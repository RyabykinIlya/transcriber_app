#!/usr/bin/python
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")

import whisperx
import gc
import torch
import threading
from pydub import AudioSegment
from pydub.utils import make_chunks
from os import environ
import shutil
from flask import Flask, request, Response, jsonify
import sys
import logging
import tempfile
from werkzeug.utils import secure_filename
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import contextlib
from dotenv import load_dotenv

@dataclass
class ModelConfig:
    language: str
    model_name: str
    batch_size: int
    compute_type: str
    device: str = "cuda"

class TranscriberApp:
    def __init__(self):
        self.app = Flask(__name__)

        # Configure root logger to ensure all logs (including threaded ones) go to stdout
        logging.basicConfig(
            level=getattr(logging, environ.get("LOGGING_LEVEL", "DEBUG").upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True
        )

        # Ensure logs are flushed immediately in threads
        for handler in logging.root.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
        sys.stdout.flush()

        self.logger = logging.getLogger(__name__)

        # Configuration
        self.port = environ.get("APP_PORT", 3000)
        self.debug = bool(environ.get("APP_DEBUG", False))
        self.hf_token = environ.get("HF_TOKEN", None)
        self.media_folder = Path(environ.get("MEDIA_FOLDER", "")).resolve()
        self.chunk_length_ms = int(environ.get("CHUNK_LENGTH_MS", 120000))
        self.chunk_overlap_ms = int(environ.get("CHUNK_OVERLAP_MS", 3000))

        # File extensions
        self.lock_ext = ".lock"
        self.transcription_ext = "_transcription.txt"
        self.error_ext = "_error.log"
        self.chunks_ext = "_chunks"

        self.model_config = ModelConfig(
            language="ru",
            model_name="large-v3",
            batch_size=int(environ.get("BATCH_SIZE", 8)),
            compute_type=environ.get("COMPUTE_TYPE", "float16")
        )

        self.active_threads: Dict[str, threading.Thread] = {}

        # Lock to prevent concurrent transcriptions (OOM protection)
        self._transcription_lock = threading.Lock()

        # Cached models (loaded once, reused across chunks and requests)
        self._whisper_model = None
        self._align_model = None
        self._align_metadata = None

        self.setup_routes()

    def log_memory(self, label: str):
        if torch.cuda.is_available():
            self.logger.info(f"{label} - GPU Memory: "
                            f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, "
                            f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
            torch.cuda.reset_peak_memory_stats()

    def setup_routes(self):
        self.app.route("/")(self.root_page)
        self.app.route("/get_status", methods=["GET"])(self.get_status)
        self.app.route("/get_transcription", methods=["GET"])(self.transcribe_from_mp3)
        self.app.route("/upload_and_transcribe", methods=["POST"])(self.upload_and_transcribe)

    def _get_whisper_model(self):
        if self._whisper_model is None:
            self.logger.info("Loading whisper model (first time)...")
            self._whisper_model = whisperx.load_model(
                self.model_config.model_name,
                self.model_config.device,
                language=self.model_config.language,
                compute_type=self.model_config.compute_type
            )
        return self._whisper_model

    def _get_align_model(self):
        if self._align_model is None:
            self.logger.info("Loading alignment model (first time)...")
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=self.model_config.language,
                device=self.model_config.device
            )
        return self._align_model, self._align_metadata

    def _cleanup_gpu(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.log_memory("Cleanup")

    def process_audio(self, audio_file: Path, speakers_from_to: Optional[Tuple[int, int]] = None):
        self.log_memory(f"Before process {audio_file.name}")
        model = self._get_whisper_model()

        audio = whisperx.load_audio(str(audio_file))
        self.log_memory(f"After load_audio {audio_file.name}")
        result = model.transcribe(
            audio,
            language=self.model_config.language,
            batch_size=self.model_config.batch_size
        )
        self.log_memory(f"After transcribe {audio_file.name}")

        model_a, metadata = self._get_align_model()

        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            self.model_config.device,
            return_char_alignments=False
        )
        self.log_memory(f"After align {audio_file.name}")

        if speakers_from_to:
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=self.hf_token,
                device=self.model_config.device
            )
            diarize_segments = diarize_model(
                audio,
                min_speakers=speakers_from_to[0],
                max_speakers=speakers_from_to[1]
            )
            result = whisperx.assign_word_speakers(diarize_segments, result)
            del diarize_model
            self.log_memory(f"After diarize {audio_file.name}")

        del audio
        self._cleanup_gpu()
        self.log_memory(f"After cleanup {audio_file.name}")

        return result

    def _slice_with_overlap(self, myaudio: AudioSegment) -> List[AudioSegment]:
        """Split audio into chunks of `chunk_length_ms` with `chunk_overlap_ms`
        overlap between neighbours. Overlap gives Whisper extra context so words
        cut at chunk boundaries stay intact; duplicates from the overlap zone
        are dropped later by `_keep_segment_in_chunk`."""
        total = len(myaudio)
        if self.chunk_overlap_ms <= 0 or total <= self.chunk_length_ms:
            return make_chunks(myaudio, self.chunk_length_ms)

        step = self.chunk_length_ms - self.chunk_overlap_ms
        if step <= 0:
            raise ValueError("CHUNK_OVERLAP_MS must be smaller than CHUNK_LENGTH_MS")

        chunks: List[AudioSegment] = []
        start = 0
        while start < total:
            end = min(start + self.chunk_length_ms, total)
            chunks.append(myaudio[start:end])
            if end >= total:
                break
            start += step
        return chunks

    def _keep_segment_in_chunk(self, segment: dict, is_last_chunk: bool) -> bool:
        """For every non-last chunk, drop segments whose end falls into the
        trailing overlap zone — those words will be re-transcribed by the next
        chunk with full leading context."""
        if is_last_chunk or self.chunk_overlap_ms <= 0:
            return True
        keep_until_sec = (self.chunk_length_ms - self.chunk_overlap_ms) / 1000.0
        return float(segment.get("end", 0.0)) <= keep_until_sec

    def create_chunks(self, source_filename: str) -> List[str]:
            print(f"Loading audio {source_filename}")
            audio_stem = Path(source_filename).stem
            chunks_folder = self.media_folder / f"{audio_stem}{self.chunks_ext}"

            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(chunks_folder)

            chunks_folder.mkdir(exist_ok=True)

            myaudio = AudioSegment.from_file(self.media_folder / source_filename)
            print("Audio loaded, creating chunks")
            chunks = self._slice_with_overlap(myaudio)
            del myaudio
            print(f"Chunks made: {len(chunks)} (overlap {self.chunk_overlap_ms}ms)")

            chunk_names = []
            for i, chunk in enumerate(chunks):
                chunk_name = f"{audio_stem}_chunk{i}.wav"
                self.logger.info(f"Creating chunk {chunk_name}")
                chunk.export(chunks_folder / chunk_name, format="wav")
                chunk_names.append(chunk_name)

            return chunk_names

    def transcribe(self, source_filename: str, speakers_count: Optional[int] = None) -> None:
        print(f"Thread transcribe started for {source_filename}")
        try:
            # Only one transcription at a time to prevent OOM
            with self._transcription_lock:
                audio_stem = Path(source_filename).stem
                chunks_folder = self.media_folder / f"{audio_stem}{self.chunks_ext}"
                transcription_file = self.media_folder / f"{audio_stem}{self.transcription_ext}"

                chunk_names = self.create_chunks(source_filename)
                print(f"Chunks created: {len(chunk_names)}")
                self.logger.info(f'Processing chunks: {", ".join(chunk_names)}')

                speakers_from_to = (1, speakers_count or 3) if speakers_count else None

                with transcription_file.open('w') as f:
                    last_index = len(chunk_names) - 1
                    for idx, chunk_name in enumerate(chunk_names):
                        audio_file = chunks_folder / chunk_name
                        self.logger.info(f"Processing {audio_file}")
                        self.log_memory(f"Before chunk {chunk_name}")

                        try:
                            self._cleanup_gpu()
                            result = self.process_audio(audio_file, speakers_from_to)
                            is_last = idx == last_index
                            for segment in result["segments"]:
                                if not self._keep_segment_in_chunk(segment, is_last):
                                    continue
                                text = f"\n{segment['speaker']}: {segment['text']}" if 'speaker' in segment else f"\n{segment['text']}"
                                f.write(text)
                            del result
                            self.log_memory(f"After chunk {chunk_name}")
                        except Exception as e:
                            error_file = self.media_folder / f"{audio_stem}{self.error_ext}"
                            error_file.write_text(f"{audio_file} {type(e).__name__}: {str(e)}\n")
                            self.logger.exception(e)
                            break

        except Exception as e:
            self.logger.exception(f"Fatal error in transcription of {source_filename}: {e}")
            try:
                audio_stem = Path(source_filename).stem
                error_file = self.media_folder / f"{audio_stem}{self.error_ext}"
                error_file.write_text(f"Fatal: {type(e).__name__}: {str(e)}\n")
            except Exception:
                pass
        finally:
            if source_filename in self.active_threads:
                del self.active_threads[source_filename]

            lock_file = self.media_folder / f"{source_filename}{self.lock_ext}"
            with contextlib.suppress(FileNotFoundError):
                lock_file.unlink()

    def transcribe_file_sync(
        self,
        audio_path: Path,
        workdir: Path,
        speakers_count: Optional[int] = None,
        lock_timeout: float = 10.0,
    ) -> Optional[str]:
        """Run the same chunk+whisper+align pipeline synchronously on an arbitrary
        file, keeping all artifacts inside `workdir`. Returns transcription text,
        or None if the transcription lock could not be acquired within timeout."""
        acquired = self._transcription_lock.acquire(timeout=lock_timeout)
        if not acquired:
            return None
        try:
            audio_stem = audio_path.stem
            chunks_folder = workdir / f"{audio_stem}{self.chunks_ext}"
            chunks_folder.mkdir(exist_ok=True)

            myaudio = AudioSegment.from_file(audio_path)
            chunks = self._slice_with_overlap(myaudio)
            del myaudio
            self.logger.info(
                f"Chunks made: {len(chunks)} for {audio_path.name} (overlap {self.chunk_overlap_ms}ms)"
            )

            chunk_paths: List[Path] = []
            for i, chunk in enumerate(chunks):
                chunk_path = chunks_folder / f"{audio_stem}_chunk{i}.wav"
                chunk.export(chunk_path, format="wav")
                chunk_paths.append(chunk_path)

            speakers_from_to = (1, speakers_count or 3) if speakers_count else None

            parts: List[str] = []
            last_index = len(chunk_paths) - 1
            for idx, chunk_path in enumerate(chunk_paths):
                self.logger.info(f"Processing {chunk_path}")
                self._cleanup_gpu()
                result = self.process_audio(chunk_path, speakers_from_to)
                is_last = idx == last_index
                for segment in result["segments"]:
                    if not self._keep_segment_in_chunk(segment, is_last):
                        continue
                    text = (
                        f"\n{segment['speaker']}: {segment['text']}"
                        if 'speaker' in segment
                        else f"\n{segment['text']}"
                    )
                    parts.append(text)
                del result

            return "".join(parts).strip()
        finally:
            self._transcription_lock.release()

    # Flask routes
    def root_page(self):
        return Response("OK")

    def get_status(self):
        status_filename = request.values.get("statusfile")
        if not status_filename:
            return jsonify({"status": "ERROR", "message": "No status filename provided"})

        base_path = Path(status_filename.replace(self.lock_ext, ""))
        error_file = self.media_folder / f"{base_path.stem}{self.error_ext}"

        if error_file.exists():
            error_message = error_file.read_text()
            return jsonify({"status": "ERROR", "message": f"Error in transcription process: {error_message}"})

        if (self.media_folder / status_filename).exists():
            return jsonify({"status": "IN PROGRESS"})

        try:
            transcription_file = self.media_folder / f"{base_path.stem}{self.transcription_ext}"
            transcription_text = transcription_file.read_text()

            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(self.media_folder / f"{base_path.stem}{self.chunks_ext}")
            transcription_file.unlink()

            return jsonify({"status": "OK", "transcription_text": transcription_text.strip()})
        except FileNotFoundError as e:
            self.logger.exception(f"Transcription {transcription_file} not found\n\n{e}")
            return jsonify({"status": "ERROR", "message": f"Transcription {transcription_file} not found"})

    def transcribe_from_mp3(self):
        audio_name = request.values.get("audio_name")
        if not audio_name:
            return jsonify({"status": "ERROR", "message": "No audio_name provided"})

        filepath = self.media_folder / audio_name
        if not filepath.exists():
            return jsonify({"status": "ERROR", "message": f"File {filepath} not found on host"})

        # Check if thread already exists and is running
        if audio_name in self.active_threads:
            thread = self.active_threads[audio_name]
            if thread.is_alive():
                return jsonify({"status": "IN_PROGRESS", "message": f"Transcription for {audio_name} is already running"})
            else:
                # Clean up completed thread
                del self.active_threads[audio_name]

        print(f"Starting transcription for {audio_name}")
        transcribe_thread = threading.Thread(
            target=self.transcribe,
            name=f"Transcriber {audio_name}",
            args=(audio_name, None),
            daemon=True,
        )
        transcribe_thread.start()
        self.active_threads[audio_name] = transcribe_thread
        print(f"Thread started for {audio_name}")

        statusfile = f"{audio_name}{self.lock_ext}"
        try:
            (self.media_folder / statusfile).touch()
        except FileExistsError:
            pass

        return jsonify({"status": "OK", "statusfile": statusfile})

    def upload_and_transcribe(self):
        uploaded = request.files.get("file")
        if uploaded is None or uploaded.filename == "":
            return jsonify({"status": "ERROR", "message": "No file provided (expected multipart field 'file')"}), 400

        filename = secure_filename(uploaded.filename) or "upload.bin"
        speakers_count = request.values.get("speakers_count", type=int)

        with tempfile.TemporaryDirectory(prefix="transcribe_") as tmp:
            workdir = Path(tmp)
            audio_path = workdir / filename
            uploaded.save(audio_path)
            self.logger.info(f"Uploaded {filename} ({audio_path.stat().st_size} bytes) -> {audio_path}")

            try:
                text = self.transcribe_file_sync(
                    audio_path=audio_path,
                    workdir=workdir,
                    speakers_count=speakers_count,
                    lock_timeout=10.0,
                )
            except Exception as e:
                self.logger.exception(f"upload_and_transcribe failed for {filename}: {e}")
                return jsonify({"status": "ERROR", "message": f"{type(e).__name__}: {e}"}), 500

            if text is None:
                return jsonify({
                    "status": "BUSY",
                    "message": "Another transcription is in progress, try again later",
                }), 503

            return jsonify({"status": "OK", "transcription_text": text})

    def run(self):
        self.app.run(debug=bool(self.debug), host="::", port=self.port)

def main():
    app = TranscriberApp()
    if len(sys.argv) > 1:
        # Preload models
        try:
            whisperx.load_model(
                app.model_config.model_name,
                app.model_config.device,
                language=app.model_config.language,
                compute_type=app.model_config.compute_type
            )
        except RuntimeError:
            pass

        try:
            whisperx.load_align_model(
                language_code=app.model_config.language,
                device=app.model_config.device
            )
        except RuntimeError:
            pass
    else:
        app.run()

if __name__ == "__main__":
    if environ.get("MODE") == "DEV":
        load_dotenv()
    main()
