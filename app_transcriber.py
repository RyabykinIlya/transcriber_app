#!/usr/bin/python
# -*- coding: utf-8 -*-

import whisperx
import gc
import torch
import threading
from pydub import AudioSegment
from pydub.utils import make_chunks
from os import mkdir, path, remove, environ, mknod
import shutil
from flask import Flask, request, Response, jsonify
import sys
import logging
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
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(environ.get("LOGGING_LEVEL", "INFO"))
        
        # Configuration
        self.port = environ.get("APP_PORT", 3000)
        self.debug = bool(environ.get("APP_DEBUG", False))
        self.hf_token = environ.get("HF_TOKEN", None)
        self.media_folder = Path(environ.get("MEDIA_FOLDER", "")).resolve()
        self.chunk_length_ms = int(environ.get("CHUNK_LENGTH_MS", 200000))
        
        # File extensions
        self.lock_ext = ".lock"
        self.transcription_ext = "_transcription.txt"
        self.error_ext = "_error.log"
        self.chunks_ext = "_chunks"
        
        self.model_config = ModelConfig(
            language="ru",
            model_name="large-v2",
            batch_size=int(environ.get("BATCH_SIZE", 16)), 
            compute_type=environ.get("COMPUTE_TYPE", "float16")
        )
        
        self.active_threads: Dict[str, threading.Thread] = {}
        self.setup_routes()

    def setup_routes(self):
        self.app.route("/")(self.root_page)
        self.app.route("/get_status", methods=["GET"])(self.get_status)
        self.app.route("/get_transcription", methods=["GET"])(self.transcribe_from_mp3)

    @contextlib.contextmanager
    def manage_resources(self):
        try:
            yield
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    def process_audio(self, audio_file: Path, speakers_from_to: Optional[Tuple[int, int]] = None):
        with self.manage_resources():
            model = whisperx.load_model(
                self.model_config.model_name,
                self.model_config.device,
                language=self.model_config.language,
                compute_type=self.model_config.compute_type
            )

            audio = whisperx.load_audio(str(audio_file))
            result = model.transcribe(
                audio,
                language=self.model_config.language,
                batch_size=self.model_config.batch_size
            )

            model_a, metadata = whisperx.load_align_model(
                language_code=self.model_config.language,
                device=self.model_config.device
            )
            
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                self.model_config.device,
                return_char_alignments=False
            )

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

        return result

    def create_chunks(self, source_filename: str) -> List[str]:
            audio_stem = Path(source_filename).stem
            chunks_folder = self.media_folder / f"{audio_stem}{self.chunks_ext}"
            
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(chunks_folder)
            
            chunks_folder.mkdir(exist_ok=True)
            
            myaudio = AudioSegment.from_file(self.media_folder / source_filename)
            chunks = make_chunks(myaudio, self.chunk_length_ms)

            chunk_names = []
            for i, chunk in enumerate(chunks):
                chunk_name = f"{audio_stem}_chunk{i}.wav"
                self.logger.debug(f"exporting {chunk_name}")
                chunk.export(chunks_folder / chunk_name, format="wav")
                chunk_names.append(chunk_name)

            return chunk_names

    def transcribe(self, source_filename: str, speakers_count: Optional[int] = None) -> None:
        try:
            audio_stem = Path(source_filename).stem
            chunks_folder = self.media_folder / f"{audio_stem}{self.chunks_ext}"
            transcription_file = self.media_folder / f"{audio_stem}{self.transcription_ext}"

            chunk_names = self.create_chunks(source_filename)
            self.logger.debug(f'Processing chunks: {", ".join(chunk_names)}')

            speakers_from_to = (1, speakers_count or 3) if speakers_count else None

            with transcription_file.open('w') as f:
                for chunk_name in chunk_names:
                    audio_file = chunks_folder / chunk_name
                    self.logger.debug(f"Processing {audio_file}")
                    
                    try:
                        result = self.process_audio(audio_file, speakers_from_to)
                        for segment in result["segments"]:
                            text = f"\n{segment['speaker']}: {segment['text']}" if 'speaker' in segment else f"\n{segment['text']}"
                            f.write(text)
                    except RuntimeError as e:
                        error_file = self.media_folder / f"{audio_stem}{self.error_ext}"
                        error_file.write_text(f"{audio_file} {str(e)}\n")
                        self.logger.exception(e)
                        break

        finally:
            if source_filename in self.active_threads:
                del self.active_threads[source_filename]
            
            lock_file = self.media_folder / f"{source_filename}{self.lock_ext}"
            with contextlib.suppress(FileNotFoundError):
                lock_file.unlink()

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

        transcribe_thread = threading.Thread(
            target=self.transcribe,
            name=f"Transcriber {audio_name}",
            args=(audio_name, None),
        )
        transcribe_thread.start()
        self.active_threads[audio_name] = transcribe_thread
        
        statusfile = f"{audio_name}{self.lock_ext}"
        try:
            (self.media_folder / statusfile).touch()
        except FileExistsError:
            pass
            
        return jsonify({"status": "OK", "statusfile": statusfile})

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
