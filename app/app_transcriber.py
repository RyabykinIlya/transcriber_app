#!/usr/bin/python
# -*- coding: utf-8 -*-
"""whisperx-based transcription service with adaptive, silence-aware chunking.

Pipeline (per file):
  1. The source is decoded ONCE by ffmpeg into 16 kHz mono float32 PCM
     (a numpy array) — no intermediate WAV chunks, no pydub, no 4 GB
     container limit; ~230 MB of RAM per hour of audio.
  2. A planner cuts the recording into a FEW large chunks (target
     CHUNK_TARGET_MS, hard bounds CHUNK_MIN_MS..CHUNK_MAX_MS), snapping
     every cut to the nearest detected silence so words are almost never
     split at a boundary. A 40-minute meeting is 4 chunks, not 20.
  3. Each chunk is a plain numpy slice (plus a small overlap margin) fed
     to whisper + align. Every segment belongs to exactly one chunk —
     decided by its START time — so there are no duplicates and no lost
     words at boundaries. Decoding is biased with a corporate glossary
     (WHISPER_INITIAL_PROMPT / WHISPER_HOTWORDS) so domain terms and
     brand spellings come out right; callers may pass their own
     vocabulary per request.
  4. If a chunk fails (CUDA hiccup, OOM, ...), it is retried once and
     then bisected recursively down to CHUNK_BISECT_MIN_MS. One bad
     chunk no longer kills the rest of the meeting; failed ranges are
     reported in the error file while the rest of the text is kept.
  5. Diarization (optional) runs ONCE over the whole recording, giving
     stable SPEAKER_XX labels across all chunks instead of per-chunk
     relabeling (which produced different labels for the same person).

Progress (chunk i/N, percent, failed count) is written into the .lock
file as JSON and surfaced by /get_status.
"""

import os

# expandable_segments breaks pyannote embedding convs under VRAM pressure
# ("CUDA driver error: device not ready", measured). It may be baked into
# the image ENV from an older deploy — drop it before torch initializes.
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

import warnings

warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")

import contextlib
import gc
import json
import shutil
import subprocess
import sys
import threading
import time
import logging
import tempfile
from dataclasses import dataclass, replace
from os import environ
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import whisperx
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request
from werkzeug.utils import secure_filename

SAMPLE_RATE = 16000


def _env_int(name: str, default: int) -> int:
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


@dataclass
class ChunkingConfig:
    """Adaptive chunking bounds (all in milliseconds)."""

    target_ms: int = 900_000
    min_ms: int = 120_000
    max_ms: int = 1_800_000
    silence_search_ms: int = 30_000
    overlap_ms: int = 3_000
    min_silence_ms: int = 500
    bisect_min_ms: int = 30_000

    def __post_init__(self) -> None:
        # Normalize so that min <= target <= max and overlap stays sane.
        self.min_ms = max(5_000, self.min_ms)
        if self.min_ms > self.target_ms:
            self.min_ms = self.target_ms
        if self.max_ms < self.target_ms:
            self.max_ms = self.target_ms
        if self.overlap_ms >= self.min_ms:
            self.overlap_ms = self.min_ms // 3
        if self.bisect_min_ms > self.min_ms:
            self.bisect_min_ms = self.min_ms

    @classmethod
    def from_env(cls) -> "ChunkingConfig":
        # CHUNK_LENGTH_MS is the legacy knob — honoured as the target.
        legacy = _env_int("CHUNK_LENGTH_MS", 0)
        return cls(
            target_ms=_env_int("CHUNK_TARGET_MS", legacy or 900_000),
            min_ms=_env_int("CHUNK_MIN_MS", 120_000),
            max_ms=_env_int("CHUNK_MAX_MS", 1_800_000),
            silence_search_ms=_env_int("CHUNK_SILENCE_SEARCH_MS", 30_000),
            overlap_ms=_env_int("CHUNK_OVERLAP_MS", 3_000),
            min_silence_ms=_env_int("CHUNK_MIN_SILENCE_MS", 500),
            bisect_min_ms=_env_int("CHUNK_BISECT_MIN_MS", 30_000),
        )


@dataclass
class ModelConfig:
    language: str
    model_name: str
    batch_size: int
    compute_type: str
    device: str = "cuda"
    beam_size: int = 5
    initial_prompt: Optional[str] = None
    hotwords: Optional[str] = None


# Default vocabulary biasing for decoding (A/B-tested on real meetings:
# roughly 2x more correctly spelled domain terms, no speed cost).
# Override per deployment with WHISPER_INITIAL_PROMPT / WHISPER_HOTWORDS
# env vars, or per request with the `initial_prompt` / `hotwords` params.
DEFAULT_GLOSSARY = (
    "ЗНИ, ФТТ, ДНИ, ИТ, КСПД, DMZ, СЭД, архсовет, архитектор, "
    "чат-бот, сценарий, виджет, рантайм, роутер, база знаний, "
    "векторная база, Chroma, OpenRouter, Runway, LLM, SLM, "
    "кредитный скоринг, кредитное управление, корпоративные коммуникации, "
    "интеграционная спека, бизнес-процесс, ИБ, C#, C-Sharp, fastapp"
)


def fmt_ts(seconds: float) -> str:
    """mm:ss for logs and error reports."""
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class TranscriberApp:
    def __init__(self):
        self.app = Flask(__name__)

        logging.basicConfig(
            level=getattr(logging, environ.get("LOGGING_LEVEL", "INFO").upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True,
        )
        self.logger = logging.getLogger(__name__)

        self.port = environ.get("APP_PORT", 3000)
        # NOTE: env vars are strings — bool("False") is True. Parse explicitly,
        # otherwise the Flask debug reloader forks the process and CUDA models
        # load twice.
        self.debug = environ.get("APP_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
        self.hf_token = environ.get("HF_TOKEN", None)
        self.media_folder = Path(environ.get("MEDIA_FOLDER", "")).resolve()
        self.chunking = ChunkingConfig.from_env()

        # File extensions / artifact suffixes (kept for API compatibility)
        self.lock_ext = ".lock"
        self.transcription_ext = "_transcription.txt"
        self.error_ext = "_error.log"
        self.chunks_ext = "_chunks"  # legacy on-disk chunks, cleaned up if found

        self.model_config = ModelConfig(
            language="ru",
            model_name="large-v3",
            batch_size=_env_int("BATCH_SIZE", 4),
            compute_type=environ.get("COMPUTE_TYPE", "float16"),
            device=environ.get("DEVICE", "cuda"),
            beam_size=_env_int("ASR_BEAM_SIZE", 5),
            # WHISPER_INITIAL_PROMPT unset -> tested default glossary;
            # set (even empty) -> used literally, allowing to disable biasing.
            initial_prompt=(
                environ["WHISPER_INITIAL_PROMPT"].strip()
                if "WHISPER_INITIAL_PROMPT" in environ
                else DEFAULT_GLOSSARY
            ),
            hotwords=(environ.get("WHISPER_HOTWORDS", "").strip() or None),
        )

        self.active_threads: Dict[str, threading.Thread] = {}
        self._threads_lock = threading.Lock()

        # Only one transcription at a time (GPU is small).
        self._transcription_lock = threading.Lock()

        # Models loaded once, reused across chunks and requests.
        self._whisper_model = None
        self._align_model = None
        self._align_metadata = None
        self._diarize_pipeline = None

        # Speaker identification (optional, see /enroll_voice).
        self._voiceprint: Optional[Dict] = None  # lazy, from voiceprint.json
        self.speaker_name = environ.get("SPEAKER_NAME", "Илья Рябыкин")
        # Calibration (cosine): own clean voice ~1.0, own voice in a real
        # meeting ~0.71 (worse mic/compression), other people 0.13-0.37.
        # 0.60 sits in the middle of that gap.
        self.voice_match_threshold = float(environ.get("VOICE_MATCH_THRESHOLD", "0.60"))

        self.setup_routes()

    # ------------------------------------------------------------------ #
    # Models
    # ------------------------------------------------------------------ #

    def _get_whisper_model(self):
        if self._whisper_model is None:
            self.logger.info("Loading whisper model (first time)...")
            asr_options = {"beam_size": self.model_config.beam_size}
            if self.model_config.initial_prompt:
                asr_options["initial_prompt"] = self.model_config.initial_prompt
            if self.model_config.hotwords:
                asr_options["hotwords"] = self.model_config.hotwords
            self._whisper_model = whisperx.load_model(
                self.model_config.model_name,
                self.model_config.device,
                language=self.model_config.language,
                compute_type=self.model_config.compute_type,
                asr_options=asr_options,
            )
        return self._whisper_model

    @contextlib.contextmanager
    def _override_asr_options(
        self, initial_prompt: Optional[str] = None, hotwords: Optional[str] = None
    ):
        """Temporarily swap decoding vocabulary for one request.

        Safe under the transcription lock (all jobs are serialized): the
        pipeline reads `options` on every generation call, and whisperx
        itself swaps the same attribute via `dataclasses.replace`.
        """
        if initial_prompt is None and hotwords is None:
            yield
            return
        if initial_prompt is not None and len(initial_prompt.strip()) < 120:
            # A/B-tested: a bare term list as initial_prompt makes Whisper
            # echo it back as hallucinated text ("ЗНИ ФТТ ЗНИ ФТТ...").
            # Bare vocabularies belong in `hotwords`, which is safe.
            self.logger.warning(
                "Custom initial_prompt is very short — Whisper may echo it back "
                "as hallucinated text. Prefer `hotwords` for bare term lists; "
                "use full sentences for initial_prompt."
            )
        model = self._get_whisper_model()
        new_kwargs: Dict[str, Optional[str]] = {}
        if initial_prompt is not None:
            new_kwargs["initial_prompt"] = initial_prompt.strip() or None
        if hotwords is not None:
            new_kwargs["hotwords"] = hotwords.strip() or None
        original = model.options
        model.options = replace(original, **new_kwargs)
        self.logger.info(
            "Per-request ASR options: "
            + ", ".join(f"{k}={bool(v)}" for k, v in new_kwargs.items())
        )
        try:
            yield
        finally:
            model.options = original

    def _get_align_model(self):
        if self._align_model is None:
            self.logger.info("Loading alignment model (first time)...")
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=self.model_config.language,
                device=self.model_config.device,
            )
        return self._align_model, self._align_metadata

    def _get_diarize_pipeline(self):
        if self._diarize_pipeline is None:
            # NOTE: in whisperx 3.x the pipeline lives in whisperx.diarize
            # (not exported at top level) and takes `token=`, not
            # `use_auth_token=`. The old top-level call raised AttributeError,
            # i.e. diarization silently never worked — fixed here.
            from whisperx.diarize import DiarizationPipeline

            self.logger.info("Loading diarization pipeline (first time)...")
            self._diarize_pipeline = DiarizationPipeline(
                model_name=environ.get("DIARIZE_MODEL") or None,
                token=self.hf_token,
                device=environ.get("DIARIZE_DEVICE", self.model_config.device),
            )
        return self._diarize_pipeline

    def _release_diarizer(self) -> None:
        """Drop the pyannote pipeline to get its VRAM back.

        Measured on the 8 GB RTX 3070 Ti: whisper+align sit at ~5.4 GB, the
        resident pyannote pipeline adds ~2.4 GB and stays forever (it used to
        be cached in `self._diarize_pipeline`). Whisper then transcribes the
        NEXT file with ~340 MB of headroom and intermittently falls into a
        slow-thrash mode: 100% GPU util at ~105 W, same 24-minute chunk taking
        4927 s instead of 142 s. Reloading the diarizer costs ~70 s once per
        file that needs speakers, which is far cheaper than that tail.
        """
        if self._diarize_pipeline is None:
            return
        self._diarize_pipeline = None
        self._cleanup_gpu()
        self.logger.info("Diarization pipeline released (VRAM returned to the whisper pool)")

    # ------------------------------------------------------------------ #
    # Speaker identification (voiceprint)
    # ------------------------------------------------------------------ #

    VOICEPRINT_FILE = "voiceprint.json"

    def _voiceprint_path(self) -> Path:
        return self.media_folder / self.VOICEPRINT_FILE

    def _load_voiceprint(self) -> Optional[Dict]:
        if self._voiceprint is not None:
            return self._voiceprint
        try:
            with open(self._voiceprint_path(), encoding="utf-8") as fh:
                data = json.load(fh)
            emb = data.get("embedding")
            if not emb or not isinstance(emb, list) or len(emb) < 1:
                raise ValueError("voiceprint file has no embedding")
            self._voiceprint = {
                "name": str(data.get("name") or "Илья Рябыкин"),
                "embedding": np.asarray(emb, dtype=np.float32),
                "enrolled_at": data.get("enrolled_at", ""),
            }
            self.logger.info(
                f"Voiceprint loaded: {self._voiceprint['name']} "
                f"(dim={len(emb)}, enrolled {self._voiceprint['enrolled_at']})"
            )
        except FileNotFoundError:
            self._voiceprint = None
        except Exception as e:
            self.logger.warning(f"Ignoring broken voiceprint file: {type(e).__name__}: {e}")
            self._voiceprint = None
        return self._voiceprint

    def _save_voiceprint(self, name: str, embedding: List[float]) -> None:
        payload = {
            "name": name,
            "embedding": [float(x) for x in embedding],
            "enrolled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        tmp = self._voiceprint_path().with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        tmp.replace(self._voiceprint_path())
        self._voiceprint = None  # force reload on next use

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _identify_speakers(
        self,
        df,
        embeddings: Dict,
        name: str,
        enroll_embedding: np.ndarray,
        threshold: float,
    ) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Map each SPEAKER_XX to the enrolled name (or keep the id).

        whisperx `speaker_embeddings` keys are the raw diarization labels
        ('A', 'B', ...), NOT int indices. The df carries both: raw `label`
        and public `speaker` (SPEAKER_XX) — the rename targets the latter.

        Returns (renames, scores) where renames[SPEAKER_XX]=name_or_id and
        scores[SPEAKER_XX]=best cosine similarity.
        """
        if not embeddings:
            return {}, {}
        label2speaker: Dict[str, str] = {}
        for _, row in df.iterrows():
            label2speaker.setdefault(row["label"], row["speaker"])

        renames: Dict[str, str] = {}
        scores: Dict[str, float] = {}
        for label, emb in embeddings.items():
            vec = np.asarray(emb, dtype=np.float32).reshape(-1)
            speaker = label2speaker.get(label, str(label))
            if vec.size != enroll_embedding.size:
                self.logger.warning(f"Embedding dim mismatch for {label} ({vec.size} vs {enroll_embedding.size})")
                continue
            sim = self._cosine_similarity(vec, enroll_embedding)
            scores[speaker] = sim
            renames[speaker] = name if sim >= threshold else speaker
            level = " => matched" if sim >= threshold else " (below threshold)"
            self.logger.info(f"Voice match {label}/{speaker} vs '{name}': cos={sim:.3f}{level}")
        return renames, scores

    def preload_models(self):
        for loader in (self._get_whisper_model, self._get_align_model):
            try:
                loader()
            except Exception as e:
                self.logger.warning(f"Model preload failed ({type(e).__name__}: {e}); will retry lazily")

    def log_memory(self, label: str):
        if torch.cuda.is_available():
            self.logger.debug(
                f"{label} - GPU Memory: "
                f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB, "
                f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB"
            )

    def _cleanup_gpu(self):
        gc.collect()
        with contextlib.suppress(Exception):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.log_memory("Cleanup")

    # ------------------------------------------------------------------ #
    # Audio decoding and chunk planning
    # ------------------------------------------------------------------ #

    @staticmethod
    def decode_audio(path: Path) -> np.ndarray:
        """Decode any ffmpeg-readable file to mono 16 kHz float32 PCM.

        Streams through ffmpeg, so neither RAM nor the 4 GB WAV container
        limit of pydub applies. ~230 MB of RAM per hour of audio.
        """
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found in PATH")
        cmd = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", str(path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-f", "s16le", "-",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        buf = bytearray()
        try:
            assert proc.stdout is not None
            while True:
                block = proc.stdout.read(1 << 23)
                if not block:
                    break
                buf += block
        finally:
            proc.stdout.close()
            stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            proc.stderr.close()
            code = proc.wait()
        if code != 0:
            tail = stderr.strip().splitlines()[-5:]
            raise RuntimeError(f"ffmpeg failed (exit {code}) for {path.name}: {' | '.join(tail)}")
        audio = np.frombuffer(bytes(buf), dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            raise RuntimeError(f"Decoded audio is empty for {path.name}")
        return audio

    def detect_silences(self, audio: np.ndarray) -> List[float]:
        """Return midpoints (seconds) of silences >= CHUNK_MIN_SILENCE_MS."""
        cfg = self.chunking
        window = max(1, int(SAMPLE_RATE * 0.03))  # 30 ms frames
        n_frames = audio.size // window
        if n_frames < 2:
            return []
        frames = audio[: n_frames * window].reshape(n_frames, window)
        rms = np.sqrt((frames.astype(np.float32) ** 2).mean(axis=1) + 1e-12)
        db = 20.0 * np.log10(rms + 1e-9)
        # Adaptive threshold: above the noise floor, but well below speech.
        floor_db = float(np.percentile(db, 10))
        loud_db = float(np.percentile(db, 95))
        threshold_db = max(floor_db + 6.0, loud_db - 25.0, -55.0)
        silent = db < threshold_db
        min_frames = max(1, int(cfg.min_silence_ms / 30))
        midpoints: List[float] = []
        i = 0
        while i < n_frames:
            if silent[i]:
                j = i
                while j < n_frames and silent[j]:
                    j += 1
                if j - i >= min_frames:
                    t0, t1 = i * window / SAMPLE_RATE, j * window / SAMPLE_RATE
                    midpoints.append((t0 + t1) / 2.0)
                i = j
            else:
                i += 1
        return midpoints

    def plan_chunks(
        self, duration_s: float, silences: List[float]
    ) -> Tuple[List[Tuple[float, float]], int]:
        """Plan (start, end) keep-ranges covering [0, duration].

        Greedy: aim for `target`, snap to the nearest silence within the
        search window, never leave a chunk shorter than `min` or longer
        than `max`. Files shorter than `max` become a single chunk.
        """
        cfg = self.chunking
        target = cfg.target_ms / 1000.0
        lo_bound, hi_bound = cfg.min_ms / 1000.0, cfg.max_ms / 1000.0
        search = cfg.silence_search_ms / 1000.0

        if duration_s <= hi_bound:
            return [(0.0, duration_s)], 0

        silences = sorted(silences)
        cuts = [0.0]
        snapped = 0
        cur = 0.0
        while duration_s - cur > hi_bound:
            aim = cur + target
            lo = max(cur + lo_bound, aim - search)
            hi = min(cur + hi_bound, aim + search, duration_s - lo_bound)
            if hi < lo:
                hi = lo
            candidates = [s for s in silences if lo <= s <= hi]
            if candidates:
                cut = min(candidates, key=lambda s: abs(s - aim))
                snapped += 1
            else:
                cut = min(max(aim, lo), hi)
            if cut <= cur + 1.0:  # safety against pathological configs
                cut = cur + lo_bound
            cuts.append(cut)
            cur = cut
        cuts.append(duration_s)
        ranges = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
        return ranges, snapped

    # ------------------------------------------------------------------ #
    # Transcription core
    # ------------------------------------------------------------------ #

    def _transcribe_slice(self, audio: np.ndarray, start_s: float, end_s: float) -> dict:
        """Transcribe+align audio[start_s-overlap : end_s+overlap].

        Returns a whisperx result whose segment/word timestamps are
        shifted to GLOBAL (whole-file) time.
        """
        cfg = self.chunking
        overlap = cfg.overlap_ms / 1000.0
        total_s = audio.size / SAMPLE_RATE
        slice_start = max(0.0, start_s - overlap)
        slice_end = min(total_s, end_s + overlap)
        audio_slice = audio[
            int(round(slice_start * SAMPLE_RATE)): int(round(slice_end * SAMPLE_RATE))
        ]

        model = self._get_whisper_model()
        result = model.transcribe(
            audio_slice,
            language=self.model_config.language,
            batch_size=self.model_config.batch_size,
        )
        segments = result.get("segments") or []
        if segments:
            model_a, metadata = self._get_align_model()
            result = whisperx.align(
                segments,
                model_a,
                metadata,
                audio_slice,
                self.model_config.device,
                return_char_alignments=False,
            )
        else:
            result = {"segments": []}

        for seg in result.get("segments", []):
            seg["start"] = float(seg.get("start", 0.0)) + slice_start
            seg["end"] = float(seg.get("end", 0.0)) + slice_start
            for word in seg.get("words", []) or []:
                word["start"] = float(word.get("start", 0.0)) + slice_start
                word["end"] = float(word.get("end", 0.0)) + slice_start
        return result

    def _process_range(self, audio: np.ndarray, start_s: float, end_s: float, depth: int = 0) -> dict:
        """Transcribe [start_s, end_s); on failure retry, then bisect."""
        last_error: Optional[Exception] = None
        for attempt in (1, 2):
            try:
                return self._transcribe_slice(audio, start_s, end_s)
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"Chunk [{fmt_ts(start_s)}-{fmt_ts(end_s)}] attempt {attempt} failed "
                    f"({type(e).__name__}: {e}); cleaning GPU and retrying"
                )
                self._cleanup_gpu()

        bisect_floor = self.chunking.bisect_min_ms / 1000.0
        if end_s - start_s >= 2 * bisect_floor and depth < 6:
            mid = (start_s + end_s) / 2.0
            self.logger.warning(
                f"Chunk [{fmt_ts(start_s)}-{fmt_ts(end_s)}] failed twice "
                f"({type(last_error).__name__}: {last_error}); splitting at {fmt_ts(mid)} and retrying halves"
            )
            left = self._process_range(audio, start_s, mid, depth + 1)
            right = self._process_range(audio, mid, end_s, depth + 1)
            return {"segments": (left.get("segments") or []) + (right.get("segments") or [])}
        raise last_error  # type: ignore[misc]

    def transcribe_array(
        self,
        audio: np.ndarray,
        speakers_from_to: Optional[Tuple[int, int]],
        sink: Callable[[dict], None],
        progress: Optional[Callable[..., None]] = None,
    ) -> List[Tuple[float, float, Exception]]:
        """Transcribe the whole decoded audio array chunk by chunk.

        Each segment is passed to `sink` exactly once (assigned by its
        start time to the chunk that owns it). Returns the list of
        (start, end, exception) ranges that failed permanently.
        """
        total_s = audio.size / SAMPLE_RATE
        silences = self.detect_silences(audio)
        ranges, snapped = self.plan_chunks(total_s, silences)
        self.logger.info(
            f"Planned {len(ranges)} chunk(s) for {fmt_ts(total_s)} of audio "
            f"(target {self.chunking.target_ms // 1000}s, max {self.chunking.max_ms // 1000}s, "
            f"{snapped}/{max(len(ranges) - 1, 0)} cuts snapped to silence, "
            f"{len(silences)} silences found)"
        )
        if progress:
            progress(stage="transcribing")

        # Diarization: run when explicitly requested OR when a voiceprint is
        # enrolled (then it must run so we can tag the owner's speech).
        voiceprint = self._load_voiceprint()
        # NOTE on stage order: transcription runs FIRST, diarization LAST.
        # The pyannote pipeline stays resident (~1-2GB VRAM) once loaded; if
        # it runs before transcription, whisper ends up transcribing against
        # a nearly-full 8GB GPU and falls into a slow-thrash mode (measured:
        # 100% util at ~110W, 7+ min without progress). The transcribe-only
        # profile (~5.5GB) and the transcribe+diarize profile (~6.5GB) are
        # both proven fine.
        failed: List[Tuple[float, float, Exception]] = []
        started = time.monotonic()
        all_segments: List[dict] = []
        for idx, (a, b) in enumerate(ranges):
            chunk_started = time.monotonic()
            try:
                result = self._process_range(audio, a, b)
            except Exception as e:
                self.logger.error(
                    f"Chunk {idx + 1}/{len(ranges)} [{fmt_ts(a)}-{fmt_ts(b)}] failed permanently: "
                    f"{type(e).__name__}: {e}"
                )
                failed.append((a, b, e))
                if progress:
                    progress(idx + 1, len(ranges), failed=len(failed))
                continue

            # Ownership rule: a segment belongs to the chunk whose
            # keep-range contains its START — no duplicates, no loss.
            segments = [
                s for s in (result.get("segments") or [])
                if a <= float(s.get("start", a)) < b
            ]
            all_segments.append(segments)
            self.logger.info(
                f"Chunk {idx + 1}/{len(ranges)} [{fmt_ts(a)}-{fmt_ts(b)}] done: "
                f"{len(segments)} segments in {time.monotonic() - chunk_started:.1f}s"
            )
            if progress:
                progress(idx + 1, len(ranges), failed=len(failed))
            del result
            self._cleanup_gpu()

        # Diarization: run when explicitly requested OR when a voiceprint is
        # enrolled (then it must run so we can tag the owner's speech).
        diarization = None
        speaker_renames: Dict[str, str] = {}
        if (speakers_from_to or voiceprint is not None) and any(all_segments):
            try:
                if progress:
                    progress(stage="diarization")
                pipeline = self._get_diarize_pipeline()
                diarization, speaker_embeddings = pipeline(
                    audio,
                    min_speakers=(speakers_from_to[0] if speakers_from_to else 1),
                    max_speakers=(speakers_from_to[1] if speakers_from_to else 3),
                    return_embeddings=True,
                )
                self.logger.info("Diarization finished for the whole file")
                if voiceprint is not None and speaker_embeddings is not None:
                    self.logger.info(f"Speaker labels found: {list(speaker_embeddings.keys())}")
                    renames, _ = self._identify_speakers(
                        diarization,
                        speaker_embeddings,
                        self.speaker_name,
                        np.asarray(voiceprint["embedding"], dtype=np.float32),
                        self.voice_match_threshold,
                    )
                    speaker_renames = renames
            except Exception as e:
                self.logger.exception(
                    f"Diarization failed ({type(e).__name__}: {e}); continuing without speakers"
                )
                diarization = None
            finally:
                # Always hand the ~2.4 GB back: the next file in the queue
                # must not inherit a nearly-full GPU (see _release_diarizer).
                self._release_diarizer()

        # Speaker labels + voiceprint renaming, then emit everything in order.
        for segments in all_segments:
            if diarization is not None and segments:
                try:
                    assigned = whisperx.assign_word_speakers(
                        diarization, {"segments": segments}
                    )
                    segments = assigned.get("segments", segments)
                except Exception as e:
                    self.logger.warning(f"Speaker assignment failed for a chunk: {e}")
            if speaker_renames:
                for seg in segments:
                    spk = seg.get("speaker")
                    if spk in speaker_renames:
                        seg["speaker"] = speaker_renames[spk]
            for seg in segments:
                sink(seg)

        self.logger.info(
            f"Transcription finished: {len(ranges)} chunk(s), {len(failed)} failed, "
            f"{time.monotonic() - started:.1f}s total"
        )
        return failed

    # ------------------------------------------------------------------ #
    # Job runners (async file job + sync upload job)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_segment(segment: dict) -> str:
        if "speaker" in segment:
            return f"\n{segment['speaker']}: {segment['text']}"
        return f"\n{segment['text']}"

    def _write_progress(
        self,
        lock_file: Path,
        done: Optional[int] = None,
        total: Optional[int] = None,
        failed: int = 0,
        stage: Optional[str] = None,
    ):
        payload: Dict[str, object] = {"ts": time.time()}
        if done is not None and total:
            payload.update({"chunk": done, "total": total, "percent": int(done * 100 / total)})
        if failed:
            payload["failed_chunks"] = failed
        if stage:
            payload["stage"] = stage
        with contextlib.suppress(OSError):
            lock_file.write_text(json.dumps(payload, ensure_ascii=False))

    def transcribe(
        self,
        source_filename: str,
        speakers_count: Optional[int] = None,
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
    ) -> None:
        """Async job behind /get_transcription: writes {stem}_transcription.txt."""
        logger = self.logger
        logger.info(f"Transcription job started for {source_filename}")
        lock_file = self.media_folder / f"{source_filename}{self.lock_ext}"
        try:
            with self._transcription_lock:
                audio_stem = Path(source_filename).stem
                transcription_file = self.media_folder / f"{audio_stem}{self.transcription_ext}"
                error_file = self.media_folder / f"{audio_stem}{self.error_ext}"

                # Legacy on-disk chunks from older versions
                with contextlib.suppress(FileNotFoundError):
                    shutil.rmtree(self.media_folder / f"{audio_stem}{self.chunks_ext}")
                with contextlib.suppress(FileNotFoundError):
                    error_file.unlink()

                self._write_progress(lock_file, stage="decoding")
                audio = self.decode_audio(self.media_folder / source_filename)

                speakers_from_to = (1, speakers_count or 3) if speakers_count else None

                def sink(segment: dict) -> None:
                    f.write(self._format_segment(segment))

                with transcription_file.open("w") as f:
                    with self._override_asr_options(initial_prompt, hotwords):
                        failed = self.transcribe_array(
                            audio,
                            speakers_from_to,
                            sink,
                            progress=lambda *a, **kw: self._write_progress(lock_file, *a, **kw),
                        )
                    f.flush()
                del audio

                if failed:
                    lines = [
                        f"{fmt_ts(a)}-{fmt_ts(b)} {type(e).__name__}: {e}" for a, b, e in failed
                    ]
                    error_file.write_text(
                        "Some chunks failed, transcription is PARTIAL.\n" + "\n".join(lines) + "\n"
                    )
                    logger.error(f"{len(failed)} chunk(s) failed for {source_filename}")

        except Exception as e:
            logger.exception(f"Fatal error in transcription of {source_filename}: {e}")
            try:
                error_file = self.media_folder / f"{Path(source_filename).stem}{self.error_ext}"
                error_file.write_text(f"Fatal: {type(e).__name__}: {e}\n")
            except Exception:
                pass
        finally:
            with self._threads_lock:
                self.active_threads.pop(source_filename, None)
            with contextlib.suppress(FileNotFoundError):
                lock_file.unlink()

    def transcribe_file_sync(
        self,
        audio_path: Path,
        speakers_count: Optional[int] = None,
        lock_timeout: float = 10.0,
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
    ) -> Optional[str]:
        """Synchronous pipeline for /upload_and_transcribe."""
        acquired = self._transcription_lock.acquire(timeout=lock_timeout)
        if not acquired:
            return None
        try:
            audio = self.decode_audio(audio_path)
            speakers_from_to = (1, speakers_count or 3) if speakers_count else None
            parts: List[str] = []
            with self._override_asr_options(initial_prompt, hotwords):
                failed = self.transcribe_array(
                    audio, speakers_from_to, lambda seg: parts.append(self._format_segment(seg))
                )
            del audio
            if failed and not parts:
                a, b, e = failed[0]
                raise RuntimeError(f"All chunks failed; first: [{fmt_ts(a)}-{fmt_ts(b)}] {type(e).__name__}: {e}")
            return "".join(parts).strip()
        finally:
            self._transcription_lock.release()

    # ------------------------------------------------------------------ #
    # Flask routes
    # ------------------------------------------------------------------ #

    def setup_routes(self):
        self.app.route("/")(self.root_page)
        self.app.route("/get_status", methods=["GET"])(self.get_status)
        self.app.route("/get_transcription", methods=["GET"])(self.transcribe_from_mp3)
        self.app.route("/upload_and_transcribe", methods=["POST"])(self.upload_and_transcribe)
        self.app.route("/enroll_voice", methods=["POST"])(self.enroll_voice)

    def enroll_voice(self):
        """POST /enroll_voice with `file` = 1-3 min of a single speaker.

        Computes a speaker embedding and stores it as voiceprint.json in
        MEDIA_FOLDER. From then on diarization auto-runs on every file and
        the matching speaker is labeled with SPEAKER_NAME.
        """
        uploaded = request.files.get("file")
        if uploaded is None or uploaded.filename == "":
            return jsonify(
                {"status": "ERROR", "message": "No file provided (expected multipart field 'file')"}
            ), 400

        name = (request.values.get("name") or self.speaker_name).strip()
        with self._transcription_lock:
            with tempfile.TemporaryDirectory(prefix="enroll_") as tmp:
                audio_path = Path(tmp) / secure_filename(uploaded.filename or "sample.wav")
                uploaded.save(audio_path)
                self.logger.info(f"Enrolling voiceprint from {audio_path.name} for '{name}'")
                try:
                    audio = self.decode_audio(audio_path)
                    pipeline = self._get_diarize_pipeline()
                    df, embeddings = pipeline(
                        audio,
                        min_speakers=1,
                        max_speakers=max(1, int(request.values.get("max_speakers", 2))),
                        return_embeddings=True,
                    )
                except Exception as e:
                    self.logger.exception(f"Enrollment failed: {e}")
                    return jsonify({"status": "ERROR", "message": f"Enrollment failed: {type(e).__name__}: {e}"}), 500
                finally:
                    self._release_diarizer()

                # Use the embedding of the dominant speaker (longest speaking
                # time). Embedding keys vary by whisperx version: raw labels
                # ('A') in some, public names ('SPEAKER_00') in others —
                # try dominant label, then its public speaker, then fallback.
                if embeddings is None or len(embeddings) == 0:
                    return jsonify({"status": "ERROR", "message": "No speaker embeddings extracted"}), 500
                durations = (
                    df.groupby("label")["end"].max() - df.groupby("label")["start"].min()
                )
                dominant_label = durations.idxmax()
                emb = embeddings.get(dominant_label)
                if emb is None:
                    rows = df.loc[df["label"] == dominant_label, "speaker"]
                    if len(rows):
                        emb = embeddings.get(rows.iloc[0])
                if emb is None and len(embeddings) == 1:
                    emb = next(iter(embeddings.values()))
                if emb is None:
                    return jsonify({
                        "status": "ERROR",
                        "message": f"Dominant speaker '{dominant_label}' has no embedding (keys: {list(embeddings.keys())})",
                    }), 500

                self._save_voiceprint(name, list(emb))
                self.logger.info(
                    f"Voiceprint saved: '{name}' from label {dominant_label} ({len(emb)} dim)"
                )
                return jsonify(
                    {
                        "status": "OK",
                        "message": f"Voiceprint enrolled for '{name}' (dominant speaker {dominant_label}, {len(embeddings)} speaker(s) found)",
                    }
                )

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
            return jsonify(
                {"status": "ERROR", "message": f"Error in transcription process: {error_message}"}
            )

        lock_path = self.media_folder / status_filename
        if lock_path.exists():
            payload = {"status": "IN PROGRESS"}
            with contextlib.suppress(Exception):
                payload.update(json.loads(lock_path.read_text()))
            return jsonify(payload)

        transcription_file = self.media_folder / f"{base_path.stem}{self.transcription_ext}"
        try:
            transcription_text = transcription_file.read_text()
        except FileNotFoundError as e:
            self.logger.exception(f"Transcription {transcription_file} not found\n{e}")
            return jsonify(
                {"status": "ERROR", "message": f"Transcription {transcription_file} not found"}
            )

        # Result is consumed on first read (keeps the legacy contract).
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(self.media_folder / f"{base_path.stem}{self.chunks_ext}")
        with contextlib.suppress(FileNotFoundError):
            transcription_file.unlink()

        return jsonify({"status": "OK", "transcription_text": transcription_text.strip()})

    def transcribe_from_mp3(self):
        audio_name = request.values.get("audio_name")
        if not audio_name:
            return jsonify({"status": "ERROR", "message": "No audio_name provided"})

        filepath = self.media_folder / audio_name
        if not filepath.exists():
            return jsonify({"status": "ERROR", "message": f"File {filepath} not found on host"})

        with self._threads_lock:
            thread = self.active_threads.get(audio_name)
            if thread is not None and thread.is_alive():
                return jsonify(
                    {"status": "IN_PROGRESS", "message": f"Transcription for {audio_name} is already running"}
                )

        speakers_count = request.values.get("speakers_count", type=int)
        initial_prompt = request.values.get("initial_prompt") or None
        hotwords = request.values.get("hotwords") or None
        self.logger.info(
            f"Starting transcription for {audio_name} (speakers_count={speakers_count}, "
            f"custom_prompt={initial_prompt is not None}, custom_hotwords={hotwords is not None})"
        )
        transcribe_thread = threading.Thread(
            target=self.transcribe,
            name=f"Transcriber {audio_name}",
            args=(audio_name, speakers_count, initial_prompt, hotwords),
            daemon=True,
        )
        transcribe_thread.start()
        with self._threads_lock:
            self.active_threads[audio_name] = transcribe_thread

        statusfile = f"{audio_name}{self.lock_ext}"
        with contextlib.suppress(OSError):
            (self.media_folder / statusfile).touch()

        return jsonify({"status": "OK", "statusfile": statusfile})

    def upload_and_transcribe(self):
        uploaded = request.files.get("file")
        if uploaded is None or uploaded.filename == "":
            return jsonify(
                {"status": "ERROR", "message": "No file provided (expected multipart field 'file')"}
            ), 400

        filename = secure_filename(uploaded.filename) or "upload.bin"
        speakers_count = request.values.get("speakers_count", type=int)
        initial_prompt = request.values.get("initial_prompt") or None
        hotwords = request.values.get("hotwords") or None

        with tempfile.TemporaryDirectory(prefix="transcribe_") as tmp:
            audio_path = Path(tmp) / filename
            uploaded.save(audio_path)
            self.logger.info(
                f"Uploaded {filename} ({audio_path.stat().st_size} bytes) -> {audio_path}"
            )
            try:
                text = self.transcribe_file_sync(
                    audio_path=audio_path,
                    speakers_count=speakers_count,
                    lock_timeout=10.0,
                    initial_prompt=initial_prompt,
                    hotwords=hotwords,
                )
            except Exception as e:
                self.logger.exception(f"upload_and_transcribe failed for {filename}: {e}")
                return jsonify({"status": "ERROR", "message": f"{type(e).__name__}: {e}"}), 500

        if text is None:
            return jsonify(
                {"status": "BUSY", "message": "Another transcription is in progress, try again later"}
            ), 503
        return jsonify({"status": "OK", "transcription_text": text})

    def run(self):
        self.app.run(debug=bool(self.debug), host="::", port=self.port)


def main():
    app = TranscriberApp()
    # Legacy build-time warm-up: `python app_transcriber.py load` fills the
    # HF cache and exits (no server).
    if len(sys.argv) > 1 and sys.argv[1] == "load":
        app.preload_models()
        return
    if environ.get("PRELOAD", "").lower() in ("1", "true", "yes"):
        app.preload_models()
    app.run()


if __name__ == "__main__":
    if environ.get("MODE") == "DEV":
        load_dotenv()
    main()
