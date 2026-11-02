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

import voiceprints

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
        self.speakers_ext = "_speakers.json"
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
        self._voiceprints: Optional[List[Dict]] = None  # lazy, from voiceprints.json
        self._gallery_lock = threading.Lock()
        self.speaker_name = environ.get("SPEAKER_NAME", "Илья Рябыкин")
        # Calibration (cosine): own clean voice ~1.0, own voice in a real
        # meeting ~0.71 (worse mic/compression), other people 0.13-0.37.
        # 0.60 sits in the middle of that gap.
        self.voice_match_threshold = float(environ.get("VOICE_MATCH_THRESHOLD", "0.60"))
        # Сколько голосов максимум искать в записи. Раньше здесь жёстко стояло
        # 3 (и это резало любую встречу до трёх голосов, потому что
        # tg_assistant не передаёт speakers_count). Теперь решает env.
        self.diarize_max_speakers = max(1, _env_int("DIARIZE_MAX_SPEAKERS", 8))

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
    # Speaker identification (voice gallery)
    # ------------------------------------------------------------------ #
    #
    # Галерея голосов: MEDIA_FOLDER/voiceprints.json (см. VOICEPRINTS_CONTRACT.md).
    # Вся логика (слияние, матчинг, окна) — в app/voiceprints.py, здесь только
    # файловый ввод-вывод, кэш и работа с пайплайном.

    VOICEPRINTS_FILE = "voiceprints.json"
    # Легаси-файл одного слепка: читается как галерея из одной записи и НЕ
    # удаляется — просто перестаёт использоваться после первой записи галереи.
    VOICEPRINT_FILE = "voiceprint.json"

    def _gallery_path(self) -> Path:
        return self.media_folder / self.VOICEPRINTS_FILE

    def _legacy_voiceprint_path(self) -> Path:
        return self.media_folder / self.VOICEPRINT_FILE

    def _load_gallery(self) -> List[Dict]:
        """Текущая галерея (кэш в памяти, инвалидируется записью).

        Если `voiceprints.json` нет или он пуст, читается легаси
        `voiceprint.json` как галерея из одной записи с именем `SPEAKER_NAME`.
        """
        with self._gallery_lock:
            if self._voiceprints is not None:
                return self._voiceprints

            gallery = voiceprints.load_gallery_file(self._gallery_path())
            if gallery:
                self.logger.info(
                    f"Voice gallery loaded: {len(gallery)} speaker(s) from {self.VOICEPRINTS_FILE}"
                )
            else:
                gallery = voiceprints.load_gallery_file(self._legacy_voiceprint_path())
                if gallery:
                    for entry in gallery:
                        if not entry.get("name"):
                            entry["name"] = self.speaker_name
                    self.logger.info(
                        f"Voice gallery: using legacy {self.VOICEPRINT_FILE} as a "
                        f"single-entry gallery ('{gallery[0]['name']}'); it is ignored "
                        f"once {self.VOICEPRINTS_FILE} is written"
                    )
            self._voiceprints = gallery
            return gallery

    def _save_gallery(self, gallery: List[Dict]) -> None:
        with self._gallery_lock:
            voiceprints.save_gallery_file(self._gallery_path(), gallery)
            self._voiceprints = list(gallery)

    def _speakers_path(self, audio_stem: str) -> Path:
        return self.media_folder / f"{audio_stem}{self.speakers_ext}"

    def _match_speakers(self, df, embeddings: Dict) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, str], Dict]:
        """Сопоставить кластеры диаризации с галереей, залогировать каждый скор.

        Возвращает (renames, scores, names, vectors) — см. voiceprints.identify_speakers.
        """
        gallery = self._load_gallery()
        if not embeddings:
            return {}, {}, {}, {}

        # У сегмента в df две метки: сырая `label` (ключ эмбеддингов) и
        # публичная `speaker` (SPEAKER_XX) — переименовываем публичную.
        label2speaker: Dict[str, str] = {}
        for row in df.to_dict("records"):
            label2speaker.setdefault(str(row.get("label")), str(row.get("speaker")))

        renames, scores, names, vectors = voiceprints.identify_speakers(
            embeddings, label2speaker, gallery, self.voice_match_threshold
        )
        for speaker, score in sorted(scores.items()):
            matched = names.get(speaker)
            level = " => matched" if matched else " (below threshold)"
            self.logger.info(
                f"Voice match {speaker} vs '{matched or 'gallery'}': "
                f"cos={score:.3f}{level} ({len(gallery)} in gallery)"
            )
        return renames, scores, names, vectors

    @staticmethod
    def _clear_speakers_file(path: Path) -> None:
        """Убрать устаревший `_speakers.json` перед новой расшифровкой."""
        with contextlib.suppress(FileNotFoundError):
            path.unlink()

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

        # Диаризация: запускается, если её попросили явно ИЛИ если галерея
        # голосов непуста (тогда она обязана отработать, чтобы подписать
        # знакомые голоса именами).
        gallery = self._load_gallery()
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

        diarization = None
        speaker_renames: Dict[str, str] = {}
        speaker_scores: Dict[str, float] = {}
        speaker_names: Dict[str, str] = {}
        speaker_vectors: Dict[str, np.ndarray] = {}
        if (speakers_from_to or gallery) and any(all_segments):
            try:
                if progress:
                    progress(stage="diarization")
                # Раньше здесь безусловно стояло max_speakers=3, и любая
                # встреча без явного speakers_count резалась на три голоса.
                # Теперь дефолт — DIARIZE_MAX_SPEAKERS (8).
                min_speakers = speakers_from_to[0] if speakers_from_to else 1
                max_speakers = (
                    speakers_from_to[1] if speakers_from_to else self.diarize_max_speakers
                )
                self.logger.info(
                    f"Diarization over the whole file "
                    f"(speakers {min_speakers}..{max_speakers})"
                )
                pipeline = self._get_diarize_pipeline()
                diarization, speaker_embeddings = pipeline(
                    audio,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    return_embeddings=True,
                )
                self.logger.info("Diarization finished for the whole file")
                if speaker_embeddings is not None:
                    self.logger.info(f"Speaker labels found: {list(speaker_embeddings.keys())}")
                # Векторы спикеров отдаём наружу даже при пустой галерее: по ним
                # tg_assistant делает ретро-разметку, когда коллегу зачислят позже.
                if speaker_embeddings:
                    (
                        speaker_renames,
                        speaker_scores,
                        speaker_names,
                        speaker_vectors,
                    ) = self._match_speakers(diarization, speaker_embeddings)
            except Exception as e:
                self.logger.exception(
                    f"Diarization failed ({type(e).__name__}: {e}); continuing without speakers"
                )
                diarization = None
            finally:
                # Always hand the ~2.4 GB back: the next file in the queue
                # must not inherit a nearly-full GPU (see _release_diarizer).
                self._release_diarizer()

        # Speaker labels + gallery renaming, then emit everything in order.
        # Строки для `speakers[]` собираются ДО переименования: контракт
        # требует отдавать исходную метку SPEAKER_XX (по ней клиент ищет
        # реплики в тексте).
        speaker_rows: List[dict] = []
        for segments in all_segments:
            if diarization is not None and segments:
                try:
                    assigned = whisperx.assign_word_speakers(
                        diarization, {"segments": segments}
                    )
                    segments = assigned.get("segments", segments)
                except Exception as e:
                    self.logger.warning(f"Speaker assignment failed for a chunk: {e}")
            for seg in segments:
                spk = seg.get("speaker")
                if spk:
                    with contextlib.suppress(TypeError, ValueError):
                        speaker_rows.append(
                            {
                                "speaker": str(spk),
                                "start": float(seg.get("start", 0.0)),
                                "end": float(seg.get("end", 0.0)),
                            }
                        )
                if speaker_renames and spk in speaker_renames:
                    seg["speaker"] = speaker_renames[spk]
                sink(seg)

        speakers = (
            voiceprints.build_speakers(
                speaker_rows, speaker_vectors, speaker_scores, speaker_names
            )
            if speaker_rows
            else []
        )
        if speakers:
            self.logger.info(
                "Speakers: "
                + ", ".join(
                    f"{s['label']}={s['name'] or '?'} ({s['total_sec']}s, "
                    f"{len(s['windows'])} window(s))"
                    for s in speakers
                )
            )

        self.logger.info(
            f"Transcription finished: {len(ranges)} chunk(s), {len(failed)} failed, "
            f"{time.monotonic() - started:.1f}s total"
        )
        return failed, speakers

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

                speakers_from_to = (
                    (1, max(1, speakers_count)) if speakers_count else None
                )
                # Устаревший _speakers.json от прошлого прогона этого же файла
                # иначе переживёт новую расшифровку и соврёт про спикеров.
                speakers_file = self._speakers_path(audio_stem)
                self._clear_speakers_file(speakers_file)

                def sink(segment: dict) -> None:
                    f.write(self._format_segment(segment))

                with transcription_file.open("w") as f:
                    with self._override_asr_options(initial_prompt, hotwords):
                        failed, speakers = self.transcribe_array(
                            audio,
                            speakers_from_to,
                            sink,
                            progress=lambda *a, **kw: self._write_progress(lock_file, *a, **kw),
                        )
                    f.flush()
                del audio

                # _speakers.json переживает выдачу расшифровки — клиент опрашивает
                # спикеров отдельно от текста (см. /get_status).
                voiceprints.save_speakers_file(speakers_file, speakers)

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
    ) -> Optional[Tuple[str, List[dict]]]:
        """Synchronous pipeline for /upload_and_transcribe.

        Возвращает `(текст, speakers)` либо None, если не дождались лока.
        `speakers` — тот же массив, что отдаёт /get_status (контракт §1.3).
        """
        acquired = self._transcription_lock.acquire(timeout=lock_timeout)
        if not acquired:
            return None
        try:
            audio = self.decode_audio(audio_path)
            speakers_from_to = (1, max(1, speakers_count)) if speakers_count else None
            parts: List[str] = []
            with self._override_asr_options(initial_prompt, hotwords):
                failed, speakers = self.transcribe_array(
                    audio, speakers_from_to, lambda seg: parts.append(self._format_segment(seg))
                )
            del audio
            if failed and not parts:
                a, b, e = failed[0]
                raise RuntimeError(f"All chunks failed; first: [{fmt_ts(a)}-{fmt_ts(b)}] {type(e).__name__}: {e}")
            return "".join(parts).strip(), speakers
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
        self.app.route("/voiceprints", methods=["GET"])(self.list_voiceprints)
        self.app.route("/voiceprints/delete", methods=["POST"])(self.delete_voiceprint)
        self.app.route("/voiceprints/match", methods=["GET", "POST"])(self.match_voiceprint)

    # --- галерея голосов ---------------------------------------------- #

    def _gallery_payload(self) -> List[Dict]:
        """Публичное представление галереи: без эмбеддингов, по имени."""
        entries = []
        for entry in self._load_gallery():
            entries.append(
                {
                    "id": entry.get("id"),
                    "name": entry.get("name"),
                    "samples": int(entry.get("samples") or 0),
                    "sample_seconds": float(entry.get("sample_seconds") or 0.0),
                    "created_at": entry.get("created_at") or "",
                    "updated_at": entry.get("updated_at") or "",
                }
            )
        entries.sort(key=lambda item: str(item.get("name") or "").casefold())
        return entries

    def list_voiceprints(self):
        """GET /voiceprints — галерея без эмбеддингов, по имени."""
        return jsonify({"status": "OK", "speakers": self._gallery_payload()})

    def delete_voiceprint(self):
        """POST /voiceprints/delete — удалить запись по `id` или по `name`."""
        payload = request.get_json(silent=True) or {}
        entry_id = request.values.get("id") or payload.get("id")
        name = request.values.get("name") or payload.get("name")
        if not entry_id and not name:
            return jsonify({"status": "ERROR", "message": "No id or name provided"}), 400

        gallery = self._load_gallery()
        kept, deleted = voiceprints.delete_speaker(gallery, entry_id=entry_id, name=name)
        if deleted is None:
            return jsonify({"status": "ERROR", "message": "Not found"}), 404
        self._save_gallery(kept)
        self.logger.info(f"Voiceprint deleted: {deleted} (left {len(kept)} in gallery)")
        return jsonify({"status": "OK", "deleted": deleted})

    def match_voiceprint(self):
        """GET /voiceprints/match?embedding=<json> — косинус против всей галереи."""
        raw = request.values.get("embedding")
        if not raw:
            payload = request.get_json(silent=True) or {}
            raw = payload.get("embedding")
        if raw is None:
            return jsonify({"status": "ERROR", "message": "No embedding provided"}), 400
        try:
            embedding = json.loads(raw) if isinstance(raw, str) else raw
            probe = np.asarray(embedding, dtype=np.float32).reshape(-1)
        except (ValueError, TypeError) as e:
            return jsonify({"status": "ERROR", "message": f"Bad embedding: {e}"}), 400
        if probe.size == 0:
            return jsonify({"status": "ERROR", "message": "Bad embedding: empty"}), 400

        matches = voiceprints.score_gallery(probe, self._load_gallery())
        return jsonify({"status": "OK", "matches": matches})

    def enroll_voice(self):
        """POST /enroll_voice — зачислить голос в галерею.

        Источник эмбеддинга — одно из двух:
          * `audio_name` (файл в MEDIA_FOLDER) + `windows` = JSON-массив
            [[start_sec, end_sec], ...]. Окна склеиваются с 0.15 с тишины между
            ними, поэтому хватает нескольких коротких чистых реплик;
          * `file` — загруженный файл (легаси-путь, как раньше).

        Вектор считается диаризатором по склейке с max_speakers=1 и пишется в
        MEDIA_FOLDER/voiceprints.json (повторное имя усредняет эмбеддинг).
        """
        name = (request.values.get("name") or "").strip()
        if not name:
            return jsonify(
                {"status": "ERROR", "message": "No name provided (expected field 'name')"}
            ), 400

        audio_name = (request.values.get("audio_name") or "").strip()
        windows_raw = request.values.get("windows")
        uploaded = request.files.get("file")
        has_file = uploaded is not None and uploaded.filename != ""

        if not audio_name and not has_file:
            return jsonify(
                {
                    "status": "ERROR",
                    "message": "No source provided (expected 'audio_name' + 'windows' or 'file')",
                }
            ), 400

        if audio_name:
            source_path = self.media_folder / audio_name
            if not source_path.exists():
                return jsonify(
                    {"status": "ERROR", "message": f"File {audio_name} not found on host"}
                ), 404
            try:
                raw_windows = json.loads(windows_raw or "")
            except ValueError as e:
                return jsonify(
                    {"status": "ERROR", "message": f"Bad 'windows' JSON: {e}"}
                ), 400
            if not isinstance(raw_windows, list) or not raw_windows:
                return jsonify(
                    {"status": "ERROR", "message": "No windows provided (expected [[start,end],...])"}
                ), 400
            # Для окон всегда одна диаризация в один кластер (контракт §1).
            max_speakers = 1
        else:
            raw_windows = None
            max_speakers = max(1, _env_int("ENROLL_MAX_SPEAKERS", 2))
            requested = request.values.get("max_speakers")
            if requested:
                with contextlib.suppress(TypeError, ValueError):
                    max_speakers = max(1, int(float(requested)))

        with self._transcription_lock:
            tmp_ctx = tempfile.TemporaryDirectory(prefix="enroll_")
            try:
                with tmp_ctx as tmp:
                    if audio_name:
                        decode_path = source_path
                    else:
                        decode_path = Path(tmp) / secure_filename(uploaded.filename or "sample.wav")
                        uploaded.save(decode_path)

                    try:
                        audio = self.decode_audio(decode_path)
                    except Exception as e:
                        self.logger.exception(f"Enrollment: cannot decode source for '{name}'")
                        return jsonify(
                            {"status": "ERROR", "message": f"Cannot decode audio: {type(e).__name__}: {e}"}
                        ), 500

                    duration = audio.size / SAMPLE_RATE
                    if audio_name:
                        windows = voiceprints.normalize_windows(raw_windows, duration)
                    else:
                        windows = voiceprints.normalize_windows([[0.0, duration]], duration)
                    speech_seconds = voiceprints.windows_total_seconds(windows)
                    if speech_seconds < voiceprints.ENROLL_MIN_TOTAL_SECONDS:
                        return jsonify(
                            {
                                "status": "ERROR",
                                "message": f"Too little speech: {speech_seconds:.1f}s",
                            }
                        ), 400

                    concat, placements = voiceprints.concat_windows(audio, windows)
                    del audio
                    self.logger.info(
                        f"Enrolling '{name}' from {len(windows)} window(s) "
                        f"({voiceprints.windows_total_seconds(windows):.1f}s of speech, "
                        f"{concat.size / SAMPLE_RATE:.1f}s concatenated, "
                        f"max_speakers={max_speakers})"
                    )
                    try:
                        pipeline = self._get_diarize_pipeline()
                        df, embeddings = pipeline(
                            concat,
                            min_speakers=1,
                            max_speakers=max_speakers,
                            return_embeddings=True,
                        )
                    except Exception as e:
                        self.logger.exception(f"Enrollment failed for '{name}': {e}")
                        return jsonify(
                            {"status": "ERROR", "message": f"Enrollment failed: {type(e).__name__}: {e}"}
                        ), 500
                    finally:
                        # Диаризатор держит ~2.4 ГБ VRAM — возвращаем всегда.
                        self._release_diarizer()
                        del concat

                    if not embeddings:
                        return jsonify(
                            {"status": "ERROR", "message": "No speaker embeddings extracted"}
                        ), 500
                    rows = df.to_dict("records") if hasattr(df, "to_dict") else []
                    dominant_label, vector = voiceprints.dominant_embedding(rows, embeddings)
                    if vector is None:
                        return jsonify(
                            {
                                "status": "ERROR",
                                "message": f"No embedding for dominant speaker "
                                f"'{dominant_label}' (keys: {list(embeddings.keys())})",
                            }
                        ), 500

                    gallery, entry = voiceprints.upsert_speaker(
                        self._load_gallery(), name, vector, sample_seconds=speech_seconds
                    )
                    self._save_gallery(gallery)
                    self.logger.info(
                        f"Voiceprint saved: '{name}' id={entry['id']} "
                        f"(label {dominant_label}, {len(embeddings)} cluster(s), "
                        f"samples={entry['samples']})"
                    )
                    return jsonify(
                        {
                            "status": "OK",
                            "id": entry["id"],
                            "name": entry["name"],
                            "samples": entry["samples"],
                            "sample_seconds": entry["sample_seconds"],
                        }
                    )
            finally:
                tmp_ctx.cleanup()
                self._cleanup_gpu()

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

        # `speakers` читаем ДО удаления текста: _speakers.json переживает выдачу
        # расшифровки, чтобы клиент мог разметить спикеров после первого чтения.
        speakers = voiceprints.load_speakers_file(self._speakers_path(base_path.stem))

        # Result is consumed on first read (keeps the legacy contract).
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(self.media_folder / f"{base_path.stem}{self.chunks_ext}")
        with contextlib.suppress(FileNotFoundError):
            transcription_file.unlink()

        return jsonify(
            {
                "status": "OK",
                "transcription_text": transcription_text.strip(),
                "speakers": speakers,
            }
        )

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
        transcription_text, speakers = text
        return jsonify(
            {
                "status": "OK",
                "transcription_text": transcription_text,
                "speakers": speakers or [],
            }
        )

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
