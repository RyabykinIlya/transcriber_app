"""A/B: whisperx decoding options on real meeting fragments.

Metrics without ground truth: chars, segments, chars/segment, duplicate
segments (repetition loops), known-hallucination phrases, glossary hits.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import whisperx

SR = 16000
FILE = Path("/records/IB_CORP_COM_CHATBOT_2026-10-22 15-06-30.mp3")

EXCERPTS = [(0, 300), (900, 1200), (2000, 2293)]

GLOSSARY = (
    "ЗНИ, ФТТ, ДНИ, ИТ, КСПД, DMZ, СЭД, СМЭВ, архсовет, архитектор, "
    "чат-бот, сценарий, виджет, рантайм, роутер, база знаний, "
    "векторная база, Chroma, OpenRouter, Runway, LLM, SLM, "
    "кредитный скоринг, кредитное управление, корпоративные коммуникации, "
    "интеграционная спека, бизнес-процесс, ИБ"
)

CONFIGS = {
    "A_baseline": {"asr_options": {"beam_size": 5}},
    "B_beam8": {"asr_options": {"beam_size": 8}},
    "C_glossary": {"asr_options": {"beam_size": 5, "initial_prompt": GLOSSARY}},
    "D_beam8_gloss_anti": {
        "asr_options": {
            "beam_size": 8,
            "initial_prompt": GLOSSARY,
            "hallucination_silence_threshold": 2.0,
        },
        "vad_options": {"vad_onset": 0.45, "vad_offset": 0.30},
    },
}

HALLUCINATION_MARKERS = [
    "продолжение следует",
    "подписывайтесь",
    "спасибо за просмотр",
    "редактор субтитров",
    "ставьте лайк",
    "субтитры сделал",
    "продолжение в следующем",
]

GLOSSARY_TERMS = ["ЗНИ", "ФТТ", "ДНИ", "КСПД", "DMZ", "архсовет", "Chroma",
                  "OpenRouter", "Runway", "скоринг", "виджет", "база знаний"]


def decode(path: Path) -> np.ndarray:
    import subprocess
    import shutil
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
           "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0


def score(segments) -> dict:
    texts = [(s.get("text") or "").strip() for s in segments]
    texts_nonempty = [t for t in texts if t]
    joined = " ".join(texts_nonempty).lower()
    dup_runs = 0
    for i in range(1, len(texts_nonempty)):
        if texts_nonempty[i] and texts_nonempty[i] == texts_nonempty[i - 1]:
            dup_runs += 1
    counts = {}
    for t in texts_nonempty:
        counts[t] = counts.get(t, 0) + 1
    heavy_repeats = sum(1 for t, c in counts.items() if c >= 3)
    halluc = sum(joined.count(m) for m in HALLUCINATION_MARKERS)
    gloss = sum(joined.count(t.lower()) for t in GLOSSARY_TERMS)
    return {
        "chars": sum(len(t) for t in texts_nonempty),
        "segments": len(texts),
        "chars_per_seg": (sum(len(t) for t in texts_nonempty) / max(len(texts_nonempty), 1)),
        "dup_runs": dup_runs,
        "heavy_repeats": heavy_repeats,
        "halluc": halluc,
        "glossary_hits": gloss,
        "tiny_segs": sum(1 for t in texts_nonempty if len(t) <= 12),
    }


def main() -> None:
    audio = decode(FILE)
    print(f"decoded {audio.size / SR:.0f}s", flush=True)
    results = {}
    texts_by_cfg = {}

    for name, cfg in CONFIGS.items():
        t0 = time.time()
        model = whisperx.load_model(
            "large-v3", "cuda", language="ru", compute_type="float16",
            asr_options=dict(cfg.get("asr_options", {})),
            vad_options=dict(cfg.get("vad_options", {})),
        )
        load_s = time.time() - t0
        agg = {k: 0 for k in ("chars", "segments", "dup_runs", "heavy_repeats",
                              "halluc", "glossary_hits", "tiny_segs")}
        per_excerpt = []
        rt = 0.0
        for a, b in EXCERPTS:
            sl = audio[a * SR: b * SR]
            t1 = time.time()
            res = model.transcribe(sl, language="ru", batch_size=4)
            rt += time.time() - t1
            sc = score(res["segments"])
            per_excerpt.append(sc)
            for k in agg:
                agg[k] += sc[k]
            if a == 2000:
                texts_by_cfg[name] = " || ".join(
                    (s.get("text") or "").strip() for s in res["segments"]
                )[:700]
        total_audio = sum(b - a for a, b in EXCERPTS)
        agg["chars_per_seg"] = agg["chars"] / max(agg["segments"], 1)
        agg["rt_factor"] = total_audio / rt
        agg["load_s"] = load_s
        results[name] = agg
        print(f"[{name}] done in {rt:.0f}s ({agg['rt_factor']:.0f}x RT), load {load_s:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()

    print("\n=== METRICS ===")
    header = f"{'config':22s} {'chars':>7s} {'segs':>5s} {'c/seg':>6s} {'dups':>5s} {'heavy':>6s} {'halluc':>7s} {'gloss':>6s} {'tiny':>5s} {'xRT':>5s}"
    print(header)
    for name, m in results.items():
        print(f"{name:22s} {m['chars']:7d} {m['segments']:5d} {m['chars_per_seg']:6.1f} "
              f"{m['dup_runs']:5d} {m['heavy_repeats']:6d} {m['halluc']:7d} "
              f"{m['glossary_hits']:6d} {m['tiny_segs']:5d} {m['rt_factor']:5.0f}")

    print("\n=== END-OF-FILE TEXT (where hallucinations live) ===")
    for name in CONFIGS:
        if name in texts_by_cfg:
            print(f"\n--- {name} ---\n{texts_by_cfg[name]}")


if __name__ == "__main__":
    main()
