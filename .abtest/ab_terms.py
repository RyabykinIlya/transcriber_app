"""A/B #2: does term biasing fix real spelling of brands/terms?

Compares baseline / initial_prompt / hotwords / both on the same excerpts,
counting exact spellings of the terms that were mangled in production.
"""
import subprocess
import time
from pathlib import Path

import torch
import whisperx
import numpy as np

SR = 16000
FILE = Path("/records/IB_CORP_COM_CHATBOT_2026-10-22 15-06-30.mp3")
EXCERPTS = [(0, 300), (900, 1200), (2000, 2293)]

GLOSSARY = (
    "ЗНИ, ФТТ, ДНИ, ИТ, КСПД, DMZ, СЭД, архсовет, архитектор, "
    "чат-бот, сценарий, виджет, рантайм, роутер, база знаний, "
    "векторная база, Chroma, OpenRouter, Runway, LLM, SLM, "
    "кредитный скоринг, кредитное управление, корпоративные коммуникации, "
    "интеграционная спека, бизнес-процесс, ИБ, C#, C-Sharp, fastapp"
)

CONFIGS = {
    "A_plain": {},
    "B_prompt": {"asr_options": {"initial_prompt": GLOSSARY}},
    "C_hotwords": {"asr_options": {"hotwords": GLOSSARY}},
    "D_both": {"asr_options": {"initial_prompt": GLOSSARY, "hotwords": GLOSSARY}},
}

# (label, variants that count as CORRECT, variants that count as MANGLED)
TERMS = [
    ("Runway", ["Runway"], ["runway", "Ранвей", "Раневей", "ранвей"]),
    ("OpenRouter", ["OpenRouter"], ["openrouter", "Опенроутер", "опенроутер", "Open Router"]),
    ("Chroma", ["Chroma"], ["Хрома", "хрома", "Хром", "chroma", "Крома"]),
    ("ЗНИ", ["ЗНИ"], ["ЗНи", "Зни", "знание"]),
    ("ФТТ", ["ФТТ"], ["ФТТ", "фтт"]),
    ("SLM", ["SLM"], ["SLM", "СЛМ", "слм"]),
    ("LLM", ["LLM"], ["LLM", "ЛЛМ", "ллм"]),
    ("C-Sharp", ["C-Sharp", "C#", "Си Шарп", "C-шарп"], ["си шарп", "СИ ШАРП"]),
    ("OpenRouter-x", [], []),
]


def decode(path: Path) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
           "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0


def run_cfg(audio, name, cfg):
    t0 = time.time()
    model = whisperx.load_model("large-v3", "cuda", language="ru", compute_type="float16",
                                asr_options=dict(cfg.get("asr_options", {})))
    texts = []
    for a, b in EXCERPTS:
        res = model.transcribe(audio[a * SR: b * SR], language="ru", batch_size=4)
        texts.extend((s.get("text") or "").strip() for s in res["segments"])
    del model
    torch.cuda.empty_cache()
    joined = " ".join(t for t in texts if t)
    return joined, time.time() - t0


def main():
    audio = decode(FILE)
    outputs = {}
    for name, cfg in CONFIGS.items():
        joined, dt = run_cfg(audio, name, cfg)
        outputs[name] = joined
        print(f"[{name}] {dt:.0f}s, {len(joined)} chars", flush=True)

    print("\n=== TERM SPELLINGS (correct / mangled) ===")
    hdr = f"{'term':12s}" + "".join(f"{n[2:]:>16s}" for n in CONFIGS)
    print(hdr)
    totals = {n: 0 for n in CONFIGS}
    for label, good, bad in TERMS:
        row = f"{label:12s}"
        for n in CONFIGS:
            j = outputs[n]
            g = sum(j.count(v) for v in good)
            b = sum(j.count(v) for v in bad)
            totals[n] += g * 2 - b
            row += f"{g:>7d} ok/{b:<3d}bad"
        print(row)
    print("\nweighted score (2*good - bad):")
    for n, v in totals.items():
        print(f"  {n:12s} {v}")

    print("\n=== SAMPLE: excerpt where brands appear ===")
    for n in CONFIGS:
        j = outputs[n]
        idx = j.lower().find("runway")
        print(f"\n--- {n} ---\n{j[max(0, idx - 200): idx + 400] if idx >= 0 else '(no runway mentioned)'}")


if __name__ == "__main__":
    main()
