# -*- coding: utf-8 -*-
"""Чистая логика галереи голосов (без torch, whisperx и Flask).

Модуль намеренно не тянет ничего тяжёлого: всё, что здесь есть, — функции над
обычными списками и numpy-массивами, поэтому тесты гоняются на машине без GPU и
без установленного torch.

Что живёт здесь:
  * нормализация окон `[[start, end], ...]` из клиента (обрезка, отброс
    коротких, снятие наложений);
  * склейка окон в один массив с вставкой тишины и картой смещений;
  * выбор окон `windows` для `/get_status` (слияние смежных сегментов, лимиты);
  * галерея: косинус, L2-нормализация, лучшее совпадение по всей галерее,
    слияние эмбеддинга при повторном зачислении, удаление записи;
  * сборка массива `speakers` из результата диаризации;
  * тонкие обёртки файлового ввода-вывода для `MEDIA_FOLDER/voiceprints.json`
    и `{stem}_speakers.json`.

Контракт — `VOICEPRINTS_CONTRACT.md`, раздел «1. transcriber_app: галерея голосов».
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SAMPLE_RATE = 16000

# --- склейка окон для зачисления (контракт §1, шаг 1) --------------------- #
ENROLL_GAP_SECONDS = 0.15      # тишины между окнами при склейке
ENROLL_MIN_WINDOW_SECONDS = 0.4
ENROLL_MAX_WINDOW_SECONDS = 30.0
ENROLL_MIN_TOTAL_SECONDS = 3.0  # меньше — 400 Too little speech

# --- окна для /get_status (контракт §1.3) --------------------------------- #
RUN_GAP_SECONDS = 1.0        # зазор меньше — считаем смежностью, клеим
WINDOWS_LIMIT = 8
WINDOW_MIN_SECONDS = 6.0
WINDOW_MAX_SECONDS = 30.0
WINDOWS_TOTAL_SECONDS = 180.0

GALLERY_VERSION = 2
#: Максимальная доля чужой речи в склейке при зачислении.
#: Выше — зачисление отклоняется: слепок получится «средним по комнате».
ENROLL_MAX_FOREIGN_RATIO = 0.15
#: Два кластера с косинусом выше этого считаются одним человеком: pyannote
#: иногда режет один голос на два кластера (тихо/громко, близко/далеко от
#: микрофона), и без склейки «чужой речью» оказалась бы половина реплик.
SAME_VOICE_COSINE = 0.80


# ------------------------------------------------------------------------- #
# Окна: нормализация
# ------------------------------------------------------------------------- #

def normalize_windows(
    windows: Iterable[Any],
    duration: Optional[float] = None,
    min_seconds: float = ENROLL_MIN_WINDOW_SECONDS,
    max_seconds: float = ENROLL_MAX_WINDOW_SECONDS,
) -> List[Tuple[float, float]]:
    """Привести окна клиента к аккуратному списку (start < end, без наложений).

    Порядок ровно как в контракте: сначала отбрасываем мусор и обрезаем окна по
    границам файла (`duration`, если известно), потом снимаем наложения
    (перекрытие обрезается по началу следующего окна), затем окна длиннее
    `max_seconds` обрезаются с начала, и в конце отбрасываются окна короче
    `min_seconds`.

    Возвращает список `(start, end)` отсортированный по началу.
    """
    cleaned: List[List[float]] = []
    for item in windows or []:
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not (np.isfinite(start) and np.isfinite(end)):
            continue
        if start < 0.0:
            start = 0.0
        if duration is not None:
            end = min(end, float(duration))
        if end <= start:
            continue
        cleaned.append([start, end])

    cleaned.sort(key=lambda w: (w[0], w[1]))

    # Снятие наложений: каждое следующее окно не может начинаться раньше
    # конца уже принятого.
    resolved: List[List[float]] = []
    for start, end in cleaned:
        if resolved and start < resolved[-1][1]:
            start = resolved[-1][1]
        if end <= start:
            continue
        resolved.append([start, end])

    normalized: List[Tuple[float, float]] = []
    for start, end in resolved:
        end = min(end, start + max_seconds)  # обрезаем с начала окна
        if end - start < min_seconds:
            continue
        normalized.append((start, end))
    return normalized


def windows_total_seconds(windows: Sequence[Sequence[float]]) -> float:
    """Суммарная длительность окон (для порога «мало речи» и `sample_seconds`)."""
    return float(sum(max(0.0, float(e) - float(s)) for s, e in windows or []))


def concat_windows(
    audio: np.ndarray,
    windows: Sequence[Sequence[float]],
    sample_rate: int = SAMPLE_RATE,
    gap_seconds: float = ENROLL_GAP_SECONDS,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Склеить куски `audio` по окнам, вставив `gap_seconds` тишины между ними.

    Возвращает `(склейка, размещения)`, где размещение — словарь с исходным
    окном (`source_start`, `source_end`) и его местом в склейке (`out_start`,
    `out_end`) в секундах. По размещениям потом можно отлаживать, что именно
    ушло в диаризатор, и проверять смещения в тестах.
    """
    audio = np.asarray(audio).reshape(-1)
    gap = np.zeros(int(round(gap_seconds * sample_rate)), dtype=audio.dtype)
    parts: List[np.ndarray] = []
    placements: List[Dict[str, float]] = []
    cursor = 0  # позиция в склейке, в сэмплах

    for index, (start, end) in enumerate(windows or []):
        a = max(0, int(round(float(start) * sample_rate)))
        b = min(audio.size, int(round(float(end) * sample_rate)))
        if b <= a:
            continue
        if parts:
            parts.append(gap)
            cursor += gap.size
        length = b - a
        parts.append(audio[a:b])
        placements.append(
            {
                "index": float(index),
                "source_start": float(start),
                "source_end": float(end),
                "out_start": cursor / sample_rate,
                "out_end": (cursor + length) / sample_rate,
            }
        )
        cursor += length

    if not parts:
        return np.zeros(0, dtype=audio.dtype), []
    return np.concatenate(parts), placements


# ------------------------------------------------------------------------- #
# Окна для /get_status: непрерывные пробеги речи
# ------------------------------------------------------------------------- #

def merge_runs(
    segments: Iterable[Dict[str, Any]],
    gap_seconds: float = RUN_GAP_SECONDS,
    blockers: Optional[Iterable[Tuple[float, float]]] = None,
) -> List[Tuple[float, float]]:
    """Слить перекрывающиеся и смежные (зазор < `gap_seconds`) сегменты.

    `blockers` — интервалы, где говорит **кто-то другой**. Через такой зазор
    склеивать нельзя: в записях, где собеседники перебивают друг друга, пауза
    в 1 секунду часто занята чужой репликой («да», «ага», «в смысле»), и склейка
    подмешивает чужой голос в слепок. Замерено на реальной встрече: окно
    842–892 с содержало 2 с речи другого спикера.
    """
    spans: List[List[float]] = []
    for seg in segments or []:
        try:
            start, end = float(seg["start"]), float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        spans.append([start, end])
    spans.sort(key=lambda s: (s[0], s[1]))

    blocked = sorted(
        (float(a), float(b)) for a, b in (blockers or []) if float(b) > float(a)
    )

    def _foreign_between(left: float, right: float) -> bool:
        """Есть ли чужой голос в зазоре (left, right)."""
        for b_start, b_end in blocked:
            if b_end <= left:
                continue
            if b_start >= right:
                break
            return True
        return False

    merged: List[List[float]] = []
    for start, end in spans:
        if (
            merged
            and start - merged[-1][1] < gap_seconds
            and not _foreign_between(merged[-1][1], start)
        ):
            if end > merged[-1][1]:
                merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def subtract_blockers(
    windows: Iterable[Tuple[float, float]],
    blockers: Iterable[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Вырезать из окон чужую речь (на случай наложений от assign_word_speakers).

    Диаризация иногда отдаёт перекрывающиеся интервалы: один сегмент помечен
    сразу двумя голосами. Если такое окно уйдёт в склейку целиком, чужой голос
    попадёт в слепок — поэтому чужие интервалы физически вычитаются.
    """
    spans = [(float(a), float(b)) for a, b in windows or [] if float(b) > float(a)]
    cuts = sorted((float(a), float(b)) for a, b in blockers or [] if float(b) > float(a))
    if not cuts:
        return spans

    result: List[Tuple[float, float]] = []
    for start, end in spans:
        pieces = [(start, end)]
        for cut_start, cut_end in cuts:
            if cut_end <= start:
                continue
            if cut_start >= end:
                break
            next_pieces: List[Tuple[float, float]] = []
            for p_start, p_end in pieces:
                if cut_end <= p_start or cut_start >= p_end:
                    next_pieces.append((p_start, p_end))
                    continue
                if cut_start > p_start:
                    next_pieces.append((p_start, cut_start))
                if cut_end < p_end:
                    next_pieces.append((cut_end, p_end))
            pieces = next_pieces
        result.extend(pieces)
    return [(a, b) for a, b in result if b > a]


def select_windows(
    segments: Iterable[Dict[str, Any]],
    limit: int = WINDOWS_LIMIT,
    min_seconds: float = WINDOW_MIN_SECONDS,
    max_seconds: float = WINDOW_MAX_SECONDS,
    total_seconds: float = WINDOWS_TOTAL_SECONDS,
    gap_seconds: float = RUN_GAP_SECONDS,
    blockers: Optional[Iterable[Tuple[float, float]]] = None,
) -> List[List[float]]:
    """До `limit` самых длинных непрерывных окон речи, отсортированных по началу.

    Каждое окно ≤ `max_seconds` (обрезаем с начала пробега), ≥ `min_seconds`
    (короткие пробеги просто не попадают в результат), суммарно ≤
    `total_seconds` (последнее окно при необходимости подрезается, но не
    короче `min_seconds`). Это ровно то, что клиент попросит склеить при
    зачислении голоса.

    `blockers` — интервалы чужой речи. Через них не склеиваем (см. `merge_runs`)
    и потом дополнительно вычитаем: окно должно содержать только этот голос,
    иначе в слепок подмешается собеседник.
    """
    foreign = [
        (float(a), float(b)) for a, b in (blockers or []) if float(b) > float(a)
    ]
    candidates: List[Tuple[float, float]] = []
    for start, end in merge_runs(segments, gap_seconds, foreign):
        end = min(end, start + max_seconds)
        for clean_start, clean_end in subtract_blockers([(start, end)], foreign):
            if clean_end - clean_start >= min_seconds:
                candidates.append((clean_start, clean_end))

    candidates.sort(key=lambda w: (-(w[1] - w[0]), w[0]))

    chosen: List[Tuple[float, float]] = []
    total = 0.0
    for start, end in candidates:
        if len(chosen) >= limit:
            break
        remaining = total_seconds - total
        if remaining < min_seconds:
            break
        if end - start > remaining:
            end = start + remaining
        chosen.append((start, end))
        total += end - start

    chosen.sort(key=lambda w: w[0])
    return [[round(s, 3), round(e, 3)] for s, e in chosen]


# ------------------------------------------------------------------------- #
# Векторы, косинус, матчинг по галерее
# ------------------------------------------------------------------------- #

def l2_normalize(vector: Any) -> np.ndarray:
    """L2-нормализация; нулевой вектор возвращается как есть (без NaN)."""
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return vec
    return (vec / norm).astype(np.float32)


def cosine_similarity(a: Any, b: Any) -> float:
    """Косинус двух векторов; 0.0, если один из них нулевой."""
    va = np.asarray(a, dtype=np.float32).reshape(-1)
    vb = np.asarray(b, dtype=np.float32).reshape(-1)
    if va.size != vb.size or va.size == 0:
        return 0.0
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def entry_vector(entry: Dict[str, Any]) -> Optional[np.ndarray]:
    """Эмбеддинг записи галереи как numpy-вектор, либо None если его нет."""
    try:
        vec = np.asarray(entry.get("embedding"), dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    return vec if vec.size else None


def score_gallery(embedding: Any, gallery: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Все записи галереи с косинусом к `embedding`, по убыванию скора.

    Записи с другой размерностью молча пропускаются — их сравнивать нечем.
    """
    probe = np.asarray(embedding, dtype=np.float32).reshape(-1)
    scored: List[Dict[str, Any]] = []
    for entry in gallery or []:
        vec = entry_vector(entry)
        if vec is None or vec.size != probe.size:
            continue
        scored.append(
            {
                "id": entry.get("id"),
                "name": entry.get("name"),
                "score": cosine_similarity(probe, vec),
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def best_match(
    embedding: Any, gallery: Sequence[Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """Лучшее совпадение по *всей* галерее: (запись, скор) или (None, None)."""
    scored = score_gallery(embedding, gallery)
    if not scored:
        return None, None
    top = scored[0]
    return {"id": top["id"], "name": top["name"]}, float(top["score"])


def identify_speakers(
    embeddings_by_label: Dict[str, Any],
    label_to_speaker: Dict[str, str],
    gallery: Sequence[Dict[str, Any]],
    threshold: float,
) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, str], Dict[str, np.ndarray]]:
    """Сопоставить каждый `SPEAKER_XX` с галереей.

    Ключи `embeddings_by_label` — сырые метки диаризации (`'A'`, `'B'`, ...), в
    df у сегмента лежат и сырая `label`, и публичная `speaker` (`SPEAKER_XX`);
    переименование (и `speakers[].label`) работает с публичной.

    Возвращает `(renames, scores, names, vectors)`:
      * `renames[SPEAKER_XX]` — имя при `cos >= threshold`, иначе сама метка;
      * `scores[SPEAKER_XX]` — лучший косинус по галерее;
      * `names` — только совпавшие (для поля `name` в `/get_status`);
      * `vectors` — эмбеддинг спикера (для ретро-разметки на стороне tg_assistant).
    """
    renames: Dict[str, str] = {}
    scores: Dict[str, float] = {}
    names: Dict[str, str] = {}
    vectors: Dict[str, np.ndarray] = {}

    for label, emb in (embeddings_by_label or {}).items():
        try:
            vec = np.asarray(emb, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            continue
        if vec.size == 0:
            continue
        speaker = label_to_speaker.get(label, str(label))
        # Вектор отдаём наружу всегда, даже при пустой галерее: по нему
        # tg_assistant потом делает ретро-разметку (контракт §2).
        vectors.setdefault(speaker, vec)
        renames[speaker] = speaker
        entry, score = best_match(vec, gallery)
        if score is None:
            # галерея пуста или все записи другой размерности — сверять не с чем
            continue
        prev = scores.get(speaker)
        if prev is not None and prev >= score:
            continue  # по этому спикеру уже есть более близкая сырая метка
        scores[speaker] = score
        vectors[speaker] = vec
        if score >= threshold and entry and entry.get("name"):
            names[speaker] = str(entry["name"])
            renames[speaker] = str(entry["name"])
    return renames, scores, names, vectors


# ------------------------------------------------------------------------- #
# Галерея: зачисление, слияние, удаление
# ------------------------------------------------------------------------- #

def _now() -> str:
    import time

    return time.strftime("%Y-%m-%d %H:%M:%S")


def _new_id() -> str:
    return "spk_" + uuid.uuid4().hex[:6]


def _legacy_id(name: str) -> str:
    """Стабильный id для легаси-слепка: ссылки из ТГ не должны меняться."""
    digest = hashlib.sha1(name.strip().lower().encode("utf-8")).hexdigest()
    return "spk_" + digest[:6]


def _name_key(name: Any) -> str:
    """Имя — ключ уникальности: case-insensitive и без краевых пробелов."""
    return str(name or "").strip().casefold()


def upsert_speaker(
    gallery: Sequence[Dict[str, Any]],
    name: str,
    embedding: Any,
    sample_seconds: float = 0.0,
    now: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Добавить или обновить запись по имени (контракт §1, шаг 5).

    Повторное зачисление того же имени не плодит запись, а усредняет вектор с
    весом по числу накопленных семплов: `(old*samples + new) / (samples + 1)`,
    затем снова L2. Возвращает `(новая галерея, запись)`.
    """
    vector = l2_normalize(embedding)
    stamp = now or _now()
    key = _name_key(name)
    result: List[Dict[str, Any]] = [dict(entry) for entry in gallery or []]

    for entry in result:
        if _name_key(entry.get("name")) != key:
            continue
        old = entry_vector(entry)
        samples = int(entry.get("samples") or 0)
        if old is not None and old.size == vector.size and samples > 0:
            merged = (old * float(samples) + vector) / float(samples + 1)
            entry["embedding"] = l2_normalize(merged).tolist()
        else:
            entry["embedding"] = vector.tolist()
        entry["samples"] = samples + 1
        entry["sample_seconds"] = round(
            float(entry.get("sample_seconds") or 0.0) + max(0.0, float(sample_seconds)), 1
        )
        entry["updated_at"] = stamp
        # Написание имени не меняем, если оно отличается только регистром/пробелами:
        # галерея и подписи в протоколах должны остаться стабильными.
        if not entry.get("name"):
            entry["name"] = str(name).strip()
        entry.setdefault("id", _legacy_id(entry["name"]))
        return result, entry

    entry = {
        "id": _new_id(),
        "name": str(name).strip(),
        "embedding": vector.tolist(),
        "samples": 1,
        "sample_seconds": round(max(0.0, float(sample_seconds)), 1),
        "created_at": stamp,
        "updated_at": stamp,
    }
    result.append(entry)
    return result, entry


def delete_speaker(
    gallery: Sequence[Dict[str, Any]],
    entry_id: Optional[str] = None,
    name: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Удалить запись по `id` или по имени.

    Возвращает `(новая галерея, что удалили)`, где второй элемент — исходное
    значение запроса (`id` или имя) либо None, если ничего не нашли.
    """
    want_id = (entry_id or "").strip()
    want_key = _name_key(name) if name else ""
    kept: List[Dict[str, Any]] = []
    deleted: Optional[str] = None
    for entry in gallery or []:
        match = (want_id and entry.get("id") == want_id) or (
            want_key and _name_key(entry.get("name")) == want_key
        )
        if match and deleted is None:
            deleted = want_id or str(name).strip()
            continue
        kept.append(dict(entry))
    return kept, deleted


# ------------------------------------------------------------------------- #
# Сериализация галереи (в т.ч. легаси voiceprint.json)
# ------------------------------------------------------------------------- #

def legacy_gallery(name: Optional[str], embedding: Any, stamp: Optional[str] = None) -> List[Dict[str, Any]]:
    """Легаси-слепок `voiceprint.json` как галерея из одной записи."""
    vector = entry_vector({"embedding": embedding})
    if vector is None:
        return []
    clean_name = str(name or "").strip()
    when = stamp or _now()
    return [
        {
            "id": _legacy_id(clean_name or "legacy"),
            "name": clean_name,
            "embedding": l2_normalize(vector).tolist(),
            "samples": 1,
            "sample_seconds": 0.0,
            "created_at": when,
            "updated_at": when,
        }
    ]


def normalize_gallery(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Починить записи, прочитать которые можно, и выкинуть остальные."""
    result: List[Dict[str, Any]] = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        vector = entry_vector(entry)
        if vector is None:
            continue
        cleaned = dict(entry)
        cleaned["name"] = str(cleaned.get("name") or "").strip()
        if not cleaned.get("id"):
            cleaned["id"] = _legacy_id(cleaned["name"] or "legacy")
        cleaned["embedding"] = l2_normalize(vector).tolist()
        cleaned["samples"] = int(cleaned.get("samples") or 1)
        cleaned["sample_seconds"] = round(float(cleaned.get("sample_seconds") or 0.0), 1)
        cleaned["created_at"] = cleaned.get("created_at") or ""
        cleaned["updated_at"] = cleaned.get("updated_at") or cleaned["created_at"]
        result.append(cleaned)
    return result


def gallery_from_data(raw: Any) -> List[Dict[str, Any]]:
    """Разобрать JSON-содержимое файла галереи.

    Понимает оба формата: текущий `{"version": 2, "speakers": [...]}` и легаси
    одиночный слепок `{"name": ..., "embedding": ..., "enrolled_at": ...}`.
    """
    if isinstance(raw, list):
        return normalize_gallery(raw)
    if not isinstance(raw, dict):
        return []
    if "speakers" in raw or "version" in raw:
        return normalize_gallery(raw.get("speakers") or [])
    if raw.get("embedding"):
        return legacy_gallery(raw.get("name"), raw.get("embedding"), raw.get("enrolled_at"))
    return []


def gallery_to_data(gallery: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {"version": GALLERY_VERSION, "speakers": list(gallery or [])}


def load_gallery_file(path: Path) -> List[Dict[str, Any]]:
    """Прочитать галерею; битый или отсутствующий файл — пустая галерея."""
    try:
        return gallery_from_data(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return []


def save_gallery_file(path: Path, gallery: Sequence[Dict[str, Any]]) -> None:
    """Атомарно записать галерею (tmp + replace)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(gallery_to_data(gallery), ensure_ascii=False), encoding="utf-8"
    )
    tmp.replace(path)


# ------------------------------------------------------------------------- #
# speakers: сборка ответа /get_status
# ------------------------------------------------------------------------- #

def build_speakers(
    segments: Iterable[Dict[str, Any]],
    vectors: Optional[Dict[str, np.ndarray]] = None,
    scores: Optional[Dict[str, float]] = None,
    names: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Собрать массив `speakers` из строк диаризации (контракт §1.3).

    `segments` — последовательность словарей `{"speaker", "start", "end"}`;
    `vectors`/`scores`/`names` — результат `identify_speakers`. `label` — всегда
    исходная метка `SPEAKER_XX`, чтобы клиент нашёл её в тексте.

    Окна речи (`windows`) каждого спикера считаются с учётом речи **остальных**:
    сквозь чужие реплики окна не склеиваются и чужие интервалы вычитаются.
    Иначе в запись, где собеседники перебивают друг друга, в склейку попадает
    второй голос — и зачисленный слепок оказывается «средним по комнате».
    """
    vectors = vectors or {}
    scores = scores or {}
    names = names or {}

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for seg in segments or []:
        speaker = seg.get("speaker")
        if not speaker:
            continue
        grouped.setdefault(str(speaker), []).append(seg)

    spans_by_speaker: Dict[str, List[Tuple[float, float]]] = {}
    for speaker, rows in grouped.items():
        spans: List[Tuple[float, float]] = []
        for row in rows:
            try:
                start, end = float(row["start"]), float(row["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if end > start:
                spans.append((start, end))
        spans_by_speaker[speaker] = spans

    result: List[Dict[str, Any]] = []
    for speaker in sorted(grouped):
        rows = grouped[speaker]
        vector = vectors.get(speaker)
        foreign = [
            span
            for other, spans in spans_by_speaker.items()
            if other != speaker
            for span in spans
        ]
        result.append(
            {
                "label": speaker,
                "name": names.get(speaker),
                "score": (
                    round(float(scores[speaker]), 4)
                    if speaker in scores and scores[speaker] is not None
                    else None
                ),
                "total_sec": round(
                    sum(max(0.0, float(r["end"]) - float(r["start"])) for r in rows), 1
                ),
                "segments": len(rows),
                "embedding": (
                    [round(float(x), 6) for x in np.asarray(vector).reshape(-1)]
                    if vector is not None
                    else None
                ),
                "windows": select_windows(rows, blockers=foreign),
            }
        )
    return result


def load_speakers_file(path: Path) -> List[Dict[str, Any]]:
    """Прочитать `{stem}_speakers.json` (переживает выдачу расшифровки)."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if isinstance(raw, dict):
        raw = raw.get("speakers")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def save_speakers_file(path: Path, speakers: Sequence[Dict[str, Any]]) -> None:
    """Атомарно записать `{stem}_speakers.json`."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps({"speakers": list(speakers or [])}, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def cluster_totals(
    rows: Iterable[Dict[str, Any]],
    embeddings: Dict[str, Any],
) -> List[Tuple[str, float]]:
    """Кластеры диаризации с суммарной длительностью, по убыванию.

    Ключи `embeddings` — сырые метки (`'A'`), а в строках `df` бывает и
    публичная `SPEAKER_XX`; возвращаем ту метку, которая есть среди ключей
    эмбеддингов, чтобы вызывающий сразу мог её использовать.
    """
    totals: Dict[str, float] = {}
    public: Dict[str, str] = {}
    for row in rows or []:
        label = str(row.get("label"))
        try:
            duration = max(0.0, float(row["end"]) - float(row["start"]))
        except (KeyError, TypeError, ValueError):
            duration = 0.0
        totals[label] = totals.get(label, 0.0) + duration
        if row.get("speaker"):
            public.setdefault(label, str(row["speaker"]))

    resolved: Dict[str, float] = {}
    for label, duration in totals.items():
        key = label
        if (embeddings or {}) and label not in embeddings:
            alternative = public.get(label)
            if alternative in embeddings:
                key = alternative
        if (embeddings or {}) and key not in embeddings:
            continue
        resolved[key] = resolved.get(key, 0.0) + duration

    return sorted(resolved.items(), key=lambda kv: kv[1], reverse=True)


def enrollment_verdict(
    rows: Iterable[Dict[str, Any]],
    embeddings: Dict[str, Any],
    max_foreign_ratio: float = ENROLL_MAX_FOREIGN_RATIO,
    same_voice_cosine: float = SAME_VOICE_COSINE,
) -> Tuple[Optional[str], Optional[np.ndarray], float, bool]:
    """Можно ли зачислять по этой склейке, и чей это голос.

    Двухступенчатая защита от подмешивания чужого голоса:

    1. **Склейка двойников.** pyannote часто режет один голос на два кластера
       (тихо/громко, ближе/дальше от микрофона). Кластеры с косинусом ≥
       `same_voice_cosine` — один человек: их речь и эмбеддинги объединяются
       (вектор — взвешенное среднее по длительности), иначе «чужой речью»
       оказалась бы половина собственных реплик.
    2. **Порог чистоты.** Оставшиеся кластеры — честно чужие. Если их доля
       больше `max_foreign_ratio`, зачисление отклоняется: усреднять чужой
       голос с чистым нельзя, такой слепок потом не матчится ни с чем.

    Возвращает `(label, vector, foreign_ratio, ok)`:
      * `label`/`vector` — доминирующий голос (для логирования и ответа);
      * `foreign_ratio` — доля чужой речи после склейки двойников;
      * `ok` — доля чужой речи не больше `max_foreign_ratio`.
    """
    totals = cluster_totals(rows, embeddings)
    if not totals:
        return None, None, 0.0, False

    # `cluster_totals` возвращает список пар, отсортированный по убыванию —
    # первый элемент и есть доминирующий голос.
    dominant_label = totals[0][0]
    dominant_vec = np.asarray(embeddings.get(dominant_label), dtype=np.float32).reshape(-1)
    if not dominant_vec.size:
        return dominant_label, None, 0.0, False
    dominant_vec = l2_normalize(dominant_vec)

    seconds_by_label = dict(totals)

    # какие кластеры — тот же самый голос (двойники доминирующего)
    twin_labels = [dominant_label]
    for label, _ in totals:
        if label == dominant_label:
            continue
        other = np.asarray(embeddings.get(label), dtype=np.float32).reshape(-1)
        if other.size == dominant_vec.size and cosine_similarity(dominant_vec, other) >= same_voice_cosine:
            twin_labels.append(label)

    twin_seconds = sum(seconds_by_label.get(label, 0.0) for label in twin_labels)
    total_seconds = sum(seconds for _, seconds in totals if seconds > 0)
    foreign = max(0.0, total_seconds - twin_seconds)
    ratio = (foreign / total_seconds) if total_seconds > 0 else 0.0

    # итоговый вектор — взвешенное по речи среднее двойников (L2 заново)
    if len(twin_labels) > 1:
        acc = np.zeros_like(dominant_vec, dtype=np.float32)
        weight = 0.0
        for label in twin_labels:
            vec = np.asarray(embeddings.get(label), dtype=np.float32).reshape(-1)
            sec = float(seconds_by_label.get(label, 0.0))
            if vec.size == acc.size:
                acc += l2_normalize(vec) * sec
                weight += sec
        if weight > 0:
            dominant_vec = l2_normalize(acc)

    return dominant_label, dominant_vec, round(ratio, 4), ratio <= max_foreign_ratio


def dominant_embedding(
    rows: Iterable[Dict[str, Any]],
    embeddings: Dict[str, Any],
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """Эмбеддинг доминирующего кластера (максимум суммарной длительности речи).

    Ключи `embeddings` зависят от версии whisperx: сырые метки (`'A'`) или
    публичные (`SPEAKER_00`), поэтому пробуем и то, и другое.
    """
    totals: Dict[str, float] = {}
    public: Dict[str, str] = {}
    for row in rows or []:
        label = str(row.get("label"))
        try:
            duration = max(0.0, float(row["end"]) - float(row["start"]))
        except (KeyError, TypeError, ValueError):
            duration = 0.0
        totals[label] = totals.get(label, 0.0) + duration
        if row.get("speaker"):
            public.setdefault(label, str(row["speaker"]))

    for label in sorted(totals, key=lambda k: totals[k], reverse=True):
        for key in (label, public.get(label)):
            if key is not None and key in (embeddings or {}):
                vec = np.asarray(embeddings[key], dtype=np.float32).reshape(-1)
                if vec.size:
                    return label, l2_normalize(vec)
    if len(embeddings or {}) == 1:
        key = next(iter(embeddings))
        return key, l2_normalize(embeddings[key])
    return None, None
