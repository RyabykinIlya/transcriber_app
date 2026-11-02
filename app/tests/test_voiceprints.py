# -*- coding: utf-8 -*-
"""Тесты чистой логики галереи голосов.

Импортируют только `app/voiceprints.py` — ни torch, ни whisperx, ни Flask не
нужны, поэтому гоняются на обычной машине без GPU.

Запуск: python3 -m unittest discover -s app/tests -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import voiceprints as vp  # noqa: E402


class NormalizeWindowsTest(unittest.TestCase):
    def test_overlap_is_cut(self):
        out = vp.normalize_windows([[0.0, 10.0], [5.0, 15.0]])
        self.assertEqual([(0.0, 10.0), (10.0, 15.0)], out)

    def test_too_short_dropped(self):
        out = vp.normalize_windows([[1.0, 1.3], [2.0, 4.0]])
        self.assertEqual([(2.0, 4.0)], out)

    def test_too_long_trimmed_from_start(self):
        out = vp.normalize_windows([[7.0, 77.0]])
        self.assertEqual([(7.0, 37.0)], out)

    def test_garbage_and_bounds(self):
        out = vp.normalize_windows(
            [["x", 3], [5.0, 4.0], [-2.0, 10.0], [95.0, 120.0], [3, 9]],
            duration=100.0,
        )
        # [-2,10] -> [0,10]; [3,9] -> полностью внутри -> отброшено как наложение;
        # [95,120] обрезано до [95,100] = 5 c (>= 0.4) — остаётся.
        self.assertEqual([(0.0, 10.0), (95.0, 100.0)], out)

    def test_sorted_and_untouched_disjoint(self):
        out = vp.normalize_windows([[40.0, 50.0], [0.0, 8.0]])
        self.assertEqual([(0.0, 8.0), (40.0, 50.0)], out)


class ConcatWindowsTest(unittest.TestCase):
    def setUp(self):
        # 10 секунд «речи»: значение сэмпла = секунда, в которой он лежит.
        self.audio = np.arange(10 * vp.SAMPLE_RATE, dtype=np.float32) / vp.SAMPLE_RATE

    def test_gap_and_offsets(self):
        concat, placements = vp.concat_windows(
            self.audio, [(0.0, 1.0), (5.0, 7.0)], gap_seconds=0.15
        )
        gap = int(round(0.15 * vp.SAMPLE_RATE))
        self.assertEqual(vp.SAMPLE_RATE + gap + 2 * vp.SAMPLE_RATE, concat.size)
        self.assertEqual(2, len(placements))
        # первое окно: [0, 1) без сдвига
        self.assertAlmostEqual(0.0, placements[0]["out_start"], places=6)
        self.assertAlmostEqual(1.0, placements[0]["out_end"], places=6)
        # второе окно сдвинуто на 1 с речи + 0.15 с тишины
        self.assertAlmostEqual(1.15, placements[1]["out_start"], places=6)
        self.assertAlmostEqual(3.15, placements[1]["out_end"], places=6)
        # в склейке реально лежит исходная речь
        second = concat[int(round(1.15 * vp.SAMPLE_RATE)):][:100]
        self.assertGreater(float(second[0]), 4.9)
        # и тишина между окнами
        self.assertTrue(np.allclose(concat[vp.SAMPLE_RATE:vp.SAMPLE_RATE + gap], 0.0))

    def test_empty_selection(self):
        concat, placements = vp.concat_windows(self.audio, [])
        self.assertEqual(0, concat.size)
        self.assertEqual([], placements)


class SelectWindowsTest(unittest.TestCase):
    def test_adjacent_segments_merge(self):
        segs = [
            {"start": 100.0, "end": 106.0},
            {"start": 106.5, "end": 112.0},  # зазор 0.5 < 1.0 -> тот же пробег
            {"start": 130.0, "end": 137.0},  # зазор 18 c -> отдельное окно
        ]
        out = vp.select_windows(segs)
        self.assertEqual([[100.0, 112.0], [130.0, 137.0]], out)

    def test_run_longer_than_30_trimmed(self):
        out = vp.select_windows([{"start": 50.0, "end": 95.0}])
        self.assertEqual([[50.0, 80.0]], out)

    def test_short_runs_ignored_and_limit(self):
        segs = [{"start": float(i * 20), "end": float(i * 20) + 7.0} for i in range(12)]
        segs.append({"start": 400.0, "end": 403.0})  # 3 c — слишком короткое
        out = vp.select_windows(segs)
        self.assertEqual(vp.WINDOWS_LIMIT, len(out))
        self.assertEqual(sorted(out), out)
        self.assertNotIn([400.0, 403.0], out)
        self.assertLessEqual(sum(e - s for s, e in out), vp.WINDOWS_TOTAL_SECONDS)

    def test_total_budget_respected(self):
        segs = [{"start": float(i * 40), "end": float(i * 40) + 29.0} for i in range(10)]
        out = vp.select_windows(segs)
        self.assertLessEqual(sum(e - s for s, e in out), vp.WINDOWS_TOTAL_SECONDS)
        self.assertTrue(all(e - s >= vp.WINDOW_MIN_SECONDS for s, e in out))


class GalleryTest(unittest.TestCase):
    def _vec(self, *values):
        return np.asarray(values, dtype=np.float32)

    def test_l2_normalize(self):
        out = vp.l2_normalize([3.0, 4.0])
        self.assertAlmostEqual(1.0, float(np.linalg.norm(out)), places=5)
        self.assertAlmostEqual(0.6, float(out[0]), places=6)
        self.assertEqual(2, vp.l2_normalize([0.0, 0.0]).size)

    def test_best_match_picks_highest_across_gallery(self):
        gallery = [
            {"id": "spk_1", "name": "A", "embedding": self._vec(1.0, 0.0).tolist()},
            {"id": "spk_2", "name": "B", "embedding": self._vec(0.6, 0.8).tolist()},
            {"id": "spk_3", "name": "C", "embedding": self._vec(0.0, 1.0).tolist()},
        ]
        entry, score = vp.best_match(self._vec(0.9, 0.1), gallery)
        self.assertEqual("spk_1", entry["id"])
        self.assertAlmostEqual(0.9939, score, places=3)

    def test_empty_gallery(self):
        entry, score = vp.best_match(self._vec(1.0, 0.0), [])
        self.assertIsNone(entry)
        self.assertIsNone(score)
        renames, scores, names, vectors = vp.identify_speakers(
            {"A": self._vec(1.0, 0.0)}, {"A": "SPEAKER_00"}, [], 0.6
        )
        self.assertEqual({}, scores)
        self.assertEqual("SPEAKER_00", renames["SPEAKER_00"])
        # вектор всё равно отдаём наружу — по нему делается ретро-разметка
        self.assertEqual(2, vectors["SPEAKER_00"].size)

    def test_threshold_keeps_label(self):
        gallery = [{"id": "spk_1", "name": "Илья", "embedding": self._vec(1.0, 0.0).tolist()}]
        renames, scores, names, _ = vp.identify_speakers(
            {"A": self._vec(0.2, 0.98), "B": self._vec(0.99, 0.05)},
            {"A": "SPEAKER_00", "B": "SPEAKER_01"},
            gallery,
            0.6,
        )
        self.assertNotIn("SPEAKER_00", names)
        self.assertEqual("SPEAKER_00", renames["SPEAKER_00"])
        self.assertLess(scores["SPEAKER_00"], 0.6)
        self.assertEqual("Илья", renames["SPEAKER_01"])
        self.assertGreaterEqual(scores["SPEAKER_01"], 0.6)

    def test_dimension_mismatch_skipped(self):
        gallery = [{"id": "spk_1", "name": "A", "embedding": [1.0, 0.0, 0.0]}]
        self.assertEqual([], vp.score_gallery(self._vec(1.0, 0.0), gallery))

    def test_upsert_new_entry(self):
        gallery, entry = vp.upsert_speaker([], "Дмитрий Морякин", self._vec(1.0, 1.0), 12.5)
        self.assertEqual(1, len(gallery))
        self.assertTrue(entry["id"].startswith("spk_"))
        self.assertEqual("Дмитрий Морякин", entry["name"])
        self.assertEqual(1, entry["samples"])
        self.assertAlmostEqual(12.5, entry["sample_seconds"])
        self.assertAlmostEqual(1.0, float(np.linalg.norm(vp.entry_vector(entry))), places=5)

    def test_re_enroll_merges_with_sample_weight(self):
        old = np.asarray([1.0, 0.0], dtype=np.float32)
        gallery = [
            {
                "id": "spk_111111",
                "name": "Иван Петров",
                "embedding": old.tolist(),
                "samples": 3,
                "sample_seconds": 40.0,
                "created_at": "x",
                "updated_at": "x",
            }
        ]
        new = np.asarray([0.0, 1.0], dtype=np.float32)
        gallery, entry = vp.upsert_speaker(gallery, "  иван петров ", new, 10.0, now="now")
        self.assertEqual(1, len(gallery))  # имя — ключ, второй записи нет
        self.assertEqual("spk_111111", entry["id"])  # id стабилен
        self.assertEqual(4, entry["samples"])
        self.assertAlmostEqual(50.0, entry["sample_seconds"])
        self.assertEqual("now", entry["updated_at"])
        self.assertEqual("x", entry["created_at"])
        # (old*3 + new)/4 = (0.75, 0.25), после L2 — те же пропорции
        vec = vp.entry_vector(entry)
        self.assertAlmostEqual(0.75 / float(np.linalg.norm([0.75, 0.25])), float(vec[0]), places=5)
        self.assertAlmostEqual(1.0, float(np.linalg.norm(vec)), places=5)

    def test_delete_by_id_and_name(self):
        gallery = [
            {"id": "spk_1", "name": "A", "embedding": [1.0, 0.0]},
            {"id": "spk_2", "name": "B", "embedding": [0.0, 1.0]},
        ]
        kept, deleted = vp.delete_speaker(gallery, entry_id="spk_1")
        self.assertEqual("spk_1", deleted)
        self.assertEqual(["spk_2"], [e["id"] for e in kept])

        kept, deleted = vp.delete_speaker(gallery, name=" b ")
        self.assertEqual("b", deleted)
        self.assertEqual(["spk_1"], [e["id"] for e in kept])

        kept, deleted = vp.delete_speaker(gallery, name="нет такого")
        self.assertIsNone(deleted)
        self.assertEqual(2, len(kept))


class FilesTest(unittest.TestCase):
    def test_legacy_voiceprint_is_one_entry_gallery(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voiceprint.json"
            path.write_text(
                json.dumps({"name": "Илья", "embedding": [3.0, 4.0], "enrolled_at": "2026"})
            )
            gallery = vp.load_gallery_file(path)
            self.assertEqual(1, len(gallery))
            self.assertEqual("Илья", gallery[0]["name"])
            self.assertTrue(gallery[0]["id"].startswith("spk_"))
            self.assertAlmostEqual(1.0, float(np.linalg.norm(gallery[0]["embedding"])), places=5)

    def test_gallery_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voiceprints.json"
            gallery, _ = vp.upsert_speaker([], "A", [0.5, 0.5], 3.0)
            vp.save_gallery_file(path, gallery)
            self.assertEqual(2, json.loads(path.read_text())["version"])
            self.assertEqual(1, len(vp.load_gallery_file(path)))

    def test_broken_file_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voiceprints.json"
            path.write_text("{не json")
            self.assertEqual([], vp.load_gallery_file(path))


class BuildSpeakersTest(unittest.TestCase):
    def test_shape_and_windows(self):
        rows = [
            {"speaker": "SPEAKER_00", "start": 1.0, "end": 8.0},
            {"speaker": "SPEAKER_00", "start": 8.5, "end": 14.0},
            {"speaker": "SPEAKER_01", "start": 20.0, "end": 22.0},
        ]
        out = vp.build_speakers(
            rows,
            vectors={"SPEAKER_00": np.asarray([0.6, 0.8], dtype=np.float32)},
            scores={"SPEAKER_00": 0.71},
            names={"SPEAKER_00": "Илья Рябыкин"},
        )
        self.assertEqual(["SPEAKER_00", "SPEAKER_01"], [s["label"] for s in out])
        first, second = out
        self.assertEqual("Илья Рябыкин", first["name"])
        self.assertAlmostEqual(0.71, first["score"])
        self.assertAlmostEqual(12.5, first["total_sec"])
        self.assertEqual(2, first["segments"])
        self.assertEqual([[1.0, 14.0]], first["windows"])
        self.assertEqual(2, len(first["embedding"]))
        self.assertIsNone(second["name"])
        self.assertIsNone(second["score"])
        self.assertIsNone(second["embedding"])
        self.assertEqual([], second["windows"])

    def test_speakers_file_survives_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x_speakers.json"
            speakers = [{"label": "SPEAKER_00", "name": None, "windows": [[1.0, 8.0]]}]
            vp.save_speakers_file(path, speakers)
            self.assertEqual(speakers, vp.load_speakers_file(path))


class DominantEmbeddingTest(unittest.TestCase):
    def test_longest_speaking_cluster_wins(self):
        rows = [
            {"label": "A", "speaker": "SPEAKER_00", "start": 0.0, "end": 2.0},
            {"label": "B", "speaker": "SPEAKER_01", "start": 2.0, "end": 20.0},
        ]
        label, vec = vp.dominant_embedding(rows, {"A": [1.0, 0.0], "B": [0.0, 2.0]})
        self.assertEqual("B", label)
        self.assertAlmostEqual(1.0, float(vec[1]), places=5)


if __name__ == "__main__":
    unittest.main()
