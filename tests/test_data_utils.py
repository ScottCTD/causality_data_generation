import json
import os
import unittest

from question_gen.data_utils import (
    coord_to_str,
    extract_cue_wall_hits,
    has_hit_index_exceeding_threshold,
    make_index,
)


class TestExtractCueWallHits(unittest.TestCase):
    def test_no_events_returns_empty(self) -> None:
        entry = {"events": [], "cushion": {}}
        self.assertEqual(extract_cue_wall_hits(entry), [])

    def test_filters_non_cue_and_non_cushion_events(self) -> None:
        entry = {
            "events": [
                {"type": "linear_cushion", "ball_id": "dummy", "cushion_id": "1", "frame": 5},
                {"type": "other", "ball_id": "cue", "cushion_id": "2", "frame": 10},
            ],
            "cushion": {"1": "red-green-wall", "2": "blue-purple-wall"},
        }
        hits = extract_cue_wall_hits(entry)
        self.assertEqual(hits, [])

    def test_maps_cushion_ids_and_sorts_by_frame(self) -> None:
        entry = {
            "events": [
                {"type": "linear_cushion", "ball_id": "cue", "cushion_id": 2, "frame": 10},
                {"type": "linear_cushion", "ball_id": "cue", "cushion_id": "1", "frame": 5},
            ],
            "cushion": {"1": "red-green-wall", "2": "blue-purple-wall"},
        }
        hits = extract_cue_wall_hits(entry)
        self.assertEqual(len(hits), 2)
        # Sorted by ascending frame
        self.assertLessEqual(hits[0]["frame"], hits[1]["frame"])
        self.assertEqual(hits[0]["name"], "red-green-wall")
        self.assertEqual(hits[1]["name"], "blue-purple-wall")
        # Index field present and integer
        self.assertEqual(hits[0]["index"], 1)
        self.assertEqual(hits[1]["index"], 2)

    def test_unknown_cushion_id_uses_unknown_label(self) -> None:
        entry = {
            "events": [
                {"type": "linear_cushion", "ball_id": "cue", "cushion_id": 99, "frame": 3},
            ],
            "cushion": {},
        }
        hits = extract_cue_wall_hits(entry)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["name"], "unknown")


class TestHasHitIndexExceedingThreshold(unittest.TestCase):
    def test_no_hits_below_threshold_returns_false(self) -> None:
        entry = {
            "events": [
                {"type": "linear_cushion", "ball_id": "cue", "cushion_id": 1, "frame": 5},
            ],
            "cushion": {"1": "red-green-wall"},
        }
        self.assertFalse(has_hit_index_exceeding_threshold(entry, 5))

    def test_hit_above_threshold_returns_true(self) -> None:
        entry = {
            "events": [
                {"type": "linear_cushion", "ball_id": "cue", "cushion_id": 10, "frame": 5},
            ],
            "cushion": {str(i): "red-green-wall" for i in range(1, 20)},
        }
        self.assertTrue(has_hit_index_exceeding_threshold(entry, 5))

    def test_non_numeric_index_is_ignored(self) -> None:
        entry = {
            "events": [
                {"type": "linear_cushion", "ball_id": "cue", "cushion_id": "abc", "frame": 5},
            ],
            "cushion": {"abc": "red-green-wall"},
        }
        self.assertFalse(has_hit_index_exceeding_threshold(entry, 5))


class TestMakeIndex(unittest.TestCase):
    def _load_fixture_path(self, name: str) -> str:
        return os.path.join(os.path.dirname(__file__), name)

    def test_basic_normalization_and_outcomes_from_fixture(self) -> None:
        # Uses the real summary from shot_2048.
        fname = self._load_fixture_path("fixtures_shot_2048_summary.json")
        with open(fname, "r") as f:
            raw = json.load(f)

        id_to_entry, index_by_pos_vel, pos_to_ids, vel_to_ids = make_index([raw])
        self.assertEqual(len(id_to_entry), 1)
        entry = id_to_entry[0]

        # Video id should match shot_id from metadata.
        self.assertEqual(entry["video"], "shot_2048")

        pos = entry["initial_state"]["position"]
        vel = entry["initial_state"]["velocity"]
        # Rounded to 2 decimal places.
        self.assertEqual(pos, [0.38, 1.66, 0.03])
        self.assertEqual(vel, [2.28, -0.95, 0.0])

        outcomes = entry["outcomes"]
        # From fixture: pocketed into orange, 2 wall hits (12 -> blue-purple, 4 -> orange-red).
        self.assertTrue(outcomes["pocketed"])
        self.assertEqual(outcomes["pocket_color"], "orange")
        self.assertEqual(outcomes["num_wall_hits"], 2)
        self.assertEqual(outcomes["wall_hits"], ["blue-purple-wall", "orange-red-wall"])

        key = (tuple(pos), tuple(vel))
        self.assertIn(key, index_by_pos_vel)
        self.assertIn(0, pos_to_ids[tuple(pos)])
        self.assertIn(0, vel_to_ids[tuple(vel)])

    def test_wall_hits_count_falls_back_to_outcomes_when_no_events(self) -> None:
        raw = {
            "metadata": {"shot_id": "test_shot", "total_frames": 10},
            "balls": {
                "cue": {
                    "initial_position": [0.1, 0.2, 0.0],
                    "initial_velocity": [1.0, 0.0, 0.0],
                    "color": "white_cue",
                    "outcomes": {"pocket": None, "wall_hits": 3, "ball_hits": 0},
                }
            },
            "events": [],
            "cushion": {},
        }
        id_to_entry, _, _, _ = make_index([raw])
        entry = id_to_entry[0]
        outcomes = entry["outcomes"]
        # No events but wall_hits present in outcomes -> num_wall_hits should be 3 and wall_hits list empty.
        self.assertEqual(outcomes["num_wall_hits"], 3)
        self.assertEqual(outcomes["wall_hits"], [])


class TestCoordToStr(unittest.TestCase):
    def test_small_values_are_rounded_to_zero(self) -> None:
        coord = (0.0001, -0.004, 0.0)
        self.assertEqual(coord_to_str(coord), "(x=0.00, y=0.00)")

    def test_prefix_is_included(self) -> None:
        coord = (1.234, -2.5, 0.0)
        self.assertEqual(coord_to_str(coord, prefix="d"), "(dx=1.23, dy=-2.50)")


if __name__ == "__main__":
    unittest.main()

