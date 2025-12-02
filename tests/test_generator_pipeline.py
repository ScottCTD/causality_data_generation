import json
import os
import sys
import types
import unittest

# Provide a lightweight stub for tqdm so that the generator can be imported
# without requiring the external tqdm package.
if "tqdm" not in sys.modules:
    tqdm_mod = types.ModuleType("tqdm")

    def _tqdm(iterable, **kwargs):
        return iterable

    tqdm_mod.tqdm = _tqdm
    sys.modules["tqdm"] = tqdm_mod

from question_gen.data_utils import make_index
from question_gen.generator import generate_sft_mcq_multilabel
from question_gen.options import OptionRenderer, facts_from_outcome
from question_gen.tense import Tense


class TestGeneratorDescriptivePredictive(unittest.TestCase):
    def _build_simple_sim_entry(self):
        return {
            "metadata": {"shot_id": "sim_test_1", "total_frames": 100},
            "balls": {
                "cue": {
                    "initial_position": [0.1, 0.2, 0.0],
                    "initial_velocity": [1.0, 0.0, 0.0],
                    "color": "white_cue",
                    "outcomes": {"pocket": None, "wall_hits": 0, "ball_hits": 0},
                }
            },
            "events": [],
            "cushion": {},
        }

    def _build_sim_entry_with_hits(self):
        # Two wall hits, one before and two after halfway point.
        return {
            "metadata": {"shot_id": "sim_test_2", "total_frames": 100},
            "balls": {
                "cue": {
                    "initial_position": [0.3, 0.4, 0.0],
                    "initial_velocity": [0.5, -0.5, 0.0],
                    "color": "white_cue",
                    "outcomes": {"pocket": None, "wall_hits": 3, "ball_hits": 0},
                }
            },
            "events": [
                {"type": "linear_cushion", "ball_id": "cue", "cushion_id": "1", "frame": 10},
                {"type": "linear_cushion", "ball_id": "cue", "cushion_id": "2", "frame": 60},
                {"type": "linear_cushion", "ball_id": "cue", "cushion_id": "3", "frame": 80},
            ],
            "cushion": {
                "1": "red-green-wall",
                "2": "blue-purple-wall",
                "3": "orange-red-wall",
            },
        }

    def test_descriptive_questions_generated_for_each_shot(self) -> None:
        sim_data = [self._build_simple_sim_entry(), self._build_sim_entry_with_hits()]
        dataset = generate_sft_mcq_multilabel(
            sim_data,
            num_options=4,
            num_correct=1,
            num_descriptive_per_shot=2,
            num_predictive_per_shot=0,
            max_velocity_cfs_per_shot=0,
            max_position_cfs_per_shot=0,
            predictive_filter_fraction=0.5,
        )
        # Expect up to 2 descriptive questions per shot when enough facts exist.
        desc = [ex for ex in dataset if ex["metadata"]["question_type"] == "descriptive"]
        self.assertGreaterEqual(len(desc), 2)
        self.assertLessEqual(len(desc), 4)
        for ex in desc:
            self.assertEqual(len(ex["options"]), 4)
            self.assertGreaterEqual(len(ex["ground_truth"]), 1)
            self.assertLessEqual(len(ex["ground_truth"]), 1)

    def test_predictive_questions_use_second_half_hits(self) -> None:
        sim_data = [self._build_sim_entry_with_hits()]
        # Build the normalized index so we can derive expected filtered outcomes.
        id_to_entry, _, _, _ = make_index(sim_data)
        entry = id_to_entry[0]
        total_frames = entry["total_frames"]
        hits_detail = entry["hits_detail"]
        threshold = 0.5 * total_frames
        filtered_hits = [
            h
            for h in hits_detail
            if isinstance(h, dict)
            and h.get("type") == "wall"
            and h.get("frame", 0) >= threshold
        ]
        filtered_wall_names = [h.get("name") for h in filtered_hits]

        dataset = generate_sft_mcq_multilabel(
            sim_data,
            num_options=4,
            num_correct=1,
            num_descriptive_per_shot=0,
            num_predictive_per_shot=1,
            max_velocity_cfs_per_shot=0,
            max_position_cfs_per_shot=0,
            predictive_filter_fraction=0.5,
        )
        pred = [ex for ex in dataset if ex["metadata"]["question_type"] == "predictive"]
        self.assertEqual(len(pred), 1)
        ex = pred[0]
        self.assertEqual(len(ex["options"]), 4)
        self.assertEqual(len(ex["ground_truth"]), 1)

        # Reconstruct the filtered outcomes and true facts to verify that
        # the correct option corresponds to a fact about the second-half hits.
        base_outcomes = entry["outcomes"]
        filtered_outcomes = dict(base_outcomes)
        filtered_outcomes["num_wall_hits"] = len(filtered_wall_names)
        filtered_outcomes["wall_hits"] = filtered_wall_names
        true_facts = facts_from_outcome(filtered_outcomes)
        renderer = OptionRenderer()
        true_strings = {renderer.render(f, Tense.FUTURE) for f in true_facts}

        options = ex["options"]
        gt_idx = ex["ground_truth"][0]
        self.assertIn(options[gt_idx], true_strings)
        # Ensure no incorrect option is also a true statement.
        for idx, opt in enumerate(options):
            if idx == gt_idx:
                continue
            self.assertNotIn(opt, true_strings)


class TestGeneratorCounterfactuals(unittest.TestCase):
    def _build_cf_sim_data(self):
        # Three entries: two share position, two share velocity, with different outcomes.
        base = {
            "balls": {
                "cue": {
                    "color": "white_cue",
                    "outcomes": {"pocket": None, "wall_hits": 0, "ball_hits": 0},
                }
            },
            "events": [],
            "cushion": {},
        }
        a = dict(base)
        a.update(
            {
                "metadata": {"shot_id": "cf_a", "total_frames": 10},
                "balls": {
                    "cue": {
                        "initial_position": [0.1, 0.2, 0.0],
                        "initial_velocity": [1.0, 0.0, 0.0],
                        "color": "white_cue",
                        "outcomes": {"pocket": "orange", "wall_hits": 1, "ball_hits": 0},
                    }
                },
            }
        )
        b = dict(base)
        b.update(
            {
                "metadata": {"shot_id": "cf_b", "total_frames": 10},
                "balls": {
                    "cue": {
                        "initial_position": [0.1, 0.2, 0.0],  # same pos as a
                        "initial_velocity": [0.0, 1.0, 0.0],
                        "color": "white_cue",
                        "outcomes": {"pocket": None, "wall_hits": 2, "ball_hits": 0},
                    }
                },
            }
        )
        c = dict(base)
        c.update(
            {
                "metadata": {"shot_id": "cf_c", "total_frames": 10},
                "balls": {
                    "cue": {
                        "initial_position": [0.5, 0.5, 0.0],
                        "initial_velocity": [1.0, 0.0, 0.0],  # same vel as a
                        "color": "white_cue",
                        "outcomes": {"pocket": "gray", "wall_hits": 0, "ball_hits": 0},
                    }
                },
            }
        )
        return [a, b, c]

    def test_counterfactual_questions_generated(self) -> None:
        sim_data = self._build_cf_sim_data()
        dataset = generate_sft_mcq_multilabel(
            sim_data,
            num_options=4,
            num_correct=1,
            num_descriptive_per_shot=0,
            num_predictive_per_shot=0,
            max_velocity_cfs_per_shot=2,
            max_position_cfs_per_shot=2,
            predictive_filter_fraction=0.5,
        )
        vel_cf = [ex for ex in dataset if ex["metadata"]["question_type"] == "counterfactual_velocity"]
        pos_cf = [ex for ex in dataset if ex["metadata"]["question_type"] == "counterfactual_position"]

        # With the constructed data there should be at least one velocity and one position CF question.
        self.assertGreaterEqual(len(vel_cf), 1)
        self.assertGreaterEqual(len(pos_cf), 1)

        for ex in vel_cf:
            self.assertIn("If the initial velocity were changed from", ex["question"])
            self.assertEqual(len(ex["options"]), 4)
            self.assertEqual(len(ex["ground_truth"]), 1)
            meta = ex["metadata"]
            self.assertIn("counterfactual_sim_id", meta)
            self.assertIn("counterfactual_initial_state", meta)

        for ex in pos_cf:
            self.assertIn("If the initial ball position were changed from", ex["question"])
            self.assertEqual(len(ex["options"]), 4)
            self.assertEqual(len(ex["ground_truth"]), 1)
            meta = ex["metadata"]
            self.assertIn("counterfactual_sim_id", meta)
            self.assertIn("counterfactual_initial_state", meta)


class TestGeneratorWithRealFixture(unittest.TestCase):
    def test_real_fixture_produces_descriptive_and_predictive(self) -> None:
        # Use the real summary fixture for shot_2048 to drive an end-to-end test.
        fixture_path = os.path.join(os.path.dirname(__file__), "fixtures_shot_2048_summary.json")
        with open(fixture_path, "r") as f:
            raw = json.load(f)

        sim_data = [raw]
        dataset = generate_sft_mcq_multilabel(
            sim_data,
            num_options=4,
            num_correct=1,
            num_descriptive_per_shot=3,
            num_predictive_per_shot=1,
            max_velocity_cfs_per_shot=0,
            max_position_cfs_per_shot=0,
            predictive_filter_fraction=0.5,
        )
        types_seen = {ex["metadata"]["question_type"] for ex in dataset}
        self.assertIn("descriptive", types_seen)
        self.assertIn("predictive", types_seen)

        for ex in dataset:
            self.assertEqual(len(ex["options"]), 4)
            self.assertGreaterEqual(len(ex["ground_truth"]), 1)
            self.assertLessEqual(len(ex["ground_truth"]), 1)


if __name__ == "__main__":
    unittest.main()

