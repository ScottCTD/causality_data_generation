import unittest

from question_gen.options import (
    DISTRACTOR_POOL_FACTS,
    OptionFact,
    OptionRenderer,
    WALL_NAMES,
    facts_from_outcome,
    random_sample,
    sample_multilabel_from_facts,
)
from question_gen.tense import Tense


class TestFactsFromOutcome(unittest.TestCase):
    def test_pocketed_and_color_fact(self) -> None:
        outcomes = {
            "num_wall_hits": 0,
            "wall_hits": [],
            "pocketed": True,
            "which_pocket": "orange",
            "pocket_color": "orange",
        }
        facts = facts_from_outcome(outcomes)
        self.assertIn(OptionFact("pocketed"), facts)
        self.assertIn(OptionFact("pocketed_in", ("orange",)), facts)

    def test_zero_wall_hits_fact(self) -> None:
        outcomes = {
            "num_wall_hits": 0,
            "wall_hits": [],
            "pocketed": False,
            "which_pocket": None,
            "pocket_color": None,
        }
        facts = facts_from_outcome(outcomes)
        self.assertIn(OptionFact("hits_0_walls"), facts)

    def test_single_wall_hit_fact(self) -> None:
        outcomes = {
            "num_wall_hits": 1,
            "wall_hits": ["blue-purple-wall"],
            "pocketed": False,
            "which_pocket": None,
            "pocket_color": None,
        }
        facts = facts_from_outcome(outcomes)
        self.assertIn(OptionFact("hits_1_wall"), facts)
        self.assertIn(OptionFact("first_wall_hit", ("blue-purple-wall",)), facts)

    def test_multiple_same_wall_hits_fact(self) -> None:
        outcomes = {
            "num_wall_hits": 3,
            "wall_hits": ["grey-orange-wall", "grey-orange-wall", "grey-orange-wall"],
            "pocketed": False,
            "which_pocket": None,
            "pocket_color": None,
        }
        facts = facts_from_outcome(outcomes)
        self.assertIn(OptionFact("hits_same_wall_n_times", (3,)), facts)
        self.assertNotIn(OptionFact("hits_n_diff_walls", (3,)), facts)

    def test_multiple_different_wall_hits_fact(self) -> None:
        outcomes = {
            "num_wall_hits": 3,
            "wall_hits": ["red-green-wall", "blue-purple-wall", "red-green-wall"],
            "pocketed": False,
            "which_pocket": None,
            "pocket_color": None,
        }
        facts = facts_from_outcome(outcomes)
        # 2 distinct walls in the sequence.
        self.assertIn(OptionFact("hits_n_diff_walls", (2,)), facts)

    def test_wall_sequence_limits_to_three_hits(self) -> None:
        walls = ["red-green-wall", "blue-purple-wall", "grey-orange-wall", "purple-grey-wall"]
        outcomes = {
            "num_wall_hits": 4,
            "wall_hits": walls,
            "pocketed": False,
            "which_pocket": None,
            "pocket_color": None,
        }
        facts = facts_from_outcome(outcomes)
        self.assertIn(OptionFact("first_wall_hit", (walls[0],)), facts)
        self.assertIn(OptionFact("second_wall_hit", (walls[1],)), facts)
        self.assertIn(OptionFact("third_wall_hit", (walls[2],)), facts)
        # No fourth-hit fact.
        self.assertNotIn(OptionFact("fourth_wall_hit", (walls[3],)), facts)

    def test_facts_are_deduplicated(self) -> None:
        outcomes = {
            "num_wall_hits": 3,
            "wall_hits": ["grey-orange-wall", "grey-orange-wall", "grey-orange-wall"],
            "pocketed": True,
            "which_pocket": "orange",
            "pocket_color": "orange",
        }
        facts = facts_from_outcome(outcomes)
        # Ensure no duplicates even if logically same fact could appear multiple times.
        self.assertEqual(len(facts), len(set(facts)))


class TestOptionRenderer(unittest.TestCase):
    def test_render_pocketed_in_all_tenses(self) -> None:
        renderer = OptionRenderer()
        fact = OptionFact("pocketed")
        self.assertEqual(renderer.render(fact, Tense.BASE), "The ball was pocketed")
        self.assertEqual(renderer.render(fact, Tense.FUTURE), "The ball will be pocketed")
        self.assertEqual(renderer.render(fact, Tense.CONDITIONAL), "The ball would be pocketed")

    def test_render_hits_n_diff_walls(self) -> None:
        renderer = OptionRenderer()
        fact = OptionFact("hits_n_diff_walls", (3,))
        self.assertEqual(renderer.render(fact, Tense.BASE), "The ball hits 3 different walls")
        self.assertEqual(renderer.render(fact, Tense.FUTURE), "The ball will hit 3 different walls")

    def test_render_wall_hit_order(self) -> None:
        renderer = OptionRenderer()
        fact1 = OptionFact("first_wall_hit", ("blue-purple-wall",))
        fact2 = OptionFact("second_wall_hit", ("red-green-wall",))
        fact3 = OptionFact("third_wall_hit", ("grey-orange-wall",))
        self.assertEqual(
            renderer.render(fact1, Tense.BASE),
            "The first wall hit was blue-purple-wall",
        )
        self.assertEqual(
            renderer.render(fact2, Tense.FUTURE),
            "The second wall hit will be red-green-wall",
        )
        self.assertEqual(
            renderer.render(fact3, Tense.CONDITIONAL),
            "The third wall hit would be grey-orange-wall",
        )


class TestSampleMultilabelFromFacts(unittest.TestCase):
    def test_basic_sampling_respects_counts(self) -> None:
        outcomes = {
            "num_wall_hits": 2,
            "wall_hits": ["blue-purple-wall", "orange-red-wall"],
            "pocketed": True,
            "which_pocket": "orange",
            "pocket_color": "orange",
        }
        true_facts = facts_from_outcome(outcomes)
        renderer = OptionRenderer()
        options, gt = sample_multilabel_from_facts(
            true_facts,
            DISTRACTOR_POOL_FACTS,
            total=4,
            num_correct=1,
            tense=Tense.BASE,
            renderer=renderer,
        )
        self.assertEqual(len(options), 4)
        self.assertEqual(len(gt), 1)
        self.assertIn(gt[0], range(4))

    def test_not_pocketed_fact_is_filtered_out(self) -> None:
        # Construct a synthetic true fact list containing only not_pocketed.
        true_facts = [OptionFact("not_pocketed")]
        renderer = OptionRenderer()
        options, gt = sample_multilabel_from_facts(
            true_facts,
            DISTRACTOR_POOL_FACTS,
            total=4,
            num_correct=2,
            tense=Tense.BASE,
            renderer=renderer,
        )
        # No permissible correct facts remain -> no question should be generated.
        self.assertEqual(options, [])
        self.assertEqual(gt, [])

    def test_all_true_facts_must_be_marked_correct_by_string_match(self) -> None:
        # Scenario similar to shot_2048 outcomes.
        outcomes = {
            "num_wall_hits": 2,
            "wall_hits": ["blue-purple-wall", "orange-red-wall"],
            "pocketed": True,
            "which_pocket": "orange",
            "pocket_color": "orange",
        }
        true_facts = facts_from_outcome(outcomes)
        renderer = OptionRenderer()
        true_strings = {renderer.render(f, Tense.BASE) for f in true_facts}

        options, gt = sample_multilabel_from_facts(
            true_facts,
            DISTRACTOR_POOL_FACTS,
            total=4,
            num_correct=1,
            tense=Tense.BASE,
            renderer=renderer,
        )

        # Any option whose surface form corresponds to a true fact must be marked correct.
        gt_set = set(gt)
        for idx, opt in enumerate(options):
            is_true_statement = opt in true_strings
            self.assertEqual(
                is_true_statement,
                idx in gt_set,
                msg=f"Option '{opt}' should {'be' if is_true_statement else 'not be'} marked correct",
            )

    def test_random_sample_handles_k_greater_than_len(self) -> None:
        facts = [OptionFact("pocketed"), OptionFact("hits_0_walls")]
        # k > len(facts) should just return all facts without error.
        sampled = random_sample(facts, 5)
        self.assertEqual(set(sampled), set(facts))

    def test_distractors_never_use_not_pocketed_kind(self) -> None:
        outcomes = {
            "num_wall_hits": 0,
            "wall_hits": [],
            "pocketed": True,
            "which_pocket": "gray",
            "pocket_color": "gray",
        }
        true_facts = facts_from_outcome(outcomes)
        renderer = OptionRenderer()
        options, gt = sample_multilabel_from_facts(
            true_facts,
            DISTRACTOR_POOL_FACTS,
            total=6,
            num_correct=1,
            tense=Tense.BASE,
            renderer=renderer,
        )
        # Ensure the surface form for "not_pocketed" never appears.
        self.assertNotIn("The ball was not pocketed", options)

    def test_wall_name_distractors_use_known_wall_names(self) -> None:
        # Make sure any option mentioning a wall name uses one of WALL_NAMES.
        outcomes = {
            "num_wall_hits": 1,
            "wall_hits": ["blue-purple-wall"],
            "pocketed": False,
            "which_pocket": None,
            "pocket_color": None,
        }
        true_facts = facts_from_outcome(outcomes)
        renderer = OptionRenderer()
        options, _ = sample_multilabel_from_facts(
            true_facts,
            DISTRACTOR_POOL_FACTS,
            total=6,
            num_correct=1,
            tense=Tense.BASE,
            renderer=renderer,
        )
        for opt in options:
            if "wall hit" in opt:
                # Extract wall name token (last word in the string).
                wall_name = opt.split()[-1]
                self.assertIn(wall_name, WALL_NAMES)


if __name__ == "__main__":
    unittest.main()

