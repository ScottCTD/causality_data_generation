"""Tests for question generation functionality."""

import unittest
import random
from question_gen import (
    convert_to_future_tense,
    filter_outcomes_for_predictive,
    outcome_options_from_outcome,
    sample_multilabel_options,
    OPTION_POOL,
)


class TestConvertToFutureTense(unittest.TestCase):
    """Tests for convert_to_future_tense function."""

    def test_pocketed_options(self):
        """Test conversion of pocketed-related options."""
        assert convert_to_future_tense("The ball was pocketed") == "The ball will be pocketed"
        assert convert_to_future_tense("The ball was pocketed in the blue pocket") == "The ball will be pocketed in the blue pocket"
        assert convert_to_future_tense("The ball was not pocketed") == "The ball will not be pocketed"

    def test_wall_hit_options(self):
        """Test conversion of wall hit options."""
        assert convert_to_future_tense("The ball hit 2 walls") == "The ball will hit 2 walls"
        assert convert_to_future_tense("The ball hit 0 walls") == "The ball will hit 0 walls"
        assert convert_to_future_tense("The ball hit 1 wall") == "The ball will hit 1 wall"
        assert convert_to_future_tense("The ball bounced off a wall") == "The ball will bounce off a wall"

    def test_stayed_on_table(self):
        """Test conversion of 'stayed on table' option."""
        assert convert_to_future_tense("The ball stayed on the table") == "The ball will stay on the table"

    def test_wall_hit_sequence(self):
        """Test conversion of wall hit sequence options."""
        assert convert_to_future_tense("The first wall hit was blue-purple-wall") == "The first wall hit will be blue-purple-wall"
        assert convert_to_future_tense("The second wall hit was red-green-wall") == "The second wall hit will be red-green-wall"
        assert convert_to_future_tense("The third wall hit was grey-orange-wall") == "The third wall hit will be grey-orange-wall"

    def test_already_future_tense(self):
        """Test that already future-tense options are handled correctly."""
        # Should not double-convert
        future = "The ball will be pocketed"
        assert convert_to_future_tense(future) == "The ball will be pocketed"


class TestFilterOutcomesForPredictive(unittest.TestCase):
    """Tests for filter_outcomes_for_predictive function."""

    def test_filters_first_half_wall_hits(self):
        """Test that wall hits in first half are filtered out."""
        outcomes = {
            'hits': [
                {'type': 'wall', 'name': 'wall1', 'frame': 10},  # First half
                {'type': 'wall', 'name': 'wall2', 'frame': 60},  # Second half
                {'type': 'wall', 'name': 'wall3', 'frame': 80},  # Second half
            ],
            'pocket': None,
            'wall_hits': 3
        }
        total_frames = 100
        filtered = filter_outcomes_for_predictive(outcomes, total_frames)
        
        assert len(filtered['hits']) == 2
        assert filtered['wall_hits'] == 2
        assert all(h['frame'] >= 50 for h in filtered['hits'] if h['type'] == 'wall')

    def test_keeps_pocketing_events(self):
        """Test that pocketing events are not filtered."""
        outcomes = {
            'hits': [
                {'type': 'wall', 'name': 'wall1', 'frame': 10},
                {'type': 'pocket', 'name': 'blue', 'frame': 90},
            ],
            'pocket': 'blue',
            'wall_hits': 1
        }
        total_frames = 100
        filtered = filter_outcomes_for_predictive(outcomes, total_frames)
        
        # Pocket event should still be there
        pocket_hits = [h for h in filtered['hits'] if h['type'] == 'pocket']
        assert len(pocket_hits) == 1
        assert filtered['pocket'] == 'blue'

    def test_handles_no_hits(self):
        """Test handling of outcomes with no hits."""
        outcomes = {
            'hits': [],
            'pocket': None,
            'wall_hits': 0
        }
        filtered = filter_outcomes_for_predictive(outcomes, 100)
        assert filtered['wall_hits'] == 0
        assert len(filtered['hits']) == 0

    def test_handles_non_dict_input(self):
        """Test handling of non-dict input."""
        assert filter_outcomes_for_predictive(None, 100) is None
        assert filter_outcomes_for_predictive("string", 100) == "string"

    def test_boundary_case_halfway(self):
        """Test boundary case at exactly halfway point."""
        outcomes = {
            'hits': [
                {'type': 'wall', 'name': 'wall1', 'frame': 49},  # Just before halfway (should be filtered)
                {'type': 'wall', 'name': 'wall2', 'frame': 50},  # Exactly at halfway (should be kept)
                {'type': 'wall', 'name': 'wall3', 'frame': 51},  # Just after halfway (should be kept)
            ],
            'wall_hits': 3
        }
        total_frames = 100
        filtered = filter_outcomes_for_predictive(outcomes, total_frames)
        
        # Frame 49 should be filtered out (< 50), frames 50 and 51 should be kept
        assert filtered['wall_hits'] == 2
        filtered_frames = [h['frame'] for h in filtered['hits'] if h['type'] == 'wall']
        assert 50 in filtered_frames
        assert 51 in filtered_frames
        assert 49 not in filtered_frames


class TestOutcomeOptionsFromOutcome(unittest.TestCase):
    """Tests for outcome_options_from_outcome function."""

    def test_pocketed_outcomes(self):
        """Test options generated for pocketed ball."""
        outcomes = {
            'pocketed': True,
            'pocket_color': 'blue',
            'num_wall_hits': 1,
            'wall_hits': ['wall1']
        }
        opts = outcome_options_from_outcome(outcomes)
        
        assert "The ball was pocketed" in opts
        assert "The ball was pocketed in the blue pocket" in opts
        assert "The ball hit 1 wall" in opts

    def test_not_pocketed_outcomes(self):
        """Test options generated for non-pocketed ball."""
        outcomes = {
            'pocketed': False,
            'num_wall_hits': 2,
            'wall_hits': ['wall1', 'wall2']
        }
        opts = outcome_options_from_outcome(outcomes)
        
        assert "The ball stayed on the table" in opts
        assert "The ball was not pocketed" in opts
        assert "The ball hit 2 walls" in opts

    def test_future_tense_conversion(self):
        """Test that future_tense parameter converts options correctly."""
        outcomes = {
            'pocketed': True,
            'pocket_color': 'red',
            'num_wall_hits': 1,
            'wall_hits': ['wall1']
        }
        opts = outcome_options_from_outcome(outcomes, future_tense=True)
        
        assert "The ball will be pocketed" in opts
        assert "The ball will be pocketed in the red pocket" in opts
        assert "The ball will hit 1 wall" in opts
        assert "The ball was pocketed" not in opts  # Should not have past tense

    def test_wall_hit_sequence(self):
        """Test that wall hit sequence options are generated."""
        outcomes = {
            'pocketed': False,
            'num_wall_hits': 3,
            'wall_hits': ['wall1', 'wall2', 'wall3']
        }
        opts = outcome_options_from_outcome(outcomes)
        
        assert "The first wall hit was wall1" in opts
        assert "The second wall hit was wall2" in opts
        assert "The third wall hit was wall3" in opts

    def test_zero_wall_hits(self):
        """Test options for zero wall hits."""
        outcomes = {
            'pocketed': False,
            'num_wall_hits': 0,
            'wall_hits': []
        }
        opts = outcome_options_from_outcome(outcomes)
        
        assert "The ball hit 0 walls" in opts
        assert "The ball bounced off a wall" not in opts


class TestSampleMultilabelOptions(unittest.TestCase):
    """Tests for sample_multilabel_options function."""

    def test_basic_sampling(self):
        """Test basic option sampling."""
        true_opts = ["The ball was pocketed", "The ball hit 2 walls"]
        pool = OPTION_POOL
        options, ground_truth = sample_multilabel_options(true_opts, pool, total=4, num_correct=2)
        
        assert len(options) == 4
        assert len(ground_truth) == 2
        assert all(i < len(options) for i in ground_truth)
        # Check that correct options are in the list
        assert all(options[i] in true_opts for i in ground_truth)

    def test_future_tense_sampling(self):
        """Test sampling with future tense conversion."""
        true_opts = ["The ball will be pocketed", "The ball will hit 2 walls"]
        pool = OPTION_POOL
        options, ground_truth = sample_multilabel_options(
            true_opts, pool, total=4, num_correct=2, future_tense=True
        )
        
        assert len(options) == 4
        # All options should be in future tense
        assert all("will" in opt or "will not" in opt for opt in options)
        # Correct options should be preserved
        assert all(options[i] in true_opts for i in ground_truth)

    def test_consistency_checking(self):
        """Test that inconsistent distractors are filtered out."""
        true_opts = ["The ball was pocketed"]
        pool = OPTION_POOL
        options, ground_truth = sample_multilabel_options(true_opts, pool, total=4, num_correct=1)
        
        # Should not have "The ball was not pocketed" or "The ball stayed on the table"
        assert "The ball was not pocketed" not in options
        assert "The ball stayed on the table" not in options

    def test_future_tense_consistency(self):
        """Test consistency checking with future tense options."""
        true_opts = ["The ball will be pocketed"]
        pool = OPTION_POOL
        options, ground_truth = sample_multilabel_options(
            true_opts, pool, total=4, num_correct=1, future_tense=True
        )
        
        # Converted distractors should not contradict correct options
        assert "The ball will not be pocketed" not in options
        assert "The ball will stay on the table" not in options

    def test_empty_true_opts(self):
        """Test handling of empty true options."""
        options, ground_truth = sample_multilabel_options([], OPTION_POOL, total=4, num_correct=1)
        assert len(options) == 4
        assert len(ground_truth) == 1
        assert "The ball stayed on the table" in options

    def test_future_tense_empty_true_opts(self):
        """Test handling of empty true options with future tense."""
        options, ground_truth = sample_multilabel_options(
            [], OPTION_POOL, total=4, num_correct=1, future_tense=True
        )
        assert len(options) == 4
        assert len(ground_truth) == 1
        assert "The ball will stay on the table" in options

    def test_num_correct_limit(self):
        """Test that num_correct is limited by available options."""
        true_opts = ["The ball was pocketed"]
        options, ground_truth = sample_multilabel_options(
            true_opts, OPTION_POOL, total=4, num_correct=5  # More than available
        )
        assert len(ground_truth) == 1  # Should be limited to 1


if __name__ == "__main__":
    # Set random seed for reproducible tests
    random.seed(42)
    unittest.main(verbosity=2)

