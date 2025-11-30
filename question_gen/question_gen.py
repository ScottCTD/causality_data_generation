# The following script takes the shot metadata JSON files generated from the simulations
# and produces a dataset of multiple-choice questions (MCQs) suitable for training/evaluating
#
# Usage example:
#   python question_gen.py --dataset ds1 --num-options 6 --num-correct 2
# This will generate MCQs with 6 options per question, 2 of which are correct.
# The output will be written to outputs/ds1/raw_qa.jsonl
#
# To override the output path:
#   python question_gen.py --dataset ds1 --output raw_qa.jsonl
# This will write to raw_qa.jsonl in the current working directory.

import argparse
import glob
import json
import os
import random
import sys
from typing import Dict, List

from tqdm import tqdm

# Support both:
#   - python -m question_gen.question_gen
#   - python question_gen/question_gen.py
if __name__ == "__main__" and __package__ is None:
    # When executed as a script, ensure sibling modules are importable.
    sys.path.insert(0, os.path.dirname(__file__))
    from data_utils import (  # type: ignore
        has_hit_index_exceeding_threshold,
    )
    from generator import generate_sft_mcq_multilabel  # type: ignore
else:
    from .data_utils import has_hit_index_exceeding_threshold
    from .generator import generate_sft_mcq_multilabel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MCQ dataset from simulation metadata."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="Dataset name (directory under outputs/).",
    )
    parser.add_argument(
        "--num-options",
        "-n",
        type=int,
        default=4,
        help="Total number of options per question.",
    )
    parser.add_argument(
        "--num-correct",
        "-c",
        type=int,
        default=1,
        help="Number of correct options per question.",
    )
    parser.add_argument(
        "--num-descriptive-per-shot",
        "-D",
        type=int,
        default=1,
        help="Number of descriptive questions to generate per shot (0 to disable).",
    )
    parser.add_argument(
        "--num-predictive-per-shot",
        "-p",
        type=int,
        default=1,
        help="Number of predictive questions to generate per shot (0 to disable).",
    )
    parser.add_argument(
        "--max-velocity-cfs-per-shot",
        "-v",
        type=int,
        default=3,
        help="Maximum number of counterfactual velocity questions per shot (0 to disable).",
    )
    parser.add_argument(
        "--max-position-cfs-per-shot",
        "-P",
        type=int,
        default=3,
        help="Maximum number of counterfactual position questions per shot (0 to disable).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: outputs/{dataset}/raw_qa.jsonl). Can be relative or absolute.",
    )
    parser.add_argument(
        "--exclude-invalid-hits",
        "-e",
        action="store_true",
        help="Exclude videos with any hit index > 18.",
    )
    parser.add_argument(
        "--predictive-filter-fraction",
        "-f",
        type=float,
        default=0.5,
        help=(
            "Fraction of video to filter out for predictive questions "
            "(e.g., 0.5 means filter out wall hits from the first half)."
        ),
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    dataset_dir = os.path.join("outputs", args.dataset)
    shots_pattern = os.path.join(dataset_dir, "shots", "shot_*", "*.json")

    sim_data: List[Dict] = []
    excluded_count = 0

    all_files = glob.glob(shots_pattern)
    for fname in tqdm(all_files, desc="Loading simulation data"):
        with open(fname, "r") as f:
            entry = json.load(f)
            if args.exclude_invalid_hits and has_hit_index_exceeding_threshold(
                entry, 18
            ):
                excluded_count += 1
                continue
            sim_data.append(entry)

    if args.exclude_invalid_hits:
        print(f"Excluded {excluded_count} shots with hit index > 18")
    print(f"Processing {len(sim_data)} shots")

    dataset = generate_sft_mcq_multilabel(
        sim_data,
        num_options=args.num_options,
        num_correct=args.num_correct,
        num_descriptive_per_shot=args.num_descriptive_per_shot,
        num_predictive_per_shot=args.num_predictive_per_shot,
        max_velocity_cfs_per_shot=args.max_velocity_cfs_per_shot,
        max_position_cfs_per_shot=args.max_position_cfs_per_shot,
        predictive_filter_fraction=args.predictive_filter_fraction,
    )

    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(dataset_dir, "raw_qa.jsonl")

    with open(output_path, "w") as f:
        for ex in dataset:
            f.write(json.dumps(ex) + "\n")

    print("Wrote", len(dataset), "examples to", output_path)


if __name__ == "__main__":
    main()
