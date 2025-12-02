#!/usr/bin/env python3
"""
Generate statistics from raw_qa.jsonl file.

This script analyzes the QA dataset and produces statistics including:
- Count of questions by category
- Distribution of options and ground truth for each category
- Other relevant statistics for debugging
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Any


def load_qa_data(file_path: str) -> List[Dict[str, Any]]:
    """Load QA data from JSONL file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue
    return data


def generate_stats(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive statistics from QA data."""
    stats: Dict[str, Any] = {
        "total_questions": len(data),
        "categories": {},
        "overall": {},
    }

    # Overall statistics
    num_options_dist = Counter()
    num_correct_dist = Counter()
    videos = set()
    questions_per_video = Counter()

    # Per-category statistics
    category_data: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "options_distribution": Counter(),
            "ground_truth_distribution": Counter(),
            "ground_truth_options_distribution": Counter(),  # Which option strings are correct
            "option_index_distribution": defaultdict(Counter),  # For each option index, count which options appear
            "num_options_dist": Counter(),
            "num_correct_dist": Counter(),
            "videos": set(),
        }
    )

    for entry in data:
        video = entry.get("video", "unknown")
        options = entry.get("options", [])
        ground_truth = entry.get("ground_truth", [])
        metadata = entry.get("metadata", {})
        question_type = metadata.get("question_type", "unknown")

        # Overall stats
        num_options_dist[len(options)] += 1
        num_correct_dist[len(ground_truth)] += 1
        videos.add(video)
        questions_per_video[video] += 1

        # Category-specific stats
        cat_stats = category_data[question_type]
        cat_stats["count"] += 1
        cat_stats["num_options_dist"][len(options)] += 1
        cat_stats["num_correct_dist"][len(ground_truth)] += 1
        cat_stats["videos"].add(video)

        # Track option strings and their positions
        for idx, option in enumerate(options):
            cat_stats["options_distribution"][option] += 1
            cat_stats["option_index_distribution"][idx][option] += 1

        # Track ground truth distribution (which option indices are correct)
        # Also track which option strings are correct
        for gt_idx in ground_truth:
            if isinstance(gt_idx, int) and 0 <= gt_idx < len(options):
                cat_stats["ground_truth_distribution"][gt_idx] += 1
                # Track the actual option string that is correct
                correct_option = options[gt_idx]
                cat_stats["ground_truth_options_distribution"][correct_option] += 1

    # Convert category data to serializable format
    for category, cat_stats in category_data.items():
        stats["categories"][category] = {
            "count": cat_stats["count"],
            "percentage": (cat_stats["count"] / len(data) * 100) if data else 0,
            "num_options_distribution": dict(cat_stats["num_options_dist"]),
            "num_correct_distribution": dict(cat_stats["num_correct_dist"]),
            "unique_videos": len(cat_stats["videos"]),
            "options_distribution": {
                "total_unique_options": len(cat_stats["options_distribution"]),
                "top_20_most_common": dict(
                    cat_stats["options_distribution"].most_common(20)
                ),
            },
            "ground_truth_distribution": {
                "distribution_by_index": dict(cat_stats["ground_truth_distribution"]),
                "total_correct_answers": sum(cat_stats["ground_truth_distribution"].values()),
                "percentage_by_index": {
                    str(idx): (count / sum(cat_stats["ground_truth_distribution"].values()) * 100)
                    if sum(cat_stats["ground_truth_distribution"].values()) > 0
                    else 0
                    for idx, count in cat_stats["ground_truth_distribution"].items()
                },
            },
            "ground_truth_options_distribution": {
                "total_unique_correct_options": len(cat_stats["ground_truth_options_distribution"]),
                "top_20_most_common_correct": dict(
                    cat_stats["ground_truth_options_distribution"].most_common(20)
                ),
                "distribution": dict(cat_stats["ground_truth_options_distribution"]),
                "percentage_by_option": {
                    option: (count / sum(cat_stats["ground_truth_options_distribution"].values()) * 100)
                    if sum(cat_stats["ground_truth_options_distribution"].values()) > 0
                    else 0
                    for option, count in cat_stats["ground_truth_options_distribution"].items()
                },
            },
            "option_position_analysis": {
                # For each position (0, 1, 2, ...), show which options appear there most often
                str(pos): dict(opt_counter.most_common(10))
                for pos, opt_counter in cat_stats["option_index_distribution"].items()
            },
        }

        # Check for biases/imbalances
        gt_dist = cat_stats["ground_truth_distribution"]
        if gt_dist:
            total_gt = sum(gt_dist.values())
            max_gt_count = max(gt_dist.values())
            min_gt_count = min(gt_dist.values())
            max_gt_idx = max(gt_dist.items(), key=lambda x: x[1])[0]
            min_gt_idx = min(gt_dist.items(), key=lambda x: x[1])[0]

            stats["categories"][category]["bias_analysis"] = {
                "most_common_correct_index": max_gt_idx,
                "most_common_correct_percentage": (max_gt_count / total_gt * 100) if total_gt > 0 else 0,
                "least_common_correct_index": min_gt_idx,
                "least_common_correct_percentage": (min_gt_count / total_gt * 100) if total_gt > 0 else 0,
                "imbalance_ratio": (max_gt_count / min_gt_count) if min_gt_count > 0 else float("inf"),
                "is_balanced": (
                    "A dataset is considered balanced if no single option index "
                    "accounts for more than 40% of correct answers"
                ),
                "is_balanced_result": (max_gt_count / total_gt <= 0.4) if total_gt > 0 else True,
            }

    # Overall statistics
    stats["overall"] = {
        "unique_videos": len(videos),
        "num_options_distribution": dict(num_options_dist),
        "num_correct_distribution": dict(num_correct_dist),
        "questions_per_video": {
            "min": min(questions_per_video.values()) if questions_per_video else 0,
            "max": max(questions_per_video.values()) if questions_per_video else 0,
            "mean": (
                sum(questions_per_video.values()) / len(questions_per_video)
                if questions_per_video
                else 0
            ),
            "distribution": dict(Counter(questions_per_video.values())),
        },
    }

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate statistics from raw_qa.jsonl file."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to raw_qa.jsonl file (or raw_qa.json).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path for raw_qa_stats.json (default: same directory as input with raw_qa_stats.json name).",
    )
    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_dir = os.path.dirname(os.path.abspath(args.input_file))
        output_path = os.path.join(input_dir, "raw_qa_stats.json")

    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_qa_data(args.input_file)
    print(f"Loaded {len(data)} questions.")

    # Generate statistics
    print("Generating statistics...")
    stats = generate_stats(data)

    # Write output
    print(f"Writing statistics to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total questions: {stats['total_questions']}")
    print(f"Unique videos: {stats['overall']['unique_videos']}")
    print("\nQuestions by category:")
    for category, cat_stats in stats["categories"].items():
        print(f"  {category}: {cat_stats['count']} ({cat_stats['percentage']:.2f}%)")
        print(f"    Unique videos: {cat_stats['unique_videos']}")
        print(f"    Ground truth distribution:")
        for idx, count in sorted(cat_stats["ground_truth_distribution"]["distribution_by_index"].items()):
            pct = cat_stats["ground_truth_distribution"]["percentage_by_index"][str(idx)]
            print(f"      Option {idx}: {count} ({pct:.2f}%)")
        print(f"    Top 5 most common correct options:")
        top_correct = list(cat_stats["ground_truth_options_distribution"]["top_20_most_common_correct"].items())[:5]
        total_correct = sum(cat_stats["ground_truth_options_distribution"]["distribution"].values())
        for option, count in top_correct:
            pct = (count / total_correct * 100) if total_correct > 0 else 0
            print(f"      \"{option[:60]}...\": {count} ({pct:.2f}%)" if len(option) > 60 else f"      \"{option}\": {count} ({pct:.2f}%)")
        if "bias_analysis" in cat_stats:
            bias = cat_stats["bias_analysis"]
            print(f"    Bias analysis:")
            print(f"      Most common correct: Option {bias['most_common_correct_index']} ({bias['most_common_correct_percentage']:.2f}%)")
            print(f"      Least common correct: Option {bias['least_common_correct_index']} ({bias['least_common_correct_percentage']:.2f}%)")
            print(f"      Imbalance ratio: {bias['imbalance_ratio']:.2f}")
            print(f"      Balanced: {bias['is_balanced_result']}")
    print("\n" + "=" * 60)
    print(f"Statistics saved to: {output_path}")


if __name__ == "__main__":
    main()

