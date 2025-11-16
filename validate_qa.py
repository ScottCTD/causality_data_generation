import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple


ALLOWED_QUESTION_TYPES = {
    "descriptive",
    "predictive",
    "counterfactual_velocity",
    "counterfactual_position",
}


def detect_tense(option: str) -> str:
    """
    Heuristic tense detector for options.

    Returns one of:
      - "future"      -> contains ' will '
      - "conditional" -> contains ' would '
      - "base"        -> neither of the above
    """
    # Use word boundaries-ish to avoid matching substrings of other words.
    if re.search(r"\bwill\b", option):
        return "future"
    if re.search(r"\bwould\b", option):
        return "conditional"
    return "base"


def validate_entry_schema(ex: Dict[str, Any], line_idx: int) -> List[str]:
    """Basic schema and type checks for a single QA example."""
    issues: List[str] = []

    # Required top-level keys and types
    if not isinstance(ex.get("video"), str):
        issues.append(f"line {line_idx}: 'video' missing or not a string")
    if not isinstance(ex.get("question"), str):
        issues.append(f"line {line_idx}: 'question' missing or not a string")

    options = ex.get("options")
    if not isinstance(options, list) or not options:
        issues.append(f"line {line_idx}: 'options' missing, not a list, or empty")
    else:
        # Ensure all options are strings
        for i, opt in enumerate(options):
            if not isinstance(opt, str):
                issues.append(
                    f"line {line_idx}: option[{i}] is not a string (type={type(opt)})"
                )

    gt = ex.get("ground_truth")
    if not isinstance(gt, list):
        issues.append(f"line {line_idx}: 'ground_truth' missing or not a list")
    else:
        for i, idx in enumerate(gt):
            if not isinstance(idx, int):
                issues.append(
                    f"line {line_idx}: ground_truth[{i}] is not an int (value={idx!r})"
                )

    meta = ex.get("metadata")
    if not isinstance(meta, dict):
        issues.append(f"line {line_idx}: 'metadata' missing or not a dict")
    else:
        q_type = meta.get("question_type")
        if q_type not in ALLOWED_QUESTION_TYPES:
            issues.append(
                f"line {line_idx}: metadata.question_type={q_type!r} "
                f"not in {sorted(ALLOWED_QUESTION_TYPES)}"
            )

    # Cross-field checks that need options / gt
    if isinstance(options, list) and options and isinstance(gt, list):
        # No duplicate options in a single example
        if len(set(options)) != len(options):
            issues.append(
                f"line {line_idx}: duplicate option strings within the same question"
            )

        # ground_truth indices within range and unique
        num_opts = len(options)
        seen_idx = set()
        for idx in gt:
            if not isinstance(idx, int):
                continue
            if idx < 0 or idx >= num_opts:
                issues.append(
                    f"line {line_idx}: ground_truth index {idx} out of bounds "
                    f"for {num_opts} options"
                )
            if idx in seen_idx:
                issues.append(
                    f"line {line_idx}: duplicate ground_truth index {idx} "
                    f"within the same example"
                )
            seen_idx.add(idx)

    return issues


def validate_tense_consistency(ex: Dict[str, Any], line_idx: int) -> List[str]:
    """Check that option tense matches question_type (base vs future vs conditional)."""
    issues: List[str] = []
    meta = ex.get("metadata") or {}
    q_type = meta.get("question_type")
    options = ex.get("options") or []

    if q_type not in ALLOWED_QUESTION_TYPES:
        return issues

    # Expected tense by question type
    if q_type == "descriptive":
        expected_tense = "base"
    elif q_type == "predictive":
        expected_tense = "future"
    else:  # counterfactual_velocity / counterfactual_position
        expected_tense = "conditional"

    for i, opt in enumerate(options):
        if not isinstance(opt, str):
            continue
        tense = detect_tense(opt)
        if expected_tense == "base":
            # For descriptive, disallow any "will"/"would" based phrasing.
            if tense != "base":
                issues.append(
                    f"line {line_idx}: option[{i}] has tense={tense!r} "
                    f"but expected base for question_type={q_type!r}"
                )
        elif expected_tense == "future":
            # For predictive, disallow "would" and require at least one "will"
            # across the options. We enforce the stronger constraint at the
            # question level below; here we just flag clearly wrong tense.
            if tense == "conditional":
                issues.append(
                    f"line {line_idx}: option[{i}] uses conditional 'would' "
                    f"but question_type is 'predictive'"
                )
        elif expected_tense == "conditional":
            # For counterfactual, disallow "will" and require "would".
            if tense == "future":
                issues.append(
                    f"line {line_idx}: option[{i}] uses 'will' "
                    f"but question_type is {q_type!r}"
                )

    # Question-level tense checks
    if options:
        tenses = {detect_tense(opt) for opt in options if isinstance(opt, str)}
        if expected_tense == "future":
            if "future" not in tenses:
                issues.append(
                    f"line {line_idx}: predictive question has no 'will' options "
                    f"(tenses seen={tenses})"
                )
        if expected_tense == "conditional":
            if "conditional" not in tenses:
                issues.append(
                    f"line {line_idx}: counterfactual question has no 'would' options "
                    f"(tenses seen={tenses})"
                )

    return issues


def validate_question_indices(
    group_indices: Dict[Tuple[Any, str], List[int]]
) -> List[str]:
    """
    Validate that question_index_within_shot, when present, is dense and 0-based
    for each (sim_id, question_type) group.
    """
    issues: List[str] = []
    for (sim_id, q_type), idxs in group_indices.items():
        if not idxs:
            continue
        uniq = sorted(set(idxs))
        expected = list(range(len(uniq)))
        if uniq != expected:
            issues.append(
                f"sim_id={sim_id}, question_type={q_type!r}: "
                f"question_index_within_shot values {uniq} not equal to {expected}"
            )
    return issues


def validate_qa_file(path: str, max_issues: int = 1000) -> None:
    """
    Run a comprehensive set of consistency checks on a QA jsonl file.

    This focuses on internal consistency of the QA dataset:
      - JSON parse + schema integrity
      - Option and ground_truth structure
      - No duplicate options per question
      - Tense consistency with question_type
      - Consistent question_index_within_shot sequencing per sim_id/question_type
    """
    total = 0
    issues: List[str] = []
    counts_by_type: Counter = Counter()
    num_correct_hist: Counter = Counter()
    num_options_hist: Counter = Counter()

    # Collect question_index_within_shot per (sim_id, question_type)
    group_indices: Dict[Tuple[Any, str], List[int]] = defaultdict(list)

    with open(path, "r") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                issues.append(f"line {line_idx}: JSON decode error: {e}")
                if len(issues) >= max_issues:
                    break
                continue

            # Basic schema and structure checks
            issues.extend(validate_entry_schema(ex, line_idx))
            if len(issues) >= max_issues:
                break

            options = ex.get("options") or []
            gt = ex.get("ground_truth") or []
            meta = ex.get("metadata") or {}
            q_type = meta.get("question_type")
            sim_id = meta.get("sim_id")

            if q_type in ALLOWED_QUESTION_TYPES:
                counts_by_type[q_type] += 1
            num_options_hist[len(options)] += 1
            # Only count valid int indices for histogram purposes
            num_correct_hist[len([i for i in gt if isinstance(i, int)])] += 1

            # Tense consistency checks
            issues.extend(validate_tense_consistency(ex, line_idx))
            if len(issues) >= max_issues:
                break

            # Track question_index_within_shot, when present
            if sim_id is not None and q_type is not None:
                q_idx = meta.get("question_index_within_shot")
                if q_idx is not None:
                    if isinstance(q_idx, int):
                        group_indices[(sim_id, q_type)].append(q_idx)
                    else:
                        issues.append(
                            f"line {line_idx}: question_index_within_shot is not an int "
                            f"(value={q_idx!r})"
                        )
                        if len(issues) >= max_issues:
                            break

    # Validate question_index_within_shot sequences
    issues.extend(validate_question_indices(group_indices))

    # Print summary
    print(f"Validated {total} examples from {os.path.abspath(path)}")
    print("Counts by question_type:", dict(counts_by_type))
    print("Histogram of number of options per question:", dict(num_options_hist))
    print("Histogram of number of correct options per question:", dict(num_correct_hist))

    if issues:
        print(f"\nFound {len(issues)} issues (showing up to {max_issues}):")
        for msg in issues[:max_issues]:
            print(" -", msg)
    else:
        print("\nNo issues found.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a MCQ jsonl dataset for internal consistency."
    )
    parser.add_argument(
        "qa_path",
        type=str,
        help="Path to raw_qa.jsonl (or any jsonl file with the same schema).",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=1000,
        help="Maximum number of issues to report before stopping.",
    )
    args = parser.parse_args()

    validate_qa_file(args.qa_path, max_issues=args.max_issues)


if __name__ == "__main__":
    main()


