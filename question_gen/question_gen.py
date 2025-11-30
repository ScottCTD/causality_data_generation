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

import json
import random
from collections import defaultdict
from typing import List, Dict, Tuple
import glob
import os
import argparse
from tqdm import tqdm

OPTION_POOL = [
    # Base/canonical phrasing (used for descriptive questions).
    # Other question types (predictive, counterfactual) are produced
    # by applying tense/voice transformations to these base strings.
    "The ball was pocketed",
    "The ball was pocketed in the gray pocket",
    "The ball was pocketed in the purple pocket",
    "The ball was pocketed in the blue pocket",
    "The ball was pocketed in the orange pocket",
    "The ball was pocketed in the green pocket",
    "The ball was pocketed in the red pocket",
    "The ball hits 0 walls",
    "The ball hits 1 wall",
    "The ball hits 2 different walls",
    "The ball hits 3 different walls",
    "The ball hits the same wall 2 times",
    "The ball hits the same wall 3 times",
    # "The ball bounced off a wall",
    # "The ball stayed on the table",
    "The ball was not pocketed",
    "The first wall hit was green-blue-wall",
    "The first wall hit was orange-red-wall",
    "The first wall hit was grey-orange-wall",
    "The first wall hit was purple-grey-wall",
    "The first wall hit was blue-purple-wall",
    "The first wall hit was red-green-wall",
    "The second wall hit was green-blue-wall",
    "The second wall hit was orange-red-wall",
    "The second wall hit was grey-orange-wall",
    "The second wall hit was purple-grey-wall",
    "The second wall hit was blue-purple-wall",
    "The second wall hit was red-green-wall",
    "The third wall hit was green-blue-wall",
    "The third wall hit was orange-red-wall",
    "The third wall hit was grey-orange-wall",
    "The third wall hit was purple-grey-wall",
    "The third wall hit was blue-purple-wall",
    "The third wall hit was red-green-wall",
    # Only include these if they can be correct in your data:
    # "The ball changed direction multiple times",
    # "The ball slowed down",
    # "The ball moved very fast",
    # "The ball stopped before reaching a wall",
    # "The ball went off the table",
    # "The ball continued in a straight line",
    # "The ball hit multiple walls",  # ambiguous, prefer explicit wall counts
]

NUM_OPTIONS = 6  # total options per question


def convert_option_tense(option: str, target: str) -> str:
    """
    Convert an option string from the base/canonical phrasing to a target tense/voice.

    Supported targets:
      - "base": leave unchanged (used for descriptive questions; past-ish tense)
      - "future": future tense (used for predictive questions)
      - "conditional": counterfactual/imagined outcome ("would" phrasing)
    """
    if target == "base":
        return option

    # We always start from the base canonical phrases, so we only need to
    # handle a small, explicit pattern set.
    if target == "future":
        option = option.replace("The ball was pocketed", "The ball will be pocketed")
        option = option.replace("The ball hits", "The ball will hit")
        option = option.replace("The ball hit", "The ball will hit")
        option = option.replace("The ball bounced", "The ball will bounce")
        option = option.replace("The ball stayed", "The ball will stay")
        option = option.replace(
            "The ball was not pocketed", "The ball will not be pocketed"
        )
        option = option.replace("The first wall hit was", "The first wall hit will be")
        option = option.replace(
            "The second wall hit was", "The second wall hit will be"
        )
        option = option.replace("The third wall hit was", "The third wall hit will be")
        return option

    if target == "conditional":
        option = option.replace("The ball was pocketed", "The ball would be pocketed")
        option = option.replace("The ball hits", "The ball would hit")
        option = option.replace("The ball hit", "The ball would hit")
        option = option.replace("The ball bounced", "The ball would bounce")
        option = option.replace("The ball stayed", "The ball would stay")
        option = option.replace(
            "The ball was not pocketed", "The ball would not be pocketed"
        )
        option = option.replace("The first wall hit was", "The first wall hit would be")
        option = option.replace(
            "The second wall hit was", "The second wall hit would be"
        )
        option = option.replace("The third wall hit was", "The third wall hit would be")
        return option

    # Fallback: if an unknown target is provided, return the original.
    return option


def has_hit_index_exceeding_threshold(sim_entry: Dict, max_hit_index: int) -> bool:
    """
    Check if any hit in the simulation entry has an index > max_hit_index.

    Args:
        sim_entry: A simulation data dictionary (loaded from JSON)
        max_hit_index: Maximum allowed hit index (exclusive)

    Returns:
        True if any hit has index > max_hit_index, False otherwise
    """
    if max_hit_index is None:
        return False

    # Check hits in all balls
    balls = sim_entry.get("balls", {})
    if not isinstance(balls, dict):
        return False

    for ball_key, ball_data in balls.items():
        if not isinstance(ball_data, dict):
            continue
        outcomes = ball_data.get("outcomes", {})
        if not isinstance(outcomes, dict):
            continue
        hits_list = outcomes.get("hits", [])
        if not isinstance(hits_list, list):
            continue

        for hit in hits_list:
            if isinstance(hit, dict):
                hit_index = hit.get("index")
                if hit_index is not None:
                    try:
                        if int(hit_index) > max_hit_index:
                            return True
                    except (ValueError, TypeError):
                        continue

    # Also check top-level outcomes if present
    outcomes_raw = sim_entry.get("outcomes", {})
    if isinstance(outcomes_raw, dict):
        hits_list = outcomes_raw.get("hits", [])
        if isinstance(hits_list, list):
            for hit in hits_list:
                if isinstance(hit, dict):
                    hit_index = hit.get("index")
                    if hit_index is not None:
                        try:
                            if int(hit_index) > max_hit_index:
                                return True
                        except (ValueError, TypeError):
                            continue

    return False


def filter_outcomes_for_predictive(outcomes_raw, total_frames, fraction=0.5):
    """
    Return a filtered version of outcomes_raw where wall hits with frame < fraction*total_frames are removed,
    and wall_hits count is adjusted accordingly.
    
    Args:
        outcomes_raw: Raw outcomes dictionary
        total_frames: Total number of frames in the video
        fraction: Fraction of video to filter out (default 0.5, meaning first half)
    """
    if not isinstance(outcomes_raw, dict):
        return outcomes_raw
    threshold_frames = fraction * total_frames if total_frames else 0
    hits_list = outcomes_raw.get("hits", [])
    filtered_hits = [
        h
        for h in hits_list
        if not (
            isinstance(h, dict)
            and h.get("type") == "wall"
            and h.get("frame", 0) < threshold_frames
        )
    ]
    wall_hits = [
        h.get("name")
        for h in filtered_hits
        if isinstance(h, dict) and h.get("type") == "wall"
    ]
    filtered = dict(outcomes_raw)
    filtered["hits"] = filtered_hits
    filtered["wall_hits"] = len(wall_hits)
    return filtered


def make_index(sim_data: List[Dict]):
    """
    Build normalized entries and a combined index keyed by (position_2dp, velocity_2dp).
    Enriches outcomes to include wall hit list (labels) and pocket color.
    Returns: (id_to_entry, index_by_pos_vel, pos_to_ids, vel_to_ids)
    """
    id_to_entry = {}
    index_by_pos_vel = {}
    pos_to_ids = defaultdict(list)
    vel_to_ids = defaultdict(list)

    for i, raw in enumerate(sim_data):
        norm = {}
        video = (
            raw.get("video")
            or (raw.get("metadata") or {}).get("shot_id")
            or f"shot_{i}"
        )

        # extract position, velocity, outcomes from possible structures
        balls = raw.get("balls", {})
        cue = balls.get("cue")
        if cue is None and isinstance(balls, dict) and balls:
            # pick first numeric-keyed ball if present (e.g., '1'), else fallback to any value
            first_key = next(iter(balls), None)
            cue = balls.get(first_key) if first_key is not None else None

        if cue is not None:
            pos_raw = cue.get("initial_position", cue.get("initial_pos", [0, 0]))
            vel_raw = cue.get("initial_velocity", cue.get("initial_vel", [0, 0]))
            outcomes_raw = cue.get("outcomes", {})
        else:
            pos_raw = raw.get("position", [0, 0])
            vel_raw = raw.get("velocity", [0, 0])
            outcomes_raw = raw.get("outcomes", {})

        # ensure numeric and take first two components (x,y); fallback to 0.0
        def round_components(arr):
            try:
                a0 = float(arr[0]) if len(arr) > 0 else 0.0
                a1 = float(arr[1]) if len(arr) > 1 else 0.0
                a2 = float(arr[2]) if len(arr) > 2 else 0.0
            except Exception:
                a0, a1, a2 = 0.0, 0.0, 0.0
            return [round(a0, 2), round(a1, 2), round(a2, 2)]

        pos2 = round_components(pos_raw)
        vel2 = round_components(vel_raw)

        # --- Enrich outcomes ---
        wall_hits = []
        pocket_val = None
        pocket_color = None
        if isinstance(outcomes_raw, dict):
            hits_list = outcomes_raw.get("hits", [])
            if isinstance(hits_list, list):
                wall_hits = [
                    h.get("name")
                    for h in hits_list
                    if isinstance(h, dict) and h.get("type") == "wall"
                ]
            pocket_val = outcomes_raw.get("pocket", outcomes_raw.get("pocketed", None))
            # If pocket_val is a dict with color, extract color
            if isinstance(pocket_val, dict):
                pocket_color = pocket_val.get("color")
            elif isinstance(pocket_val, str):
                pocket_color = pocket_val
            elif pocket_val is not None:
                pocket_color = str(pocket_val)
            num_wall_hits = outcomes_raw.get(
                "wall_hits", outcomes_raw.get("num_wall_hits", None)
            )
            if num_wall_hits is None and hits_list:
                num_wall_hits = sum(
                    1
                    for h in hits_list
                    if isinstance(h, dict) and h.get("type") == "wall"
                )
        else:
            num_wall_hits = 0

        pocketed = pocket_val is not None
        which_pocket = pocket_val if pocketed else None

        norm["video"] = video
        norm["initial_state"] = {"position": pos2, "velocity": vel2}
        norm["outcomes"] = {
            "num_wall_hits": int(num_wall_hits) if num_wall_hits is not None else 0,
            "wall_hits": wall_hits,
            "pocketed": bool(pocketed),
            "which_pocket": which_pocket,
            "pocket_color": pocket_color,
        }

        id_to_entry[i] = norm

        key = (tuple(pos2), tuple(vel2))
        # store first encountered sim id for this (pos, vel) pair
        if key not in index_by_pos_vel:
            index_by_pos_vel[key] = i
        pos_to_ids[tuple(pos2)].append(i)
        vel_to_ids[tuple(vel2)].append(i)

    return id_to_entry, index_by_pos_vel, pos_to_ids, vel_to_ids


def outcome_options_from_outcome(outcomes: Dict, tense: str = "base") -> List[str]:
    """Return a set of plausible true-option strings given outcomes dict.

    Args:
        outcomes: Dictionary containing outcome information
        tense: One of {"base", "future", "conditional"} indicating how to phrase options.
    """
    opts = []
    hits = outcomes.get("num_wall_hits", 0)
    wall_hits = outcomes.get("wall_hits", [])
    pocketed = outcomes.get("pocketed", False)
    which = outcomes.get("which_pocket", None)
    pocket_color = outcomes.get("pocket_color", None)

    if pocketed:
        opts.append("The ball was pocketed")
        if pocket_color:
            opts.append(f"The ball was pocketed in the {pocket_color} pocket")
    else:
        # opts.append("The ball stayed on the table")
        opts.append("The ball was not pocketed")

    # wall-hit statements
    if hits == 0:
        opts.append("The ball hits 0 walls")
    elif hits == 1:
        opts.append("The ball hits 1 wall")
    elif hits >= 2:
        # Check if all walls are the same or different
        # Only check for same wall if we have complete wall_hits data
        if wall_hits and len(wall_hits) == hits:
            unique_walls = len(set(wall_hits))
            if unique_walls == 1:
                # Same wall hit multiple times
                opts.append(f"The ball hits the same wall {hits} times")
            else:
                # Different walls
                opts.append(f"The ball hits {hits} different walls")
        else:
            # Default to "different walls" if wall_hits data is incomplete
            opts.append(f"The ball hits {hits} different walls")
    # if hits >= 1:
    #     opts.append("The ball bounced off a wall")

    # Add wall hit sequence options
    if wall_hits:
        opts.append(f"The first wall hit was {wall_hits[0]}")
        if len(wall_hits) > 1:
            opts.append(f"The second wall hit was {wall_hits[1]}")
        if len(wall_hits) > 2:
            opts.append(f"The third wall hit was {wall_hits[2]}")
    opts = list(dict.fromkeys(opts))  # dedupe, preserve order

    # Convert to desired tense if requested
    if tense != "base":
        opts = [convert_option_tense(opt, tense) for opt in opts]

    return opts


def sample_multilabel_options(
    true_opts: List[str], pool: List[str], total=4, num_correct=2, tense: str = "base"
):
    """
    Pick num_correct correct options randomly from true_opts,
    then fill to `total` with distractors from pool not overlapping chosen corrects.
    Only sample distractors that are not logically inconsistent with the true options.

    Args:
        true_opts: List of correct option strings (already phrased in the desired tense).
        pool: Pool of all possible options to sample distractors from (in base tense).
        total: Total number of options to return
        num_correct: Number of correct options to include
        tense: Desired tense/voice for all returned options
    """
    if not true_opts:
        # Fallback default when there is no information: use a simple "not pocketed" statement.
        base_default = "The ball was not pocketed"
        true_opts = [convert_option_tense(base_default, tense)]

    num_correct = min(num_correct, len(true_opts), total)
    chosen_correct = random.sample(true_opts, num_correct)

    # Filter distractors to avoid logical inconsistency
    def is_consistent(opt, correct_opts):
        # Normalize option for comparison (remove tense differences)
        def normalize_tense(s):
            # Convert future/conditional/present tense to past/base tense for comparison
            s = s.replace("will be pocketed", "was pocketed")
            s = s.replace("will hit", "hit")
            s = s.replace("hits", "hit")  # Normalize present tense "hits" to "hit"
            s = s.replace("will bounce", "bounced")
            s = s.replace("will stay", "stayed")
            s = s.replace("will not be pocketed", "was not pocketed")
            s = s.replace("The first wall hit will be", "The first wall hit was")
            s = s.replace("The second wall hit will be", "The second wall hit was")
            s = s.replace("The third wall hit will be", "The third wall hit was")
            s = s.replace("would be pocketed", "was pocketed")
            s = s.replace("would hit", "hit")
            s = s.replace("would bounce", "bounced")
            s = s.replace("would stay", "stayed")
            s = s.replace("would not be pocketed", "was not pocketed")
            s = s.replace("The first wall hit would be", "The first wall hit was")
            s = s.replace("The second wall hit would be", "The second wall hit was")
            s = s.replace("The third wall hit would be", "The third wall hit was")
            s = s.replace("will be", "was")
            return s

        opt_norm = normalize_tense(opt)
        correct_opts_norm = [normalize_tense(o) for o in correct_opts]

        # Don't allow "The ball was not pocketed" if a correct option says it was pocketed, and vice versa
        if (
            "The ball was pocketed" in correct_opts_norm
            and opt_norm == "The ball was not pocketed"
        ) or (
            "The ball was not pocketed" in correct_opts_norm
            and opt_norm == "The ball was pocketed"
        ):
            return False
        # Don't allow "The ball hit 0 walls" if a correct option says it hit walls, and vice versa
        if (
            any(o.startswith("The ball hit 0 wall") for o in correct_opts_norm)
            and "wall hit" in opt_norm
            and not opt_norm.startswith("The ball hit 0 wall")
        ):
            return False
        if any(
            o.startswith("The ball hit") and "0 wall" not in o
            for o in correct_opts_norm
        ) and opt_norm.startswith("The ball hit 0 wall"):
            return False
        # Don't allow "The ball stayed on the table" if a correct option says it was pocketed
        if (
            "The ball was pocketed" in correct_opts_norm
            and opt_norm == "The ball stayed on the table"
        ):
            return False
        return True

    distractor_candidates = [
        o for o in pool if o not in chosen_correct and is_consistent(o, chosen_correct)
    ]
    # We may not always have enough consistent distractors to fill `total - num_correct`.
    # In that case, just take as many as we can without error.
    num_distractors_needed = max(0, total - num_correct)
    num_distractors_taken = min(num_distractors_needed, len(distractor_candidates))
    distractors = random.sample(distractor_candidates, k=num_distractors_taken)

    # Convert distractors to desired tense if needed
    if tense != "base":
        distractors = [convert_option_tense(d, tense) for d in distractors]

    # Merge correct options and distractors, removing any duplicate strings
    # that might arise after tense conversion (e.g., a distractor mapping to
    # the same surface form as a correct option). Correct options are listed
    # first so they are preserved when duplicates occur.
    labeled_opts = [(opt, True) for opt in chosen_correct] + [
        (opt, False) for opt in distractors
    ]
    seen = set()
    deduped_labeled = []
    for opt, is_correct in labeled_opts:
        if opt in seen:
            continue
        seen.add(opt)
        deduped_labeled.append((opt, is_correct))

    # Now shuffle while preserving which options are correct.
    random.shuffle(deduped_labeled)
    all_opts = [opt for opt, _ in deduped_labeled]
    ground_indices = [
        i for i, (_, is_correct) in enumerate(deduped_labeled) if is_correct
    ]
    return all_opts, ground_indices


# ---------------------------
# Counterfactual neighbor selection (using combined index keyed by (pos,vel))
# ---------------------------
def find_velocity_cfs(
    pos: Tuple[float, float], vel: Tuple[float, float], pos_to_ids, id2entry, n=3
):
    """
    Find up to n random shots with the same position but different velocity.
    """
    candidates = pos_to_ids.get(tuple(pos), [])
    candidates = [
        i
        for i in candidates
        if tuple(id2entry[i]["initial_state"]["velocity"]) != tuple(vel)
    ]
    if not candidates:
        return []
    return random.sample(candidates, min(n, len(candidates)))


def find_position_cfs(
    pos: Tuple[float, float], vel: Tuple[float, float], vel_to_ids, id2entry, n=3
):
    """
    Find up to n random shots with the same velocity but different position.
    """
    candidates = vel_to_ids.get(tuple(vel), [])
    candidates = [
        i
        for i in candidates
        if tuple(id2entry[i]["initial_state"]["position"]) != tuple(pos)
    ]
    if not candidates:
        return []
    return random.sample(candidates, min(n, len(candidates)))


def coord_to_str(coord, prefix="") -> str:
    # Round very small values to zero to avoid unnatural formats like "-0.00"
    x = coord[0] if abs(coord[0]) >= 0.005 else 0.0
    y = coord[1] if abs(coord[1]) >= 0.005 else 0.0
    return f"({prefix}x={x:.2f}, {prefix}y={y:.2f})"


def generate_sft_mcq_multilabel(
    sim_data: List[Dict],
    num_options: int,
    num_correct: int,
    num_descriptive_per_shot: int = 1,
    num_predictive_per_shot: int = 1,
    max_velocity_cfs_per_shot: int = 3,
    max_position_cfs_per_shot: int = 3,
    predictive_filter_fraction: float = 0.5,
):
    """
    Generate dataset with the following schema for each example:
      - video: str
      - question: str
      - options: List[str]  # length num_options
      - ground_truth: List[int]  # list of 0-based indices into options
      - metadata: {...}
    """
    id2entry, idx_pos_vel, pos_to_ids, vel_to_ids = make_index(sim_data)
    out_dataset = []

    for sim_id, entry in tqdm(id2entry.items(), desc="Generating questions"):
        video = entry["video"]
        pos = tuple(entry["initial_state"]["position"])
        vel = tuple(entry["initial_state"]["velocity"])
        outcomes = entry["outcomes"]

        w, h = 0.9906, 1.9812
        context_text = f"Pocket locations: red at (0, 0), green at ({w}, 0), gray at (0, {h}), and purple at ({w}, {h}). Walls are named by the colors of the two pockets they connect (e.g., the 'red-green' wall is between the red and green pockets)."

        # --- DESCRIPTIVE question(s) (full video) ---
        # Use base/canonical phrasing (past-ish tense is fine for "What happened?").
        if num_descriptive_per_shot > 0:
            true_opts_desc = outcome_options_from_outcome(outcomes, tense="base")
            for q_idx in range(num_descriptive_per_shot):
                # Only generate additional descriptive questions beyond the first
                # if we have enough distinct true options to support some variety.
                if q_idx > 0 and len(true_opts_desc) <= num_correct:
                    break
                options_list, ground_indices = sample_multilabel_options(
                    true_opts_desc,
                    OPTION_POOL,
                    total=num_options,
                    num_correct=num_correct,
                    tense="base",
                )
                question_text = (
                    "Context: "
                    + context_text
                    + "\nQuestion: What happened in this video?"
                )
                out_dataset.append(
                    {
                        "video": video,
                        "question": question_text,
                        "options": options_list,
                        "ground_truth": ground_indices,
                        "metadata": {
                            "question_type": "descriptive",
                            "sim_id": sim_id,
                            "question_index_within_shot": q_idx,
                        },
                    }
                )

        # --- PREDICTIVE question (first-half video) ---
        # Filter wall hits for predictive question only
        # Get total_frames from root or metadata
        raw = sim_data[sim_id]
        total_frames = raw.get("total_frames")
        if total_frames is None:
            total_frames = (raw.get("metadata") or {}).get("total_frames")
        if total_frames is None:
            total_frames = 0
        # Reparse outcomes_raw for predictive
        balls = raw.get("balls", {})
        cue = balls.get("cue")
        if cue is None and isinstance(balls, dict) and balls:
            first_key = next(iter(balls), None)
            cue = balls.get(first_key) if first_key is not None else None
        if cue is not None:
            outcomes_raw = cue.get("outcomes", {})
        else:
            outcomes_raw = raw.get("outcomes", {})
        filtered_outcomes_raw = filter_outcomes_for_predictive(
            outcomes_raw, total_frames, fraction=predictive_filter_fraction
        )
        # Build filtered outcomes dict for options
        hits_list = filtered_outcomes_raw.get("hits", [])
        wall_hits = [
            h.get("name")
            for h in hits_list
            if isinstance(h, dict) and h.get("type") == "wall"
        ]
        pocket_val = filtered_outcomes_raw.get(
            "pocket", filtered_outcomes_raw.get("pocketed", None)
        )
        if isinstance(pocket_val, dict):
            pocket_color = pocket_val.get("color")
        elif isinstance(pocket_val, str):
            pocket_color = pocket_val
        elif pocket_val is not None:
            pocket_color = str(pocket_val)
        else:
            pocket_color = None
        num_wall_hits = filtered_outcomes_raw.get(
            "wall_hits", filtered_outcomes_raw.get("num_wall_hits", None)
        )
        if num_wall_hits is None and hits_list:
            num_wall_hits = sum(
                1 for h in hits_list if isinstance(h, dict) and h.get("type") == "wall"
            )
        pocketed = pocket_val is not None
        which_pocket = pocket_val if pocketed else None
        filtered_outcomes = {
            "num_wall_hits": int(num_wall_hits) if num_wall_hits is not None else 0,
            "wall_hits": wall_hits,
            "pocketed": bool(pocketed),
            "which_pocket": which_pocket,
            "pocket_color": pocket_color,
        }
        if num_predictive_per_shot > 0:
            true_opts_predictive = outcome_options_from_outcome(
                filtered_outcomes, tense="future"
            )
            for q_idx in range(num_predictive_per_shot):
                # Only generate additional predictive questions beyond the first
                # if we have enough distinct true options to support some variety.
                if q_idx > 0 and len(true_opts_predictive) <= num_correct:
                    break
                options_list, ground_indices = sample_multilabel_options(
                    true_opts_predictive,
                    OPTION_POOL,
                    total=num_options,
                    num_correct=num_correct,
                    tense="future",
                )
                question_text = (
                    "Context: "
                    + context_text
                    + "\nQuestion: Based on the first part of the video, what will happen in "
                    "STRICTLY the second part of the video?"
                )
                out_dataset.append(
                    {
                        "video": video.replace(".mp4", "_partial.mp4"),
                        "question": question_text,
                        "options": options_list,
                        "ground_truth": ground_indices,
                        "metadata": {
                            "question_type": "predictive",
                            "sim_id": sim_id,
                            "question_index_within_shot": q_idx,
                        },
                    }
                )

        # --- COUNTERFACTUALS: up to N velocity and M position neighbors ---
        if max_velocity_cfs_per_shot > 0:
            vel_cf_ids = find_velocity_cfs(
                pos, vel, pos_to_ids, id2entry, n=max_velocity_cfs_per_shot
            )
            for vel_cf_id in vel_cf_ids:
                cf_entry = id2entry[vel_cf_id]
                cf_out = cf_entry["outcomes"]
                # Counterfactuals: phrase options in conditional tense ("would ...").
                true_opts_cf = outcome_options_from_outcome(cf_out, tense="conditional")
                options_list, ground_indices = sample_multilabel_options(
                    true_opts_cf,
                    OPTION_POOL,
                    total=num_options,
                    num_correct=num_correct,
                    tense="conditional",
                )
                question_text = (
                    f"Context: {context_text}\n"
                    f"Question: If the initial velocity were changed from {coord_to_str(vel, prefix='d')} "
                    f"to {coord_to_str(cf_entry['initial_state']['velocity'], prefix='d')} "
                    f"(assume all other variables are unchanged), what would happen?"
                )
                out_dataset.append(
                    {
                        "video": video,
                        "question": question_text,
                        "options": options_list,
                        "ground_truth": ground_indices,
                        "metadata": {
                            "question_type": "counterfactual_velocity",
                            "sim_id": sim_id,
                            "counterfactual_sim_id": vel_cf_id,
                            "counterfactual_video": cf_entry["video"],
                            "counterfactual_initial_state": cf_entry["initial_state"],
                        },
                    }
                )

        if max_position_cfs_per_shot > 0:
            pos_cf_ids = find_position_cfs(
                pos, vel, vel_to_ids, id2entry, n=max_position_cfs_per_shot
            )
            for pos_cf_id in pos_cf_ids:
                cf_entry = id2entry[pos_cf_id]
                cf_out = cf_entry["outcomes"]
                true_opts_cf = outcome_options_from_outcome(cf_out, tense="conditional")
                options_list, ground_indices = sample_multilabel_options(
                    true_opts_cf,
                    OPTION_POOL,
                    total=num_options,
                    num_correct=num_correct,
                    tense="conditional",
                )
                question_text = (
                    f"Context: {context_text}\n"
                    f"Question: If the initial ball position were changed from {coord_to_str(pos)} "
                    f"to {coord_to_str(cf_entry['initial_state']['position'])} "
                    f"(assume all other variables are unchanged), what would happen?"
                )
                out_dataset.append(
                    {
                        "video": video,
                        "question": question_text,
                        "options": options_list,
                        "ground_truth": ground_indices,
                        "metadata": {
                            "question_type": "counterfactual_position",
                            "sim_id": sim_id,
                            "counterfactual_sim_id": pos_cf_id,
                            "counterfactual_video": cf_entry["video"],
                            "counterfactual_initial_state": cf_entry["initial_state"],
                        },
                    }
                )

    return out_dataset


if __name__ == "__main__":
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
        default=2,
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
        help="Fraction of video to filter out for predictive questions (default: 0.5, meaning first half). "
        "E.g., 0.3 means filter out wall hits from first 30%% of video.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    dataset_dir = os.path.join("outputs", args.dataset)
    shots_pattern = os.path.join(dataset_dir, "shots", "shot_*", "*.json")

    sim_data = []
    excluded_count = 0
    # i= 0
    # limit=100
    all_files = glob.glob(shots_pattern)
    for fname in tqdm(all_files, desc="Loading simulation data"):
        # if i >= limit:
        #     break
        with open(fname, "r") as f:
            entry = json.load(f)
            # Filter out entries with hit index > 18 if flag is set
            if args.exclude_invalid_hits and has_hit_index_exceeding_threshold(
                entry, 18
            ):
                excluded_count += 1
                continue
            sim_data.append(entry)
        # i += 1

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
    # Write to jsonl
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(dataset_dir, "raw_qa.jsonl")
    with open(output_path, "w") as f:
        for ex in dataset:
            f.write(json.dumps(ex) + "\n")
    print("Wrote", len(dataset), "examples to", output_path)
