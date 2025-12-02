from __future__ import annotations

from typing import Dict, List

from tqdm import tqdm

try:  # Package import (python -m question_gen.generator)
    from .data_utils import (
        coord_to_str,
        find_position_cfs,
        find_velocity_cfs,
        make_index,
    )
    from .options import (
        DISTRACTOR_POOL_FACTS,
        OptionRenderer,
        Tense,
        facts_from_outcome,
        sample_multilabel_from_facts,
    )
except ImportError:  # Script-level import (import generator)
    from data_utils import (  # type: ignore
        coord_to_str,
        find_position_cfs,
        find_velocity_cfs,
        make_index,
    )
    from options import (  # type: ignore
        DISTRACTOR_POOL_FACTS,
        OptionRenderer,
        Tense,
        facts_from_outcome,
        sample_multilabel_from_facts,
    )


def generate_sft_mcq_multilabel(
    sim_data: List[Dict],
    num_options: int,
    num_correct: int,
    num_descriptive_per_shot: int = 1,
    num_predictive_per_shot: int = 1,
    max_velocity_cfs_per_shot: int = 3,
    max_position_cfs_per_shot: int = 3,
    predictive_filter_fraction: float = 0.5,
) -> List[Dict]:
    """
    Generate a multi-label MCQ dataset from simulation metadata.

    Returns a list of examples, each with:
      - ``video``: str
      - ``question``: str
      - ``options``: List[str]
      - ``ground_truth``: List[int]
      - ``metadata``: Dict
    """
    id2entry, _, pos_to_ids, vel_to_ids = make_index(sim_data)
    renderer = OptionRenderer()
    out_dataset: List[Dict] = []

    for sim_id, entry in tqdm(id2entry.items(), desc="Generating questions"):
        video = entry["video"]
        pos = tuple(entry["initial_state"]["position"])
        vel = tuple(entry["initial_state"]["velocity"])
        outcomes = entry["outcomes"]

        w, h = 0.9906, 1.9812
        m = round(h / 2, 4)
        context_text = (
            f"The pool table has a width of {w} and a height of {h}. "
            "Pockets are marked by colored squares near them. Pocket locations: red at (0, 0), green at "
            f"({w}, 0), orange at (0, {m}), blue at ({w}, {m}), gray at (0, {h}), and purple at ({w}, {h}). "
            "Walls are named by the colors of the two pockets they connect "
            "(e.g., the 'red-green' wall is between the red and green pockets). "
            "Answer the following question by considering the cue ball (white) movements on the pool table."
        )

        # --- DESCRIPTIVE question(s) (full video) ---
        if num_descriptive_per_shot > 0:
            true_facts_desc = facts_from_outcome(outcomes)
            for q_idx in range(num_descriptive_per_shot):
                if q_idx > 0 and len(true_facts_desc) <= num_correct:
                    break
                options_list, ground_indices = sample_multilabel_from_facts(
                    true_facts_desc,
                    DISTRACTOR_POOL_FACTS,
                    total=num_options,
                    num_correct=num_correct,
                    tense=Tense.BASE,
                    renderer=renderer,
                )
                # If sampling produced no valid correct options (e.g., all
                # candidate facts were filtered out), skip this question.
                if not options_list or not ground_indices:
                    continue
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

        # --- PREDICTIVE question (first-half vs. second-half video) ---
        raw = sim_data[sim_id]
        total_frames = entry.get("total_frames")
        if total_frames is None:
            total_frames = (raw.get("metadata") or {}).get("total_frames")
        if total_frames is None:
            total_frames = 0

        hits_detail = entry.get("hits_detail", [])
        threshold_frames = (
            predictive_filter_fraction * total_frames if total_frames else 0
        )
        filtered_hits = [
            h
            for h in hits_detail
            if isinstance(h, dict)
            and h.get("type") == "wall"
            and h.get("frame", 0) >= threshold_frames
        ]
        wall_hits = [h.get("name") for h in filtered_hits if isinstance(h, dict)]

        base_outcomes = entry["outcomes"]
        filtered_outcomes = dict(base_outcomes)
        filtered_outcomes["num_wall_hits"] = len(wall_hits)
        filtered_outcomes["wall_hits"] = wall_hits

        if num_predictive_per_shot > 0:
            true_facts_predictive = facts_from_outcome(filtered_outcomes)
            for q_idx in range(num_predictive_per_shot):
                if q_idx > 0 and len(true_facts_predictive) <= num_correct:
                    break
                options_list, ground_indices = sample_multilabel_from_facts(
                    true_facts_predictive,
                    DISTRACTOR_POOL_FACTS,
                    total=num_options,
                    num_correct=num_correct,
                    tense=Tense.FUTURE,
                    renderer=renderer,
                )
                if not options_list or not ground_indices:
                    continue
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
                true_facts_cf = facts_from_outcome(cf_out)
                options_list, ground_indices = sample_multilabel_from_facts(
                    true_facts_cf,
                    DISTRACTOR_POOL_FACTS,
                    total=num_options,
                    num_correct=num_correct,
                    tense=Tense.CONDITIONAL,
                    renderer=renderer,
                )
                if not options_list or not ground_indices:
                    continue
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
                            "counterfactual_initial_state": cf_entry[
                                "initial_state"
                            ],
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
                true_facts_cf = facts_from_outcome(cf_out)
                options_list, ground_indices = sample_multilabel_from_facts(
                    true_facts_cf,
                    DISTRACTOR_POOL_FACTS,
                    total=num_options,
                    num_correct=num_correct,
                    tense=Tense.CONDITIONAL,
                    renderer=renderer,
                )
                if not options_list or not ground_indices:
                    continue
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
                            "counterfactual_initial_state": cf_entry[
                                "initial_state"
                            ],
                        },
                    }
                )

    return out_dataset
