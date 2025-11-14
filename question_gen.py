# The following script takes the shot metadata JSON files generated from the simulations
# and produces a dataset of multiple-choice questions (MCQs) suitable for training/evaluating

import json
import random
from collections import defaultdict
from typing import List, Dict, Tuple
import glob
import os


OPTION_POOL = [
    "The ball was pocketed",
    "The ball was pocketed in the gray pocket",
    "The ball was pocketed in the purple pocket",
    "The ball was pocketed in the blue pocket",
    "The ball was pocketed in the orange pocket",
    "The ball was pocketed in the green pocket",
    "The ball was pocketed in the red pocket",
    "The ball hit 0 walls",
    "The ball hit 1 wall",
    "The ball hit 2 walls",
    "The ball hit 3 different walls",
    "The ball bounced off a wall",
    "The ball stayed on the table",
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
        video = raw.get('video') or (raw.get('metadata') or {}).get('shot_id') or f'shot_{i}'

        # extract position, velocity, outcomes from possible structures
        balls = raw.get('balls', {})
        cue = balls.get('cue')
        if cue is None and isinstance(balls, dict) and balls:
            # pick first numeric-keyed ball if present (e.g., '1'), else fallback to any value
            first_key = next(iter(balls), None)
            cue = balls.get(first_key) if first_key is not None else None

        if cue is not None:
            pos_raw = cue.get('initial_position', cue.get('initial_pos', [0, 0]))
            vel_raw = cue.get('initial_velocity', cue.get('initial_vel', [0, 0]))
            outcomes_raw = cue.get('outcomes', {})
        else:
            pos_raw = raw.get('position', [0, 0])
            vel_raw = raw.get('velocity', [0, 0])
            outcomes_raw = raw.get('outcomes', {})

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
            hits_list = outcomes_raw.get('hits', [])
            if isinstance(hits_list, list):
                wall_hits = [h.get('name') for h in hits_list if isinstance(h, dict) and h.get('type') == 'wall']
            pocket_val = outcomes_raw.get('pocket', outcomes_raw.get('pocketed', None))
            # If pocket_val is a dict with color, extract color
            if isinstance(pocket_val, dict):
                pocket_color = pocket_val.get('color')
            elif isinstance(pocket_val, str):
                pocket_color = pocket_val
            elif pocket_val is not None:
                pocket_color = str(pocket_val)
            num_wall_hits = outcomes_raw.get('wall_hits', outcomes_raw.get('num_wall_hits', None))
            if num_wall_hits is None and hits_list:
                num_wall_hits = sum(1 for h in hits_list if isinstance(h, dict) and h.get('type') == 'wall')
        else:
            num_wall_hits = 0

        pocketed = pocket_val is not None
        which_pocket = pocket_val if pocketed else None

        norm['video'] = video
        norm['initial_state'] = {'position': pos2, 'velocity': vel2}
        norm['outcomes'] = {
            'num_wall_hits': int(num_wall_hits) if num_wall_hits is not None else 0,
            'wall_hits': wall_hits,
            'pocketed': bool(pocketed),
            'which_pocket': which_pocket,
            'pocket_color': pocket_color
        }

        id_to_entry[i] = norm

        key = (tuple(pos2), tuple(vel2))
        # store first encountered sim id for this (pos, vel) pair
        if key not in index_by_pos_vel:
            index_by_pos_vel[key] = i
        pos_to_ids[tuple(pos2)].append(i)
        vel_to_ids[tuple(vel2)].append(i)

    return id_to_entry, index_by_pos_vel, pos_to_ids, vel_to_ids

def outcome_options_from_outcome(outcomes: Dict) -> List[str]:
    """Return a set of plausible true-option strings given outcomes dict, including wall hit labels and pocket color."""
    opts = []
    hits = outcomes.get("num_wall_hits", 0)
    wall_hits = outcomes.get("wall_hits", [])
    pocketed = outcomes.get("pocketed", False)
    which = outcomes.get("which_pocket", None)
    pocket_color = outcomes.get("pocket_color", None)

    if pocketed:
        opts.append("The ball was pocketed")
        if which in (1,2,3,4,5,6):
            opts.append(f"The ball fell into pocket {which}")
        if pocket_color:
            opts.append(f"The ball was pocketed in the {pocket_color} pocket")
    else:
        opts.append("The ball stayed on the table")
        opts.append("The ball was not pocketed")

    # wall-hit statements
    opts.append(f"The ball hit {hits} wall" + ("s" if hits != 1 else ""))
    if hits >= 1:
        opts.append("The ball bounced off a wall")
    if hits >= 3:
        opts.append("The ball hit 3 different walls")
    if hits == 0:
        opts.append("The ball hit 0 walls")

    # Add wall hit sequence options
    if wall_hits:
        opts.append(f"The first wall hit was {wall_hits[0]}")
        if len(wall_hits) > 1:
            opts.append(f"The second wall hit was {wall_hits[1]}")
        if len(wall_hits) > 2:
            opts.append(f"The third wall hit was {wall_hits[2]}")
    return list(dict.fromkeys(opts))  # dedupe, preserve order

def sample_multilabel_options(true_opts: List[str], pool: List[str], total=6):
    """
    Pick t = 1..min(3, len(true_opts)) correct options randomly from true_opts,
    then fill to `total` with distractors from pool not overlapping chosen corrects.
    Only sample distractors that are not logically inconsistent with the true options.
    """
    if not true_opts:
        true_opts = ["The ball stayed on the table"]

    num_correct = random.randint(1, min(3, len(true_opts)))
    chosen_correct = random.sample(true_opts, num_correct)

    # Filter distractors to avoid logical inconsistency
    def is_consistent(opt, correct_opts):
        # Don't allow "The ball was not pocketed" if a correct option says it was pocketed, and vice versa
        if ("The ball was pocketed" in correct_opts and opt == "The ball was not pocketed") or \
           ("The ball was not pocketed" in correct_opts and opt == "The ball was pocketed"):
            return False
        # Don't allow "The ball hit 0 walls" if a correct option says it hit walls, and vice versa
        if any(o.startswith("The ball hit 0 wall") for o in correct_opts) and "wall hit" in opt and not opt.startswith("The ball hit 0 wall"):
            return False
        if any(o.startswith("The ball hit") and "0 wall" not in o for o in correct_opts) and opt.startswith("The ball hit 0 wall"):
            return False
        # Don't allow "The ball stayed on the table" if a correct option says it was pocketed
        if ("The ball was pocketed" in correct_opts and opt == "The ball stayed on the table"):
            return False
        return True

    distractor_candidates = [o for o in pool if o not in chosen_correct and is_consistent(o, chosen_correct)]
    distractors = random.sample(distractor_candidates, k=max(0, total - num_correct))
    all_opts = chosen_correct + distractors
    random.shuffle(all_opts)
    ground_indices = [i for i, opt in enumerate(all_opts) if opt in chosen_correct]
    return all_opts, ground_indices

# ---------------------------
# Counterfactual neighbor selection (using combined index keyed by (pos,vel))
# ---------------------------
def find_velocity_cfs(pos: Tuple[float,float], vel: Tuple[float,float], pos_to_ids, id2entry, n=3):
    """
    Find up to n random shots with the same position but different velocity.
    """
    candidates = pos_to_ids.get(tuple(pos), [])
    candidates = [i for i in candidates if tuple(id2entry[i]['initial_state']['velocity']) != tuple(vel)]
    if not candidates:
        return []
    return random.sample(candidates, min(n, len(candidates)))

def find_position_cfs(pos: Tuple[float,float], vel: Tuple[float,float], vel_to_ids, id2entry, n=3):
    """
    Find up to n random shots with the same velocity but different position.
    """
    candidates = vel_to_ids.get(tuple(vel), [])
    candidates = [i for i in candidates if tuple(id2entry[i]['initial_state']['position']) != tuple(pos)]
    if not candidates:
        return []
    return random.sample(candidates, min(n, len(candidates)))

def generate_sft_mcq_multilabel(sim_data: List[Dict]):
    """
    Generate dataset with the following schema for each example:
      - video: str
      - question: str
      - options: List[str]  # length NUM_OPTIONS
      - ground_truth: List[int]  # list of 0-based indices into options
      - metadata: {...}
    """
    id2entry, idx_pos_vel, pos_to_ids, vel_to_ids = make_index(sim_data)
    out_dataset = []

    for sim_id, entry in id2entry.items():
        video = entry["video"]
        pos = tuple(entry["initial_state"]["position"])
        vel = tuple(entry["initial_state"]["velocity"])
        outcomes = entry["outcomes"]

        # --- DESCRIPTIVE question (full video) ---
        true_opts = outcome_options_from_outcome(outcomes)
        options_list, ground_indices = sample_multilabel_options(true_opts, OPTION_POOL, total=NUM_OPTIONS)
        question_text = "What happened in this video?"
        out_dataset.append({
            "video": video,
            "question": question_text,
            "options": options_list,
            "ground_truth": ground_indices,
            "metadata": {"question_type": "descriptive", "sim_id": sim_id}
        })

        # --- PREDICTIVE question (first-half video) ---
        options_list, ground_indices = sample_multilabel_options(true_opts, OPTION_POOL, total=NUM_OPTIONS)
        question_text = "Based on the first half of the video, what will happen by the end?"
        out_dataset.append({
            "video": video.replace(".mp4","_firsthalf.mp4"),
            "question": question_text,
            "options": options_list,
            "ground_truth": ground_indices,
            "metadata": {"question_type": "predictive", "sim_id": sim_id}
        })

        # --- COUNTERFACTUALS: up to 3 velocity and 3 position neighbors ---
        vel_cf_ids = find_velocity_cfs(pos, vel, pos_to_ids, id2entry, n=3)
        for vel_cf_id in vel_cf_ids:
            cf_entry = id2entry[vel_cf_id]
            cf_out = cf_entry["outcomes"]
            true_opts_cf = outcome_options_from_outcome(cf_out)
            options_list, ground_indices = sample_multilabel_options(true_opts_cf, OPTION_POOL, total=NUM_OPTIONS)
            question_text = f"If the initial velocity were changed from {vel} to {tuple(cf_entry['initial_state']['velocity'])} (assume all other variables are unchanged), what would happen?"
            out_dataset.append({
                "video": video,
                "question": question_text,
                "options": options_list,
                "ground_truth": ground_indices,
                "metadata": {
                    "question_type": "counterfactual_velocity",
                    "sim_id": sim_id,
                    "counterfactual_sim_id": vel_cf_id,
                    "counterfactual_video": cf_entry["video"],
                    "counterfactual_initial_state": cf_entry["initial_state"]
                }
            })

        pos_cf_ids = find_position_cfs(pos, vel, vel_to_ids, id2entry, n=3)
        for pos_cf_id in pos_cf_ids:
            cf_entry = id2entry[pos_cf_id]
            cf_out = cf_entry["outcomes"]
            true_opts_cf = outcome_options_from_outcome(cf_out)
            options_list, ground_indices = sample_multilabel_options(true_opts_cf, OPTION_POOL, total=NUM_OPTIONS)
            question_text = f"If the ball had started at position {tuple(cf_entry['initial_state']['position'])} instead of {pos} (assume all other variables are unchanged), what would happen?"
            out_dataset.append({
                "video": video,
                "question": question_text,
                "options": options_list,
                "ground_truth": ground_indices,
                "metadata": {
                    "question_type": "counterfactual_position",
                    "sim_id": sim_id,
                    "counterfactual_sim_id": pos_cf_id,
                    "counterfactual_video": cf_entry["video"],
                    "counterfactual_initial_state": cf_entry["initial_state"]
                }
            })

    return out_dataset

# Read all JSON files from ./data/output/
sim_data = []
# i= 0
# limit=100
for fname in glob.glob(os.path.join("data", "output", "shot_*", "*.json")):
    # if i >= limit:
    #     break
    with open(fname, "r") as f:
        sim_data.append(json.load(f))
    # i += 1

dataset = generate_sft_mcq_multilabel(sim_data)
# Write to jsonl
with open("mcq_multilabel.jsonl", "w") as f:
    for ex in dataset:
        f.write(json.dumps(ex) + "\n")
print("Wrote", len(dataset), "examples")