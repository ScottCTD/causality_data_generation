from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple


def extract_cue_wall_hits(sim_entry: Dict) -> List[Dict]:
    """
    Extract ordered wall-hit events for the cue ball from a summary.json-style
    simulation entry produced by pool_simulate/shot_utils/summary.py.
    """
    events = sim_entry.get("events", [])
    cushion_map = sim_entry.get("cushion", {})
    hits: List[Dict] = []

    if not isinstance(events, list) or not isinstance(cushion_map, dict):
        return hits

    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("ball_id") != "cue":
            continue
        if ev.get("type") not in ("linear_cushion", "circular_cushion"):
            continue

        cushion_id = ev.get("cushion_id")
        cushion_id_str = str(cushion_id) if cushion_id is not None else None
        wall_name = cushion_map.get(cushion_id_str, "unknown")

        hit: Dict = {
            "type": "wall",
            "name": wall_name,
            "frame": ev.get("frame", 0),
        }
        try:
            if cushion_id is not None:
                hit["index"] = int(cushion_id)
        except (TypeError, ValueError):
            pass

        hits.append(hit)

    hits.sort(key=lambda h: h.get("frame", 0))
    return hits


def has_hit_index_exceeding_threshold(sim_entry: Dict, max_hit_index: int) -> bool:
    """
    Check if any cue-ball wall hit in the simulation entry has an index > max_hit_index.
    """
    if max_hit_index is None:
        return False

    for hit in extract_cue_wall_hits(sim_entry):
        hit_index = hit.get("index")
        if hit_index is None:
            continue
        try:
            if int(hit_index) > max_hit_index:
                return True
        except (ValueError, TypeError):
            continue

    return False


def make_index(sim_data: List[Dict]):
    """
    Build normalized entries and a combined index keyed by (position_2dp, velocity_2dp).

    The index is currently built only for the cue ball, but the structure is chosen
    to be extensible to multi-ball scenarios in the future.

    Returns: (id_to_entry, index_by_pos_vel, pos_to_ids, vel_to_ids)
    """
    id_to_entry: Dict[int, Dict] = {}
    index_by_pos_vel: Dict[Tuple[Tuple[float, float, float], Tuple[float, float, float]], int] = {}
    pos_to_ids: Dict[Tuple[float, float, float], List[int]] = defaultdict(list)
    vel_to_ids: Dict[Tuple[float, float, float], List[int]] = defaultdict(list)

    for i, raw in enumerate(sim_data):
        norm: Dict = {}
        video = (
            raw.get("video")
            or (raw.get("metadata") or {}).get("shot_id")
            or f"shot_{i}"
        )

        balls = raw.get("balls", {})
        cue = balls.get("cue") if isinstance(balls, dict) else None

        if cue is None:
            # Fallbacks for older formats; only cue-ball shots are expected.
            pos_raw = raw.get("position", [0.0, 0.0, 0.0])
            vel_raw = raw.get("velocity", [0.0, 0.0, 0.0])
            outcomes_raw = raw.get("outcomes", {})
        else:
            pos_raw = cue.get("initial_position", [0.0, 0.0, 0.0])
            vel_raw = cue.get("initial_velocity", [0.0, 0.0, 0.0])
            outcomes_raw = cue.get("outcomes", {})

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

        # Enrich outcomes using summary.json schema (cue-ball only).
        wall_hits_detail = extract_cue_wall_hits(raw)
        wall_hits = [
            h.get("name")
            for h in wall_hits_detail
            if isinstance(h, dict) and h.get("type") == "wall" and h.get("name")
        ]

        pocket_val = None
        pocket_color = None
        num_wall_hits = len(wall_hits)

        if isinstance(outcomes_raw, dict):
            pocket_val = outcomes_raw.get("pocket")
            if isinstance(pocket_val, dict):
                pocket_color = pocket_val.get("color")
            elif isinstance(pocket_val, str):
                pocket_color = pocket_val
            elif pocket_val is not None:
                pocket_color = str(pocket_val)

            if num_wall_hits == 0:
                stored_hits = outcomes_raw.get(
                    "wall_hits", outcomes_raw.get("num_wall_hits", None)
                )
                if stored_hits is not None:
                    try:
                        num_wall_hits = int(stored_hits)
                    except (TypeError, ValueError):
                        num_wall_hits = 0

        pocketed = pocket_val is not None
        which_pocket = pocket_val if pocketed else None

        meta = raw.get("metadata") or {}
        total_frames = meta.get("total_frames", 0)

        norm["video"] = video
        norm["initial_state"] = {"position": pos2, "velocity": vel2}
        norm["outcomes"] = {
            "num_wall_hits": int(num_wall_hits) if num_wall_hits is not None else 0,
            "wall_hits": wall_hits,
            "pocketed": bool(pocketed),
            "which_pocket": which_pocket,
            "pocket_color": pocket_color,
        }
        norm["hits_detail"] = wall_hits_detail
        norm["total_frames"] = total_frames

        id_to_entry[i] = norm

        key = (tuple(pos2), tuple(vel2))
        if key not in index_by_pos_vel:
            index_by_pos_vel[key] = i
        pos_to_ids[tuple(pos2)].append(i)
        vel_to_ids[tuple(vel2)].append(i)

    return id_to_entry, index_by_pos_vel, pos_to_ids, vel_to_ids


def find_velocity_cfs(
    pos: Tuple[float, float, float],
    vel: Tuple[float, float, float],
    pos_to_ids,
    id2entry,
    n: int = 3,
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
    pos: Tuple[float, float, float],
    vel: Tuple[float, float, float],
    vel_to_ids,
    id2entry,
    n: int = 3,
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


def coord_to_str(coord, prefix: str = "") -> str:
    """
    Nicely formatted coordinate string, used in question text for counterfactuals.
    """
    x = coord[0] if abs(coord[0]) >= 0.005 else 0.0
    y = coord[1] if abs(coord[1]) >= 0.005 else 0.0
    return f"({prefix}x={x:.2f}, {prefix}y={y:.2f})"


