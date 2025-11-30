from __future__ import annotations

import re
from typing import Dict, Iterable, List

import pooltool as pt
from pooltool.events.datatypes import AgentType, EventType

from . import config

BALL_COLOR_MAP = {
    "cue": "white_cue",
    "1": "yellow_solid",
    "2": "blue_solid",
    "3": "red_solid",
    "4": "pink_solid",
    "5": "orange_solid",
    "6": "green_solid",
    "7": "brown_solid",
    "8": "black_solid",
    "9": "yellow_stripe",
    "10": "blue_stripe",
    "11": "red_stripe",
    "12": "pink_stripe",
    "13": "orange_stripe",
    "14": "green_stripe",
    "15": "brown_stripe",
}


def _natural_key(value: str) -> List[int | str]:
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part for part in parts if part]


def _build_cushion_index(table: pt.Table) -> Dict[str, int]:
    ordered = sorted(table.cushion_segments.linear.keys(), key=_natural_key)
    ordered += sorted(table.cushion_segments.circular.keys(), key=_natural_key)
    return {seg_id: idx for idx, seg_id in enumerate(ordered, start=1)}


def _build_pocket_index(table: pt.Table) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    ordered = [pid for pid in config.POCKET_ORDER if pid in table.pockets]
    ordered += sorted(
        [pid for pid in table.pockets if pid not in config.POCKET_ORDER],
        key=_natural_key,
    )
    for idx, pocket_id in enumerate(ordered, start=1):
        mapping[pocket_id] = idx
    return mapping


def _first_nonzero_velocity(ball: pt.Ball, eps: float = 1e-6) -> tuple[float, float, float]:
    history = ball.history
    states: Iterable = history if not history.empty else [ball.state]

    first_state = None
    for state in states:
        if first_state is None:
            first_state = state
        vel = state.rvw[1]
        if any(abs(comp) > eps for comp in vel):
            return tuple(float(comp) for comp in vel)

    assert first_state is not None
    return tuple(float(comp) for comp in first_state.rvw[1])


def summarize_system(
    system: pt.System, metadata: dict[str, object] | None = None
) -> dict[str, dict[str, object]]:
    meta = metadata or {}
    fps = float(meta.get("fps", config.FPS))
    duration = float(system.t or 0.0)
    total_frames_meta = meta.get("total_frames")
    # For error tolerance, May not really be called anytime
    if total_frames_meta is not None:
        try:
            total_frames = max(1, int(total_frames_meta))
        except (TypeError, ValueError):
            total_frames = int(max(1, round(fps * duration))) if fps > 0 else 1
    else:
        total_frames = int(max(1, round(fps * duration))) if fps > 0 else 1

    cushion_index = _build_cushion_index(system.table)
    pocket_index = _build_pocket_index(system.table)
    pocket_color_lookup = {
        idx: config.POCKET_COLOR_MAP.get(pid, pid) for pid, idx in pocket_index.items()
    }

    wall_hits: dict[str, list[int]] = {ball_id: [] for ball_id in system.balls}
    ball_hits: dict[str, list[str]] = {ball_id: [] for ball_id in system.balls}
    hit_sequence: dict[str, list[dict[str, object]]] = {
        ball_id: [] for ball_id in system.balls}
    pocket_results: dict[str, int | None] = {
        ball_id: None for ball_id in system.balls}
    
    # Collect all events for the events array
    events_list: list[dict[str, object]] = []

    for event in system.events:
        event_time = getattr(event, "time", 0.0) or 0.0
        if fps > 0:
            clamped_time = max(0.0, min(duration, float(event_time)))
            frame_idx = int(round(clamped_time * fps))
            frame_idx = min(frame_idx, total_frames - 1)
        else:
            frame_idx = 0
        ball_ids = [
            agent.id for agent in event.agents if agent.agent_type == AgentType.BALL]
        if not ball_ids:
            continue

        if event.event_type in (
            EventType.BALL_LINEAR_CUSHION,
            EventType.BALL_CIRCULAR_CUSHION,
        ):
            cushion_agent = next(
                (
                    agent
                    for agent in event.agents
                    if agent.agent_type
                    in (AgentType.LINEAR_CUSHION_SEGMENT, AgentType.CIRCULAR_CUSHION_SEGMENT)
                ),
                None,
            )
            if cushion_agent is None:
                continue
            cushion_id = cushion_index.get(cushion_agent.id)
            if cushion_id is None:
                continue
            
            event_type_str = "linear_cushion" if event.event_type == EventType.BALL_LINEAR_CUSHION else "circular_cushion"
            
            for ball_id in ball_ids:
                wall_hits[ball_id].append(cushion_id)
                hit_sequence[ball_id].append(
                    {
                        "type": "wall",
                        "index": cushion_id,
                        "name": config.CUSHION_COLOR_LOOKUP.get(cushion_id, "unknown"),
                        "frame": frame_idx,
                    }
                )
                # Add to events list
                events_list.append({
                    "type": event_type_str,
                    "ball_id": ball_id,
                    "cushion_id": str(cushion_id),
                    "frame": frame_idx,
                })

        elif event.event_type == EventType.BALL_POCKET:
            pocket_agent = next(
                (agent for agent in event.agents if agent.agent_type == AgentType.POCKET),
                None,
            )
            if pocket_agent is None:
                continue
            pocket_id = pocket_index.get(pocket_agent.id)
            for ball_id in ball_ids:
                if pocket_results[ball_id] is None:
                    pocket_results[ball_id] = pocket_id
                # Add to events list
                events_list.append({
                    "type": "pocket",
                    "pocket_id": pocket_id,
                    "ball_id": ball_id,
                    "frame": frame_idx,
                })

        elif event.event_type == EventType.BALL_BALL:
            other_balls = [
                agent.id for agent in event.agents if agent.agent_type == AgentType.BALL]
            for hitter in other_balls:
                for target in other_balls:
                    if target == hitter:
                        continue
                    ball_hits[hitter].append(target)
                    hit_sequence[hitter].append(
                        {"type": "ball", "name": target, "frame": frame_idx})
                    # Add to events list (only once per pair)
                    if hitter < target:  # Avoid duplicates by ordering
                        events_list.append({
                            "type": "ball",
                            "hitter_id": hitter,
                            "target_id": target,
                            "frame": frame_idx,
                        })

    summary: dict[str, dict[str, object]] = {}
    for ball_id, ball in system.balls.items():
        history = ball.history
        state = history[0] if not history.empty else ball.state
        pos = tuple(float(coord) for coord in state.rvw[0])
        vel = _first_nonzero_velocity(ball)
        hits = wall_hits[ball_id]
        pocket_idx = pocket_results[ball_id]
        summary[ball_id] = {
            "initial_position": pos,
            "initial_velocity": vel,
            "color": BALL_COLOR_MAP.get(ball_id, "unknown"),
            "outcomes": {
                "pocket": pocket_color_lookup.get(pocket_idx) if pocket_idx else None,
                "wall_hits": len(hits),
                "ball_hits": len(ball_hits[ball_id]),
            },
        }
    # Ensure total_frames is in metadata
    meta["total_frames"] = total_frames
    
    # Build cushion mapping from CUSHION_COLOR_LOOKUP (convert keys to strings)
    cushion_mapping = {str(k): v for k, v in config.CUSHION_COLOR_LOOKUP.items()}
    
    return {
        "metadata": meta,
        "balls": summary,
        "events": events_list,
        "cushion": cushion_mapping,
    }
