from __future__ import annotations

import re
from typing import Dict, List

import pooltool as pt
from pooltool.events.datatypes import AgentType, EventType

from . import config


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


def summarize_system(system: pt.System) -> dict[str, dict[str, object]]:
    cushion_index = _build_cushion_index(system.table)
    pocket_index = _build_pocket_index(system.table)
    pocket_color_lookup = {
        idx: config.POCKET_COLOR_MAP.get(pid, pid) for pid, idx in pocket_index.items()
    }

    wall_hits: dict[str, list[int]] = {ball_id: [] for ball_id in system.balls}
    pocket_results: dict[str, int | None] = {ball_id: None for ball_id in system.balls}

    for event in system.events:
        ball_ids = [agent.id for agent in event.agents if agent.agent_type == AgentType.BALL]
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
            for ball_id in ball_ids:
                wall_hits[ball_id].append(cushion_id)

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

    summary: dict[str, dict[str, object]] = {}
    for ball_id, ball in system.balls.items():
        history = ball.history
        state = history[0] if not history.empty else ball.state
        pos = tuple(float(coord) for coord in state.rvw[0])
        vel = tuple(float(coord) for coord in state.rvw[1])
        hits = wall_hits[ball_id]
        first_hit = hits[0] if hits else None
        pocket_idx = pocket_results[ball_id]
        summary[ball_id] = {
            "initial_position": pos,
            "initial_velocity": vel,
            "outcomes": {
                "num_wall_hits": len(hits),
                "pocketed": pocket_idx is not None,
                "pocket_id": pocket_color_lookup.get(pocket_idx) if pocket_idx else None,
                "first_wall_hit": config.CUSHION_COLOR_LOOKUP.get(first_hit) if first_hit else None,
            },
        }
    return summary
