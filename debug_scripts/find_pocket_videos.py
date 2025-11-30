#!/usr/bin/env python3
"""Scan the output dataset and list videos where any ball ends up pocketed."""

from __future__ import annotations

import json
from pathlib import Path

from shot_utils import config


def summary_paths(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted(base.glob("shot_*/summary_*.json"))


def video_with_pocket(summary_path: Path) -> tuple[str, bool]:
    with open(summary_path, "r", encoding="utf-8") as fp:
        summary = json.load(fp)
    shot_id = summary.get("metadata", {}).get("shot_id", summary_path.parent.name)
    for ball_data in summary.get("balls", {}).values():
        outcomes = ball_data.get("outcomes", {})
        if outcomes.get("pocket"):
            return shot_id, True
    return shot_id, False


def main() -> None:
    base = config.BASE_OUTPUT
    candidates = summary_paths(base)
    if not candidates:
        print(f"No summaries found in {base}")
        return

    hits: list[str] = []
    for path in candidates:
        shot_id, pocketed = video_with_pocket(path)
        if pocketed:
            hits.append(str(path.parent / f"video_{shot_id}.mp4"))

    if not hits:
        print("No videos end with a pocketed ball.")
    else:
        print("Videos with at least one pocketed ball:")
        for video in hits:
            print(video)


if __name__ == "__main__":
    main()
