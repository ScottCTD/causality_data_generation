"""Debug script for visualizing a few multi-ball collisions."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import pooltool as pt
from pooltool.objects.ball.sets import BallSet
from shot_utils import config
from shot_utils.rendering import encode_video, render_frames
from shot_utils.simulation import simulate_shot
from shot_utils.summary import summarize_system

SHOT_PRESETS: tuple[tuple[str, float, float], ...] = (
    ("soft_cut", 1.6, 60.0),
    ("standard_chain", 2.0, 72.0),
    ("fast_cut", 2.4, 82.0),
)


def _build_three_ball_collision_system(table: pt.Table, cue_speed: float, cue_phi: float) -> tuple[pt.System, tuple[float, float]]:
    """Place a cue ball and two object balls along the cue direction for chained collisions."""
    radius = pt.BallParams.default().R
    spacing = 7 * radius
    cue_start = (table.w * 0.3, table.l * 0.25)

    phi_rad = math.radians(cue_phi)
    direction = (math.cos(phi_rad), math.sin(phi_rad))

    balls = {"cue": pt.Ball.create("cue", xy=cue_start)}
    for idx, ball_id in enumerate(("1", "2"), start=1):
        offset_x = direction[0] * spacing * idx
        offset_y = direction[1] * spacing * idx
        balls[ball_id] = pt.Ball.create(
            ball_id,
            xy=(
                cue_start[0] + offset_x,
                cue_start[1] + offset_y,
            ),
        )

    cue = pt.Cue.default()
    system = pt.System(table=table, balls=balls, cue=cue)
    system.set_ballset(BallSet("pooltool_pocket"))
    system.cue.set_state(V0=cue_speed, phi=cue_phi)
    return system, cue_start


def _render_shot(output_dir: Path, shot_id: str, cue_speed: float, cue_phi: float, fps: int, keep_frames: bool) -> None:
    """Simulate, render, and encode a single shot."""
    table = pt.Table.default()
    system, cue_start = _build_three_ball_collision_system(table, cue_speed, cue_phi)
    simulate_shot(system, fps)

    shot_dir = output_dir / shot_id
    shot_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "shot_id": shot_id,
        "cue_start": {"x": cue_start[0], "y": cue_start[1]},
        "velocity": cue_speed,
        "phi": cue_phi,
        "fps": fps,
    }

    frames_dir = render_frames(system, shot_dir, fps)
    frame_count = len(list(frames_dir.glob(f"{config.FRAME_PREFIX}_*.png")))
    metadata["total_frames"] = frame_count

    summary = summarize_system(system, metadata=metadata)
    summary_path = shot_dir / f"summary_{shot_id}.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    video_path = shot_dir / f"{shot_id}.mp4"
    encode_video(frames_dir, fps, video_path)
    if not keep_frames:
        try:
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
        except Exception:
            pass

    print(f"Stored shot '{shot_id}' in {shot_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a few debug multi-ball collision videos.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.BASE_OUTPUT / "debug" / "test0",
        help="Directory to store rendered assets.",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep the intermediate PNG frames instead of deleting them after encoding.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for shot_id, cue_speed, cue_phi in SHOT_PRESETS:
        _render_shot(
            output_dir=args.output_dir,
            shot_id=shot_id,
            cue_speed=cue_speed,
            cue_phi=cue_phi,
            fps=config.FPS,
            keep_frames=bool(args.keep_frames),
        )

    print(f"Generated {len(SHOT_PRESETS)} videos in {args.output_dir}")


if __name__ == "__main__":
    main()
