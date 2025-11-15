"""Debug script for visualizing a multi-ball collision."""

from __future__ import annotations

import json
import math
import shutil

import pooltool as pt
from pooltool.objects.ball.sets import BallSet
from shot_utils import config
from shot_utils.rendering import encode_video, render_frames
from shot_utils.simulation import simulate_shot
from shot_utils.summary import summarize_system


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


def main() -> None:
    config.BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

    reference_table = pt.Table.default()
    cue_speed = 2.0
    cue_phi = 72.0  # shoot slightly off-axis to avoid degenerate configurations
    system, cue_start = _build_three_ball_collision_system(
        reference_table, cue_speed, cue_phi)

    simulate_shot(system, config.FPS)

    shot_dir = config.BASE_OUTPUT / "debug" / "multi_ball_collision"
    shot_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "shot_id": "multi_ball_collision",
        "cue_start": {"x": cue_start[0], "y": cue_start[1]},
        "velocity": cue_speed,
        "phi": cue_phi,
        "fps": config.FPS,
    }

    frames_dir = render_frames(system, shot_dir, config.FPS)
    frame_count = len(list(frames_dir.glob(f"{config.FRAME_PREFIX}_*.png")))
    metadata["total_frames"] = frame_count

    summary = summarize_system(system, metadata=metadata)
    summary_path = shot_dir / "summary_multi_ball_collision.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    video_path = shot_dir / "video.mp4"
    encode_video(frames_dir, config.FPS, video_path)
    try:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
    except Exception:
        pass

    print(f"Multi-ball collision debug shot stored in {shot_dir}")


if __name__ == "__main__":
    main()
