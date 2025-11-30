"""Debug script for visualizing a few multi-ball collisions."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

from panda3d.core import loadPrcFileData
from tqdm import tqdm

# Disable audio completely (no more ALSA/OpenAL spam)
loadPrcFileData("", "audio-library-name null\n")

# Force offscreen / headless rendering (no window, avoids :0.0 errors if supported)
loadPrcFileData("", "window-type offscreen\n")

# Reduce log verbosity for these subsystems
loadPrcFileData("", "notify-level audio error\n")
loadPrcFileData("", "notify-level-display error\n")

import pooltool as pt
from pooltool.objects.ball.sets import BallSet
from shot_utils import config
from shot_utils.rendering import encode_video, render_frames
from shot_utils.simulation import BallState, build_system, simulate_shot
from shot_utils.summary import summarize_system

SHOT_PRESETS: tuple[tuple[str, float, float], ...] = (
    ("soft_cut", 1.6, 60.0),
    ("standard_chain", 2.0, 72.0),
    ("fast_cut", 2.4, 82.0),
)

# Test presets with initial velocities for object balls
# Format: (shot_id, cue_speed, cue_phi, ball1_speed, ball1_phi, ball2_speed, ball2_phi)
SHOT_PRESETS_WITH_VELOCITIES: tuple[tuple[str, float, float, float, float, float, float], ...] = (
    ("ball1_moving_toward_cue", 1.5, 90.0, 0.8, 270.0, 0.0, 0.0),  # Ball 1 moving toward cue ball
    ("ball1_moving_away", 2.0, 90.0, 1.2, 90.0, 0.0, 0.0),  # Ball 1 moving in same direction
    ("both_balls_moving", 1.8, 45.0, 1.0, 225.0, 0.9, 135.0),  # Both balls moving at angles
    ("ball2_moving_perpendicular", 2.2, 0.0, 0.0, 0.0, 1.5, 90.0),  # Ball 2 moving perpendicular
    ("high_speed_collision", 3.0, 90.0, 2.0, 270.0, 0.0, 0.0),  # High speed head-on collision
)


def _build_three_ball_collision_system(
    table: pt.Table,
    cue_speed: float,
    cue_phi: float,
    ball1_speed: float = 0.0,
    ball1_phi: float = 0.0,
    ball2_speed: float = 0.0,
    ball2_phi: float = 0.0,
) -> tuple[pt.System, dict[str, BallState], tuple[float, float]]:
    """Place a cue ball and two object balls along the cue direction for chained collisions.
    
    Args:
        table: The pool table
        cue_speed: Cue stick speed (V0) in m/s
        cue_phi: Cue stick angle in degrees
        ball1_speed: Initial speed for ball "1" in m/s (default: 0.0)
        ball1_phi: Initial angle for ball "1" in degrees (default: 0.0)
        ball2_speed: Initial speed for ball "2" in m/s (default: 0.0)
        ball2_phi: Initial angle for ball "2" in degrees (default: 0.0)
    
    Returns:
        Tuple of (system, ball_states, cue_start)
    """
    radius = pt.BallParams.default().R
    spacing = 7 * radius
    cue_start = (table.w * 0.3, table.l * 0.25)

    phi_rad = math.radians(cue_phi)
    direction = (math.cos(phi_rad), math.sin(phi_rad))

    # Create ball states dictionary
    ball_states = {
        "cue": BallState(x=cue_start[0], y=cue_start[1], speed=cue_speed, phi=cue_phi),
    }
    
    # Ball 1
    offset_x = direction[0] * spacing
    offset_y = direction[1] * spacing
    ball_states["1"] = BallState(
        x=cue_start[0] + offset_x,
        y=cue_start[1] + offset_y,
        speed=ball1_speed,
        phi=ball1_phi,
    )
    
    # Ball 2
    offset_x = direction[0] * spacing * 2
    offset_y = direction[1] * spacing * 2
    ball_states["2"] = BallState(
        x=cue_start[0] + offset_x,
        y=cue_start[1] + offset_y,
        speed=ball2_speed,
        phi=ball2_phi,
    )

    system = build_system(ball_states)
    system.set_ballset(BallSet("pooltool_pocket"))
    return system, ball_states, cue_start


def _render_shot(
    output_dir: Path,
    shot_id: str,
    cue_speed: float,
    cue_phi: float,
    fps: int,
    keep_frames: bool,
    ball1_speed: float = 0.0,
    ball1_phi: float = 0.0,
    ball2_speed: float = 0.0,
    ball2_phi: float = 0.0,
) -> None:
    """Simulate, render, and encode a single shot."""
    table = pt.Table.default()
    system, ball_states, cue_start = _build_three_ball_collision_system(
        table, cue_speed, cue_phi, ball1_speed, ball1_phi, ball2_speed, ball2_phi
    )
    simulate_shot(system, fps)

    shot_dir = output_dir / shot_id
    shot_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = render_frames(system, shot_dir, fps)
    frame_count = len(list(frames_dir.glob(f"{config.FRAME_PREFIX}_*.png")))
    
    # Serialize ball_states to dict format for JSON (matching main.py format)
    initial_ball_states = {
        ball_id: {
            "x": state.x,
            "y": state.y,
            "speed": state.speed,
            "phi": state.phi,
        }
        for ball_id, state in ball_states.items()
    }
    metadata = {
        "shot_id": shot_id,
        "fps": fps,
        "total_frames": frame_count,
        "initial_ball_states": initial_ball_states,
    }

    summary = summarize_system(system, metadata=metadata)
    summary_path = shot_dir / "summary.json"
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
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=None,
        help="Number of worker processes (defaults to CPU count)",
    )
    return parser.parse_args()


def _run_shot_from_tuple(args: tuple[Path, str, float, float, int, bool, float, float, float, float]) -> None:
    """Worker function that unpacks tuple arguments for multiprocessing."""
    (
        output_dir,
        shot_id,
        cue_speed,
        cue_phi,
        fps,
        keep_frames,
        ball1_speed,
        ball1_phi,
        ball2_speed,
        ball2_phi,
    ) = args
    _render_shot(
        output_dir=output_dir,
        shot_id=shot_id,
        cue_speed=cue_speed,
        cue_phi=cue_phi,
        fps=fps,
        keep_frames=keep_frames,
        ball1_speed=ball1_speed,
        ball1_phi=ball1_phi,
        ball2_speed=ball2_speed,
        ball2_phi=ball2_phi,
    )


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks = []
    
    # Add standard presets (stationary object balls)
    for shot_id, cue_speed, cue_phi in SHOT_PRESETS:
        tasks.append((
            args.output_dir,
            shot_id,
            cue_speed,
            cue_phi,
            config.FPS,
            bool(args.keep_frames),
            0.0,  # ball1_speed
            0.0,  # ball1_phi
            0.0,  # ball2_speed
            0.0,  # ball2_phi
        ))

    # Add presets with initial velocities for object balls
    for shot_id, cue_speed, cue_phi, ball1_speed, ball1_phi, ball2_speed, ball2_phi in SHOT_PRESETS_WITH_VELOCITIES:
        tasks.append((
            args.output_dir,
            shot_id,
            cue_speed,
            cue_phi,
            config.FPS,
            bool(args.keep_frames),
            ball1_speed,
            ball1_phi,
            ball2_speed,
            ball2_phi,
        ))

    # Process tasks in parallel
    proc_count = args.processes or cpu_count()
    worker = partial(_run_shot_from_tuple)
    with Pool(processes=proc_count) as pool:
        list(
            tqdm(
                pool.imap(worker, tasks),
                total=len(tasks),
                desc="Generating shots",
                unit="shot",
                file=sys.stdout,
            )
        )

    total_shots = len(SHOT_PRESETS) + len(SHOT_PRESETS_WITH_VELOCITIES)
    print(f"Generated {total_shots} videos in {args.output_dir} using {proc_count} processes")


if __name__ == "__main__":
    main()
