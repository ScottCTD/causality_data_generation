"""Dataset generator for pooltool shots."""

from __future__ import annotations

import argparse
import itertools
import json
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
from shot_utils import config
from shot_utils.rendering import render_and_encode_video
from shot_utils.simulation import (
    BallState,
    build_system,
    extract_trajectories,
    simulate_shot,
)
from shot_utils.summary import summarize_system

CAMERA_STATES = [
    "7_foot_offcenter",
    "LongSideView",
    "WidthSideView",
    "7_foot_overhead",
]

# Map camera names to IDs
CAMERA_ID_MAP = {name: f"cam{i:02d}" for i, name in enumerate(CAMERA_STATES)}
CAMERA_NAME_MAP = {cam_id: name for name, cam_id in CAMERA_ID_MAP.items()}


def run_shot(
    base_output: Path,
    shot_id: str,
    ball_states: dict[str, BallState],
    camera_name: str,
) -> None:
    # shot_id is per unique shot (ball_states configuration)
    # Each shot directory contains multiple video files, one per camera
    outdir = base_output / "shots" / shot_id
    outdir.mkdir(parents=True, exist_ok=True)

    camera_id = CAMERA_ID_MAP[camera_name]
    video_path = outdir / f"video_{camera_id}.mp4"
    summary_path = outdir / f"summary.json"

    # Skip if already exists
    if video_path.exists() and summary_path.exists():
        return

    # Build and simulate system (needed for rendering)
    system = build_system(ball_states)
    simulate_shot(system, config.FPS)

    # Render video for this camera
    frame_count = render_and_encode_video(
        system=system,
        outdir=outdir,
        fps=config.FPS,
        video_path=video_path,
        camera_name=camera_name,
    )

    # Create summary if it doesn't exist (first camera for this shot)
    if not summary_path.exists():
        # Serialize ball_states to dict format for JSON
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
            "fps": config.FPS,
            "total_frames": frame_count,
            "initial_ball_states": initial_ball_states,
        }
        summary = summarize_system(system, metadata=metadata)
        with open(summary_path, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)


def _scaled_positions(table: pt.Table, num: int) -> list[tuple[float, float]]:
    num = max(1, num)
    grid = int(num**0.5)
    while grid * grid < num:
        grid += 1
    positions: list[tuple[float, float]] = []
    margin_x = 0.05 * table.w
    margin_y = 0.05 * table.l
    usable_w = table.w - 2 * margin_x
    usable_l = table.l - 2 * margin_y
    for i in range(grid):
        for j in range(grid):
            if len(positions) >= num:
                break
            fx = (i + 0.5) / grid
            fy = (j + 0.5) / grid
            x = margin_x + fx * usable_w
            y = margin_y + fy * usable_l
            positions.append((x, y))
    return positions


def _segment_values(low: float, high: float, num: int) -> list[float]:
    if num <= 1:
        return [low]
    step = (high - low) / (num - 1)
    return [low + i * step for i in range(num)]


def _segment_angles(num: int) -> list[float]:
    if num <= 1:
        return [0.0]
    step = 360.0 / num
    return [(i * step) % 360.0 for i in range(num)]


def main(
    processes: int | None = None,
    dataset_name: str = "default",
    num_shots: int | None = None,
) -> None:
    # Set up dataset-specific output directory
    base_output = Path("outputs") / dataset_name
    base_output.mkdir(parents=True, exist_ok=True)

    reference_table = pt.Table.default()
    positions = _scaled_positions(reference_table, num=16)
    speeds = _segment_values(0.3, 1.8, num=16)
    phis = _segment_angles(num=16)

    combos = list(itertools.product(positions, speeds, phis))
    tasks = []
    shot_counter = 1
    # Each unique shot (ball_states configuration) gets one shot_id
    # Each shot will have multiple videos (one per camera) in the same directory
    for (x, y), speed, phi in combos:
        shot_label = f"shot_{shot_counter:02d}"
        # Create ball_states dict with cue ball
        ball_states = {
            "cue": BallState(x=x, y=y, speed=speed, phi=phi),
        }
        for camera_name in CAMERA_STATES:
            tasks.append((base_output, shot_label, ball_states, camera_name))
        shot_counter += 1

    # Limit to num_shots if specified (for test runs)
    if num_shots is not None:
        tasks = tasks[:num_shots]

    worker = partial(_run_shot_from_tuple)
    proc_count = processes or cpu_count()
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

    print(f"Generated {len(tasks)} shots using {proc_count} processes")


def _run_shot_from_tuple(args: tuple[Path, str, dict[str, BallState], str]) -> None:
    base_output, shot_id, ball_states, camera_name = args
    run_shot(base_output, shot_id, ball_states, camera_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate poolshot dataset")
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=64,
        help="Number of worker processes (defaults to CPU count)",
    )
    parser.add_argument(
        "-d",
        "--dataset-name",
        type=str,
        default="default",
        help="Name of the dataset (outputs will be stored under outputs/dataset_name)",
    )
    parser.add_argument(
        "-k",
        "--test-shots",
        type=int,
        default=None,
        help="Number of shots to generate for test run (defaults to all shots)",
    )
    args = parser.parse_args()
    main(
        processes=args.processes,
        dataset_name=args.dataset_name,
        num_shots=args.test_shots,
    )
