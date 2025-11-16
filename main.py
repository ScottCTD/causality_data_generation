"""Dataset generator for pooltool shots."""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
import sys

from tqdm import tqdm

from panda3d.core import loadPrcFileData

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
from shot_utils.simulation import (build_system_one_ball_hit_cushion,
                                   extract_trajectories, simulate_shot)
from shot_utils.summary import summarize_system

CAMERA_STATES = [
    "7_foot_offcenter",
    "LongSideView",
    "WidthSideView",
    "7_foot_overhead",
]


def run_shot(
    shot_id: str,
    x: float,
    y: float,
    velocity: float,
    phi: float,
    camera_name: str,
) -> dict[str, object]:
    # shot_id is now already just the shot number (e.g., "shot_01")
    # Each camera+shot combo gets a unique shot number
    outdir = config.BASE_OUTPUT / "shots" / shot_id
    outdir.mkdir(parents=True, exist_ok=True)

    video_path = outdir / f"video.mp4"
    summary_path = outdir / f"summary_{shot_id}.json"

    # Skip if both video and summary already exist
    if video_path.exists() and summary_path.exists():
        # Load existing summary to get metadata
        with open(summary_path, "r", encoding="utf-8") as fp:
            summary = json.load(fp)
        metadata = summary.get("metadata", {})
        
        return { 
            "shot_id": shot_id,
            "cue_start": metadata.get("cue_start", {"x": x, "y": y}),
            "velocity": metadata.get("velocity", velocity),
            "phi": metadata.get("phi", phi),
            "camera": metadata.get("camera_name", camera_name),
            "paths": {
                "directory": str(outdir),
                "summary": str(summary_path),
                "video": str(video_path),
            },
        }

    system = build_system_one_ball_hit_cushion(x, y, velocity, phi)
    simulate_shot(system, config.FPS)

    # df = extract_trajectories(system)
    # df = df[df["t"] <= system.t].copy()
    #
    # trajectory_path = outdir / f"trajectory_{shot_id}.csv"
    # df.to_csv(trajectory_path, index=False)
    #
    # system_path = outdir / f"system_{shot_id}.json"
    # system.save(system_path)

    metadata = {
        "shot_id": shot_id,
        "cue_start": {"x": x, "y": y},
        "velocity": velocity,
        "phi": phi,
        "fps": config.FPS,
    }

    frame_count = render_and_encode_video(
        system=system,
        outdir=outdir,
        fps=config.FPS,
        video_path=video_path,
        camera_name=camera_name,
    )
    metadata["total_frames"] = frame_count
    metadata["camera_name"] = camera_name

    summary = summarize_system(system, metadata=metadata)
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    return {
        "shot_id": shot_id,
        "cue_start": metadata["cue_start"],
        "velocity": velocity,
        "phi": phi,
        "camera": camera_name,
        "paths": {
            "directory": str(outdir),
            # "trajectory": str(trajectory_path),
            # "system": str(system_path),
            "summary": str(summary_path),
            "video": str(video_path),
        },
    }


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


def main(processes: int | None = None, dataset_name: str = "default", num_shots: int | None = None) -> None:
    # Set up dataset-specific output directory
    config.BASE_OUTPUT = Path("outputs") / dataset_name
    config.GLOBAL_INDEX_PATH = config.BASE_OUTPUT / "global_index.json"

    config.BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

    reference_table = pt.Table.default()
    positions = _scaled_positions(reference_table, num=16)
    velocities = _segment_values(0.3, 1.8, num=16)
    phis = _segment_angles(num=16)

    combos = list(itertools.product(positions, velocities, phis))
    tasks = []
    shot_counter = 1
    # Treat each camera * shot combo as a unique shot
    for (x, y), velocity, phi in combos:
        for camera_name in CAMERA_STATES:
            shot_label = f"shot_{shot_counter:02d}"
            tasks.append((shot_label, x, y, velocity, phi, camera_name))
            shot_counter += 1

    # Limit to num_shots if specified (for test runs)
    if num_shots is not None:
        tasks = tasks[:num_shots]

    worker = partial(_run_shot_from_tuple)
    proc_count = processes or cpu_count()
    with Pool(processes=proc_count) as pool:
        results = list(
            tqdm(
                pool.imap(worker, tasks),
                total=len(tasks),
                desc="Generating shots",
                unit="shot",
                file=sys.stdout,
            )
        )

    index_path = config.GLOBAL_INDEX_PATH
    with open(index_path, "w", encoding="utf-8") as fp:
        json.dump({"shots": results}, fp, indent=2)

    print(f"Global index written to {index_path}")
    print(f"Generated {len(results)} shots using {proc_count} processes")


def _run_shot_from_tuple(args: tuple[str, float, float, float, float, str]):
    shot_id, x, y, velocity, phi, camera_name = args
    result = run_shot(shot_id, x, y, velocity, phi, camera_name)
    return result


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
    main(processes=args.processes, dataset_name=args.dataset_name,
         num_shots=args.test_shots)
