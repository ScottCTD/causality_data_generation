# make_video.py
# Requirements: pooltool==0.5.0, numpy, pandas, ffmpeg (CLI installed)

import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

import pooltool as pt
from pooltool.ani.animate import FrameStepper
from pooltool.ani.camera import camera_states
from pooltool.ani.image import ImageExt, ImageZip, save_images
from pooltool.events.datatypes import AgentType, EventType

# -----------------------
# Config (edit as needed)
# -----------------------
FPS = 10                 # target video FPS
DURATION = 5.0           # seconds
OUTDIR = Path("shot_out")  # output directory
VIDEO_NAME = "shot.mp4"    # output video filename
FRAME_SIZE = (int(1.6 * 720), 720)  # matches pooltool's default aspect
CAMERA_NAME = "7_foot_offcenter"
SUMMARY_NAME = "shot_summary.json"

FRAME_PREFIX = "frame"
FRAME_PATTERN = f"{FRAME_PREFIX}_%06d.png"
POCKET_ORDER = ("lb", "lc", "lt", "rb", "rc", "rt")
POCKET_COLOR_MAP: dict[str, str] = {
    "lb": "red",
    "lc": "orange",
    "lt": "grey",
    "rb": "green",
    "rc": "blue",
    "rt": "purple",
}
CUSHION_COLOR_LOOKUP: dict[int, str] = {
    1: "red-green-wall",
    2: "orange-red-wall",
    3: "orange-red-wall",
    4: "orange-red-wall",
    5: "grey-orange-wall",
    6: "grey-orange-wall",
    7: "grey-orange-wall",
    8: "purple-grey-wall",
    9: "purple-grey-wall",
    10: "purple-grey-wall",
    11: "blue-purple-wall",
    12: "blue-purple-wall",
    13: "blue-purple-wall",
    14: "green-blue-wall",
    15: "green-blue-wall",
    16: "green-blue-wall",
    17: "red-green-wall",
    18: "red-green-wall",
    19: "purple-grey-wall",
    20: "blue-purple-wall",
    21: "blue-purple-wall",
    22: "green-blue-wall",
    23: "green-blue-wall",
    24: "red-green-wall",
    25: "red-green-wall",
    26: "orange-red-wall",
    27: "orange-red-wall",
    28: "grey-orange-wall",
    29: "grey-orange-wall",
    30: "purple-grey-wall",
}


def _natural_key(value: str) -> list[int | str]:
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part for part in parts if part]


def _build_cushion_index(table: pt.Table) -> dict[str, int]:
    indices: dict[str, int] = {}
    ordered_ids = sorted(
        table.cushion_segments.linear.keys(), key=_natural_key)
    ordered_ids += sorted(table.cushion_segments.circular.keys(),
                          key=_natural_key)
    for idx, seg_id in enumerate(ordered_ids, start=1):
        indices[seg_id] = idx
    return indices


def _build_pocket_index(table: pt.Table) -> dict[str, int]:
    mapping: dict[str, int] = {}
    ordered = [pid for pid in POCKET_ORDER if pid in table.pockets]
    ordered += sorted(
        [pid for pid in table.pockets if pid not in POCKET_ORDER], key=_natural_key
    )
    for idx, pocket_id in enumerate(ordered, start=1):
        mapping[pocket_id] = idx
    return mapping


def _state_vectors(ball: pt.Ball) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    history = ball.history
    state = history[0] if not history.empty else ball.state
    pos = tuple(float(coord) for coord in state.rvw[0])
    vel = tuple(float(coord) for coord in state.rvw[1])
    return pos, vel


def summarize_system(system: pt.System) -> dict[str, dict[str, object]]:
    cushion_index = _build_cushion_index(system.table)
    pocket_index = _build_pocket_index(system.table)
    pocket_color_lookup = {
        idx: POCKET_COLOR_MAP.get(pid, pid) for pid, idx in pocket_index.items()
    }

    wall_hits: dict[str, list[int]] = {ball_id: [] for ball_id in system.balls}
    pocket_results: dict[str, int | None] = {
        ball_id: None for ball_id in system.balls}

    for event in system.events:
        ball_ids = [
            agent.id for agent in event.agents if agent.agent_type == AgentType.BALL
        ]
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
        initial_position, initial_velocity = _state_vectors(ball)
        hits = wall_hits[ball_id]
        first_hit = hits[0] if hits else None
        pocket_idx = pocket_results[ball_id]
        pocket_color = (
            pocket_color_lookup.get(
                pocket_idx) if pocket_idx is not None else None
        )
        wall_label = (
            CUSHION_COLOR_LOOKUP.get(
                first_hit) if first_hit is not None else None
        )
        summary[ball_id] = {
            "initial_position": initial_position,
            "initial_velocity": initial_velocity,
            "outcomes": {
                "num_wall_hits": len(hits),
                "pocketed": pocket_idx is not None,
                "pocket_id": pocket_color,
                "first_wall_hit": wall_label,
            },
        }
    return summary


def build_system_one_ball_hit_cushion() -> pt.System:
    """
    Create a system with only the cue ball, positioned and aimed so it will hit a cushion.
    This avoids the single-ball/no-event bug by ensuring at least one rail collision.
    """
    table = pt.Table.default()
    balls = {
        # near left, centered in y
        "cue": pt.Ball.create("cue", xy=(0.6, table.w / 2.0)),
        "1": pt.Ball.create("1", xy=(5.0, 6.0)),
    }
    cue = pt.Cue.default()
    system = pt.System(table=table, balls=balls, cue=cue)

    # Aim roughly to the right so it strikes the right cushion.
    # Angle convention: 0° = +x (right), 90° = +y (up table). Adjust as you like.
    V0 = 2.0          # m/s initial speed
    phi = 0.0         # degrees (straight to +x)
    system.cue.set_state(V0=V0, phi=phi)

    return system


def simulate(system: pt.System, duration: float, fps: int) -> None:
    """
    Run the simulation and densify the continuous history at dt = 1/fps.
    """
    # Pooltool will run until motion ends; to ensure we have 5s worth, we'll densify
    # and later trim/pad to DURATION as needed.
    pt.simulate(system, continuous=True, dt=1.0 / fps, inplace=True)


def extract_trajectories(system: pt.System) -> pd.DataFrame:
    """
    Extract (t, x, y, vx, vy, omega_x, omega_y, omega_z) for each ball at the
    continuous sample times. Returns a tidy DataFrame.
    """
    records = []
    for ball_id, ball in system.balls.items():
        hist = ball.history_cts
        if hist is None or len(hist) == 0:
            # If continuous wasn’t generated, fallback to discrete history
            hist = ball.history
        rvw, s, t = hist.vectorize()  # rvw: (N,3,3): [pos, vel, ang_vel]
        pos = rvw[:, 0, :]            # (x, y, z)
        vel = rvw[:, 1, :]            # (vx, vy, vz)
        omg = rvw[:, 2, :]            # (wx, wy, wz)
        for i in range(len(t)):
            records.append({
                "ball_id": ball_id,
                "t": float(t[i]),
                "x": float(pos[i, 0]),
                "y": float(pos[i, 1]),
                "z": float(pos[i, 2]),
                "vx": float(vel[i, 0]),
                "vy": float(vel[i, 1]),
                "vz": float(vel[i, 2]),
                "wx": float(omg[i, 0]),
                "wy": float(omg[i, 1]),
                "wz": float(omg[i, 2]),
            })
    df = pd.DataFrame.from_records(records)
    # Sort and reset index
    df.sort_values(["t", "ball_id"], inplace=True, kind="stable")
    df.reset_index(drop=True, inplace=True)
    return df


def render_with_pooltool(system: pt.System, outdir: Path, fps: int) -> Path:
    """
    Use pooltool's Panda3D renderer to dump per-frame PNGs for the shot.
    """
    frames_dir = outdir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)

    interface = FrameStepper()
    exporter = ImageZip(
        path=frames_dir,
        ext=ImageExt.PNG,
        prefix=FRAME_PREFIX,
        compress=False,
    )

    try:
        save_images(
            exporter=exporter,
            system=system,
            interface=interface,
            size=FRAME_SIZE,
            fps=fps,
            camera_state=camera_states[CAMERA_NAME],
            gray=False,
            show_hud=False,
        )
    finally:
        # Tear down the offscreen window so repeat runs don't accumulate resources.
        interface.destroy()

    return frames_dir


def encode_video(frames_dir: Path, fps: int, video_path: Path):
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / FRAME_PATTERN),
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # ensure even dimensions
        str(video_path),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    system = build_system_one_ball_hit_cushion()
    simulate(system, DURATION, FPS)

    # Trajectories
    df = extract_trajectories(system)
    # Clamp/pad to requested duration if simulation ended early/late
    df = df[df["t"] <= DURATION].copy()
    df.to_csv(OUTDIR / "trajectory.csv", index=False)

    # Save the simulated System too (handy for later)
    system.save(OUTDIR / "shot.json")
    summary = summarize_system(system)
    with open(OUTDIR / SUMMARY_NAME, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    # Frames + video using pooltool's renderer
    frames_dir = render_with_pooltool(system, OUTDIR, FPS)
    encode_video(frames_dir, FPS, OUTDIR / VIDEO_NAME)

    print(
        f"Done. CSV: {OUTDIR/'trajectory.csv'}  Video: {OUTDIR/VIDEO_NAME}  Summary: {OUTDIR/SUMMARY_NAME}"
    )


if __name__ == "__main__":
    main()
