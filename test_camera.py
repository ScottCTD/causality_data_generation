"""Interactive script for testing custom camera parameters."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pooltool as pt
from pooltool.ani.camera._camera import CameraState
from pooltool.ani.camera.states import camera_states
from pooltool.objects.ball.sets import BallSet

from shot_utils import config
from shot_utils.rendering import encode_video, render_frames
from shot_utils.simulation import simulate_shot
from shot_utils.summary import summarize_system


def _build_demo_system(phi: float, speed: float) -> tuple[pt.System, tuple[float, float]]:
    """Create a simple two-ball setup for visualization."""
    table = pt.Table.default()
    radius = pt.BallParams.default().R
    spacing = 6 * radius
    cue_start = (table.w * 0.35, table.l * 0.25)

    balls = {"cue": pt.Ball.create("cue", xy=cue_start)}
    balls["1"] = pt.Ball.create("1", xy=(cue_start[0], cue_start[1] + spacing))

    cue = pt.Cue.default()
    system = pt.System(table=table, balls=balls, cue=cue)
    system.set_ballset(BallSet("pooltool_pocket"))
    system.cue.set_state(V0=speed, phi=phi)
    return system, cue_start


def _parse_triplet(values: list[float] | None, fallback: tuple[float, float, float] | None) -> tuple[float, float, float] | None:
    if values is None:
        return fallback
    return tuple(values)  # type: ignore[return-value]


def _load_camera_state(base_name: str) -> CameraState:
    state_path = Path("pooltool") / "pooltool" / "ani" / "camera" / "states" / f"{base_name}.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Camera state '{base_name}' not found at {state_path}")
    return CameraState.from_json(state_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a debug shot using custom camera parameters.")
    parser.add_argument("--base-state", default="7_foot_offcenter", help="Name of the camera state JSON to start from.")
    parser.add_argument("--cam-pos", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Override camera position.")
    parser.add_argument("--cam-hpr", type=float, nargs=3, metavar=("H", "P", "R"), help="Override camera heading/pitch/roll.")
    parser.add_argument("--fix-pos", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Override fixation position.")
    parser.add_argument("--fix-hpr", type=float, nargs=3, metavar=("H", "P", "R"), help="Override fixation heading/pitch/roll.")
    parser.add_argument("--phi", type=float, default=72.0, help="Cue direction in degrees.")
    parser.add_argument("--speed", type=float, default=2.0, help="Cue speed in m/s.")
    parser.add_argument("--output-dir", type=Path, default=config.BASE_OUTPUT / "debug" / "test_camera", help="Directory for the rendered assets.")
    parser.add_argument("--state-name", default="custom_camera_test", help="Temporary name for the injected camera state.")
    parser.add_argument("--keep-frames", action="store_true", help="Keep raw frame images (skip cleanup).")
    args = parser.parse_args()

    base_state = _load_camera_state(args.base_state)
    cam_pos = _parse_triplet(args.cam_pos, base_state.cam_pos)
    cam_hpr = _parse_triplet(args.cam_hpr, base_state.cam_hpr)
    fix_pos = _parse_triplet(args.fix_pos, base_state.fixation_pos)
    fix_hpr = _parse_triplet(args.fix_hpr, base_state.fixation_hpr)

    if cam_pos is None or cam_hpr is None:
        raise ValueError("Camera position and orientation must be defined.")

    custom_state = CameraState(
        cam_hpr=cam_hpr,
        cam_pos=cam_pos,
        fixation_hpr=fix_hpr,
        fixation_pos=fix_pos,
    )

    system, cue_start = _build_demo_system(phi=args.phi, speed=args.speed)
    simulate_shot(system, config.FPS)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Inject the custom camera state and render.
    original_camera = config.CAMERA_NAME
    camera_states[args.state_name] = custom_state
    config.CAMERA_NAME = args.state_name

    try:
        frames_dir = render_frames(system, args.output_dir, config.FPS)
    finally:
        config.CAMERA_NAME = original_camera
        camera_states.pop(args.state_name, None)

    frame_count = len(list(frames_dir.glob(f"{config.FRAME_PREFIX}_*.png")))
    metadata = {
        "shot_id": "test_camera",
        "cue_start": {"x": cue_start[0], "y": cue_start[1]},
        "velocity": args.speed,
        "phi": args.phi,
        "fps": config.FPS,
        "total_frames": frame_count,
        "camera": {
            "base_state": args.base_state,
            "cam_pos": cam_pos,
            "cam_hpr": cam_hpr,
            "fixation_pos": fix_pos,
            "fixation_hpr": fix_hpr,
        },
    }

    summary = summarize_system(system, metadata=metadata)
    summary_path = args.output_dir / "summary_test_camera.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    video_path = args.output_dir / "video.mp4"
    encode_video(frames_dir, config.FPS, video_path)

    if not args.keep_frames:
        try:
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
        except Exception:
            pass

    print("Rendered test shot with camera parameters:")
    print(json.dumps(metadata["camera"], indent=2))
    print(f"Video: {video_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
