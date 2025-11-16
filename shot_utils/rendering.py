from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from subprocess import DEVNULL

from pooltool.ani.animate import FrameStepper
from pooltool.ani.camera import CameraState, camera_states
import pooltool.ani.camera.states as camera_states_module
from pooltool.ani.image import ImageExt, ImageZip, save_images

from . import config


_PACKAGE_STATE_DIR = Path(camera_states_module.__file__).parent
_REPO_STATE_DIR = (
    Path(__file__).resolve().parent.parent / "pooltool" / "pooltool" / "ani" / "camera" / "states"
)
_STATE_DIRS = []
if _REPO_STATE_DIR.exists():
    _STATE_DIRS.append(_REPO_STATE_DIR)
_STATE_DIRS.append(_PACKAGE_STATE_DIR)


def _get_camera_state(name: str) -> CameraState:
    if name in camera_states:
        return camera_states[name]
    for state_dir in _STATE_DIRS:
        state_path = state_dir / f"{name}.json"
        if state_path.exists():
            state = CameraState.from_json(state_path)
            camera_states[name] = state
            return state
    raise KeyError(
        f"Camera state '{name}' not found. Checked: {[str(d / f'{name}.json') for d in _STATE_DIRS]}"
    )


def render_frames(system, outdir: Path, fps: int, camera_name: str | None = None) -> Path:
    frames_dir = outdir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)

    interface = FrameStepper()
    exporter = ImageZip(path=frames_dir, ext=ImageExt.PNG,
                        prefix=config.FRAME_PREFIX, compress=False)

    try:
        state_name = camera_name or config.CAMERA_NAME
        camera_state = _get_camera_state(state_name)
        offset = getattr(config, "CAMERA_DISTANCE_OFFSET", 0.0)
        if offset:
            cam_pos = (
                camera_state.cam_pos[0] + offset,
                camera_state.cam_pos[1],
                camera_state.cam_pos[2],
            )
            camera_state = CameraState(
                cam_hpr=camera_state.cam_hpr,
                cam_pos=cam_pos,
                fixation_hpr=camera_state.fixation_hpr,
                fixation_pos=camera_state.fixation_pos,
            )

        save_images(
            exporter=exporter,
            system=system,
            interface=interface,
            size=config.FRAME_SIZE,
            fps=fps,
            camera_state=camera_state,
            gray=False,
            show_hud=False,
        )
    finally:
        interface.destroy()

    return frames_dir


def encode_video(frames_dir: Path, fps: int, video_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",              # suppress ffmpeg logs (only show errors)
        "-framerate", str(fps),
        "-i", str(frames_dir / config.FRAME_PATTERN),
        "-pix_fmt", "yuv420p",
        "-crf", "20",                      # optional: slightly higher CRF for smaller 240p files
        "-vf", "scale=-2:240:flags=lanczos,pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-movflags", "+faststart",         # optional: better web playback
        str(video_path),
    ]

    subprocess.run(cmd, check=True, stdout=DEVNULL)
