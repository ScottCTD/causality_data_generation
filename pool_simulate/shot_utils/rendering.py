from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from subprocess import DEVNULL

import numpy as np
from numpy.typing import NDArray

from pooltool.ani.animate import FrameStepper
from pooltool.ani.camera import CameraState, camera_states
import pooltool.ani.camera.states as camera_states_module
from pooltool.ani.image import ImageExt, ImageZip, image_stack, save_images

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


def _render_frame_stack(
    system,
    fps: int,
    camera_name: str | None = None,
) -> NDArray[np.uint8]:
    """Render a stack of RGB frames for a given system and camera.

    This mirrors the logic in `render_frames` but keeps everything in memory so it
    can be streamed directly to ffmpeg instead of writing PNGs to disk.
    """
    interface = FrameStepper()
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

        frames = image_stack(
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

    # Ensure frames are uint8 and RGB (drop alpha if present)
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4:
        raise ValueError(f"Expected frame stack with shape (N, H, W, C), got {frames.shape}")

    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    elif frames.shape[-1] != 3:
        raise ValueError(f"Expected 3 or 4 channels, got {frames.shape[-1]}")

    return frames


def encode_video_stream(frames: NDArray[np.uint8], fps: int, video_path: Path) -> None:
    """Encode a video by streaming raw frames to ffmpeg via stdin.

    This avoids writing individual PNGs to disk and is typically much faster when
    generating large datasets.
    """
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4:
        raise ValueError(f"Expected frame stack with shape (N, H, W, C), got {frames.shape}")

    num_frames, height, width, channels = frames.shape
    if channels == 4:
        frames = frames[..., :3]
        channels = 3
    if channels != 3:
        raise ValueError(f"Expected 3-channel RGB frames, got {channels}")

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",  # suppress ffmpeg logs (only show errors)
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",  # optional: slightly higher CRF for smaller 240p files
        "-vf",
        "scale=-2:240:flags=lanczos,pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-movflags",
        "+faststart",  # optional: better web playback
        str(video_path),
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        assert proc.stdin is not None
        proc.stdin.write(frames.tobytes())
        proc.stdin.close()

        return_code = proc.wait()
        if return_code != 0:
            stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
            raise RuntimeError(f"ffmpeg failed with return code {return_code}: {stderr}")
    finally:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        if proc.stderr:
            proc.stderr.close()


def render_and_encode_video(
    system,
    outdir: Path,
    fps: int,
    video_path: Path,
    camera_name: str | None = None,
) -> int:
    """High-level helper to render frames in-memory and encode via ffmpeg streaming.

    Returns:
        The number of frames rendered/encoded.
    """
    frames = _render_frame_stack(system, fps=fps, camera_name=camera_name)
    encode_video_stream(frames, fps=fps, video_path=video_path)
    return frames.shape[0]


def encode_video(frames_dir: Path, fps: int, video_path: Path) -> None:
    """Legacy disk-based encoder kept for backward compatibility.

    New code should prefer `render_and_encode_video` + `encode_video_stream`.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",  # suppress ffmpeg logs (only show errors)
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / config.FRAME_PATTERN),
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",  # optional: slightly higher CRF for smaller 240p files
        "-vf",
        "scale=-2:240:flags=lanczos,pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-movflags",
        "+faststart",  # optional: better web playback
        str(video_path),
    ]

    subprocess.run(cmd, check=True, stdout=DEVNULL)
