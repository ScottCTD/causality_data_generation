from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from pooltool.ani.animate import FrameStepper
from pooltool.ani.camera import camera_states
from pooltool.ani.image import ImageExt, ImageZip, save_images

from . import config


def render_frames(system, outdir: Path, fps: int) -> Path:
    frames_dir = outdir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)

    interface = FrameStepper()
    exporter = ImageZip(path=frames_dir, ext=ImageExt.PNG, prefix=config.FRAME_PREFIX, compress=False)

    try:
        save_images(
            exporter=exporter,
            system=system,
            interface=interface,
            size=config.FRAME_SIZE,
            fps=fps,
            camera_state=camera_states[config.CAMERA_NAME],
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
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / config.FRAME_PATTERN),
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(video_path),
    ]
    subprocess.run(cmd, check=True)
