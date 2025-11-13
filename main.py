# make_video.py
# Requirements: pooltool==0.5.0, numpy, pandas, ffmpeg (CLI installed)

import json

from shot_utils import config
from shot_utils.rendering import encode_video, render_frames
from shot_utils.simulation import (
    build_system_one_ball_hit_cushion,
    extract_trajectories,
    simulate_shot,
)
from shot_utils.summary import summarize_system


def ensure_output_dir() -> None:
    config.OUTDIR.mkdir(parents=True, exist_ok=True)


def persist_summary(system) -> None:
    summary = summarize_system(system)
    with open(config.OUTDIR / config.SUMMARY_NAME, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)


def persist_trajectory(df) -> None:
    df.to_csv(config.OUTDIR / "trajectory.csv", index=False)


def main():
    ensure_output_dir()

    system = build_system_one_ball_hit_cushion()
    simulate_shot(system, config.DURATION, config.FPS)

    df = extract_trajectories(system)
    df = df[df["t"] <= config.DURATION].copy()
    persist_trajectory(df)

    system.save(config.OUTDIR / "shot.json")
    persist_summary(system)

    frames_dir = render_frames(system, config.OUTDIR, config.FPS)
    encode_video(frames_dir, config.FPS, config.OUTDIR / config.VIDEO_NAME)

    print(
        f"Done. CSV: {config.OUTDIR/'trajectory.csv'}  "
        f"Video: {config.OUTDIR/config.VIDEO_NAME}  Summary: {config.OUTDIR/config.SUMMARY_NAME}"
    )


if __name__ == "__main__":
    main()
