"""Debug script to generate two videos with different ball positions but same phi."""

from pathlib import Path
import shutil

import pooltool as pt
from shot_utils import config
from shot_utils.rendering import encode_video, render_frames
from shot_utils.simulation import build_system_one_ball_hit_cushion, simulate_shot


def main():
    """Generate two videos with the specified parameters."""
    output_dir = Path("debug_output")
    output_dir.mkdir(exist_ok=True)

    # Shot 1: Ball at top (near head rail)
    shot1_params = {
        "x": 0.16097250000000002,
        "y": 1.659255,
        "velocity": 1.5,
        "phi": 216.0,
    }

    # Shot 2: Ball at bottom (near foot rail)
    shot2_params = {
        "x": 0.3838575,
        "y": 0.32194500000000004,
        "velocity": 1.5,
        "phi": 216.0,
    }

    shots = [
        ("shot1_top", shot1_params),
        ("shot2_bottom", shot2_params),
    ]

    for shot_id, params in shots:
        print(f"\nProcessing {shot_id}...")
        print(f"  Position: ({params['x']:.4f}, {params['y']:.4f})")
        print(f"  Velocity: {params['velocity']:.4f}")
        print(f"  Phi: {params['phi']:.1f}°")

        outdir = output_dir / shot_id
        outdir.mkdir(parents=True, exist_ok=True)

        # Build system
        system = build_system_one_ball_hit_cushion(
            params["x"],
            params["y"],
            params["velocity"],
            params["phi"],
        )

        # Use 15 fps
        fps = 15

        # Simulate
        print("  Simulating...")
        simulate_shot(system, fps)

        # Render frames
        print("  Rendering frames...")
        frames_dir = render_frames(system, outdir, fps)

        # Encode video
        video_path = outdir / f"video_{shot_id}.mp4"
        print(f"  Encoding video to {video_path}...")
        encode_video(frames_dir, fps, video_path)

        # Cleanup frames
        try:
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
        except Exception:
            pass

        print(f"  ✓ Completed: {video_path}")

    print(f"\n✓ All videos generated in {output_dir}/")


if __name__ == "__main__":
    main()
