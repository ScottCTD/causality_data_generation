"""Debug script to generate videos at table corners."""

from pathlib import Path
import shutil

import pooltool as pt
from shot_utils import config
from shot_utils.rendering import encode_video, render_frames
from shot_utils.simulation import build_system_one_ball_hit_cushion, simulate_shot


def main():
    """Generate videos at the four table corners."""
    output_dir = Path("debug_output")
    output_dir.mkdir(exist_ok=True)

    # Get table dimensions
    table = pt.Table.default()
    w = table.w
    l = table.l  # length (height in y-direction)
    
    print(f"Table dimensions: w={w:.4f}, l={l:.4f}")

    # Four corner positions
    corners = [
        ("corner_00", 0.0, 0.0),      # Bottom-left (head rail, left)
        ("corner_w0", w, 0.0),        # Bottom-right (head rail, right)
        ("corner_0l", 0.0, l),        # Top-left (foot rail, left)
        ("corner_wl", w, l),          # Top-right (foot rail, right)
    ]

    # Use same parameters as debug2.py
    velocity = 1.5
    phi = 216.0
    fps = 15

    for corner_id, x, y in corners:
        print(f"\nProcessing {corner_id}...")
        print(f"  Position: ({x:.4f}, {y:.4f})")
        print(f"  Velocity: {velocity:.4f}")
        print(f"  Phi: {phi:.1f}°")

        outdir = output_dir / corner_id
        outdir.mkdir(parents=True, exist_ok=True)

        # Build system
        system = build_system_one_ball_hit_cushion(x, y, velocity, phi)

        # Simulate
        print("  Simulating...")
        simulate_shot(system, fps)

        # Render frames
        print("  Rendering frames...")
        frames_dir = render_frames(system, outdir, fps)

        # Encode video
        video_path = outdir / f"video_{corner_id}.mp4"
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
