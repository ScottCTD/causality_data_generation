"""Statistics calculation for poolshot dataset."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from shot_utils import config


def compute_dataset_stats(base_output: Path) -> dict[str, Any]:
    """
    Compute aggregate statistics from all shot summaries in the dataset.
    
    Args:
        base_output: Path to the dataset directory (e.g., outputs/dataset_name)
        
    Returns:
        Dictionary containing computed statistics
    """
    shots_dir = base_output / "shots"
    if not shots_dir.exists():
        return {}
    
    # Collect data from all summary.json files
    video_lengths: list[int] = []
    video_lengths_seconds: list[float] = []
    wall_hits_per_shot: list[int] = []
    wall_hits_per_ball: list[int] = []
    pockets_per_shot: list[int] = []
    total_pockets = 0
    
    shot_dirs = sorted(shots_dir.iterdir())
    for shot_dir in shot_dirs:
        if not shot_dir.is_dir():
            continue
            
        summary_path = shot_dir / "summary.json"
        if not summary_path.exists():
            continue
            
        try:
            with open(summary_path, "r", encoding="utf-8") as fp:
                summary = json.load(fp)
        except Exception:
            continue
        
        # Extract video length (total_frames and duration in seconds)
        metadata = summary.get("metadata", {})
        total_frames = metadata.get("total_frames")
        fps = metadata.get("fps", config.FPS)
        if total_frames is not None:
            try:
                frames_int = int(total_frames)
                video_lengths.append(frames_int)
                # Calculate duration in seconds
                try:
                    fps_float = float(fps)
                    if fps_float > 0:
                        duration_seconds = frames_int / fps_float
                        video_lengths_seconds.append(duration_seconds)
                except (TypeError, ValueError):
                    pass
            except (TypeError, ValueError):
                pass
        
        # Extract wall hits and pockets per ball
        balls = summary.get("balls", {})
        shot_wall_hits = 0
        shot_pockets = 0
        
        for ball_id, ball_data in balls.items():
            outcomes = ball_data.get("outcomes", {})
            
            # Count wall hits
            wall_hits = outcomes.get("wall_hits", 0)
            try:
                wall_hits_int = int(wall_hits)
                wall_hits_per_ball.append(wall_hits_int)
                shot_wall_hits += wall_hits_int
            except (TypeError, ValueError):
                pass
            
            # Count pockets
            pocket = outcomes.get("pocket")
            if pocket is not None:
                shot_pockets += 1
                total_pockets += 1
        
        wall_hits_per_shot.append(shot_wall_hits)
        pockets_per_shot.append(shot_pockets)
    
    # Compute statistics
    stats: dict[str, Any] = {
        "total_shots": len(video_lengths),
    }
    
    # Video length statistics (in frames)
    if video_lengths:
        stats["video_frames"] = {
            "min": min(video_lengths),
            "max": max(video_lengths),
            "mean": statistics.mean(video_lengths),
            "median": statistics.median(video_lengths),
            "stdev": statistics.stdev(video_lengths) if len(video_lengths) > 1 else 0.0,
            "total": sum(video_lengths),
        }
    else:
        stats["video_frames"] = {}
    
    # Video length statistics (in seconds)
    if video_lengths_seconds:
        stats["video_seconds"] = {
            "min": min(video_lengths_seconds),
            "max": max(video_lengths_seconds),
            "mean": statistics.mean(video_lengths_seconds),
            "median": statistics.median(video_lengths_seconds),
            "stdev": statistics.stdev(video_lengths_seconds) if len(video_lengths_seconds) > 1 else 0.0,
            "total": sum(video_lengths_seconds),
        }
    else:
        stats["video_seconds"] = {}
    
    # Wall hits statistics
    if wall_hits_per_shot:
        stats["wall_hits_per_shot"] = {
            "min": min(wall_hits_per_shot),
            "max": max(wall_hits_per_shot),
            "mean": statistics.mean(wall_hits_per_shot),
            "median": statistics.median(wall_hits_per_shot),
            "stdev": statistics.stdev(wall_hits_per_shot) if len(wall_hits_per_shot) > 1 else 0.0,
            "total": sum(wall_hits_per_shot),
        }
    else:
        stats["wall_hits_per_shot"] = {}
    
    if wall_hits_per_ball:
        stats["wall_hits_per_ball"] = {
            "min": min(wall_hits_per_ball),
            "max": max(wall_hits_per_ball),
            "mean": statistics.mean(wall_hits_per_ball),
            "median": statistics.median(wall_hits_per_ball),
            "stdev": statistics.stdev(wall_hits_per_ball) if len(wall_hits_per_ball) > 1 else 0.0,
            "total": sum(wall_hits_per_ball),
        }
    else:
        stats["wall_hits_per_ball"] = {}
    
    # Pockets statistics
    stats["pockets"] = {
        "total": total_pockets,
    }
    
    if pockets_per_shot:
        stats["pockets"]["per_shot"] = {
            "min": min(pockets_per_shot),
            "max": max(pockets_per_shot),
            "mean": statistics.mean(pockets_per_shot),
            "median": statistics.median(pockets_per_shot),
            "stdev": statistics.stdev(pockets_per_shot) if len(pockets_per_shot) > 1 else 0.0,
        }
        stats["pockets"]["shots_with_pockets"] = sum(1 for p in pockets_per_shot if p > 0)
        stats["pockets"]["shots_without_pockets"] = sum(1 for p in pockets_per_shot if p == 0)
    else:
        stats["pockets"]["per_shot"] = {}
        stats["pockets"]["shots_with_pockets"] = 0
        stats["pockets"]["shots_without_pockets"] = 0
    
    return stats

