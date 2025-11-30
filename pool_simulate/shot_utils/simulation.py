from __future__ import annotations

from dataclasses import dataclass

import attrs
import numpy as np
import pandas as pd

import pooltool as pt

from pooltool import BallParams


@dataclass
class BallState:

    x: float
    y: float
    speed: float = 0.0
    phi: float = 0.0
    param: BallParams = BallParams()


def set_ball_velocity(ball: pt.Ball, speed: float, phi: float) -> None:
    """Set a ball's initial velocity from speed and angle (rolling motion)
    
    Note: This is a simplified direct velocity setting. The cue ball's velocity is actually
    calculated from physics (cue stick mass, contact point, spin, etc.), not directly from
    V0 and phi. This function provides a simple way to set velocity for non-cue balls.
    
    Args:
        ball: The ball to set velocity for
        speed: Initial speed in m/s (this is the ball speed, not cue stick speed)
        phi: Direction angle in degrees (same convention as cue: 0=right, 90=foot rail, 180=left, 270=head rail)
    """
    # Convert phi to radians and compute velocity components
    phi_rad = np.deg2rad(phi)
    vx = speed * np.cos(phi_rad)
    vy = speed * np.sin(phi_rad)
    ball.state.rvw[1] = np.array([vx, vy, 0.0], dtype=np.float64)
    
    # For rolling motion, set angular velocity to match linear velocity
    # For a ball rolling without slipping on a flat surface: ω = v / R
    # The angular velocity is around the z-axis (perpendicular to table)
    # Using right-hand rule: for velocity in xy plane, ωz should be negative
    R = ball.params.R
    omega_z = -speed / R  # Negative for right-hand rule with forward motion
    ball.state.rvw[2] = np.array([0.0, 0.0, omega_z], dtype=np.float64)
    ball.state.s = pt.constants.rolling  # Set motion state to rolling


def create_ball_from_state(ball_id: str, ball_state: BallState) -> pt.Ball:
    """Create a pooltool Ball from a BallState configuration
    
    Args:
        ball_id: The ID for the ball
        ball_state: The BallState configuration (x, y, speed, phi, param)
    
    Returns:
        A pooltool Ball initialized with the specified state
    """
    # Create ball with position and custom params
    ball = pt.Ball.create(ball_id, xy=(ball_state.x, ball_state.y), **attrs.asdict(ball_state.param))
    
    # Set velocity if speed > 0 (stationary balls already have default zero velocity)
    if ball_state.speed > 0:
        set_ball_velocity(ball, ball_state.speed, ball_state.phi)
    
    return ball


def build_system(ball_states: dict[str, BallState]) -> pt.System:
    """Build a system with balls initialized from BallState configurations
    
    The cue ball's velocity is set by the cue stick strike during simulation,
    not manually. The cue stick parameters (V0, phi) are taken from the
    cue ball's state (speed, phi).
    
    Args:
        ball_states: Dictionary mapping ball IDs to BallState configurations.
                     Must contain "cue" key.
    
    Returns:
        A pooltool System with balls initialized according to ball_states
    """
    if "cue" not in ball_states:
        raise ValueError('ball_states must contain a "cue" ball')
    
    table = pt.Table.default()
    cue_state = ball_states["cue"]
    
    # Create all balls from ball_states
    # Note: Don't set velocity for cue ball - let cue stick strike handle it
    balls = {}
    for ball_id, ball_state in ball_states.items():
        if ball_id == "cue":
            # Create cue ball without velocity (cue stick will set it)
            balls[ball_id] = pt.Ball.create(
                ball_id, xy=(ball_state.x, ball_state.y), **attrs.asdict(ball_state.param)
            )
        else:
            balls[ball_id] = create_ball_from_state(ball_id, ball_state)
    
    # If there's only one ball (the cue ball), add a dummy ball to avoid simulation errors
    # Place dummy ball far away and mark it as pocketed so it doesn't interfere
    if len(balls) == 1:
        dummy_ball = pt.Ball.dummy("dummy")
        # Place dummy ball far outside the table bounds
        dummy_ball.state.rvw[0] = np.array([table.w * 10, table.l * 10, dummy_ball.params.R], dtype=np.float64)
        # Mark as pocketed so it doesn't participate in collisions
        dummy_ball.state.s = pt.constants.pocketed
        balls["dummy"] = dummy_ball
    
    # Initialize cue stick with parameters from cue ball state
    cue = pt.Cue.default()
    cue.cue_ball_id = "cue"
    system = pt.System(table=table, balls=balls, cue=cue)
    system.cue.set_state(V0=cue_state.speed, phi=cue_state.phi)
    
    return system


def simulate_shot(system: pt.System, fps: int, max_second: float | None = None) -> None:
    pt.simulate(system, continuous=True, dt=1.0 / fps, inplace=True, t_final=max_second)


def extract_trajectories(system: pt.System) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for ball_id, ball in system.balls.items():
        history = ball.history_cts if not ball.history_cts.empty else ball.history
        rvw, _, t = history.vectorize()
        pos = rvw[:, 0, :]
        vel = rvw[:, 1, :]
        omg = rvw[:, 2, :]
        for i in range(len(t)):
            records.append(
                {
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
                }
            )
    df = pd.DataFrame.from_records(records)
    df.sort_values(["t", "ball_id"], inplace=True, kind="stable")
    df.reset_index(drop=True, inplace=True)
    return df
