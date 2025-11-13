from __future__ import annotations

import pooltool as pt
import pandas as pd


def build_system_one_ball_hit_cushion() -> pt.System:
    table = pt.Table.default()
    balls = {
        "cue": pt.Ball.create("cue", xy=(0.6, table.w / 2.0)),
        "1": pt.Ball.create("1", xy=(5.0, 6.0)),
    }
    cue = pt.Cue.default()
    system = pt.System(table=table, balls=balls, cue=cue)
    system.cue.set_state(V0=2.0, phi=0.0)
    return system


def simulate_shot(system: pt.System, duration: float, fps: int) -> None:
    pt.simulate(system, continuous=True, dt=1.0 / fps, inplace=True)


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
