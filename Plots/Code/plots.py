"""
CLF-CBF Robot Trajectory Visualization
---------------------------------------
Plots the robot's full path together with the moving stick/puck,
the static obstacles (with their safety margins), and the arena
boundary — matching the layout of the reference scene image.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# ----------------------------- Config ------------------------------------
TRAJ_PATH = "robot_trajectory.csv"
OBS_PATH = "obstacles.csv"
OUT_PATH = "trajectory_plot.png"

SAFE_DISTANCE = 0.50          # obstacle safety-circle radius (m)
ARENA_BOUNDS = (-2.0, 2.0)    # square arena, matches the reference image
OBSTACLE_SIZE = 0.18          # visual size of the square obstacle markers


def load_data(traj_path: str, obs_path: str):
    """Load and lightly validate the trajectory / obstacle CSVs."""
    traj = pd.read_csv(traj_path)
    obs = pd.read_csv(obs_path)

    required_traj_cols = {"robot_x", "robot_y", "stick_x", "stick_y", "puck_x", "puck_y"}
    missing = required_traj_cols - set(traj.columns)
    if missing:
        raise ValueError(f"robot_trajectory.csv is missing columns: {missing}")

    required_obs_cols = {"robot_id", "x", "y"}
    missing_obs = required_obs_cols - set(obs.columns)
    if missing_obs:
        raise ValueError(f"obstacles.csv is missing columns: {missing_obs}")

    # Drop back-to-back duplicate rows (the raw log repeats samples when the
    # controller stalls) so the path/gradient isn't skewed by dead time.
    traj = traj.loc[(traj.shift() != traj).any(axis=1)].reset_index(drop=True)

    return traj, obs


def add_gradient_path(ax, x, y, cmap="Blues", label=None):
    """Draw a path whose color darkens over time, so direction of travel
    is visible at a glance (arrows alone get cluttered on a dense path)."""
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidth=2.5)
    lc.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(lc)
    if label:
        # Proxy artist so the gradient path still gets one legend entry.
        proxy = Line2D([0], [0], color=plt.get_cmap(cmap)(0.7), lw=2.5, label=label)
        return proxy
    return None


def plot_scene(traj: pd.DataFrame, obs: pd.DataFrame, out_path: str = OUT_PATH):
    fig, ax = plt.subplots(figsize=(9, 8))

    # --- Arena boundary -----------------------------------------------
    lo, hi = ARENA_BOUNDS
    ax.add_patch(Rectangle((lo, lo), hi - lo, hi - lo,
                            fill=False, edgecolor="black", linewidth=2, zorder=1))

    # --- Obstacles + safety margins ------------------------------------
    for _, row in obs.iterrows():
        ox, oy, rid = row["x"], row["y"], int(row["robot_id"])

        ax.add_patch(Rectangle((ox - OBSTACLE_SIZE / 2, oy - OBSTACLE_SIZE / 2),
                                OBSTACLE_SIZE, OBSTACLE_SIZE,
                                facecolor="black", edgecolor="black", zorder=3))
        ax.add_patch(Circle((ox, oy), SAFE_DISTANCE, fill=False,
                             linestyle="--", color="crimson", alpha=0.6, zorder=2))
        ax.annotate(str(rid), (ox, oy), xytext=(6, 6), textcoords="offset points",
                     fontsize=9, fontweight="bold", color="crimson", zorder=4)

    # --- Robot path (gradient = time direction) -------------------------
    robot_proxy = add_gradient_path(ax, traj["robot_x"], traj["robot_y"],
                                     cmap="Blues", label="Robot path")

    # --- Stick and puck paths (both move over the episode) --------------
    stick_proxy = add_gradient_path(ax, traj["stick_x"], traj["stick_y"],
                                     cmap="Oranges", label="Stick path")
    puck_proxy = add_gradient_path(ax, traj["puck_x"], traj["puck_y"],
                                    cmap="Purples", label="Puck path")

    # --- Key point markers ----------------------------------------------
    ax.scatter(*traj[["robot_x", "robot_y"]].iloc[0], color="green", s=140,
               zorder=5, edgecolor="black", label="Robot start")
    ax.scatter(*traj[["robot_x", "robot_y"]].iloc[-1], color="red", s=140,
               zorder=5, edgecolor="black", label="Robot end")
    ax.scatter(*traj[["stick_x", "stick_y"]].iloc[-1], color="darkorange", marker="s",
               s=160, zorder=5, edgecolor="black", label="Stick (final)")
    ax.scatter(*traj[["puck_x", "puck_y"]].iloc[-1], color="purple", marker="o",
               s=160, zorder=5, edgecolor="black", label="Puck (final)")

    # --- Cosmetics --------------------------------------------------------
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("CLF-CBF Robot Trajectory")
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(lo - 0.5, hi + 0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    extra = [h for h in (robot_proxy, stick_proxy, puck_proxy) if h is not None]
    ax.legend(handles=extra + handles, loc="upper left", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.show()
    return fig, ax


if __name__ == "__main__":
    traj_df, obs_df = load_data(TRAJ_PATH, OBS_PATH)
    plot_scene(traj_df, obs_df)