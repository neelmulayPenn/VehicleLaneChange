# controllers/intersection_animation.py
import numpy as np
from Bicycle_Model.bicycle_model_visualizer import VehicleBicycleVisualizer
from planners.intersection_env import IntersectionGeometry
from matplotlib.animation import FuncAnimation

def create_intersection_animation(x_traj, u_traj, dt, num_lanes=3, lane_width=5.0):
    vis = VehicleBicycleVisualizer(show_velocity=True, show_trail=True)
    fig, ax = vis.fig, vis.ax

    # Create intersection geometry
    geom = IntersectionGeometry(num_lanes=num_lanes, lane_width=lane_width)

    # Draw all lane centerlines for all directions
    for direction in ["east", "west", "north", "south"]:
        for xs, ys in geom.lane_lines(direction):
            ax.plot(xs, ys, color='red', linestyle='--', alpha=0.5)

    # Set axis limits to include all lanes
    L = geom.road_length
    road_half_width = geom.total_width / 2
    ax.set_xlim(-L - road_half_width, L + road_half_width)
    ax.set_ylim(-L - road_half_width, L + road_half_width)
    ax.set_aspect('equal')

    N = x_traj.shape[0]

    def update(frame):
        vis.draw(x_traj[frame],
                 u_traj[frame if frame < len(u_traj) else -1],
                 frame * dt)
        return []

    ani = FuncAnimation(fig, update, frames=N, interval=dt * 1000, blit=False)
    return fig, ani
