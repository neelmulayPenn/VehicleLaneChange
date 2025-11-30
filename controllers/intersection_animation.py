# controllers/intersection_animation.py
import numpy as np
from Bicycle_Model.bicycle_model_visualizer import VehicleBicycleVisualizer
from planners.intersection_env import IntersectionGeometry
from matplotlib.animation import FuncAnimation

class SimpleVehicle:
    def __init__(self, x, y, psi=0.0):
        self.x = x
        self.y = y
        self.psi = psi

def create_intersection_animation(x_traj, u_traj, dt, obstacle_trajs=None):
    vis = VehicleBicycleVisualizer(show_velocity=True, show_trail=True)
    fig, ax = vis.fig, vis.ax

    geom = IntersectionGeometry()

    draw_intersection_lanes(ax, geom)

    # ---- (NEW) Add traffic vehicles if provided ----
    traffic_vehicles = []
    if obstacle_trajs is not None:
        M = obstacle_trajs.shape[1]
        for i in range(M):
            x0 = obstacle_trajs[0, i, 0]
            y0 = obstacle_trajs[0, i, 1]
            psi0 = obstacle_trajs[0, i, 4]   # <-- using yaw column
            traffic_vehicles.append(SimpleVehicle(x0, y0, psi0))
        vis.add_traffic(traffic_vehicles)
    print("Lane centers (per direction):")
    for d in ["east", "north"]:
        print(f" {d}: {geom.lanes[d]}")

    # Setup axes
    L = geom.road_length
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect('equal')

    N = x_traj.shape[0]

    def update(frame):
        # Update traffic if present
        if obstacle_trajs is not None:
            for i in range(len(traffic_vehicles)):
                traffic_vehicles[i].x = obstacle_trajs[frame, i, 0]
                traffic_vehicles[i].y = obstacle_trajs[frame, i, 1]
                
                traffic_vehicles[i].psi = obstacle_trajs[frame, i, 4]

        vis.draw(x_traj[frame], u_traj[frame if frame < len(u_traj) else -1], frame*dt)
        return []

    ani = FuncAnimation(fig, update, frames=N, interval=dt*1000, blit=False)
    return fig, ani

def draw_intersection_lanes(ax, geom):
    """Draws ONLY the true lane centerlines from IntersectionGeometry."""
    directions = ["east", "west", "north", "south"]

    for direction in directions:
        for lane_id in range(geom.num_lanes):
            xs, ys = geom.straight_path(direction, lane_id, N=500)
            # print(f"Direction: {direction}, Lane ID: {lane_id}")
            # print("xs unique:", np.unique(xs)[:5])
            # print("lane offset:", geom.lanes[direction][lane_id])
            ax.plot(xs, ys, 'k', linewidth=1.5, zorder=-10)