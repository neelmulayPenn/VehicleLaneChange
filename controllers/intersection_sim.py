# controllers/intersection_sim.py
import numpy as np
from Bicycle_Model.bicycle_model_dynamics import VehicleBicycleModel
from controllers.mpc_controller import DrakeMPC
from planners.intersection_env import IntersectionGeometry
from controllers.intersection_reference_provider import IntersectionReferenceProvider

def run_intersection_sim(tf=12.0, dt=0.05, direction="west"):
    geom = IntersectionGeometry()
    ref_provider = IntersectionReferenceProvider(geom, direction=direction, lane_id=2, speed=10.0)
    mpc = DrakeMPC(reference_provider=ref_provider)

    N = int(tf / dt)
    t_traj = np.linspace(0, tf, N+1)

    # --- Choose starting position based on direction ---
  # Choose starting position based on direction
    if direction == "east":
        start_x = -geom.road_length
        start_y = geom.lane_center("east", lane_id=0)  # <-- remove [1]
        psi0=0.0

    elif direction == "west":
        start_x = geom.road_length
        start_y = geom.lane_center("west", lane_id=0)  # <-- remove [1]
        psi0=np.pi

    elif direction == "north":
        start_y = -geom.road_length
        start_x = geom.lane_center("north", lane_id=0)  # <-- remove [0]
        psi0=np.pi/2

    elif direction == "south":
        start_y = geom.road_length
        start_x = geom.lane_center("south", lane_id=0)  # <-- remove [0]
        psi0=-np.pi/2

    else:
        raise ValueError(f"Unknown direction '{direction}'")

    # Initial 6-state
    x0 = np.array([start_x, start_y, psi0, 10.0, 0.0, 0.0])

    # Storage
    x_traj = np.zeros((N+1, 6))
    u_traj = np.zeros((N, 2))
    x_traj[0] = x0

    bicycle = VehicleBicycleModel(
        m=1000, Iz=2500, Lf=1.1, Lr=1.1,
        Cf=80000, Cr=80000, d_limit=np.deg2rad(30)
    )

    # --- Main simulation loop ---
    for k in range(N):
        x = x_traj[k]

        ref = ref_provider.get_ref_horizon(x, mpc.cfg.T, mpc.cfg.DT)
        u = mpc.solve_mpc(x, ref)
        u_traj[k] = u

        xdot = bicycle.continuous_time_full_dynamics(x, u)
        x_traj[k+1] = x + dt * xdot

    return x_traj, u_traj, t_traj, dt, N
