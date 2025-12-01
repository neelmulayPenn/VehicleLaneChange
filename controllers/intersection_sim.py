# controllers/intersection_sim.py
import numpy as np
from dataclasses import dataclass
from Bicycle_Model.bicycle_model_dynamics import VehicleBicycleModel
from controllers.mpc_controller import DrakeMPC
from planners.intersection_env import IntersectionGeometry
from controllers.reference_provider import ReferenceProvider
from planners.obstacle_aware_planner import ObstacleAwarePlanner
from planners.perception import NoisyPerception

# Direction lookup (unit vectors)
DIRECTION_VECTORS = {
    "east":  np.array([1.0, 0.0]),
    "west":  np.array([-1.0, 0.0]),
    "north": np.array([0.0, 1.0]),
    "south": np.array([0.0, -1.0]),
}
DIRECTION_YAWS = {
    "east":  0.0,
    "west":  np.pi,
    "north": np.pi,
    "south": 1.5*np.pi,
}


@dataclass
class IntVehicle:
    """
    Intersection-capable vehicle for the simulation.
    Attributes match what NoisyPerception expects: x, y, v, lane.
    Adds direction field and step() to move in the correct axis.
    """
    x: float
    y: float
    v: float         # speed (m/s)
    lane: int
    direction: str   # one of "east","west","north","south"

    def step(self, dt: float):
        vec = DIRECTION_VECTORS[self.direction]
        self.x += vec[0] * self.v * dt
        self.y += vec[1] * self.v * dt
    
    @property
    def psi(self):
        return DIRECTION_YAWS[self.direction]


def run_intersection_sim(tf: float = 20.0,
                         dt: float = 0.05,
                         ego_speed: float = 20.0,
                         direction: str = "east"):
    """
    Intersection scenario that re-uses the straight-road TTC/ObstacleAwarePlanner
    + perception + ReferenceProvider + DrakeMPC pipeline.

    Returns: x_traj, u_traj, t_traj, obstacle_trajs, dt, N
      - obstacle_trajs shape: (N+1, M, 4) with columns [x, y, v, lane]
    """
    # === Geometry & ego init ===
    geom = IntersectionGeometry()

    # lane offsets for each lane in ego approach direction
    lane_offsets = [geom.lane_center(direction, lane_id=i) for i in range(geom.num_lanes)]

    # Set ego start far upstream along its approach axis
    if direction == "east":
        start_x = -geom.road_length
        start_y = lane_offsets[1]  # start in center lane by default
        psi0 = 0.0
    elif direction == "west":
        start_x = geom.road_length
        start_y = lane_offsets[1]
        psi0 = np.pi
    elif direction == "north":
        start_y = -geom.road_length
        start_x = lane_offsets[1]
        psi0 = np.pi / 2
    elif direction == "south":
        start_y = geom.road_length
        start_x = lane_offsets[1]
        psi0 = -np.pi / 2
    else:
        raise ValueError(f"Unknown direction '{direction}'")

    x0 = np.array([start_x, start_y, psi0, ego_speed, 0.0, 0.0])

    # === Create obstacles as IntVehicle objects (true world) ===
    # spawn_specs: (direction, lane_id, speed, offset_from_start)
    # offset_from_start is how far along their approach they already are (meters)
    spawn_specs = [
        # Cross traffic (north -> south or south -> north), and some along ego route
        # Example scenario: cross traffic coming north->south across ego east-west
        ("east", 0, 8.0,  20.0),
        ("east", 1, 2.0, 0.0),
        ("east",1, 6.0,  50.0),
        ("east", 2, 3.0,  10.0),
        ("east", 2, 7.0,  00.0)

        # Also a vehicle in ego direction (ahead) to test following behavior (optional)
        # (direction, lane_id, speed, offset)
        # ("east", 1, 6.0, 30.0),
    ]

    # Build IntVehicle list with consistent starting coordinates using geom
    obstacles = []
    for (d, lane_id, speed, offset) in spawn_specs:
        if d in ("east", "west"):
            # x coordinate along approach axis
            if d == "east":
                x_init = -geom.road_length + offset
            else:
                x_init = geom.road_length - offset
            y_init = geom.lane_center(d, lane_id)
        else:  # north/south -> y varies, x from lane center for that direction
            if d == "north":
                y_init = -geom.road_length + offset
            else:
                y_init = geom.road_length - offset
            x_init = geom.lane_center(d, lane_id)

        obstacles.append(IntVehicle(x=x_init, y=y_init, v=speed, lane=lane_id, direction=d))

    # === Perception (low-rate noisy) using same class as straight-road ===
    perception = NoisyPerception(obstacles, perception_dt=0.2, sigma_x=0.5, sigma_y=0.3)

    # === Planner + ReferenceProvider + MPC setup ===
    bicycle = VehicleBicycleModel(
        m=1000, Iz=2500, Lf=1.1, Lr=1.6,
        Cf=80000, Cr=80000, d_limit=np.deg2rad(30)
    )

    bp = ObstacleAwarePlanner(preferred_lane=2, cruise_speed=ego_speed, bicycle=bicycle)
    # Turn on intersection logic and provide stop-line coordinates for ego approach:
    # For east/west ego approaches the stop line is at x=0
    # For north/south ego approaches the stop line is at y=0
    bp.intersection_mode = False
    bp.intersection_x = 0.0
    bp.intersection_y = 0.0

    # initially give planner the first perceived set
    perception.step(0.0)
    bp.obstacles = perception.perceived_obstacles
    print("CROSS TRAFFIC:", bp.obstacles)

    ref_provider = ReferenceProvider(bp, horizon_s=5.0)
    mpc = DrakeMPC(reference_provider=ref_provider)

    # === Simulation storage ===
    N = int(tf / dt)
    t_traj = np.linspace(0.0, tf, N + 1)
    x_traj = np.zeros((N + 1, 6))
    u_traj = np.zeros((N, 2))
    M = len(obstacles)
    obstacle_trajs = np.zeros((N + 1, M, 5))  # x, y, v, lane

    x_traj[0] = x0
    for i, o in enumerate(obstacles):
        obstacle_trajs[0, i, :] = [o.x, o.y, o.v, o.lane, o.psi]

    # === Main loop: mirrors run_overtake_sim_ttc but with intersection geometry ===
    for k in range(N):
        x = x_traj[k]

        # 1) Advance perception (low-rate noisy) so planner sees perceived obstacles
        perception.step(dt)
        bp.obstacles = perception.perceived_obstacles

        # 2) Get reference trajectory from ReferenceProvider (calls bp.update internally)
        try:
            ref = ref_provider.get_ref_horizon(x0=x, T=mpc.cfg.T, DT=mpc.cfg.DT)
        except Exception as e:
            # If ref provider crashed (shouldn't normally), fallback to simple straight ref
            # Build a trivial straight reference along ego direction using geom
            xs, ys = geom.straight_path(direction, lane_id=1, N=mpc.cfg.T + 1)
            ref = np.zeros((6, mpc.cfg.T + 1))
            ref[0, :] = xs[:mpc.cfg.T + 1]
            ref[1, :] = ys[:mpc.cfg.T + 1]
            ref[2, :] = np.arctan2(np.diff(ref[1], prepend=ref[1, 0]), np.diff(ref[0], prepend=ref[0, 0]))
            ref[3, :] = bp.cruise_speed
            ref[4, :] = 0.0
            ref[5, :] = 0.0

        # 3) Solve MPC (same pattern as straight-road)
        try:
            u = mpc.solve_mpc(x, ref)
        except Exception as e:
            # MPC failed: fallback like straight-road behavior - gently center into preferred lane
            target_y = geom.lane_center(direction, bp.preferred_lane)
            y_error = target_y - x[1]
            # produce small lateral control to slowly center; longitudinal 0
            u = np.array([0.0, np.clip(0.3 * y_error, -0.3, 0.3)])

        u_traj[k] = u

        # 4) Integrate ego dynamics
        x_traj[k + 1] = x + dt * bicycle.continuous_time_full_dynamics(x, u)

        # 5) Advance true obstacles in the world and log their truth trajectories
        for i, o in enumerate(obstacles):
            o.step(dt)
            obstacle_trajs[k + 1, i, :] = [o.x, o.y, o.v, o.lane, o.psi]

    return x_traj, u_traj, t_traj, obstacle_trajs, dt, N
