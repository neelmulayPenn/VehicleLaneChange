import numpy as np
from math import copysign
from planners.env import lane_center_y
from planners.behavior_planner import BehaviorPlanner
from planners.env import OtherVehicle
from planners.perception import NoisyPerception
from controllers.reference_provider import ReferenceProvider
from controllers.mpc_controller import DrakeMPC
from Bicycle_Model.bicycle_model_dynamics import VehicleBicycleModel


class ObstacleAwarePlanner(BehaviorPlanner):
    """
    Behavior planner wrapper that supports multiple obstacles:
    - Triggers lane-change using distance and TTC thresholds
    - Slows to match obstacle speed when boxed in
    - Adds hysteresis + commitment to avoid bouncing between lanes
    """

    def __init__(self, preferred_lane=1, cruise_speed=20.0, bicycle=None):
        super().__init__(preferred_lane, cruise_speed, bicycle)
        self.obstacles = []  # list[OtherVehicle]

        # Safety / decision thresholds
        self.safety_distance = 25.0      # meters (earlier trigger)
        self.ttc_threshold = 2.0         # seconds (time-to-collision)
        self.slow_down_distance = 30.0   # follow distance for speed matching

        # Hysteresis / commitment
        self.default_cruise = cruise_speed
        self.lane_change_committed = False    # True once we decide to change
        self.current_target_lane = None       # Lane we are moving toward
        self.eval_candidate_lane = None       # Lane currently being evaluated
        self.eval_counter = 0                 # How many consecutive frames it stayed best
        self.required_stability_steps = 3     # frames of consistent best lane before commit
        self.post_change_cooldown = 0.5       # seconds after change before considering next
        self.cooldown_timer = 0.0

        #intersection parameters
        
        self.intersection_mode = True    # default OFF
        self.intersection_x = None        # x-position of stop line
        self.cross_traffic = []           # vehicles from N/S or W/E
        self.safety_gap = 0.01             # seconds additional buffer

    def _y_to_lane(self, y):
        centers = [lane_center_y(0), lane_center_y(1), lane_center_y(2)]
        diffs = np.abs(np.array(centers) - y)
        return int(np.argmin(diffs))

    def _nearest_ahead_per_lane(self, ego_x, ego_vx):
        lane_info = {
            0: dict(dx=np.inf, v=None, ttc=np.inf),
            1: dict(dx=np.inf, v=None, ttc=np.inf),
            2: dict(dx=np.inf, v=None, ttc=np.inf),
        }
        for obs in (self.obstacles or []):
            lane = self._y_to_lane(obs.y)
            dx = obs.x - ego_x
            if dx <= 0:
                continue  # behind or at same x
            v_rel = ego_vx - obs.v
            ttc = dx / v_rel if v_rel > 0.0 else np.inf
            if dx < lane_info[lane]['dx']:
                lane_info[lane] = dict(dx=dx, v=obs.v, ttc=ttc)
        return lane_info

    def _score_lane(self, info, lane_id):
        # Larger dx and TTC are better; slight bias for center lane
        dx_score = (info['dx'] if np.isfinite(info['dx']) else 1e6) * 0.5
        ttc_score = (info['ttc'] if np.isfinite(info['ttc']) else 1e3) * 1.0
        center_bonus = 5.0 if lane_id == 1 else 0.0
        return dx_score + ttc_score + center_bonus

    def _select_best_adjacent_lane(self, current_lane, lane_info):
        candidates = [l for l in [current_lane - 1, current_lane + 1] if 0 <= l <= 2]
        if not candidates:
            return None
        best_lane = None
        best_score = -np.inf
        for l in candidates:
            score = self._score_lane(lane_info[l], l)
            if score > best_score:
                best_score = score
                best_lane = l
        return best_lane

    def update(self, ego_x: float, ego_y: float, ego_vx: float, dt: float):
        # Baseline: reset cruise speed each step and apply following logic
        self.cruise_speed = self.default_cruise
        # Intersection: override speed if unsafe
        if self.intersection_mode:
            unsafe = self._intersection_conflict(ego_x, ego_vx)
            if unsafe:
                self.cruise_speed = 0.0   # STOP

        # Cooldown after a completed lane change
        if self.cooldown_timer > 0.0:
            self.cooldown_timer = max(0.0, self.cooldown_timer - dt)

        current_lane = self._y_to_lane(ego_y)
        lane_info = self._nearest_ahead_per_lane(ego_x, ego_vx)
        cur = lane_info[current_lane]

        # Following behavior: start matching obstacle speed if too close
        if cur['dx'] < self.slow_down_distance and cur['v'] is not None and (ego_vx > cur['v']):
            self.cruise_speed = min(self.cruise_speed, max(cur['v'] - 0.5, 0.0))

        # If we are already committed to a lane change, do not re‑evaluate.
        if self.lane_change_committed:
            # Let base BehaviorPlanner execute the maneuver toward preferred_lane.
            # Consider the lane change complete once the inferred lane matches target.
            if current_lane == self.current_target_lane:
                self.lane_change_committed = False
                self.current_target_lane = None
                self.eval_candidate_lane = None
                self.eval_counter = 0
                self.cooldown_timer = self.post_change_cooldown
            return super().update(ego_x, ego_y, ego_vx, dt)

        # Hazard in current lane?
        hazard = (cur['dx'] < self.safety_distance) or (cur['ttc'] < self.ttc_threshold)

        # Only consider new lane change if no cooldown
        if hazard and self.cooldown_timer <= 0.0:
            best_lane = self._select_best_adjacent_lane(current_lane, lane_info)

            if best_lane is not None and best_lane != current_lane:
                # Hysteresis: require the same best lane for several consecutive steps
                if best_lane != self.eval_candidate_lane:
                    # New candidate lane: reset counter
                    self.eval_candidate_lane = best_lane
                    self.eval_counter = 1
                else:
                    self.eval_counter += 1

                # Once stable enough, commit to lane change
                if self.eval_counter >= self.required_stability_steps:
                    self.preferred_lane = best_lane
                    self.current_target_lane = best_lane
                    self.lane_change_committed = True
            else:
                # No good adjacent lane or same lane; reset evaluation
                self.eval_candidate_lane = None
                self.eval_counter = 0
        else:
            # No hazard or still cooling down: reset evaluation
            self.eval_candidate_lane = None
            self.eval_counter = 0

        # Call base BehaviorPlanner to generate TrajectoryRequest toward preferred_lane
        return super().update(ego_x, ego_y, ego_vx, dt)
    
    def _intersection_conflict(self, ego_x, ego_vx):
        if not self.intersection_mode:
            return False

        # Ego time-to-intersection
        dx = self.intersection_x - ego_x
        if dx <= 0:
            print(f"[DEBUG] Ego already at/past intersection: ego_x={ego_x}, intersection_x={self.intersection_x}")
            return False  # already at or past intersection

        TTC_ego = dx / max(ego_vx, 0.1)
        print(f"[DEBUG] Ego position: x={ego_x}, velocity={ego_vx}, TTC_ego={TTC_ego}")

        # Check all cross-traffic vehicles
        for i, veh in enumerate(self.cross_traffic):
            x_cross, y_cross, v_cross = veh
            dy_cross = self.intersection_y - y_cross  # distance along y to stop line
            if dy_cross <= 0:
                print(f"[DEBUG] Cross vehicle {i} already past intersection: y_cross={y_cross}")
                continue  # already past intersection
            ttc_cross = dy_cross / max(v_cross, 0.1)
            print(f"[DEBUG] Cross vehicle {i}: y={y_cross}, v={v_cross}, dy={dy_cross}, TTC_cross={ttc_cross}")

            # Unsafe if cross traffic will reach intersection sooner than ego + safety gap
            if ttc_cross < TTC_ego + self.safety_gap:
                print(f"[DEBUG] Intersection UNSAFE due to vehicle {i}: TTC_cross={ttc_cross}, TTC_ego+gap={TTC_ego+self.safety_gap}")
                return True

        print("[DEBUG] Intersection SAFE")
        return False



def run_overtake_sim_ttc(
    tf: float = 20.0,
    dt: float = 0.05,
    ego_speed: float = 15.0,
    obstacles_init=None,
):
    """
    Runs the overtaking scenario with multiple obstacles and TTC-based lane-change.
    Returns: x_traj, u_traj, t_traj, obstacle_trajs (N+1, M, 4), dt, N
    """
    # Vehicle model params
    m = 1000
    Iz = 2500
    Lf = 1.1
    Lr = 1.6
    Cf = 80000
    Cr = 80000
    d_limit = np.deg2rad(30)

    bicycle = VehicleBicycleModel(m, Iz, Lf, Lr, Cf, Cr, d_limit)

    # Initial state: ego behind obstacles
    x0 = np.array([0.0, lane_center_y(1), 0.0, ego_speed, 0.0, 0.0])

    # Obstacles across all lanes (default set if none provided)
    if obstacles_init is None:
        obstacles_init = [
            OtherVehicle(x=20.0, y=lane_center_y(0), v=10.0, lane=0),
            OtherVehicle(x=30.0, y=lane_center_y(1), v=6.0, lane=1),
            OtherVehicle(x=15.0, y=lane_center_y(2), v=8.0, lane=2),
        ]
    obstacles = [OtherVehicle(x=o.x, y=o.y, v=o.v, lane=o.lane) for o in obstacles_init]

    perception = NoisyPerception(
        obstacles,
        perception_dt=0.2,
        sigma_x=0.5,
        sigma_y=0.3,
    )

    # Planner and MPC
    bp = ObstacleAwarePlanner(preferred_lane=0, cruise_speed=ego_speed, bicycle=bicycle)
    bp.obstacles = perception.perceived_obstacles
    ref_provider = ReferenceProvider(bp, horizon_s=5.0)
    mpc = DrakeMPC(reference_provider=ref_provider)

    N = int(tf / dt)
    t_traj = np.linspace(0.0, tf, N + 1)
    x_traj = np.zeros((N + 1, 6))
    u_traj = np.zeros((N, 2))
    M = len(obstacles)
    obstacle_trajs = np.zeros((N + 1, M, 4))  # x, y, v, lane per obstacle

    x_traj[0] = x0
    for i, o in enumerate(obstacles):
        obstacle_trajs[0, i, :] = [o.x, o.y, o.v, o.lane]

    for k in range(N):
        x = x_traj[k]

        # Control via MPC (fallback: gentle centering accel)
        try:
            ref = ref_provider.get_ref_horizon(x0=x, T=mpc.cfg.T, DT=mpc.cfg.DT)
            u = mpc.solve_mpc(x, ref)
            
        except Exception as e:
            print("MPC FAILED:", e)
            target_lane_y = lane_center_y(bp.preferred_lane)
            y_error = target_lane_y - x[1]
            u = np.array([0.0, np.clip(0.3 * y_error, -0.3, 0.3)])

        u_traj[k] = u

        # Euler integrate
        xdot = bicycle.continuous_time_full_dynamics(x, u)
        x_next = x + dt * xdot
        x_traj[k + 1] = x_next

        # Advance obstacles (true world)
        for i, o in enumerate(obstacles):
            o.step(dt)
            obstacle_trajs[k + 1, i, :] = [o.x, o.y, o.v, o.lane]

        # Update noisy, low‑rate perception after world moves
        perception.step(dt)
        bp.obstacles = perception.perceived_obstacles
        bp.cross_traffic = [
            (obs.x, obs.y, obs.v)
            for obs in perception.perceived_obstacles
            if obs.direction in ("north", "south")  # directions that cross ego's path
        ]
        print("CROSS TRAFFIC:", bp.cross_traffic)

    return x_traj, u_traj, t_traj, obstacle_trajs, dt, N


