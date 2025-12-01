# planners/intersection_ref_provider.py
import numpy as np
from planners.intersection_env import IntersectionGeometry

class IntersectionReferenceProvider:
    """
    Provides 6D reference trajectory through a 4-way intersection.
    Supports:
    - straight-through
    - future: right-turn, left-turn
    """

    def __init__(self, geometry: IntersectionGeometry, direction="east", lane_id=1, speed=10.0):
        self.geometry = geometry
        self.direction = direction
        self.lane_id = lane_id
        self.speed = speed

        self.path_x, self.path_y = geometry.straight_path(direction, lane_id, N=500)

    def get_ref_horizon(self, x0, T, dt, traj_req=None):
        """
        Returns reference of shape (6, T+1):
        [x, y, psi, vx, vy, yaw_rate]
        """
        steps = T + 1
        ref = np.zeros((6, steps))

        # Truncate or pad path to T+1
        idx0 = np.argmin((self.path_x - x0[0])**2 + (self.path_y - x0[1])**2)
        end = idx0 + steps
        if end >= len(self.path_x):
            end = len(self.path_x)-1

        xs = self.path_x[idx0:end]
        ys = self.path_y[idx0:end]

        if len(xs) < steps:
            xs = np.pad(xs, (0, steps-len(xs)), constant_values=xs[-1])
            ys = np.pad(ys, (0, steps-len(ys)), constant_values=ys[-1])

        # Apply lateral lane-change if requested
        if traj_req is not None and traj_req.lateral_coeffs is not None:
            a0, a1, a2, a3, a4, a5 = traj_req.lateral_coeffs
            t_local = np.arange(steps) * dt - traj_req.maneuver_start_time
            # clamp t_local to [0, maneuver_duration]
            t_local = np.clip(t_local, 0.0, traj_req.maneuver_duration)
            lateral_offset = a0 + a1*t_local + a2*t_local**2 + a3*t_local**3 + a4*t_local**4 + a5*t_local**5
            ys += lateral_offset

        if traj_req is not None:
            self.speed = traj_req.v_ref

        # Compute heading
        psi = np.arctan2(np.diff(ys, prepend=ys[0]), np.diff(xs, prepend=xs[0]))

        # Fill reference state
        ref[0,:] = xs
        ref[1,:] = ys
        ref[2,:] = psi
        ref[3,:] = self.speed   # vx
        ref[4,:] = 0.0          # vy
        ref[5,:] = 0.0          # yaw rate

        return ref
