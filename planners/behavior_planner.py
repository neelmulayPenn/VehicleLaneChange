from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass
class TrajectoryRequest:
    intent: str                 # "CRUISE" | "LANE_CHANGE_TO"
    target_lane: Optional[int]  # for lane change
    v_ref: float                # desired speed cap (m/s)
    start_in: float = 0.0       # relative start time (s)
    horizon_s: float = 5.0      # planning horizon
    
    # New fields for fixed trajectory execution
    lateral_coeffs: Optional[Tuple[float, float, float, float, float, float]] = None
    maneuver_duration: float = 4.0
    maneuver_start_time: float = 0.0 # Absolute simulation time when maneuver started

class BehaviorPlanner:
    def __init__(self, preferred_lane=1, cruise_speed=20.0, bicycle=None):
        self.preferred_lane = preferred_lane
        self.cruise_speed = cruise_speed
        self.state = "CRUISE"
        self.bicycle = bicycle
        
        # Internal state for maneuvers
        self.sim_time = 0.0
        self.maneuver_start_time = 0.0
        self.maneuver_coeffs = None
        self.maneuver_duration = 4.0 # Fixed duration for lane change

    def _get_quintic_coeffs(self, y_start, y_end, T):
        """Helper to calculate quintic polynomial coefficients."""
        a0 = y_start
        a1 = 0.0
        a2 = 0.0
        h = y_end - y_start
        if T <= 0:
            return (a0, a1, a2, 0.0, 0.0, 0.0)
        
        a3 = 10 * h / (T**3)
        a4 = -15 * h / (T**4)
        a5 = 6 * h / (T**5)
        return (a0, a1, a2, a3, a4, a5)

    def update(self, ego_x: float, ego_y: float, ego_vx: float, dt: float):
        # Track internal simulation time
        self.sim_time += dt

        # infer current lane by nearest centerline
        lane_idx = round((ego_y - 0.0)/3.7) + 1  # inverse of lane_center_y()
        lane_idx = max(0, min(2, lane_idx))

        # State Machine Logic
        # Transition: CRUISE -> LANE_CHANGE
        if self.state == "CRUISE":
            if lane_idx != self.preferred_lane:
                self.state = "LANE_CHANGE_TO"
                self.maneuver_start_time = self.sim_time
                
                # Calculate the fixed trajectory once
                y_target = (self.preferred_lane - 1) * 3.7
                self.maneuver_coeffs = self._get_quintic_coeffs(
                    y_start=ego_y, 
                    y_end=y_target, 
                    T=self.maneuver_duration
                )

        # Transition: LANE_CHANGE -> CRUISE
        elif self.state == "LANE_CHANGE_TO":
            if self.sim_time > self.maneuver_start_time + self.maneuver_duration:
                self.state = "CRUISE"
                self.maneuver_coeffs = None # Reset coeffs

        # Output Generation
        if self.state == "CRUISE":
            return TrajectoryRequest(intent="CRUISE",
                                     target_lane=self.preferred_lane,
                                     v_ref=self.cruise_speed,
                                     start_in=0.0,
                                     horizon_s=5.0)
        else:
            return TrajectoryRequest(intent="LANE_CHANGE_TO",
                                     target_lane=self.preferred_lane,
                                     v_ref=self.cruise_speed,
                                     start_in=0.0,
                                     horizon_s=5.0,
                                     lateral_coeffs=self.maneuver_coeffs,
                                     maneuver_duration=self.maneuver_duration,
                                     maneuver_start_time=self.maneuver_start_time)