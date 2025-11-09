from dataclasses import dataclass
from typing import Optional

@dataclass
class TrajectoryRequest:
    intent: str                 # "CRUISE" | "LANE_CHANGE_TO"
    target_lane: Optional[int]  # for lane change
    v_ref: float                # desired speed cap (m/s)
    start_in: float = 0.0       # when to start maneuver (s)
    horizon_s: float = 5.0      # planning horizon

class BehaviorPlanner:
    def __init__(self, preferred_lane=1, cruise_speed=20.0, bicycle=None):
        self.preferred_lane = preferred_lane
        self.cruise_speed = cruise_speed
        self.state = "CRUISE"
        self.side = None
        self.bicycle = bicycle # reference to bicycle model if needed

    def update(self, ego_x: float, ego_y: float, ego_vx: float, dt: float):
        # infer current lane by nearest centerline
        lane_idx = round((ego_y - 0.0)/3.7) + 1  # inverse of lane_center_y()
        lane_idx = max(0, min(2, lane_idx))

        if lane_idx != self.preferred_lane:
            self.state = "LANE_CHANGE_TO"
        else:
            self.state = "CRUISE"

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
                                     horizon_s=5.0)
