import numpy as np
from dataclasses import dataclass

LANE_WIDTH = 3.7
LANE_IDS = [0, 1, 2]  # bottom (0), middle (1), top (2)
ROAD_HEADING = 0.0    # straight along +X

def lane_center_y(lane_id: int, y0=0.0):
    # Lane 1 centered at y0; lane 0 below, lane 2 above
    return y0 + (lane_id - 1) * LANE_WIDTH

@dataclass
class OtherVehicle:
    x: float
    y: float
    v: float  # along +X (road aligned)
    lane: int

    def step(self, dt: float):
        self.x += self.v * dt
        # keep straight; extend as needed

class World:
    def __init__(self, moving_traffic_lanes=(0,2), y0=0.0):
        self.traffic = []
        # Spawn a few cars in lane 0 and 2
        for lane in moving_traffic_lanes:
            yc = lane_center_y(lane, y0)
            # staggered starting positions and speeds
            for i in range(3):
                self.traffic.append(OtherVehicle(x=30+40*i, y=yc, v=14+2*i, lane=lane))

    def step(self, dt: float):
        for car in self.traffic:
            car.step(dt)
