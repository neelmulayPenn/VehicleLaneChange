import numpy as np
from dataclasses import dataclass
from scipy.interpolate import CubicSpline

LANE_WIDTH = 3.7
LANE_IDS = [0, 1, 2]  # left (0), middle (1), right (2)
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


class RoadGeometry:
    def __init__(self, xs, ys):
        s = np.zeros(len(xs))
        for i in range(1, len(xs)):
            ds = np.hypot(xs[i]-xs[i-1], ys[i]-ys[i-1])
            s[i] = s[i-1] + ds
        self.s = s
        self.cx = CubicSpline(s, xs)
        self.cy = CubicSpline(s, ys)
        self.length = s[-1]

    def center(self, s):
        return np.array([self.cx(s), self.cy(s)])

    def tangent(self, s):
        dx = self.cx.derivative()(s)
        dy = self.cy.derivative()(s)
        t = np.array([dx, dy])
        return t / np.linalg.norm(t)

    def normal(self, s):
        t = self.tangent(s)
        return np.array([-t[1], t[0]])
    
def lane_offset(lane_id, num_lanes):
    center_index = (num_lanes - 1) / 2.0
    return (lane_id - center_index) * LANE_WIDTH

def lane_xy(road: RoadGeometry, s, lane_id, num_lanes):
    offset = lane_offset(lane_id, num_lanes)
    C = road.center(s)
    N = road.normal(s)
    return C + offset * N

from dataclasses import dataclass

@dataclass
class OtherVehicle_curves:
    x: float
    y: float
    v: float
    lane: int
    s: float          # internal arc-length along the road
    road: RoadGeometry
    num_lanes: int

    def update_xy(self):
        self.x, self.y = lane_xy(self.road, self.s, self.lane, self.num_lanes)
        tangent = self.road.tangent(self.s)
        self.psi = np.arctan2(tangent[1], tangent[0])
        
    def step(self, dt):
        self.s += self.v * dt
        self.s %= self.road.length   # wrap / clamp
        self.update_xy()

class World_curves:
    def __init__(self, road, num_lanes=3, moving_lanes=None):
        self.road = road
        self.num_lanes = num_lanes
        self.traffic = []

        # e.g. lanes = [0, 2] or [1,3,4] or range(num_lanes)
        if moving_lanes is None:
            moving_lanes = [0, num_lanes-1]  # bottom + top

        for lane in moving_lanes:
            for i in range(3):
                s0 = 30 + 40*i
                v = 14 + 2*i
                x, y = lane_xy(road, s0, lane, num_lanes)
                self.traffic.append(
                    OtherVehicle_curves(x=x, y=y, v=v,
                                 lane=lane, s=s0,
                                 road=road, num_lanes=num_lanes)
                )

    def step(self, dt):
        for car in self.traffic:
            car.step(dt)



