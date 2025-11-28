import numpy as np

class IntersectionGeometry:
    """
    4-way intersection with multiple lanes.
    Road width = num_lanes * lane_width
    Lane centers calculated from left edge of road
    """
    def __init__(self, lane_width=5.0, num_lanes=3, road_length=80):
        self.lane_width = lane_width
        self.num_lanes = num_lanes
        self.road_length = road_length
        self.total_width = self.lane_width * self.num_lanes  # full road width

        # Lane centers for each direction
        # If you want symmetric around 0, use -total_width/2 + lane_width/2 etc
        # Here we use left-to-right: 0.5*width, 1.5*width, ...
        offsets = (-self.total_width/2 + self.lane_width/2) + np.arange(self.num_lanes) * self.lane_width

        self.lanes = {
            "east": offsets,
            "west": offsets,
            "north": offsets,
            "south": offsets
        }

    def lane_center(self, direction, lane_id=0):
        """Return center of a given lane (scalar)."""
        return self.lanes[direction][lane_id]

    def lane_lines(self, direction):
        """Return list of (xs, ys) for all lane centerlines."""
        lines = []
        for offset in self.lanes[direction]:
            if direction in ["east", "west"]:
                xs = np.linspace(-self.road_length, self.road_length, 500)
                ys = np.full_like(xs, offset)
                if direction == "west":
                    xs = xs[::-1]
            elif direction in ["north", "south"]:
                ys = np.linspace(-self.road_length, self.road_length, 500)
                xs = np.full_like(ys, offset)
                if direction == "south":
                    ys = ys[::-1]
            lines.append((xs, ys))
        return lines

    def straight_path(self, direction, lane_id=0, N=500):
        """Return centerline path (x, y) for a specific lane."""
        lane_offset = self.lanes[direction][lane_id]
        if direction == "east":
            xs = np.linspace(-self.road_length, self.road_length, N)
            ys = np.full_like(xs, lane_offset)
        elif direction == "west":
            xs = np.linspace(self.road_length, -self.road_length, N)
            ys = np.full_like(xs, lane_offset)
        elif direction == "north":
            ys = np.linspace(-self.road_length, self.road_length, N)
            xs = np.full_like(ys, lane_offset)
        elif direction == "south":
            ys = np.linspace(self.road_length, -self.road_length, N)
            xs = np.full_like(ys, lane_offset)
        else:
            raise ValueError("Unknown direction " + str(direction))
        return xs, ys

    def reference_path(self, direction, lane_id=0, N=500, speed=10.0):
        """6D reference trajectory for the ego car in selected lane."""
        xs, ys = self.straight_path(direction, lane_id=lane_id, N=N+1)
        psi = np.arctan2(np.diff(ys, prepend=ys[0]), np.diff(xs, prepend=xs[0]))
        ref = np.zeros((6, N+1))
        ref[0,:] = xs
        ref[1,:] = ys
        ref[2,:] = psi
        ref[3,:] = speed
        ref[4,:] = 0.0
        ref[5,:] = 0.0
        return ref
