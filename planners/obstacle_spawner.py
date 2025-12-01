from __future__ import annotations
from typing import Iterable, List, Optional, Sequence
import numpy as np
from planners.env import LANE_IDS, OtherVehicle, lane_center_y


def spawn_random_obstacles(
    num_obstacles: int = 3,
    lanes: Optional[Sequence[int]] = None,
    x_range: tuple[float, float] = (10.0, 30.0),
    speed_range: tuple[float, float] = (5.0, 20.0),
    min_gap: float = 8.0,
    seed: Optional[int] = None,
    ego_x: float = 0.0,
    min_front_offset: float = 5.0,
    guarantee_lane: Optional[int] = None,
) -> List[OtherVehicle]:
    """Generate a list of `OtherVehicle` instances with randomized states.

    Parameters
    ----------
    num_obstacles:
        Total number of vehicles to spawn.
    lanes:
        Subset of lane IDs to place traffic in. Defaults to all lanes in `planners.env.LANE_IDS`.
    x_range:
        Tuple specifying the uniform range for the longitudinal spawn positions (meters).
    speed_range:
        Tuple specifying the uniform range of obstacle speeds (m/s).
    min_gap:
        Minimum longitudinal spacing enforced within each lane to avoid overlapping vehicles.
    seed:
        Optional RNG seed for reproducible spawns.
    ego_x:
        Ego vehicle x-position; used to guarantee at least one lead obstacle.
    min_front_offset:
        Minimum distance in front of ego where the guaranteed obstacle should appear.
    guarantee_lane:
        Optional lane index that must contain a vehicle ahead of ego.
    """

    if num_obstacles <= 0:
        return []

    lane_choices = list(lanes) if lanes is not None else list(LANE_IDS)
    if not lane_choices:
        raise ValueError("At least one lane must be provided to spawn obstacles.")

    if x_range[0] >= x_range[1]:
        raise ValueError("x_range must be an increasing interval.")
    if speed_range[0] >= speed_range[1]:
        raise ValueError("speed_range must be an increasing interval.")

    rng = np.random.default_rng(seed)
    lane_positions: dict[int, List[float]] = {lane: [] for lane in lane_choices}
    obstacles: List[OtherVehicle] = []

    lane_sequence: List[int]
    if num_obstacles >= len(lane_choices):
        lane_sequence = list(lane_choices)
        remaining = num_obstacles - len(lane_choices)
        if remaining > 0:
            lane_sequence.extend(rng.choice(lane_choices, size=remaining, replace=True))
        rng.shuffle(lane_sequence)
    else:
        lane_sequence = list(rng.choice(lane_choices, size=num_obstacles, replace=True))

    max_attempts_per_vehicle = 20
    for lane in lane_sequence:
        attempts = 0
        placed = False
        while attempts < max_attempts_per_vehicle and not placed:
            x = float(rng.uniform(*x_range))
            if all(abs(x - existing_x) >= min_gap for existing_x in lane_positions[lane]):
                v = float(rng.uniform(*speed_range))
                y = lane_center_y(lane)
                obstacles.append(OtherVehicle(x=x, y=y, v=v, lane=lane))
                lane_positions[lane].append(x)
                placed = True
            attempts += 1
        # If we fail to place after attempts, skip to avoid infinite loop

    ahead_threshold = ego_x + min_front_offset
    needs_lane = guarantee_lane if guarantee_lane is not None else None
    def _place_specific_lane(target_lane: int):
        y = lane_center_y(target_lane)
        x_target = max(ahead_threshold, x_range[0])
        x = float(min(x_target, x_range[1]))
        v = float(rng.uniform(*speed_range))
        existing = lane_positions[target_lane]
        if existing:
            for _ in range(5):
                gaps = [abs(x - ex) for ex in existing]
                if gaps and min(gaps) < min_gap:
                    x = min(x_range[1], max(x_range[0], x + min_gap))
                else:
                    break
        obstacles.append(OtherVehicle(x=x, y=y, v=v, lane=target_lane))
        lane_positions[target_lane].append(x)

    if needs_lane is not None and needs_lane not in lane_positions:
        lane_positions[needs_lane] = []
        lane_choices.append(needs_lane)

    if needs_lane is not None:
        if not any((obs.x > ahead_threshold) and (obs.lane == needs_lane) for obs in obstacles):
            _place_specific_lane(needs_lane)
    elif not any(obs.x > ahead_threshold for obs in obstacles):
        _place_specific_lane(int(rng.choice(lane_choices)))

    return obstacles
