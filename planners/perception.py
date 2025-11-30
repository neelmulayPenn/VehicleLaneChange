import numpy as np
from planners.env import OtherVehicle

class NoisyPerception:
    """
    Wraps a list of true OtherVehicle objects and exposes a 'perceived'
    version with:
      - Lower update rate (perception_dt > sim dt)
      - Additive Gaussian noise on x, y
    Use perceived obstacles for ObstacleAwarePlanner instead of the true ones.
    """

    def __init__(self, obstacles, perception_dt=0.2,
                 sigma_x=0.5, sigma_y=0.3, rng=None):
        """
        obstacles: list[OtherVehicle]  (the true world obstacles)
        perception_dt: seconds between perception updates
        sigma_x, sigma_y: std dev of noise in meters
        rng: np.random.Generator or None
        """
        self.true_obstacles = obstacles
        self.perception_dt = perception_dt
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.rng = rng or np.random.default_rng()

        self._time_since_update = 0.0
        # Start with a perfect measurement
        self._perceived = [self._copy_with_noise(o, noise=False)
                           for o in self.true_obstacles]

    def _copy_with_noise(self, obs: OtherVehicle, noise: bool = True):
        dx = self.rng.normal(0.0, self.sigma_x) if noise else 0.0
        dy = self.rng.normal(0.0, self.sigma_y) if noise else 0.0
        return OtherVehicle(
            x=obs.x + dx,
            y=obs.y + dy,
            v=obs.v,
            lane=obs.lane
        )

    def step(self, dt: float):
        """
        Advance perception time. Only when accumulated time exceeds
        perception_dt do we refresh the perceived obstacle list.
        """
        self._time_since_update += dt
        if self._time_since_update >= self.perception_dt:
            self._time_since_update = 0.0
            # Re‑sense all obstacles with fresh noise
            self._perceived = [
                self._copy_with_noise(o, noise=True)
                for o in self.true_obstacles
            ]

    @property
    def perceived_obstacles(self):
        """
        Returns the last perceived obstacle list (low‑rate + noisy).
        """
        return self._perceived