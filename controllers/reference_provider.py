import numpy as np
from trajectory_planner import make_ref_function
from behavior_planner import BehaviorPlanner


class ReferenceProvider:
    """
    Bridges BehaviorPlanner + make_ref_function with MPC.
    """

    def __init__(self, behavior_planner: BehaviorPlanner, horizon_s=5.0):
        self.bp = behavior_planner
        self.horizon_s = horizon_s

    def get_ref_horizon(self, x0: np.ndarray, T: int, DT: float):
        """
        Returns a reference trajectory of shape (6, T+1) for MPC.
        """
        X, Y, psi, vx, vy, w = x0

        # 1. Behavior planner decides CRUISE or LANE_CHANGE
        req = self.bp.update(X, Y, vx, DT)

        # 2. Build a reference function ref(t)
        ref_func, _ = make_ref_function(
            intent=req.intent,
            ego_state=x0,
            v_ref=req.v_ref,
            target_lane=req.target_lane,
            horizon_s=req.horizon_s,
        )

        # 3. Sample it into a horizon
        ref = np.zeros((6, T+1))

        for k in range(T+1):
            t = k * DT
            xr, yr, psir, vref = ref_func(t)

            ref[0, k] = xr
            ref[1, k] = yr
            ref[2, k] = psir
            ref[3, k] = vref    # vx_ref
            ref[4, k] = 0.0     # vy_ref (no lateral velocity reference)
            ref[5, k] = 0.0     # w_ref  (no yaw rate reference)

        return ref
