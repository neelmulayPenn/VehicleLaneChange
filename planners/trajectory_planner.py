import numpy as np
from typing import Callable, Tuple
from env import lane_center_y, LANE_WIDTH, ROAD_HEADING

def _quintic_coeff(d0, dT, T):
    # d(0)=d0, d'(0)=0, d''(0)=0; d(T)=dT, d'(T)=0, d''(T)=0
    a0 = d0; a1 = 0.0; a2 = 0.0
    a3 = 10*(dT - d0)/(T**3)
    a4 = -15*(dT - d0)/(T**4)
    a5 = 6*(dT - d0)/(T**5)
    return a0,a1,a2,a3,a4,a5

def make_ref_function(intent: str,
                      ego_state: np.ndarray,
                      v_ref: float,
                      target_lane: int,
                      horizon_s: float,
                      T_lc: float = 4.0) -> Tuple[Callable[[float], Tuple[float,float,float,float]], float]:
    """
    Returns ref(t) -> (x_ref, y_ref, psi_ref, v_ref), and suggested dt for sampling.
    """
    X,Y,psi,vx,vy,w = ego_state  # aligns with your model state

    # Compute current and target lateral centers
    y_cur = Y
    y_tar = lane_center_y(target_lane, y0=0.0)

    if intent == "CRUISE" or abs(y_tar - y_cur) < 0.1:
        # centerline of target lane, straight road, heading = ROAD_HEADING
        def ref(t):
            s = v_ref * t
            x = X + s*np.cos(ROAD_HEADING)
            y = y_tar + s*np.sin(ROAD_HEADING)  # straight road; keeps y near lane center
            psi_r = ROAD_HEADING
            return x, y, psi_r, v_ref
        return ref, dt_suggest

    elif intent == "LANE_CHANGE_TO":
        # lateral quintic over T_lc
        a0,a1,a2,a3,a4,a5 = _quintic_coeff(y_cur, y_tar, T_lc)

        def d_of_t(t):
            tt = np.clip(t, 0.0, T_lc)
            return a0 + a1*tt + a2*tt**2 + a3*tt**3 + a4*tt**4 + a5*tt**5

        def ref(t):
            s = v_ref * t
            x_c = X + s*np.cos(ROAD_HEADING)
            y_c = d_of_t(t)  # directly treat global Y as lateral for a straight road
            psi_r = ROAD_HEADING
            return x_c, y_c, psi_r, v_ref

        return ref, dt_suggest

    else:
        # default to cruise
        def ref(t):
            s = v_ref * t
            x = X + s*np.cos(ROAD_HEADING)
            y = y_tar
            psi_r = ROAD_HEADING
            return x, y, psi_r, v_ref
        return ref, dt_suggest
