import numpy as np
from typing import Callable, Tuple, Optional
from planners.env import lane_center_y, LANE_WIDTH, ROAD_HEADING

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
                      T_lc: float = 4.0,
                      lateral_coeffs: Optional[Tuple[float, float, float, float, float, float]] = None,
                      t_current: float = 0.0,
                      maneuver_start_time: float = 0.0
                      ) -> Tuple[Callable[[float], Tuple[float,float,float,float]], float]:
    """
    Returns ref(t) -> (x_ref, y_ref, psi_ref, v_ref), and suggested dt for sampling.
    Now supports executing a fixed trajectory passed via lateral_coeffs.
    """
    X,Y,psi,vx,vy,w = ego_state  # aligns with your model state

    # Suggest a sampling time based on horizon
    dt_suggest = min(0.1, max(0.02, horizon_s/200.0))

    # Compute current and target lateral centers
    y_cur = Y
    y_tar = lane_center_y(target_lane, y0=0.0)

    if intent == "CRUISE":
        # centerline of target lane, straight road, heading = ROAD_HEADING
        def ref(t):
            s = v_ref * t
            x = X + s*np.cos(ROAD_HEADING)
            y = y_tar + s*np.sin(ROAD_HEADING)  # straight road; keeps y near lane center
            psi_r = ROAD_HEADING
            return x, y, psi_r, v_ref
        return ref, dt_suggest

    elif intent == "LANE_CHANGE_TO":
        # Check if we have a pre-calculated trajectory to follow
        if lateral_coeffs is not None:
            a0, a1, a2, a3, a4, a5 = lateral_coeffs
            
            # Calculate how far into the maneuver we are
            t_elapsed = t_current - maneuver_start_time

            def d_of_t_and_deriv(t_plan):
                # t_plan is relative time from now (0 to horizon)
                # tau is the absolute time along the maneuver curve
                tau = t_plan + t_elapsed
                
                # Clip tau so we don't extrapolate past the maneuver end
                tt = np.clip(tau, 0.0, T_lc)
                
                # Position
                pos = a0 + a1*tt + a2*tt**2 + a3*tt**3 + a4*tt**4 + a5*tt**5
                
                # Derivative (dy/dt) for heading calculation
                # Only valid if we are within the maneuver duration
                if 0.0 <= tau <= T_lc:
                    vel = a1 + 2*a2*tt + 3*a3*tt**2 + 4*a4*tt**3 + 5*a5*tt**4
                else:
                    vel = 0.0
                return pos, vel

            def ref(t):
                s = v_ref * t
                x_c = X + s*np.cos(ROAD_HEADING)
                
                y_c, dy_c = d_of_t_and_deriv(t)
                
                # Compute heading: psi = atan(dy/dx) + ROAD_HEADING
                # dy/dx = (dy/dt) / (dx/dt) ~= dy_c / v_ref
                # This aligns the car with the lane change path
                psi_r = np.arctan2(dy_c, v_ref) + ROAD_HEADING
                
                return x_c, y_c, psi_r, v_ref

            return ref, dt_suggest

        else:
            # Fallback for dynamic/stateless lane change (old behavior)
            # Recalculates from current position every step
            a0,a1,a2,a3,a4,a5 = _quintic_coeff(y_cur, y_tar, T_lc)

            def d_of_t(t):
                tt = np.clip(t, 0.0, T_lc)
                return a0 + a1*tt + a2*tt**2 + a3*tt**3 + a4*tt**4 + a5*tt**5

            def ref(t):
                s = v_ref * t
                x_c = X + s*np.cos(ROAD_HEADING)
                y_c = d_of_t(t)
                psi_r = ROAD_HEADING
                return x_c, y_c, psi_r, v_ref

            return ref, dt_suggest

    else:
        # Default
        def ref(t):
            s = v_ref * t
            x = X + s*np.cos(ROAD_HEADING)
            y = y_tar
            psi_r = ROAD_HEADING
            return x, y, psi_r, v_ref
        return ref, dt_suggest