import numpy as np
from math import sin, cos, pi
from scipy.integrate import solve_ivp
from bicycle_model_dynamics import VehicleBicycleModel
import matplotlib.pyplot as plt

def simulate_bicycle_model(x0, tf, bicycle):
    t0 = 0.0 
    n_points = 1000

    dt = 1e-2

    x = [x0]
    u = [np.zeros((2,))]
    t = [t0]

    while np.linalg.norm(np.array(x[-1][0:2])) > 1e-3 and t[-1] < tf:
        current_time = t[-1]
        current_x = x[-1]
        current_u_command = np.zeros(2)
        
        #TODO:compute control input u = (steer angle, longitudinal accel)
        current_u_real = 0

        #apply control input limits
        current_d_real = np.clip(current_u_command[0], -bicycle.d_limit, bicycle.d_limit)
        
        # Autonomous ODE for constant inputs to work with solve_ivp
        def f(t, x):
            return bicycle.continuous_time_full_dynamics(current_x, current_u_real)
        # Integrate one step
        sol = solve_ivp(f, (0, dt), current_x, first_step=dt)

        # Record time, state, and inputs
        t.append(t[-1] + dt)
        x.append(sol.y[:, -1])
        u.append(current_u_command)

    x = np.array(x)
    u = np.array(u)
    t = np.array(t)
    return x, u, t