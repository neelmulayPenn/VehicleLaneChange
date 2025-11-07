import numpy as np
import matplotlib as plt
from scipy.integrate import solve_ivp

class VehicleBicycleModel:
    def __init__(self, m, Iz, Lf, Lr, Cf, Cr, d_limit):
        # m  : float  - vehicle mass [kg]
        # Iz : float  - yaw moment of inertia [kg*m^2]
        # Lf : float  - distance from CG to front axle [m]
        # Lr : float  - distance from CG to rear axle [m]
        # Cf : float  - front tire cornering stiffness [N/rad]
        # Cr : float  - rear tire cornering stiffness [N/rad]
        # d_limit : float - steering angle limit either direction [rad]
        
        self.m = m
        self.Iz = Iz
        self.Lf = Lf
        self.Lr = Lr
        self.Cf = Cf
        self.Cr = Cr
        self.d_limit = d_limit

    def continuous_time_full_dynamics(self, x, u):
        X, Y, psi, vx, vy, w = x # x_position, y_position, yaw angle, long. vel, lateral. vel, yaw rate
        delta, ax = u # steering angle, long. accel

        # vehicle parameters
        m = self.m
        Iz = self.Iz
        Lf = self.Lf
        Lr = self.Lr
        Cf = self.Cf
        Cr = self.Cr
        d_limit = self.d_limit

        # no divide by 0
        vx = np.sign(vx) * max(abs(vx), 1e-3)

        # small angle approx tire slip angles
        alpha_f = np.arctan2(vy + Lf * w, vx) - delta
        alpha_r = np.arctan2(vy - Lr * w, vx)

        # lateral force
        Fyf = -Cf * alpha_f
        Fyr = -Cr * alpha_r

        # longitudinal force
        Fxf = 0 #0.5 * m * ax #commented out for rear wheel drive
        Fxr = 0.5 * m * ax

        # dynamics
        Xdot = vx * np.cos(psi) - vy * np.sin(psi)
        Ydot = vx * np.sin(psi) + vy * np.cos(psi)
        psidot = w
        vxdot = ax + w * vy + (1/m) * (Fxf * np.cos(delta) - Fyf * np.sin(delta) + Fxr)
        vydot = -w * vx + (1/m) * (Fxf * np.sin(delta) + Fyf * np.cos(delta) + Fyr)
        wdot = (1/Iz) * (Lf * (Fyf * np.cos(delta) + Fxf * np.sin(delta)) - Lr * Fyr)

        xdot = np.array([Xdot, Ydot, psidot, vxdot, vydot, wdot])
        return xdot
    
    def continuous_time_linearized_dynamics(self, x0, u0, eps=1e-5):
        """
        Numerically linearize the nonlinear dynamics about (x0, u0).

        Args:
            x0 : np.array (6,)  - state
            u0 : np.array (2,)  - input
            eps : float - finite difference step

        Returns:
            A : (6,6) continuous-time Jacobian wrt x
            B : (6,2) continuous-time Jacobian wrt u
        """
        n = len(x0)
        m = len(u0)

        f0 = self.continuous_time_full_dynamics(x0, u0)
        A = np.zeros((n, n))
        B = np.zeros((n, m))

        # df/dx_i = A
        for i in range(n):
            dx = np.zeros_like(x0)
            dx[i] = eps
            f_plus = self.continuous_time_full_dynamics(x0 + dx, u0)
            f_minus = self.continuous_time_full_dynamics(x0 - dx, u0)
            A[:, i] = (f_plus - f_minus) / (2 * eps)

        # df/du_i = B
        for j in range(m):
            du = np.zeros_like(u0)
            du[j] = eps
            f_plus = self.continuous_time_full_dynamics(x0, u0 + du)
            f_minus = self.continuous_time_full_dynamics(x0, u0 - du)
            B[:, j] = (f_plus - f_minus) / (2 * eps)

        return A, B

    def discrete_time_linearized_dynamics(self, x0, u0, T, eps=1e-5):
        # euler integration discrete linearized dynamics
        A_c, B_c = self.continuous_time_linearized_dynamics(x0, u0, eps)
        A_d = np.eye(A_c.shape[0]) + T * A_c
        B_d = T * B_c
        return A_d, B_d