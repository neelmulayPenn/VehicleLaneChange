#!/usr/bin/env python3
import numpy as np
from pydrake.systems.framework import LeafSystem, BasicVector
from pydrake.solvers import MathematicalProgram, SnoptSolver


class DrakeMPCConfig:
    NX = 4  # [x, y, v, yaw]
    NU = 2  # [a, delta]
    T = 15  # horizon
    DT = 0.1

    # vehicle params (must match bicycle model: Lf=1.2, Lr=1.2)
    WB = 2.4  # wheelbase = Lf + Lr
    MAX_STEER = np.deg2rad(30)  # match bicycle model d_limit
    MIN_STEER = -np.deg2rad(30)
    MAX_SPEED = 25.0  # allow higher speeds
    MIN_SPEED = 0.0
    MAX_ACCEL = 3.0  # allow stronger acceleration

    # cost weights
    Q = np.diag([10.0, 10.0, 5.0, 8.0])
    Qf = np.diag([20.0, 20.0, 10.0, 15.0])
    R = np.diag([0.1, 10.0])
    Rd = np.diag([0.1, 10.0])


class DrakeMPC(LeafSystem):
    """
    A Drake-based receding horizon MPC controller using the bicycle model of the car.
    The controller solves an optimization problem at each time step to compute the optimal control inputs
    over a finite horizon, minimizing a cost function that penalizes deviation from a reference trajectory.
    """

    def __init__(self, reference_provider):
        """
        ref_traj_callback(t, state) → produces a reference trajectory of shape (4, T+1)
        """

        super().__init__()
        self.cfg = DrakeMPCConfig()
        #self.ref_callback = ref_traj_callback
        self.ref_provider = reference_provider

        # Input port: full vehicle state [x, y, v, yaw]
        self.DeclareVectorInputPort("state", BasicVector(self.cfg.NX))

        # Output port: control [a, delta]
        self.DeclareVectorOutputPort("control", BasicVector(self.cfg.NU),
                                     self.CalcControl)

        self.solver = SnoptSolver()

    # ------------------------------------------------------------
    # Discrete Kinematic Bicycle Model (same as ROS MPC)
    # ------------------------------------------------------------
    def linearized_discrete_dynamics(self, x0):
        # u0 = zero input linearization  
        u0 = np.array([0.0, 0.0])
        A_d, B_d = self.model.discrete_time_linearized_dynamics(
            x0, u0, self.DT
        )
        return A_d, B_d
    
    # ------------------------------------------------------------
    # Discrete Kinematic Bicycle Model (Euler step)
    # State x = [x, y, v, yaw], control u = [a, delta]
    # ------------------------------------------------------------
    def discrete_dynamics(self, x, u):
        cfg = self.cfg

        x_pos = x[0]
        y_pos = x[1]
        v     = x[2]
        yaw   = x[3]

        a     = u[0]
        delta = u[1]

        # Continuous-time derivatives
        dx   = v * np.cos(yaw)
        dy   = v * np.sin(yaw)
        dv   = a
        dyaw = v / cfg.WB * np.tan(delta)

        # Explicit Euler step to get x_{k+1}
        return np.array([
            x_pos + cfg.DT * dx,
            y_pos + cfg.DT * dy,
            v     + cfg.DT * dv,
            yaw   + cfg.DT * dyaw,
        ])


    # ------------------------------------------------------------
    # MPC Solve Step
    # ------------------------------------------------------------
    def solve_mpc(self, x0, ref):
        cfg = self.cfg
        prog = MathematicalProgram()

        # Coerce shapes
        x0 = np.asarray(x0, dtype=float).reshape(cfg.NX,)
        ref = np.asarray(ref, dtype=float).reshape(cfg.NX, cfg.T + 1)

        # Decision vars
        x = prog.NewContinuousVariables(cfg.NX, cfg.T + 1, "x")
        u = prog.NewContinuousVariables(cfg.NU, cfg.T, "u")

        # ----- Initial state: add constraints component-wise -----
        for i in range(cfg.NX):
            prog.AddLinearConstraint(x[i, 0] == x0[i])

        # ----- Dynamics constraints -----
        for k in range(cfg.T):
            xk = x[:, k]
            uk = u[:, k]
            x_next = x[:, k + 1]

            x_dyn = self.discrete_dynamics(xk, uk)
            for i in range(cfg.NX):
                prog.AddConstraint(x_next[i] == x_dyn[i])

        # ----- State bounds -----
        for k in range(cfg.T + 1):
            prog.AddConstraint(x[2, k] <= cfg.MAX_SPEED)   # v
            prog.AddConstraint(x[2, k] >= cfg.MIN_SPEED)

        # ----- Input bounds -----
        prog.AddBoundingBoxConstraint(-cfg.MAX_ACCEL, cfg.MAX_ACCEL, u[0, :])
        prog.AddBoundingBoxConstraint(cfg.MIN_STEER, cfg.MAX_STEER, u[1, :])

        # ----- Objective function -----
        cost = 0.0

        # Tracking cost over horizon
        for k in range(cfg.T):
            e = x[:, k] - ref[:, k]
            cost += e @ cfg.Q @ e

        # Terminal cost
        eT = x[:, cfg.T] - ref[:, cfg.T]
        cost += eT @ cfg.Qf @ eT

        # Input effort
        for k in range(cfg.T):
            cost += u[:, k] @ cfg.R @ u[:, k]

        # Input smoothness
        for k in range(cfg.T - 1):
            du = u[:, k + 1] - u[:, k]
            cost += du @ cfg.Rd @ du

        prog.AddCost(cost)

        result = self.solver.Solve(prog)

        if not result.is_success():
            # You can log more info here if you like
            print("WARNING: MPC solve failed; returning zero control")
            return np.array([0.0, 0.0])

        u0 = result.GetSolution(u[:, 0])
        return np.array(u0).reshape(cfg.NU,)

    # ------------------------------------------------------------
    # Drake Output Port: returns control command
    # ------------------------------------------------------------
    def CalcControl(self, context, output):
        x = self.get_input_port(0).Eval(context)
        t = context.get_time()

        # Query reference trajectory generator (lane-center, lane change, etc.)
        ref = self.ref_provider.get_ref_horizon(
            x0=x,
            T=self.cfg.T,
            dt=self.cfg.DT
        )

        u = self.solve_mpc(x, ref)
        output.SetFromVector(u)
