#!/usr/bin/env python3
import numpy as np
from pydrake.systems.framework import LeafSystem, BasicVector
from pydrake.solvers.mathematicalprogram import MathematicalProgram
from pydrake.solvers.snopt import SnoptSolver
from math import tan, cos, sin


class DrakeMPCConfig:
    NX = 4  # [x, y, v, yaw]
    NU = 2  # [a, delta]
    T = 15  # horizon
    DT = 0.1

    # vehicle params
    WB = 0.33
    MAX_STEER = 0.4189 * 1.75
    MIN_STEER = -0.4189 * 1.75
    MAX_SPEED = 6.0
    MIN_SPEED = 0.0
    MAX_ACCEL = 2.0

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
    # MPC Solve Step
    # ------------------------------------------------------------
    def solve_mpc(self, x0, ref):
        cfg = self.cfg
        prog = MathematicalProgram()

        # Decision vars
        x = prog.NewContinuousVariables(cfg.NX, cfg.T + 1, "x")
        u = prog.NewContinuousVariables(cfg.NU, cfg.T, "u")

        # Initial state
        prog.AddConstraint(x[:, 0] == x0)

        # Dynamics constraints
        for k in range(cfg.T):
            xk = x[:, k]
            uk = u[:, k]
            x_next = x[:, k + 1]

            # Use explicit Euler
            x_dyn = self.dynamics(xk, uk)
            prog.AddConstraint(x_next == x_dyn)

        # State bounds
        for k in range(cfg.T + 1):
            prog.AddConstraint(x[2, k] <= cfg.MAX_SPEED)   # v
            prog.AddConstraint(x[2, k] >= cfg.MIN_SPEED)

        # Input bounds
        prog.AddBoundingBoxConstraint(-cfg.MAX_ACCEL, cfg.MAX_ACCEL, u[0, :])
        prog.AddBoundingBoxConstraint(cfg.MIN_STEER, cfg.MAX_STEER, u[1, :])

        # Objective function
        cost = 0

        # Tracking cost
        for k in range(cfg.T):
            e = x[:, k] - ref[:, k]
            cost += e @ cfg.Q @ e

        # terminal
        eT = x[:, cfg.T] - ref[:, cfg.T]
        cost += eT @ cfg.Qf @ eT

        # Input effort
        for k in range(cfg.T):
            cost += u[:, k] @ cfg.R @ u[:, k]

        # Smoothness
        for k in range(cfg.T - 1):
            du = u[:, k + 1] - u[:, k]
            cost += du @ cfg.Rd @ du

        prog.AddCost(cost)

        result = self.solver.Solve(prog)

        if not result.is_success():
            print("WARNING: MPC failed; returning zero control")
            return np.array([0.0, 0.0])

        u0 = result.GetSolution(u[:, 0])
        return u0

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
