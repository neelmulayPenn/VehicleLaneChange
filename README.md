# Vehicle Lane Changing and Intersection Control


This repository presents a research-oriented autonomous driving stack developed for MEAM 5170 (Control and Optimization with Applications in Robotics). The work integrates nonlinear vehicle dynamics, hierarchical planning, uncertain perception, and constrained receding-horizon control for lane-change and intersection scenarios.


## Abstract


We study closed-loop autonomy for structured road environments by combining:


- A 6-state dynamic bicycle model with tire lateral force effects.
- A behavior layer with lane-change commitment, TTC hazard logic, and hysteresis.
- A trajectory layer based on quintic lateral motion profiles.
- A linearized MPC solved with Drake + SNOPT over a finite horizon.
- A perception wrapper that injects sensing latency and noise.


The resulting pipeline demonstrates robust behavior under dynamic traffic and imperfect state information while maintaining physically plausible trajectories.


## Problem Formulation


Let the ego state be:


$$
\mathbf{x} = [X,\;Y,\;\psi,\;v_x,\;v_y,\;\omega]^\top
$$


and the control input be:


$$
\mathbf{u} = [\delta,\;a_x]^\top
$$


where $\delta$ is steering angle and $a_x$ is longitudinal acceleration.


At each control step, we solve a finite-horizon optimal control problem:


$$
\min_{\{\mathbf{x}_k,\mathbf{u}_k\}} \sum_{k=0}^{T-1}
(\mathbf{x}_k-\mathbf{x}^{ref}_k)^\top Q(\mathbf{x}_k-\mathbf{x}^{ref}_k)
+ \mathbf{u}_k^\top R\mathbf{u}_k
+ \sum_{k=0}^{T-2}(\Delta \mathbf{u}_k)^\top R_d(\Delta \mathbf{u}_k)
+ (\mathbf{x}_T-\mathbf{x}^{ref}_T)^\top Q_f(\mathbf{x}_T-\mathbf{x}^{ref}_T)
$$


subject to linearized dynamics and actuator/state constraints.


## Method Overview


### 1. Vehicle Dynamics


Implemented in Bicycle_Model/bicycle_model_dynamics.py.


- Nonlinear bicycle dynamics with front/rear slip-angle lateral forces.
- Numerical Jacobian linearization around reference operating points.
- Euler-discretized local linear model for MPC prediction.


This supports both realistic simulation rollout and tractable constrained optimization.


### 2. Behavioral Decision Layer


Implemented in planners/behavior_planner.py and planners/obstacle_aware_planner.py.


- Finite-state behavior: CRUISE and LANE_CHANGE_TO.
- TTC and distance-based hazard detection.
- Adjacent-lane utility scoring.
- Hysteresis and commitment logic to avoid oscillatory lane switching.
- Cooldown logic after lane changes.
- Speed adaptation for car-following when lateral escape is unsafe.
- Optional intersection conflict logic using ego/cross-traffic TTC comparison.


### 3. Trajectory Generation


Implemented in planners/trajectory_planner.py.


- CRUISE: lane-center tracking with forward progression.
- LANE_CHANGE_TO: quintic lateral polynomial with smooth boundary conditions.
- Fixed maneuver coefficient reuse for temporal consistency.


The lane-change profile satisfies smoothness objectives and limits abrupt curvature/yaw demands.


### 4. Perception Realism


Implemented in planners/perception.py.


- Perception updates at a slower rate than simulation integration.
- Additive Gaussian noise on obstacle position estimates.
- Planner/controller operate on stale/noisy obstacle snapshots.


This introduces realistic sensing imperfections and stress-tests planning robustness.


### 5. MPC Controller


Implemented in controllers/mpc_controller.py.


- Horizon: $T=15$, $\Delta t=0.1$ s.
- Linearized discrete dynamics around the reference trajectory.
- State and control constraints (speed, steering, acceleration).
- Quadratic tracking/smoothness objective.
- Solver: SNOPT through Drake MathematicalProgram.


## Closed-Loop Stack Interaction


```text
True traffic -> NoisyPerception -> ObstacleAwarePlanner -> ReferenceProvider -> DrakeMPC -> ego control
      ^                                                                                      |
      |--------------------------------------------------------------------------------------|
                         Ego propagated by nonlinear bicycle dynamics
```


Per step:


1. Perception updates obstacle estimates.
2. Behavior planner selects intent and target lane/speed.
3. Trajectory generator produces a local reference horizon.
4. MPC solves and outputs steering/acceleration.
5. Ego state is integrated through nonlinear dynamics.


## Experiments and Scenarios


### Straight-Road Overtake


- Function: run_overtake_sim_ttc(...) in planners/obstacle_aware_planner.py.
- Multi-lane traffic with mixed speeds.
- Demonstrates TTC-triggered lane changes and follow behavior.


### 4-Way Intersection


- Functions: run_intersection_sim(...) in controllers/intersection_sim.py and create_intersection_animation(...) in controllers/intersection_animation.py.
- Multi-direction traffic crossing ego path.
- Demonstrates geometry-aware reference generation and intersection-aware behavior.


### Trajectory Smoothness Validation


- Script: planners/test_continuity.py.
- Checks continuity and feasibility properties of generated references:
  - C0 position continuity
  - C1 velocity continuity
  - C2 acceleration continuity
  - lateral acceleration magnitude sanity check




## Repository Map


```text
VehicleLaneChanging/
  Bicycle_Model/
    bicycle_model_dynamics.py
    bicycle_model_sim.py
    bicycle_model_visualizer.py
  planners/
    env.py
    intersection_env.py
    behavior_planner.py
    trajectory_planner.py
    obstacle_aware_planner.py
    perception.py
    obstacle_spawner.py
    test_continuity.py
  controllers/
    mpc_controller.py
    reference_provider.py
    intersection_reference_provider.py
    intersection_sim.py
    intersection_animation.py
```


## Setup


Dependencies:


- numpy
- scipy
- matplotlib
- drake / pydrake


Example environment setup:


```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy scipy matplotlib drake
```


## Quick Reproduction


1. Run continuity tests:


```powershell
python planners/test_continuity.py
```


2. Run a lane-change or intersection simulation from a notebook/script by calling:


- run_overtake_sim_ttc(...)
- run_intersection_sim(...)


3. Visualize with animation helpers:


- create_animation(...)
- create_intersection_animation(...)


## Contribution to Controls and Autonomous Driving


This work demonstrates competency in:


- Nonlinear modeling and local linearization of vehicle dynamics.
- Formulating and solving constrained finite-horizon control problems.
- Hierarchical autonomy design (behavior-planning-control decomposition).
- Safety-oriented decision heuristics (TTC, hysteresis, commitment).
- Robustness considerations under sensing uncertainty and asynchronous updates.


From a graduate-level perspective, the key value is the end-to-end integration: modeling assumptions, planner design choices, and optimization-based control are all implemented, connected, and validated in simulation.





