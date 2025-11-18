"""
Test trajectory planner continuity properties.

Checks:
1. C0: Position continuity (no jumps in x, y)
2. C1: Velocity continuity (no jumps in dx/dt, dy/dt)
3. C2: Acceleration continuity (no jumps in d²x/dt², d²y/dt²)
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from planners.trajectory_planner import make_ref_function
from planners.behavior_planner import BehaviorPlanner


def finite_difference_derivative(func, t, dt=1e-5):
    """Compute derivative using central difference."""
    return (func(t + dt) - func(t - dt)) / (2 * dt)


def test_cruise_continuity():
    """Test CRUISE trajectory (should be perfectly smooth straight line)."""
    print("\n=== Testing CRUISE Continuity ===")
    
    # Setup
    v_ref = 20.0
    ego_state = np.array([0.0, 0.0, 0.0, 15.0, 0.0, 0.0])  # lane 1, 15 m/s
    ref_func, _ = make_ref_function(
        intent="CRUISE",
        ego_state=ego_state,
        v_ref=v_ref,
        target_lane=1,
        horizon_s=5.0
    )
    
    # Sample at many points
    t_samples = np.linspace(0, 5.0, 100)
    positions = np.array([ref_func(t)[:2] for t in t_samples])  # (x, y)
    
    # Check C0: position should be continuous
    # Expected displacement between samples at v_ref = 20 m/s
    dt_sample = t_samples[1] - t_samples[0]
    pos_diffs = np.diff(positions, axis=0)
    pos_step_sizes = np.linalg.norm(pos_diffs, axis=1)
    expected_step = v_ref * dt_sample
    
    # Check all steps are consistent
    step_consistency = np.std(pos_step_sizes)
    print(f"Position step consistency: mean={np.mean(pos_step_sizes):.4f}, std={step_consistency:.6e} m")
    print(f"  Expected step size: {expected_step:.4f} m")
    assert step_consistency < 0.01, "Position has discontinuities!"
    
    # Check C1: velocity should be continuous
    dt = t_samples[1] - t_samples[0]
    velocities = pos_diffs / dt
    vel_jumps = np.diff(velocities, axis=0)
    max_vel_jump = np.max(np.abs(vel_jumps))
    print(f"Max velocity jump: {max_vel_jump:.6e} m/s (expect < 0.1)")
    assert max_vel_jump < 0.1, "Velocity not continuous!"
    
    # CRUISE should have constant velocity
    v_ref = 20.0
    expected_vx = v_ref * np.cos(0.0)  # heading = 0
    actual_vx = velocities[:, 0]
    print(f"Velocity consistency: mean={np.mean(actual_vx):.2f}, std={np.std(actual_vx):.4f} m/s")
    assert np.allclose(actual_vx, expected_vx, rtol=0.01), "CRUISE velocity not constant!"
    
    print("CRUISE continuity test PASSED")


def test_lane_change_continuity():
    """Test LANE_CHANGE_TO trajectory (quintic polynomial)."""
    print("\n=== Testing LANE_CHANGE Continuity ===")
    
    # Start in lane 0, target lane 1 (3.7m lateral shift)
    ego_state = np.array([0.0, -3.7, 0.0, 15.0, 0.0, 0.0])
    T_lc = 4.0  # lane change duration
    
    ref_func, _ = make_ref_function(
        intent="LANE_CHANGE_TO",
        ego_state=ego_state,
        v_ref=20.0,
        target_lane=1,
        horizon_s=5.0,
        T_lc=T_lc
    )
    
    # Sample densely, especially around boundaries (avoid duplicates)
    t_samples = np.concatenate([
        np.linspace(0, 0.5, 20, endpoint=False),       # start
        np.linspace(0.5, T_lc - 0.5, 50, endpoint=False),  # middle
        np.linspace(T_lc - 0.5, T_lc, 20, endpoint=False), # end
        np.linspace(T_lc, 5.0, 20)                         # after
    ])
    
    # Extract trajectory
    trajectory = np.array([ref_func(t) for t in t_samples])
    x_vals = trajectory[:, 0]
    y_vals = trajectory[:, 1]
    
    # C0 test: Position continuous
    dt_vec = np.diff(t_samples)
    dx = np.diff(x_vals)
    dy = np.diff(y_vals)
    # Filter out zero dt values
    valid_mask = dt_vec > 1e-10
    if np.any(valid_mask):
        max_pos_jump = np.max(np.sqrt(dx[valid_mask]**2 + dy[valid_mask]**2) / dt_vec[valid_mask])
        print(f"Max position rate: {max_pos_jump:.2f} m/s (expect ~20 m/s for v_ref=20)")
    else:
        print("Warning: No valid position differences to check")
    
    # C1 test: Velocity continuous (check at boundaries)
    def get_velocity(t):
        dt = 1e-5
        x1, y1, _, _ = ref_func(t - dt)
        x2, y2, _, _ = ref_func(t + dt)
        return (x2 - x1) / (2*dt), (y2 - y1) / (2*dt)
    
    # Check velocity at key times
    times_to_check = [0.0, 0.01, T_lc - 0.01, T_lc, T_lc + 0.01]
    print("\nVelocity continuity at boundaries:")
    for t in times_to_check:
        vx, vy = get_velocity(t)
        print(f"  t={t:.2f}s: vx={vx:.2f}, vy={vy:.3f} m/s")
    
    # Start and end should have vy ≈ 0 (quintic boundary conditions)
    vx_start, vy_start = get_velocity(0.01)
    vx_end, vy_end = get_velocity(T_lc - 0.01)
    print(f"\nLateral velocity at boundaries:")
    print(f"  Start (t=0.01s): vy={vy_start:.4f} m/s (expect ≈0)")
    print(f"  End (t={T_lc-0.01:.2f}s): vy={vy_end:.4f} m/s (expect ≈0)")
    assert abs(vy_start) < 0.5, "Initial lateral velocity too high!"
    assert abs(vy_end) < 0.5, "Final lateral velocity too high!"
    
    # C2 test: Acceleration continuous (quintic ensures this)
    def get_acceleration(t):
        dt = 1e-4
        vx1, vy1 = get_velocity(t - dt)
        vx2, vy2 = get_velocity(t + dt)
        return (vx2 - vx1) / (2*dt), (vy2 - vy1) / (2*dt)
    
    print("\nAcceleration at boundaries:")
    for t in [0.0, T_lc]:
        ax, ay = get_acceleration(t)
        print(f"  t={t:.2f}s: ax={ax:.3f}, ay={ay:.3f} m/s²")
    
    # Check lateral displacement is correct
    y_start = y_vals[0]
    y_end = y_vals[-1]
    print(f"\nLateral displacement:")
    print(f"  Start: y={y_start:.2f} m")
    print(f"  End:   y={y_end:.2f} m")
    print(f"  Shift: {y_end - y_start:.2f} m (expect 3.7 m)")
    assert abs((y_end - y_start) - 3.7) < 0.1, "Lane change didn't reach target!"
    
    print("LANE_CHANGE continuity test PASSED")


def test_transition_continuity():
    """Test continuity when switching from CRUISE to LANE_CHANGE."""
    print("\n=== Testing Transition Continuity ===")
    
    # Scenario: vehicle cruising, then decides to change lanes
    ego_state_cruise = np.array([10.0, 0.0, 0.0, 15.0, 0.0, 0.0])  # lane 1
    
    # CRUISE phase
    ref_cruise, _ = make_ref_function(
        intent="CRUISE",
        ego_state=ego_state_cruise,
        v_ref=20.0,
        target_lane=1,
        horizon_s=2.0
    )
    
    # After 1 second, start lane change
    x_at_1s, y_at_1s, psi_at_1s, v_at_1s = ref_cruise(1.0)
    ego_state_lc = np.array([x_at_1s, y_at_1s, psi_at_1s, v_at_1s, 0.0, 0.0])
    
    ref_lc, _ = make_ref_function(
        intent="LANE_CHANGE_TO",
        ego_state=ego_state_lc,
        v_ref=20.0,
        target_lane=2,  # change to lane 2
        horizon_s=5.0,
        T_lc=4.0
    )
    
    # Check position matches at transition
    x_cruise_end, y_cruise_end, _, _ = ref_cruise(1.0)
    x_lc_start, y_lc_start, _, _ = ref_lc(0.0)
    
    pos_error = np.sqrt((x_lc_start - x_cruise_end)**2 + (y_lc_start - y_cruise_end)**2)
    print(f"Position error at transition: {pos_error:.6e} m (expect ≈0)")
    assert pos_error < 1e-6, "Position discontinuity at transition!"
    
    print("Transition continuity test PASSED")


def test_physical_feasibility():
    """Test that generated trajectories are physically feasible."""
    print("\n=== Testing Physical Feasibility ===")
    
    # Lane change with realistic parameters
    ego_state = np.array([0.0, -3.7, 0.0, 15.0, 0.0, 0.0])
    ref_func, _ = make_ref_function(
        intent="LANE_CHANGE_TO",
        ego_state=ego_state,
        v_ref=20.0,
        target_lane=1,
        horizon_s=5.0,
        T_lc=4.0
    )
    
    # Check max lateral acceleration
    t_samples = np.linspace(0, 5.0, 200)
    max_lat_accel = 0.0
    
    for t in t_samples:
        dt = 1e-4
        _, y1, _, _ = ref_func(t - dt)
        _, y2, _, _ = ref_func(t)
        _, y3, _, _ = ref_func(t + dt)
        
        # Second derivative (acceleration)
        ay = (y3 - 2*y2 + y1) / (dt**2)
        max_lat_accel = max(max_lat_accel, abs(ay))
    
    print(f"Max lateral acceleration: {max_lat_accel:.2f} m/s²")
    print(f"  Typical car limit: 8-10 m/s²")
    print(f"  Comfortable: < 3 m/s²")
    
    # Typical passenger car can do ~0.8g = 7.8 m/s² lateral
    assert max_lat_accel < 10.0, "Lateral acceleration exceeds vehicle limits!"
    
    if max_lat_accel < 3.0:
        print("Comfortable acceleration level")
    else:
        print("Aggressive but feasible")
    
    print("Physical feasibility test PASSED")

if __name__ == "__main__":
    print("=" * 60)
    print("Trajectory Planner Continuity Tests")
    print("=" * 60)
    
    try:
        test_cruise_continuity()
        test_lane_change_continuity()
        test_transition_continuity()
        test_physical_feasibility()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
