"""Booster T1 Shoot (augmented) MPC example using SPC C++ backend.

Uses the T1ShootAugmented task with CEM optimization. The control vector is
extended to 15 dims:
  - control[0:3]  : velocity commands (vx, vy, vtheta) for the RL policy
  - control[3:15] : residual adjustments for the 12 leg joint targets

The mocap marker defines a goal *region*: its yaw sets the shooting direction
and goal_half_width the region's lateral extent. Unlike T1PassAugmented, the
ball flying past the goal line costs nothing (overshoot is free), and ball
speed along the shot direction is rewarded — the planner should wind up for a
maximum-power kick rather than a controlled dribble.

Usage: python3 examples/t1_shoot_augmented.py [--no_policy]
"""

import os

# Prevent OpenMP thread migration for better L1/L2 cache locality
os.environ["OMP_PROC_BIND"] = "true"
os.environ["OMP_PLACES"] = "cores"

import argparse

import spc_py

from utils import run_interactive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_policy", action="store_true", help="Run without ONNX policy")
    parser.add_argument("--record", action="store_true", help="Record an mp4 of the viewer to recordings/")
    args = parser.parse_args()

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/t1/scene_shoot.xml"))
    policy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../policies/t1_navigation.onnx"))

    print(f"Loading model from {model_path}")
    env = spc_py.SpcEnv(model_path)

    task_params = {
        "action_scale": 1.0,
        "gait_freq": 1.5,
        "target_height": 0.665,

        # Shoot-specific weights
        "standoff_distance": 0.35,  # augmented: leg-swing residuals extend the foot's reach
        "ball_goal_weight": 2.0,  # shortfall to the goal line (zero once crossed)
        "goal_half_width": 1.2,  # matches the scene's goal mouth (posts at +-1.25)
        "lateral_weight": 1.0,  # penalize lateral miss beyond the half-width
        "shoot_power_weight": 0.5,  # reward ball speed along the shooting direction
        "pos_weight": 0.3,
        "ori_weight": 0.2,
        "height_weight": 0.5,
        "upright_weight": 2.0,  # penalize trunk tilt (prevents falls)
        "behind_weight": 0.0,  # off: residual kicks need close contact
        "ctrl_weight": 0.01,
        "vx_limit": 1.5,
        "vy_limit": 1.2,
        "vtheta_limit": 1.5,

        # Augmented (leg residual) params
        # T1 joint order is head(2), arms(8), waist(1), left leg(6), right leg(6),
        # so the 12 leg joints start at index 11.
        "leg_joint_start": 11,
        "leg_joint_count": 12,
        # Very cheap residuals for power (0.2 -> 0.1 lifted mean crossing speed
        # 1.9 -> 2.8 m/s over a 4-seed eval; 0.05 adds variance without gains).
        "residual_weight": 0.15,
        # Wider gate than the pass task (0.45/0.75): the kick wind-up happens
        # during the approach, so residuals must engage before the ball is in
        # reach. Widening to 0.60/1.20 nearly doubled crossing speed; 1.5 is
        # too wide (residuals disturb the approach gait).
        "gate_near": 0.60,
        "gate_far": 1.20,
    }

    # We pass the same policy used for navigation, as it accepts velocity commands
    task = spc_py.create_task("T1ShootAugmented", env, task_params)

    policy = None
    if not args.no_policy:
        if os.path.exists(policy_path):
            print(f"Loading policy from {policy_path}")
            policy = spc_py.ONNXPolicy(policy_path)
        else:
            print(f"WARNING: Policy not found at {policy_path}")
            print("Continuing without policy...")

    config = spc_py.CEMConfig()
    # Rollouts plan on a coarser model (dt=0.004 x5 substeps = 0.02s/control
    # step, ~1.7x cheaper than the real dt=0.002 x10); the saving buys the
    # sample count and horizon the 15-dim control needs. ~0.45x realtime on 8
    # cores.
    config.num_samples = 64
    config.num_elites = 12
    config.num_knots = 4
    config.num_iterations = 2
    config.plan_horizon_steps = 48
    config.sim_substeps = 5
    config.plan_timestep = 0.004  # coarse planning dt (real sim stays at the model's 0.002)
    config.control_dim = 15  # vx, vy, vtheta + 12 leg residuals
    config.obs_dim = 85
    config.num_threads = 8
    config.sigma_init = 0.5
    config.sigma_min = 0.05
    config.explore_fraction = 0.5
    # Warm-start shift and elite reuse stay off: leg residuals are
    # gait-phase-locked, so re-timed means and stale elites from the previous
    # replan desynchronize from the gait (verified to hurt the pass task).
    config.replan_shift_steps = 0
    config.elite_keep = 0

    # Per-dimension bounds and sampling spread. Residual bounds +-0.6 rad give
    # the kick a real wind-up amplitude and keep sigma=0.5 sampling from
    # saturating at the bounds (joint limits still clamp the final targets).
    config.u_min = [-1.5, -1.2, -1.5] + [-0.6] * 12
    config.u_max = [1.5, 1.2, 1.5] + [0.6] * 12
    config.sigma_init_per_dim = [0.5] * 3 + [0.5] * 12

    cem = spc_py.CEM(env, task, policy, config)

    # The scene keyframe sets up the shot: robot at (3.8, 0) facing the goal,
    # ball on the penalty spot (5, 0). The mocap goal region defaults to the
    # goal mouth at x=7 (its yaw is the shooting direction; drag/rotate it in
    # the viewer to move the target).
    run_interactive(
        env,
        cem,
        model_path,
        sim_dt=0.02,
        sim_steps_per_replan=10,
        init_kwargs={"keyframe_name": "home"},
        record=args.record,
    )


if __name__ == "__main__":
    main()
