"""Booster T1 Shoot MPC example using SPC C++ backend (velocity-only control).

Uses the T1Shoot task with CEM optimization to drive a soccer ball through a
goal *region* at maximum power. The mocap marker defines a goal line: its yaw
sets the shooting direction and goal_half_width the region's lateral extent.
Unlike T1Pass, the ball flying past the line costs nothing (overshoot is free).

The velocity commands (vx, vy, vtheta) are optimized by CEM, and a trained
ONNX locomotion policy converts them to motor targets at each step.

Usage: python3 examples/t1_shoot.py [--no_policy]
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
        "standoff_distance": 0.25,  # T1 is smaller than G1: stand closer so the feet reach the ball
        "ball_goal_weight": 2.0,  # shortfall to the goal line (zero once crossed)
        "goal_half_width": 1.2,  # matches the scene's goal mouth (posts at +-1.25)
        "lateral_weight": 1.0,  # penalize lateral miss beyond the half-width
        "shoot_power_weight": 0.5,  # reward ball speed along the shooting direction
        "pos_weight": 0.3,
        "ori_weight": 0.2,
        "height_weight": 0.5,
        "upright_weight": 2.0,  # penalize trunk tilt (prevents falls)
        "behind_weight": 2.0,  # stay behind the ball relative to the shot direction
        "ctrl_weight": 0.01,
        "vx_limit": 1.5,
        "vy_limit": 1.2,
        "vtheta_limit": 1.5,
    }

    # We pass the same policy used for navigation, as it accepts velocity commands
    task = spc_py.create_task("T1Shoot", env, task_params)

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
    # step, ~1.7x cheaper than the real dt=0.002 x10); the saving buys the long
    # horizon the approach needs. ~0.5x realtime on 8 cores.
    config.num_samples = 16
    config.num_elites = 8
    config.num_knots = 4
    config.num_iterations = 1
    config.plan_horizon_steps = 64
    config.sim_substeps = 5
    config.plan_timestep = 0.004  # coarse planning dt (real sim stays at the model's 0.002)
    config.control_dim = 3  # vx, vy, vtheta
    config.obs_dim = 85
    config.num_threads = 8
    config.sigma_init = 0.5
    config.sigma_min = 0.05
    config.explore_fraction = 0.5
    config.replan_shift_steps = 1  # replan every ctrl step: warm-start shifted mean
    config.elite_keep = 2  # re-inject previous replan's best samples (iCEM)

    # Velocity commands:
    config.u_min = [-1.5, -1.2, -1.5]
    config.u_max = [1.5, 1.2, 1.5]

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
