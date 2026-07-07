"""Booster T1 Navigation MPC example using SPC C++ backend.

Uses the T1Navigation task with CEM optimization to navigate a Booster T1
humanoid robot to a goal position defined by a mocap body marker.

The velocity commands (vx, vy, vtheta) are optimized by CEM, and a trained
locomotion policy (ONNX) converts them to motor targets at each step.

Usage: python3 examples/run_t1_navigation.py [--no_policy]
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

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/t1/scene_navigation.xml"))
    policy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../policies/t1_navigation.onnx"))

    print(f"Loading model from {model_path}")
    env = spc_py.SpcEnv(model_path)

    task_params = {
        "action_scale": 1.0,
        "gait_freq": 1.5,
        "target_height": 0.665,
        "pos_weight": 1.0,
        "ori_weight": 1.0,
        "upright_weight": 2.0,
        "height_weight": 0.5,
        "ctrl_weight": 0.01,
        "vx_limit": 1.5,
        "vy_limit": 1.2,
        "vtheta_limit": 1.5,
    }
    task = spc_py.create_task("T1Navigation", env, task_params)

    policy = None
    if not args.no_policy:
        if os.path.exists(policy_path):
            print(f"Loading policy from {policy_path}")
            policy = spc_py.ONNXPolicy(policy_path)
        else:
            print(f"WARNING: Policy not found at {policy_path}")
            print("Continuing without policy...")

    config = spc_py.CEMConfig()
    config.num_samples = 8  # 1 rollout per core; realtime on 8 physical cores
    config.num_elites = 4
    config.num_knots = 4
    config.num_iterations = 1
    config.plan_horizon_steps = 25
    config.sim_substeps = 10  # dt=0.002, ctrl_dt=0.02 -> 10 substeps (mujoco_playground settings)
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

    run_interactive(
        env,
        cem,
        model_path,
        sim_dt=0.02,
        sim_steps_per_replan=10,
        init_kwargs={"keyframe_name": "home", "mocap_defaults": {0: ([3.0, 1.0, 0.05], [1.0, 0.0, 0.0, 0.0])}},
        record=args.record,
    )


if __name__ == "__main__":
    main()
