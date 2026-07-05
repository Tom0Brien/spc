"""G1 Navigation MPC example using SPC C++ backend.

Uses the G1Navigation task with CEM optimization to navigate a G1 humanoid
robot to a goal position defined by a mocap body marker.

The velocity commands (vx, vy, vtheta) are optimized by CEM, and a trained
locomotion policy (hand-rolled MLP) converts them to motor targets at each step.

Usage: python3 examples/run_g1_navigation.py [--no_policy]
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
    args = parser.parse_args()

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/g1/scene_navigation.xml"))
    policy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../policies/g1_navigation.mlp"))

    print(f"Loading model from {model_path}")
    env = spc_py.SpcEnv(model_path)

    task_params = {
        "action_scale": 0.5,
        "gait_freq": 1.5,
        "target_height": 0.75,
        "pos_weight": 1.0,
        "ori_weight": 1.0,
        "upright_weight": 2.0,
        "height_weight": 0.5,
        "ctrl_weight": 0.01,
    }
    task = spc_py.create_task("G1Navigation", env, task_params)

    policy = None
    if not args.no_policy:
        if os.path.exists(policy_path):
            print(f"Loading policy from {policy_path}")
            policy = spc_py.MLPPolicy(policy_path)
        else:
            print(f"WARNING: Policy not found at {policy_path}")
            print("Continuing without policy...")

    config = spc_py.CEMConfig()
    config.num_samples = 8  # 1 rollout per core; realtime on 8 physical cores
    config.num_elites = 4
    config.num_knots = 4
    config.num_iterations = 1
    config.plan_horizon_steps = 40
    config.sim_substeps = 5  # dt=0.004, ctrl_dt=0.02 -> 5 substeps
    config.control_dim = 3  # vx, vy, vtheta
    config.obs_dim = 103
    config.num_threads = 8
    config.sigma_init = 0.5
    config.sigma_min = 0.05
    config.explore_fraction = 0.5
    config.replan_shift_steps = 1  # replan every ctrl step: warm-start shifted mean
    config.elite_keep = 2  # re-inject previous replan's best samples (iCEM)

    # Velocity commands bounded to the RL policy's training range
    config.u_min = [-1.0, -1.0, -1.0]
    config.u_max = [1.0, 1.0, 1.0]

    cem = spc_py.CEM(env, task, policy, config)

    run_interactive(
        env,
        cem,
        model_path,
        sim_dt=0.02,
        sim_steps_per_replan=5,
        init_kwargs={"keyframe_name": "knees_bent", "mocap_defaults": {0: ([3.0, 1.0, 0.05], [1.0, 0.0, 0.0, 0.0])}},
    )


if __name__ == "__main__":
    main()
