"""Booster K1 Pass MPC example using SPC C++ backend.

Uses the K1Pass task with CEM optimization to navigate a Booster K1 humanoid
robot to a soccer ball and push it toward a goal position.

The velocity commands (vx, vy, vtheta) are optimized by CEM, and a trained
ONNX locomotion policy converts them to motor targets at each step.

Usage: python3 examples/k1_pass.py [--no_policy]
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

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/k1/scene_soccer.xml"))
    policy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../policies/k1_navigation.onnx"))

    print(f"Loading model from {model_path}")
    env = spc_py.SpcEnv(model_path)

    task_params = {
        "action_scale": 1.0,
        "gait_freq": 1.5,
        "target_height": 0.543,

        # Pass-specific weights
        "standoff_distance": 0.25,  # K1 is smaller than G1: stand closer so the feet reach the ball
        "ball_goal_weight": 2.0,
        "pos_weight": 0.3,
        "ori_weight": 0.2,
        "height_weight": 0.5,
        "upright_weight": 2.0,  # penalize trunk tilt (prevents falls)
        "behind_weight": 2.0,  # stay on the far side of the ball from the goal
        "ctrl_weight": 0.01,
        "vx_limit": 1.0,
        "vy_limit": 0.8,
        "vtheta_limit": 1.0,
    }

    # We pass the same policy used for navigation, as it accepts velocity commands
    task = spc_py.create_task("K1Pass", env, task_params)

    policy = None
    if not args.no_policy:
        if os.path.exists(policy_path):
            print(f"Loading policy from {policy_path}")
            policy = spc_py.ONNXPolicy(policy_path)
        else:
            print(f"WARNING: Policy not found at {policy_path}")
            print("Continuing without policy...")

    config = spc_py.CEMConfig()
    # Planning uses the exact dt the RL policy was trained on (sim_dt=0.002,
    # ctrl_dt=0.02 -> 10 substeps, matching mujoco_playground K1 joystick). A
    # coarser planning dt desyncs the policy's gait phase and causes
    # stutter-stepping. ~0.5x realtime on 8 cores.
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

    # Velocity commands bounded to the RL policy's training range
    config.u_min = [-1.0, -0.8, -1.0]
    config.u_max = [1.0, 0.8, 1.0]

    cem = spc_py.CEM(env, task, policy, config)

    def custom_init(m_py, d_py, spc_env):
        import mujoco
        from utils import init_env_state

        init_env_state(m_py, d_py, spc_env, keyframe_name="home")

        # Set soccer ball initial position
        ball_id = mujoco.mj_name2id(m_py, mujoco.mjtObj.mjOBJ_BODY, "soccer_ball")
        if ball_id != -1:
            jnt_id = m_py.body_jntadr[ball_id]
            qpos_adr = m_py.jnt_qposadr[jnt_id]
            d_py.qpos[qpos_adr:qpos_adr + 3] = [2.0, 0.5, 0.09]
            d_py.qpos[qpos_adr + 3:qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

        # Set goal position via mocap body
        if m_py.nmocap > 0:
            d_py.mocap_pos[0] = [4.75, 0.0, 0.05]
            d_py.mocap_quat[0] = [1.0, 0.0, 0.0, 0.0]

        mujoco.mj_forward(m_py, d_py)
        spc_env.set_qpos(d_py.qpos)
        spc_env.set_mocap_pos(0, d_py.mocap_pos[0])
        spc_env.set_mocap_quat(0, d_py.mocap_quat[0])
        spc_env.forward()

    run_interactive(
        env,
        cem,
        model_path,
        sim_dt=0.02,
        sim_steps_per_replan=10,
        init_kwargs={"custom_init_fn": custom_init},
        record=args.record,
    )


if __name__ == "__main__":
    main()
