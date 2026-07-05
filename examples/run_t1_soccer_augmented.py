"""Booster T1 Soccer (augmented) MPC example using SPC C++ backend.

Uses the T1SoccerAugmented task with CEM optimization. The control vector is
extended to 15 dims:
  - control[0:3]  : velocity commands (vx, vy, vtheta) for the RL policy
  - control[3:15] : residual adjustments for the 12 leg joint targets

A trained ONNX locomotion policy converts the velocity commands to motor
targets at each step; CEM additionally optimizes the leg residuals for fine
kicking adjustments.

Usage: python3 examples/run_t1_soccer_augmented.py [--no_policy]
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

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/t1/scene_soccer.xml"))
    policy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../policies/t1_navigation.onnx"))

    print(f"Loading model from {model_path}")
    env = spc_py.SpcEnv(model_path)

    task_params = {
        "action_scale": 1.0,
        "gait_freq": 1.5,
        "target_height": 0.665,

        # Soccer-specific weights
        "standoff_distance": 0.35,  # augmented: leg-swing residuals extend the foot's reach
        "ball_goal_weight": 1.0,
        "pos_weight": 0.3,
        "ori_weight": 0.2,
        "height_weight": 0.5,
        "upright_weight": 2.0,  # penalize trunk tilt (prevents falls)
        "behind_weight": 0.0,  # off: residual kicks need close contact
        "ctrl_weight": 0.01,

        # Augmented (leg residual) params
        # T1 joint order is head(2), arms(8), waist(1), left leg(6), right leg(6),
        # so the 12 leg joints start at index 11.
        "leg_joint_start": 11,
        "leg_joint_count": 12,
        "residual_weight": 0.5,
    }

    # We pass the same policy used for navigation, as it accepts velocity commands
    task = spc_py.create_task("T1SoccerAugmented", env, task_params)

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
    config.plan_horizon_steps = 32  # longer lookahead so CEM can discover the leg-swing kick
    config.sim_substeps = 10  # dt=0.002, ctrl_dt=0.02 -> 10 substeps (mujoco_playground settings)
    config.control_dim = 15  # vx, vy, vtheta + 12 leg residuals
    config.obs_dim = 85
    config.num_threads = 8
    config.sigma_init = 0.5
    config.sigma_min = 0.05
    config.explore_fraction = 0.5
    # Warm-start shift and elite reuse stay off: leg residuals are
    # gait-phase-locked, so re-timed means and stale elites from the previous
    # replan desynchronize from the gait (verified to hurt the task).
    config.replan_shift_steps = 0
    config.elite_keep = 0

    # Per-dimension bounds and sampling spread:
    # velocity commands in [-1, 1] (vy in [-0.8, 0.8]), leg residuals in [-0.3, 0.3] rad
    config.u_min = [-1.0, -0.8, -1.0] + [-0.3] * 12
    config.u_max = [1.0, 0.8, 1.0] + [0.3] * 12
    config.sigma_init_per_dim = [0.5] * 3 + [0.15] * 12

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
            d_py.qpos[qpos_adr:qpos_adr + 3] = [2.0, 0.5, 0.117]
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
    )


if __name__ == "__main__":
    main()
