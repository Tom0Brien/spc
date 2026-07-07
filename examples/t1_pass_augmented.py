"""Booster T1 Pass (augmented) MPC example using SPC C++ backend.

Uses the T1PassAugmented task with CEM optimization. The control vector is
extended to 15 dims:
  - control[0:3]  : velocity commands (vx, vy, vtheta) for the RL policy
  - control[3:15] : residual adjustments for the 12 leg joint targets

A trained ONNX locomotion policy converts the velocity commands to motor
targets at each step; CEM additionally optimizes the leg residuals for fine
kicking adjustments.

Usage: python3 examples/t1_pass_augmented.py [--no_policy]
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

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/t1/scene_soccer.xml"))
    policy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../policies/t1_navigation.onnx"))

    print(f"Loading model from {model_path}")
    env = spc_py.SpcEnv(model_path)

    task_params = {
        "action_scale": 1.0,
        "gait_freq": 1.5,
        "target_height": 0.665,

        # Pass-specific weights
        "standoff_distance": 0.35,  # augmented: leg-swing residuals extend the foot's reach
        "ball_goal_weight": 1.0,
        "pos_weight": 0.3,
        "ori_weight": 0.2,
        "height_weight": 0.5,
        "upright_weight": 2.0,  # penalize trunk tilt (prevents falls)
        "behind_weight": 0.0,  # off: residual kicks need close contact
        "ctrl_weight": 0.01,
        # Experiment: raise the policy-command clamp above the default
        # {1.0, 0.8, 1.0} (must match the CEM velocity bounds below).
        "vx_limit": 1.5,
        "vy_limit": 1.2,
        "vtheta_limit": 1.5,

        # Augmented (leg residual) params
        # T1 joint order is head(2), arms(8), waist(1), left leg(6), right leg(6),
        # so the 12 leg joints start at index 11.
        "leg_joint_start": 11,
        "leg_joint_count": 12,
        # Tuned: 0.5 -> 0.2 makes kicks cheap enough to beat pure dribbling
        # (median time-to-goal 8.2s -> 7.3s, worst case 15.6s -> 8.1s over a
        # 6-episode eval; no falls).
        "residual_weight": 0.2,
        # Residuals fade to zero beyond gate_far and reach full strength within
        # gate_near, keeping the far-field approach velocity-only.
        "gate_near": 0.45,
        "gate_far": 0.75,
    }

    # We pass the same policy used for navigation, as it accepts velocity commands
    task = spc_py.create_task("T1PassAugmented", env, task_params)

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
    config.num_samples = 24
    config.num_elites = 12
    config.num_knots = 4
    config.num_iterations = 1
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
    # replan desynchronize from the gait (verified to hurt the task).
    config.replan_shift_steps = 0
    config.elite_keep = 0

    # Per-dimension bounds and sampling spread:
    config.u_min = [-1.5, -1.2, -1.5] + [-0.3] * 12
    config.u_max = [1.5, 1.2, 1.5] + [0.3] * 12
    # Residual sigma 0.15 -> 0.25 (tuned with residual_weight=0.2): wider kick
    # exploration finds contact earlier without destabilizing the gait.
    config.sigma_init_per_dim = [0.5] * 3 + [0.25] * 12

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
            d_py.qpos[qpos_adr:qpos_adr + 3] = [3.0, 0.5, 0.117]
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
