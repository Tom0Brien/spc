"""G1 Soccer MPC example using SPC C++ backend.

Uses the G1Soccer task with CEM optimization to navigate a G1 humanoid
robot to a soccer ball and push it toward a goal position.

The velocity commands (vx, vy, vtheta) are optimized by CEM, and a trained
ONNX locomotion policy converts them to motor targets at each step.

Usage: python3 examples/run_g1_soccer.py [--no_policy]
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

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/g1/scene_soccer.xml"))
    policy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../policies/g1_navigation.mlp"))

    print(f"Loading model from {model_path}")
    env = spc_py.SpcEnv(model_path)

    # You can now configure the C++ task dynamically from Python!
    task_params = {
        "action_scale": 0.5,
        "gait_freq": 1.5,
        "target_height": 0.75,
        
        # Soccer-specific weights
        "standoff_distance": 0.5,
        "ball_goal_weight": 1.0,
        "pos_weight": 0.3,
        "ori_weight": 0.2,
        "height_weight": 1.0,
        "upright_weight": 2.0,  # penalize pelvis tilt (prevents falls)
        "behind_weight": 2.0,  # stay on the far side of the ball from the goal
        "ctrl_weight": 0.01,
    }
    
    # We pass the same policy used for navigation, as it accepts velocity commands
    task = spc_py.create_task("G1Soccer", env, task_params)

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

    def custom_init(m_py, d_py, spc_env):
        import mujoco
        from utils import init_env_state
        init_env_state(m_py, d_py, spc_env, keyframe_name="knees_bent")
        
        # Set soccer ball initial position
        ball_id = mujoco.mj_name2id(m_py, mujoco.mjtObj.mjOBJ_BODY, "soccer_ball")
        if ball_id != -1:
            jnt_id = m_py.body_jntadr[ball_id]
            qpos_adr = m_py.jnt_qposadr[jnt_id]
            d_py.qpos[qpos_adr:qpos_adr+3] = [2.0, 0.5, 0.117]
            d_py.qpos[qpos_adr+3:qpos_adr+7] = [1.0, 0.0, 0.0, 0.0]
            
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
        sim_steps_per_replan=5,
        init_kwargs={"custom_init_fn": custom_init},
    )


if __name__ == "__main__":
    main()
