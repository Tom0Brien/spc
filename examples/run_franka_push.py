import argparse
import os

import mujoco
import mujoco.viewer
import numpy as np
import spc_py

from utils import run_interactive

# Prevent OpenMP thread migration for better L1/L2 cache locality
os.environ["OMP_PROC_BIND"] = "true"
os.environ["OMP_PLACES"] = "cores"


def init_hydrax_state(m_py, d_py, env):
    # Home keyframe values (extracted from model)
    home_qpos = np.array(
        [
            -0.182772,
            0.146282,
            0.172246,
            -2.24238,
            -0.0788546,
            2.45127,
            0.0160022,  # arm (7)
            0.8,
            0.8,
            0.8,
            0.8,
            0.8,
            0.8,  # gripper (6)
            0.56784,
            -0.0253974,
            0.0306525,  # box position (3)
            0.0,
            0.0,
            0.0,
            1.0,  # box quaternion wxyz (4)
        ]
    )
    home_ctrl = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.82])

    # Joint limits and perturbation
    jnt_range = np.array(
        [
            [-2.8973, 2.8973],
            [-1.7628, 1.7628],
            [-2.8973, 2.8973],
            [-3.0718, -0.0698],
            [-2.8973, 2.8973],
            [-0.0175, 3.7525],
            [-2.8973, 2.8973],
        ]
    )
    joint_range_init_percent_limit = np.array([0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3])

    # Randomize arm joints
    arm_noise = 0.3 * np.random.uniform(
        low=jnt_range[:, 0] * joint_range_init_percent_limit,
        high=jnt_range[:, 1] * joint_range_init_percent_limit,
        size=(7,),
    )

    # Initial object position from keyframe
    init_obj_pos = home_qpos[13:16]

    # Sampling bounds
    OBJ_SAMPLE_MIN = np.array([0.4, -0.2, -0.005])
    OBJ_SAMPLE_MAX = np.array([0.65, 0.2, 0.04])

    # Box position randomization
    box_offset = 0.15
    box_x = np.random.uniform(init_obj_pos[0] - box_offset, init_obj_pos[0] + box_offset)
    box_y = np.random.uniform(init_obj_pos[1] - box_offset, init_obj_pos[1] + box_offset)
    box_x = np.clip(box_x, OBJ_SAMPLE_MIN[0], OBJ_SAMPLE_MAX[0])
    box_y = np.clip(box_y, OBJ_SAMPLE_MIN[1], OBJ_SAMPLE_MAX[1])

    # Box quaternion: random rotation around Z axis
    box_theta = np.random.uniform(0, 2 * np.pi)
    box_quat = np.array([np.cos(box_theta / 2), 0.0, 0.0, np.sin(box_theta / 2)])

    # Target position randomization
    target_offset = 0.05
    target_x = np.random.uniform(init_obj_pos[0] - target_offset, init_obj_pos[0] + target_offset)
    target_y = np.random.uniform(init_obj_pos[1] - target_offset, init_obj_pos[1] + target_offset)
    target_x = np.clip(target_x, OBJ_SAMPLE_MIN[0], OBJ_SAMPLE_MAX[0])
    target_y = np.clip(target_y, OBJ_SAMPLE_MIN[1], OBJ_SAMPLE_MAX[1])
    target_z = init_obj_pos[2]

    # Target quaternion
    target_theta = np.random.uniform(0, 90 * np.pi / 180)
    target_quat = np.array([np.cos(target_theta / 2), 0.0, 0.0, np.sin(target_theta / 2)])

    # Build qpos
    qpos = home_qpos.copy()
    qpos[:7] += arm_noise
    qpos[13] = box_x
    qpos[14] = box_y
    qpos[16:20] = box_quat

    d_py.qpos[:] = qpos
    d_py.qvel[:] = 0.0
    d_py.ctrl[:] = home_ctrl

    d_py.mocap_pos[0] = [target_x, target_y, target_z]
    d_py.mocap_quat[0] = target_quat
    mujoco.mj_forward(m_py, d_py)

    env.set_qpos(d_py.qpos)
    env.set_qvel(d_py.qvel)
    env.set_ctrl(d_py.ctrl)
    env.set_mocap_pos(0, d_py.mocap_pos[0])
    env.set_mocap_quat(0, d_py.mocap_quat[0])
    env.forward()


def main():
    parser = argparse.ArgumentParser(description="Run Franka Push with RL policy + CEM residuals in C++")
    parser.add_argument("--no_policy", action="store_true", default=False, help="Disable the RL policy base action")
    args = parser.parse_args()

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/franka_push/scene.xml"))
    # Hand-rolled MLP inference (converted from franka_push.onnx): identical
    # outputs, but skips ONNX Runtime session overhead, which dominates for
    # this small (17k param) policy and is what makes policy+CEM realtime.
    policy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../policies/franka_push.mlp"))

    print(f"Loading model from {model_path}")
    env = spc_py.SpcEnv(model_path)

    # You can now configure the C++ task dynamically from Python!
    task_params = {
        "obj_target_weight": 10.0,
        "gripper_obj_weight": 5.0,
        "orientation_weight": 1.0,
        "residual_weight": 0.1,
        "action_scale": 0.1,
    }
    task = spc_py.create_task("FrankaPushTask", env, task_params)

    policy = None
    if not args.no_policy:
        print(f"Loading policy from {policy_path}")
        policy = spc_py.MLPPolicy(policy_path)

    config = spc_py.CEMConfig()
    
    config.num_samples = 48
    config.num_elites = 8
    config.num_knots = 4
    config.num_iterations = 1
    config.plan_horizon_steps = 25
    config.sim_substeps = 2
    config.control_dim = 7
    config.obs_dim = 48
    config.num_threads = 8
    config.sigma_init = 0.1
    config.sigma_min = 0.05
    config.explore_fraction = 0.5

    cem = spc_py.CEM(env, task, policy, config)

    # Initialize using the custom Franka state function before running
    m_py = mujoco.MjModel.from_xml_path(model_path)
    d_py = mujoco.MjData(m_py)
    init_hydrax_state(m_py, d_py, env)
    run_interactive(
        env, cem, model_path, sim_dt=0.02, sim_steps_per_replan=2, init_kwargs={"custom_init_fn": init_hydrax_state}
    )


if __name__ == "__main__":
    main()
