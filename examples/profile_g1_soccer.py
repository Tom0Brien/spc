import os
import time

# Prevent OpenMP thread migration for better L1/L2 cache locality
os.environ["OMP_PROC_BIND"] = "true"
os.environ["OMP_PLACES"] = "cores"

import spc_py

def main():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/g1/scene_soccer.xml"))
    mlp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../policies/g1_navigation.mlp"))

    env = spc_py.SpcEnv(model_path)
    
    task_params = {
        "action_scale": 0.5,
        "gait_freq": 1.5,
        "target_height": 0.75,
        "standoff_distance": 0.5,
        "ball_goal_weight": 1.0,
        "pos_weight": 0.3,
        "ori_weight": 0.2,
        "height_weight": 0.5,
        "ctrl_weight": 0.01,
    }
    
    task = spc_py.create_task("G1Soccer", env, task_params)
    policy = spc_py.MLPPolicy(mlp_path)

    config = spc_py.CEMConfig()
    config.num_samples = 32
    config.num_elites = 8
    config.num_knots = 4
    config.num_iterations = 1
    config.plan_horizon_steps = 25
    config.sim_substeps = 5  # dt=0.004, ctrl_dt=0.02 -> 5 substeps
    config.control_dim = 3
    config.obs_dim = 103
    config.num_threads = 8
    config.sigma_init = 0.5
    config.sigma_min = 0.05
    config.explore_fraction = 0.5

    cem = spc_py.CEM(env, task, policy, config)

    # Initialize env
    import mujoco
    from utils import init_env_state
    
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    init_env_state(mj_model, mj_data, env, keyframe_name="knees_bent")
    
    # Custom init logic for soccer
    ball_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "soccer_ball")
    if ball_id != -1:
        jnt_id = mj_model.body_jntadr[ball_id]
        qpos_adr = mj_model.jnt_qposadr[jnt_id]
        mj_data.qpos[qpos_adr:qpos_adr+3] = [2.0, 0.5, 0.117]
        mj_data.qpos[qpos_adr+3:qpos_adr+7] = [1.0, 0.0, 0.0, 0.0]
        
    if mj_model.nmocap > 0:
        mj_data.mocap_pos[0] = [4.75, 0.0, 0.05]
        mj_data.mocap_quat[0] = [1.0, 0.0, 0.0, 0.0]
        
    mujoco.mj_forward(mj_model, mj_data)
    env.set_qpos(mj_data.qpos)
    env.set_mocap_pos(0, mj_data.mocap_pos[0])
    env.set_mocap_quat(0, mj_data.mocap_quat[0])
    env.forward()
    
    print("Warming up...")
    for _ in range(5):
        env.step_mpc(cem, 5)
        
    print("Profiling...")
    start_time = time.time()
    num_plans = 100
    for _ in range(num_plans):
        env.step_mpc(cem, 5)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"Total time for {num_plans} plans: {duration:.4f}s")
    print(f"Average time per plan: {(duration / num_plans) * 1000:.2f}ms")
    
    print(f"Target time for real-time: 20.00ms")
    print(f"Current speed: {20.0 / ((duration / num_plans) * 1000):.2f}x real-time")

if __name__ == "__main__":
    main()
