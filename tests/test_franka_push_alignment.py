import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

sys.path.insert(0, "/home/tom/OneDrive/Phd/Papers/spc_rl/pg_spc/hydrax")
sys.path.insert(0, "/home/tom/OneDrive/Phd/Papers/spc_rl/pg_spc/mujoco_playground")

from hydrax.tasks.franka.franka_push_geometry import FrankaPushGeometry

# Add build directory to path to find spc_py module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
import spc_py

def main():
    print("Testing Franka Push alignment...")
    
    # 1. Initialize Python MJX Task
    py_task = FrankaPushGeometry(geometry="cube", use_rl_policy=False)
    m_py = py_task.mj_model
    d_py = mujoco.MjData(m_py)
    
    # Initialize to a non-zero state
    if m_py.nkey > 0:
        mujoco.mj_resetDataKeyframe(m_py, d_py, 0)
        
    d_py.qpos += np.random.uniform(-0.1, 0.1, m_py.nq)
    d_py.qvel += np.random.uniform(-0.1, 0.1, m_py.nv)
    d_py.ctrl += np.random.uniform(-1.0, 1.0, m_py.nu)
    
    mujoco.mj_forward(m_py, d_py)
    
    # Convert to MJX
    mx = mjx.put_model(m_py)
    dx = mjx.put_data(m_py, d_py)
    
    # Call JAX task functions
    # For cost, wait, running_cost takes dx? Let's check hydrax signature.
    # hydrax tasks usually have self.running_cost(dx, action) or something.
    try:
        # Some envs have running_cost(state, action)
        pass 
    except Exception as e:
        pass
        
    # Wait, the observation in hydrax is structured, then flattened by brax.
    # Let's just compare the observation explicitly since it's easier.
    obs_jax = py_task._get_obs(dx)
    obs_flat_py = jnp.concatenate([jnp.atleast_1d(x) for x in jax.tree_util.tree_leaves(obs_jax)]).flatten()
    
    # 2. Initialize C++ Task independently
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/franka_push/scene.xml"))
    env = spc_py.SpcEnv(model_path)
    
    # Set C++ state to match JAX/Python state exactly
    env.set_qpos(d_py.qpos)
    env.set_qvel(d_py.qvel)
    env.set_ctrl(d_py.ctrl)
    env.forward()
    
    cpp_task = spc_py.FrankaPushTask(env)
    
    # 3. Call C++ observation and running cost
    obs_cpp = np.array(cpp_task.get_observation(env, 48))
    
    # We must pass the control as a float32 numpy array
    ctrl_np = np.array(d_py.ctrl, dtype=np.float32)
    cost_cpp = cpp_task.running_cost(env, ctrl_np)
    
    # Cost from JAX
    cost_jax = py_task.running_cost(dx, dx.ctrl)
    
    print("--------------------------------------------------")
    print(f"JAX observation size: {obs_flat_py.shape}")
    print(f"C++ observation size: {obs_cpp.shape}")
    
    # Compare observation arrays
    max_obs_diff = np.max(np.abs(obs_flat_py - obs_cpp))
    print(f"Max observation difference: {max_obs_diff:.6f}")
    
    print(f"JAX running cost: {float(cost_jax):.6f}")
    print(f"C++ running cost: {cost_cpp:.6f}")
    
    if max_obs_diff < 1e-4 and abs(float(cost_jax) - cost_cpp) < 1e-2:
        print("SUCCESS! Implementations match.")
    else:
        print("WARNING: Implementations do not match perfectly.")
    
if __name__ == "__main__":
    main()
