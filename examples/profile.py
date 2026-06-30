import sys
import os
import time
os.environ["OMP_PROC_BIND"] = "true"
os.environ["OMP_PLACES"] = "cores"
import mujoco
import spc_py
env = spc_py.SpcEnv("models/franka_push/scene.xml")
task = spc_py.create_task("FrankaPushTask", env)
policy = spc_py.ONNXPolicy("policies/franka_push.onnx")
config = spc_py.CEMConfig()
config.num_samples = 96
config.num_elites = 8
config.num_knots = 6
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

N = 10
# Warmup
env.step_mpc(cem, 2)

start = time.time()
for _ in range(N):
    env.step_mpc(cem, 2)
avg = (time.time() - start) / N
print(f"With policy: avg {avg*1000:.1f}ms over {N} steps ({0.02/avg:.2f}x realtime)")

cem_no_policy = spc_py.CEM(env, task, None, config)
env.step_mpc(cem_no_policy, 2)
start = time.time()
for _ in range(N):
    env.step_mpc(cem_no_policy, 2)
avg = (time.time() - start) / N
print(f"Without policy: avg {avg*1000:.1f}ms over {N} steps ({0.02/avg:.2f}x realtime)")
