"""Profile G1 Navigation."""

import os
import time

os.environ["OMP_PROC_BIND"] = "true"
os.environ["OMP_PLACES"] = "cores"
import spc_py


def main():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/g1/scene_navigation.xml"))
    policy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../policies/g1_navigation.onnx"))

    env = spc_py.SpcEnv(model_path)
    task = spc_py.create_task("G1Navigation", env)
    policy = spc_py.ONNXPolicy(policy_path)

    config = spc_py.CEMConfig()
    config.num_samples = 32
    config.num_elites = 8
    config.num_knots = 4
    config.num_iterations = 1
    config.plan_horizon_steps = 5
    config.sim_substeps = 10
    config.control_dim = 3
    config.obs_dim = 103
    config.num_threads = 8

    cem = spc_py.CEM(env, task, policy, config)

    N = 100
    # Warmup
    env.step_mpc(cem, 10)

    start = time.time()
    for _ in range(N):
        env.step_mpc(cem, 10)

    avg = (time.time() - start) / N
    print(f"With policy: avg {avg * 1000:.1f}ms over {N} steps ({0.02 / avg:.2f}x realtime)")


if __name__ == "__main__":
    main()
