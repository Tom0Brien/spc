import os

import spc_py

from utils import run_interactive


def main():
    model_path = os.path.join(os.path.dirname(__file__), "../models/particle/scene.xml")

    # Initialize the C++ environment
    env = spc_py.SpcEnv(model_path)

    # You can now configure the C++ task dynamically from Python!
    task_params = {"pos_weight": 5.0, "vel_weight": 0.1, "ctrl_weight": 0.1}
    task = spc_py.create_task("ParticleTask", env, task_params)

    # Configure and create CEM optimizer
    config = spc_py.CEMConfig()
    config.num_samples = 64
    config.num_elites = 16
    config.num_knots = 10
    config.num_iterations = 3
    config.plan_horizon_steps = 10
    config.control_dim = env.nu
    config.obs_dim = 0
    config.sigma_init = 0.5

    # We pass None for the policy since the Particle task has no base policy
    optimizer = spc_py.CEM(env, task, None, config)

    # Run simulation loop
    run_interactive(
        env, optimizer, model_path, sim_dt=0.02, sim_steps_per_replan=10, init_kwargs={"keyframe_name": "home"}
    )


if __name__ == "__main__":
    main()
