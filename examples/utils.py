import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from video import VideoRecorder


def _derive_recording_name(model_path):
    """Derive a stable task-like recording name from a model path."""
    model_dir = os.path.basename(os.path.dirname(model_path))
    model_stem = os.path.splitext(os.path.basename(model_path))[0]

    if model_stem == "scene":
        return model_dir
    if model_stem.startswith("scene_"):
        return f"{model_dir}_{model_stem[len('scene_'):]}"
    return model_stem


def init_env_state(m_py, d_py, env, keyframe_name="knees_bent", mocap_defaults=None):
    """Generically initialize the environment to a keyframe and sync to C++."""
    # Reset to keyframe if available
    if m_py.nkey > 0:
        key_id = mujoco.mj_name2id(m_py, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
        if key_id < 0:
            key_id = 0  # Default to first keyframe
        mujoco.mj_resetDataKeyframe(m_py, d_py, key_id)
    else:
        mujoco.mj_resetData(m_py, d_py)

    # Set optional mocap defaults
    if mocap_defaults:
        for i, (pos, quat) in mocap_defaults.items():
            if i < m_py.nmocap:
                if pos is not None:
                    d_py.mocap_pos[i] = pos
                if quat is not None:
                    d_py.mocap_quat[i] = quat

    mujoco.mj_forward(m_py, d_py)

    # Sync to C++ env
    env.set_qpos(d_py.qpos)
    env.set_qvel(d_py.qvel)
    env.set_ctrl(d_py.ctrl)

    for i in range(m_py.nmocap):
        env.set_mocap_pos(i, d_py.mocap_pos[i])
        env.set_mocap_quat(i, d_py.mocap_quat[i])

    env.forward()


def run_interactive(
    env,
    optimizer,
    model_path,
    sim_dt=0.02,
    sim_steps_per_replan=10,
    init_kwargs=None,
    record=False,
    record_dir=None,
    record_size=(720, 480),
    record_name=None,
):
    """Generic interactive simulation loop for SPC tasks.

    Set ``record=True`` to save an mp4 of the viewer to ``record_dir``
    (default: ``<repo>/recordings``). The filename is prefixed with
    ``record_name`` or a task-like name derived from ``model_path``.
    Recording captures one frame per replan step, so the video plays at
    ``1 / sim_dt`` fps (real time).
    """
    print("Starting interactive simulation...")
    print("Double click the target/goal marker and right-click drag to move it.")

    m_py = mujoco.MjModel.from_xml_path(model_path)
    d_py = mujoco.MjData(m_py)

    init_kwargs = init_kwargs or {}
    custom_init_fn = init_kwargs.pop("custom_init_fn", None)
    if custom_init_fn:
        custom_init_fn(m_py, d_py, env)
    else:
        init_env_state(m_py, d_py, env, **init_kwargs)

    recorder = None
    renderer = None
    if record:
        width, height = record_size
        if record_dir is None:
            record_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../recordings"))
        if record_name is None:
            record_name = _derive_recording_name(model_path)
        recorder = VideoRecorder(
            output_dir=record_dir,
            width=width,
            height=height,
            fps=1.0 / sim_dt,
            filename_prefix=record_name,
        )
        # The offscreen buffer must be large enough for the requested resolution.
        m_py.vis.global_.offwidth = width
        m_py.vis.global_.offheight = height
        if recorder.start():
            renderer = mujoco.Renderer(m_py, height=height, width=width)
        else:
            recorder = None

    try:
        with mujoco.viewer.launch_passive(m_py, d_py) as viewer:
            while viewer.is_running():
                step_start = time.time()

                # Read all mocap bodies from Python viewer and send to C++ SpcEnv
                for i in range(m_py.nmocap):
                    env.set_mocap_pos(i, np.array(d_py.mocap_pos[i]))
                    env.set_mocap_quat(i, np.array(d_py.mocap_quat[i]))

                # Step C++ MPC and simulation
                compute_start = time.time()
                env.step_mpc(optimizer, sim_steps_per_replan)
                compute_time = time.time() - compute_start

                # Sync back to Python viewer
                d_py.qpos[:] = env.get_qpos()
                mujoco.mj_forward(m_py, d_py)
                viewer.sync()

                # Capture a frame from the viewer camera if recording
                if renderer is not None and recorder.is_recording:
                    renderer.update_scene(d_py, viewer.cam)
                    recorder.add_frame(renderer.render().tobytes())

                # Real-time pacing
                elapsed_compute = time.time() - step_start
                time_until_next_step = sim_dt - elapsed_compute
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                total_elapsed = time.time() - step_start
                rtf = sim_dt / total_elapsed if total_elapsed > 0 else 0.0
                sys.stdout.write(f"\rRealtime rate: {rtf:.2f}x | Compute time: {compute_time * 1000:.2f}ms   ")
                sys.stdout.flush()
    finally:
        if recorder is not None:
            recorder.stop()

