#include "spc/tasks/t1_navigation.h"

#include <cmath>
#include <cstring>
#include <iostream>

namespace spc {
namespace tasks {

T1Navigation::T1Navigation(mjModel* model, const spc::core::TaskConfig& config) {
    std::string gyro_name =
        config.string_params.count("gyro_name") ? config.string_params.at("gyro_name") : "gyro";
    std::string linvel_name =
        config.string_params.count("linvel_name") ? config.string_params.at("linvel_name") : "local_linvel";
    std::string imu_site =
        config.string_params.count("imu_site") ? config.string_params.at("imu_site") : "imu";

    int gyro_id = mj_name2id(model, mjOBJ_SENSOR, gyro_name.c_str());
    int linvel_id = mj_name2id(model, mjOBJ_SENSOR, linvel_name.c_str());

    gyro_adr_ = (gyro_id >= 0) ? model->sensor_adr[gyro_id] : -1;
    linvel_adr_ = (linvel_id >= 0) ? model->sensor_adr[linvel_id] : -1;

    imu_site_id_ = mj_name2id(model, mjOBJ_SITE, imu_site.c_str());

    // Default pose from mujoco_playground T1 Joystick "home" keyframe
    // These are the default joint positions the RL policy was trained with
    const float dp[23] = {
        0.0f,  0.0f,                              // head (2)
        0.0f,  -1.4f, 0.0f, -0.4f,                // left arm (4)
        0.0f,  1.4f,  0.0f, 0.4f,                 // right arm (4)
        0.0f,                                     // waist (1)
        -0.2f, 0.0f,  0.0f, 0.4f,  -0.2f, 0.0f,   // left leg (6)
        -0.2f, 0.0f,  0.0f, 0.4f,  -0.2f, 0.0f    // right leg (6)
    };
    std::memcpy(default_pose_, dp, sizeof(dp));

    action_scale_ = config.numeric_params.count("action_scale")
                        ? static_cast<float>(config.numeric_params.at("action_scale"))
                        : 1.0f;
    gait_freq_ =
        config.numeric_params.count("gait_freq") ? static_cast<float>(config.numeric_params.at("gait_freq")) : 1.5f;

    // Velocity command limits (symmetric): defaults match the RL policy training bounds.
    vel_limit_[0] =
        config.numeric_params.count("vx_limit") ? static_cast<float>(config.numeric_params.at("vx_limit")) : 1.0f;
    vel_limit_[1] =
        config.numeric_params.count("vy_limit") ? static_cast<float>(config.numeric_params.at("vy_limit")) : 0.8f;
    vel_limit_[2] = config.numeric_params.count("vtheta_limit")
                        ? static_cast<float>(config.numeric_params.at("vtheta_limit"))
                        : 1.0f;
    target_height_ = config.numeric_params.count("target_height") ? config.numeric_params.at("target_height") : 0.665;

    pos_weight_ = config.numeric_params.count("pos_weight") ? config.numeric_params.at("pos_weight") : 1.0;
    ori_weight_ = config.numeric_params.count("ori_weight") ? config.numeric_params.at("ori_weight") : 1.0;
    upright_weight_ = config.numeric_params.count("upright_weight") ? config.numeric_params.at("upright_weight") : 2.0;
    height_weight_ = config.numeric_params.count("height_weight") ? config.numeric_params.at("height_weight") : 0.5;
    ctrl_weight_ = config.numeric_params.count("ctrl_weight") ? config.numeric_params.at("ctrl_weight") : 0.01;

    // Store joint limits (joints 1-23, skipping floating base joint 0)
    for (int i = 0; i < 23; ++i) {
        jnt_range_low_[i] = model->jnt_range[(i + 1) * 2];
        jnt_range_high_[i] = model->jnt_range[(i + 1) * 2 + 1];
    }
}

void T1Navigation::GetObservation(const mjModel* model, const mjData* data, float* obs_out) const {
    int idx = 0;

    // 1. Local linear velocity of trunk (3)
    if (linvel_adr_ >= 0) {
        for (int i = 0; i < 3; ++i)
            obs_out[idx++] = static_cast<float>(data->sensordata[linvel_adr_ + i]);
    } else {
        for (int i = 0; i < 3; ++i)
            obs_out[idx++] = 0.0f;
    }

    // 2. Gyro (angular velocity) of trunk (3)
    if (gyro_adr_ >= 0) {
        for (int i = 0; i < 3; ++i)
            obs_out[idx++] = static_cast<float>(data->sensordata[gyro_adr_ + i]);
    } else {
        for (int i = 0; i < 3; ++i)
            obs_out[idx++] = 0.0f;
    }

    // 3. Gravity vector in trunk frame (3)
    // gravity = site_xmat[imu].T @ [0, 0, -1]
    if (imu_site_id_ >= 0) {
        const mjtNum* xmat = data->site_xmat + 9 * imu_site_id_;
        // xmat is 3x3 row-major.
        // gravity = xmat.T @ [0,0,-1] = [-xmat[6], -xmat[7], -xmat[8]]
        obs_out[idx++] = static_cast<float>(-xmat[6]);
        obs_out[idx++] = static_cast<float>(-xmat[7]);
        obs_out[idx++] = static_cast<float>(-xmat[8]);
    } else {
        obs_out[idx++] = 0.0f;
        obs_out[idx++] = 0.0f;
        obs_out[idx++] = -1.0f;
    }

    // 4. Command (3) - stored in obs slots, will be filled by ApplyControl
    // For GetObservation, we use zeros (the actual command comes from the MPC control)
    for (int i = 0; i < 3; ++i)
        obs_out[idx++] = 0.0f;

    // 5. Joint angles - default_pose (23)
    // Robot joints are qpos[7:30]; head joints (first 2) are zeroed to match
    // the sim2sim T1 deployment.
    for (int i = 0; i < 23; ++i) {
        float dq = static_cast<float>(data->qpos[7 + i]) - default_pose_[i];
        obs_out[idx++] = (i < 2) ? 0.0f : dq;
    }

    // 6. Joint velocities (23)
    // Robot joint velocities are qvel[6:29]; head joints zeroed as above.
    for (int i = 0; i < 23; ++i) {
        float v = static_cast<float>(data->qvel[6 + i]);
        obs_out[idx++] = (i < 2) ? 0.0f : v;
    }

    // 7. Last action (23)
    // last_act = (ctrl - default_pose) / action_scale
    for (int i = 0; i < 23; ++i) {
        float motor_target = static_cast<float>(data->ctrl[i]);
        obs_out[idx++] = (motor_target - default_pose_[i]) / action_scale_;
    }

    // 8. Gait phase (4): cos(phase_left), cos(phase_right), sin(phase_left), sin(phase_right)
    float phase_base = 2.0f * static_cast<float>(M_PI) * static_cast<float>(data->time) * gait_freq_;
    float phase_left = phase_base;
    float phase_right = phase_base + static_cast<float>(M_PI);
    obs_out[idx++] = std::cos(phase_left);
    obs_out[idx++] = std::cos(phase_right);
    obs_out[idx++] = std::sin(phase_left);
    obs_out[idx++] = std::sin(phase_right);

    // Total: 3+3+3+3+23+23+23+4 = 85
}

double T1Navigation::RunningCost(const mjModel* model, const mjData* data, const float* control) const {
    // Goal from mocap body 0
    const mjtNum* goal_pos = data->mocap_pos;    // x, y, z
    const mjtNum* goal_quat = data->mocap_quat;  // w, x, y, z

    // Robot position (x, y from floating base)
    double robot_x = data->qpos[0];
    double robot_y = data->qpos[1];

    // Position error (2D)
    double dx = robot_x - goal_pos[0];
    double dy = robot_y - goal_pos[1];
    double position_cost = dx * dx + dy * dy;

    // Yaw orientation error
    // Robot quaternion: qpos[3:7] = [qw, qx, qy, qz]
    const mjtNum* robot_quat = data->qpos + 3;
    // Quaternion difference for yaw: use qz component of relative quaternion
    mjtNum goal_inv[4];
    mju_negQuat(goal_inv, goal_quat);
    mjtNum quat_diff[4];
    mju_mulQuat(quat_diff, robot_quat, goal_inv);
    double orientation_cost = quat_diff[3] * quat_diff[3];  // qz^2

    // Upright cost: penalize tilt from vertical
    // Use trunk orientation to check uprightness
    // gravity_z should be close to -1 if upright
    double upright_cost = 0.0;
    if (imu_site_id_ >= 0) {
        const mjtNum* xmat = data->site_xmat + 9 * imu_site_id_;
        // gravity = xmat.T @ [0,0,-1]
        double gx = -xmat[6];
        double gy = -xmat[7];
        // x^2 + y^2 of gravity vector (0 when upright)
        upright_cost = gx * gx + gy * gy;
    }

    // Height cost: keep trunk at target height
    double height_cost = 0.0;
    if (imu_site_id_ >= 0) {
        double trunk_z = data->site_xpos[3 * imu_site_id_ + 2];
        double height_err = trunk_z - target_height_;
        height_cost = height_err * height_err;
    }

    // Control regularization (velocity commands)
    double ctrl_cost = 0.0;
    for (int i = 0; i < 3; ++i) {
        ctrl_cost += control[i] * control[i];
    }

    return pos_weight_ * position_cost + ori_weight_ * orientation_cost + upright_weight_ * upright_cost +
           height_weight_ * height_cost + ctrl_weight_ * ctrl_cost;
}

double T1Navigation::TerminalCost(const mjModel* model, const mjData* data) const {
    float zero_ctrl[3] = {0.0f, 0.0f, 0.0f};
    return RunningCost(model, data, zero_ctrl);
}

void T1Navigation::ApplyControl(const mjModel* model, mjData* data, const float* action) const {
    // action[0:3] = velocity command (vx, vy, vtheta)

    // Build the 85-dim observation with the velocity command inserted
    float obs[96];  // 85 needed, use 96 for alignment
    GetObservation(model, data, obs);
    // Overwrite the command slots (indices 9, 10, 11), clamping to velocity limits.
    float cmd_norm_sq = 0.0f;
    for (int i = 0; i < 3; ++i) {
        float cmd = action[i];
        if (cmd < -vel_limit_[i])
            cmd = -vel_limit_[i];
        if (cmd > vel_limit_[i])
            cmd = vel_limit_[i];
        obs[9 + i] = cmd;
        cmd_norm_sq += cmd * cmd;
    }

    // Standing: the policy was trained with the gait phase pinned to pi when
    // the command is (near) zero.
    if (cmd_norm_sq < 0.01f * 0.01f) {
        obs[81] = -1.0f;  // cos(pi)
        obs[82] = -1.0f;
        obs[83] = 0.0f;   // sin(pi)
        obs[84] = 0.0f;
    }

    // Run ONNX policy to get 23 motor actions
    float policy_action[32];  // 23 needed
    if (policy_) {
        policy_->ComputeAction(obs, 85, policy_action, 23);
    } else {
        for (int i = 0; i < 23; ++i)
            policy_action[i] = 0.0f;
    }

    // Convert policy output to motor targets: default_pose + action * scale
    for (int i = 0; i < 23; ++i) {
        float motor_target = default_pose_[i] + policy_action[i] * action_scale_;

        // Clamp to joint limits
        if (motor_target < jnt_range_low_[i])
            motor_target = jnt_range_low_[i];
        if (motor_target > jnt_range_high_[i])
            motor_target = jnt_range_high_[i];

        data->ctrl[i] = motor_target;
    }
}

}  // namespace tasks
}  // namespace spc

#include "spc/core/task_factory.h"
REGISTER_TASK("T1Navigation", spc::tasks::T1Navigation, T1Navigation)
