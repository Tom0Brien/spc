#include "spc/tasks/g1_navigation.h"

#include <cmath>
#include <cstring>
#include <iostream>

namespace spc {
namespace tasks {

G1Navigation::G1Navigation(mjModel* model, const spc::core::TaskConfig& config) {
    std::string gyro_name =
        config.string_params.count("gyro_name") ? config.string_params.at("gyro_name") : "gyro_pelvis";
    std::string linvel_name =
        config.string_params.count("linvel_name") ? config.string_params.at("linvel_name") : "local_linvel_pelvis";
    std::string pelvis_imu =
        config.string_params.count("pelvis_imu") ? config.string_params.at("pelvis_imu") : "imu_in_pelvis";
    std::string torso_imu =
        config.string_params.count("torso_imu") ? config.string_params.at("torso_imu") : "imu_in_torso";

    int gyro_id = mj_name2id(model, mjOBJ_SENSOR, gyro_name.c_str());
    int linvel_id = mj_name2id(model, mjOBJ_SENSOR, linvel_name.c_str());

    pelvis_gyro_adr_ = (gyro_id >= 0) ? model->sensor_adr[gyro_id] : -1;
    pelvis_linvel_adr_ = (linvel_id >= 0) ? model->sensor_adr[linvel_id] : -1;

    pelvis_imu_site_id_ = mj_name2id(model, mjOBJ_SITE, pelvis_imu.c_str());
    torso_site_id_ = mj_name2id(model, mjOBJ_SITE, torso_imu.c_str());

    // Default pose from mujoco_playground G1JoystickFlatTerrain
    // These are the default joint positions the RL policy was trained with
    const float dp[29] = {
        -0.312f, 0.0f,  0.0f,   0.669f, -0.363f, 0.0f,        // left leg (6)
        -0.312f, 0.0f,  0.0f,   0.669f, -0.363f, 0.0f,        // right leg (6)
        0.0f,    0.0f,  0.073f,                               // waist (3)
        0.2f,    0.2f,  0.0f,   0.6f,   0.0f,    0.0f, 0.0f,  // left arm (7)
        0.2f,    -0.2f, 0.0f,   0.6f,   0.0f,    0.0f, 0.0f   // right arm (7)
    };
    std::memcpy(default_pose_, dp, sizeof(dp));

    action_scale_ = config.numeric_params.count("action_scale")
                        ? static_cast<float>(config.numeric_params.at("action_scale"))
                        : 0.5f;
    gait_freq_ =
        config.numeric_params.count("gait_freq") ? static_cast<float>(config.numeric_params.at("gait_freq")) : 1.5f;
    target_height_ = config.numeric_params.count("target_height") ? config.numeric_params.at("target_height") : 0.75;

    pos_weight_ = config.numeric_params.count("pos_weight") ? config.numeric_params.at("pos_weight") : 1.0;
    ori_weight_ = config.numeric_params.count("ori_weight") ? config.numeric_params.at("ori_weight") : 1.0;
    upright_weight_ = config.numeric_params.count("upright_weight") ? config.numeric_params.at("upright_weight") : 2.0;
    height_weight_ = config.numeric_params.count("height_weight") ? config.numeric_params.at("height_weight") : 0.5;
    ctrl_weight_ = config.numeric_params.count("ctrl_weight") ? config.numeric_params.at("ctrl_weight") : 0.01;

    // Store joint limits (joints 1-29, skipping floating base joint 0)
    for (int i = 0; i < 29; ++i) {
        jnt_range_low_[i] = model->jnt_range[(i + 1) * 2];
        jnt_range_high_[i] = model->jnt_range[(i + 1) * 2 + 1];
    }
}

void G1Navigation::GetObservation(const mjModel* model, const mjData* data, float* obs_out) const {
    int idx = 0;

    // 1. Local linear velocity of pelvis (3)
    if (pelvis_linvel_adr_ >= 0) {
        for (int i = 0; i < 3; ++i)
            obs_out[idx++] = static_cast<float>(data->sensordata[pelvis_linvel_adr_ + i]);
    } else {
        for (int i = 0; i < 3; ++i)
            obs_out[idx++] = 0.0f;
    }

    // 2. Gyro (angular velocity) of pelvis (3)
    if (pelvis_gyro_adr_ >= 0) {
        for (int i = 0; i < 3; ++i)
            obs_out[idx++] = static_cast<float>(data->sensordata[pelvis_gyro_adr_ + i]);
    } else {
        for (int i = 0; i < 3; ++i)
            obs_out[idx++] = 0.0f;
    }

    // 3. Gravity vector in pelvis frame (3)
    // gravity = site_xmat[pelvis_imu].T @ [0, 0, -1]
    if (pelvis_imu_site_id_ >= 0) {
        const mjtNum* xmat = data->site_xmat + 9 * pelvis_imu_site_id_;
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

    // 5. Joint angles - default_pose (29)
    // Robot joints are qpos[7:36]
    for (int i = 0; i < 29; ++i)
        obs_out[idx++] = static_cast<float>(data->qpos[7 + i]) - default_pose_[i];

    // 6. Joint velocities (29)
    // Robot joint velocities are qvel[6:35]
    for (int i = 0; i < 29; ++i)
        obs_out[idx++] = static_cast<float>(data->qvel[6 + i]);

    // 7. Last action (29)
    // last_act = (ctrl - default_pose) / action_scale
    for (int i = 0; i < 29; ++i) {
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

    // Total: 3+3+3+3+29+29+29+4 = 103
}

double G1Navigation::RunningCost(const mjModel* model, const mjData* data, const float* control) const {
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
    // Use pelvis orientation to check uprightness
    // gravity_z should be close to -1 if upright
    double upright_cost = 0.0;
    if (pelvis_imu_site_id_ >= 0) {
        const mjtNum* xmat = data->site_xmat + 9 * pelvis_imu_site_id_;
        // gravity = xmat.T @ [0,0,-1]
        double gx = -xmat[6];
        double gy = -xmat[7];
        // x^2 + y^2 of gravity vector (0 when upright)
        upright_cost = gx * gx + gy * gy;
    }

    // Height cost: keep torso at target height
    double height_cost = 0.0;
    if (torso_site_id_ >= 0) {
        double torso_z = data->site_xpos[3 * torso_site_id_ + 2];
        double height_err = torso_z - target_height_;
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

double G1Navigation::TerminalCost(const mjModel* model, const mjData* data) const {
    float zero_ctrl[3] = {0.0f, 0.0f, 0.0f};
    return RunningCost(model, data, zero_ctrl);
}

void G1Navigation::ApplyControl(const mjModel* model, mjData* data, const float* action) const {
    // action[0:3] = velocity command (vx, vy, vtheta)

    // Build the 103-dim observation with the velocity command inserted
    float obs[128];  // 103 needed, use 128 for alignment
    GetObservation(model, data, obs);
    // Overwrite the command slots (indices 9, 10, 11)
    obs[9] = action[0];
    obs[10] = action[1];
    obs[11] = action[2];

    // Run ONNX policy to get 29 motor actions
    float policy_action[32];  // 29 needed
    if (policy_) {
        policy_->ComputeAction(obs, 103, policy_action, 29);
    } else {
        for (int i = 0; i < 29; ++i)
            policy_action[i] = 0.0f;
    }

    // Convert policy output to motor targets: default_pose + action * scale
    for (int i = 0; i < 29; ++i) {
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
REGISTER_TASK("G1Navigation", spc::tasks::G1Navigation, G1Navigation)
