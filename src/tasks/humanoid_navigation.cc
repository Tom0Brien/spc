#include "spc/tasks/humanoid_navigation.h"

#include <cmath>
#include <utility>

namespace spc {
namespace tasks {

HumanoidNavigation::HumanoidNavigation(mjModel* model, const core::TaskConfig& config, HumanoidSpec spec)
    : spec_(std::move(spec)) {
    auto num = [&config](const char* key, double fallback) {
        auto it = config.numeric_params.find(key);
        return it != config.numeric_params.end() ? it->second : fallback;
    };
    auto str = [&config](const char* key, const std::string& fallback) {
        auto it = config.string_params.find(key);
        return it != config.string_params.end() ? it->second : fallback;
    };

    std::string gyro_name = str("gyro_name", spec_.gyro_name);
    std::string linvel_name = str("linvel_name", spec_.linvel_name);
    std::string upright_site = str("upright_site", spec_.upright_site);
    std::string height_site = str("height_site", spec_.height_site);

    int gyro_id = mj_name2id(model, mjOBJ_SENSOR, gyro_name.c_str());
    int linvel_id = mj_name2id(model, mjOBJ_SENSOR, linvel_name.c_str());
    gyro_adr_ = (gyro_id >= 0) ? model->sensor_adr[gyro_id] : -1;
    linvel_adr_ = (linvel_id >= 0) ? model->sensor_adr[linvel_id] : -1;

    upright_site_id_ = mj_name2id(model, mjOBJ_SITE, upright_site.c_str());
    height_site_id_ = mj_name2id(model, mjOBJ_SITE, height_site.c_str());

    action_scale_ = static_cast<float>(num("action_scale", spec_.action_scale));
    action_scale_joint_.assign(spec_.njoints, action_scale_);
    if (!spec_.action_scale_vec.empty()) {
        for (int i = 0; i < spec_.njoints; ++i) {
            action_scale_joint_[i] = spec_.action_scale_vec[i] * action_scale_;
        }
    }
    gait_freq_ = static_cast<float>(num("gait_freq", spec_.gait_freq));

    vel_limit_[0] = static_cast<float>(num("vx_limit", spec_.vel_limit[0]));
    vel_limit_[1] = static_cast<float>(num("vy_limit", spec_.vel_limit[1]));
    vel_limit_[2] = static_cast<float>(num("vtheta_limit", spec_.vel_limit[2]));
    cmd_deadzone_ = static_cast<float>(num("cmd_deadzone", 0.1));

    target_height_ = num("target_height", spec_.target_height);
    pos_weight_ = num("pos_weight", 1.0);
    ori_weight_ = num("ori_weight", 1.0);
    upright_weight_ = num("upright_weight", 2.0);
    height_weight_ = num("height_weight", 0.5);
    ctrl_weight_ = num("ctrl_weight", 0.01);
    pos_scale_ = num("pos_scale", 0.5);
    ori_scale_ = num("ori_scale", 0.5);

    // Joint limits (joints 1..njoints, skipping the floating base joint 0)
    jnt_range_low_.resize(spec_.njoints);
    jnt_range_high_.resize(spec_.njoints);
    for (int i = 0; i < spec_.njoints; ++i) {
        jnt_range_low_[i] = static_cast<float>(model->jnt_range[(i + 1) * 2]);
        jnt_range_high_[i] = static_cast<float>(model->jnt_range[(i + 1) * 2 + 1]);
    }
}

void HumanoidNavigation::GetObservation(const mjModel* model, const mjData* data, float* obs_out) const {
    const int n = spec_.njoints;
    int idx = 0;

    // 1. Local linear velocity of the base (3)
    if (linvel_adr_ >= 0) {
        for (int i = 0; i < 3; ++i)
            obs_out[idx++] = static_cast<float>(data->sensordata[linvel_adr_ + i]);
    } else {
        for (int i = 0; i < 3; ++i)
            obs_out[idx++] = 0.0f;
    }

    // 2. Gyro (angular velocity) of the base (3)
    if (gyro_adr_ >= 0) {
        for (int i = 0; i < 3; ++i)
            obs_out[idx++] = static_cast<float>(data->sensordata[gyro_adr_ + i]);
    } else {
        for (int i = 0; i < 3; ++i)
            obs_out[idx++] = 0.0f;
    }

    // 3. Gravity vector in the IMU frame (3): xmat.T @ [0,0,-1] with xmat 3x3
    // row-major = [-xmat[6], -xmat[7], -xmat[8]]
    if (upright_site_id_ >= 0) {
        const mjtNum* xmat = data->site_xmat + 9 * upright_site_id_;
        obs_out[idx++] = static_cast<float>(-xmat[6]);
        obs_out[idx++] = static_cast<float>(-xmat[7]);
        obs_out[idx++] = static_cast<float>(-xmat[8]);
    } else {
        obs_out[idx++] = 0.0f;
        obs_out[idx++] = 0.0f;
        obs_out[idx++] = -1.0f;
    }

    // 4. Command (3): zeros here; ApplyControl overwrites these slots with the
    // clamped velocity command before running the policy.
    for (int i = 0; i < 3; ++i)
        obs_out[idx++] = 0.0f;

    // 5. Joint angles - default_pose (njoints). Robot joints are qpos[7:].
    // The leading zero_obs_joints entries are zeroed (T1 head joints, per its
    // sim2sim deployment).
    for (int i = 0; i < n; ++i) {
        float dq = static_cast<float>(data->qpos[7 + i]) - spec_.default_pose[i];
        obs_out[idx++] = (i < spec_.zero_obs_joints) ? 0.0f : dq;
    }

    // 6. Joint velocities (njoints). Robot joint velocities are qvel[6:].
    for (int i = 0; i < n; ++i) {
        float v = static_cast<float>(data->qvel[6 + i]);
        obs_out[idx++] = (i < spec_.zero_obs_joints) ? 0.0f : v;
    }

    // 7. Last action (njoints): (ctrl - default_pose) / action_scale
    for (int i = 0; i < n; ++i) {
        float motor_target = static_cast<float>(data->ctrl[i]);
        obs_out[idx++] = (motor_target - spec_.default_pose[i]) / action_scale_joint_[i];
    }

    // 8. Gait phase (4): cos(phase_left), cos(phase_right), sin(phase_left), sin(phase_right)
    float phase_base = 2.0f * static_cast<float>(M_PI) * static_cast<float>(data->time) * gait_freq_;
    float phase_left = phase_base;
    float phase_right = phase_base + static_cast<float>(M_PI);
    obs_out[idx++] = std::cos(phase_left);
    obs_out[idx++] = std::cos(phase_right);
    obs_out[idx++] = std::sin(phase_left);
    obs_out[idx++] = std::sin(phase_right);

    // Total: 3+3+3+3 + 3*njoints + 4 = 16 + 3*njoints
}

double HumanoidNavigation::RunningCost(const mjModel* model, const mjData* data, const float* control) const {
    // Goal from mocap body 0
    const mjtNum* goal_pos = data->mocap_pos;
    const mjtNum* goal_quat = data->mocap_quat;

    // Position error (2D, floating base): pseudo-Huber so a distant goal gives
    // a constant gradient instead of quadratically dominating the stability terms
    double dx = data->qpos[0] - goal_pos[0];
    double dy = data->qpos[1] - goal_pos[1];
    double position_cost = PseudoHuber(std::sqrt(dx * dx + dy * dy), pos_scale_);

    // Yaw error, wrapped to [-pi, pi]
    double yaw_err = WrapAngle(YawFromQuat(data->qpos + 3) - YawFromQuat(goal_quat));
    double orientation_cost = PseudoHuber(yaw_err, ori_scale_);

    // Upright cost: xy components of gravity in the IMU frame (0 when upright)
    double upright_cost = 0.0;
    if (upright_site_id_ >= 0) {
        const mjtNum* xmat = data->site_xmat + 9 * upright_site_id_;
        double gx = -xmat[6];
        double gy = -xmat[7];
        upright_cost = gx * gx + gy * gy;
    }

    // Height cost: keep the height site at target_height
    double height_cost = 0.0;
    if (height_site_id_ >= 0) {
        double z = data->site_xpos[3 * height_site_id_ + 2];
        double err = z - target_height_;
        height_cost = err * err;
    }

    // Control regularization (velocity commands)
    double ctrl_cost = 0.0;
    for (int i = 0; i < 3; ++i) {
        ctrl_cost += control[i] * control[i];
    }

    return pos_weight_ * position_cost + ori_weight_ * orientation_cost + upright_weight_ * upright_cost +
           height_weight_ * height_cost + ctrl_weight_ * ctrl_cost;
}

double HumanoidNavigation::TerminalCost(const mjModel* model, const mjData* data) const {
    float zero_ctrl[3] = {0.0f, 0.0f, 0.0f};
    return RunningCost(model, data, zero_ctrl);
}

void HumanoidNavigation::ApplyControl(const mjModel* model, mjData* data, const float* action) const {
    const int n = spec_.njoints;

    // Build the observation with the velocity command inserted at slots 9..11,
    // clamped to the command limits.
    float obs[128];  // max obs dim is 16 + 3*29 = 103
    GetObservation(model, data, obs);
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

    // Commands below the policy's trained stand threshold (playground
    // stand_still cost gates on |cmd| < 0.1) snap to exactly zero: the policy
    // only stands cleanly at zero, and a sampling optimizer never lands there
    // on its own, leaving the robot shuffling at the stand/walk boundary.
    if (cmd_norm_sq < cmd_deadzone_ * cmd_deadzone_) {
        obs[9] = obs[10] = obs[11] = 0.0f;
        cmd_norm_sq = 0.0f;
    }

    // Some policies (T1) were trained with the gait phase pinned to pi when
    // the command is near zero (standing).
    if (spec_.pin_phase_when_standing && cmd_norm_sq < 0.01f * 0.01f) {
        const int phase = GetObsDim() - 4;
        obs[phase + 0] = -1.0f;  // cos(pi)
        obs[phase + 1] = -1.0f;
        obs[phase + 2] = 0.0f;  // sin(pi)
        obs[phase + 3] = 0.0f;
    }

    // Run the policy to get motor actions
    float policy_action[32];  // max njoints is 29
    if (policy_) {
        policy_->ComputeAction(obs, GetObsDim(), policy_action, n);
    } else {
        for (int i = 0; i < n; ++i)
            policy_action[i] = 0.0f;
    }

    // Motor targets: default_pose + action * scale, clamped to joint limits
    for (int i = 0; i < n; ++i) {
        float motor_target = spec_.default_pose[i] + policy_action[i] * action_scale_joint_[i];

        if (motor_target < jnt_range_low_[i])
            motor_target = jnt_range_low_[i];
        if (motor_target > jnt_range_high_[i])
            motor_target = jnt_range_high_[i];

        data->ctrl[i] = motor_target;
    }
}

}  // namespace tasks
}  // namespace spc
