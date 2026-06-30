#include "spc/tasks/franka_push.h"
#include <cmath>
#include <iostream>
#include <vector>

namespace spc {
namespace tasks {

FrankaPush::FrankaPush(mjModel* model, const spc::core::TaskConfig& config) {
    std::string obj_name = config.string_params.count("obj_name") ? config.string_params.at("obj_name") : "box";
    std::string gripper_name = config.string_params.count("gripper_name") ? config.string_params.at("gripper_name") : "gripper";
    
    obj_body_ = mj_name2id(model, mjOBJ_BODY, obj_name.c_str());
    gripper_site_ = mj_name2id(model, mjOBJ_SITE, gripper_name.c_str());
    mocap_target_ = config.numeric_params.count("mocap_target") ? static_cast<int>(config.numeric_params.at("mocap_target")) : 0;
    action_scale_ = config.numeric_params.count("action_scale") ? static_cast<float>(config.numeric_params.at("action_scale")) : 0.1f;
    
    obj_target_weight_ = config.numeric_params.count("obj_target_weight") ? config.numeric_params.at("obj_target_weight") : 10.0;
    gripper_obj_weight_ = config.numeric_params.count("gripper_obj_weight") ? config.numeric_params.at("gripper_obj_weight") : 5.0;
    orientation_weight_ = config.numeric_params.count("orientation_weight") ? config.numeric_params.at("orientation_weight") : 1.0;
    residual_weight_ = config.numeric_params.count("residual_weight") ? config.numeric_params.at("residual_weight") : 0.1;
}

void FrankaPush::GetObservation(const mjModel* model, const mjData* data, float* obs_out) const {
    int idx = 0;

    // 1. target_pos (3)
    for (int i = 0; i < 3; ++i) obs_out[idx++] = data->mocap_pos[3 * mocap_target_ + i];

    // 2. target_orientation (6) - from target_mat (3x3), removing first column/row?
    // In Python: target_mat.ravel()[3:] -> elements 3 to 8 of the 3x3 matrix.
    mjtNum target_mat[9];
    mju_quat2Mat(target_mat, data->mocap_quat + 4 * mocap_target_);
    for (int i = 3; i < 9; ++i) obs_out[idx++] = target_mat[i];

    // 3. last_action (7)
    for (int i = 0; i < 7; ++i) obs_out[idx++] = data->ctrl[i] / action_scale_;

    // 4. robot_qpos (7)
    for (int i = 0; i < 7; ++i) obs_out[idx++] = data->qpos[i];

    // 5. robot_qvel (7)
    for (int i = 0; i < 7; ++i) obs_out[idx++] = data->qvel[i];

    // 6. gripper_pos (3)
    for (int i = 0; i < 3; ++i) obs_out[idx++] = data->site_xpos[3 * gripper_site_ + i];

    // 7. gripper_mat.ravel()[3:] (6)
    for (int i = 3; i < 9; ++i) obs_out[idx++] = data->site_xmat[9 * gripper_site_ + i];

    // 8. obj_orientation (6)
    mjtNum obj_mat[9];
    mju_quat2Mat(obj_mat, data->xquat + 4 * obj_body_);
    for (int i = 3; i < 9; ++i) obs_out[idx++] = obj_mat[i];

    // 9. obj_pos (3)
    for (int i = 0; i < 3; ++i) obs_out[idx++] = data->xpos[3 * obj_body_ + i];
    
    // Total should be 3+6+7+7+7+3+6+6+3 = 48
}

double FrankaPush::RunningCost(const mjModel* model, const mjData* data, const float* control) const {
    const mjtNum* target_pos = data->mocap_pos + 3 * mocap_target_;
    const mjtNum* obj_pos = data->xpos + 3 * obj_body_;
    const mjtNum* gripper_pos = data->site_xpos + 3 * gripper_site_;

    // Box to target cost
    double obj_target_cost = 0;
    for (int i = 0; i < 2; ++i) {
        double d = obj_pos[i] - target_pos[i];
        obj_target_cost += d * d;
    }

    // Gripper to object cost
    double side_dir[3];
    double side_dir_norm = 0;
    for (int i = 0; i < 3; ++i) {
        side_dir[i] = obj_pos[i] - target_pos[i];
        side_dir_norm += side_dir[i] * side_dir[i];
    }
    side_dir_norm = std::sqrt(side_dir_norm) + 1e-6;
    
    double gripper_obj_cost = 0;
    for (int i = 0; i < 3; ++i) {
        double adjusted_dir = (side_dir_norm > 1e-3) ? (side_dir[i] / side_dir_norm * 0.1) : 0.0;
        double obj_side_pos = adjusted_dir + obj_pos[i];
        double d = obj_side_pos - gripper_pos[i];
        gripper_obj_cost += d * d;
    }

    // Orientation cost
    const mjtNum* target_quat = data->mocap_quat + 4 * mocap_target_;
    const mjtNum* obj_quat = data->xquat + 4 * obj_body_;
    
    mjtNum target_inv[4];
    mju_negQuat(target_inv, target_quat);
    mjtNum quat_diff[4];
    mju_mulQuat(quat_diff, obj_quat, target_inv);
    
    double norm_img = std::sqrt(quat_diff[1]*quat_diff[1] + quat_diff[2]*quat_diff[2] + quat_diff[3]*quat_diff[3]);
    if (norm_img > 1.0) norm_img = 1.0;
    double ori_error = 2.0 * std::asin(norm_img);
    double orientation_cost = ori_error * ori_error;

    // Residual penalty (simplified for C++ as quadratic penalty for now)
    double residual_cost = 0;
    for (int i = 0; i < 7; ++i) {
        residual_cost += control[i] * control[i];
    }
    residual_cost *= residual_weight_;

    return obj_target_weight_ * obj_target_cost + 
           gripper_obj_weight_ * gripper_obj_cost + 
           orientation_weight_ * orientation_cost + 
           residual_cost;
}

double FrankaPush::TerminalCost(const mjModel* model, const mjData* data) const {
    std::vector<float> zero_ctrl(7, 0.0f);
    return RunningCost(model, data, zero_ctrl.data());
}

void FrankaPush::ApplyControl(const mjModel* model, mjData* data, const float* residual) const {
    float max_torque[7] = {87.0f, 87.0f, 87.0f, 87.0f, 12.0f, 12.0f, 12.0f};
    float gear[7] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}; // Assuming gear=1, check limits
    
    float base_action[128];
    for (int j = 0; j < 7; ++j) base_action[j] = 0.0f;
    
    if (policy_) {
        float obs[512];
        GetObservation(model, data, obs);
        policy_->ComputeAction(obs, GetObsDim(), base_action, 7);
    }
    
    for (int i = 0; i < 7; ++i) {
        float ctrl = (base_action[i] + residual[i]) * action_scale_;
        
        // Clamp torque
        if (ctrl > max_torque[i]) ctrl = max_torque[i];
        if (ctrl < -max_torque[i]) ctrl = -max_torque[i];
        
        // The MuJoCo model bounds might be different but we clamp here manually
        data->ctrl[i] = ctrl;
    }
    
    // Gripper (index 7)
    if (model->nu > 7) {
        data->ctrl[7] = 0.82f;
    }
}

} // namespace tasks
} // namespace spc

#include "spc/core/task_factory.h"
REGISTER_TASK("FrankaPushTask", spc::tasks::FrankaPush, FrankaPushTask)
