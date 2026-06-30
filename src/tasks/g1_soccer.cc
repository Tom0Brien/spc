#include "spc/tasks/g1_soccer.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace spc {
namespace tasks {

G1Soccer::G1Soccer(mjModel* model, const spc::core::TaskConfig& config)
    : G1Navigation(model, config) {  // Call base class constructor

    std::string ball_name =
        config.string_params.count("ball_name") ? config.string_params.at("ball_name") : "soccer_ball";
    soccer_ball_id_ = mj_name2id(model, mjOBJ_BODY, ball_name.c_str());
    if (soccer_ball_id_ < 0) {
        throw std::runtime_error("Could not find soccer ball body: " + ball_name);
    }

    // Default weights for soccer
    standoff_distance_ =
        config.numeric_params.count("standoff_distance") ? config.numeric_params.at("standoff_distance") : 0.5;
    ball_goal_weight_ =
        config.numeric_params.count("ball_goal_weight") ? config.numeric_params.at("ball_goal_weight") : 1.0;

    // Override navigation weights with soccer-specific ones if not provided
    pos_weight_ = config.numeric_params.count("pos_weight") ? config.numeric_params.at("pos_weight") : 0.3;
    ori_weight_ = config.numeric_params.count("ori_weight") ? config.numeric_params.at("ori_weight") : 0.2;
    height_weight_ = config.numeric_params.count("height_weight") ? config.numeric_params.at("height_weight") : 0.5;
    ctrl_weight_ = config.numeric_params.count("ctrl_weight") ? config.numeric_params.at("ctrl_weight") : 0.01;
}

double G1Soccer::RunningCost(const mjModel* model, const mjData* data, const float* control) const {
    // Ball position (x, y)
    const mjtNum* ball_pos = data->xpos + 3 * soccer_ball_id_;
    // Goal position from mocap 0 (since it inherits, mocap_target could be used, but let's assume mocap 0)
    const mjtNum* goal_pos = data->mocap_pos;

    // Ball to goal cost
    double b2g_x = goal_pos[0] - ball_pos[0];
    double b2g_y = goal_pos[1] - ball_pos[1];
    double ball_goal_dist = std::sqrt(b2g_x * b2g_x + b2g_y * b2g_y);
    double ball_goal_cost = ball_goal_dist * ball_goal_dist;

    // Desired robot position (behind ball)
    double dir_x = b2g_x / (ball_goal_dist + 1e-6);
    double dir_y = b2g_y / (ball_goal_dist + 1e-6);
    double desired_x = ball_pos[0] - standoff_distance_ * dir_x;
    double desired_y = ball_pos[1] - standoff_distance_ * dir_y;

    // Robot position
    double rx = data->qpos[0];
    double ry = data->qpos[1];

    // Robot position cost
    double px = rx - desired_x;
    double py = ry - desired_y;
    double robot_pos_cost = px * px + py * py;
    double dist_to_desired = std::sqrt(robot_pos_cost);

    // Robot orientation cost
    // Robot yaw: from quaternion
    const mjtNum* quat = data->qpos + 3;
    double yaw =
        std::atan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]));

    double approach_angle = std::atan2(desired_y - ry, desired_x - rx);
    double kick_angle = std::atan2(goal_pos[1] - ry, goal_pos[0] - rx);

    // Sigmoid transition: far -> approach, close -> kick
    double transition = 1.0 / (1.0 + std::exp(-((dist_to_desired - 0.6) / 0.2)));
    double desired_yaw = transition * approach_angle + (1.0 - transition) * kick_angle;

    double ori_err = yaw - desired_yaw;
    ori_err = std::atan2(std::sin(ori_err), std::cos(ori_err));
    double robot_ori_cost = ori_err * ori_err;

    // Height cost
    double height_cost = 0.0;
    if (torso_site_id_ >= 0) {
        double tz = data->site_xpos[3 * torso_site_id_ + 2];
        double err = tz - target_height_;
        height_cost = err * err;
    }

    // Control cost
    double ctrl_cost = 0.0;
    for (int i = 0; i < 3; ++i)
        ctrl_cost += control[i] * control[i];

    return ball_goal_weight_ * ball_goal_cost + pos_weight_ * robot_pos_cost + ori_weight_ * robot_ori_cost +
           height_weight_ * height_cost + ctrl_weight_ * ctrl_cost;
}

double G1Soccer::TerminalCost(const mjModel* model, const mjData* data) const {
    float zero_ctrl[3] = {0.0f, 0.0f, 0.0f};
    return RunningCost(model, data, zero_ctrl);
}

}  // namespace tasks
}  // namespace spc

#include "spc/core/task_factory.h"
REGISTER_TASK("G1Soccer", spc::tasks::G1Soccer, G1SoccerTask)
