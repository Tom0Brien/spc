#include "spc/tasks/humanoid_pass.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace spc {
namespace tasks {

HumanoidPass::HumanoidPass(mjModel* model, const core::TaskConfig& config, HumanoidSpec spec)
    : HumanoidNavigation(model, config, std::move(spec)) {
    auto num = [&config](const char* key, double fallback) {
        auto it = config.numeric_params.find(key);
        return it != config.numeric_params.end() ? it->second : fallback;
    };

    std::string ball_name =
        config.string_params.count("ball_name") ? config.string_params.at("ball_name") : "soccer_ball";
    soccer_ball_id_ = mj_name2id(model, mjOBJ_BODY, ball_name.c_str());
    if (soccer_ball_id_ < 0) {
        throw std::runtime_error("Could not find soccer ball body: " + ball_name);
    }
    ball_dof_adr_ =
        model->body_jntnum[soccer_ball_id_] > 0 ? model->jnt_dofadr[model->body_jntadr[soccer_ball_id_]] : -1;

    standoff_distance_ = num("standoff_distance", 0.5);
    ball_goal_weight_ = num("ball_goal_weight", 1.0);
    ball_goal_scale_ = num("ball_goal_scale", 0.5);
    // Keeps the velocity-only controller from bumping the ball off-line. The
    // augmented task overrides this to 0 (residual kicks need close,
    // momentarily misaligned contact).
    behind_weight_ = num("behind_weight", 2.0);

    // Dribbling shaping, off by default (see header).
    ball_decel_ = num("ball_decel", 0.0);
    approach_distance_ = num("approach_distance", standoff_distance_);
    avoid_weight_ = num("avoid_weight", 0.0);
    avoid_radius_ = num("avoid_radius", 0.35);
    align_lo_ = num("align_lo", 0.5);
    align_hi_ = num("align_hi", 0.9);

    // Pass-specific defaults for the navigation weights
    pos_weight_ = num("pos_weight", 0.3);
    ori_weight_ = num("ori_weight", 0.2);
    height_weight_ = num("height_weight", 0.5);
    ctrl_weight_ = num("ctrl_weight", 0.01);
}

double HumanoidPass::AlignGate(double align) const {
    double t = (align - align_lo_) / (align_hi_ - align_lo_);
    t = std::fmin(1.0, std::fmax(0.0, t));
    return t * t * (3.0 - 2.0 * t);  // smoothstep
}

double HumanoidPass::RunningCost(const mjModel* model, const mjData* data, const float* control) const {
    const mjtNum* ball_pos = data->xpos + 3 * soccer_ball_id_;
    const mjtNum* goal_pos = data->mocap_pos;  // goal from mocap body 0

    // Ball to goal cost: pseudo-Huber so a distant goal gives a constant
    // gradient instead of quadratically dominating the stability terms.
    // With ball_decel set, the ball is scored where it will come to rest
    // (s = v|v| / 2a under constant rolling deceleration).
    double rest_x = ball_pos[0];
    double rest_y = ball_pos[1];
    if (ball_decel_ > 0.0 && ball_dof_adr_ >= 0) {
        double vx = data->qvel[ball_dof_adr_];
        double vy = data->qvel[ball_dof_adr_ + 1];
        double roll = std::sqrt(vx * vx + vy * vy) / (2.0 * ball_decel_);
        rest_x += vx * roll;
        rest_y += vy * roll;
    }
    double rest_dx = goal_pos[0] - rest_x;
    double rest_dy = goal_pos[1] - rest_y;
    double ball_goal_cost = PseudoHuber(std::sqrt(rest_dx * rest_dx + rest_dy * rest_dy), ball_goal_scale_);

    // Ball->goal direction from the current ball position
    double b2g_x = goal_pos[0] - ball_pos[0];
    double b2g_y = goal_pos[1] - ball_pos[1];
    double ball_goal_dist = std::sqrt(b2g_x * b2g_x + b2g_y * b2g_y);
    double dir_x = b2g_x / (ball_goal_dist + 1e-6);
    double dir_y = b2g_y / (ball_goal_dist + 1e-6);

    double rx = data->qpos[0];
    double ry = data->qpos[1];

    // Robot alignment behind the ball: 1 = robot->ball points along ball->goal
    double rb_x = ball_pos[0] - rx;
    double rb_y = ball_pos[1] - ry;
    double rb_dist = std::sqrt(rb_x * rb_x + rb_y * rb_y) + 1e-6;
    double align = (rb_x * dir_x + rb_y * dir_y) / rb_dist;
    double gate = AlignGate(align);

    // Desired robot position: behind the ball along the ball->goal line, at
    // approach_distance while misaligned, closing to standoff_distance (the
    // dribbling distance) once lined up.
    double standoff = approach_distance_ + gate * (standoff_distance_ - approach_distance_);
    double desired_x = ball_pos[0] - standoff * dir_x;
    double desired_y = ball_pos[1] - standoff * dir_y;

    double px = rx - desired_x;
    double py = ry - desired_y;
    double dist_to_desired = std::sqrt(px * px + py * py);
    double robot_pos_cost = PseudoHuber(dist_to_desired, pos_scale_);

    // Robot yaw from the base quaternion
    double yaw = YawFromQuat(data->qpos + 3);

    double approach_angle = std::atan2(desired_y - ry, desired_x - rx);
    double kick_angle = std::atan2(goal_pos[1] - ry, goal_pos[0] - rx);

    // Sigmoid transition: far -> face the approach point, close -> face the goal
    double transition = 1.0 / (1.0 + std::exp(-((dist_to_desired - 0.6) / 0.2)));
    double desired_yaw = transition * approach_angle + (1.0 - transition) * kick_angle;

    double ori_err = WrapAngle(yaw - desired_yaw);
    double robot_ori_cost = PseudoHuber(ori_err, ori_scale_);

    // Height cost
    double height_cost = 0.0;
    if (height_site_id_ >= 0) {
        double z = data->site_xpos[3 * height_site_id_ + 2];
        double err = z - target_height_;
        height_cost = err * err;
    }

    // Upright cost (same as HumanoidNavigation)
    double upright_cost = 0.0;
    if (upright_site_id_ >= 0) {
        const mjtNum* xmat = data->site_xmat + 9 * upright_site_id_;
        double gx = -xmat[6];
        double gy = -xmat[7];
        upright_cost = gx * gx + gy * gy;
    }

    // Behind-ball alignment: penalize being on the goal side of the ball,
    // gated by proximity so it only acts near the ball
    double proximity = std::exp(-rb_dist / 0.5);
    double behind_cost = (1.0 - align) * proximity;

    // Misaligned ball avoidance: the ball is an obstacle until the robot is
    // lined up behind it, so circling around beats pushing it off-line.
    double avoid_r = rb_dist / avoid_radius_;
    double avoid_cost = (1.0 - gate) * std::exp(-avoid_r * avoid_r);

    double ctrl_cost = 0.0;
    for (int i = 0; i < 3; ++i)
        ctrl_cost += control[i] * control[i];

    return ball_goal_weight_ * ball_goal_cost + pos_weight_ * robot_pos_cost + ori_weight_ * robot_ori_cost +
           height_weight_ * height_cost + upright_weight_ * upright_cost + behind_weight_ * behind_cost +
           avoid_weight_ * avoid_cost + ctrl_weight_ * ctrl_cost;
}

double HumanoidPass::TerminalCost(const mjModel* model, const mjData* data) const {
    float zero_ctrl[3] = {0.0f, 0.0f, 0.0f};
    return RunningCost(model, data, zero_ctrl);
}

}  // namespace tasks
}  // namespace spc
