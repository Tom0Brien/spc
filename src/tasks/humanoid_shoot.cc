#include "spc/tasks/humanoid_shoot.h"

#include <cmath>
#include <utility>

namespace spc {
namespace tasks {

HumanoidShoot::HumanoidShoot(mjModel* model, const core::TaskConfig& config, HumanoidSpec spec)
    : HumanoidPass(model, config, std::move(spec)) {
    auto num = [&config](const char* key, double fallback) {
        auto it = config.numeric_params.find(key);
        return it != config.numeric_params.end() ? it->second : fallback;
    };

    // Ball freejoint qvel address (for the shot-power reward)
    int ball_jnt = model->body_jntadr[soccer_ball_id_];
    ball_dofadr_ = (ball_jnt >= 0) ? model->jnt_dofadr[ball_jnt] : -1;

    goal_half_width_ = num("goal_half_width", 0.5);
    lateral_weight_ = num("lateral_weight", 1.0);
    shoot_power_weight_ = num("shoot_power_weight", 0.5);
}

double HumanoidShoot::RunningCost(const mjModel* model, const mjData* data, const float* control) const {
    const mjtNum* ball_pos = data->xpos + 3 * soccer_ball_id_;
    const mjtNum* goal_pos = data->mocap_pos;  // goal region center from mocap body 0

    // Shooting direction n: the mocap marker's yaw (+x axis). The goal line
    // runs perpendicular to n through goal_pos.
    double goal_yaw = YawFromQuat(data->mocap_quat);
    double nx = std::cos(goal_yaw);
    double ny = std::sin(goal_yaw);

    // Ball position relative to the goal line: s along n (negative in front of
    // the line), t lateral.
    double rel_x = ball_pos[0] - goal_pos[0];
    double rel_y = ball_pos[1] - goal_pos[1];
    double s = rel_x * nx + rel_y * ny;
    double t = -rel_x * ny + rel_y * nx;

    // Shortfall: distance remaining to the goal line; zero once crossed, so
    // overshoot is free (this is what distinguishes shoot from pass).
    double shortfall = std::max(0.0, -s);
    double shortfall_cost = PseudoHuber(shortfall, ball_goal_scale_);

    // Lateral miss: free inside the goal mouth, penalized beyond it.
    double lateral_miss = std::max(0.0, std::abs(t) - goal_half_width_);
    double lateral_cost = PseudoHuber(lateral_miss, ball_goal_scale_);

    // Shot power: reward ball velocity along n, NOT gated on being in front of
    // the line. Gated, the reward would integrate over the rollout to the
    // distance covered before crossing -- the same constant for every scoring
    // rollout regardless of speed. Ungated, it integrates to the total ball
    // travel along n within the horizon, which scales with shot speed: a hard
    // shot keeps earning past the line, a trickled one stops.
    double power_cost = 0.0;
    if (ball_dofadr_ >= 0) {
        const mjtNum* ball_vel = data->qvel + ball_dofadr_;  // freejoint: linear vel first
        power_cost = -std::max(0.0, ball_vel[0] * nx + ball_vel[1] * ny);
    }

    // Desired robot position: behind the ball along the shooting direction, so
    // contact drives the ball along n. Unlike pass, n is constant, so this
    // stays well-defined even at/past the goal line.
    double desired_x = ball_pos[0] - standoff_distance_ * nx;
    double desired_y = ball_pos[1] - standoff_distance_ * ny;

    double rx = data->qpos[0];
    double ry = data->qpos[1];

    double px = rx - desired_x;
    double py = ry - desired_y;
    double dist_to_desired = std::sqrt(px * px + py * py);
    double robot_pos_cost = PseudoHuber(dist_to_desired, pos_scale_);

    // Orientation: far -> face the approach point, close -> face along n.
    double yaw = YawFromQuat(data->qpos + 3);
    double approach_angle = std::atan2(desired_y - ry, desired_x - rx);
    double transition = 1.0 / (1.0 + std::exp(-((dist_to_desired - 0.6) / 0.2)));
    double desired_yaw = transition * approach_angle + (1.0 - transition) * goal_yaw;

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
    double rb_x = ball_pos[0] - rx;
    double rb_y = ball_pos[1] - ry;
    double rb_dist = std::sqrt(rb_x * rb_x + rb_y * rb_y) + 1e-6;
    double align = (rb_x * nx + rb_y * ny) / rb_dist;  // 1 = robot directly behind ball
    double proximity = std::exp(-rb_dist / 0.5);
    double behind_cost = (1.0 - align) * proximity;

    double ctrl_cost = 0.0;
    for (int i = 0; i < 3; ++i)
        ctrl_cost += control[i] * control[i];

    return ball_goal_weight_ * shortfall_cost + lateral_weight_ * lateral_cost +
           shoot_power_weight_ * power_cost + pos_weight_ * robot_pos_cost + ori_weight_ * robot_ori_cost +
           height_weight_ * height_cost + upright_weight_ * upright_cost + behind_weight_ * behind_cost +
           ctrl_weight_ * ctrl_cost;
}

double HumanoidShoot::TerminalCost(const mjModel* model, const mjData* data) const {
    float zero_ctrl[3] = {0.0f, 0.0f, 0.0f};
    return RunningCost(model, data, zero_ctrl);
}

}  // namespace tasks
}  // namespace spc
