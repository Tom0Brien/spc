#include "spc/tasks/humanoid_soccer_augmented.h"

#include <cmath>
#include <utility>

namespace spc {
namespace tasks {

HumanoidSoccerAugmented::HumanoidSoccerAugmented(mjModel* model, const core::TaskConfig& config, HumanoidSpec spec)
    : HumanoidSoccer(model, config, std::move(spec)) {
    auto num = [&config](const char* key, double fallback) {
        auto it = config.numeric_params.find(key);
        return it != config.numeric_params.end() ? it->second : fallback;
    };

    leg_joint_start_ = static_cast<int>(num("leg_joint_start", spec_.leg_joint_start));
    leg_joint_count_ = static_cast<int>(num("leg_joint_count", 12));
    residual_weight_ = num("residual_weight", 0.05);

    // Distance-to-ball residual gate; disabled unless configured.
    gate_near_ = num("gate_near", 0.0);
    gate_far_ = num("gate_far", 0.0);

    // Residual kicks need close contact that the behind-ball penalty fights
    // against, so it is disabled by default for the augmented task.
    behind_weight_ = num("behind_weight", 0.0);
}

double HumanoidSoccerAugmented::ResidualGate(const mjData* data) const {
    if (gate_far_ <= gate_near_)
        return 1.0;
    const mjtNum* ball_pos = data->xpos + 3 * soccer_ball_id_;
    double dx = ball_pos[0] - data->qpos[0];
    double dy = ball_pos[1] - data->qpos[1];
    double rb = std::sqrt(dx * dx + dy * dy);
    if (rb <= gate_near_)
        return 1.0;
    if (rb >= gate_far_)
        return 0.0;
    double t = (gate_far_ - rb) / (gate_far_ - gate_near_);
    return t * t * (3.0 - 2.0 * t);  // smoothstep
}

void HumanoidSoccerAugmented::ApplyControl(const mjModel* model, mjData* data, const float* control) const {
    // control[0:3] = velocity command, control[3:] = leg residuals.

    // Run the RL policy on the velocity command to write base motor targets.
    HumanoidNavigation::ApplyControl(model, data, control);

    float gate = static_cast<float>(ResidualGate(data));
    if (gate <= 0.0f)
        return;

    // Add gated residuals to the leg motor targets and re-clamp.
    const float* residuals = control + 3;
    for (int i = 0; i < leg_joint_count_; ++i) {
        int j = leg_joint_start_ + i;
        float motor_target = static_cast<float>(data->ctrl[j]) + gate * residuals[i];

        if (motor_target < jnt_range_low_[j])
            motor_target = jnt_range_low_[j];
        if (motor_target > jnt_range_high_[j])
            motor_target = jnt_range_high_[j];

        data->ctrl[j] = motor_target;
    }
}

double HumanoidSoccerAugmented::RunningCost(const mjModel* model, const mjData* data, const float* control) const {
    // Base soccer cost; the base uses control[0:3] for the velocity control cost.
    double cost = HumanoidSoccer::RunningCost(model, data, control);

    // Regularize the leg residuals control[3:3+leg_joint_count_].
    double residual_cost = 0.0;
    const float* residuals = control + 3;
    for (int i = 0; i < leg_joint_count_; ++i)
        residual_cost += residuals[i] * residuals[i];

    return cost + residual_weight_ * residual_cost;
}

double HumanoidSoccerAugmented::TerminalCost(const mjModel* model, const mjData* data) const {
    float zero_ctrl[3 + 32] = {0.0f};
    return RunningCost(model, data, zero_ctrl);
}

}  // namespace tasks
}  // namespace spc
