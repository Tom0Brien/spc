#pragma once

#include <cmath>
#include <utility>

#include "spc/tasks/humanoid_pass.h"

namespace spc {
namespace tasks {

/**
 * @brief Augmented-control wrapper for the humanoid ball tasks.
 *
 * Extends a base ball task (HumanoidPass, HumanoidShoot, ...) with a
 * hierarchical control space:
 *   - control[0:3]                : velocity commands (vx, vy, vtheta) for the RL policy
 *   - control[3:3+leg_joint_count]: residual adjustments for the leg joint targets
 *
 * ApplyControl runs the RL policy on the velocity command (via the base class)
 * to obtain motor targets, then adds the leg residuals (starting at the spec's
 * leg_joint_start) and re-clamps to joint limits. The running cost additionally
 * regularizes the residuals to discourage destabilizing leg motions.
 *
 * When gate_near/gate_far are configured, the residuals are gated by the
 * robot's distance to the ball: zero beyond gate_far, full strength within
 * gate_near. This keeps the far-field approach a velocity-only problem and
 * engages the leg-swing residuals only near the ball.
 */
template <typename Base>
class HumanoidAugmented : public Base {
public:
    HumanoidAugmented(mjModel* model, const core::TaskConfig& config, HumanoidSpec spec)
        : Base(model, config, std::move(spec)) {
        auto num = [&config](const char* key, double fallback) {
            auto it = config.numeric_params.find(key);
            return it != config.numeric_params.end() ? it->second : fallback;
        };

        leg_joint_start_ = static_cast<int>(num("leg_joint_start", this->spec_.leg_joint_start));
        leg_joint_count_ = static_cast<int>(num("leg_joint_count", 12));
        residual_weight_ = num("residual_weight", 0.05);

        // Distance-to-ball residual gate; disabled unless configured.
        gate_near_ = num("gate_near", 0.0);
        gate_far_ = num("gate_far", 0.0);

        // Residual kicks need close contact that the behind-ball penalty fights
        // against, so it is disabled by default for the augmented tasks.
        this->behind_weight_ = num("behind_weight", 0.0);
    }

    void ApplyControl(const mjModel* model, mjData* data, const float* control) const override {
        // control[0:3] = velocity command, control[3:] = leg residuals.

        // Run the RL policy on the velocity command to write base motor targets.
        Base::ApplyControl(model, data, control);

        float gate = static_cast<float>(ResidualGate(data));
        if (gate <= 0.0f)
            return;

        // Add gated residuals to the leg motor targets and re-clamp.
        const float* residuals = control + 3;
        for (int i = 0; i < leg_joint_count_; ++i) {
            int j = leg_joint_start_ + i;
            float motor_target = static_cast<float>(data->ctrl[j]) + gate * residuals[i];

            if (motor_target < this->jnt_range_low_[j])
                motor_target = this->jnt_range_low_[j];
            if (motor_target > this->jnt_range_high_[j])
                motor_target = this->jnt_range_high_[j];

            data->ctrl[j] = motor_target;
        }
    }

    double RunningCost(const mjModel* model, const mjData* data, const float* control) const override {
        // Base cost; the base uses control[0:3] for the velocity control cost.
        double cost = Base::RunningCost(model, data, control);

        // Regularize the leg residuals control[3:3+leg_joint_count_].
        double residual_cost = 0.0;
        const float* residuals = control + 3;
        for (int i = 0; i < leg_joint_count_; ++i)
            residual_cost += residuals[i] * residuals[i];

        return cost + residual_weight_ * residual_cost;
    }

    double TerminalCost(const mjModel* model, const mjData* data) const override {
        float zero_ctrl[3 + 32] = {0.0f};
        return RunningCost(model, data, zero_ctrl);
    }

private:
    // Smoothstep gate on the robot-ball distance; 1.0 when gating is disabled.
    double ResidualGate(const mjData* data) const {
        if (gate_far_ <= gate_near_)
            return 1.0;
        const mjtNum* ball_pos = data->xpos + 3 * this->soccer_ball_id_;
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

    int leg_joint_start_;
    int leg_joint_count_;
    double residual_weight_;
    double gate_near_;  // full residual strength at/below this distance
    double gate_far_;   // zero residual strength at/above this distance (<=0 disables gating)
};

}  // namespace tasks
}  // namespace spc
