#pragma once

#include "spc/tasks/humanoid_soccer.h"

namespace spc {
namespace tasks {

/**
 * @brief Humanoid soccer task with an augmented control vector.
 *
 * Extends HumanoidSoccer with a hierarchical control space:
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
class HumanoidSoccerAugmented : public HumanoidSoccer {
public:
    HumanoidSoccerAugmented(mjModel* model, const core::TaskConfig& config, HumanoidSpec spec);

    double RunningCost(const mjModel* model, const mjData* data, const float* control) const override;
    double TerminalCost(const mjModel* model, const mjData* data) const override;
    void ApplyControl(const mjModel* model, mjData* data, const float* control) const override;

private:
    // Smoothstep gate on the robot-ball distance; 1.0 when gating is disabled.
    double ResidualGate(const mjData* data) const;

    int leg_joint_start_;
    int leg_joint_count_;
    double residual_weight_;
    double gate_near_;  // full residual strength at/below this distance
    double gate_far_;   // zero residual strength at/above this distance (<=0 disables gating)
};

}  // namespace tasks
}  // namespace spc
