#pragma once

#include "spc/tasks/t1_soccer.h"

namespace spc {
namespace tasks {

/**
 * @brief T1 soccer task with an augmented control vector.
 *
 * Extends T1Soccer with a hierarchical control space:
 *   - control[0:3]                : velocity commands (vx, vy, vtheta) for the RL policy
 *   - control[3:3+leg_joint_count]: residual adjustments for the leg joint targets
 *
 * ApplyControl runs the RL policy on the velocity command (via the base class)
 * to obtain motor targets, then adds the leg residuals to the leg joint targets
 * and re-clamps to joint limits. The running cost additionally regularizes the
 * residuals to discourage destabilizing leg motions.
 *
 * Unlike the G1 (whose 12 leg joints come first), the T1 joint order is
 * head(2), arms(8), waist(1), left leg(6), right leg(6), so the leg residuals
 * are applied starting at leg_joint_start (default 11).
 *
 * The residuals are gated by the robot's distance to the ball: they fade to
 * zero beyond gate_far and reach full strength within gate_near. This keeps
 * the far-field approach a clean velocity-only problem (so CEM's samples all
 * inform the velocity command instead of being diluted across 12 useless
 * residual dims), and only engages the leg-swing residuals for the kick when
 * the robot is actually next to the ball.
 */
class T1SoccerAugmented : public T1Soccer {
public:
    T1SoccerAugmented(mjModel* model, const spc::core::TaskConfig& config);

    double RunningCost(const mjModel* model, const mjData* data, const float* control) const override;
    double TerminalCost(const mjModel* model, const mjData* data) const override;
    void ApplyControl(const mjModel* model, mjData* data, const float* control) const override;

private:
    // Smoothstep gate on the robot-ball distance (metres).
    double ResidualGate(const mjData* data) const;

    int leg_joint_start_;  // index of the first leg joint (11 for T1)
    int leg_joint_count_;
    double residual_weight_;
    double gate_near_;  // full residual strength at/below this distance
    double gate_far_;   // zero residual strength at/above this distance
};

}  // namespace tasks
}  // namespace spc
