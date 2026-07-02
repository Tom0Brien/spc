#pragma once

#include "spc/tasks/g1_soccer.h"

namespace spc {
namespace tasks {

/**
 * @brief G1 soccer task with an augmented control vector.
 *
 * Extends G1Soccer with a hierarchical control space:
 *   - control[0:3]  : velocity commands (vx, vy, vtheta) for the RL policy
 *   - control[3:15] : residual adjustments for the first 12 (leg) joint targets
 *
 * ApplyControl runs the RL policy on the velocity command (via the base class)
 * to obtain motor targets, then adds the leg residuals to the first 12 targets
 * and re-clamps to joint limits. The running cost additionally regularizes the
 * residuals to discourage destabilizing leg motions.
 */
class G1SoccerAugmented : public G1Soccer {
public:
    G1SoccerAugmented(mjModel* model, const spc::core::TaskConfig& config);

    double RunningCost(const mjModel* model, const mjData* data, const float* control) const override;
    double TerminalCost(const mjModel* model, const mjData* data) const override;
    void ApplyControl(const mjModel* model, mjData* data, const float* control) const override;

private:
    int leg_joint_count_;
    double residual_weight_;
};

}  // namespace tasks
}  // namespace spc
