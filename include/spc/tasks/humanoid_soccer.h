#pragma once

#include "spc/tasks/humanoid_navigation.h"

namespace spc {
namespace tasks {

/**
 * @brief Humanoid soccer task: navigate to a ball and push it toward a mocap goal.
 *
 * Inherits the navigation control pipeline (velocity commands -> RL policy ->
 * motor targets) and replaces the cost: ball-to-goal distance, a standoff
 * position behind the ball, an approach/kick orientation blend, and the
 * navigation stability terms.
 */
class HumanoidSoccer : public HumanoidNavigation {
public:
    HumanoidSoccer(mjModel* model, const core::TaskConfig& config, HumanoidSpec spec);

    double RunningCost(const mjModel* model, const mjData* data, const float* control) const override;
    double TerminalCost(const mjModel* model, const mjData* data) const override;

protected:
    int soccer_ball_id_;
    int ball_dofadr_;  // qvel address of the ball freejoint

    double standoff_distance_;
    double ball_goal_weight_;
    double ball_vel_weight_;  // reward ball velocity toward goal
    double behind_weight_;    // penalize being on the goal side of the ball when close
};

}  // namespace tasks
}  // namespace spc
