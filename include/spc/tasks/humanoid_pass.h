#pragma once

#include "spc/tasks/humanoid_navigation.h"

namespace spc {
namespace tasks {

/**
 * @brief Humanoid pass task: navigate to a ball and deliver it precisely to a
 * mocap target point.
 *
 * Inherits the navigation control pipeline (velocity commands -> RL policy ->
 * motor targets) and replaces the cost: ball-to-target distance (symmetric, so
 * overshoot is penalized like shortfall), a standoff position behind the ball,
 * an approach/kick orientation blend, and the navigation stability terms.
 *
 * For shooting through a goal line where overshoot is free, use HumanoidShoot.
 */
class HumanoidPass : public HumanoidNavigation {
public:
    HumanoidPass(mjModel* model, const core::TaskConfig& config, HumanoidSpec spec);

    double RunningCost(const mjModel* model, const mjData* data, const float* control) const override;
    double TerminalCost(const mjModel* model, const mjData* data) const override;

protected:
    int soccer_ball_id_;

    double standoff_distance_;
    double ball_goal_weight_;
    double ball_goal_scale_;  // pseudo-Huber scale for the ball->goal distance (meters)
    double behind_weight_;    // penalize being on the goal side of the ball when close
};

}  // namespace tasks
}  // namespace spc
