#pragma once

#include "spc/tasks/humanoid_pass.h"

namespace spc {
namespace tasks {

/**
 * @brief Humanoid shoot task: drive the ball through a goal line at maximum
 * power while staying stable.
 *
 * The mocap goal body defines a goal *region*, not a point: its yaw sets the
 * shooting direction n (the +x axis of the marker), and goal_half_width sets
 * the region's lateral extent. The ball cost is
 *   - shortfall: distance remaining along n to the goal line, zero once the
 *     ball has crossed it — overshoot is free, unlike HumanoidPass;
 *   - lateral miss: lateral distance beyond the region half-width, so shots
 *     inside the mouth are free laterally;
 *   - power: a reward on the ball's velocity component along n while the ball
 *     is still in front of the line ("maximum power").
 *
 * The robot approach (standoff behind the ball along -n), orientation blend,
 * and stability terms mirror HumanoidPass.
 */
class HumanoidShoot : public HumanoidPass {
public:
    HumanoidShoot(mjModel* model, const core::TaskConfig& config, HumanoidSpec spec);

    double RunningCost(const mjModel* model, const mjData* data, const float* control) const override;
    double TerminalCost(const mjModel* model, const mjData* data) const override;

protected:
    int ball_dofadr_;  // qvel address of the ball freejoint (power reward)

    double goal_half_width_;     // lateral half-extent of the goal region (meters)
    double lateral_weight_;      // penalize lateral miss beyond the half-width
    double shoot_power_weight_;  // reward ball speed along the shooting direction
};

}  // namespace tasks
}  // namespace spc
