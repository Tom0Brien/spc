#pragma once

#include "spc/tasks/g1_navigation.h"

namespace spc {
namespace tasks {

class G1Soccer : public G1Navigation {
public:
    G1Soccer(mjModel* model, const spc::core::TaskConfig& config);

    double RunningCost(const mjModel* model, const mjData* data, const float* control) const override;
    double TerminalCost(const mjModel* model, const mjData* data) const override;

protected:
    int soccer_ball_id_;
    int ball_dofadr_;  // qvel address of the ball freejoint

    // config params for soccer behavior
    double standoff_distance_;
    double ball_goal_weight_;
    double ball_vel_weight_;  // reward ball velocity toward goal
    double behind_weight_;    // penalize being on the goal side of the ball when close
};

}  // namespace tasks
}  // namespace spc
