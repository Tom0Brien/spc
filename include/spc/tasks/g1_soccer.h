#pragma once

#include "spc/tasks/g1_navigation.h"

namespace spc {
namespace tasks {

class G1Soccer : public G1Navigation {
public:
    G1Soccer(mjModel* model, const spc::core::TaskConfig& config);

    double RunningCost(const mjModel* model, const mjData* data, const float* control) const override;
    double TerminalCost(const mjModel* model, const mjData* data) const override;

private:
    int soccer_ball_id_;

    // config params for soccer behavior
    double standoff_distance_;
    double ball_goal_weight_;
};

}  // namespace tasks
}  // namespace spc
