#pragma once

#include "spc/core/task.h"
#include <mujoco/mujoco.h>

namespace spc {
namespace tasks {

class Particle : public core::Task {
public:
    Particle(mjModel* model, const spc::core::TaskConfig& config);

    void GetObservation(const mjModel* model, const mjData* data, float* obs_out) const override;
    double RunningCost(const mjModel* model, const mjData* data, const float* control) const override;
    double TerminalCost(const mjModel* model, const mjData* data) const override;

private:
    int pointmass_site_id_;
    int nu_;
    double pos_weight_;
    double vel_weight_;
    double ctrl_weight_;
};

} // namespace tasks
} // namespace spc
