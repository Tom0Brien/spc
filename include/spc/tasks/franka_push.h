#pragma once

#include <mujoco/mujoco.h>

#include "spc/core/task.h"

namespace spc {
namespace tasks {

class FrankaPush : public core::Task {
public:
    FrankaPush(mjModel* model, const spc::core::TaskConfig& config);
    ~FrankaPush() override = default;

    void GetObservation(const mjModel* model, const mjData* data, float* obs_out) const override;
    double RunningCost(const mjModel* model, const mjData* data, const float* control) const override;
    double TerminalCost(const mjModel* model, const mjData* data) const override;
    void ApplyControl(const mjModel* model, mjData* data, const float* action) const override;
    int GetObsDim() const { return 48; }

private:
    int obj_body_;
    int gripper_site_;
    int mocap_target_;
    float action_scale_;

    double obj_target_weight_;
    double gripper_obj_weight_;
    double orientation_weight_;
    double residual_weight_;
};

}  // namespace tasks
}  // namespace spc
