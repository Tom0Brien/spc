#pragma once

#include "spc/core/task.h"
#include <mujoco/mujoco.h>
#include <cmath>

namespace spc {
namespace tasks {

/**
 * @brief G1 humanoid navigation task.
 *
 * Control dim is 3 (vx, vy, vtheta velocity commands).
 * The ONNX policy takes a 103-dim observation and outputs 29 motor targets.
 * ApplyControl converts velocity commands → observation → policy inference → motor targets.
 *
 * Cost minimizes distance and yaw error to a mocap goal body, penalizes tilt,
 * and regularizes controls.
 */
class G1Navigation : public core::Task {
public:
    G1Navigation(mjModel* model, const spc::core::TaskConfig& config);
    ~G1Navigation() override = default;

    void GetObservation(const mjModel* model, const mjData* data, float* obs_out) const override;
    double RunningCost(const mjModel* model, const mjData* data, const float* control) const override;
    double TerminalCost(const mjModel* model, const mjData* data) const override;
    void ApplyControl(const mjModel* model, mjData* data, const float* action) const override;
    int GetObsDim() const { return 103; }
    int GetActionDim() const { return 29; }

private:
    // Sensor addresses
    int pelvis_gyro_adr_;       // gyro_pelvis sensor address (3)
    int pelvis_linvel_adr_;     // local_linvel_pelvis sensor address (3)
    int pelvis_imu_site_id_;    // pelvis IMU site for gravity projection

    // Body/site IDs
    int torso_site_id_;

    // Default pose (29 robot joints)
    float default_pose_[29];

    // Action scale from environment config
    float action_scale_;

    // Gait phase tracking (mutable since it changes with time)
    float gait_freq_;

    // Joint limits for clamping motor targets
    float jnt_range_low_[29];
    float jnt_range_high_[29];
};

} // namespace tasks
} // namespace spc
