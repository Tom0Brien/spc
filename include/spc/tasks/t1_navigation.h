#pragma once

#include <mujoco/mujoco.h>

#include <cmath>

#include "spc/core/task.h"

namespace spc {
namespace tasks {

/**
 * @brief Booster T1 humanoid navigation task.
 *
 * Control dim is 3 (vx, vy, vtheta velocity commands).
 * The ONNX policy takes an 85-dim observation and outputs 23 motor targets.
 * ApplyControl converts velocity commands → observation → policy inference → motor targets.
 *
 * Cost minimizes distance and yaw error to a mocap goal body, penalizes tilt,
 * and regularizes controls.
 */
class T1Navigation : public core::Task {
public:
    T1Navigation(mjModel* model, const spc::core::TaskConfig& config);
    ~T1Navigation() override = default;

    void GetObservation(const mjModel* model, const mjData* data, float* obs_out) const override;
    double RunningCost(const mjModel* model, const mjData* data, const float* control) const override;
    double TerminalCost(const mjModel* model, const mjData* data) const override;
    void ApplyControl(const mjModel* model, mjData* data, const float* action) const override;
    int GetObsDim() const { return 85; }
    int GetActionDim() const { return 23; }

protected:
    // Sensor addresses
    int gyro_adr_;         // gyro sensor address (3)
    int linvel_adr_;       // local_linvel sensor address (3)
    int imu_site_id_;      // trunk IMU site for gravity projection and height

    // Default pose (23 robot joints)
    float default_pose_[23];

    // Action scale from environment config
    float action_scale_;
    float gait_freq_;

    // Velocity command limits (vx, vy, vtheta) applied to the high-level commands.
    float vel_limit_[3];

    double target_height_;
    double pos_weight_;
    double ori_weight_;
    double upright_weight_;
    double height_weight_;
    double ctrl_weight_;

    // Joint limits for clamping motor targets
    float jnt_range_low_[23];
    float jnt_range_high_[23];
};

}  // namespace tasks
}  // namespace spc
