#pragma once

#include <mujoco/mujoco.h>

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "spc/core/task.h"

namespace spc {
namespace tasks {

/**
 * @brief Pseudo-Huber norm of a scaled error: sqrt((err/scale)^2 + 1) - 1.
 *
 * Quadratic near zero, linear beyond ~scale. Used for all tracking terms so a
 * distant goal contributes a constant gradient instead of a quadratic blowup
 * that drowns the stability terms. Returns ~1 at err ~= 1.7*scale, so weights
 * are comparable across terms.
 */
inline double PseudoHuber(double err, double scale) {
    double x = err / scale;
    return std::sqrt(x * x + 1.0) - 1.0;
}

/// Yaw (z rotation) extracted from a wxyz quaternion.
inline double YawFromQuat(const mjtNum* q) {
    return std::atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]));
}

/// Wrap an angle difference to [-pi, pi].
inline double WrapAngle(double a) {
    return std::atan2(std::sin(a), std::cos(a));
}

/**
 * @brief Robot-specific constants for the shared humanoid tasks.
 *
 * The G1 and T1 task logic is identical; only these constants differ. Config
 * params with the same names override the spec defaults at construction.
 */
struct HumanoidSpec {
    int njoints = 0;                  // actuated robot joints (excludes floating base)
    std::vector<float> default_pose;  // default joint targets the RL policy was trained with
    std::string gyro_name;
    std::string linvel_name;
    std::string upright_site;  // IMU site used for gravity projection
    std::string height_site;   // site whose z is regulated to target_height
    float action_scale = 0.5f;
    // Per-joint action scales (K1: 0.25*effort_limit/kp per booster_train).
    // Empty means uniform action_scale; otherwise njoints entries, each
    // multiplied by action_scale.
    std::vector<float> action_scale_vec;
    float gait_freq = 1.5f;
    std::array<float, 3> vel_limit = {1.0f, 1.0f, 1.0f};  // vx, vy, vtheta command bounds
    double target_height = 0.0;
    int zero_obs_joints = 0;              // leading joints zeroed in the observation (T1 head)
    bool pin_phase_when_standing = false;  // gait phase pinned to pi at ~zero command (T1)
    int leg_joint_start = 0;               // first leg joint index (augmented residuals)
};

/**
 * @brief Humanoid navigation task.
 *
 * Control dim is 3 (vx, vy, vtheta velocity commands). The RL policy takes the
 * observation (16 + 3*njoints dims) and outputs njoints motor targets;
 * ApplyControl converts velocity commands -> observation -> policy inference
 * -> motor targets.
 *
 * Cost minimizes distance and yaw error to a mocap goal body, penalizes tilt
 * and height error, and regularizes controls.
 */
class HumanoidNavigation : public core::Task {
public:
    HumanoidNavigation(mjModel* model, const core::TaskConfig& config, HumanoidSpec spec);
    ~HumanoidNavigation() override = default;

    void GetObservation(const mjModel* model, const mjData* data, float* obs_out) const override;
    double RunningCost(const mjModel* model, const mjData* data, const float* control) const override;
    double TerminalCost(const mjModel* model, const mjData* data) const override;
    void ApplyControl(const mjModel* model, mjData* data, const float* action) const override;
    int GetObsDim() const { return 16 + 3 * spec_.njoints; }
    int GetActionDim() const { return spec_.njoints; }

protected:
    HumanoidSpec spec_;

    // Sensor addresses and site IDs
    int gyro_adr_;
    int linvel_adr_;
    int upright_site_id_;
    int height_site_id_;

    float action_scale_;
    std::vector<float> action_scale_joint_;  // per-joint scale (njoints)
    float gait_freq_;
    float vel_limit_[3];
    float cmd_deadzone_;  // |cmd| below this snaps to zero (policy stand threshold)

    double target_height_;
    double pos_weight_;
    double ori_weight_;
    double upright_weight_;
    double height_weight_;
    double ctrl_weight_;

    // Characteristic error scales for the pseudo-Huber tracking terms: the
    // cost transitions from quadratic to linear around these magnitudes.
    double pos_scale_;  // meters
    double ori_scale_;  // radians

    // Joint limits for clamping motor targets
    std::vector<float> jnt_range_low_;
    std::vector<float> jnt_range_high_;
};

}  // namespace tasks
}  // namespace spc
