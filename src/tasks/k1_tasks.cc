// Booster K1 (22 DoF) registrations of the shared humanoid tasks.

#include "spc/core/task_factory.h"
#include "spc/tasks/humanoid_navigation.h"

namespace spc {
namespace tasks {
namespace {

HumanoidSpec K1Spec() {
    HumanoidSpec spec;
    spec.njoints = 22;
    // Default pose from the mujoco_playground K1 Joystick "home" keyframe (the
    // joint targets the RL policy was trained with).
    spec.default_pose = {
        0.0f,  0.0f,                              // head (2)
        0.0f,  -1.3f, 0.0f, 0.0f,                 // left arm (4)
        0.0f,  1.3f,  0.0f, 0.0f,                 // right arm (4)
        -0.2f, 0.0f,  0.0f, 0.4f,  -0.2f, 0.0f,   // left leg (6)
        -0.2f, 0.0f,  0.0f, 0.4f,  -0.2f, 0.0f    // right leg (6)
    };
    spec.gyro_name = "gyro";
    spec.linvel_name = "local_linvel";
    spec.upright_site = "imu";
    spec.height_site = "imu";
    // Per-joint action scales (0.25 * effort_limit / kp), following
    // booster_train; the K1 policy was trained with these.
    spec.action_scale = 1.0f;
    spec.action_scale_vec = {
        0.37994f, 0.37994f,                                          // head
        0.88653f, 0.88653f, 0.88653f, 0.88653f,                      // left arm
        0.88653f, 0.88653f, 0.88653f, 0.88653f,                      // right arm
        0.56289f, 0.88586f, 0.53653f, 0.46356f, 0.26827f, 0.26827f,  // left leg
        0.56289f, 0.88586f, 0.53653f, 0.46356f, 0.26827f, 0.26827f   // right leg
    };
    spec.vel_limit = {1.0f, 0.8f, 1.0f};
    spec.target_height = 0.543;
    // The K1 policy observes all joints (no zeroed head entries) and pins the
    // gait phase at zero command, matching the mujoco_playground K1 joystick.
    spec.zero_obs_joints = 0;
    spec.pin_phase_when_standing = true;
    // Joint order is head(2), arms(8), legs(12): legs start at 10.
    spec.leg_joint_start = 10;
    return spec;
}

class K1Navigation : public HumanoidNavigation {
public:
    K1Navigation(mjModel* model, const core::TaskConfig& config) : HumanoidNavigation(model, config, K1Spec()) {}
};

}  // namespace
}  // namespace tasks
}  // namespace spc

REGISTER_TASK("K1Navigation", spc::tasks::K1Navigation, K1Navigation)
