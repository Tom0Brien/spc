// Booster T1 (23 DoF) registrations of the shared humanoid tasks.

#include "spc/core/task_factory.h"
#include "spc/tasks/humanoid_navigation.h"
#include "spc/tasks/humanoid_soccer.h"
#include "spc/tasks/humanoid_soccer_augmented.h"

namespace spc {
namespace tasks {
namespace {

HumanoidSpec T1Spec() {
    HumanoidSpec spec;
    spec.njoints = 23;
    // Default pose from the mujoco_playground T1 Joystick "home" keyframe (the
    // joint targets the RL policy was trained with).
    spec.default_pose = {
        0.0f,  0.0f,                              // head (2)
        0.0f,  -1.4f, 0.0f, -0.4f,                // left arm (4)
        0.0f,  1.4f,  0.0f, 0.4f,                 // right arm (4)
        0.0f,                                     // waist (1)
        -0.2f, 0.0f,  0.0f, 0.4f,  -0.2f, 0.0f,   // left leg (6)
        -0.2f, 0.0f,  0.0f, 0.4f,  -0.2f, 0.0f    // right leg (6)
    };
    spec.gyro_name = "gyro";
    spec.linvel_name = "local_linvel";
    spec.upright_site = "imu";
    spec.height_site = "imu";
    spec.action_scale = 1.0f;
    spec.vel_limit = {1.0f, 0.8f, 1.0f};
    spec.target_height = 0.665;
    // Head joint angles/velocities are zeroed in the observation and the gait
    // phase is pinned at zero command, matching the T1 sim2sim deployment.
    spec.zero_obs_joints = 2;
    spec.pin_phase_when_standing = true;
    // Joint order is head(2), arms(8), waist(1), legs(12): legs start at 11.
    spec.leg_joint_start = 11;
    return spec;
}

class T1Navigation : public HumanoidNavigation {
public:
    T1Navigation(mjModel* model, const core::TaskConfig& config) : HumanoidNavigation(model, config, T1Spec()) {}
};

class T1Soccer : public HumanoidSoccer {
public:
    T1Soccer(mjModel* model, const core::TaskConfig& config) : HumanoidSoccer(model, config, T1Spec()) {}
};

class T1SoccerAugmented : public HumanoidSoccerAugmented {
public:
    T1SoccerAugmented(mjModel* model, const core::TaskConfig& config)
        : HumanoidSoccerAugmented(model, config, T1Spec()) {}
};

}  // namespace
}  // namespace tasks
}  // namespace spc

REGISTER_TASK("T1Navigation", spc::tasks::T1Navigation, T1Navigation)
REGISTER_TASK("T1Soccer", spc::tasks::T1Soccer, T1Soccer)
REGISTER_TASK("T1SoccerAugmented", spc::tasks::T1SoccerAugmented, T1SoccerAugmented)
