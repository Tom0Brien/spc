// Unitree G1 (29 DoF) registrations of the shared humanoid tasks.

#include "spc/core/task_factory.h"
#include "spc/tasks/humanoid_navigation.h"
#include "spc/tasks/humanoid_soccer.h"
#include "spc/tasks/humanoid_soccer_augmented.h"

namespace spc {
namespace tasks {
namespace {

HumanoidSpec G1Spec() {
    HumanoidSpec spec;
    spec.njoints = 29;
    // Default pose from mujoco_playground G1JoystickFlatTerrain (the joint
    // targets the RL policy was trained with).
    spec.default_pose = {
        -0.312f, 0.0f,  0.0f,   0.669f, -0.363f, 0.0f,        // left leg (6)
        -0.312f, 0.0f,  0.0f,   0.669f, -0.363f, 0.0f,        // right leg (6)
        0.0f,    0.0f,  0.073f,                               // waist (3)
        0.2f,    0.2f,  0.0f,   0.6f,   0.0f,    0.0f, 0.0f,  // left arm (7)
        0.2f,    -0.2f, 0.0f,   0.6f,   0.0f,    0.0f, 0.0f   // right arm (7)
    };
    spec.gyro_name = "gyro_pelvis";
    spec.linvel_name = "local_linvel_pelvis";
    spec.upright_site = "imu_in_pelvis";
    spec.height_site = "imu_in_torso";
    spec.action_scale = 0.5f;
    spec.vel_limit = {1.0f, 1.0f, 1.0f};
    spec.target_height = 0.75;
    return spec;
}

class G1Navigation : public HumanoidNavigation {
public:
    G1Navigation(mjModel* model, const core::TaskConfig& config) : HumanoidNavigation(model, config, G1Spec()) {}
};

class G1Soccer : public HumanoidSoccer {
public:
    G1Soccer(mjModel* model, const core::TaskConfig& config) : HumanoidSoccer(model, config, G1Spec()) {}
};

class G1SoccerAugmented : public HumanoidSoccerAugmented {
public:
    G1SoccerAugmented(mjModel* model, const core::TaskConfig& config)
        : HumanoidSoccerAugmented(model, config, G1Spec()) {}
};

}  // namespace
}  // namespace tasks
}  // namespace spc

REGISTER_TASK("G1Navigation", spc::tasks::G1Navigation, G1Navigation)
REGISTER_TASK("G1Soccer", spc::tasks::G1Soccer, G1Soccer)
REGISTER_TASK("G1SoccerAugmented", spc::tasks::G1SoccerAugmented, G1SoccerAugmented)
