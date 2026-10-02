// Booster K1 (22 DoF) registrations of the shared humanoid tasks.

#include "spc/core/task_factory.h"
#include "spc/tasks/humanoid_augmented.h"
#include "spc/tasks/humanoid_navigation.h"
#include "spc/tasks/humanoid_pass.h"

namespace spc {
namespace tasks {
namespace {

HumanoidSpec K1Spec() {
    HumanoidSpec spec;
    spec.njoints = 22;
    // Walk policy trained with booster_mjlab's velocity task
    // (github.com/IntelligentRoboticsLab/booster_mjlab, sim-to-real tested;
    // policies/k1_navigation.onnx is run 2026-09-19_15-15-30) on the robot in
    // models/k1/k1.xml. Default pose is its HOME_KEYFRAME (the action offset).
    spec.default_pose = {
        0.0f,  0.0f,                             // head (2)
        0.0f,  -1.4f, 0.0f, -0.4f,               // left arm (4)
        0.0f,  1.4f,  0.0f, 0.4f,                // right arm (4)
        -0.4f, 0.0f,  0.0f, 0.8f,  -0.4f, 0.0f,  // left leg (6)
        -0.4f, 0.0f,  0.0f, 0.8f,  -0.4f, 0.0f   // right leg (6)
    };
    spec.gyro_name = "imu_ang_vel";
    spec.linvel_name = "imu_lin_vel";  // unused by the mjlab observation
    spec.upright_site = "imu";
    spec.height_site = "imu";
    // Per-joint action scales (0.25 * effort_limit / kp), K1_ACTION_SCALE in
    // booster_mjlab.
    spec.action_scale = 1.0f;
    spec.action_scale_vec = {
        0.375f,  0.375f,                                        // head
        0.35f,   0.35f,   0.35f,      0.35f,                    // left arm
        0.35f,   0.35f,   0.35f,      0.35f,                    // right arm
        0.2125f, 0.2375f, 0.1196875f, 0.35f, 0.1915f, 0.1915f,  // left leg
        0.2125f, 0.2375f, 0.1196875f, 0.35f, 0.1915f, 0.1915f   // right leg
    };
    // Inside the final command curriculum stage (x [-1.5, 1.75], y +-1.75,
    // yaw +-1.5); vx is symmetric here so it is bounded by the backward range.
    spec.vel_limit = {1.0f, 1.0f, 1.5f};
    spec.target_height = 0.54;
    spec.obs_layout = ObsLayout::kMjlab;
    spec.clamp_targets = false;
    // Joint order is head(2), arms(8), legs(12): legs start at 10.
    spec.leg_joint_start = 10;
    return spec;
}

class K1Navigation : public HumanoidNavigation {
public:
    K1Navigation(mjModel* model, const core::TaskConfig& config) : HumanoidNavigation(model, config, K1Spec()) {}
};

class K1Pass : public HumanoidPass {
public:
    K1Pass(mjModel* model, const core::TaskConfig& config) : HumanoidPass(model, config, K1Spec()) {}
};

class K1PassAugmented : public HumanoidAugmented<HumanoidPass> {
public:
    K1PassAugmented(mjModel* model, const core::TaskConfig& config)
        : HumanoidAugmented<HumanoidPass>(model, config, K1Spec()) {}
};

}  // namespace
}  // namespace tasks
}  // namespace spc

REGISTER_TASK("K1Navigation", spc::tasks::K1Navigation, K1Navigation)
REGISTER_TASK("K1Pass", spc::tasks::K1Pass, K1Pass)
REGISTER_TASK("K1PassAugmented", spc::tasks::K1PassAugmented, K1PassAugmented)
