#include <gtest/gtest.h>
#include <mujoco/mujoco.h>

#include "spc/tasks/particle.h"

// We use the absolute path to the local spc/models directory for testing
// MODELS_DIR is passed as a macro via CMake
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define MODEL_PATH TOSTRING(MODELS_DIR) "/particle/scene.xml"

TEST(ParticleTaskTest, InitializationAndCost) {
    char error[1000];
    mjModel* m = mj_loadXML(MODEL_PATH, nullptr, error, 1000);
    ASSERT_NE(m, nullptr) << "Failed to load particle model: " << error;

    spc::tasks::Particle task(m, spc::core::TaskConfig{});

    mjData* d = mj_makeData(m);

    // Shift mocap to guarantee cost > 0
    d->mocap_pos[0] = 0.0;
    d->mocap_pos[1] = 0.1;
    d->mocap_pos[2] = 0.0;

    // Forward kinematics to compute site positions
    mj_forward(m, d);

    // Test terminal cost is > 0 when not at target
    double tc = task.TerminalCost(m, d);
    EXPECT_GT(tc, 0.0);

    std::vector<float> dummy_control(m->nu, 0.0f);
    double rc = task.RunningCost(m, d, dummy_control.data());
    EXPECT_GT(rc, 0.0);

    // Since control is zero, running cost is just state_cost + 0.1*0 = tc
    EXPECT_FLOAT_EQ(rc, tc);

    mj_deleteData(d);
    mj_deleteModel(m);
}
