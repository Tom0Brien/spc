#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include "spc/algs/cem.h"
#include "spc/core/task.h"
#include "spc/core/policy.h"

using namespace spc;

// Dummy implementations for testing
class DummyTask : public core::Task {
public:
    void GetObservation(const mjModel* model, const mjData* data, float* obs_out) const override {}
    double RunningCost(const mjModel* model, const mjData* data, const float* control) const override { return 0.0; }
    double TerminalCost(const mjModel* model, const mjData* data) const override { return 0.0; }
};

class QuadraticTask : public core::Task {
public:
    void GetObservation(const mjModel* model, const mjData* data, float* obs_out) const override {}
    double RunningCost(const mjModel* model, const mjData* data, const float* control) const override { 
        // Cost is simply (u - 0.5)^2 for the single actuator
        double diff = control[0] - 0.5;
        return diff * diff;
    }
    double TerminalCost(const mjModel* model, const mjData* data) const override { return 0.0; }
};

class DummyPolicy : public core::Policy {
public:
    void Load(const std::string& model_path) override {}
    void ComputeAction(const float* obs, int obs_dim, float* action, int action_dim) const override {}
};

TEST(CEMTest, Initialization) {
    // Create a minimal mujoco model with 1 actuator (nu=1) from an XML string
    const char* xml = R"(
    <mujoco>
        <worldbody>
            <body pos="0 0 0">
                <joint name="j" type="slide" axis="1 0 0"/>
                <geom type="sphere" size="1"/>
            </body>
        </worldbody>
        <actuator>
            <motor joint="j" ctrlrange="-1 1"/>
        </actuator>
    </mujoco>)";
    
    const char* filename = "/tmp/dummy_model.xml";
    FILE* fp = fopen(filename, "w");
    fputs(xml, fp);
    fclose(fp);

    char error[1000];
    mjModel* m = mj_loadXML(filename, nullptr, error, 1000);
    ASSERT_NE(m, nullptr) << "Failed to load dummy model: " << error;

    auto task = std::make_shared<DummyTask>();
    auto policy = std::make_shared<DummyPolicy>();
    algs::CEMConfig config;
    config.num_samples = 4; // Keep small for basic testing

    // Initialize the optimizer
    algs::CEM cem(m, task, policy, config);

    // Run optimization
    std::vector<float> best_action(config.control_dim, 0.0f);
    mjData* current_state = mj_makeData(m);
    
    // Optimize should execute the multi-threaded rollout loop
    ASSERT_NO_THROW(cem.Optimize(current_state, best_action.data()));
    
    mj_deleteData(current_state);
    mj_deleteModel(m);
}

TEST(CEMTest, QuadraticOptimization) {
    const char* xml = R"(
    <mujoco>
        <worldbody>
            <body pos="0 0 0">
                <joint name="j" type="slide" axis="1 0 0"/>
                <geom type="sphere" size="1"/>
            </body>
        </worldbody>
        <actuator>
            <motor joint="j" ctrlrange="-1 1"/>
        </actuator>
    </mujoco>)";
    const char* filename = "/tmp/dummy_model2.xml";
    FILE* fp = fopen(filename, "w");
    fputs(xml, fp);
    fclose(fp);

    char error[1000];
    mjModel* m = mj_loadXML(filename, nullptr, error, 1000);
    ASSERT_NE(m, nullptr) << "Failed to load dummy model: " << error;

    auto task = std::make_shared<QuadraticTask>();
    auto policy = std::make_shared<DummyPolicy>();
    
    algs::CEMConfig config;
    config.num_samples = 256;
    config.num_elites = 24;
    config.num_knots = 4;
    config.num_iterations = 5; // Enough iterations to converge to 0.5
    config.plan_horizon_steps = 10;
    config.control_dim = 1;
    config.sigma_init = 0.5f;

    algs::CEM cem(m, task, policy, config);

    std::vector<float> best_action(config.control_dim, 0.0f);
    mjData* current_state = mj_makeData(m);
    
    cem.Optimize(current_state, best_action.data());
    
    // We expect the optimal action to be close to 0.5
    EXPECT_NEAR(best_action[0], 0.5f, 0.05f);
    
    mj_deleteData(current_state);
    mj_deleteModel(m);
}
