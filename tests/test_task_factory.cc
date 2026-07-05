#include <gtest/gtest.h>
#include <mujoco/mujoco.h>

#include <memory>
#include <stdexcept>

#include "spc/core/task.h"
#include "spc/core/task_factory.h"

using spc::core::Task;
using spc::core::TaskConfig;
using spc::core::TaskFactory;

namespace {

// Minimal task whose costs echo a config param, so we can verify the factory
// forwards the TaskConfig to the constructor.
class FactoryDummyTask : public Task {
public:
    FactoryDummyTask(mjModel* /*model*/, const TaskConfig& config) {
        auto it = config.numeric_params.find("weight");
        weight_ = (it != config.numeric_params.end()) ? it->second : -1.0;
    }

    void GetObservation(const mjModel*, const mjData*, float*) const override {}
    double RunningCost(const mjModel*, const mjData*, const float*) const override { return weight_; }
    double TerminalCost(const mjModel*, const mjData*) const override { return weight_; }

    double weight_ = 0.0;
};

}  // namespace

TEST(TaskFactoryTest, RegisterAndCreate) {
    auto& factory = TaskFactory::GetInstance();
    factory.Register("FactoryDummyTask", [](mjModel* model, const TaskConfig& config) {
        return std::make_shared<FactoryDummyTask>(model, config);
    });

    TaskConfig config;
    config.numeric_params["weight"] = 42.0;

    auto task = factory.Create("FactoryDummyTask", nullptr, config);
    ASSERT_NE(task, nullptr);

    // The config must have reached the constructed task.
    EXPECT_DOUBLE_EQ(task->TerminalCost(nullptr, nullptr), 42.0);
}

TEST(TaskFactoryTest, UnknownTaskThrows) {
    auto& factory = TaskFactory::GetInstance();
    EXPECT_THROW(factory.Create("NoSuchTaskRegistered", nullptr, TaskConfig{}), std::invalid_argument);
}

TEST(TaskFactoryTest, GetInstanceIsSingleton) { EXPECT_EQ(&TaskFactory::GetInstance(), &TaskFactory::GetInstance()); }
