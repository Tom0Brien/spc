#pragma once

#include <mujoco/mujoco.h>

#include <memory>
#include <vector>

#include "spc/core/policy.h"
#include "spc/core/task.h"

namespace spc {
namespace algs {

/**
 * @brief Base configuration for any Sampling-Based Optimizer.
 */
struct OptimizerConfig {
    int num_samples = 128;
    int num_knots = 5;       // Number of control spline knots
    int num_iterations = 1;  // Iterations per Optimize() call
    int plan_horizon_steps = 25;
    int sim_substeps = 4;
    int control_dim = 1;
    int obs_dim = 1;      // Size of the observation array
    int num_threads = 8;  // Number of OpenMP threads to use
};

/**
 * @brief Abstract base class for Sampling-Based Optimizers.
 *
 * Owns the task, policy, and MuJoCo model required to execute rollouts.
 * Implements the core multi-threaded rollout evaluation logic.
 */
class Optimizer {
public:
    Optimizer(mjModel* model, std::shared_ptr<core::Task> task, std::shared_ptr<core::Policy> policy,
              const OptimizerConfig& config);

    virtual ~Optimizer();

    /**
     * @brief Get the configured control dimension.
     */
    int GetControlDim() const { return config_.control_dim; }

    /**
     * @brief Get the associated task.
     */
    std::shared_ptr<core::Task> GetTask() const { return task_; }

    /**
     * @brief Execute the optimization loop.
     */
    virtual void Optimize(const mjData* current_state, float* best_action_out);

protected:
    // Called once at the start of each Optimize() call, before sampling.
    // Lets subclasses warm-start the search distribution for a receding horizon.
    virtual void PrepareForReplan() {}

    virtual void SampleKnots(std::vector<float>& samples) = 0;
    virtual void UpdateDistribution(const std::vector<float>& samples, const std::vector<double>& costs) = 0;
    virtual void GetBestAction(const mjData* current_state, float* best_action_out) = 0;

    void EvaluateRollouts(const mjData* current_state, const std::vector<float>& samples, std::vector<double>& costs);

    mjModel* model_;
    std::shared_ptr<core::Task> task_;
    std::shared_ptr<core::Policy> policy_;
    OptimizerConfig config_;

    std::vector<mjData*> thread_datas_;
};

}  // namespace algs
}  // namespace spc
