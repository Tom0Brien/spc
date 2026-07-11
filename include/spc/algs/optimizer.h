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

    // Phase-indexed control dims: the trailing phase_dims entries of the
    // control vector are interpolated periodically over one cycle of
    // phase_freq (Hz) against absolute sim time, instead of over the horizon.
    // Useful for gait-locked terms (e.g. leg residuals), which then stay
    // valid across replans without any warm-start re-timing.
    int phase_dims = 0;
    float phase_freq = 0.0f;

    // Early termination of dominated rollouts: a rollout whose accumulated
    // cost already exceeds the k-th best completed cost this iteration can
    // never enter the elite set, so it stops stepping. Lossless for the CEM
    // elite update, but requires nonnegative running/terminal costs; disable
    // for tasks whose cost can be negative. Only saves wall time when
    // num_samples > num_threads. (MPPI ignores this: it weights all samples.)
    bool prune_dominated = true;

    // Coarse planning model: physics stepping dominates rollout cost, so the
    // rollouts can use a cheaper model than the real one. When plan_timestep>0,
    // rollouts run on a clone of the model with this larger timestep (and,
    // optionally, cheaper solver iterations). Pair it with a proportionally
    // smaller sim_substeps so one control step still advances the same sim time
    // (e.g. real dt=0.002 x10 substeps == plan dt=0.004 x5 substeps = 0.02s),
    // which roughly halves rollout cost. 0 = use the real model's setting.
    double plan_timestep = 0.0;
    int plan_iterations = 0;
    int plan_ls_iterations = 0;
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

    // Overwrite the trailing phase_dims entries of control with the periodic
    // interpolation of knots at the gait phase corresponding to sim time.
    void InterpPhaseDims(const float* knots, double time, float* control) const;

    mjModel* model_;  // model used for rollouts (coarse plan clone if configured)
    mjModel* plan_model_ = nullptr;  // owned coarse planning clone, else null
    std::shared_ptr<core::Task> task_;
    std::shared_ptr<core::Policy> policy_;
    OptimizerConfig config_;

    // Rollouts i whose cost cannot rank in the top prune_rank_ are terminated
    // early (see OptimizerConfig::prune_dominated). Subclasses set this to
    // their elite cutoff; 0 disables pruning.
    int prune_rank_ = 0;

    std::vector<mjData*> thread_datas_;  // one per worker thread
};

}  // namespace algs
}  // namespace spc
