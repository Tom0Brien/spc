#include "spc/algs/optimizer.h"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>

#include "spc/utils/spline.h"

namespace spc {
namespace algs {

Optimizer::Optimizer(mjModel* model, std::shared_ptr<core::Task> task, std::shared_ptr<core::Policy> policy,
                     const OptimizerConfig& config)
    : model_(model), task_(task), policy_(policy), config_(config) {
    if (!model) {
        throw std::invalid_argument("MuJoCo model cannot be null");
    }

    // Build a cheaper clone for rollouts if a coarse planning model is
    // configured. It shares the real model's dimensions (only opt.* differs),
    // so mjData stays layout-compatible with the real state passed to Optimize.
    if (config.plan_timestep > 0.0 || config.plan_iterations > 0 || config.plan_ls_iterations > 0) {
        plan_model_ = mj_copyModel(nullptr, model);
        if (config.plan_timestep > 0.0)
            plan_model_->opt.timestep = config.plan_timestep;
        if (config.plan_iterations > 0)
            plan_model_->opt.iterations = config.plan_iterations;
        if (config.plan_ls_iterations > 0)
            plan_model_->opt.ls_iterations = config.plan_ls_iterations;
        model_ = plan_model_;  // rollouts step the coarse model
    }

    if (task_ && policy_) {
        task_->SetPolicy(policy_);
    }

    // One mjData per worker thread (not per sample): every rollout starts by
    // copying the root state, so data can be reused across the samples a
    // thread processes, keeping the resident working set at num_threads
    // mjDatas regardless of population size.
    int num_datas = std::min(config.num_threads, config.num_samples);
    thread_datas_.resize(std::max(num_datas, 1), nullptr);
    for (mjData*& d : thread_datas_) {
        d = mj_makeData(model_);
    }
}

Optimizer::~Optimizer() {
    for (mjData* d : thread_datas_) {
        if (d)
            mj_deleteData(d);
    }
    if (plan_model_)
        mj_deleteModel(plan_model_);
}

void Optimizer::Optimize(const mjData* current_state, float* best_action_out) {
    int num_samples = config_.num_samples;
    int n_params = config_.control_dim * config_.num_knots;

    std::vector<float> samples(num_samples * n_params);
    std::vector<double> costs(num_samples);

    PrepareForReplan();

    for (int iter = 0; iter < config_.num_iterations; ++iter) {
        SampleKnots(samples);
        EvaluateRollouts(current_state, samples, costs);
        UpdateDistribution(samples, costs);
    }

    GetBestAction(current_state, best_action_out);
}

void Optimizer::EvaluateRollouts(const mjData* current_state, const std::vector<float>& samples,
                                 std::vector<double>& costs) {
    int nu = config_.control_dim;
    int num_knots = config_.num_knots;
    int horizon_steps = config_.plan_horizon_steps;
    int n_params = nu * num_knots;
    int num_samples = config_.num_samples;

    // Dominated-rollout pruning: once prune_rank_ rollouts have completed, any
    // rollout whose accumulated cost exceeds the prune_rank_-th best completed
    // cost can never rank in the top prune_rank_, so it stops stepping (costs
    // are nonnegative, see OptimizerConfig::prune_dominated). The threshold
    // only tightens as more rollouts finish, so pruning never discards a true
    // elite and the distribution update is unchanged.
    const bool prune = config_.prune_dominated && prune_rank_ > 0 && prune_rank_ < num_samples;
    const double kInf = std::numeric_limits<double>::infinity();
    std::atomic<double> prune_threshold(kInf);
    std::mutex completed_mutex;
    std::vector<double> completed_costs;
    if (prune)
        completed_costs.reserve(num_samples);

// Rollout costs are heterogeneous (contact bursts, early divergence), so a
// static schedule leaves threads idle behind the slowest chunk; dynamic
// scheduling balances the makespan (same reason mujoco's rollout.cc chunks
// work through a thread pool).
#pragma omp parallel for schedule(dynamic, 1) num_threads(static_cast<int>(thread_datas_.size()))
    for (int i = 0; i < num_samples; ++i) {
        mjData* d = thread_datas_[omp_get_thread_num()];
        mj_copyData(d, model_, current_state);

        double total_cost = 0.0;
        bool pruned = false;
        const float* sample_knots = &samples[i * n_params];
        float current_control[128];

        for (int step = 0; step < horizon_steps; ++step) {
            // Interpolate residual control from knots
            utils::InterpLinear(nu, num_knots, sample_knots, step, horizon_steps, current_control);
            InterpPhaseDims(sample_knots, d->time, current_control);

            // Set control in MuJoCo data by delegating to Task
            if (task_) {
                task_->ApplyControl(model_, d, current_control);
            } else {
                for (int j = 0; j < nu; ++j) {
                    d->ctrl[j] = current_control[j];
                }
            }

            // Step physics engine
            for (int sub = 0; sub < config_.sim_substeps; ++sub) {
                mj_step(model_, d);
            }

            // Accumulate running cost
            if (task_) {
                total_cost += task_->RunningCost(model_, d, current_control);
            }

            if (prune && total_cost > prune_threshold.load(std::memory_order_relaxed)) {
                pruned = true;
                break;
            }
        }

        if (pruned) {
            costs[i] = kInf;
            continue;
        }

        // Terminal cost
        if (task_) {
            total_cost += task_->TerminalCost(model_, d);
        }
        costs[i] = total_cost;

        if (prune) {
            std::lock_guard<std::mutex> lock(completed_mutex);
            completed_costs.push_back(total_cost);
            if (static_cast<int>(completed_costs.size()) >= prune_rank_) {
                std::nth_element(completed_costs.begin(), completed_costs.begin() + prune_rank_ - 1,
                                 completed_costs.end());
                prune_threshold.store(completed_costs[prune_rank_ - 1], std::memory_order_relaxed);
            }
        }
    }

}

void Optimizer::InterpPhaseDims(const float* knots, double time, float* control) const {
    int phase_dims = config_.phase_dims;
    if (phase_dims <= 0)
        return;

    int nu = config_.control_dim;
    int num_knots = config_.num_knots;

    // Knots span one gait cycle and wrap around (periodic spline).
    double cycle = time * config_.phase_freq;
    cycle -= std::floor(cycle);  // [0, 1)
    float knot_pos = static_cast<float>(cycle) * num_knots;
    int idx0 = std::min(num_knots - 1, static_cast<int>(knot_pos));
    int idx1 = (idx0 + 1) % num_knots;
    float alpha = knot_pos - idx0;

    for (int dim = nu - phase_dims; dim < nu; ++dim) {
        float k0 = knots[idx0 * nu + dim];
        float k1 = knots[idx1 * nu + dim];
        control[dim] = k0 + alpha * (k1 - k0);
    }
}

}  // namespace algs
}  // namespace spc
