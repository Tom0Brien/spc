#include "spc/algs/optimizer.h"
#include "spc/utils/spline.h"
#include <stdexcept>
#include <omp.h>
#include <cstring>

namespace spc {
namespace algs {

Optimizer::Optimizer(mjModel* model, 
                     std::shared_ptr<core::Task> task,
                     std::shared_ptr<core::Policy> policy,
                     const OptimizerConfig& config)
    : model_(model), task_(task), policy_(policy), config_(config) {
    
    if (!model) {
        throw std::invalid_argument("MuJoCo model cannot be null");
    }

    if (task_ && policy_) {
        task_->SetPolicy(policy_);
    }

    thread_datas_.resize(config.num_samples, nullptr);
    for (int i = 0; i < config.num_samples; ++i) {
        thread_datas_[i] = mj_makeData(model);
    }
}

Optimizer::~Optimizer() {
    for (mjData* d : thread_datas_) {
        if (d) mj_deleteData(d);
    }
}

void Optimizer::Optimize(const mjData* current_state, float* best_action_out) {
    int num_samples = config_.num_samples;
    int n_params = config_.control_dim * config_.num_knots;
    
    std::vector<float> samples(num_samples * n_params);
    std::vector<double> costs(num_samples);

    for (int iter = 0; iter < config_.num_iterations; ++iter) {
        SampleKnots(samples);
        EvaluateRollouts(current_state, samples, costs);
        UpdateDistribution(samples, costs);
    }

    GetBestAction(current_state, best_action_out);
}

void Optimizer::EvaluateRollouts(const mjData* current_state, const std::vector<float>& samples, std::vector<double>& costs) {
    int nu = config_.control_dim;
    int num_knots = config_.num_knots;
    int horizon_steps = config_.plan_horizon_steps;
    int n_params = nu * num_knots;
    int num_samples = config_.num_samples;

    #pragma omp parallel for num_threads(config_.num_threads)
    for (int i = 0; i < num_samples; ++i) {
        mjData* d = thread_datas_[i];
        mj_copyData(d, model_, current_state);
        
        double total_cost = 0.0;
        const float* sample_knots = &samples[i * n_params];
        float current_control[128];
        float obs[512];
        float base_action[128];
        for (int j = 0; j < nu; ++j) base_action[j] = 0.0f;

        for (int step = 0; step < horizon_steps; ++step) {
            // Interpolate residual control from knots
            utils::InterpLinear(nu, num_knots, sample_knots, step, horizon_steps, current_control);

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
        }
        
        // Terminal cost
        if (task_) {
            total_cost += task_->TerminalCost(model_, d);
        }
        costs[i] = total_cost;
    }
}

} // namespace algs
} // namespace spc
