#include "spc/algs/cem.h"
#include "spc/utils/spline.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <omp.h>

namespace spc {
namespace algs {

CEM::CEM(mjModel* model, 
         std::shared_ptr<core::Task> task,
         std::shared_ptr<core::Policy> policy,
         const CEMConfig& config)
    : Optimizer(model, task, policy, config), cem_config_(config) {
    
    int n_params = config.control_dim * config.num_knots;
    mean_.resize(n_params, 0.0f);
    stddev_.resize(n_params, config.sigma_init);

    int max_threads = omp_get_max_threads();
    rngs_.resize(max_threads);
    for(int i = 0; i < max_threads; ++i) {
        rngs_[i].seed(1337 + i); // TODO: pass seed from config
    }
}

void CEM::SampleKnots(std::vector<float>& samples) {
    int n_params = cem_config_.control_dim * cem_config_.num_knots;
    int num_explore = static_cast<int>(cem_config_.num_samples * cem_config_.explore_fraction);
    int num_main = cem_config_.num_samples - num_explore;
    
    #pragma omp parallel for num_threads(cem_config_.num_threads)
    for (int i = 0; i < cem_config_.num_samples; ++i) {
        int tid = omp_get_thread_num();
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        bool is_explore = (i >= num_main);
        
        for (int k = 0; k < n_params; ++k) {
            float stddev = is_explore ? cem_config_.sigma_init : stddev_[k];
            samples[i * n_params + k] = mean_[k] + stddev * dist(rngs_[tid]);
        }
    }
}

void CEM::UpdateDistribution(const std::vector<float>& samples, const std::vector<double>& costs) {
    int num_samples = cem_config_.num_samples;
    int num_elites = std::min(cem_config_.num_elites, num_samples);
    int n_params = cem_config_.control_dim * cem_config_.num_knots;

    std::vector<int> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return costs[a] < costs[b];
    });

    std::fill(mean_.begin(), mean_.end(), 0.0f);
    for (int e = 0; e < num_elites; ++e) {
        int idx = indices[e];
        const float* elite_knots = &samples[idx * n_params];
        for (int k = 0; k < n_params; ++k) {
            mean_[k] += elite_knots[k];
        }
    }
    for (float& m : mean_) m /= num_elites;

    std::fill(stddev_.begin(), stddev_.end(), 0.0f);
    for (int e = 0; e < num_elites; ++e) {
        int idx = indices[e];
        const float* elite_knots = &samples[idx * n_params];
        for (int k = 0; k < n_params; ++k) {
            float diff = elite_knots[k] - mean_[k];
            stddev_[k] += diff * diff;
        }
    }
    for (float& s : stddev_) {
        s = std::sqrt(s / num_elites);
        s = std::max(s, cem_config_.sigma_min);
    }
}

void CEM::GetBestAction(const mjData* current_state, float* best_action_out) {
    int nu = cem_config_.control_dim;
    int num_knots = cem_config_.num_knots;
    
    // Evaluate mean spline at t=0 to get the residual control
    utils::InterpLinear(nu, num_knots, mean_.data(), 0, cem_config_.plan_horizon_steps, best_action_out);
}

} // namespace algs
} // namespace spc
