#include "spc/algs/cem.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "spc/utils/spline.h"

namespace spc {
namespace algs {

CEM::CEM(mjModel* model, std::shared_ptr<core::Task> task, std::shared_ptr<core::Policy> policy,
         const CEMConfig& config)
    : Optimizer(model, task, policy, config), cem_config_(config) {
    int nu = config.control_dim;
    int n_params = nu * config.num_knots;

    // Expand per-dimension sigma and bounds, falling back to scalar sigma / unbounded.
    sigma_init_dim_.resize(nu);
    u_min_dim_.resize(nu);
    u_max_dim_.resize(nu);
    for (int j = 0; j < nu; ++j) {
        sigma_init_dim_[j] = (j < static_cast<int>(config.sigma_init_per_dim.size())) ? config.sigma_init_per_dim[j]
                                                                                      : config.sigma_init;
        u_min_dim_[j] = (j < static_cast<int>(config.u_min.size())) ? config.u_min[j]
                                                                    : -std::numeric_limits<float>::infinity();
        u_max_dim_[j] =
            (j < static_cast<int>(config.u_max.size())) ? config.u_max[j] : std::numeric_limits<float>::infinity();
    }

    mean_.resize(n_params, 0.0f);
    stddev_.resize(n_params);
    for (int k = 0; k < n_params; ++k)
        stddev_[k] = sigma_init_dim_[k % nu];

    // Reserve population slots for re-injected elites: samples[0] is always
    // the mean, and at least two slots must remain for fresh exploration.
    num_kept_ = 0;
    cem_config_.elite_keep = std::min(cem_config_.elite_keep, config.num_samples - 2);
    if (cem_config_.elite_keep > 0)
        kept_elites_.resize(cem_config_.elite_keep * n_params, 0.0f);

    int max_threads = omp_get_max_threads();
    rngs_.resize(max_threads);
    for (int i = 0; i < max_threads; ++i) {
        rngs_[i].seed(config.seed + i);
    }
}

void CEM::PrepareForReplan() {
    // Receding-horizon warm start: the previous plan's first replan_shift_steps
    // control steps have been executed, so re-time the mean spline to start at
    // the current instant (holding the last knot beyond the old horizon).
    int shift = cem_config_.replan_shift_steps;
    if (shift <= 0)
        return;

    int nu = cem_config_.control_dim;
    int num_knots = cem_config_.num_knots;
    int horizon = cem_config_.plan_horizon_steps;
    if (num_knots < 2 || horizon < 2)
        return;

    std::vector<float> shifted(mean_.size());
    auto shift_knots = [&](float* knots) {
        for (int j = 0; j < num_knots; ++j) {
            float t = static_cast<float>(j) / (num_knots - 1) + static_cast<float>(shift) / (horizon - 1);
            utils::InterpLinearNorm(nu, num_knots, knots, t, &shifted[j * nu]);
        }
        std::copy(shifted.begin(), shifted.end(), knots);
    };

    shift_knots(mean_.data());
    for (int e = 0; e < num_kept_; ++e)
        shift_knots(&kept_elites_[e * nu * num_knots]);
}

void CEM::SampleKnots(std::vector<float>& samples) {
    int nu = cem_config_.control_dim;
    int n_params = nu * cem_config_.num_knots;
    int num_samples = cem_config_.num_samples;
    int num_explore = static_cast<int>(num_samples * cem_config_.explore_fraction);
    int num_main = num_samples - num_explore;

    auto clamp_dim = [&](float value, int dim) {
        if (value < u_min_dim_[dim])
            value = u_min_dim_[dim];
        if (value > u_max_dim_[dim])
            value = u_max_dim_[dim];
        return value;
    };

    // Sample 0 always carries the current mean (clamped), so the elite set
    // never regresses between replans. This matters at small population sizes.
    for (int k = 0; k < n_params; ++k)
        samples[k] = clamp_dim(mean_[k], k % nu);

    // iCEM-style elite reuse: re-inject the previous replan's best samples.
    for (int e = 0; e < num_kept_; ++e) {
        for (int k = 0; k < n_params; ++k)
            samples[(1 + e) * n_params + k] = clamp_dim(kept_elites_[e * n_params + k], k % nu);
    }

    // Remaining samples are antithetic pairs (mean +/- sigma*eps): mirrored
    // draws halve the variance of the elite estimate at small population
    // sizes. A pair's sigma class (adapted vs explore) follows its leader.
    // With noise_rho > 0, noise is AR(1)-correlated across knots (iCEM
    // colored noise), which biases exploration toward temporally smooth
    // control variations.
    bool mppi = (cem_config_.update_rule == 1);
    float rho = cem_config_.noise_rho;
    float rho_c = std::sqrt(std::max(0.0f, 1.0f - rho * rho));
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::mt19937& rng = rngs_[0];
    std::vector<float> eps(n_params);
    for (int i = 1 + num_kept_; i < num_samples; i += 2) {
        bool is_explore = (i >= num_main);
        bool has_mirror = (i + 1 < num_samples);

        for (int k = 0; k < n_params; ++k) {
            float white = dist(rng);
            eps[k] = (rho > 0.0f && k >= nu) ? rho * eps[k - nu] + rho_c * white : white;
        }

        for (int k = 0; k < n_params; ++k) {
            int dim = k % nu;
            // MPPI keeps a fixed sampling covariance; CEM adapts it via elites.
            float stddev = (mppi || is_explore) ? sigma_init_dim_[dim] : stddev_[k];
            float perturb = stddev * eps[k];
            samples[i * n_params + k] = clamp_dim(mean_[k] + perturb, dim);
            if (has_mirror)
                samples[(i + 1) * n_params + k] = clamp_dim(mean_[k] - perturb, dim);
        }
    }
}

void CEM::UpdateDistribution(const std::vector<float>& samples, const std::vector<double>& costs) {
    int num_samples = cem_config_.num_samples;
    int num_elites = std::min(cem_config_.num_elites, num_samples);
    int n_params = cem_config_.control_dim * cem_config_.num_knots;

    int sort_n = std::max(num_elites, cem_config_.elite_keep);
    std::vector<int> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + sort_n, indices.end(),
                      [&](int a, int b) { return costs[a] < costs[b]; });

    if (cem_config_.update_rule == 1) {
        // MPPI: softmax-weighted average over ALL samples. Every rollout
        // contributes, which wastes nothing at small population sizes; the
        // sampling covariance stays fixed at sigma_init.
        double cost_min = costs[indices[0]];
        std::vector<float> weights(num_samples);
        float weight_sum = 0.0f;
        for (int i = 0; i < num_samples; ++i) {
            weights[i] = std::exp(static_cast<float>(-(costs[i] - cost_min)) / cem_config_.mppi_lambda);
            weight_sum += weights[i];
        }
        for (float& w : weights)
            w /= weight_sum;

        std::fill(mean_.begin(), mean_.end(), 0.0f);
        for (int i = 0; i < num_samples; ++i) {
            const float* knots = &samples[i * n_params];
            for (int k = 0; k < n_params; ++k) {
                mean_[k] += weights[i] * knots[k];
            }
        }
    } else {
        // Rank-based elite weights (CMA-ES style): w_e ~ ln(E + 0.5) - ln(e + 1).
        // Emphasizing the best elites extracts more from small populations than a
        // uniform average over a hard elite cut.
        std::vector<float> weights(num_elites);
        float weight_sum = 0.0f;
        for (int e = 0; e < num_elites; ++e) {
            weights[e] = std::log(num_elites + 0.5f) - std::log(e + 1.0f);
            weight_sum += weights[e];
        }
        for (float& w : weights)
            w /= weight_sum;

        std::fill(mean_.begin(), mean_.end(), 0.0f);
        for (int e = 0; e < num_elites; ++e) {
            const float* elite_knots = &samples[indices[e] * n_params];
            for (int k = 0; k < n_params; ++k) {
                mean_[k] += weights[e] * elite_knots[k];
            }
        }

        std::fill(stddev_.begin(), stddev_.end(), 0.0f);
        for (int e = 0; e < num_elites; ++e) {
            const float* elite_knots = &samples[indices[e] * n_params];
            for (int k = 0; k < n_params; ++k) {
                float diff = elite_knots[k] - mean_[k];
                stddev_[k] += weights[e] * diff * diff;
            }
        }
        for (float& s : stddev_) {
            s = std::sqrt(s);
            s = std::max(s, cem_config_.sigma_min);
        }
    }

    // Remember the best samples for re-injection at the next replan.
    num_kept_ = cem_config_.elite_keep;
    for (int e = 0; e < num_kept_; ++e) {
        const float* knots = &samples[indices[e] * n_params];
        std::copy(knots, knots + n_params, &kept_elites_[e * n_params]);
    }
}

void CEM::GetBestAction(const mjData* current_state, float* best_action_out) {
    int nu = cem_config_.control_dim;
    int num_knots = cem_config_.num_knots;

    // Evaluate mean spline at t=0 to get the residual control
    utils::InterpLinear(nu, num_knots, mean_.data(), 0, cem_config_.plan_horizon_steps, best_action_out);
}

}  // namespace algs
}  // namespace spc
