#pragma once

#include <random>
#include <vector>

#include "spc/algs/optimizer.h"

namespace spc {
namespace algs {

/**
 * @brief Configuration for the Cross-Entropy Method (CEM) optimizer.
 */
struct CEMConfig : public OptimizerConfig {
    int num_elites = 12;
    float sigma_init = 0.3f;
    float sigma_min = 0.05f;
    float explore_fraction = 0.0f;

    // Optional per-dimension overrides (size = control_dim). When empty, the
    // scalar sigma_init is used and samples are unbounded. Bounds are applied
    // at sampling time so the elite distribution stays inside the feasible box.
    std::vector<float> sigma_init_per_dim;
    std::vector<float> u_min;
    std::vector<float> u_max;
};

/**
 * @brief Cross-Entropy Method (CEM) for sampling-based predictive control.
 */
class CEM : public Optimizer {
public:
    CEM(mjModel* model, std::shared_ptr<core::Task> task, std::shared_ptr<core::Policy> policy,
        const CEMConfig& config);

    ~CEM() override = default;

protected:
    void SampleKnots(std::vector<float>& samples) override;
    void UpdateDistribution(const std::vector<float>& samples, const std::vector<double>& costs) override;
    void GetBestAction(const mjData* current_state, float* best_action_out) override;

private:
    CEMConfig cem_config_;

    std::vector<float> mean_;
    std::vector<float> stddev_;
    std::vector<std::mt19937> rngs_;

    // Expanded per-dimension arrays (size = control_dim)
    std::vector<float> sigma_init_dim_;
    std::vector<float> u_min_dim_;
    std::vector<float> u_max_dim_;
};

}  // namespace algs
}  // namespace spc
