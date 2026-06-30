#pragma once

#include <random>

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
};

}  // namespace algs
}  // namespace spc
