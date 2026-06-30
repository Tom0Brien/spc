#pragma once

#include <string>
#include <vector>

#include "spc/core/policy.h"

namespace spc {
namespace core {

/**
 * @brief Hand-rolled MLP policy.
 *
 * Replicates the exported actor network (input normalization, fully-connected
 * layers with SiLU/swish activations, and an output slice) without the
 * ONNX Runtime dispatch overhead. ComputeAction is allocation-free and uses
 * only const, shared weights, so it is safe to call concurrently from the
 * OpenMP rollout threads.
 *
 * Weights are loaded from a self-describing binary produced from the ONNX file.
 */
class MLPPolicy : public Policy {
public:
    MLPPolicy(const std::string& model_path);
    ~MLPPolicy() override = default;

    void Load(const std::string& model_path) override;

    void ComputeAction(const float* obs, int obs_dim, float* action, int action_dim) const override;

private:
    struct Layer {
        int in_dim;
        int out_dim;
        std::vector<float> weight;  // (in_dim, out_dim) row-major
        std::vector<float> bias;    // (out_dim)
    };

    int obs_dim_ = 0;
    int out_dim_ = 0;  // action size after slice
    std::vector<float> mean_;
    std::vector<float> inv_std_;  // 1 / (std + eps), precomputed
    std::vector<Layer> layers_;
};

}  // namespace core
}  // namespace spc
