#pragma once

#include <vector>
#include <string>

namespace spc {
namespace core {

/**
 * @brief Abstract base class for Neural Network Policies.
 *
 * This defines the interface for evaluating the base RL policy during C++ rollouts.
 * Because rollouts are parallelized using OpenMP, the ComputeAction method
 * must be thread-safe for concurrent execution.
 */
class Policy {
public:
    virtual ~Policy() = default;

    /**
     * @brief Load the policy weights or model definition from disk.
     * @param model_path Path to the exported model (e.g., .onnx file).
     */
    virtual void Load(const std::string& model_path) = 0;

    /**
     * @brief Compute the base control action given an observation.
     *
     * @param obs Pointer to the flattened observation array.
     * @param obs_dim Number of elements in the observation.
     * @param action Pointer to the pre-allocated action array (output).
     * @param action_dim Number of elements in the action.
     */
    virtual void ComputeAction(const float* obs, int obs_dim, float* action, int action_dim) const = 0;
};

} // namespace core
} // namespace spc
