#pragma once

#include <mujoco/mujoco.h>
#include <vector>
#include <memory>
#include "spc/core/policy.h"

#include <unordered_map>
#include <string>

namespace spc {
namespace core {

struct TaskConfig {
    std::unordered_map<std::string, double> numeric_params;
    std::unordered_map<std::string, std::string> string_params;
};

/**
 * @brief Abstract base class for Tasks.
 *
 * Defines the reward/cost structure, constraints, and observation extraction
 * for a specific environment. Since costs are evaluated on the CPU during
 * the hot rollout loop, this implementation must be extremely fast.
 */
class Task {
public:
    virtual ~Task() = default;

    /**
     * @brief Extract the observation array from the current MuJoCo state.
     * This is passed to the RL policy.
     * 
     * @param model Pointer to the MuJoCo model.
     * @param data Pointer to the MuJoCo data (state).
     * @param obs_out Pre-allocated array to write the observation into.
     */
    virtual void GetObservation(const mjModel* model, const mjData* data, float* obs_out) const = 0;

    /**
     * @brief Compute the running cost at a specific time step.
     * 
     * @param model Pointer to the MuJoCo model.
     * @param data Pointer to the MuJoCo data (state).
     * @param control The applied control action.
     * @return double The scalar running cost.
     */
    virtual double RunningCost(const mjModel* model, const mjData* data, const float* control) const = 0;

    /**
     * @brief Compute the terminal cost at the end of a rollout.
     * 
     * @param model Pointer to the MuJoCo model.
     * @param data Pointer to the MuJoCo data (state).
     * @return double The scalar terminal cost.
     */
    virtual double TerminalCost(const mjModel* model, const mjData* data) const = 0;

    /**
     * @brief Compute constraint violations.
     * 
     * @param model Pointer to the MuJoCo model.
     * @param data Pointer to the MuJoCo data (state).
     * @param control The applied control action.
     * @return double Constraint cost. >0 indicates a violation. <=0 is safe.
     */
    virtual double ConstraintCost(const mjModel* model, const mjData* data, const float* control) const {
        return 0.0; // Default: unconstrained
    }

    /**
     * @brief Apply the control action. The task evaluates the base policy if any,
     * combines it with the residual, applies scaling, and writes to data->ctrl.
     * 
     * @param model Pointer to the MuJoCo model.
     * @param data Pointer to the MuJoCo data (state) to be modified.
     * @param residual The residual control from the optimizer.
     */
    virtual void ApplyControl(const mjModel* model, mjData* data, const float* residual) const {
        for (int i = 0; i < model->nu; ++i) {
            data->ctrl[i] = residual[i];
        }
    }

    /**
     * @brief Set the base policy for the task to evaluate during ApplyControl.
     */
    virtual void SetPolicy(std::shared_ptr<Policy> policy) {
        policy_ = policy;
    }

protected:
    std::shared_ptr<Policy> policy_ = nullptr;
};

} // namespace core
} // namespace spc
