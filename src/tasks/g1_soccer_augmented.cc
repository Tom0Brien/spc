#include "spc/tasks/g1_soccer_augmented.h"

namespace spc {
namespace tasks {

G1SoccerAugmented::G1SoccerAugmented(mjModel* model, const spc::core::TaskConfig& config)
    : G1Soccer(model, config) {  // Call base class constructor

    // Number of leg joints receiving residuals (first 12 of 29 robot joints).
    leg_joint_count_ = config.numeric_params.count("leg_joint_count")
                           ? static_cast<int>(config.numeric_params.at("leg_joint_count"))
                           : 12;

    // Regularization weight on leg residuals (prevent instability).
    residual_weight_ =
        config.numeric_params.count("residual_weight") ? config.numeric_params.at("residual_weight") : 0.05;

    // Residual kicks need close contact that the behind-ball penalty fights
    // against, so it is disabled by default for the augmented task.
    behind_weight_ = config.numeric_params.count("behind_weight") ? config.numeric_params.at("behind_weight") : 0.0;
}

void G1SoccerAugmented::ApplyControl(const mjModel* model, mjData* data, const float* control) const {
    // control[0:3] = velocity command, control[3:15] = leg residuals.

    // Run the RL policy on the velocity command to write base motor targets.
    G1Navigation::ApplyControl(model, data, control);

    // Add residuals to the first leg_joint_count_ motor targets and re-clamp.
    const float* residuals = control + 3;
    for (int i = 0; i < leg_joint_count_; ++i) {
        float motor_target = static_cast<float>(data->ctrl[i]) + residuals[i];

        if (motor_target < jnt_range_low_[i])
            motor_target = jnt_range_low_[i];
        if (motor_target > jnt_range_high_[i])
            motor_target = jnt_range_high_[i];

        data->ctrl[i] = motor_target;
    }
}

double G1SoccerAugmented::RunningCost(const mjModel* model, const mjData* data, const float* control) const {
    // Base soccer cost (ball-to-goal, positioning, orientation, height, velocity reg).
    // The base uses control[0:3] for the velocity control cost.
    double cost = G1Soccer::RunningCost(model, data, control);

    // Additional regularization on leg residuals control[3:3+leg_joint_count_].
    double residual_cost = 0.0;
    const float* residuals = control + 3;
    for (int i = 0; i < leg_joint_count_; ++i)
        residual_cost += residuals[i] * residuals[i];

    return cost + residual_weight_ * residual_cost;
}

double G1SoccerAugmented::TerminalCost(const mjModel* model, const mjData* data) const {
    float zero_ctrl[3 + 32] = {0.0f};
    return RunningCost(model, data, zero_ctrl);
}

}  // namespace tasks
}  // namespace spc

#include "spc/core/task_factory.h"
REGISTER_TASK("G1SoccerAugmented", spc::tasks::G1SoccerAugmented, G1SoccerAugmentedTask)
