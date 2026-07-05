#include "spc/tasks/t1_soccer_augmented.h"

namespace spc {
namespace tasks {

T1SoccerAugmented::T1SoccerAugmented(mjModel* model, const spc::core::TaskConfig& config)
    : T1Soccer(model, config) {  // Call base class constructor

    // Index of the first leg joint. T1 order is head(2), arms(8), waist(1),
    // left leg(6), right leg(6), so the legs start at joint 11.
    leg_joint_start_ = config.numeric_params.count("leg_joint_start")
                           ? static_cast<int>(config.numeric_params.at("leg_joint_start"))
                           : 11;

    // Number of leg joints receiving residuals (12 legs: 6 left + 6 right).
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

void T1SoccerAugmented::ApplyControl(const mjModel* model, mjData* data, const float* control) const {
    // control[0:3] = velocity command, control[3:3+leg_joint_count] = leg residuals.

    // Run the RL policy on the velocity command to write base motor targets.
    T1Navigation::ApplyControl(model, data, control);

    // Add residuals to the leg motor targets and re-clamp.
    const float* residuals = control + 3;
    for (int i = 0; i < leg_joint_count_; ++i) {
        int j = leg_joint_start_ + i;
        float motor_target = static_cast<float>(data->ctrl[j]) + residuals[i];

        if (motor_target < jnt_range_low_[j])
            motor_target = jnt_range_low_[j];
        if (motor_target > jnt_range_high_[j])
            motor_target = jnt_range_high_[j];

        data->ctrl[j] = motor_target;
    }
}

double T1SoccerAugmented::RunningCost(const mjModel* model, const mjData* data, const float* control) const {
    // Base soccer cost (ball-to-goal, positioning, orientation, height, velocity reg).
    // The base uses control[0:3] for the velocity control cost.
    double cost = T1Soccer::RunningCost(model, data, control);

    // Additional regularization on leg residuals control[3:3+leg_joint_count_].
    double residual_cost = 0.0;
    const float* residuals = control + 3;
    for (int i = 0; i < leg_joint_count_; ++i)
        residual_cost += residuals[i] * residuals[i];

    return cost + residual_weight_ * residual_cost;
}

double T1SoccerAugmented::TerminalCost(const mjModel* model, const mjData* data) const {
    float zero_ctrl[3 + 32] = {0.0f};
    return RunningCost(model, data, zero_ctrl);
}

}  // namespace tasks
}  // namespace spc

#include "spc/core/task_factory.h"
REGISTER_TASK("T1SoccerAugmented", spc::tasks::T1SoccerAugmented, T1SoccerAugmentedTask)
