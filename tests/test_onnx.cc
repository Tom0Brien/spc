#include <gtest/gtest.h>

#include <vector>

#include "spc/core/onnx_policy.h"

TEST(ONNXPolicyTest, LoadAndComputeAction) {
    // Note: Assuming the tests are run from the build directory
    std::string model_path = "../policies/franka_push.onnx";

    // Create the policy
    spc::core::ONNXPolicy policy(model_path);

    // Franka Push observation size is 48
    int obs_dim = 48;
    // Franka Push action size is 7
    int action_dim = 7;

    std::vector<float> obs(obs_dim, 0.0f);
    std::vector<float> action(action_dim, 0.0f);

    // Give some random values to observation
    for (int i = 0; i < obs_dim; ++i) {
        obs[i] = static_cast<float>(i) * 0.01f;
    }

    // Compute action
    policy.ComputeAction(obs.data(), obs_dim, action.data(), action_dim);

    // Check that actions are reasonable (not all zeros if inputs are not zeros)
    float sum_abs_action = 0.0f;
    for (int i = 0; i < action_dim; ++i) {
        sum_abs_action += std::abs(action[i]);
    }

    // Because the neural network has biases, the output should definitely not be zero
    EXPECT_GT(sum_abs_action, 0.0f);
}
