#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "spc/core/mlp_policy.h"
#include "spc/core/onnx_policy.h"

// POLICIES_DIR is passed as a macro via CMake so the test is independent of the
// working directory it runs from.
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define POLICIES_DIR_STR TOSTRING(POLICIES_DIR)

using spc::core::MLPPolicy;
using spc::core::ONNXPolicy;

TEST(MLPPolicyTest, BadPathThrows) {
    EXPECT_THROW(MLPPolicy{"/nonexistent/path/does_not_exist.mlp"}, std::runtime_error);
}

TEST(MLPPolicyTest, BadMagicThrows) {
    // A file that exists but does not start with the SPCMLP01 magic must be
    // rejected rather than silently misparsed.
    const std::string path = "/tmp/spc_bad_magic.mlp";
    FILE* fp = std::fopen(path.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    const char garbage[16] = "NOTAMLPFILE!!!";
    std::fwrite(garbage, 1, sizeof(garbage), fp);
    std::fclose(fp);

    EXPECT_THROW(MLPPolicy{path}, std::runtime_error);
    std::remove(path.c_str());
}

// The .mlp format is a hand-rolled reimplementation of the exported .onnx
// actor. For any input the two must agree; this guards the fast inference path
// against silently drifting from the reference.
TEST(MLPPolicyTest, MatchesOnnxReference) {
    const std::string mlp_path = std::string(POLICIES_DIR_STR) + "/franka_push.mlp";
    const std::string onnx_path = std::string(POLICIES_DIR_STR) + "/franka_push.onnx";

    MLPPolicy mlp(mlp_path);
    ONNXPolicy onnx(onnx_path);

    const int obs_dim = 48;
    const int action_dim = 7;

    std::vector<float> obs(obs_dim);
    std::vector<float> action_mlp(action_dim, 0.0f);
    std::vector<float> action_onnx(action_dim, 0.0f);

    // A few deterministic observation patterns spanning positive/negative and
    // varying magnitudes, so the comparison is not limited to a single point.
    for (int trial = 0; trial < 3; ++trial) {
        for (int i = 0; i < obs_dim; ++i) {
            obs[i] = std::sin(0.13f * i + 0.7f * trial) * (0.5f + 0.5f * trial);
        }

        mlp.ComputeAction(obs.data(), obs_dim, action_mlp.data(), action_dim);
        onnx.ComputeAction(obs.data(), obs_dim, action_onnx.data(), action_dim);

        for (int i = 0; i < action_dim; ++i) {
            EXPECT_NEAR(action_mlp[i], action_onnx[i], 1e-3f)
                << "Mismatch at trial " << trial << ", action index " << i;
        }
    }
}

// ComputeAction must be deterministic: identical inputs yield identical outputs.
TEST(MLPPolicyTest, Deterministic) {
    const std::string mlp_path = std::string(POLICIES_DIR_STR) + "/franka_push.mlp";
    MLPPolicy mlp(mlp_path);

    const int obs_dim = 48;
    const int action_dim = 7;
    std::vector<float> obs(obs_dim);
    for (int i = 0; i < obs_dim; ++i)
        obs[i] = 0.01f * static_cast<float>(i);

    std::vector<float> a1(action_dim, 0.0f);
    std::vector<float> a2(action_dim, 0.0f);
    mlp.ComputeAction(obs.data(), obs_dim, a1.data(), action_dim);
    mlp.ComputeAction(obs.data(), obs_dim, a2.data(), action_dim);

    for (int i = 0; i < action_dim; ++i)
        EXPECT_FLOAT_EQ(a1[i], a2[i]);
}
