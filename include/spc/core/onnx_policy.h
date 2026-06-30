#pragma once

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

#include "spc/core/policy.h"

namespace spc {
namespace core {

class ONNXPolicy : public Policy {
public:
    ONNXPolicy(const std::string& model_path);
    ~ONNXPolicy() override = default;

    void Load(const std::string& model_path) override;

    // Computes action from observation using the ONNX model
    // obs should be size obs_dim, action_out will be filled with size action_dim
    void ComputeAction(const float* obs, int obs_dim, float* action_out, int action_dim) const override;

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_{nullptr};

    // Model metadata
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;

    // Pre-allocated strings for names because Ort::Session::GetInputNameAllocated returns allocated strings
    // and we need to manage their lifetime during execution.
    std::vector<Ort::AllocatedStringPtr> input_names_allocated_;
    std::vector<Ort::AllocatedStringPtr> output_names_allocated_;
};

}  // namespace core
}  // namespace spc
