#include "spc/core/onnx_policy.h"
#include <stdexcept>
#include <iostream>

namespace spc {
namespace core {

ONNXPolicy::ONNXPolicy(const std::string& model_path) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ONNXPolicy");
    memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Load(model_path);
}

void ONNXPolicy::Load(const std::string& model_path) {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    // Get input names
    size_t num_inputs = session_->GetInputCount();
    input_names_.clear();
    input_names_allocated_.clear();
    for (size_t i = 0; i < num_inputs; i++) {
        input_names_allocated_.push_back(session_->GetInputNameAllocated(i, allocator));
        input_names_.push_back(input_names_allocated_.back().get());
    }

    // Get output names
    size_t num_outputs = session_->GetOutputCount();
    output_names_.clear();
    output_names_allocated_.clear();
    for (size_t i = 0; i < num_outputs; i++) {
        output_names_allocated_.push_back(session_->GetOutputNameAllocated(i, allocator));
        output_names_.push_back(output_names_allocated_.back().get());
    }
}

void ONNXPolicy::ComputeAction(const float* obs, int obs_dim, float* action_out, int action_dim) const {
    if (!session_) {
        throw std::runtime_error("ONNXPolicy: Model not loaded.");
    }

    int64_t input_shape[2] = {1, obs_dim}; // batch_size=1
    
    // Const cast is required because Ort::Value::CreateTensor takes non-const pointer, 
    // but we only use it as an input tensor, so it won't be modified.
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float*>(obs), obs_dim, input_shape, 2);

    int64_t output_shape[2] = {1, action_dim}; // batch_size=1
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, action_out, action_dim, output_shape, 2);

    // Run inference with pre-allocated tensor
    Ort::RunOptions run_options{nullptr};
    session_->Run(
        run_options, 
        input_names_.data(), &input_tensor, 1, 
        output_names_.data(), &output_tensor, 1
    );
}

} // namespace core
} // namespace spc
