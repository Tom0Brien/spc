#include "spc/core/mlp_policy.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace spc {
namespace core {

namespace {
template <typename T>
T Read(std::ifstream& f) {
    T v;
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    return v;
}

void ReadFloats(std::ifstream& f, std::vector<float>& out, int n) {
    out.resize(n);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n) * sizeof(float));
}
}  // namespace

MLPPolicy::MLPPolicy(const std::string& model_path) { Load(model_path); }

void MLPPolicy::Load(const std::string& model_path) {
    std::ifstream f(model_path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("MLPPolicy: failed to open " + model_path);
    }

    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "SPCMLP01", 8) != 0) {
        throw std::runtime_error("MLPPolicy: bad magic in " + model_path);
    }

    obs_dim_ = Read<int32_t>(f);
    out_dim_ = Read<int32_t>(f);
    int n_layers = Read<int32_t>(f);

    std::vector<int> dims(n_layers + 1);
    for (int i = 0; i <= n_layers; ++i)
        dims[i] = Read<int32_t>(f);

    std::vector<float> std_vec;
    float eps;
    ReadFloats(f, mean_, obs_dim_);
    ReadFloats(f, std_vec, obs_dim_);
    f.read(reinterpret_cast<char*>(&eps), sizeof(float));

    inv_std_.resize(obs_dim_);
    for (int i = 0; i < obs_dim_; ++i)
        inv_std_[i] = 1.0f / (std_vec[i] + eps);

    layers_.resize(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        Layer& layer = layers_[l];
        layer.in_dim = dims[l];
        layer.out_dim = dims[l + 1];
        ReadFloats(f, layer.weight, layer.in_dim * layer.out_dim);
        ReadFloats(f, layer.bias, layer.out_dim);
    }

    if (!f) {
        throw std::runtime_error("MLPPolicy: truncated weights file " + model_path);
    }
}

void MLPPolicy::ComputeAction(const float* obs, int obs_dim, float* action, int action_dim) const {
    // Scratch buffers sized to the widest layers seen in practice; the network
    // is small so fixed stack storage avoids per-call allocation.
    constexpr int kMaxWidth = 1024;
    float buf_a[kMaxWidth];
    float buf_b[kMaxWidth];

    // Input normalization: x0 = (obs - mean) / (std + eps)
    for (int i = 0; i < obs_dim_; ++i)
        buf_a[i] = (obs[i] - mean_[i]) * inv_std_[i];

    float* cur = buf_a;
    float* nxt = buf_b;

    for (size_t l = 0; l < layers_.size(); ++l) {
        const Layer& layer = layers_[l];
        const bool last = (l + 1 == layers_.size());
        const int out_dim = layer.out_dim;
        const float* w = layer.weight.data();

        // Output-stationary GEMV: nxt[:] = bias + sum_i cur[i] * W[i, :].
        // The inner loop over outputs is a contiguous, dependency-free FMA that
        // the compiler vectorizes at full throughput.
        for (int j = 0; j < out_dim; ++j)
            nxt[j] = layer.bias[j];
        for (int i = 0; i < layer.in_dim; ++i) {
            const float xi = cur[i];
            const float* wi = w + static_cast<size_t>(i) * out_dim;
#pragma omp simd
            for (int j = 0; j < out_dim; ++j)
                nxt[j] += xi * wi[j];
        }

        if (!last) {
            // SiLU/swish: x * sigmoid(x)
            for (int j = 0; j < out_dim; ++j)
                nxt[j] = nxt[j] / (1.0f + std::exp(-nxt[j]));
        }
        std::swap(cur, nxt);
    }

    int n = action_dim < out_dim_ ? action_dim : out_dim_;
    for (int i = 0; i < n; ++i)
        action[i] = cur[i];
}

}  // namespace core
}  // namespace spc
