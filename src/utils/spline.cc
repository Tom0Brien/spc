#include "spc/utils/spline.h"

#include <algorithm>

namespace spc {
namespace utils {

void InterpLinearNorm(int nu, int num_knots, const float* knots, float t, float* out) {
    if (num_knots == 1) {
        for (int i = 0; i < nu; ++i)
            out[i] = knots[i];
        return;
    }

    if (t < 0.0f)
        t = 0.0f;
    if (t > 1.0f)
        t = 1.0f;
    float knot_idx_f = t * (num_knots - 1);
    int idx0 = std::min(num_knots - 2, static_cast<int>(knot_idx_f));
    int idx1 = idx0 + 1;
    float alpha = knot_idx_f - idx0;

    for (int i = 0; i < nu; ++i) {
        float k0 = knots[idx0 * nu + i];
        float k1 = knots[idx1 * nu + i];
        out[i] = k0 + alpha * (k1 - k0);
    }
}

void InterpLinear(int nu, int num_knots, const float* knots, int step, int total_steps, float* out) {
    float t = (total_steps <= 1) ? 0.0f : static_cast<float>(step) / (total_steps - 1);
    InterpLinearNorm(nu, num_knots, knots, t, out);
}

}  // namespace utils
}  // namespace spc
