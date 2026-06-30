#include "spc/utils/spline.h"
#include <algorithm>

namespace spc {
namespace utils {

void InterpLinear(int nu, int num_knots, const float* knots, int step, int total_steps, float* out) {
    if (num_knots == 1 || total_steps <= 1) {
        for (int i = 0; i < nu; ++i) out[i] = knots[i];
        return;
    }
    
    float t = static_cast<float>(step) / (total_steps - 1);
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

} // namespace utils
} // namespace spc
