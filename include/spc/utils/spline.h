#pragma once

namespace spc {
namespace utils {

/**
 * @brief Linear interpolation for control splines over a fixed horizon.
 *
 * @param nu Dimension of the control vector.
 * @param num_knots Number of knots in the spline.
 * @param knots Flat array of knots (size = nu * num_knots).
 * @param step Current timestep in the horizon.
 * @param total_steps Total number of timesteps in the horizon.
 * @param out Output array to write the interpolated control action.
 */
void InterpLinear(int nu, int num_knots, const float* knots, int step, int total_steps, float* out);

/**
 * @brief Linear interpolation at a normalized position t in [0, 1] (clamped).
 */
void InterpLinearNorm(int nu, int num_knots, const float* knots, float t, float* out);

}  // namespace utils
}  // namespace spc
