#include <gtest/gtest.h>

#include <vector>

#include "spc/utils/spline.h"

using namespace spc::utils;

TEST(SplineTest, InterpLinear_ZeroOrder) {
    int nu = 1;
    int num_knots = 1;
    std::vector<float> knots = {1.5f};
    std::vector<float> out(nu);

    InterpLinear(nu, num_knots, knots.data(), 0, 5, out.data());
    EXPECT_FLOAT_EQ(out[0], 1.5f);

    InterpLinear(nu, num_knots, knots.data(), 4, 5, out.data());
    EXPECT_FLOAT_EQ(out[0], 1.5f);
}

TEST(SplineTest, InterpLinear_MultiKnot) {
    int nu = 1;
    int num_knots = 3;
    std::vector<float> knots = {0.0f, 1.0f, 2.0f};
    std::vector<float> out(nu);
    int horizon = 5;  // step 0, 1, 2, 3, 4

    // t = 0 / 4 = 0.0 -> knot idx 0.0 -> knot[0]
    InterpLinear(nu, num_knots, knots.data(), 0, horizon, out.data());
    EXPECT_FLOAT_EQ(out[0], 0.0f);

    // t = 1 / 4 = 0.25 -> knot idx = 0.25 * 2 = 0.5 -> 50% between knot[0] and knot[1]
    InterpLinear(nu, num_knots, knots.data(), 1, horizon, out.data());
    EXPECT_FLOAT_EQ(out[0], 0.5f);

    // t = 2 / 4 = 0.5 -> knot idx = 0.5 * 2 = 1.0 -> knot[1]
    InterpLinear(nu, num_knots, knots.data(), 2, horizon, out.data());
    EXPECT_FLOAT_EQ(out[0], 1.0f);

    // t = 3 / 4 = 0.75 -> knot idx = 0.75 * 2 = 1.5 -> 50% between knot[1] and knot[2]
    InterpLinear(nu, num_knots, knots.data(), 3, horizon, out.data());
    EXPECT_FLOAT_EQ(out[0], 1.5f);

    // t = 4 / 4 = 1.0 -> knot idx = 1.0 * 2 = 2.0 -> knot[2]
    InterpLinear(nu, num_knots, knots.data(), 4, horizon, out.data());
    EXPECT_FLOAT_EQ(out[0], 2.0f);
}

TEST(SplineTest, InterpLinear_MultiDim) {
    int nu = 2;
    int num_knots = 2;
    // Knot 0: (0, 10), Knot 1: (10, 20)
    std::vector<float> knots = {0.0f, 10.0f, 10.0f, 20.0f};
    std::vector<float> out(nu);
    int horizon = 3;  // step 0, 1, 2

    // step 0 (t=0)
    InterpLinear(nu, num_knots, knots.data(), 0, horizon, out.data());
    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_FLOAT_EQ(out[1], 10.0f);

    // step 1 (t=0.5)
    InterpLinear(nu, num_knots, knots.data(), 1, horizon, out.data());
    EXPECT_FLOAT_EQ(out[0], 5.0f);
    EXPECT_FLOAT_EQ(out[1], 15.0f);

    // step 2 (t=1.0)
    InterpLinear(nu, num_knots, knots.data(), 2, horizon, out.data());
    EXPECT_FLOAT_EQ(out[0], 10.0f);
    EXPECT_FLOAT_EQ(out[1], 20.0f);
}

TEST(SplineTest, InterpLinear_SingleStepHorizon) {
    // total_steps <= 1 must fall back to t=0 (first knot) without dividing by zero.
    int nu = 1;
    int num_knots = 3;
    std::vector<float> knots = {7.0f, 8.0f, 9.0f};
    std::vector<float> out(nu);

    InterpLinear(nu, num_knots, knots.data(), 0, 1, out.data());
    EXPECT_FLOAT_EQ(out[0], 7.0f);

    // total_steps = 0 is also guarded.
    InterpLinear(nu, num_knots, knots.data(), 0, 0, out.data());
    EXPECT_FLOAT_EQ(out[0], 7.0f);
}

TEST(SplineTest, InterpLinearNorm_ClampsOutOfRange) {
    int nu = 1;
    int num_knots = 3;
    std::vector<float> knots = {0.0f, 1.0f, 2.0f};
    std::vector<float> out(nu);

    // t < 0 clamps to the first knot; no extrapolation below the range.
    InterpLinearNorm(nu, num_knots, knots.data(), -0.5f, out.data());
    EXPECT_FLOAT_EQ(out[0], 0.0f);

    // t > 1 clamps to the last knot; no extrapolation above the range.
    InterpLinearNorm(nu, num_knots, knots.data(), 1.5f, out.data());
    EXPECT_FLOAT_EQ(out[0], 2.0f);

    // Interior value interpolates as usual.
    InterpLinearNorm(nu, num_knots, knots.data(), 0.25f, out.data());
    EXPECT_FLOAT_EQ(out[0], 0.5f);
}

TEST(SplineTest, InterpLinearNorm_SingleKnot) {
    // A single knot is a constant regardless of t (including out-of-range t).
    int nu = 2;
    int num_knots = 1;
    std::vector<float> knots = {3.0f, -4.0f};
    std::vector<float> out(nu);

    InterpLinearNorm(nu, num_knots, knots.data(), 0.0f, out.data());
    EXPECT_FLOAT_EQ(out[0], 3.0f);
    EXPECT_FLOAT_EQ(out[1], -4.0f);

    InterpLinearNorm(nu, num_knots, knots.data(), 5.0f, out.data());
    EXPECT_FLOAT_EQ(out[0], 3.0f);
    EXPECT_FLOAT_EQ(out[1], -4.0f);
}
