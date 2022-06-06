#include <iostream>
#include <random>
#include <vector>
#include <Eigen/Eigen>
#include <map>
#include <memory>
#include <gtest/gtest.h>

#include "model/mobile_robot_model.h"
#include "distribution/uniform_distribution.h"
#include "distribution/normal_distribution.h"
#include "distribution/four_dimensional_normal_distribution.h"

using namespace MobileRobot;

TEST(MobileRobotModel, getObservationMoments)
{
    const double epsilon = 0.01;

    MobileRobotModel model;

    // Normal Distribution
    {
        const Eigen::Vector4d mean = {0.4, 0.4, 3.0, 0.8};
        Eigen::Matrix4d cov;
        cov << 0.012, -0.0015,  0.0014, -0.004,
                -0.0015, 0.011991, 0.0014,  0.0042,
                0.0014,  0.0014,     0.02,    0.0,
                -0.004,  0.0042,      0.0,   0.011;

        FourDimensionalNormalDistribution dist(mean, cov);
        auto wx_dist = NormalDistribution(0.0, 1.0);
        auto wy_dist = NormalDistribution(0.0, 1.0);
        auto wv_dist = NormalDistribution(0.0, 0.3*0.3);
        auto wyaw_dist = NormalDistribution(0.0, M_PI*M_PI/100);

        MobileRobotModel::measurementInputStateMoments input_moments;
        input_moments.xPow1 = dist.calc_moment(STATE::IDX::X, 1);
        input_moments.yPow1 = dist.calc_moment(STATE::IDX::Y, 1);
        input_moments.vPow1 = dist.calc_moment(STATE::IDX::V, 1);
        input_moments.cyawPow1 = dist.calc_cos_moment(STATE::IDX::YAW, 1);
        input_moments.syawPow1 = dist.calc_sin_moment(STATE::IDX::YAW, 1);

        input_moments.xPow2 = dist.calc_moment(STATE::IDX::X, 2);
        input_moments.yPow2 = dist.calc_moment(STATE::IDX::Y, 2);
        input_moments.vPow2 = dist.calc_moment(STATE::IDX::V, 2);
        input_moments.cyawPow2 = dist.calc_cos_moment(STATE::IDX::YAW, 2);
        input_moments.syawPow2 = dist.calc_sin_moment(STATE::IDX::YAW, 2);
        input_moments.xPow1_cyawPow1 = dist.calc_x_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
        input_moments.xPow1_syawPow1 = dist.calc_x_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
        input_moments.yPow1_cyawPow1 = dist.calc_x_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
        input_moments.yPow1_syawPow1 = dist.calc_x_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
        input_moments.vPow1_cyawPow1 = dist.calc_x_cos_z_moment(STATE::IDX::V, STATE::IDX::YAW);
        input_moments.vPow1_syawPow1 = dist.calc_x_sin_z_moment(STATE::IDX::V, STATE::IDX::YAW);
        input_moments.xPow1_yPow1 = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::Y);
        input_moments.xPow1_vPow1 = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::V);
        input_moments.yPow1_vPow1 = dist.calc_cross_second_moment(STATE::IDX::Y, STATE::IDX::V);
        input_moments.cyawPow1_syawPow1 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 1, 1);

        input_moments.vPow2_cyawPow1 = dist.calc_xx_cos_z_moment(STATE::IDX::V, STATE::IDX::YAW);
        input_moments.vPow2_cyawPow1 = dist.calc_xx_sin_z_moment(STATE::IDX::V, STATE::IDX::YAW);
        input_moments.vPow1_cyawPow2 = dist.calc_x_cos_y_cos_y_moment(STATE::IDX::V, STATE::IDX::YAW);
        input_moments.vPow1_syawPow2 = dist.calc_x_sin_y_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW);
        input_moments.xPow1_vPow1_syawPow1 = dist.calc_xy_sin_z_moment(STATE::IDX::X, STATE::IDX::V, STATE::IDX::YAW);
        input_moments.xPow1_vPow1_cyawPow1 = dist.calc_xy_cos_z_moment(STATE::IDX::X, STATE::IDX::V, STATE::IDX::YAW);
        input_moments.yPow1_vPow1_syawPow1 = dist.calc_xy_sin_z_moment(STATE::IDX::Y, STATE::IDX::V, STATE::IDX::YAW);
        input_moments.yPow1_vPow1_cyawPow1 = dist.calc_xy_cos_z_moment(STATE::IDX::Y, STATE::IDX::V, STATE::IDX::YAW);
        input_moments.vPow1_cyawPow1_syawPow1 = dist.calc_x_cos_y_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW);

        input_moments.vPow2_cyawPow2 = dist.calc_xx_cos_y_cos_y_moment(STATE::IDX::V, STATE::IDX::YAW);
        input_moments.vPow2_syawPow2 = dist.calc_xx_sin_y_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW);
        input_moments.vPow2_cyawPow1_syawPow1 = dist.calc_xx_cos_y_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW);

        MobileRobotModel::ObservationNoiseMoments observation_noise;
        observation_noise.wxPow1 = wx_dist.calc_moment(1);
        observation_noise.wyPow1 = wy_dist.calc_moment(1);
        observation_noise.wvPow1 = wv_dist.calc_moment(1);
        observation_noise.cwyawPow1 = wyaw_dist.calc_cos_moment(1);
        observation_noise.swyawPow1 = wyaw_dist.calc_sin_moment(1);
        observation_noise.wxPow2 = wx_dist.calc_moment(2);
        observation_noise.wyPow2 = wy_dist.calc_moment(2);
        observation_noise.wvPow2 = wv_dist.calc_moment(2);
        observation_noise.cwyawPow2 = wyaw_dist.calc_cos_moment(2);
        observation_noise.swyawPow2 = wyaw_dist.calc_sin_moment(2);
        observation_noise.cwyawPow1_swyawPow1 = wyaw_dist.calc_cos_sin_moment(1, 1);

        const auto measurement_moments = model.getObservationMoments(input_moments, observation_noise);

        EXPECT_NEAR(measurement_moments.xPow1, 0.4, epsilon);
        EXPECT_NEAR(measurement_moments.yPow1, 0.4, epsilon);
        EXPECT_NEAR(measurement_moments.vcPow1, 1.9784823133548737, epsilon);
        EXPECT_NEAR(measurement_moments.xPow2, 9.19077369535576, epsilon);
        EXPECT_NEAR(measurement_moments.yPow2, 9.1906030645321, epsilon);
        EXPECT_NEAR(measurement_moments.vcPow2, 4.447890181853698, epsilon);
        EXPECT_NEAR(measurement_moments.xPow1_yPow1, 0.15874613778556418, epsilon);
        EXPECT_NEAR(measurement_moments.xPow1_vcPow1, 0.7997687963699138, epsilon);
        EXPECT_NEAR(measurement_moments.yPow1_vcPow1, 0.784075118599483, epsilon);
    }
}