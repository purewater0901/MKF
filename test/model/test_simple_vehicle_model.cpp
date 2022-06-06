#include <iostream>
#include <random>
#include <vector>
#include <Eigen/Eigen>
#include <map>
#include <memory>
#include <gtest/gtest.h>

#include "model/simple_vehicle_model.h"
#include "distribution/uniform_distribution.h"
#include "distribution/normal_distribution.h"
#include "distribution/four_dimensional_normal_distribution.h"

using namespace SimpleVehicle;

TEST(SimpleVehicleModel, getObservationMoments_indpendent)
{
    const double landmark_x = 10.0;
    const double landmark_y = 20.0;
    const double epsilon = 0.001;
    SimpleVehicleModel model;

    const double x_mean = 3.5;
    const double y_mean = 5.5;
    const double yaw_mean = M_PI/6.0;
    const double x_cov = 0.5;
    const double y_cov = 0.5;
    const double yaw_cov = 0.05;
    const Eigen::Vector3d mean = {x_mean, y_mean, yaw_mean};
    Eigen::Matrix3d cov;
    cov <<x_cov, 0.0, 0.0,
          0.0, y_cov, 0.0,
          0.0, 0.0, yaw_cov;

    ThreeDimensionalNormalDistribution dist(mean, cov);
    SimpleVehicleModel::ReducedStateMoments reduced_moments;
    reduced_moments.cPow1= dist.calc_cos_moment(STATE::IDX::YAW, 1);
    reduced_moments.sPow1= dist.calc_sin_moment(STATE::IDX::YAW, 1);

    reduced_moments.cPow2= dist.calc_cos_moment(STATE::IDX::YAW, 2);
    reduced_moments.sPow2= dist.calc_sin_moment(STATE::IDX::YAW, 2);
    reduced_moments.xPow1_cPow1 = dist.calc_x_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow1_cPow1 = dist.calc_x_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    reduced_moments.xPow1_sPow1 = dist.calc_x_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow1_sPow1 = dist.calc_x_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    reduced_moments.cPow1_sPow1 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 1, 1);

    reduced_moments.xPow1_cPow2 = dist.calc_x_cos_z_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow1_cPow2 = dist.calc_x_cos_z_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    reduced_moments.xPow1_sPow2 = dist.calc_x_sin_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow1_sPow2 = dist.calc_x_sin_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    reduced_moments.xPow1_cPow1_sPow1 = dist.calc_x_cos_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow1_cPow1_sPow1 = dist.calc_x_cos_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);

    reduced_moments.xPow2_cPow2 = dist.calc_xx_cos_z_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow2_cPow2 = dist.calc_xx_cos_z_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    reduced_moments.xPow2_sPow2 = dist.calc_xx_sin_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow2_sPow2 = dist.calc_xx_sin_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    reduced_moments.xPow2_cPow1_sPow1 = dist.calc_xx_cos_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow2_cPow1_sPow1 = dist.calc_xx_cos_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    reduced_moments.xPow1_yPow1_cPow2 = dist.calc_xy_cos_z_cos_z_moment();
    reduced_moments.xPow1_yPow1_sPow2 = dist.calc_xy_sin_z_sin_z_moment();
    reduced_moments.xPow1_yPow1_cPow1_sPow1 = dist.calc_xy_cos_z_sin_z_moment();

    // Step2. Create Observation Noise
    const double wr_mean = 0.0;
    const double wr_cov = std::pow(0.1, 2);
    const double wa_mean = 0.0;
    const double wa_cov = std::pow(M_PI/60, 2);
    auto wr_dist = NormalDistribution(wr_mean, wr_cov);
    auto wa_dist = NormalDistribution(wa_mean, wa_cov);
    SimpleVehicleModel::ObservationNoiseMoments observation_noise;
    observation_noise.wrPow1 = wr_dist.calc_moment(1);
    observation_noise.wrPow2 = wr_dist.calc_moment(2);

    observation_noise.cwaPow1 = wa_dist.calc_cos_moment(1);
    observation_noise.swaPow1 = wa_dist.calc_sin_moment(1);
    observation_noise.cwaPow2 = wa_dist.calc_cos_moment(2);
    observation_noise.swaPow2 = wa_dist.calc_sin_moment(2);
    observation_noise.cwaPow1_swaPow1 = wa_dist.calc_cos_sin_moment(1, 1);

    // Step3. Get Observation Moments
    const auto observation_moments = model.getObservationMoments(reduced_moments, observation_noise, {landmark_x, landmark_y});

    // Monte Carlo
    SimpleVehicleModel::ObservationMoments montecarlo_observation_moments;
    {
        std::random_device seed_gen;
        std::default_random_engine engine(seed_gen());

        const size_t num_of_samples = 10000 * 10000;
        std::normal_distribution<double> x_dist_monte(mean(0), std::sqrt(cov(0,0)));
        std::normal_distribution<double> y_dist_monte(mean(1), std::sqrt(cov(1,1)));
        std::normal_distribution<double> yaw_dist_monte(mean(2), std::sqrt(cov(2,2)));
        std::normal_distribution<double> wr_dist_monte(wr_mean, std::sqrt(wr_cov));
        std::normal_distribution<double> wa_dist_monte(wa_mean, std::sqrt(wa_cov));


        for(size_t i=0; i<num_of_samples; ++i) {
            const auto x = x_dist_monte(engine);
            const auto y = y_dist_monte(engine);
            const auto yaw = yaw_dist_monte(engine);
            const auto wr = wr_dist_monte(engine);
            const auto wa = wa_dist_monte(engine);

            const auto y_observe = model.observe({x,y,yaw}, {wr, wa}, {landmark_x, landmark_y});

            montecarlo_observation_moments.rcosPow1 += y_observe[0];
            montecarlo_observation_moments.rsinPow1 += y_observe[1];
            montecarlo_observation_moments.rcosPow2 += y_observe[0]*y_observe[0];
            montecarlo_observation_moments.rsinPow2 += y_observe[1]*y_observe[1];
            montecarlo_observation_moments.rcosPow1_rsinPow1 += y_observe[0] * y_observe[1];
        }

        montecarlo_observation_moments.rcosPow1 /= num_of_samples;
        montecarlo_observation_moments.rsinPow1 /= num_of_samples;
        montecarlo_observation_moments.rcosPow2 /= num_of_samples;
        montecarlo_observation_moments.rsinPow2 /= num_of_samples;
        montecarlo_observation_moments.rcosPow1_rsinPow1 /= num_of_samples;

    }

    EXPECT_NEAR(observation_moments.rcosPow1, montecarlo_observation_moments.rcosPow1, epsilon);
    EXPECT_NEAR(observation_moments.rsinPow1, montecarlo_observation_moments.rsinPow1, epsilon);
    EXPECT_NEAR(observation_moments.rcosPow2, montecarlo_observation_moments.rcosPow2, epsilon);
    EXPECT_NEAR(observation_moments.rsinPow2, montecarlo_observation_moments.rsinPow2, epsilon);
    EXPECT_NEAR(observation_moments.rcosPow1_rsinPow1, montecarlo_observation_moments.rcosPow1_rsinPow1, epsilon);
}