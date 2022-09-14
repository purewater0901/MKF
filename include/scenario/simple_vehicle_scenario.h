#ifndef UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_SCENARIO_H
#define UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_SCENARIO_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <Eigen/Eigen>

#include "model/simple_vehicle_model.h"
#include "distribution/normal_distribution.h"
#include "distribution/uniform_distribution.h"
#include "distribution/exponential_distribution.h"
#include "utilities.h"

using namespace SimpleVehicle;

struct SimpleVehicleGaussianScenario
{
    SimpleVehicleGaussianScenario() : filename_("/simple_vehicle_gaussian.csv")
    {
        // Position normal distribution
        const double x_cov = 0.01*0.01;
        const double y_cov = 0.01*0.01;
        const double yaw_cov = 0.01*0.01;

        // Normal Distribution
        ini_cov_ << x_cov, 0.0, 0.0,
                    0.0, y_cov, 0.0,
                    0.0, 0.0, yaw_cov;

        // Observation Noise
        const double mean_wr = 1.0;
        const double cov_wr = std::pow(0.09, 2);
        const double mean_wa = 0.0;
        const double cov_wa = std::pow(M_PI/120.0, 2);
        observation_noise_map_ = {
                {OBSERVATION_NOISE::IDX::WR, std::make_shared<NormalDistribution>(mean_wr, cov_wr)},
                {OBSERVATION_NOISE::IDX::WA, std::make_shared<NormalDistribution>(mean_wa, cov_wa)}};
    }

    const std::string filename_{"/simple_vehicle_gaussian.csv"};

    // Initial Distribution
    Eigen::Matrix3d ini_cov_;

    // Noise
    std::map<int, std::shared_ptr<BaseDistribution>> observation_noise_map_;
};

struct SimpleVehicleNonGaussianScenario
{
    SimpleVehicleNonGaussianScenario() : filename_("/simple_vehicle_non_gaussian.csv")
    {
        // Position normal distribution
        const double x_cov = 0.01*0.01;
        const double y_cov = 0.01*0.01;
        const double yaw_cov = 0.01*0.01;

        // Normal Distribution
        ini_cov_ << x_cov, 0.0, 0.0,
                    0.0, y_cov, 0.0,
                    0.0, 0.0, yaw_cov;

        // Observation Noise
        const double lambda_wr = 1.0;
        const double lower_bearing = -M_PI/12;
        const double upper_bearing = M_PI/12;
        observation_noise_map_ = {
                {OBSERVATION_NOISE::IDX::WR , std::make_shared<ExponentialDistribution>(lambda_wr)},
                {OBSERVATION_NOISE::IDX::WA , std::make_shared<UniformDistribution>(lower_bearing, upper_bearing)}};

        wr_dist_ = std::exponential_distribution<double>(lambda_wr);
        wa_dist_ = std::uniform_real_distribution<double>(lower_bearing, upper_bearing);
    }

    const std::string filename_{"/simple_vehicle_non_gaussian.csv"};

    // Initial Distribution
    Eigen::Matrix3d ini_cov_;

    // Noise
    std::map<int, std::shared_ptr<BaseDistribution>> observation_noise_map_;

    //std::uniform_real_distribution<double> wr_dist_;
    std::exponential_distribution<double> wr_dist_;
    std::uniform_real_distribution<double> wa_dist_;
};

#endif //UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_SCENARIO_H
