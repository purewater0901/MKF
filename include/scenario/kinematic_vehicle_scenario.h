#ifndef UNCERTAINTY_PROPAGATION_KINEMATIC_VEHICLE_SCENARIO_H
#define UNCERTAINTY_PROPAGATION_KINEMATIC_VEHICLE_SCENARIO_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <Eigen/Eigen>

#include "model/kinematic_vehicle_model.h"
#include "distribution/normal_distribution.h"
#include "distribution/uniform_distribution.h"
#include "distribution/exponential_distribution.h"
#include "utilities.h"

struct KinematicVehicleGaussianScenario
{
    KinematicVehicleGaussianScenario() : filename_("/kinematic_vehicle_gaussian.csv")
    {
        // Position normal distribution
        const double x_mean = 0.0;
        const double x_cov = 0.1*0.1;
        const double y_mean = 0.0;
        const double y_cov = 0.1*0.1;
        const double v_mean = 3.0;
        const double v_cov = 0.1*0.1;
        const double yaw_mean = M_PI/4.0;
        const double yaw_cov = 0.1*0.1;

        // Normal Distribution
        NormalDistribution x0_dist(x_mean, x_cov);
        NormalDistribution y0_dist(y_mean, y_cov);
        NormalDistribution v0_dist(v_mean, v_cov);
        NormalDistribution yaw0_dist(yaw_mean, yaw_cov);

        ini_mean_ = {x0_dist.calc_mean(), y0_dist.calc_mean(), v0_dist.calc_mean(), yaw0_dist.calc_mean()};
        ini_cov_ << x0_dist.calc_variance(), 0.0, 0.0, 0.0,
                0.0, y0_dist.calc_variance(), 0.0, 0.0,
                0.0, 0.0, v0_dist.calc_variance(), 0.0,
                0.0, 0.0, 0.0, yaw0_dist.calc_variance();

        // Input
        a_input_ = Eigen::VectorXd::Constant(N, 0.03);
        u_input_ = Eigen::VectorXd::Constant(N, 0.05);

        // System Noise
        const double mean_wv = 0.0;
        const double cov_wv = std::pow(1.0, 2);
        const double mean_wyaw = 0.0;
        const double cov_wyaw = std::pow(0.2, 2);
        system_noise_map_ = {
                {SYSTEM_NOISE::IDX::WV, std::make_shared<NormalDistribution>(mean_wv*dt, cov_wv*dt*dt)},
                {SYSTEM_NOISE::IDX::WYAW, std::make_shared<NormalDistribution>(mean_wyaw*dt, cov_wyaw*dt*dt)}};

        // Observation Noise
        const double mean_meas_noise_r = 10.0;
        const double cov_meas_noise_r = std::pow(1.5, 2);
        const double mean_meas_noise_vc = 0.0;
        const double cov_meas_noise_vc = std::pow(2.5, 2);
        const double mean_meas_noise_yaw = 0.0;
        const double cov_meas_noise_yaw = std::pow(M_PI/10.0, 2);
        observation_noise_map_ = {
                {OBSERVATION_NOISE::IDX::WR, std::make_shared<NormalDistribution>(mean_meas_noise_r, cov_meas_noise_r)},
                {OBSERVATION_NOISE::IDX::WVC, std::make_shared<NormalDistribution>(mean_meas_noise_vc, cov_meas_noise_vc)},
                {OBSERVATION_NOISE::IDX::WYAW, std::make_shared<NormalDistribution>(mean_meas_noise_yaw, cov_meas_noise_yaw)}};

        // Random Variable Generator
        wv_dist_ = std::normal_distribution<double>(mean_wv*dt, std::sqrt(cov_wv)*dt);
        wyaw_dist_ = std::normal_distribution<double>(mean_wyaw*dt, std::sqrt(cov_wyaw)*dt);
        mr_dist_ = std::normal_distribution<double>(mean_meas_noise_r, std::sqrt(cov_meas_noise_r));
        mvc_dist_ = std::normal_distribution<double>(mean_meas_noise_vc, std::sqrt(cov_meas_noise_vc));
        myaw_dist_ = std::normal_distribution<double>(mean_meas_noise_yaw, std::sqrt(cov_meas_noise_yaw));
    }

    // Initial Setting
    const size_t N{1000};
    const double dt{0.1};
    const std::string filename_{"/kinematic_vehicle_gaussian.csv"};

    // Initial Distribution
    Eigen::Vector4d ini_mean_;
    Eigen::Matrix4d ini_cov_;

    //Input
    Eigen::VectorXd a_input_;
    Eigen::VectorXd u_input_;

    // Noise
    std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map_;
    std::map<int, std::shared_ptr<BaseDistribution>> observation_noise_map_;

    std::normal_distribution<double> wv_dist_;
    std::normal_distribution<double> wyaw_dist_;
    std::normal_distribution<double> mr_dist_;
    std::normal_distribution<double> mvc_dist_;
    std::normal_distribution<double> myaw_dist_;
};

struct KinematicVehicleNonGaussianScenario
{
    KinematicVehicleNonGaussianScenario() : filename_("/kinematic_vehicle_non_gaussian.csv")
    {
        // Position normal distribution
        const double x_mean = 0.0;
        const double x_cov = 0.1*0.1;
        const double y_mean = 0.0;
        const double y_cov = 0.1*0.1;
        const double v_mean = 3.0;
        const double v_cov = 0.1*0.1;
        const double yaw_mean = M_PI/4.0;
        const double yaw_cov = 0.1*0.1;

        // Normal Distribution
        NormalDistribution x0_dist(x_mean, x_cov);
        NormalDistribution y0_dist(y_mean, y_cov);
        NormalDistribution v0_dist(v_mean, v_cov);
        NormalDistribution yaw0_dist(yaw_mean, yaw_cov);

        ini_mean_ = {x0_dist.calc_mean(), y0_dist.calc_mean(), v0_dist.calc_mean(), yaw0_dist.calc_mean()};
        ini_cov_ << x0_dist.calc_variance(), 0.0, 0.0, 0.0,
                0.0, y0_dist.calc_variance(), 0.0, 0.0,
                0.0, 0.0, v0_dist.calc_variance(), 0.0,
                0.0, 0.0, 0.0, yaw0_dist.calc_variance();

        // Input
        a_input_ = Eigen::VectorXd::Constant(N, 0.03);
        u_input_ = Eigen::VectorXd::Constant(N, 0.05);

        // System Noise
        const double mean_wv = 0.0;
        const double cov_wv = std::pow(1.0, 2);
        const double lower_wyaw = -0.2;
        const double upper_wyaw = 0.2;
        system_noise_map_ = {
                {SYSTEM_NOISE::IDX::WV, std::make_shared<NormalDistribution>(mean_wv*dt, cov_wv*dt*dt)},
                {SYSTEM_NOISE::IDX::WYAW, std::make_shared<UniformDistribution>(lower_wyaw*dt, upper_wyaw*dt)}};


        // Observation Noise
        const double lower_mr = 0.0;
        const double upper_mr = 10.0;
        const double lambda_mvc = 0.5;
        const double lower_myaw = -M_PI/10.0;
        const double upper_myaw = M_PI/10.0;
        observation_noise_map_ = {
                {OBSERVATION_NOISE::IDX::WR, std::make_shared<UniformDistribution>(lower_mr, upper_mr)},
                {OBSERVATION_NOISE::IDX::WVC, std::make_shared<ExponentialDistribution>(lambda_mvc)},
                {OBSERVATION_NOISE::IDX::WYAW, std::make_shared<UniformDistribution>(lower_myaw, upper_myaw)}};


        // Random Variable Generator
        wv_dist_ = std::normal_distribution<double>(mean_wv*dt, std::sqrt(cov_wv)*dt);
        wyaw_dist_ = std::uniform_real_distribution<double>(lower_wyaw*dt, upper_wyaw*dt);
        mr_dist_ = std::uniform_real_distribution<double>(lower_mr, upper_mr);
        mvc_dist_ = std::exponential_distribution<double>(lambda_mvc);
        myaw_dist_ = std::uniform_real_distribution<double>(lower_myaw, upper_myaw);
    }

    // Initial Setting
    const size_t N{1000};
    const double dt{0.1};
    const std::string filename_{"/kinematic_vehicle_non_gaussian.csv"};

    // Initial Distribution
    Eigen::Vector4d ini_mean_;
    Eigen::Matrix4d ini_cov_;

    //Input
    Eigen::VectorXd a_input_;
    Eigen::VectorXd u_input_;

    // Noise
    std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map_;
    std::map<int, std::shared_ptr<BaseDistribution>> observation_noise_map_;

    std::normal_distribution<double> wv_dist_;
    std::uniform_real_distribution<double> wyaw_dist_;
    std::uniform_real_distribution<double> mr_dist_;
    std::exponential_distribution<double> mvc_dist_;
    std::uniform_real_distribution<double> myaw_dist_;
};

#endif //UNCERTAINTY_PROPAGATION_KINEMATIC_VEHICLE_SCENARIO_H
