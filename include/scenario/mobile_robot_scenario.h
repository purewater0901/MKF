#ifndef UNCERTAINTY_PROPAGATION_MOBILE_ROBOT_SCENARIO_H
#define UNCERTAINTY_PROPAGATION_MOBILE_ROBOT_SCENARIO_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <Eigen/Eigen>

#include "model/mobile_robot_model.h"
#include "distribution/normal_distribution.h"
#include "distribution/uniform_distribution.h"
#include "distribution/exponential_distribution.h"
#include "utilities.h"

struct MobileRobotGaussianScenario
{
    MobileRobotGaussianScenario() : N(1000), dt(0.1), filename_("/mobile_robot_gaussian.csv")
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

        auto x0_dist = NormalDistribution(x_mean, x_cov);
        auto y0_dist = NormalDistribution(y_mean, y_cov);
        auto v0_dist = NormalDistribution(v_mean, v_cov);
        auto yaw0_dist = NormalDistribution(yaw_mean, yaw_cov);
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
        const double cov_wv = std::pow(0.5, 2);
        const double mean_wyaw = 0.0;
        const double cov_wyaw = std::pow(0.15, 2);
        system_noise_map_ = {
                {SYSTEM_NOISE::IDX::WV, std::make_shared<NormalDistribution>(mean_wv*dt, cov_wv*dt*dt)},
                {SYSTEM_NOISE::IDX::WYAW, std::make_shared<NormalDistribution>(mean_wyaw*dt, cov_wyaw*dt*dt)}};

        // Observation Noise
        const double mean_meas_noise_x = 0.0;
        const double cov_meas_noise_x = std::pow(3.5, 2);
        const double mean_meas_noise_y = 0.0;
        const double cov_meas_noise_y = std::pow(3.5, 2);
        const double mean_meas_noise_vc = 0.0;
        const double cov_meas_noise_vc = std::pow(3.0, 2);
        const double mean_meas_noise_yaw = 0.0;
        const double cov_meas_noise_yaw = std::pow(M_PI/10.0, 2);

        const auto meas_x_dist = std::make_shared<NormalDistribution>(mean_meas_noise_x, cov_meas_noise_x);
        const auto meas_y_dist = std::make_shared<NormalDistribution>(mean_meas_noise_y, cov_meas_noise_y);
        const auto meas_vc_dist = std::make_shared<NormalDistribution>(mean_meas_noise_vc, cov_meas_noise_vc);
        const auto meas_yaw_dist = std::make_shared<NormalDistribution>(mean_meas_noise_yaw, cov_meas_noise_yaw);
        observation_noise_map_ = {
                {OBSERVATION_NOISE::IDX::WX, meas_x_dist},
                {OBSERVATION_NOISE::IDX::WY, meas_y_dist},
                {OBSERVATION_NOISE::IDX::WVC, meas_vc_dist},
                {OBSERVATION_NOISE::IDX::WYAW, meas_yaw_dist}};

        // Random Variable Generator
        wv_dist_ = std::normal_distribution<double>(mean_wv*dt, std::sqrt(cov_wv)*dt);
        wyaw_dist_ = std::normal_distribution<double>(mean_wyaw*dt, std::sqrt(cov_wyaw)*dt);
        mx_dist_ = std::normal_distribution<double>(mean_meas_noise_x, std::sqrt(cov_meas_noise_x));
        my_dist_ = std::normal_distribution<double>(mean_meas_noise_y, std::sqrt(cov_meas_noise_y));
        mvc_dist_ = std::normal_distribution<double>(mean_meas_noise_vc, std::sqrt(cov_meas_noise_vc));
        myaw_dist_ = std::normal_distribution<double>(mean_meas_noise_yaw, std::sqrt(cov_meas_noise_yaw));
    }

    // Initial Setting
    const size_t N{1000};
    const double dt{0.1};
    const std::string filename_{"/mobile_robot_gaussian.csv"};

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
    std::normal_distribution<double> mx_dist_;
    std::normal_distribution<double> my_dist_;
    std::normal_distribution<double> mvc_dist_;
    std::normal_distribution<double> myaw_dist_;
};

struct MobileRobotNonGaussianScenario
{
    MobileRobotNonGaussianScenario() : N(1000), dt(0.1), filename_("/mobile_robot_non_gaussian.csv")
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

        auto x0_dist = NormalDistribution(x_mean, x_cov);
        auto y0_dist = NormalDistribution(y_mean, y_cov);
        auto v0_dist = NormalDistribution(v_mean, v_cov);
        auto yaw0_dist = NormalDistribution(yaw_mean, yaw_cov);
        ini_mean_ = {x0_dist.calc_mean(), y0_dist.calc_mean(), v0_dist.calc_mean(), yaw0_dist.calc_mean()};
        ini_cov_ << x0_dist.calc_variance(), 0.0, 0.0, 0.0,
                0.0, y0_dist.calc_variance(), 0.0, 0.0,
                0.0, 0.0, v0_dist.calc_variance(), 0.0,
                0.0, 0.0, 0.0, yaw0_dist.calc_variance();

        // Input
        a_input_ = Eigen::VectorXd::Constant(N, 0.03);
        u_input_ = Eigen::VectorXd::Constant(N, 0.05);

        // System Noise
        // System Noise
        const double lower_wv = -3.0;
        const double upper_wv = 3.0;
        const double lower_wyaw = -0.65;
        const double upper_wyaw = 0.65;
        system_noise_map_ = {
                {SYSTEM_NOISE::IDX::WV, std::make_shared<UniformDistribution>(lower_wv*dt, upper_wv*dt)},
                {SYSTEM_NOISE::IDX::WYAW, std::make_shared<UniformDistribution>(lower_wyaw*dt, upper_wyaw*dt)}};

        // Observation Noise
        const double lower_meas_x = -1.5;
        const double upper_meas_x =  1.0;
        const double lower_meas_y = -1.0;
        const double upper_meas_y =  1.5;
        const double lambda_vc = 0.4;
        const double lower_meas_yaw = -M_PI/10.0;
        const double upper_meas_yaw = M_PI/10.0;
        observation_noise_map_ = {
                {OBSERVATION_NOISE::IDX::WX, std::make_shared<UniformDistribution>(lower_meas_x, upper_meas_x)},
                {OBSERVATION_NOISE::IDX::WY, std::make_shared<UniformDistribution>(lower_meas_y, upper_meas_y)},
                {OBSERVATION_NOISE::IDX::WVC, std::make_shared<ExponentialDistribution>(lambda_vc)},
                {OBSERVATION_NOISE::IDX::WYAW, std::make_shared<UniformDistribution>(lower_meas_yaw, upper_meas_yaw)}};

        // Random Variable Generator
        wv_dist_ = std::uniform_real_distribution<double>(lower_wv*dt, upper_wv*dt);
        wyaw_dist_ = std::uniform_real_distribution<double>(lower_wyaw*dt, upper_wyaw*dt);
        mx_dist_ = std::uniform_real_distribution<double>(lower_meas_x, upper_meas_x);
        my_dist_ = std::uniform_real_distribution<double>(lower_meas_y, upper_meas_y);
        mvc_dist_ = std::exponential_distribution<double>(lambda_vc);
        myaw_dist_ = std::uniform_real_distribution<double>(lower_meas_yaw, upper_meas_yaw);
    }

    // Initial Setting
    const size_t N{1000};
    const double dt{0.1};
    const std::string filename_{"/mobile_robot_non_gaussian.csv"};

    // Initial Distribution
    Eigen::Vector4d ini_mean_;
    Eigen::Matrix4d ini_cov_;

    //Input
    Eigen::VectorXd a_input_;
    Eigen::VectorXd u_input_;

    // Noise
    std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map_;
    std::map<int, std::shared_ptr<BaseDistribution>> observation_noise_map_;

    std::uniform_real_distribution<double> wv_dist_;
    std::uniform_real_distribution<double> wyaw_dist_;
    std::uniform_real_distribution<double> mx_dist_;
    std::uniform_real_distribution<double> my_dist_;
    std::exponential_distribution<double> mvc_dist_;
    std::uniform_real_distribution<double> myaw_dist_;
};

#endif //UNCERTAINTY_PROPAGATION_MOBILE_ROBOT_SCENARIO_H
