#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Eigen>

#include "matplotlibcpp.h"
#include "distribution/normal_distribution.h"
#include "model/normal_vehicle_model.h"
#include "filter/normal_vehicle_nkf.h"
#include "filter/normal_vehicle_ukf.h"
#include "filter/normal_vehicle_ekf.h"

using namespace NormalVehicle;

int main()
{
    const size_t N = 10000;
    const double dt = 0.1;

    // Normal Vehicle Nonlinear Kalman Filter
    NormalVehicleNKF normal_vehicle_nkf;

    // Normal Vehicle Unscented Kalman Filter
    NormalVehicleUKF normal_vehicle_ukf;

    // Normal Vehicle Extended Kalman Filter
    NormalVehicleEKF normal_vehicle_ekf;

    // Position normal distribution
    const double x_mean = 0.0;
    const double x_cov = 0.1*0.1;
    const double y_mean = 0.0;
    const double y_cov = 0.1*0.1;
    const double yaw_mean = M_PI/4.0;
    const double yaw_cov = 0.1*0.1;

    // Uniform Distribution
    NormalDistribution x_dist(x_mean, x_cov);
    NormalDistribution y_dist(y_mean, y_cov);
    NormalDistribution yaw_dist(yaw_mean, yaw_cov);

    StateInfo nkf_state_info;
    nkf_state_info.mean = {x_dist.calc_mean(), y_dist.calc_mean(), yaw_dist.calc_mean()};
    nkf_state_info.covariance << x_dist.calc_variance(), 0.0, 0.0,
                                 0.0, y_dist.calc_variance(), 0.0,
                                 0.0, 0.0, yaw_dist.calc_variance();
    auto ekf_state_info = nkf_state_info;
    auto ukf_state_info = nkf_state_info;

    // Initial State
    Eigen::Vector3d x_true = nkf_state_info.mean;

    // Input
    const Eigen::VectorXd v_input = Eigen::VectorXd::Constant(N, 2.0);
    const Eigen::VectorXd u_input = Eigen::VectorXd::Constant(N, 0.05);

    // System Noise
    const double mean_wx = 0.0;
    const double cov_wx = std::pow(0.1*dt, 2);
    const double mean_wy = 0.0;
    const double cov_wy = std::pow(0.1*dt, 2);
    const double mean_wyaw = 0.0;
    const double cov_wyaw = std::pow(0.1*dt, 2);
    std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map{
            {SYSTEM_NOISE::IDX::WX, std::make_shared<NormalDistribution>(mean_wx, cov_wx)},
            {SYSTEM_NOISE::IDX::WY, std::make_shared<NormalDistribution>(mean_wy, cov_wy)},
            {SYSTEM_NOISE::IDX::WYAW, std::make_shared<NormalDistribution>(mean_wyaw, cov_wyaw)}};

    // Observation Noise
    const double mean_meas_noise_r = 100.0;
    const double cov_meas_noise_r = std::pow(10.5, 2);
    const double mean_meas_noise_yaw = 0.0;
    const double cov_meas_noise_yaw = std::pow(M_PI/10.0, 2);

    std::map<int, std::shared_ptr<BaseDistribution>> observation_noise_map{
            {OBSERVATION_NOISE::IDX::WR, std::make_shared<NormalDistribution>(mean_meas_noise_r, cov_meas_noise_r)},
            {OBSERVATION_NOISE::IDX::WYAW, std::make_shared<NormalDistribution>(mean_meas_noise_yaw, cov_meas_noise_yaw)}};

    // Random Variable Generator
    std::default_random_engine generator;
    std::normal_distribution<double> wx_dist(mean_wx, std::sqrt(cov_wx));
    std::normal_distribution<double> wy_dist(mean_wy, std::sqrt(cov_wy));
    std::normal_distribution<double> wyaw_dist(mean_wyaw, std::sqrt(cov_wyaw));
    std::normal_distribution<double> mr_dist(mean_meas_noise_r, std::sqrt(cov_meas_noise_r));
    std::normal_distribution<double> myaw_dist(mean_meas_noise_yaw, std::sqrt(cov_meas_noise_yaw));

    NormalVehicleModel vehicle_model;
    std::vector<double> times(N);
    std::vector<double> ukf_xy_errors(N);
    std::vector<double> nkf_xy_errors(N);
    std::vector<double> ekf_xy_errors(N);
    std::vector<double> x_true_vec(N);
    std::vector<double> y_true_vec(N);
    std::vector<double> nkf_x_estimate(N);
    std::vector<double> nkf_y_estimate(N);
    std::vector<double> ukf_x_estimate(N);
    std::vector<double> ukf_y_estimate(N);
    std::vector<double> ekf_x_estimate(N);
    std::vector<double> ekf_y_estimate(N);
    for(size_t i=0; i < N; ++i) {
        std::cout << "iteration: " << i << std::endl;
        // Control Inputs
        Eigen::Vector2d controls(v_input(i)*dt, u_input(i)*dt);

        // Simulate
        Eigen::Vector3d system_noise{wx_dist(generator), wy_dist(generator), wyaw_dist(generator)};
        Eigen::Vector2d observation_noise{std::max(0.0, mr_dist(generator)), myaw_dist(generator)};
        x_true = vehicle_model.propagate(x_true, controls, system_noise);
        auto nkf_y = vehicle_model.observe(x_true, observation_noise);
        auto ukf_y = nkf_y;
        auto ekf_y = nkf_y;

        // Predict
        const auto nkf_predicted_info = normal_vehicle_nkf.predict(nkf_state_info, controls, system_noise_map);
        const auto ekf_predicted_info = normal_vehicle_ekf.predict(ekf_state_info, controls, system_noise_map);
        const auto ukf_predicted_info = normal_vehicle_ukf.predict(ukf_state_info, controls, system_noise_map, observation_noise_map);

        // Recalculate Yaw Angle to avoid the angle over 2*pi
        const double nkf_yaw_error = normalizeRadian(nkf_y(OBSERVATION::IDX::YAW) - nkf_predicted_info.mean(STATE::IDX::YAW));
        const double ekf_yaw_error = normalizeRadian(ekf_y(OBSERVATION::IDX::YAW) - ekf_predicted_info.mean(STATE::IDX::YAW));
        const double ukf_yaw_error = normalizeRadian(ukf_y(OBSERVATION::IDX::YAW) - ukf_predicted_info.mean(STATE::IDX::YAW));
        nkf_y(OBSERVATION::IDX::YAW) = nkf_yaw_error + nkf_predicted_info.mean(STATE::IDX::YAW);
        ekf_y(OBSERVATION::IDX::YAW) = ekf_yaw_error + ekf_predicted_info.mean(STATE::IDX::YAW);
        ukf_y(OBSERVATION::IDX::YAW) = ukf_yaw_error + ukf_predicted_info.mean(STATE::IDX::YAW);

        // Update
        const auto nkf_updated_info = normal_vehicle_nkf.update(nkf_predicted_info, nkf_y, observation_noise_map);
        const auto ekf_updated_info = normal_vehicle_ekf.update(ekf_predicted_info, ekf_y, observation_noise_map);
        const auto ukf_updated_info = normal_vehicle_ukf.update(ukf_predicted_info, ukf_y, system_noise_map, observation_noise_map);
        nkf_state_info = nkf_updated_info;
        ekf_state_info = ekf_updated_info;
        ukf_state_info = ukf_updated_info;

        // NKF
        {
            const double dx = x_true(STATE::IDX::X) - nkf_updated_info.mean(STATE::IDX::X);
            const double dy = x_true(STATE::IDX::Y) - nkf_updated_info.mean(STATE::IDX::Y);
            const double xy_error = std::sqrt(dx*dx + dy*dy);
            const double nkf_yaw_error = std::abs(x_true(STATE::IDX::YAW) - nkf_updated_info.mean(STATE::IDX::YAW));

            std::cout << "nkf_xy_error: " << xy_error << std::endl;
            std::cout << "nkf_yaw_error: " << nkf_yaw_error << std::endl;
            nkf_xy_errors.at(i) = xy_error;
            nkf_x_estimate.at(i) = nkf_state_info.mean(STATE::IDX::X);
            nkf_y_estimate.at(i) = nkf_state_info.mean(STATE::IDX::Y);
        }

        // UKF
        {
            const double dx = x_true(STATE::IDX::X) - ukf_updated_info.mean(STATE::IDX::X);
            const double dy = x_true(STATE::IDX::Y) - ukf_updated_info.mean(STATE::IDX::Y);
            const double xy_error = std::sqrt(dx*dx + dy*dy);
            const double ukf_yaw_error = std::abs(x_true(STATE::IDX::YAW) - ukf_updated_info.mean(STATE::IDX::YAW));

            std::cout << "ukf_xy_error: " << xy_error << std::endl;
            std::cout << "ukf_yaw_error: " << ukf_yaw_error << std::endl;
            ukf_xy_errors.at(i) = xy_error;
            ukf_x_estimate.at(i) = ukf_updated_info.mean(STATE::IDX::X);
            ukf_y_estimate.at(i) = ukf_updated_info.mean(STATE::IDX::Y);
        }

        // EKF
        {
            const double dx = x_true(STATE::IDX::X) - ekf_updated_info.mean(STATE::IDX::X);
            const double dy = x_true(STATE::IDX::Y) - ekf_updated_info.mean(STATE::IDX::Y);
            const double xy_error = std::sqrt(dx*dx + dy*dy);
            const double ekf_yaw_error = std::abs(x_true(STATE::IDX::YAW) - ekf_updated_info.mean(STATE::IDX::YAW));

            std::cout << "ekf_xy_error: " << xy_error << std::endl;
            std::cout << "ekf_yaw_error: " << ekf_yaw_error << std::endl;
            ekf_xy_errors.at(i) = xy_error;
            ekf_x_estimate.at(i) = ekf_state_info.mean(STATE::IDX::X);
            ekf_y_estimate.at(i) = ekf_state_info.mean(STATE::IDX::Y);
        }

        times.at(i) = i*dt;
        x_true_vec.at(i) = x_true(0);
        y_true_vec.at(i) = x_true(1);
    }

    matplotlibcpp::figure_size(1500, 900);
    std::map<std::string, std::string> nkf_keywords;
    std::map<std::string, std::string> ukf_keywords;
    std::map<std::string, std::string> ekf_keywords;
    nkf_keywords.insert(std::pair<std::string, std::string>("label", "nkf error"));
    ukf_keywords.insert(std::pair<std::string, std::string>("label", "ukf error"));
    ekf_keywords.insert(std::pair<std::string, std::string>("label", "ekf error"));
    //matplotlibcpp::plot(times, nkf_xy_errors, nkf_keywords);
    //matplotlibcpp::plot(times, ukf_xy_errors, ukf_keywords);
    matplotlibcpp::plot(nkf_x_estimate, nkf_y_estimate, nkf_keywords);
    matplotlibcpp::plot(ukf_x_estimate, ukf_y_estimate, ukf_keywords);
    matplotlibcpp::plot(ekf_x_estimate, ekf_y_estimate, ekf_keywords);
    matplotlibcpp::named_plot("true", x_true_vec, y_true_vec);
    matplotlibcpp::legend();
    matplotlibcpp::title("Result");
    matplotlibcpp::show();

    return 0;
}
