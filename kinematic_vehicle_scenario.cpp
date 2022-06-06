#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Eigen>
#include <filesystem>

#include "matplotlibcpp.h"
#include "distribution/normal_distribution.h"
#include "model/kinematic_vehicle_model.h"
#include "filter/kinematic_vehilce_nkf.h"
#include "filter/kinematic_vehicle_ukf.h"
#include "filter/kinematic_vehicle_ekf.h"
#include "scenario/kinematic_vehicle_scenario.h"

using namespace KinematicVehicle;

int main()
{
    KinematicVehicleGaussianScenario scenario;
    const size_t N = scenario.N;
    const double dt = scenario.dt;

    // Kinematic Vehicle Nonlinear Kalman Filter
    KinematicVehicleNKF kinematic_vehicle_nkf;

    // Normal Vehicle Unscented Kalman Filter
    KinematicVehicleUKF kinematic_vehicle_ukf;

    // Normal Vehicle Extended Kalman Filter
    KinematicVehicleEKF kinematic_vehicle_ekf;

    StateInfo nkf_state_info;
    nkf_state_info.mean = scenario.ini_mean_;
    nkf_state_info.covariance = scenario.ini_cov_;
    auto ekf_state_info = nkf_state_info;
    auto ukf_state_info = nkf_state_info;

    // Initial State
    auto x_true = nkf_state_info.mean;

    // Noise
    const auto& system_noise_map = scenario.system_noise_map_;
    const auto& observation_noise_map = scenario.observation_noise_map_;

    // Random Variable Generator
    std::default_random_engine generator;
    auto& wv_dist = scenario.wv_dist_;
    auto& wyaw_dist = scenario.wyaw_dist_;
    auto& mr_dist = scenario.mr_dist_;
    auto& mvc_dist = scenario.mvc_dist_;
    auto& myaw_dist = scenario.myaw_dist_;

    KinematicVehicleModel vehicle_model;
    std::vector<double> times(N);
    std::vector<double> nkf_xy_errors(N);
    std::vector<double> ekf_xy_errors(N);
    std::vector<double> ukf_xy_errors(N);
    std::vector<double> nkf_yaw_errors(N);
    std::vector<double> ekf_yaw_errors(N);
    std::vector<double> ukf_yaw_errors(N);
    std::vector<double> nkf_v_errors(N);
    std::vector<double> ekf_v_errors(N);
    std::vector<double> ukf_v_errors(N);
    std::vector<double> x_true_vec(N);
    std::vector<double> y_true_vec(N);
    std::vector<double> v_true_vec(N);
    std::vector<double> yaw_true_vec(N);
    std::vector<double> nkf_x_estimate(N);
    std::vector<double> nkf_y_estimate(N);
    std::vector<double> nkf_v_estimate(N);
    std::vector<double> nkf_yaw_estimate(N);
    std::vector<double> ekf_x_estimate(N);
    std::vector<double> ekf_y_estimate(N);
    std::vector<double> ekf_v_estimate(N);
    std::vector<double> ekf_yaw_estimate(N);
    std::vector<double> ukf_x_estimate(N);
    std::vector<double> ukf_y_estimate(N);
    std::vector<double> ukf_v_estimate(N);
    std::vector<double> ukf_yaw_estimate(N);
    for(size_t i=0; i < N; ++i) {
        std::cout << "iteration: " << i << std::endl;
        Eigen::Vector2d controls(scenario.a_input_(i)*dt, scenario.u_input_(i)*dt);

        // Simulate
        Eigen::Vector2d system_noise{wv_dist(generator), wyaw_dist(generator)};
        Eigen::Vector3d observation_noise{std::max(0.0, mr_dist(generator)), mvc_dist(generator), myaw_dist(generator)};
        x_true = vehicle_model.propagate(x_true, controls, system_noise, dt);
        auto y_nkf = vehicle_model.observe(x_true, observation_noise);
        auto y_ekf = y_nkf;
        auto y_ukf = y_nkf;

        // Predict
        const auto nkf_predicted_info = kinematic_vehicle_nkf.predict(nkf_state_info, controls, dt, system_noise_map);
        const auto ekf_predicted_info = kinematic_vehicle_ekf.predict(ekf_state_info, controls, dt, system_noise_map);
        const auto ukf_predicted_info = kinematic_vehicle_ukf.predict(ukf_state_info, controls, dt, system_noise_map, observation_noise_map);

        const double nkf_yaw_error = normalizeRadian(y_nkf(OBSERVATION::IDX::YAW) - nkf_predicted_info.mean(STATE::IDX::YAW));
        const double ekf_yaw_error = normalizeRadian(y_ekf(OBSERVATION::IDX::YAW) - ekf_predicted_info.mean(STATE::IDX::YAW));
        const double ukf_yaw_error = normalizeRadian(y_ukf(OBSERVATION::IDX::YAW) - ukf_predicted_info.mean(STATE::IDX::YAW));
        y_nkf(OBSERVATION::IDX::YAW) = nkf_yaw_error + nkf_predicted_info.mean(STATE::IDX::YAW);
        y_ekf(OBSERVATION::IDX::YAW) = ekf_yaw_error + ekf_predicted_info.mean(STATE::IDX::YAW);
        y_ukf(OBSERVATION::IDX::YAW) = ukf_yaw_error + ukf_predicted_info.mean(STATE::IDX::YAW);

        // Update
        const auto nkf_updated_info = kinematic_vehicle_nkf.update(nkf_predicted_info, y_nkf, observation_noise_map);
        const auto ekf_updated_info = kinematic_vehicle_ekf.update(ekf_predicted_info, y_ekf, observation_noise_map);
        const auto ukf_updated_info = kinematic_vehicle_ukf.update(ukf_predicted_info, y_ukf, system_noise_map, observation_noise_map);
        nkf_state_info = nkf_updated_info;
        ekf_state_info = ekf_updated_info;
        ukf_state_info = ukf_updated_info;

        std::cout << "x_true" << std::endl;
        std::cout << x_true << std::endl;
        std::cout << "nkf predicted mean" << std::endl;
        std::cout << nkf_predicted_info.mean << std::endl;
        std::cout << "ekf predicted mean" << std::endl;
        std::cout << ekf_predicted_info.mean << std::endl;
        std::cout << "ukf predicted mean" << std::endl;
        std::cout << ukf_predicted_info.mean << std::endl;
        std::cout << "nkf updated mean" << std::endl;
        std::cout << nkf_updated_info.mean << std::endl;
        std::cout << "ekf updated mean" << std::endl;
        std::cout << ekf_updated_info.mean << std::endl;
        std::cout << "ukf updated mean" << std::endl;
        std::cout << ukf_updated_info.mean << std::endl;
        std::cout << "observation" << std::endl;
        std::cout << y_ekf << std::endl;
        std::cout << "---------------" << std::endl;

        {
            const double dx = x_true(STATE::IDX::X) - nkf_updated_info.mean(STATE::IDX::X);
            const double dy = x_true(STATE::IDX::Y) - nkf_updated_info.mean(STATE::IDX::Y);
            const double xy_error = std::sqrt(dx*dx + dy*dy);
            const double dv = x_true(STATE::IDX::V) - nkf_updated_info.mean(STATE::IDX::V);
            const double nkf_yaw_error = std::abs(x_true(STATE::IDX::YAW) - nkf_updated_info.mean(STATE::IDX::YAW));

            std::cout << "nkf_xy_error: " << xy_error << std::endl;
            std::cout << "nkf_yaw_error: " << nkf_yaw_error << std::endl;
            std::cout << "nkf dv: " << dv << std::endl;
            nkf_xy_errors.at(i) = xy_error;
            nkf_yaw_errors.at(i) = std::fabs(nkf_yaw_error);
            nkf_v_errors.at(i) = std::fabs(dv);
            nkf_x_estimate.at(i) = nkf_updated_info.mean(STATE::IDX::X);
            nkf_y_estimate.at(i) = nkf_updated_info.mean(STATE::IDX::Y);
            nkf_v_estimate.at(i) = nkf_updated_info.mean(STATE::IDX::V);
            nkf_yaw_estimate.at(i) = nkf_updated_info.mean(STATE::IDX::YAW);
        }

        // EKF
        {
            const double dx = x_true(STATE::IDX::X) - ekf_updated_info.mean(STATE::IDX::X);
            const double dy = x_true(STATE::IDX::Y) - ekf_updated_info.mean(STATE::IDX::Y);
            const double xy_error = std::sqrt(dx*dx + dy*dy);
            const double dv = x_true(STATE::IDX::V) - ekf_updated_info.mean(STATE::IDX::V);
            const double ekf_yaw_error = std::abs(x_true(STATE::IDX::YAW) - ekf_updated_info.mean(STATE::IDX::YAW));

            std::cout << "ekf_xy_error: " << xy_error << std::endl;
            std::cout << "ekf_yaw_error: " << ekf_yaw_error << std::endl;
            std::cout << "nkf dv: " << dv << std::endl;
            ekf_xy_errors.at(i) = xy_error;
            ekf_yaw_errors.at(i) = std::fabs(ekf_yaw_error);
            ekf_v_errors.at(i) = std::fabs(dv);
            ekf_x_estimate.at(i) = ekf_updated_info.mean(STATE::IDX::X);
            ekf_y_estimate.at(i) = ekf_updated_info.mean(STATE::IDX::Y);
            ekf_v_estimate.at(i) = ekf_updated_info.mean(STATE::IDX::V);
            ekf_yaw_estimate.at(i) = ekf_updated_info.mean(STATE::IDX::YAW);
        }

        // UKF
        {
            const double dx = x_true(STATE::IDX::X) - ukf_updated_info.mean(STATE::IDX::X);
            const double dy = x_true(STATE::IDX::Y) - ukf_updated_info.mean(STATE::IDX::Y);
            const double xy_error = std::sqrt(dx*dx + dy*dy);
            const double dv = x_true(STATE::IDX::V) - ukf_updated_info.mean(STATE::IDX::V);
            const double ukf_yaw_error = std::abs(x_true(STATE::IDX::YAW) - ukf_updated_info.mean(STATE::IDX::YAW));

            std::cout << "ukf_xy_error: " << xy_error << std::endl;
            std::cout << "ukf_yaw_error: " << ukf_yaw_error << std::endl;
            std::cout << "nkf dv: " << dv << std::endl;
            ukf_xy_errors.at(i) = xy_error;
            ukf_yaw_errors.at(i) = std::fabs(ukf_yaw_error);
            ukf_v_errors.at(i) = std::fabs(dv);
            ukf_x_estimate.at(i) = ukf_updated_info.mean(STATE::IDX::X);
            ukf_y_estimate.at(i) = ukf_updated_info.mean(STATE::IDX::Y);
            ukf_v_estimate.at(i) = ukf_updated_info.mean(STATE::IDX::V);
            ukf_yaw_estimate.at(i) = ukf_updated_info.mean(STATE::IDX::YAW);
        }
        times.at(i) = i*dt;
        x_true_vec.at(i) = x_true(STATE::IDX::X);
        y_true_vec.at(i) = x_true(STATE::IDX::Y);
        v_true_vec.at(i) = x_true(STATE::IDX::V);
        yaw_true_vec.at(i) = x_true(STATE::IDX::YAW);
    }

    // Output data to file
    {
        std::string parent_dir = "/home/yutaka/CLionProjects/uncertainty_propagation/result";
        for(const auto& p : std::filesystem::directory_iterator("../result/"))
        {
            const auto abs_p = std::filesystem::canonical(p);
            const auto flag_find = abs_p.string().find("data");
            if(flag_find != std::string::npos) {
                parent_dir = abs_p.string();
            }
        }
        const std::string filename = parent_dir + scenario.filename_;
        outputResultToFile(filename, times,
                           x_true_vec, y_true_vec, v_true_vec, yaw_true_vec,
                           nkf_x_estimate, nkf_y_estimate, nkf_v_estimate, nkf_yaw_estimate,
                           ekf_x_estimate, ekf_y_estimate, ekf_v_estimate, ekf_yaw_estimate,
                           ukf_x_estimate, ukf_y_estimate, ukf_v_estimate, ukf_yaw_estimate,
                           nkf_xy_errors, nkf_v_errors, nkf_yaw_errors,
                           ekf_xy_errors, ekf_v_errors, ekf_yaw_errors,
                           ukf_xy_errors, ukf_v_errors, ukf_yaw_errors);
    }

    double nkf_xy_error_sum = 0.0;
    double ekf_xy_error_sum = 0.0;
    double ukf_xy_error_sum = 0.0;
    double nkf_yaw_error_sum = 0.0;
    double ekf_yaw_error_sum = 0.0;
    double ukf_yaw_error_sum = 0.0;
    double nkf_v_error_sum = 0.0;
    double ekf_v_error_sum = 0.0;
    double ukf_v_error_sum = 0.0;
    for(size_t i=0; i<ukf_xy_errors.size(); ++i) {
        nkf_xy_error_sum += nkf_xy_errors.at(i);
        ekf_xy_error_sum += ekf_xy_errors.at(i);
        ukf_xy_error_sum += ukf_xy_errors.at(i);
        nkf_yaw_error_sum += nkf_yaw_errors.at(i);
        ekf_yaw_error_sum += ekf_yaw_errors.at(i);
        ukf_yaw_error_sum += ukf_yaw_errors.at(i);
        nkf_v_error_sum += nkf_v_errors.at(i);
        ekf_v_error_sum += ekf_v_errors.at(i);
        ukf_v_error_sum += ukf_v_errors.at(i);
    }

    std::cout << "nkf_xy_error mean: " << nkf_xy_error_sum / N << std::endl;
    std::cout << "ekf_xy_error mean: " << ekf_xy_error_sum / N << std::endl;
    std::cout << "ukf_xy_error mean: " << ukf_xy_error_sum / N << std::endl;
    std::cout << "nkf_yaw_error mean: " << nkf_yaw_error_sum / N << std::endl;
    std::cout << "ekf_yaw_error mean: " << ekf_yaw_error_sum / N << std::endl;
    std::cout << "ukf_yaw_error mean: " << ukf_yaw_error_sum / N << std::endl;
    std::cout << "nkf_v_error mean: " << nkf_v_error_sum / N << std::endl;
    std::cout << "ekf_v_error mean: " << ekf_v_error_sum / N << std::endl;
    std::cout << "ukf_v_error mean: " << ukf_v_error_sum / N << std::endl;

    matplotlibcpp::figure_size(1500, 900);
    std::map<std::string, std::string> nkf_keywords;
    std::map<std::string, std::string> ekf_keywords;
    std::map<std::string, std::string> ukf_keywords;
    nkf_keywords.insert(std::pair<std::string, std::string>("label", "nkf error"));
    ekf_keywords.insert(std::pair<std::string, std::string>("label", "ekf error"));
    ukf_keywords.insert(std::pair<std::string, std::string>("label", "ukf error"));
    //matplotlibcpp::plot(times, nkf_xy_errors, nkf_keywords);
    //matplotlibcpp::plot(times, ukf_xy_errors, ukf_keywords);
    matplotlibcpp::plot(nkf_x_estimate, nkf_y_estimate, nkf_keywords);
    matplotlibcpp::plot(ekf_x_estimate, ekf_y_estimate, ekf_keywords);
    matplotlibcpp::plot(ukf_x_estimate, ukf_y_estimate, ukf_keywords);
    matplotlibcpp::named_plot("true", x_true_vec, y_true_vec);
    matplotlibcpp::legend();
    matplotlibcpp::title("Result");
    matplotlibcpp::show();
    return 0;
}