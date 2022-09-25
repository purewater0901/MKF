#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Eigen>
#include <fstream>
#include <filesystem>
#include <algorithm>

#include "matplotlibcpp.h"
#include "distribution/two_dimensional_normal_distribution.h"
#include "filter/simple_vehicle_nkf.h"
#include "filter/simple_vehicle_ukf.h"
#include "filter/simple_vehicle_ekf.h"
#include "model/simple_vehicle_model.h"
#include "scenario/simple_vehicle_scenario.h"

using namespace SimpleVehicle;

struct LandMark
{
    LandMark(const double _x, const double _y, const double _std_x, const double _std_y) : x(_x), y(_y), std_x(_std_x), std_y(_std_y)
    {
    }
    double x;
    double y;
    double std_x;
    double std_y;
};

int main() {
    const int robot_num = 2;
    // Creating Map
    std::map<int, int> barcode_map;
    barcode_map.insert(std::make_pair(23, 5));
    barcode_map.insert(std::make_pair(72, 6));
    barcode_map.insert(std::make_pair(27, 7));
    barcode_map.insert(std::make_pair(54, 8));
    barcode_map.insert(std::make_pair(70, 9));
    barcode_map.insert(std::make_pair(36, 10));
    barcode_map.insert(std::make_pair(18, 11));
    barcode_map.insert(std::make_pair(25, 12));
    barcode_map.insert(std::make_pair(9, 13));
    barcode_map.insert(std::make_pair(81, 14));
    barcode_map.insert(std::make_pair(16, 15));
    barcode_map.insert(std::make_pair(90, 16));
    barcode_map.insert(std::make_pair(61, 17));
    barcode_map.insert(std::make_pair(45, 18));
    barcode_map.insert(std::make_pair(7, 19));
    barcode_map.insert(std::make_pair(63, 20));

    std::map<size_t, LandMark> landmark_map;
    std::ifstream landmark_file("/home/yutaka/CLionProjects/MKF/data/MRCLAM_Dataset1/Landmark_Groundtruth.dat");
    if(landmark_file.fail()) {
        std::cout << "Failed to Open the landmark truth file" << std::endl;
        return -1;
    }
    {
        size_t id;
        double x, y, std_x, std_y;
        landmark_file >> id >> x >> y >> std_x >> std_y;
        while(!landmark_file.eof())
        {
            landmark_map.insert(std::make_pair(id, LandMark(x, y, std_x, std_y)));
            landmark_file >> id >> x >> y >> std_x >> std_y;
        }
        landmark_file.close();
    }

    // Reading files
    const std::string odometry_filename = "/home/yutaka/CLionProjects/MKF/data/MRCLAM_Dataset1/Robot" + std::to_string(robot_num) + "_Odometry.dat";
    std::ifstream odometry_file(odometry_filename);
    if(odometry_file.fail()) {
        std::cout << "Failed to Open the ground truth file" << std::endl;
        return -1;
    }
    std::vector<double> odometry_time;
    std::vector<double> odometry_v;
    std::vector<double> odometry_w;
    {
        double time, v, w;
        odometry_file >> time >> v >> w;
        while(!odometry_file.eof())
        {
            odometry_time.push_back(time);
            odometry_v.push_back(v);
            odometry_w.push_back(w);
            odometry_file >> time >> v >> w;
        }
        odometry_file.close();
    }
    const double base_time = odometry_time.front();
    for(size_t i=0; i<odometry_time.size(); ++i){
        odometry_time.at(i) -= base_time;
    }

    const std::string ground_truth_filename = "/home/yutaka/CLionProjects/MKF/data/MRCLAM_Dataset1/Robot" + std::to_string(robot_num) + "_Groundtruth.dat";
    std::ifstream ground_truth_file(ground_truth_filename);
    if(ground_truth_file.fail()) {
        std::cout << "Failed to Open the ground truth file" << std::endl;
        return -1;
    }
    std::vector<double> ground_truth_time;
    std::vector<double> ground_truth_x;
    std::vector<double> ground_truth_y;
    std::vector<double> ground_truth_yaw;
    {
        double time, x, y, yaw;
        ground_truth_file >> time >> x >> y >> yaw;
        while(!ground_truth_file.eof())
        {
            if(time - base_time < 0.0) {
                ground_truth_file >> time >> x >> y >> yaw;
                continue;
            }

            ground_truth_time.push_back(time - base_time);
            ground_truth_x.push_back(x);
            ground_truth_y.push_back(y);
            ground_truth_yaw.push_back(yaw);
            ground_truth_file >> time >> x >> y >> yaw;
        }
        ground_truth_file.close();
    }

    const std::string measurement_filename = "/home/yutaka/CLionProjects/MKF/data/MRCLAM_Dataset1/Robot" + std::to_string(robot_num) + "_Measurement_updated.dat";
    std::ifstream measurement_file(measurement_filename);
    if(measurement_file.fail()) {
        std::cout << "Failed to Open the ground truth file" << std::endl;
        return -1;
    }
    std::vector<double> measurement_time;
    std::vector<size_t> measurement_subject;
    std::vector<double> measurement_range;
    std::vector<double> measurement_bearing;
    std::vector<double> measurement_true_range;
    std::vector<double> measurement_true_bearing;
    {
        double time, range, bearing, range_true, bearing_true;
        int id;
        measurement_file >> time >> id >> range >> bearing >> range_true >> bearing_true;
        while(!measurement_file.eof())
        {
            if( (0 <= id  && id <=5) || id==11 || id == 17 || time < 0.0){
            // if( (0 <= id  && id <=5) || time < 0.0){
                measurement_file >> time >> id >> range >> bearing >> range_true >> bearing_true;
                continue;
            }
            std::cout << id << std::endl;
            measurement_time.push_back(time);
            measurement_subject.push_back(id);
            measurement_range.push_back(range);
            measurement_bearing.push_back(bearing);
            measurement_true_range.push_back(range_true);
            measurement_true_bearing.push_back(bearing_true);
            measurement_file >> time >> id >> range >> bearing >> range_true >> bearing_true;
        }
        measurement_file.close();
    }

    /////////////////////////////////
    ///// Setup each filter /////////
    /////////////////////////////////
    SimpleVehicleNonGaussianScenario scenario;

    SimpleVehicleEKF ekf;
    SimpleVehicleUKF ukf;
    SimpleVehicleNKF nkf;

    // External disturbances
    const double mean_wv = 0.0;
    const double cov_wv = std::pow(0.1, 2);
    const double mean_wu = 0.0;
    const double cov_wu = std::pow(1.0, 2);

    std::default_random_engine generator;
    auto wr_dist = scenario.wr_dist_;
    auto wa_dist = scenario.wa_dist_;

    // Observation Noise
    const auto measurement_noise_map = scenario.observation_noise_map_;

    StateInfo nkf_state_info;
    nkf_state_info.mean = {ground_truth_x.front(), ground_truth_y.front(), ground_truth_yaw.front()};
    nkf_state_info.covariance = scenario.ini_cov_;
    auto ekf_state_info = nkf_state_info;
    auto ukf_state_info = nkf_state_info;

    ////////////////////////////////////////////////////
    // Start Simulation
    ////////////////////////////////////////////////////
    std::vector<double> times;
    std::vector<double> ekf_xy_errors;
    std::vector<double> ekf_yaw_errors;
    std::vector<double> ukf_xy_errors;
    std::vector<double> ukf_yaw_errors;
    std::vector<double> nkf_xy_errors;
    std::vector<double> nkf_yaw_errors;
    std::vector<double> x_true_vec;
    std::vector<double> y_true_vec;
    std::vector<double> yaw_true_vec;
    std::vector<double> nkf_x_estimate;
    std::vector<double> nkf_y_estimate;
    std::vector<double> nkf_yaw_estimate;
    std::vector<double> ekf_x_estimate;
    std::vector<double> ekf_y_estimate;
    std::vector<double> ekf_yaw_estimate;
    std::vector<double> ukf_x_estimate;
    std::vector<double> ukf_y_estimate;
    std::vector<double> ukf_yaw_estimate;

    size_t measurement_id = 0;
    size_t ground_truth_id = 0;
    for(size_t odo_id = 0; odo_id < 20000; ++odo_id){
        double current_time = odometry_time.at(odo_id);
        const double next_time = odometry_time.at(odo_id+1);

        // Check if we need update
        if(measurement_id < measurement_time.size() && next_time - measurement_time.at(measurement_id) > 0.0) {
            // predict till measurement time and update
            while(measurement_id < measurement_time.size()) {
                if(next_time - measurement_time.at(measurement_id) < 0.0) {
                    break;
                }

                const double dt = measurement_time.at(measurement_id) - current_time;

                // predict
                if(dt > 1e-5) {
                    const Eigen::Vector2d inputs = {odometry_v.at(odo_id)*dt, odometry_w.at(odo_id)*dt};
                    const std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map = {
                            {SYSTEM_NOISE::IDX::WV, std::make_shared<NormalDistribution>(mean_wv*dt, cov_wv*dt*dt)},
                            {SYSTEM_NOISE::IDX::WU, std::make_shared<NormalDistribution>(mean_wu*dt, cov_wu*dt*dt)}};
                    ekf_state_info = ekf.predict(ekf_state_info, inputs, dt, system_noise_map);
                    ukf_state_info = ukf.predict(ukf_state_info, inputs, system_noise_map, measurement_noise_map);
                    nkf_state_info = nkf.predict(nkf_state_info, inputs, system_noise_map);
                }

                // update
                const Eigen::Vector2d meas = {measurement_true_range[measurement_id], measurement_true_bearing[measurement_id]}; // measurement value
                const Eigen::Vector2d meas_noise = {wr_dist(generator), wa_dist(generator)};
                const Eigen::Vector2d y = {meas(0)*std::cos(meas(1) + meas_noise(1))*meas_noise(0), meas(0)*std::sin(meas(1) + meas_noise(1))*meas_noise(0)}; // transform
                const auto landmark = landmark_map.at(measurement_subject.at(measurement_id));
                const double updated_dt = std::max(1e-5, dt);
                const std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map = {
                        {SYSTEM_NOISE::IDX::WV, std::make_shared<NormalDistribution>(mean_wv*updated_dt, cov_wv*updated_dt*updated_dt)},
                        {SYSTEM_NOISE::IDX::WU, std::make_shared<NormalDistribution>(mean_wu*updated_dt, cov_wu*updated_dt*updated_dt)}};
                ekf_state_info = ekf.update(ekf_state_info, y, {landmark.x, landmark.y}, measurement_noise_map);
                ukf_state_info = ukf.update(ukf_state_info, y, {landmark.x, landmark.y}, system_noise_map, measurement_noise_map);
                nkf_state_info = nkf.update(nkf_state_info, y, {landmark.x, landmark.y}, measurement_noise_map);
                std::cout << "y: " << y << std::endl;

                current_time = measurement_time.at(measurement_id);
                ++measurement_id;
            }

            // End simulation if we cannot receive measurement values anymore
            if(measurement_id == measurement_time.size()) {
                break;
            }
        }

        // predict till ground truth
        while(ground_truth_time.at(ground_truth_id) < current_time && ground_truth_time.at(ground_truth_id) < next_time) {
            ++ground_truth_id;
        }

        if(current_time < ground_truth_time.at(ground_truth_id) && ground_truth_time.at(ground_truth_id) < next_time) {
            while(true) {
                if(next_time < ground_truth_time.at(ground_truth_id)) {
                    break;
                }

                const double dt = ground_truth_time.at(ground_truth_id) - current_time;
                if(dt > 1e-5) {
                    const Eigen::Vector2d inputs = {odometry_v.at(odo_id)*dt, odometry_w.at(odo_id)*dt};
                    const std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map = {
                            {SYSTEM_NOISE::IDX::WV, std::make_shared<NormalDistribution>(mean_wv*dt, cov_wv*dt*dt)},
                            {SYSTEM_NOISE::IDX::WU, std::make_shared<NormalDistribution>(mean_wu*dt, cov_wu*dt*dt)}};
                    ekf_state_info = ekf.predict(ekf_state_info, inputs, dt, system_noise_map);
                    ukf_state_info = ukf.predict(ukf_state_info, inputs, system_noise_map, measurement_noise_map);
                    nkf_state_info = nkf.predict(nkf_state_info, inputs, system_noise_map);
                }

                // Compare
                // update time
                times.push_back(current_time);

                // Ground truth value
                const double true_x = ground_truth_x.at(ground_truth_id);
                const double true_y = ground_truth_y.at(ground_truth_id);
                const double true_yaw = ground_truth_yaw.at(ground_truth_id);

                // Ground Truth Value
                x_true_vec.push_back(true_x);
                y_true_vec.push_back(true_y);
                yaw_true_vec.push_back(true_yaw);

                // EKF
                {
                    const double dx = true_x - ekf_state_info.mean(0);
                    const double dy = true_y - ekf_state_info.mean(1);
                    const double xy_error = std::hypot(dx, dy);
                    const double dyaw = normalizeRadian(true_yaw - ekf_state_info.mean(2));

                    ekf_xy_errors.push_back(xy_error);
                    ekf_yaw_errors.push_back(std::fabs(dyaw));
                    ekf_x_estimate.push_back(ekf_state_info.mean(0));
                    ekf_y_estimate.push_back(ekf_state_info.mean(1));
                    ekf_yaw_estimate.push_back(ekf_state_info.mean(2));

                    //std::cout << "ekf_xy_error: " << xy_error << std::endl;
                    //std::cout << "ekf_yaw_error: " << dyaw << std::endl;
                }

                // UKF
                {
                    const double dx = true_x - ukf_state_info.mean(0);
                    const double dy = true_y - ukf_state_info.mean(1);
                    const double xy_error = std::hypot(dx, dy);
                    const double dyaw = normalizeRadian(true_yaw - ukf_state_info.mean(2));

                    ukf_xy_errors.push_back(xy_error);
                    ukf_yaw_errors.push_back(std::fabs(dyaw));
                    ukf_x_estimate.push_back(ukf_state_info.mean(0));
                    ukf_y_estimate.push_back(ukf_state_info.mean(1));
                    ukf_yaw_estimate.push_back(ukf_state_info.mean(2));

                    //std::cout << "ukf_xy_error: " << xy_error << std::endl;
                    //std::cout << "ukf_yaw_error: " << dyaw << std::endl;
                }

                // NKF
                {
                    const double dx = true_x - nkf_state_info.mean(0);
                    const double dy = true_y - nkf_state_info.mean(1);
                    const double xy_error = std::hypot(dx, dy);
                    const double dyaw = normalizeRadian(true_yaw - nkf_state_info.mean(2));

                    nkf_xy_errors.push_back(xy_error);
                    nkf_yaw_errors.push_back(std::fabs(dyaw));
                    nkf_x_estimate.push_back(nkf_state_info.mean(0));
                    nkf_y_estimate.push_back(nkf_state_info.mean(1));
                    nkf_yaw_estimate.push_back(nkf_state_info.mean(2));

                    //std::cout << "nkf_xy_error: " << xy_error << std::endl;
                    //std::cout << "nkf_yaw_error: " << dyaw << std::endl;
                    //std::cout << "-----------------------" << std::endl;
                }

                current_time = ground_truth_time.at(ground_truth_id);
                ++ground_truth_id;
            }
        }

        // normal predict till next time
        const double dt = next_time - current_time;
        if(dt > 1e-5) {
            const Eigen::Vector2d inputs = {odometry_v.at(odo_id)*dt, odometry_w.at(odo_id)*dt};
            const std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map = {
                    {SYSTEM_NOISE::IDX::WV, std::make_shared<NormalDistribution>(mean_wv*dt, cov_wv*dt*dt)},
                    {SYSTEM_NOISE::IDX::WU, std::make_shared<NormalDistribution>(mean_wu*dt, cov_wu*dt*dt)}};
            ekf_state_info = ekf.predict(ekf_state_info, inputs, dt, system_noise_map);
            ukf_state_info = ukf.predict(ukf_state_info, inputs, system_noise_map, measurement_noise_map);
            nkf_state_info = nkf.predict(nkf_state_info, inputs, system_noise_map);
        }
    }

    double nkf_xy_error_sum = 0.0;
    double ekf_xy_error_sum = 0.0;
    double ukf_xy_error_sum = 0.0;
    double nkf_yaw_error_sum = 0.0;
    double ekf_yaw_error_sum = 0.0;
    double ukf_yaw_error_sum = 0.0;
    for(size_t i=0; i<ukf_xy_errors.size(); ++i) {
        nkf_xy_error_sum += nkf_xy_errors.at(i);
        ekf_xy_error_sum += ekf_xy_errors.at(i);
        ukf_xy_error_sum += ukf_xy_errors.at(i);
        nkf_yaw_error_sum += nkf_yaw_errors.at(i);
        ekf_yaw_error_sum += ekf_yaw_errors.at(i);
        ukf_yaw_error_sum += ukf_yaw_errors.at(i);
    }

    // Output data to file
    {
        std::string parent_dir = "/home/yutaka/CLionProjects/MKF/result";
        for(const auto& p : std::filesystem::directory_iterator("../result/"))
        {
            const auto abs_p = std::filesystem::canonical(p);
            const auto flag_find = abs_p.string().find("data");
            if(flag_find != std::string::npos) {
                parent_dir = abs_p.string();
            }
        }
        parent_dir += "/robot" + std::to_string(robot_num);
        std::filesystem::create_directories(parent_dir);
        const std::string filename = parent_dir + scenario.filename_;
        outputResultToFile(filename, times,
                           x_true_vec, y_true_vec, yaw_true_vec,
                           nkf_x_estimate, nkf_y_estimate, nkf_yaw_estimate,
                           ekf_x_estimate, ekf_y_estimate, ekf_yaw_estimate,
                           ukf_x_estimate, ukf_y_estimate, ukf_yaw_estimate,
                           nkf_xy_errors, nkf_yaw_errors,
                           ekf_xy_errors, ekf_yaw_errors,
                           ukf_xy_errors, ukf_yaw_errors);
    }

    std::cout << "ekf_xy_error mean: " << ekf_xy_error_sum / ekf_xy_errors.size() << std::endl;
    std::cout << "ukf_xy_error mean: " << ukf_xy_error_sum / ukf_xy_errors.size() << std::endl;
    std::cout << "nkf_xy_error mean: " << nkf_xy_error_sum / nkf_xy_errors.size() << std::endl;
    std::cout << "ekf_yaw_error mean: " << ekf_yaw_error_sum / ekf_yaw_errors.size() << std::endl;
    std::cout << "ukf_yaw_error mean: " << ukf_yaw_error_sum / ukf_yaw_errors.size() << std::endl;
    std::cout << "nkf_yaw_error mean: " << nkf_yaw_error_sum / nkf_yaw_errors.size() << std::endl;

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
