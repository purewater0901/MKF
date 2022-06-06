#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Eigen>
#include <fstream>
#include <algorithm>

#include "matplotlibcpp.h"
#include "distribution/uniform_distribution.h"
#include "distribution/normal_distribution.h"
#include "distribution/two_dimensional_normal_distribution.h"
#include "filter/simple_vehicle_nkf.h"
#include "filter/simple_vehicle_ukf.h"
#include "model/simple_vehicle_model.h"

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
    std::ifstream landmark_file("/home/yutaka/CLionProjects/uncertainty_propagation/data/MRCLAM_Dataset1/Landmark_Groundtruth.dat");
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
    std::ifstream odometry_file("/home/yutaka/CLionProjects/uncertainty_propagation/data/MRCLAM_Dataset1/Robot1_Odometry.dat");
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
    for(size_t i=0; i < odometry_time.size(); ++i) {
        odometry_time.at(i) -= base_time;
    }

    std::ifstream ground_truth_file("/home/yutaka/CLionProjects/uncertainty_propagation/data/MRCLAM_Dataset1/Robot1_Groundtruth.dat");
    if(ground_truth_file.fail()) {
        std::cout << "Failed to Open the ground truth file" << std::endl;
        return -1;
    }
    std::vector<double> ground_truth_time{0.0};
    std::vector<double> ground_truth_x{3.57323240};
    std::vector<double> ground_truth_y{-3.33283870};
    std::vector<double> ground_truth_yaw{2.34080000};
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

    std::ifstream measurement_file("/home/yutaka/CLionProjects/uncertainty_propagation/data/MRCLAM_Dataset1/Robot1_Measurement.dat");
    if(measurement_file.fail()) {
        std::cout << "Failed to Open the ground truth file" << std::endl;
        return -1;
    }
    std::vector<double> measurement_time;
    std::vector<size_t> measurement_subject;
    std::vector<double> measurement_range;
    std::vector<double> measurement_bearing;
    {
        double time, range, bearing;
        int id;
        measurement_file >> time >> id >> range >> bearing;
        while(!measurement_file.eof())
        {
            if(id == 5 || id ==14 || id == 41 || id == 32 || id == 23 || id == 18 || id == 61 || time - base_time < 0.0){
                measurement_file >> time >> id >> range >> bearing;
                continue;
            }
            measurement_time.push_back(time - base_time);
            measurement_subject.push_back(barcode_map.at(id));
            measurement_range.push_back(range);
            measurement_bearing.push_back(bearing);
            measurement_file >> time >> id >> range >> bearing;
        }
        measurement_file.close();
    }

    /*
    std::vector<double> error_yaw;
    SimpleVehicleModel model;
    for(size_t i=0; i<10000;  ++i) {
        Eigen::Vector3d x = {3.57323240, -3.33283870, 2.34080000};
        {
            int closest_id = 0;
            double min_dev = std::numeric_limits<double>::max();
            const double current_time = odometry_time.at(i);
            for(size_t j=0; j < ground_truth_time.size(); ++j) {
                const double time_dev = std::fabs(current_time - ground_truth_time.at(j));
                if (time_dev < min_dev) {
                    min_dev = time_dev;
                    closest_id = j;
                }
            }
            x = {ground_truth_x.at(closest_id), ground_truth_y.at(closest_id), ground_truth_yaw.at(closest_id)};
        }

        const double dt = odometry_time.at(i+1) - odometry_time.at(i);
        const double odo_time = odometry_time.at(i+1);
        const Eigen::Vector2d input{odometry_v.at(i)*dt, odometry_w.at(i)*dt};
        const Eigen::Vector2d noise{0.0*dt, 0.0*dt};

        x = model.propagate(x, input, noise);
        x(2) = normalizeRadian(x(2));

        size_t closest_id = 0;
        double min_dev = std::numeric_limits<double>::max();
        for(size_t j=0; j < ground_truth_time.size(); ++j) {
            const double time_dev = std::fabs(odo_time - ground_truth_time.at(j));
            if(time_dev < min_dev) {
                min_dev = time_dev;
                closest_id = j;
            }
        }

        const double true_x = ground_truth_x.at(closest_id);
        const double true_y = ground_truth_y.at(closest_id);
        const double true_yaw = ground_truth_yaw.at(closest_id);

        const double current_error_yaw = true_yaw - x(2);
        error_yaw.push_back(current_error_yaw);
    }
    const double mean_yaw_error = std::accumulate(error_yaw.begin(), error_yaw.end(), 0.0) / static_cast<double>(error_yaw.size());
    {
        double sum_cov = 0.0;
        for(size_t i=0; i<error_yaw.size(); ++i) {
            sum_cov += (error_yaw.at(i) - mean_yaw_error) * (error_yaw.at(i) - mean_yaw_error);
        }
        std::cout << "E[error_yaw]: " << mean_yaw_error << std::endl;
        std::cout << "V[error_yaw]: " << normalizeRadian(sum_cov / error_yaw.size()) << std::endl;
    }
    matplotlibcpp::figure_size(1500, 900);
    matplotlibcpp::hist(error_yaw, 50);
    matplotlibcpp::legend();
    matplotlibcpp::title("Error");
    matplotlibcpp::show();
    */

    std::vector<double> error_range;
    std::vector<double> error_square_range;
    std::vector<double> error_bearing;
    std::vector<double> error_meas1;
    std::vector<double> error_meas2;
    for(size_t i=0; i<measurement_time.size(); ++i) {
        const double meas_time = measurement_time.at(i);
        const size_t meas_id = measurement_subject.at(i);
        const double land_x = landmark_map.at(meas_id).x;
        const double land_y = landmark_map.at(meas_id).y;

        // Get the closest ground truth value
        size_t closest_id = 0;
        double min_dev = std::numeric_limits<double>::max();
        for(size_t j=0; j < ground_truth_time.size(); ++j) {
            const double time_dev = std::fabs(meas_time - ground_truth_time.at(j));
            if(time_dev < min_dev) {
                min_dev = time_dev;
                closest_id = j;
            }
        }

        const double true_x = ground_truth_x.at(closest_id);
        const double true_y = ground_truth_y.at(closest_id);
        const double true_yaw = ground_truth_yaw.at(closest_id);

        const double true_range = std::hypot(true_x - land_x, true_y - land_y);
        const double true_bearing = std::atan2(land_y - true_y, land_x - true_x) - true_yaw;

        const double meas_range = measurement_range.at(i);
        const double meas_bearing = measurement_bearing.at(i);

        const double current_error_range = meas_range - true_range;
        const double current_error_bearing = normalizeRadian(meas_bearing - true_bearing);
        const double current_meas1 = meas_range * std::cos(meas_bearing) - (land_x - true_x)*cos(true_yaw) - (land_y - true_y)*std::sin(true_yaw);
        const double current_meas2 = meas_range * std::sin(meas_bearing) - (land_y - true_y)*cos(true_yaw) + (land_x - true_x)*std::sin(true_yaw);
        error_range.push_back(current_error_range);
        error_square_range.push_back(current_error_range*current_error_range);
        error_bearing.push_back(current_error_bearing);
        error_meas1.push_back(current_meas1);
        error_meas2.push_back(current_meas2);
    }
    const double mean_error_range = std::accumulate(error_range.begin(), error_range.end(), 0.0) / static_cast<double>(error_range.size());
    const double mean_error_square_range = std::accumulate(error_square_range.begin(), error_square_range.end(), 0.0) / static_cast<double>(error_square_range.size());
    const double mean_error_bearing = std::accumulate(error_bearing.begin(), error_bearing.end(), 0.0) / static_cast<double>(error_bearing.size());
    const double mean_error_meas1 = std::accumulate(error_meas1.begin(), error_meas1.end(), 0.0) / static_cast<double>(error_meas1.size());
    const double mean_error_meas2 = std::accumulate(error_meas2.begin(), error_meas2.end(), 0.0) / static_cast<double>(error_meas2.size());
    std::cout << "E[error_range]: " << mean_error_range << std::endl;
    std::cout << "E[error_square_range]: " << mean_error_square_range << std::endl;
    std::cout << "E[error_bearing]: " << mean_error_bearing << std::endl;
    std::cout << "E[error_meas1]: " << mean_error_meas1 << std::endl;
    std::cout << "E[error_meas2]: " << mean_error_meas2 << std::endl;
    {
        double sum_cov = 0.0;
        double sum_square_cov = 0.0;
        for(size_t i=0; i<error_range.size(); ++i) {
            sum_cov += (error_range.at(i) - mean_error_range) * (error_range.at(i) - mean_error_range);
            sum_square_cov += (error_square_range.at(i) - mean_error_square_range) * (error_square_range.at(i) - mean_error_square_range);
        }
        std::cout << "V[error_range]: " << sum_cov / error_range.size() << std::endl;
        std::cout << "V[error_suqare_range]: " << sum_square_cov / error_square_range.size() << std::endl;
    }

    {
        double sum_cov = 0.0;
        for(size_t i=0; i<error_bearing.size(); ++i) {
            sum_cov += (error_bearing.at(i) - mean_error_bearing) * (error_bearing.at(i) - mean_error_bearing);
        }
        std::cout << "V[error_bearing]: " << sum_cov / error_bearing.size() << std::endl;
    }

    {
        double sum_cov = 0.0;
        for(size_t i=0; i<error_meas1.size(); ++i) {
            sum_cov += (error_meas1.at(i) - mean_error_meas1) * (error_meas1.at(i) - mean_error_meas1);
        }
        std::cout << "V[error_meas1]: " << sum_cov / error_meas1.size() << std::endl;
    }
    {
        double sum_cov = 0.0;
        for(size_t i=0; i<error_meas2.size(); ++i) {
            sum_cov += (error_meas2.at(i) - mean_error_meas2) * (error_meas2.at(i) - mean_error_meas2);
        }
        std::cout << "V[error_meas2]: " << sum_cov / error_meas2.size() << std::endl;
    }

    matplotlibcpp::figure_size(1500, 900);
    matplotlibcpp::hist(error_meas2, 500);
    matplotlibcpp::legend();
    matplotlibcpp::title("Error");
    matplotlibcpp::show();

    return 0;
}
