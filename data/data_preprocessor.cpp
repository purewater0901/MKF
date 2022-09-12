#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Eigen>
#include <fstream>
#include <algorithm>

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

inline double normalizeRadian(const double rad, const double min_rad = -M_PI)
{
    const auto max_rad = min_rad + 2 * M_PI;

    const auto value = std::fmod(rad, 2 * M_PI);
    if (min_rad <= value && value < max_rad) {
        return value;
    }

    return value - std::copysign(2 * M_PI, value);
}

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
    const std::string odometry_filename = "/home/yutaka/CLionProjects/uncertainty_propagation/data/MRCLAM_Dataset1/Robot" + std::to_string(robot_num) + "_Odometry.dat";
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

    const std::string ground_truth_filename = "/home/yutaka/CLionProjects/uncertainty_propagation/data/MRCLAM_Dataset1/Robot" + std::to_string(robot_num) + "_Groundtruth.dat";
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

    const std::string measurement_filename = "/home/yutaka/CLionProjects/uncertainty_propagation/data/MRCLAM_Dataset1/Robot" + std::to_string(robot_num) + "_Measurement_updated.dat";
    std::ifstream measurement_file(measurement_filename);
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
            if(id == 5 || id ==14 || id == 41 || id == 32 || id == 23 || time - base_time < 0.0){
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

    std::vector<double> ground_truth_range;
    std::vector<double> ground_truth_bearing;
    ground_truth_range.reserve(odometry_time.size());
    ground_truth_bearing.reserve(odometry_time.size());
    for(size_t meas_id = 0; meas_id < measurement_time.size(); ++meas_id)
    {
        size_t closest_id = 0;
        double min_dev = std::numeric_limits<double>::max();
        const double current_time = measurement_time.at(meas_id);
        for(size_t j=0; j < ground_truth_time.size(); ++j)
        {
            const double time_dev = std::fabs(current_time - ground_truth_time.at(j));
            if (time_dev < min_dev) {
                min_dev = time_dev;
                closest_id = j;
            }
        }

        const double x_true = ground_truth_x.at(closest_id);
        const double y_true = ground_truth_y.at(closest_id);
        const double yaw_true = ground_truth_yaw.at(closest_id);
        const auto landmark = landmark_map.at(measurement_subject.at(meas_id));
        const double x_land = landmark.x;
        const double y_land = landmark.y;

        // compute true range
        const double r_true = std::hypot(x_land - x_true, y_land - y_true);
        const double bearing_true = normalizeRadian(std::atan2(y_land - y_true, x_land - x_true) - yaw_true);
        ground_truth_range.push_back(r_true);
        ground_truth_bearing.push_back(bearing_true);
    }

    const std::string measurement_updated_filename = "/home/yutaka/CLionProjects/uncertainty_propagation/data/MRCLAM_Dataset1/Robot" + std::to_string(robot_num) + "_Measurement_updated.dat";
    std::ofstream  measurement_updated_file(measurement_updated_filename);
    for(size_t i=0; i<measurement_time.size(); ++i) {
        measurement_updated_file << measurement_time.at(i) << " " << measurement_subject.at(i) << " "
                                 << measurement_range.at(i) << " " << measurement_bearing.at(i) << " "
                                 << ground_truth_range.at(i) << " " << ground_truth_bearing.at(i) << std::endl;
    }

    return 0;
}