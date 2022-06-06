#ifndef UNCERTAINTY_PROPAGATION_UTILITIES_H
#define UNCERTAINTY_PROPAGATION_UTILITIES_H

#include <iostream>
#include <vector>
#include <Eigen/Eigen>
#include <complex>
#include <cmath>

int fact(const int n);

int nCr(const int n, const int r);

inline double normalizeRadian(const double rad, const double min_rad = -M_PI)
{
    const auto max_rad = min_rad + 2 * M_PI;

    const auto value = std::fmod(rad, 2 * M_PI);
    if (min_rad <= value && value < max_rad) {
        return value;
    }

    return value - std::copysign(2 * M_PI, value);
}

void outputResultToFile(const std::string& filename, const std::vector<double>& time,
                        const std::vector<double>& x_true, const std::vector<double>& y_true, const std::vector<double>& v_true, const std::vector<double>& yaw_true,
                        const std::vector<double>& nkf_x, const std::vector<double>& nkf_y, const std::vector<double>& nkf_v, const std::vector<double>& nkf_yaw,
                        const std::vector<double>& ekf_x, const std::vector<double>& ekf_y, const std::vector<double>& ekf_v, const std::vector<double>& ekf_yaw,
                        const std::vector<double>& ukf_x, const std::vector<double>& ukf_y, const std::vector<double>& ukf_v, const std::vector<double>& ukf_yaw,
                        const std::vector<double>& nkf_xy_errors, const std::vector<double>& nkf_v_errors, const std::vector<double>& nkf_yaw_errors,
                        const std::vector<double>& ekf_xy_errors, const std::vector<double>& ekf_v_errors, const std::vector<double>& ekf_yaw_errors,
                        const std::vector<double>& ukf_xy_errors, const std::vector<double>& ukf_v_errors, const std::vector<double>& ukf_yaw_errors);

void outputResultToFile(const std::string& filename, const std::vector<double>& time,
                        const std::vector<double>& x_true, const std::vector<double>& y_true, const std::vector<double>& yaw_true,
                        const std::vector<double>& nkf_x, const std::vector<double>& nkf_y, const std::vector<double>& nkf_yaw,
                        const std::vector<double>& ekf_x, const std::vector<double>& ekf_y, const std::vector<double>& ekf_yaw,
                        const std::vector<double>& ukf_x, const std::vector<double>& ukf_y, const std::vector<double>& ukf_yaw,
                        const std::vector<double>& nkf_xy_errors, const std::vector<double>& nkf_yaw_errors,
                        const std::vector<double>& ekf_xy_errors, const std::vector<double>& ekf_yaw_errors,
                        const std::vector<double>& ukf_xy_errors, const std::vector<double>& ukf_yaw_errors);

#endif //UNCERTAINTY_PROPAGATION_UTILITIES_H
