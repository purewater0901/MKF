#include "utilities.h"
#include <fstream>

int fact(const int n)
{
    int res = 1;
    for(int i=2; i<=n; ++i)
        res *= i;

    return res;
}

int nCr(const int n, const int r)
{
    return fact(n) / (fact(r) * fact(n - r));
}

void outputResultToFile(const std::string& filename, const std::vector<double>& time,
                        const std::vector<double>& x_true, const std::vector<double>& y_true, const std::vector<double>& v_true, const std::vector<double>& yaw_true,
                        const std::vector<double>& nkf_x, const std::vector<double>& nkf_y, const std::vector<double>& nkf_v, const std::vector<double>& nkf_yaw,
                        const std::vector<double>& ekf_x, const std::vector<double>& ekf_y, const std::vector<double>& ekf_v, const std::vector<double>& ekf_yaw,
                        const std::vector<double>& ukf_x, const std::vector<double>& ukf_y, const std::vector<double>& ukf_v, const std::vector<double>& ukf_yaw,
                        const std::vector<double>& nkf_xy_errors, const std::vector<double>& nkf_v_errors, const std::vector<double>& nkf_yaw_errors,
                        const std::vector<double>& ekf_xy_errors, const std::vector<double>& ekf_v_errors, const std::vector<double>& ekf_yaw_errors,
                        const std::vector<double>& ukf_xy_errors, const std::vector<double>& ukf_v_errors, const std::vector<double>& ukf_yaw_errors)
{
    std::ofstream writing_file;
    writing_file.open(filename, std::ios::out);
    writing_file << "time," << "x_true," << "y_true," << "v_true," << "yaw_true,"
                 << "nkf_x," << "nkf_y," << "nkf_v," << "nkf_yaw,"
                 << "ekf_x," << "ekf_y," << "ekf_v," << "ekf_yaw,"
                 << "ukf_x," << "ukf_y," << "ukf_v," << "ukf_yaw,"
                 << "nkf_xy_error," << "nkf_v_error," << "nkf_yaw_error,"
                 << "ekf_xy_error," << "ekf_v_error," << "ekf_yaw_error,"
                 << "ukf_xy_error," << "ukf_v_error," << "ukf_yaw_error" << std::endl;
    for(size_t i=0; i<nkf_xy_errors.size(); ++i) {
        writing_file << time.at(i) << ","
                     << x_true.at(i) << "," << y_true.at(i) << "," << v_true.at(i) << "," << yaw_true.at(i) << ","
                     << nkf_x.at(i) << "," << nkf_y.at(i) << "," << nkf_v.at(i) << "," << nkf_yaw.at(i) << ","
                     << ekf_x.at(i) << "," << ekf_y.at(i) << "," << ekf_v.at(i) << "," << ekf_yaw.at(i) << ","
                     << ukf_x.at(i) << "," << ukf_y.at(i) << "," << ukf_v.at(i) << "," << ukf_yaw.at(i) << ","
                     << nkf_xy_errors.at(i) << "," << nkf_v_errors.at(i) << "," << nkf_yaw_errors.at(i) << ","
                     << ekf_xy_errors.at(i) << "," << ekf_v_errors.at(i) << "," << ekf_yaw_errors.at(i) << ","
                     << ukf_xy_errors.at(i) << "," << ukf_v_errors.at(i) << "," << ukf_yaw_errors.at(i) << std::endl;
    }
    writing_file.close();
}

void outputResultToFile(const std::string& filename, const std::vector<double>& time,
                        const std::vector<double>& x_true, const std::vector<double>& y_true, const std::vector<double>& yaw_true,
                        const std::vector<double>& nkf_x, const std::vector<double>& nkf_y, const std::vector<double>& nkf_yaw,
                        const std::vector<double>& ekf_x, const std::vector<double>& ekf_y, const std::vector<double>& ekf_yaw,
                        const std::vector<double>& ukf_x, const std::vector<double>& ukf_y, const std::vector<double>& ukf_yaw,
                        const std::vector<double>& nkf_xy_errors, const std::vector<double>& nkf_yaw_errors,
                        const std::vector<double>& ekf_xy_errors, const std::vector<double>& ekf_yaw_errors,
                        const std::vector<double>& ukf_xy_errors, const std::vector<double>& ukf_yaw_errors)
{
    std::ofstream writing_file;
    writing_file.open(filename, std::ios::out);
    writing_file << "time," << "x_true," << "y_true," << "yaw_true,"
                 << "nkf_x," << "nkf_y," << "nkf_yaw,"
                 << "ekf_x," << "ekf_y," << "ekf_yaw,"
                 << "ukf_x," << "ukf_y," << "ukf_yaw,"
                 << "nkf_xy_error," << "nkf_yaw_error,"
                 << "ekf_xy_error," << "ekf_yaw_error,"
                 << "ukf_xy_error," << "ukf_yaw_error" << std::endl;
    for(size_t i=0; i<nkf_xy_errors.size(); ++i) {
        writing_file << time.at(i) << ","
                     << x_true.at(i) << "," << y_true.at(i) << "," << yaw_true.at(i) << ","
                     << nkf_x.at(i) << "," << nkf_y.at(i) << "," << nkf_yaw.at(i) << ","
                     << ekf_x.at(i) << "," << ekf_y.at(i) << "," << ekf_yaw.at(i) << ","
                     << ukf_x.at(i) << "," << ukf_y.at(i) << "," << ukf_yaw.at(i) << ","
                     << nkf_xy_errors.at(i) << "," << nkf_yaw_errors.at(i) << ","
                     << ekf_xy_errors.at(i) << "," << ekf_yaw_errors.at(i) << ","
                     << ukf_xy_errors.at(i) << "," << ukf_yaw_errors.at(i) << std::endl;
    }
    writing_file.close();
}