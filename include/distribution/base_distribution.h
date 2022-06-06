#ifndef UNCERTAINTY_PROPAGATION_BASE_DISTRIBUTION_H
#define UNCERTAINTY_PROPAGATION_BASE_DISTRIBUTION_H

#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Eigen>

class BaseDistribution{
public:
    BaseDistribution() = default;

    virtual double calc_mean() = 0;
    virtual double calc_variance() = 0;

    virtual std::complex<double> calc_characteristic(const int t) = 0;
    virtual std::complex<double> calc_first_diff_characteristic(const int t) = 0;
    virtual std::complex<double> calc_second_diff_characteristic(const int t) = 0;
    virtual std::complex<double> calc_third_diff_characteristic(const int t) = 0;
    virtual std::complex<double> calc_fourth_diff_characteristic(const int t) = 0;

    double calc_moment(const int order);
    double calc_cos_moment(const int order);
    double calc_sin_moment(const int order);
    double calc_cos_sin_moment(const int cos_order, const int sin_order);
    double calc_x_cos_moment(const int x_order, const int cos_order);
    double calc_x_sin_moment(const int x_order, const int sin_order);
    double calc_x_cos_sin_moment(const int x_order, const int cos_order, const int sin_order);
};

#endif //UNCERTAINTY_PROPAGATION_BASE_DISTRIBUTION_H
