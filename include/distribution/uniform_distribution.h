#ifndef UNCERTAINTY_PROPAGATION_UNIFORM_DISTRIBUTION_H
#define UNCERTAINTY_PROPAGATION_UNIFORM_DISTRIBUTION_H

#include <iostream>
#include <vector>
#include <Eigen/Eigen>
#include <complex>

#include "distribution/base_distribution.h"
#include "utilities.h"

class UniformDistribution : public BaseDistribution{
public:
    UniformDistribution(const double l, const double u);

    double calc_mean();
    double calc_variance();

    std::complex<double> calc_characteristic(const int t);
    std::complex<double> calc_first_diff_characteristic(const int t);
    std::complex<double> calc_second_diff_characteristic(const int t);
    std::complex<double> calc_third_diff_characteristic(const int t);
    std::complex<double> calc_fourth_diff_characteristic(const int t);

private:
    const double l_;
    const double u_;
};

#endif //UNCERTAINTY_PROPAGATION_UNIFORM_DISTRIBUTION_H
