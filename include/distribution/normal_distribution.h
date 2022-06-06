#ifndef UNCERTAINTY_PROPAGATION_NORMAL_DISTRIBUTION_H
#define UNCERTAINTY_PROPAGATION_NORMAL_DISTRIBUTION_H

#include <iostream>
#include <vector>
#include <Eigen/Eigen>
#include <complex>

#include "distribution/base_distribution.h"
#include "utilities.h"

class NormalDistribution : public BaseDistribution{
public:
    NormalDistribution() : mean_(0.0), variance_(1.0) {}
    NormalDistribution(const double mean, const double variance);

    double calc_mean();
    double calc_variance();

    std::complex<double> calc_characteristic(const int t);
    std::complex<double> calc_first_diff_characteristic(const int t);
    std::complex<double> calc_second_diff_characteristic(const int t);
    std::complex<double> calc_third_diff_characteristic(const int t);
    std::complex<double> calc_fourth_diff_characteristic(const int t);

    double mean_;
    double variance_;

private:
};

#endif //UNCERTAINTY_PROPAGATION_NORMAL_DISTRIBUTION_H
