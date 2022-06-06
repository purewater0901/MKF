#include "distribution/exponential_distribution.h"

ExponentialDistribution::ExponentialDistribution(const double lambda) : lambda_(lambda)
{
}

double ExponentialDistribution::calc_mean()
{
    return 1.0/lambda_;
}

double ExponentialDistribution::calc_variance()
{
    return  1.0/(lambda_*lambda_);
}

std::complex<double> ExponentialDistribution::calc_characteristic(const int t)
{
    const std::complex<double> i(0.0, 1.0);
    const auto t_double = static_cast<double>(t);
    return lambda_ / (lambda_ - i *t_double);
}

std::complex<double> ExponentialDistribution::calc_first_diff_characteristic(const int t)
{
    const std::complex<double> i(0.0, 1.0);
    const auto t_double = static_cast<double>(t);
    const auto tmp = lambda_ - i * t_double;
    return i*lambda_ / (tmp*tmp);
}

std::complex<double> ExponentialDistribution::calc_second_diff_characteristic(const int t)
{
    const std::complex<double> i(0.0, 1.0);
    const auto t_double = static_cast<double>(t);
    const auto tmp = lambda_ - i * t_double;
    return -2.0*lambda_ / (tmp*tmp*tmp);
}

std::complex<double> ExponentialDistribution::calc_third_diff_characteristic(const int t)
{
    const std::complex<double> i(0.0, 1.0);
    const auto t_double = static_cast<double>(t);
    const auto tmp = lambda_ - i * t_double;
    return -6.0*i*lambda_ / (tmp*tmp*tmp*tmp);
}

std::complex<double> ExponentialDistribution::calc_fourth_diff_characteristic(const int t)
{
    const std::complex<double> i(0.0, 1.0);
    const auto t_double = static_cast<double>(t);
    const auto tmp = lambda_ - i * t_double;
    return 24*lambda_/std::pow(tmp, 5);
}
