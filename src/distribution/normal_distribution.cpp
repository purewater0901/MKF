#include "distribution/normal_distribution.h"

NormalDistribution::NormalDistribution(const double mean, const double variance) : mean_(mean), variance_(variance)
{
}

double NormalDistribution::calc_mean()
{
    return mean_;
}

double NormalDistribution::calc_variance()
{
    return  variance_;
}

std::complex<double> NormalDistribution::calc_characteristic(const int t)
{
    if(t == 0)
    {
        return {1.0, 0.0};
    }

    const std::complex<double> i(0.0, 1.0);
    const auto t_double = static_cast<double>(t);
    return std::exp(i*t_double*mean_ - variance_*std::pow(t_double, 2)*0.5);
}

std::complex<double> NormalDistribution::calc_first_diff_characteristic(const int t)
{
    if(t == 0)
    {
        return {0.0, mean_};
    }

    const std::complex<double> i(0.0, 1.0);
    const auto t_double = static_cast<double>(t);
    const auto tmp = i*mean_ - variance_*t_double;
    return tmp * calc_characteristic(t);
}

std::complex<double> NormalDistribution::calc_second_diff_characteristic(const int t)
{
    if(t == 0)
    {
        return {-variance_ - mean_*mean_, 0.0};
    }

    const std::complex<double> i(0.0, 1.0);
    const auto t_double = static_cast<double>(t);
    const auto tmp = (i*mean_ - variance_*t_double);
    return -variance_ * calc_characteristic(t) + tmp * calc_first_diff_characteristic(t);
}

std::complex<double> NormalDistribution::calc_third_diff_characteristic(const int t)
{
    const std::complex<double> i(0.0, 1.0);
    const auto t_double = static_cast<double>(t);
    const auto tmp = (i*mean_ - variance_*t_double);
    const auto exp_term = std::exp(i*t_double*mean_ - variance_*std::pow(t_double, 2)*0.5);
    return -3.0 * variance_ * tmp * exp_term + tmp*tmp*tmp*exp_term;
}

std::complex<double> NormalDistribution::calc_fourth_diff_characteristic(const int t)
{
    const std::complex<double> i(0.0, 1.0);
    const auto t_double = static_cast<double>(t);
    const auto tmp = (i*mean_ - variance_*t_double);
    const auto exp_term = std::exp(i*t_double*mean_ - variance_*std::pow(t_double, 2)*0.5);
    return 3.0 * std::pow(variance_, 2) * exp_term - 6.0 * variance_ * tmp * tmp * exp_term + std::pow(tmp, 4) * exp_term;
}
