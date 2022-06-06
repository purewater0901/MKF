#include "distribution/uniform_distribution.h"

UniformDistribution::UniformDistribution(const double l, const double u) : BaseDistribution(), l_(l), u_(u) {}

double UniformDistribution::calc_mean()
{
    return 0.5 * (l_ + u_);
}

double UniformDistribution::calc_variance()
{
    return  (u_ - l_) * (u_ - l_) / 12.0;
}

std::complex<double> UniformDistribution::calc_characteristic(const int t)
{
    if(t == 0)
    {
        return {1.0, 0.0};
    }

    const std::complex<double> i(0.0, 1.0);
    const auto t_double = static_cast<double>(t);
    return (std::exp(i*t_double*u_) - std::exp(i*t_double*l_))/ (i*t_double*(u_-l_));
}

std::complex<double> UniformDistribution::calc_first_diff_characteristic(const int t)
{
    if(t == 0)
    {
        return {0.0, (u_+l_)/2.0};
    }

    const std::complex<double> i(0.0, 1.0);
    const auto t_double = static_cast<double>(t);
    return (u_*std::exp(i*t_double*u_) - l_*std::exp(i*t_double*l_)) / (t_double*(u_-l_)) - calc_characteristic(t) / t_double;
}

std::complex<double> UniformDistribution::calc_second_diff_characteristic(const int t)
{
    if(t == 0)
    {
        return {-(u_*u_+l_*l_+u_*l_)/3.0, 0.0};
    }

    const std::complex<double> i(0.0, 1.0);
    const auto t_double = static_cast<double>(t);

    return (i*(u_*u_*std::exp(i*t_double*u_) - l_*l_*std::exp(i*t_double*l_)))/(t_double*(u_-l_)) - 2.0 * calc_first_diff_characteristic(t) / t_double;
}

std::complex<double> UniformDistribution::calc_third_diff_characteristic(const int t)
{
    if(t == 0)
    {
        return {0.0, -(u_+l_)*(u_*u_+l_*l_)/4.0};
    }

    const std::complex<double> i(0.0, 1.0);
    const auto t_double = static_cast<double>(t);

    return (-std::pow(u_, 3)*std::exp(i*t_double*u_) + std::pow(l_, 3)*std::exp(i*t_double*l_))/(t_double*(u_-l_)) - 3.0 * calc_second_diff_characteristic(t) / t_double;
}

std::complex<double> UniformDistribution::calc_fourth_diff_characteristic(const int t)
{
    if(t == 0)
    {
        return {(std::pow(u_, 5) - std::pow(l_, 5))/(5.0*(u_-l_)), 0.0};
    }

    const std::complex<double> i(0.0, 1.0);
    const auto t_double = static_cast<double>(t);

    return (i*(-std::pow(u_, 4)*std::exp(i*t_double*u_) + std::pow(l_, 4)*std::exp(i*t_double*l_)))/(t_double*(u_-l_)) - 4.0 * calc_third_diff_characteristic(t) / t_double;
}
