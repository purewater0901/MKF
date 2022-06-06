#ifndef UNCERTAINTY_PROPAGATION_PAPER_EXAMPLE_2D_UKF_H
#define UNCERTAINTY_PROPAGATION_PAPER_EXAMPLE_2D_UKF_H

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <Eigen/Eigen>

class PaperExample2DUKF
{
public:
    PaperExample2DUKF(const Eigen::Vector2d& mean, const Eigen::Matrix2d& cov);

    double predict(const std::string& type);

    const Eigen::Vector2d mean_;
    const Eigen::Matrix2d cov_;

    const int size_{2};
    const double alpha_squared_{1.0};
    const double beta_{0.0};
    const double kappa_{0.0};
    const double lambda_;

    double Sigma_WM0_;
    double Sigma_WC0_;
    double Sigma_WMI_;
    double Sigma_WCI_;
};

#endif //UNCERTAINTY_PROPAGATION_PAPER_EXAMPLE_2D_UKF_H
