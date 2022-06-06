#ifndef UNCERTAINTY_PROPAGATION_PAPER_EXAMPLE_3D_UKF_H
#define UNCERTAINTY_PROPAGATION_PAPER_EXAMPLE_3D_UKF_H

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <Eigen/Eigen>

class PaperExample3DUKF
{
public:
    PaperExample3DUKF(const Eigen::Vector3d& mean, const Eigen::Matrix3d& cov);

    double predict(const std::string& type);

    const Eigen::Vector3d mean_;
    const Eigen::Matrix3d cov_;

    const int size_{3};
    const double alpha_squared_{1.0};
    const double beta_{0.0};
    const double kappa_{0.0};
    const double lambda_;

    double Sigma_WM0_;
    double Sigma_WC0_;
    double Sigma_WMI_;
    double Sigma_WCI_;
};

#endif //UNCERTAINTY_PROPAGATION_PAPER_EXAMPLE_3D_UKF_H
