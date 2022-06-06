#ifndef UNCERTAINTY_PROPAGATION_THREE_DIMENSIONAL_NORMAL_DISTRIBUTION_H
#define UNCERTAINTY_PROPAGATION_THREE_DIMENSIONAL_NORMAL_DISTRIBUTION_H

#include <iostream>
#include <vector>
#include <Eigen/Eigen>
#include <complex>

#include "distribution/normal_distribution.h"
#include "distribution/two_dimensional_normal_distribution.h"
#include "utilities.h"

class ThreeDimensionalNormalDistribution
{
public:
    ThreeDimensionalNormalDistribution(const double cov_threshold=1e-6) : cov_threshold_(cov_threshold) {};
    ThreeDimensionalNormalDistribution(const Eigen::Vector3d& mean,
                                       const Eigen::Matrix3d& covariance,
                                       const double cov_threshold=1e-6);

    bool checkPositiveDefiniteness(const Eigen::Matrix3d& covariance);
    bool initializeData();

    void setValues(const Eigen::Vector3d& mean, const Eigen::Matrix3d& covariance);

    TwoDimensionalNormalDistribution create2DNormalDistribution(const int dim1, const int dim2);

    double calc_mean(const int dim);
    double calc_covariance(const int dim);
    double calc_moment(const int dim ,const int moment);
    double calc_sin_moment(const int dim ,const int moment);
    double calc_cos_moment(const int dim ,const int moment);
    double calc_cos_sin_moment(const int dim ,const int cos_moment, const int sin_moment);
    double calc_x_sin_x_moment(const int dim, const int moment, const int sin_moment);
    double calc_x_cos_x_moment(const int dim, const int moment, const int cos_moment);

    double calc_cross_second_moment(const int dim1, const int dim2);
    double calc_x_sin_z_moment(const int dim_x, const int dim_z);
    double calc_x_cos_z_moment(const int dim_x, const int dim_z);

    double calc_cross_third_moment(const int dim1, const int dim2, const int moment1, const int moment2);
    double calc_xx_sin_z_moment(const int dim_x, const int dim_z);
    double calc_xx_cos_z_moment(const int dim_x, const int dim_z);
    double calc_xy_cos_z_moment();
    double calc_xy_sin_z_moment();
    double calc_xy_cos_y_moment(const int dim_x, const int dim_y);
    double calc_xy_sin_y_moment(const int dim_x, const int dim_y);
    double calc_x_cos_z_cos_z_moment(const int dim_x, const int dim_z);
    double calc_x_sin_z_sin_z_moment(const int dim_x, const int dim_z);
    double calc_x_cos_z_sin_z_moment(const int dim_x, const int dim_z);

    double calc_xxyy_moment(const int dim_x, const int dim_y);
    double calc_xx_cos_z_cos_z_moment(const int dim_x, const int dim_z);
    double calc_xx_sin_z_sin_z_moment(const int dim_x, const int dim_z);
    double calc_xx_cos_z_sin_z_moment(const int dim_x, const int dim_z);
    double calc_xxy_cos_z_moment();
    double calc_xy_cos_z_cos_z_moment();
    double calc_xy_sin_z_sin_z_moment();
    double calc_xy_cos_z_sin_z_moment();

    Eigen::Vector3d mean_;
    Eigen::Matrix3d covariance_;

    Eigen::Vector3d eigen_values_;
    Eigen::Matrix3d T_;

    const double cov_threshold_{1e-6};
    bool initialization_ = false;
};

#endif //UNCERTAINTY_PROPAGATION_THREE_DIMENSIONAL_NORMAL_DISTRIBUTION_H
