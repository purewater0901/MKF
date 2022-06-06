#ifndef UNCERTAINTY_PROPAGATION_FOUR_DIMENSIONAL_NORMAL_DISTRIBUTION_H
#define UNCERTAINTY_PROPAGATION_FOUR_DIMENSIONAL_NORMAL_DISTRIBUTION_H

#include <iostream>
#include <vector>
#include <Eigen/Eigen>
#include <complex>

#include "distribution/normal_distribution.h"
#include "distribution/two_dimensional_normal_distribution.h"
#include "distribution/three_dimensional_normal_distribution.h"
#include "utilities.h"

class FourDimensionalNormalDistribution
{
public:
    FourDimensionalNormalDistribution(const double cov_threshold=1e-6) : cov_threshold_(cov_threshold) {};
    FourDimensionalNormalDistribution(const Eigen::Vector4d& mean,
                                      const Eigen::Matrix4d& covariance,
                                      const double cov_threshold=1e-6);

    bool checkPositiveDefiniteness(const Eigen::Matrix4d& covariance);

    void setValues(const Eigen::Vector4d& mean, const Eigen::Matrix4d& covariance);

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
    double calc_x_cos_y_cos_y_moment(const int dim_x, const int dim_y);
    double calc_x_sin_y_sin_y_moment(const int dim_x, const int dim_y);
    double calc_x_cos_y_sin_y_moment(const int dim_x, const int dim_y);
    double calc_xy_sin_y_moment(const int dim_x, const int dim_y);
    double calc_xy_cos_y_moment(const int dim_x, const int dim_y);

    double calc_cross_third_moment(const int dim1, const int dim2, const int moment1, const int moment2);
    double calc_xx_sin_z_moment(const int dim_x, const int dim_z);
    double calc_xx_cos_z_moment(const int dim_x, const int dim_z);

    double calc_xxyy_moment(const int dim_x, const int dim_y);
    double calc_xx_cos_y_cos_y_moment(const int dim_x, const int dim_y);
    double calc_xx_sin_y_sin_y_moment(const int dim_x, const int dim_y);
    double calc_xx_cos_y_sin_y_moment(const int dim_x, const int dim_y);

    double calc_xy_cos_z_moment(const int dim_x, const int dim_y, const int dim_z);
    double calc_xy_sin_z_moment(const int dim_x, const int dim_y, const int dim_z);
    double calc_xxy_cos_z_moment(const int dim_x, const int dim_y, const int dim_z);

    TwoDimensionalNormalDistribution create2DNormalDistribution(const int dim1, const int dim2);
    ThreeDimensionalNormalDistribution create3DNormalDistribution(const int dim1, const int dim2, const int dim3);

    Eigen::Vector4d mean_;
    Eigen::Matrix4d covariance_;

    const double cov_threshold_{1e-6};
};

#endif //UNCERTAINTY_PROPAGATION_FOUR_DIMENSIONAL_NORMAL_DISTRIBUTION_H
