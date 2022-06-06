#include "distribution/four_dimensional_normal_distribution.h"

FourDimensionalNormalDistribution::FourDimensionalNormalDistribution(const Eigen::Vector4d& mean,
                                                                     const Eigen::Matrix4d& covariance,
                                                                     const double cov_threshold)
                                                                     : mean_(mean),
                                                                       covariance_(covariance),
                                                                       cov_threshold_(cov_threshold)
{
    if(!checkPositiveDefiniteness(covariance_)) {
        throw std::runtime_error("Possibly non semi-positive definitie matrix!");
    }
}

void FourDimensionalNormalDistribution::setValues(const Eigen::Vector4d &mean, const Eigen::Matrix4d &covariance)
{
    // Set Value
    mean_ = mean;
    covariance_ = covariance;

    if(!checkPositiveDefiniteness(covariance_)) {
        throw std::runtime_error("Possibly non semi-positive definitie matrix!");
    }
}

bool FourDimensionalNormalDistribution::checkPositiveDefiniteness(const Eigen::Matrix4d& covariance)
{
    Eigen::LLT<Eigen::MatrixXd> lltOfA(covariance_); // compute the Cholesky decomposition of A
    if(lltOfA.info() == Eigen::NumericalIssue) {
        return false;
    }

    return true;
}

TwoDimensionalNormalDistribution FourDimensionalNormalDistribution::create2DNormalDistribution(const int dim1, const int dim2)
{
    const Eigen::Vector2d mean = {mean_(dim1), mean_(dim2)};
    Eigen::Matrix2d cov;
    cov << covariance_(dim1, dim1), covariance_(dim1, dim2),
            covariance_(dim2, dim1), covariance_(dim2, dim2);

    return TwoDimensionalNormalDistribution(mean, cov);
}

ThreeDimensionalNormalDistribution FourDimensionalNormalDistribution::create3DNormalDistribution(const int dim1, const int dim2, const int dim3)
{
    const Eigen::Vector3d mean = {mean_(dim1), mean_(dim2), mean_(dim3)};
    Eigen::Matrix3d cov;
    cov << covariance_(dim1, dim1), covariance_(dim1, dim2), covariance_(dim1, dim3),
           covariance_(dim2, dim1), covariance_(dim2, dim2), covariance_(dim2, dim3),
           covariance_(dim3, dim1), covariance_(dim3, dim2), covariance_(dim3, dim3);

    return ThreeDimensionalNormalDistribution(mean, cov);
}

double FourDimensionalNormalDistribution::calc_mean(const int dim)
{
    if(dim > 3) {
        throw std::invalid_argument("Dim is larger than the size of the distribution");
    }
    return mean_(dim);
}

double FourDimensionalNormalDistribution::calc_covariance(const int dim)
{
    if(dim > 3) {
        throw std::invalid_argument("Dim is larger than the size of the distribution");
    }
    return covariance_(dim, dim);
}

double FourDimensionalNormalDistribution::calc_moment(const int dim, const int moment)
{
    if(dim > 3) {
        throw std::invalid_argument("Dim is larger than the size of the distribution");
    }
    NormalDistribution dist(mean_(dim), covariance_(dim, dim));
    return dist.calc_moment(moment);
}

double FourDimensionalNormalDistribution::calc_sin_moment(const int dim ,const int moment)
{
    if(dim > 3) {
        throw std::invalid_argument("Dim is larger than the size of the distribution");
    }
    NormalDistribution dist(mean_(dim), covariance_(dim, dim));
    return dist.calc_sin_moment(moment);
}

double FourDimensionalNormalDistribution::calc_cos_moment(const int dim ,const int moment)
{
    if(dim > 3) {
        throw std::invalid_argument("Dim is larger than the size of the distribution");
    }
    NormalDistribution dist(mean_(dim), covariance_(dim, dim));
    return dist.calc_cos_moment(moment);
}

double FourDimensionalNormalDistribution::calc_cos_sin_moment(const int dim ,const int cos_moment, const int sin_moment)
{
    if(dim > 3) {
        throw std::invalid_argument("Dim is larger than the size of the distribution");
    }
    NormalDistribution dist(mean_(dim), covariance_(dim, dim));
    return dist.calc_cos_sin_moment(cos_moment, sin_moment);
}

double FourDimensionalNormalDistribution::calc_x_sin_x_moment(const int dim, const int moment, const int sin_moment)
{
    if(dim > 3) {
        throw std::invalid_argument("Dim is larger than the size of the distribution");
    }
    NormalDistribution dist(mean_(dim), covariance_(dim, dim));
    return dist.calc_x_sin_moment(moment, sin_moment);
}

double FourDimensionalNormalDistribution::calc_x_cos_x_moment(const int dim, const int moment, const int cos_moment)
{
    if(dim > 3) {
        throw std::invalid_argument("Dim is larger than the size of the distribution");
    }
    NormalDistribution dist(mean_(dim), covariance_(dim, dim));
    return dist.calc_x_cos_moment(moment, cos_moment);
}

double FourDimensionalNormalDistribution::calc_cross_second_moment(const int dim1, const int dim2)
{
    auto dist = create2DNormalDistribution(dim1, dim2);
    return dist.calc_xy_moment();
}

double FourDimensionalNormalDistribution::calc_x_sin_z_moment(const int dim_x, const int dim_z)
{
    if(dim_x > 3 || dim_z > 3) {
        throw std::invalid_argument("dim_x or dim_z is larger than the size of the distribution");
    }

    auto dist = create2DNormalDistribution(dim_x, dim_z);
    return dist.calc_x_sin_y_moment();
}

double FourDimensionalNormalDistribution::calc_x_cos_z_moment(const int dim_x, const int dim_z)
{
    if(dim_x > 3 || dim_z > 3) {
        throw std::invalid_argument("dim_x or dim_z is larger than the size of the distribution");
    }

    auto dist = create2DNormalDistribution(dim_x, dim_z);
    return dist.calc_x_cos_y_moment();
}

double FourDimensionalNormalDistribution::calc_x_cos_y_cos_y_moment(const int dim_x, const int dim_y)
{
    if(dim_x > 3 || dim_y > 3) {
        throw std::invalid_argument("dim_x or dim_z is larger than the size of the distribution");
    }

    auto dist = create2DNormalDistribution(dim_x, dim_y);
    return dist.calc_x_cos_y_cos_y_moment();
}

double FourDimensionalNormalDistribution::calc_x_sin_y_sin_y_moment(const int dim_x, const int dim_y)
{
    if(dim_x > 3 || dim_y > 3) {
        throw std::invalid_argument("dim_x or dim_z is larger than the size of the distribution");
    }

    auto dist = create2DNormalDistribution(dim_x, dim_y);
    return dist.calc_x_sin_y_sin_y_moment();
}

double FourDimensionalNormalDistribution::calc_x_cos_y_sin_y_moment(const int dim_x, const int dim_y)
{
    if(dim_x > 3 || dim_y > 3) {
        throw std::invalid_argument("dim_x or dim_z is larger than the size of the distribution");
    }

    auto dist = create2DNormalDistribution(dim_x, dim_y);
    return dist.calc_x_cos_y_sin_y_moment();
}

double FourDimensionalNormalDistribution::calc_xy_sin_y_moment(const int dim_x, const int dim_y)
{
    if(dim_x > 3 || dim_y > 3) {
        throw std::invalid_argument("dim_x or dim_z is larger than the size of the distribution");
    }

    auto dist = create2DNormalDistribution(dim_x, dim_y);
    return dist.calc_x_y_sin_y_moment();
}

double FourDimensionalNormalDistribution::calc_xy_cos_y_moment(const int dim_x, const int dim_y)
{
    if(dim_x > 3 || dim_y > 3) {
        throw std::invalid_argument("dim_x or dim_z is larger than the size of the distribution");
    }

    auto dist = create2DNormalDistribution(dim_x, dim_y);
    return dist.calc_x_y_cos_y_moment();
}

double FourDimensionalNormalDistribution::calc_cross_third_moment(const int dim1, const int dim2, const int moment1, const int moment2)
{
    if(moment1 + moment2 != 3) {
        throw std::invalid_argument("moment1 + moment2 is not 3");
    }

    if(dim1 > 3 || dim2 > 3) {
        throw std::invalid_argument("dim1 or dim2 is over 3");
    }

    auto dist = create2DNormalDistribution(dim1, dim2);

    return dist.calc_third_moment(moment1, moment2);
}

double FourDimensionalNormalDistribution::calc_xx_sin_z_moment(const int dim_x, const int dim_z)
{
    if(dim_x > 3 || dim_z > 3) {
        throw std::invalid_argument("dim1 or dim2 is over 3");
    }

    auto dist = create2DNormalDistribution(dim_x, dim_z);
    return dist.calc_xx_sin_y_moment();
}

double FourDimensionalNormalDistribution::calc_xx_cos_z_moment(const int dim_x, const int dim_z)
{
    if(dim_x > 3 || dim_z > 3) {
        throw std::invalid_argument("dim1 or dim2 is over 3");
    }

    auto dist = create2DNormalDistribution(dim_x, dim_z);
    return dist.calc_xx_cos_y_moment();
}

double FourDimensionalNormalDistribution::calc_xy_cos_z_moment(const int dim_x, const int dim_y, const int dim_z)
{
    if(dim_x > 3 || dim_y > 3 || dim_z > 3) {
        throw std::invalid_argument("dim1, dim2 or dim3 is over 3");
    }

    auto dist = create3DNormalDistribution(dim_x, dim_y, dim_z);
    return dist.calc_xy_cos_z_moment();
}

double FourDimensionalNormalDistribution::calc_xy_sin_z_moment(const int dim_x, const int dim_y, const int dim_z)
{
    if(dim_x > 3 || dim_y > 3 || dim_z > 3) {
        throw std::invalid_argument("dim1, dim2 or dim3 is over 3");
    }

    auto dist = create3DNormalDistribution(dim_x, dim_y, dim_z);
    return dist.calc_xy_sin_z_moment();
}

double FourDimensionalNormalDistribution::calc_xxyy_moment(const int dim_x, const int dim_y)
{
    if(dim_x > 3 || dim_y > 3) {
        throw std::invalid_argument("dim1 or dim2 is over 3");
    }

    auto dist = create2DNormalDistribution(dim_x, dim_y);
    return dist.calc_xxyy_moment();
}

double FourDimensionalNormalDistribution::calc_xx_cos_y_cos_y_moment(const int dim_x, const int dim_y)
{
    if(dim_x > 3 || dim_y > 3) {
        throw std::invalid_argument("dim1 or dim2 is over 3");
    }

    auto dist = create2DNormalDistribution(dim_x, dim_y);
    return dist.calc_xx_cos_y_cos_y_moment();
}

double FourDimensionalNormalDistribution::calc_xx_sin_y_sin_y_moment(const int dim_x, const int dim_y)
{
    if(dim_x > 3 || dim_y > 3) {
        throw std::invalid_argument("dim1 or dim2 is over 3");
    }

    auto dist = create2DNormalDistribution(dim_x, dim_y);
    return dist.calc_xx_sin_y_sin_y_moment();
}

double FourDimensionalNormalDistribution::calc_xx_cos_y_sin_y_moment(const int dim_x, const int dim_y)
{
    if(dim_x > 3 || dim_y > 3) {
        throw std::invalid_argument("dim1 or dim2 is over 3");
    }

    auto dist = create2DNormalDistribution(dim_x, dim_y);
    return dist.calc_xx_cos_y_sin_y_moment();
}

double FourDimensionalNormalDistribution::calc_xxy_cos_z_moment(const int dim_x, const int dim_y, const int dim_z)
{
    if(dim_x > 3 || dim_y > 3 || dim_z > 3) {
        throw std::invalid_argument("dim1, dim2 or dim3 is over 3");
    }

    auto dist = create3DNormalDistribution(dim_x, dim_y, dim_z);
    return dist.calc_xxy_cos_z_moment();
}
