#include "distribution/three_dimensional_normal_distribution.h"
#include "distribution/two_dimensional_normal_distribution.h"

ThreeDimensionalNormalDistribution::ThreeDimensionalNormalDistribution(const Eigen::Vector3d& mean,
                                                                       const Eigen::Matrix3d& covariance,
                                                                       const double cov_threshold)
                                                                       : mean_(mean),
                                                                         covariance_(covariance),
                                                                         cov_threshold_(cov_threshold)
{
    if(!checkPositiveDefiniteness(covariance_)) {
        throw std::runtime_error("Possibly non semi-positive definitie matrix!");
    }

    if(!initializeData()) {
        throw std::runtime_error("Failed to do Eigen Decomposition.");
    }

    // enable initialization
    initialization_ = true;
}

void ThreeDimensionalNormalDistribution::setValues(const Eigen::Vector3d &mean, const Eigen::Matrix3d &covariance)
{
    // Set Value
    mean_ = mean;
    covariance_ = covariance;

    if(!checkPositiveDefiniteness(covariance_)) {
        throw std::runtime_error("Possibly non semi-positive definitie matrix!");
    }

    if(!initializeData()) {
        throw std::runtime_error("Failed to do Eigen Decomposition.");
    }

    initialization_ = true;
}

bool ThreeDimensionalNormalDistribution::checkPositiveDefiniteness(const Eigen::Matrix3d& covariance)
{
    Eigen::LLT<Eigen::MatrixXd> lltOfA(covariance_); // compute the Cholesky decomposition of A
    if(lltOfA.info() == Eigen::NumericalIssue) {
        return false;
    }

    return true;
}

bool ThreeDimensionalNormalDistribution::initializeData()
{
    // Inverse Matrix
    const Eigen::Matrix3d inv_covariance = covariance_.inverse();

    // Eigen Decomposition
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(inv_covariance);
    if(es.info() != Eigen::Success) {
        return false;
    }

    eigen_values_ = es.eigenvalues();
    T_ = es.eigenvectors();

    return true;
}

TwoDimensionalNormalDistribution ThreeDimensionalNormalDistribution::create2DNormalDistribution(const int dim1, const int dim2)
{
    const Eigen::Vector2d mean = {mean_(dim1), mean_(dim2)};
    Eigen::Matrix2d cov;
    cov << covariance_(dim1, dim1), covariance_(dim1, dim2),
            covariance_(dim2, dim1), covariance_(dim2, dim2);

    return TwoDimensionalNormalDistribution(mean, cov);
}

double ThreeDimensionalNormalDistribution::calc_mean(const int dim)
{
    if(dim > 2) {
        throw std::invalid_argument("Dim is larger than the size of the distribution");
    }
    return mean_(dim);
}

double ThreeDimensionalNormalDistribution::calc_covariance(const int dim)
{
    if(dim > 2) {
        throw std::invalid_argument("Dim is larger than the size of the distribution");
    }
    return covariance_(dim, dim);
}

double ThreeDimensionalNormalDistribution::calc_moment(const int dim, const int moment)
{
    if(dim > 2) {
        throw std::invalid_argument("Dim is larger than the size of the distribution");
    }
    NormalDistribution dist(mean_(dim), covariance_(dim, dim));
    return dist.calc_moment(moment);
}

double ThreeDimensionalNormalDistribution::calc_sin_moment(const int dim ,const int moment)
{
    if(dim > 2) {
        throw std::invalid_argument("Dim is larger than the size of the distribution");
    }
    NormalDistribution dist(mean_(dim), covariance_(dim, dim));
    return dist.calc_sin_moment(moment);
}

double ThreeDimensionalNormalDistribution::calc_cos_moment(const int dim ,const int moment)
{
    if(dim > 2) {
        throw std::invalid_argument("Dim is larger than the size of the distribution");
    }
    NormalDistribution dist(mean_(dim), covariance_(dim, dim));
    return dist.calc_cos_moment(moment);
}

double ThreeDimensionalNormalDistribution::calc_cos_sin_moment(const int dim ,const int cos_moment, const int sin_moment)
{
    if(dim > 2) {
        throw std::invalid_argument("Dim is larger than the size of the distribution");
    }
    NormalDistribution dist(mean_(dim), covariance_(dim, dim));
    return dist.calc_cos_sin_moment(cos_moment, sin_moment);
}

double ThreeDimensionalNormalDistribution::calc_x_sin_x_moment(const int dim, const int moment, const int sin_moment)
{
    if(dim > 2) {
        throw std::invalid_argument("Dim is larger than the size of the distribution");
    }
    NormalDistribution dist(mean_(dim), covariance_(dim, dim));
    return dist.calc_x_sin_moment(moment, sin_moment);
}

double ThreeDimensionalNormalDistribution::calc_x_cos_x_moment(const int dim, const int moment, const int cos_moment)
{
    if(dim > 2) {
        throw std::invalid_argument("Dim is larger than the size of the distribution");
    }
    NormalDistribution dist(mean_(dim), covariance_(dim, dim));
    return dist.calc_x_cos_moment(moment, cos_moment);
}

double ThreeDimensionalNormalDistribution::calc_cross_second_moment(const int dim1, const int dim2)
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    const auto y_mean = T_.transpose()*mean_;
    const double T_dim1_1 = T_(dim1, 0);
    const double T_dim1_2 = T_(dim1, 1);
    const double T_dim1_3 = T_(dim1, 2);
    const double T_dim2_1 = T_(dim2, 0);
    const double T_dim2_2 = T_(dim2, 1);
    const double T_dim2_3 = T_(dim2, 2);

    const double y1_mean = y_mean(0);
    const double y1_cov = 1.0/eigen_values_(0);
    NormalDistribution normal_y1(y1_mean, y1_cov);

    const double y2_mean = y_mean(1);
    const double y2_cov = 1.0/eigen_values_(1);
    NormalDistribution normal_y2(y2_mean, y2_cov);

    const double y3_mean = y_mean(2);
    const double y3_cov = 1.0/eigen_values_(2);
    NormalDistribution normal_y3(y3_mean, y3_cov);

    const double y1_first_moment = normal_y1.calc_moment(1);
    const double y2_first_moment = normal_y2.calc_moment(1);
    const double y3_first_moment = normal_y3.calc_moment(1);
    const double y1_second_moment = normal_y1.calc_moment(2);
    const double y2_second_moment = normal_y2.calc_moment(2);
    const double y3_second_moment = normal_y3.calc_moment(2);

    return T_dim1_1*T_dim2_1*y1_second_moment + T_dim1_1*T_dim2_2*y1_first_moment*y2_first_moment + T_dim1_1*T_dim2_3*y1_first_moment*y3_first_moment +
           T_dim1_2*T_dim2_1*y1_first_moment*y2_first_moment + T_dim1_2*T_dim2_2*y2_second_moment + T_dim1_2*T_dim2_3*y2_first_moment*y3_first_moment +
           T_dim1_3*T_dim2_1*y1_first_moment*y3_first_moment + T_dim1_3*T_dim2_2*y2_first_moment*y3_first_moment + T_dim1_3*T_dim2_3*y3_second_moment;
}

double ThreeDimensionalNormalDistribution::calc_x_sin_z_moment(const int dim_x, const int dim_z)
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(dim_x > 2 || dim_z > 2) {
        throw std::invalid_argument("dim_x or dim_z is larger than the size of the distribution");
    }

    if(std::fabs(covariance_(dim_x, dim_z)) < cov_threshold_) {
        NormalDistribution normal_x(mean_(dim_x), covariance_(dim_x, dim_x));
        NormalDistribution normal_z(mean_(dim_z), covariance_(dim_z, dim_z));

        return normal_x.calc_moment(1) * normal_z.calc_sin_moment(1);
    }

    const auto y_mean = T_.transpose()*mean_;
    const double tx1 = T_(dim_x, 0);
    const double tx2 = T_(dim_x, 1);
    const double tx3 = T_(dim_x, 2);
    const double tz1 = T_(dim_z, 0);
    const double tz2 = T_(dim_z, 1);
    const double tz3 = T_(dim_z, 2);

    const double l1_mean = tz1*y_mean(0);
    const double l1_cov = tz1*tz1/eigen_values_(0);
    NormalDistribution normal1(l1_mean, l1_cov);

    const double l2_mean = tz2*y_mean(1);
    const double l2_cov = tz2*tz2/eigen_values_(1);
    NormalDistribution normal2(l2_mean, l2_cov);

    const double l3_mean = tz3*y_mean(2);
    const double l3_cov = tz3*tz3/eigen_values_(2);
    NormalDistribution normal3(l3_mean, l3_cov);

    // compute E[xsin(theta)]
    const double l1_cosl1 = normal1.calc_x_cos_moment(1, 1);
    const double l1_sinl1 = normal1.calc_x_sin_moment(1, 1);
    const double l2_cosl2 = normal2.calc_x_cos_moment(1, 1);
    const double l2_sinl2 = normal2.calc_x_sin_moment(1, 1);
    const double l3_cosl3 = normal3.calc_x_cos_moment(1, 1);
    const double l3_sinl3 = normal3.calc_x_sin_moment(1, 1);
    const double sinl1 = normal1.calc_sin_moment(1);
    const double cosl1 = normal1.calc_cos_moment(1);
    const double sinl2 = normal2.calc_sin_moment(1);
    const double cosl2 = normal2.calc_cos_moment(1);
    const double sinl3 = normal3.calc_sin_moment(1);
    const double cosl3 = normal3.calc_cos_moment(1);

    return tx1/tz1 *l1_sinl1 * (cosl2*cosl3 - sinl2*sinl3) + tx1/tz1 * l1_cosl1 * (sinl2*cosl3 + cosl2*sinl3)
           + tx2/tz2 *l2_sinl2 * (cosl1*cosl3 - sinl1*sinl3) + tx2/tz2 * l2_cosl2 * (sinl1*cosl3 + cosl1*sinl3)
           + tx3/tz3 *l3_sinl3 * (cosl1*cosl2 - sinl1*sinl2) + tx3/tz3 * l3_cosl3 * (sinl1*cosl2 + cosl1*sinl2);
}

double ThreeDimensionalNormalDistribution::calc_x_cos_z_moment(const int dim_x, const int dim_z)
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(dim_x > 2 || dim_z > 2) {
        throw std::invalid_argument("dim_x or dim_z is larger than the size of the distribution");
    }

    if(std::fabs(covariance_(dim_x, dim_z)) < cov_threshold_) {
        NormalDistribution normal_x(mean_(dim_x), covariance_(dim_x, dim_x));
        NormalDistribution normal_z(mean_(dim_z), covariance_(dim_z, dim_z));

        return normal_x.calc_moment(1) * normal_z.calc_cos_moment(1);
    }

    const auto y_mean = T_.transpose()*mean_;
    const double tx1 = T_(dim_x, 0);
    const double tx2 = T_(dim_x, 1);
    const double tx3 = T_(dim_x, 2);
    const double tz1 = T_(dim_z, 0);
    const double tz2 = T_(dim_z, 1);
    const double tz3 = T_(dim_z, 2);

    const double l1_mean = tz1*y_mean(0);
    const double l1_cov = tz1*tz1/eigen_values_(0);
    NormalDistribution normal1(l1_mean, l1_cov);

    const double l2_mean = tz2*y_mean(1);
    const double l2_cov = tz2*tz2/eigen_values_(1);
    NormalDistribution normal2(l2_mean, l2_cov);

    const double l3_mean = tz3*y_mean(2);
    const double l3_cov = tz3*tz3/eigen_values_(2);
    NormalDistribution normal3(l3_mean, l3_cov);

    // compute E[xsin(theta)]
    const double l1_cosl1 = normal1.calc_x_cos_moment(1, 1);
    const double l1_sinl1 = normal1.calc_x_sin_moment(1, 1);
    const double l2_cosl2 = normal2.calc_x_cos_moment(1, 1);
    const double l2_sinl2 = normal2.calc_x_sin_moment(1, 1);
    const double l3_cosl3 = normal3.calc_x_cos_moment(1, 1);
    const double l3_sinl3 = normal3.calc_x_sin_moment(1, 1);
    const double sinl1 = normal1.calc_sin_moment(1);
    const double cosl1 = normal1.calc_cos_moment(1);
    const double sinl2 = normal2.calc_sin_moment(1);
    const double cosl2 = normal2.calc_cos_moment(1);
    const double sinl3 = normal3.calc_sin_moment(1);
    const double cosl3 = normal3.calc_cos_moment(1);

    return -tx1/tz1*l1_sinl1*(sinl2*cosl3 + cosl2*sinl3) + tx1/tz1*l1_cosl1*(cosl2*cosl3 - sinl2*sinl3)
           -tx2/tz2*l2_sinl2*(sinl3*cosl1 + sinl1*cosl3) + tx2/tz2*l2_cosl2*(cosl1*cosl3 - sinl1*sinl3)
           -tx3/tz3*l3_sinl3*(sinl2*cosl1 + sinl1*cosl2) + tx3/tz3*l3_cosl3*(cosl1*cosl2 - sinl1*sinl2);
}

double ThreeDimensionalNormalDistribution::calc_cross_third_moment(const int dim1, const int dim2, const int moment1, const int moment2)
{
    if(moment1 + moment2 != 3) {
        throw std::invalid_argument("moment1 + moment2 is not 3");
    }

    if(dim1 > 2 || dim2 > 2) {
        throw std::invalid_argument("dim1 or dim2 is over 3");
    }

    const Eigen::Vector2d mean(mean_(dim1), mean_(dim2));
    Eigen::Matrix2d covariance;
    covariance << covariance_(dim1, dim1), covariance_(dim1, dim2),
                  covariance_(dim2, dim1), covariance_(dim2, dim2);

    TwoDimensionalNormalDistribution dist(mean, covariance);

   return dist.calc_third_moment(moment1, moment2);
}

double ThreeDimensionalNormalDistribution::calc_xx_sin_z_moment(const int dim_x, const int dim_z)
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(dim_x > 2 || dim_z > 2) {
        throw std::invalid_argument("dim_x or dim_z is larger than the size of the distribution");
    }

    if(std::fabs(covariance_(dim_x, dim_z)) < cov_threshold_) {
        NormalDistribution normal_x(mean_(dim_x), covariance_(dim_x, dim_x));
        NormalDistribution normal_z(mean_(dim_z), covariance_(dim_z, dim_z));

        return normal_x.calc_moment(2) * normal_z.calc_sin_moment(1);
    }

    const auto y_mean = T_.transpose()*mean_;
    const double tx1 = T_(dim_x, 0);
    const double tx2 = T_(dim_x, 1);
    const double tx3 = T_(dim_x, 2);
    const double tz1 = T_(dim_z, 0);
    const double tz2 = T_(dim_z, 1);
    const double tz3 = T_(dim_z, 2);

    const double l1_mean = tz1*y_mean(0);
    const double l1_cov = tz1*tz1/eigen_values_(0);
    NormalDistribution normal1(l1_mean, l1_cov);

    const double l2_mean = tz2*y_mean(1);
    const double l2_cov = tz2*tz2/eigen_values_(1);
    NormalDistribution normal2(l2_mean, l2_cov);

    const double l3_mean = tz3*y_mean(2);
    const double l3_cov = tz3*tz3/eigen_values_(2);
    NormalDistribution normal3(l3_mean, l3_cov);

    // compute E[xsin(theta)]
    const double l1_cosl1 = normal1.calc_x_cos_moment(1, 1);
    const double l1_sinl1 = normal1.calc_x_sin_moment(1, 1);
    const double l2_cosl2 = normal2.calc_x_cos_moment(1, 1);
    const double l2_sinl2 = normal2.calc_x_sin_moment(1, 1);
    const double l3_cosl3 = normal3.calc_x_cos_moment(1, 1);
    const double l3_sinl3 = normal3.calc_x_sin_moment(1, 1);
    const double l1_square_cosl1 = normal1.calc_x_cos_moment(2, 1);
    const double l1_square_sinl1 = normal1.calc_x_sin_moment(2, 1);
    const double l2_square_cosl2 = normal2.calc_x_cos_moment(2, 1);
    const double l2_square_sinl2 = normal2.calc_x_sin_moment(2, 1);
    const double l3_square_cosl3 = normal3.calc_x_cos_moment(2, 1);
    const double l3_square_sinl3 = normal3.calc_x_sin_moment(2, 1);
    const double sinl1 = normal1.calc_sin_moment(1);
    const double cosl1 = normal1.calc_cos_moment(1);
    const double sinl2 = normal2.calc_sin_moment(1);
    const double cosl2 = normal2.calc_cos_moment(1);
    const double sinl3 = normal3.calc_sin_moment(1);
    const double cosl3 = normal3.calc_cos_moment(1);

    return std::pow(tx1/tz1, 2) *l1_square_sinl1 * (cosl2*cosl3 - sinl2*sinl3) + std::pow(tx1/tz1, 2) * l1_square_cosl1 * (sinl2*cosl3 + cosl2*sinl3) +
           std::pow(tx2/tz2, 2) *l2_square_sinl2 * (cosl1*cosl3 - sinl1*sinl3) + std::pow(tx2/tz2, 2) * l2_square_cosl2 * (sinl1*cosl3 + cosl1*sinl3) +
           std::pow(tx3/tz3, 2) *l3_square_sinl3 * (cosl1*cosl2 - sinl1*sinl2) + std::pow(tx3/tz3, 2) * l3_square_cosl3 * (sinl1*cosl2 + cosl1*sinl2) +
           2*(tx1*tx2)/(tz1*tz2)*sinl3*(l1_cosl1*l2_cosl2 - l1_sinl1*l2_sinl2) +
           2*(tx1*tx2)/(tz1*tz2)*cosl3*(l1_sinl1*l2_cosl2 + l2_sinl2*l1_cosl1) +
           2*(tx1*tx3)/(tz1*tz3)*sinl2*(l1_cosl1*l3_cosl3 - l1_sinl1*l3_sinl3) +
           2*(tx1*tx3)/(tz1*tz3)*cosl2*(l1_sinl1*l3_cosl3 + l3_sinl3*l1_cosl1) +
           2*(tx2*tx3)/(tz2*tz3)*sinl1*(l2_cosl2*l3_cosl3 - l2_sinl2*l3_sinl3) +
           2*(tx2*tx3)/(tz2*tz3)*cosl1*(l2_sinl2*l3_cosl3 + l3_sinl3*l2_cosl2);
}

double ThreeDimensionalNormalDistribution::calc_xx_cos_z_moment(const int dim_x, const int dim_z)
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(dim_x > 2 || dim_z > 2) {
        throw std::invalid_argument("dim_x or dim_z is larger than the size of the distribution");
    }

    if(std::fabs(covariance_(dim_x, dim_z)) < cov_threshold_) {
        NormalDistribution normal_x(mean_(dim_x), covariance_(dim_x, dim_x));
        NormalDistribution normal_z(mean_(dim_z), covariance_(dim_z, dim_z));

        return normal_x.calc_moment(2) * normal_z.calc_cos_moment(1);
    }

    const auto y_mean = T_.transpose()*mean_;
    const double tx1 = T_(dim_x, 0);
    const double tx2 = T_(dim_x, 1);
    const double tx3 = T_(dim_x, 2);
    const double tz1 = T_(dim_z, 0);
    const double tz2 = T_(dim_z, 1);
    const double tz3 = T_(dim_z, 2);

    const double l1_mean = tz1*y_mean(0);
    const double l1_cov = tz1*tz1/eigen_values_(0);
    NormalDistribution normal1(l1_mean, l1_cov);

    const double l2_mean = tz2*y_mean(1);
    const double l2_cov = tz2*tz2/eigen_values_(1);
    NormalDistribution normal2(l2_mean, l2_cov);

    const double l3_mean = tz3*y_mean(2);
    const double l3_cov = tz3*tz3/eigen_values_(2);
    NormalDistribution normal3(l3_mean, l3_cov);

    // compute E[xsin(theta)]
    const double l1_cosl1 = normal1.calc_x_cos_moment(1, 1);
    const double l1_sinl1 = normal1.calc_x_sin_moment(1, 1);
    const double l2_cosl2 = normal2.calc_x_cos_moment(1, 1);
    const double l2_sinl2 = normal2.calc_x_sin_moment(1, 1);
    const double l3_cosl3 = normal3.calc_x_cos_moment(1, 1);
    const double l3_sinl3 = normal3.calc_x_sin_moment(1, 1);
    const double l1_square_cosl1 = normal1.calc_x_cos_moment(2, 1);
    const double l1_square_sinl1 = normal1.calc_x_sin_moment(2, 1);
    const double l2_square_cosl2 = normal2.calc_x_cos_moment(2, 1);
    const double l2_square_sinl2 = normal2.calc_x_sin_moment(2, 1);
    const double l3_square_cosl3 = normal3.calc_x_cos_moment(2, 1);
    const double l3_square_sinl3 = normal3.calc_x_sin_moment(2, 1);
    const double sinl1 = normal1.calc_sin_moment(1);
    const double cosl1 = normal1.calc_cos_moment(1);
    const double sinl2 = normal2.calc_sin_moment(1);
    const double cosl2 = normal2.calc_cos_moment(1);
    const double sinl3 = normal3.calc_sin_moment(1);
    const double cosl3 = normal3.calc_cos_moment(1);

    return - std::pow(tx1/tz1, 2)*l1_square_sinl1*(sinl2*cosl3 + sinl3*cosl2)
           + std::pow(tx1/tz1, 2)*l1_square_cosl1*(cosl2*cosl3 - sinl2*sinl3)
           - std::pow(tx2/tz2, 2)*l2_square_sinl2*(sinl1*cosl3 + sinl3*cosl1)
           + std::pow(tx2/tz2, 2)*l2_square_cosl2*(cosl1*cosl3 - sinl1*sinl3)
           - std::pow(tx3/tz3, 2)*l3_square_sinl3*(sinl1*cosl2 + sinl2*cosl1)
           + std::pow(tx3/tz3, 2)*l3_square_cosl3*(cosl1*cosl2 - sinl1*sinl2)
           - 2*(tx1*tx2)/(tz1*tz2)*sinl3*(l1_sinl1*l2_cosl2 + l1_cosl1*l2_sinl2)
           + 2*(tx1*tx2)/(tz1*tz2)*cosl3*(l1_cosl1*l2_cosl2 - l1_sinl1*l2_sinl2)
           - 2*(tx1*tx3)/(tz1*tz3)*sinl2*(l1_sinl1*l3_cosl3 + l1_cosl1*l3_sinl3)
           + 2*(tx1*tx3)/(tz1*tz3)*cosl2*(l1_cosl1*l3_cosl3 - l1_sinl1*l3_sinl3)
           - 2*(tx2*tx3)/(tz2*tz3)*sinl1*(l2_sinl2*l3_cosl3 + l2_cosl2*l3_sinl3)
           + 2*(tx2*tx3)/(tz2*tz3)*cosl1*(l2_cosl2*l3_cosl3 - l2_sinl2*l3_sinl3);
}

double ThreeDimensionalNormalDistribution::calc_xy_cos_z_moment()
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(std::fabs(covariance_(0, 2)) < cov_threshold_ && std::fabs(covariance_(1, 2)) < cov_threshold_ && std::fabs(covariance_(0, 1))<cov_threshold_) {
        NormalDistribution dist_x(mean_(0), covariance_(0, 0));
        NormalDistribution dist_y(mean_(1), covariance_(1, 1));
        NormalDistribution dist_z(mean_(2), covariance_(2, 2));

        return dist_x.calc_moment(1) * dist_y.calc_moment(1) * dist_z.calc_cos_moment(1);
    }

    const auto y_mean = T_.transpose()*mean_;
    const double t11 = T_(0, 0);
    const double t12 = T_(0, 1);
    const double t13 = T_(0, 2);
    const double t21 = T_(1, 0);
    const double t22 = T_(1, 1);
    const double t23 = T_(1, 2);
    const double t31 = T_(2, 0);
    const double t32 = T_(2, 1);
    const double t33 = T_(2, 2);

    const double l1_mean = t31*y_mean(0);
    const double l1_cov = t31*t31/eigen_values_(0);
    NormalDistribution normal1(l1_mean, l1_cov);

    const double l2_mean = t32*y_mean(1);
    const double l2_cov = t32*t32/eigen_values_(1);
    NormalDistribution normal2(l2_mean, l2_cov);

    const double l3_mean = t33*y_mean(2);
    const double l3_cov = t33*t33/eigen_values_(2);
    NormalDistribution normal3(l3_mean, l3_cov);

    // compute E[xycos(theta)]
    const double l1_cosl1 = normal1.calc_x_cos_moment(1, 1);
    const double l1_sinl1 = normal1.calc_x_sin_moment(1, 1);
    const double l2_cosl2 = normal2.calc_x_cos_moment(1, 1);
    const double l2_sinl2 = normal2.calc_x_sin_moment(1, 1);
    const double l3_cosl3 = normal3.calc_x_cos_moment(1, 1);
    const double l3_sinl3 = normal3.calc_x_sin_moment(1, 1);
    const double l1_square_cosl1 = normal1.calc_x_cos_moment(2, 1);
    const double l1_square_sinl1 = normal1.calc_x_sin_moment(2, 1);
    const double l2_square_cosl2 = normal2.calc_x_cos_moment(2, 1);
    const double l2_square_sinl2 = normal2.calc_x_sin_moment(2, 1);
    const double l3_square_cosl3 = normal3.calc_x_cos_moment(2, 1);
    const double l3_square_sinl3 = normal3.calc_x_sin_moment(2, 1);
    const double sinl1 = normal1.calc_sin_moment(1);
    const double cosl1 = normal1.calc_cos_moment(1);
    const double sinl2 = normal2.calc_sin_moment(1);
    const double cosl2 = normal2.calc_cos_moment(1);
    const double sinl3 = normal3.calc_sin_moment(1);
    const double cosl3 = normal3.calc_cos_moment(1);

    return - (t11*t21)/(t31*t31) * l1_square_sinl1 * sinl2 * cosl3
           - (t11*t21)/(t31*t31) * l1_square_sinl1 * sinl3 * cosl2
           - (t11*t21)/(t31*t31) * l1_square_cosl1 * sinl2 * sinl3
           + (t11*t21)/(t31*t31) * l1_square_cosl1 * cosl2 * cosl3
           - (t11*t22)/(t31*t32) * l1_sinl1 * l2_cosl2 * sinl3
           - (t11*t22)/(t31*t32) * l1_cosl1 * l2_sinl2 * sinl3
           - (t11*t22)/(t31*t32) * l1_sinl1 * l2_sinl2 * cosl3
           + (t11*t22)/(t31*t32) * l1_cosl1 * l2_cosl2 * cosl3
           - (t11*t23)/(t31*t33) * l1_sinl1 * l3_cosl3 * sinl2
           - (t11*t23)/(t31*t33) * l1_cosl1 * l3_sinl3 * sinl2
           - (t11*t23)/(t31*t33) * l1_sinl1 * l3_sinl3 * cosl2
           + (t11*t23)/(t31*t33) * l1_cosl1 * l3_cosl3 * cosl2
           - (t12*t21)/(t31*t32) * l1_sinl1 * l2_cosl2 * sinl3
           - (t12*t21)/(t31*t32) * l2_sinl2 * l1_cosl1 * sinl3
           - (t12*t21)/(t31*t32) * l1_sinl1 * l2_sinl2 * cosl3
           + (t12*t21)/(t31*t32) * l1_cosl1 * l2_cosl2 * cosl3
           - (t12*t22)/(t32*t32) * l2_square_sinl2 * sinl1 * cosl3
           - (t12*t22)/(t32*t32) * l2_square_sinl2 * sinl3 * cosl1
           - (t12*t22)/(t32*t32) * l2_square_cosl2 * sinl1 * sinl3
           + (t12*t22)/(t32*t32) * l2_square_cosl2 * cosl1 * cosl3
           - (t12*t23)/(t32*t33) * l2_sinl2 * l3_cosl3 * sinl1
           - (t12*t23)/(t32*t33) * l2_cosl2 * l3_sinl3 * sinl1
           - (t12*t23)/(t32*t33) * l2_sinl2 * l3_sinl3 * cosl1
           + (t12*t23)/(t32*t33) * l2_cosl2 * l3_cosl3 * cosl1
           - (t13*t21)/(t31*t33) * l1_sinl1 * l3_cosl3 * sinl2
           - (t13*t21)/(t31*t33) * l1_cosl1 * l3_sinl3 * sinl2
           - (t13*t21)/(t31*t33) * l1_sinl1 * l3_sinl3 * cosl2
           + (t13*t21)/(t31*t33) * l1_cosl1 * l3_cosl3 * cosl2
           - (t13*t22)/(t32*t33) * l2_sinl2 * l3_cosl3 * sinl1
           - (t13*t22)/(t32*t33) * l2_cosl2 * l3_sinl3 * sinl1
           - (t13*t22)/(t32*t33) * l2_sinl2 * l3_sinl3 * cosl1
           + (t13*t22)/(t32*t33) * l2_cosl2 * l3_cosl3 * cosl1
           - (t13*t23)/(t33*t33) * l3_square_sinl3 * sinl1 * cosl2
           - (t13*t23)/(t33*t33) * l3_square_sinl3 * sinl2 * cosl1
           - (t13*t23)/(t33*t33) * l3_square_cosl3 * sinl1 * sinl2
           + (t13*t23)/(t33*t33) * l3_square_cosl3 * cosl1 * cosl2;
}

double ThreeDimensionalNormalDistribution::calc_xy_sin_z_moment()
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(std::fabs(covariance_(0, 2)) < cov_threshold_ && std::fabs(covariance_(1, 2)) < cov_threshold_ && std::fabs(covariance_(0, 1))<cov_threshold_) {
        NormalDistribution dist_x(mean_(0), covariance_(0, 0));
        NormalDistribution dist_y(mean_(1), covariance_(1, 1));
        NormalDistribution dist_z(mean_(2), covariance_(2, 2));

        return dist_x.calc_moment(1) * dist_y.calc_moment(1) * dist_z.calc_sin_moment(1);
    }

    const auto y_mean = T_.transpose()*mean_;
    const double t11 = T_(0, 0);
    const double t12 = T_(0, 1);
    const double t13 = T_(0, 2);
    const double t21 = T_(1, 0);
    const double t22 = T_(1, 1);
    const double t23 = T_(1, 2);
    const double t31 = T_(2, 0);
    const double t32 = T_(2, 1);
    const double t33 = T_(2, 2);

    const double l1_mean = t31*y_mean(0);
    const double l1_cov = t31*t31/eigen_values_(0);
    NormalDistribution normal1(l1_mean, l1_cov);

    const double l2_mean = t32*y_mean(1);
    const double l2_cov = t32*t32/eigen_values_(1);
    NormalDistribution normal2(l2_mean, l2_cov);

    const double l3_mean = t33*y_mean(2);
    const double l3_cov = t33*t33/eigen_values_(2);
    NormalDistribution normal3(l3_mean, l3_cov);

    // compute E[xycos(theta)]
    const double l1_cosl1 = normal1.calc_x_cos_moment(1, 1);
    const double l1_sinl1 = normal1.calc_x_sin_moment(1, 1);
    const double l2_cosl2 = normal2.calc_x_cos_moment(1, 1);
    const double l2_sinl2 = normal2.calc_x_sin_moment(1, 1);
    const double l3_cosl3 = normal3.calc_x_cos_moment(1, 1);
    const double l3_sinl3 = normal3.calc_x_sin_moment(1, 1);
    const double l1_square_cosl1 = normal1.calc_x_cos_moment(2, 1);
    const double l1_square_sinl1 = normal1.calc_x_sin_moment(2, 1);
    const double l2_square_cosl2 = normal2.calc_x_cos_moment(2, 1);
    const double l2_square_sinl2 = normal2.calc_x_sin_moment(2, 1);
    const double l3_square_cosl3 = normal3.calc_x_cos_moment(2, 1);
    const double l3_square_sinl3 = normal3.calc_x_sin_moment(2, 1);
    const double sinl1 = normal1.calc_sin_moment(1);
    const double cosl1 = normal1.calc_cos_moment(1);
    const double sinl2 = normal2.calc_sin_moment(1);
    const double cosl2 = normal2.calc_cos_moment(1);
    const double sinl3 = normal3.calc_sin_moment(1);
    const double cosl3 = normal3.calc_cos_moment(1);

    return   (t11*t21)/(t31*t31) * l1_square_sinl1 * (-sinl2 * sinl3 + cosl2 * cosl3)
           + (t11*t21)/(t31*t31) * l1_square_cosl1 * (sinl2  * cosl3 + cosl2 * sinl3)
           + (t11*t22)/(t31*t32) * sinl3 * (-l1_sinl1 * l2_sinl2 + l1_cosl1 * l2_cosl2)
           + (t11*t22)/(t31*t32) * cosl3 * ( l1_sinl1 * l2_cosl2 + l2_sinl2 * l1_cosl1)
           + (t11*t23)/(t31*t33) * sinl2 * (-l1_sinl1 * l3_sinl3 + l1_cosl1 * l3_cosl3)
           + (t11*t23)/(t31*t33) * cosl2 * (l1_sinl1 * l3_cosl3 + l3_sinl3 * l1_cosl1)
           + (t12*t21)/(t31*t32) * sinl3 * (-l1_sinl1 * l2_sinl2 + l1_cosl1 * l2_cosl2)
           + (t12*t21)/(t31*t32) * cosl3 * (l1_sinl1 * l2_cosl2 + l2_sinl2 * l1_cosl1)
           + (t12*t22)/(t32*t32) * l2_square_sinl2 * (-sinl1 * sinl3 + cosl1 * cosl3)
           + (t12*t22)/(t32*t32) * l2_square_cosl2 * ( sinl1 * cosl3 + sinl3 * cosl1)
           + (t12*t23)/(t32*t33) * sinl1 * (-l2_sinl2 * l3_sinl3 + l2_cosl2 * l3_cosl3)
           + (t12*t23)/(t32*t33) * cosl1 * (l2_sinl2 * l3_cosl3 + l3_sinl3 * l2_cosl2)
           + (t13*t21)/(t31*t33) * sinl2 * (-l1_sinl1 * l3_sinl3 + l1_cosl1 * l3_cosl3)
           + (t13*t21)/(t31*t33) * cosl2 * (l1_sinl1 * l3_cosl3 + l1_cosl1 * l3_sinl3)
           + (t13*t22)/(t32*t33) * sinl1 * (-l2_sinl2 * l3_sinl3 + l2_cosl2 * l3_cosl3)
           + (t13*t22)/(t32*t33) * cosl1 * (l2_sinl2 * l3_cosl3 + l3_sinl3 * l2_cosl2)
           + (t13*t23)/(t33*t33) * l3_square_sinl3 * (-sinl1 * sinl2 + cosl1 * cosl2)
           + (t13*t23)/(t33*t33) * l3_square_cosl3 * ( sinl1 * cosl2 + sinl2 * cosl1);
}

double ThreeDimensionalNormalDistribution::calc_xy_cos_y_moment(const int dim_x, const int dim_y)
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(dim_x > 2 || dim_y > 2) {
        throw std::invalid_argument("dim_x or dim_y is larger than the size of the distribution");
    }

    auto dist = create2DNormalDistribution(dim_x, dim_y);

    return dist.calc_x_y_cos_y_moment();
}

double ThreeDimensionalNormalDistribution::calc_xy_sin_y_moment(const int dim_x, const int dim_y)
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(dim_x > 2 || dim_y > 2) {
        throw std::invalid_argument("dim_x or dim_y is larger than the size of the distribution");
    }

    auto dist = create2DNormalDistribution(dim_x, dim_y);

    return dist.calc_x_y_sin_y_moment();
}

double ThreeDimensionalNormalDistribution::calc_x_cos_z_cos_z_moment(const int dim_x, const int dim_z)
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(dim_x > 2 || dim_z > 2) {
        throw std::invalid_argument("dim_x or dim_z is larger than the size of the distribution");
    }

    auto dist = create2DNormalDistribution(dim_x, dim_z);

    return dist.calc_x_cos_y_cos_y_moment();
}

double ThreeDimensionalNormalDistribution::calc_x_sin_z_sin_z_moment(const int dim_x, const int dim_z)
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(dim_x > 2 || dim_z > 2) {
        throw std::invalid_argument("dim_x or dim_z is larger than the size of the distribution");
    }

    auto dist = create2DNormalDistribution(dim_x, dim_z);

    return dist.calc_x_sin_y_sin_y_moment();
}

double ThreeDimensionalNormalDistribution::calc_x_cos_z_sin_z_moment(const int dim_x, const int dim_z)
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(dim_x > 2 || dim_z > 2) {
        throw std::invalid_argument("dim_x or dim_z is larger than the size of the distribution");
    }

    auto dist = create2DNormalDistribution(dim_x, dim_z);

    return dist.calc_x_cos_y_sin_y_moment();
}

double ThreeDimensionalNormalDistribution::calc_xxyy_moment(const int dim_x, const int dim_y)
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(dim_x > 2 || dim_y > 2) {
        throw std::invalid_argument("dim_x or dim_z is larger than the size of the distribution");
    }

    const auto y_mean = T_.transpose()*mean_;
    const double tx1 = T_(dim_x, 0);
    const double tx2 = T_(dim_x, 1);
    const double tx3 = T_(dim_x, 2);
    const double ty1 = T_(dim_y, 0);
    const double ty2 = T_(dim_y, 1);
    const double ty3 = T_(dim_y, 2);

    const double y1_mean = y_mean(0);
    const double y1_cov = 1.0/eigen_values_(0);
    NormalDistribution normal_y1(y1_mean, y1_cov);

    const double y2_mean = y_mean(1);
    const double y2_cov = 1.0/eigen_values_(1);
    NormalDistribution normal_y2(y2_mean, y2_cov);

    const double y3_mean = y_mean(2);
    const double y3_cov = 1.0/eigen_values_(2);
    NormalDistribution normal_y3(y3_mean, y3_cov);

    const double y1_first_moment = normal_y1.calc_moment(1);
    const double y2_first_moment = normal_y2.calc_moment(1);
    const double y3_first_moment = normal_y3.calc_moment(1);
    const double y1_second_moment = normal_y1.calc_moment(2);
    const double y2_second_moment = normal_y2.calc_moment(2);
    const double y3_second_moment = normal_y3.calc_moment(2);
    const double y1_third_moment = normal_y1.calc_moment(3);
    const double y2_third_moment = normal_y2.calc_moment(3);
    const double y3_third_moment = normal_y3.calc_moment(3);
    const double y1_fourth_moment = normal_y1.calc_moment(4);
    const double y2_fourth_moment = normal_y2.calc_moment(4);
    const double y3_fourth_moment = normal_y3.calc_moment(4);

    return std::pow(tx1, 2)*std::pow(ty1, 2)*y1_fourth_moment + 2*std::pow(tx1, 2)*ty1*ty2*y1_third_moment*y2_first_moment
           + 2*std::pow(tx1, 2)*ty1*ty3*y1_third_moment*y3_first_moment
           + std::pow(tx1, 2)*std::pow(ty2, 2)*y1_second_moment*y2_second_moment + 2*std::pow(tx1, 2)*ty2*ty3*y1_second_moment*y2_first_moment*y3_first_moment
           + std::pow(tx1, 2)*std::pow(ty3, 2)*y1_second_moment*y3_second_moment + 2*tx1*tx2*std::pow(ty1, 2)*y1_third_moment*y2_first_moment
           + 4*tx1*tx2*ty1*ty2*y1_second_moment*y2_second_moment + 4*tx1*tx2*ty1*ty3*y1_second_moment*y2_first_moment*y3_first_moment
           + 2*tx1*tx2*std::pow(ty2, 2)*y1_first_moment*y2_third_moment + 4*tx1*tx2*ty2*ty3*y1_first_moment*y2_second_moment*y3_first_moment
           + 2*tx1*tx2*std::pow(ty3, 2)*y1_first_moment*y2_first_moment*y3_second_moment + 2*tx1*tx3*std::pow(ty1, 2)*y1_third_moment*y3_first_moment
           + 4*tx1*tx3*ty1*ty2*y1_second_moment*y2_first_moment*y3_first_moment + 4*tx1*tx3*ty1*ty3*y1_second_moment*y3_second_moment
           + 2*tx1*tx3*std::pow(ty2, 2)*y1_first_moment*y2_second_moment*y3_first_moment + 4*tx1*tx3*ty2*ty3*y1_first_moment*y2_first_moment*y3_second_moment
           + 2*tx1*tx3*std::pow(ty3, 2)*y1_first_moment*y3_third_moment + std::pow(tx2, 2)*std::pow(ty1, 2)*y1_second_moment*y2_second_moment
           + 2*std::pow(tx2, 2)*ty1*ty2*y1_first_moment*y2_third_moment + 2*std::pow(tx2, 2)*ty1*ty3*y1_first_moment*y2_second_moment*y3_first_moment
           + std::pow(tx2, 2)*std::pow(ty2, 2)*y2_fourth_moment + 2*std::pow(tx2, 2)*ty2*ty3*y2_third_moment*y3_first_moment
           + std::pow(tx2, 2)*std::pow(ty3, 2)*y2_second_moment*y3_second_moment + 2*tx2*tx3*std::pow(ty1, 2)*y1_second_moment*y2_first_moment*y3_first_moment
           + 4*tx2*tx3*ty1*ty2*y1_first_moment*y2_second_moment*y3_first_moment + 4*tx2*tx3*ty1*ty3*y1_first_moment*y2_first_moment*y3_second_moment
           + 2*tx2*tx3*std::pow(ty2, 2)*y2_third_moment*y3_first_moment + 4*tx2*tx3*ty2*ty3*y2_second_moment*y3_second_moment
           + 2*tx2*tx3*std::pow(ty3, 2)*y2_first_moment*y3_third_moment + std::pow(tx3, 2)*std::pow(ty1, 2)*y1_second_moment*y3_second_moment
           + 2*std::pow(tx3, 2)*ty1*ty2*y1_first_moment*y2_first_moment*y3_second_moment + 2*std::pow(tx3, 2)*ty1*ty3*y1_first_moment*y3_third_moment
           + std::pow(tx3, 2)*std::pow(ty2, 2)*y2_second_moment*y3_second_moment
           + 2*std::pow(tx3, 2)*ty2*ty3*y2_first_moment*y3_third_moment + std::pow(tx3, 2)*std::pow(ty3, 2)*y3_fourth_moment;
}

double ThreeDimensionalNormalDistribution::calc_xx_cos_z_cos_z_moment(const int dim_x, const int dim_z)
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(dim_x > 2 || dim_z > 2) {
        throw std::invalid_argument("dim_x or dim_z is larger than the size of the distribution");
    }

    auto dist = create2DNormalDistribution(dim_x, dim_z);

    return dist.calc_xx_cos_y_cos_y_moment();
}

double ThreeDimensionalNormalDistribution::calc_xx_sin_z_sin_z_moment(const int dim_x, const int dim_z)
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(dim_x > 2 || dim_z > 2) {
        throw std::invalid_argument("dim_x or dim_z is larger than the size of the distribution");
    }

    auto dist = create2DNormalDistribution(dim_x, dim_z);

    return dist.calc_xx_sin_y_sin_y_moment();
}

double ThreeDimensionalNormalDistribution::calc_xx_cos_z_sin_z_moment(const int dim_x, const int dim_z)
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(dim_x > 2 || dim_z > 2) {
        throw std::invalid_argument("dim_x or dim_z is larger than the size of the distribution");
    }

    auto dist = create2DNormalDistribution(dim_x, dim_z);

    return dist.calc_xx_cos_y_sin_y_moment();
}

double ThreeDimensionalNormalDistribution::calc_xxy_cos_z_moment()
{
    if (!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if (std::fabs(covariance_(0, 2)) < cov_threshold_ && std::fabs(covariance_(1, 2)) < cov_threshold_ && std::fabs(covariance_(0,1))<cov_threshold_) {
        NormalDistribution dist_x(mean_(0), covariance_(0, 0));
        NormalDistribution dist_y(mean_(1), covariance_(1, 1));
        NormalDistribution dist_z(mean_(2), covariance_(2, 2));

        return dist_x.calc_moment(2) * dist_y.calc_moment(1) * dist_z.calc_cos_moment(1);
    }

    const auto y_mean = T_.transpose() * mean_;
    const double t11 = T_(0, 0);
    const double t12 = T_(0, 1);
    const double t13 = T_(0, 2);
    const double t21 = T_(1, 0);
    const double t22 = T_(1, 1);
    const double t23 = T_(1, 2);
    const double t31 = T_(2, 0);
    const double t32 = T_(2, 1);
    const double t33 = T_(2, 2);

    const double l1_mean = t31 * y_mean(0);
    const double l1_cov = t31 * t31 / eigen_values_(0);
    NormalDistribution normal1(l1_mean, l1_cov);

    const double l2_mean = t32 * y_mean(1);
    const double l2_cov = t32 * t32 / eigen_values_(1);
    NormalDistribution normal2(l2_mean, l2_cov);

    const double l3_mean = t33 * y_mean(2);
    const double l3_cov = t33 * t33 / eigen_values_(2);
    NormalDistribution normal3(l3_mean, l3_cov);

    // compute E[xycos(theta)]
    const double l1_cosl1 = normal1.calc_x_cos_moment(1, 1);
    const double l1_sinl1 = normal1.calc_x_sin_moment(1, 1);
    const double l2_cosl2 = normal2.calc_x_cos_moment(1, 1);
    const double l2_sinl2 = normal2.calc_x_sin_moment(1, 1);
    const double l3_cosl3 = normal3.calc_x_cos_moment(1, 1);
    const double l3_sinl3 = normal3.calc_x_sin_moment(1, 1);
    const double l1Pow2_cosl1 = normal1.calc_x_cos_moment(2, 1);
    const double l1Pow2_sinl1 = normal1.calc_x_sin_moment(2, 1);
    const double l2Pow2_cosl2 = normal2.calc_x_cos_moment(2, 1);
    const double l2Pow2_sinl2 = normal2.calc_x_sin_moment(2, 1);
    const double l3Pow2_cosl3 = normal3.calc_x_cos_moment(2, 1);
    const double l3Pow2_sinl3 = normal3.calc_x_sin_moment(2, 1);
    const double l1Pow3_cosl1 = normal1.calc_x_cos_moment(3, 1);
    const double l1Pow3_sinl1 = normal1.calc_x_sin_moment(3, 1);
    const double l2Pow3_cosl2 = normal2.calc_x_cos_moment(3, 1);
    const double l2Pow3_sinl2 = normal2.calc_x_sin_moment(3, 1);
    const double l3Pow3_cosl3 = normal3.calc_x_cos_moment(3, 1);
    const double l3Pow3_sinl3 = normal3.calc_x_sin_moment(3, 1);
    const double sinl1 = normal1.calc_sin_moment(1);
    const double cosl1 = normal1.calc_cos_moment(1);
    const double sinl2 = normal2.calc_sin_moment(1);
    const double cosl2 = normal2.calc_cos_moment(1);
    const double sinl3 = normal3.calc_sin_moment(1);
    const double cosl3 = normal3.calc_cos_moment(1);

    return -std::pow(t11 / t31, 2) * t21 / t31 * sinl3 * (l1Pow3_sinl1 * cosl2 + l1Pow3_cosl1 * sinl2)
           - std::pow(t11 / t31, 2) * t21 / t31 * cosl3 * (l1Pow3_sinl1 * sinl2 - l1Pow3_cosl1 * cosl2)
           - std::pow(t11 / t31, 2) * t22 / t32 * sinl3 * (l1Pow2_sinl1 * l2_cosl2 + l1Pow2_cosl1 * l2_sinl2)
           - std::pow(t11 / t31, 2) * t22 / t32 * cosl3 * (l1Pow2_sinl1 * l2_sinl2 - l1Pow2_cosl1 * l2_cosl2)
           - std::pow(t11 / t31, 2) * t23 / t33 * sinl2 * (l1Pow2_sinl1 * l3_cosl3 + l1Pow2_cosl1 * l3_sinl3)
           - std::pow(t11 / t31, 2) * t23 / t33 * cosl2 * (l1Pow2_sinl1 * l3_sinl3 - l1Pow2_cosl1 * l3_cosl3)
           - std::pow(t12 / t32, 2) * t21 / t31 * sinl3 * (l1_sinl1 * l2Pow2_cosl2 + l1_cosl1 * l2Pow2_sinl2)
           - std::pow(t12 / t32, 2) * t21 / t31 * cosl3 * (l1_sinl1 * l2Pow2_sinl2 - l1_cosl1 * l2Pow2_cosl2)
           - std::pow(t12 / t32, 2) * t22 / t32 * sinl3 * (l2Pow3_sinl2 * cosl1 + l2Pow3_cosl2 * sinl1)
           - std::pow(t12 / t32, 2) * t22 / t32 * cosl3 * (l2Pow3_sinl2 * sinl1 - l2Pow3_cosl2 * cosl1)
           - std::pow(t12 / t32, 2) * t23 / t33 * sinl1 * (l3_sinl3 * l2Pow2_cosl2 + l3_cosl3 * l2Pow2_sinl2)
           - std::pow(t12 / t32, 2) * t23 / t33 * cosl1 * (l3_sinl3 * l2Pow2_sinl2 - l3_cosl3 * l2Pow2_cosl2)
           - std::pow(t13 / t33, 2) * t21 / t31 * sinl2 * (l3Pow2_sinl3 * l1_cosl1 + l3Pow2_cosl3 * l1_sinl1)
           - std::pow(t13 / t33, 2) * t21 / t31 * cosl2 * (l3Pow2_sinl3 * l1_sinl1 - l3Pow2_cosl3 * l1_cosl1)
           - std::pow(t13 / t33, 2) * t22 / t32 * sinl1 * (l3Pow2_sinl3 * l2_cosl2 + l3Pow2_cosl3 * l2_sinl2)
           - std::pow(t13 / t33, 2) * t22 / t32 * cosl1 * (l3Pow2_sinl3 * l2_sinl2 - l3Pow2_cosl3 * l2_cosl2)
           - std::pow(t13 / t33, 2) * t23 / t33 * sinl2 * (l3Pow3_sinl3 * cosl1 + l3Pow3_cosl3 * sinl1)
           - std::pow(t13 / t33, 2) * t23 / t33 * cosl2 * (l3Pow3_sinl3 * sinl1 - l3Pow3_cosl3 * cosl1)
           - 2 * (t11 * t12 * t21) / (t31 * t31 * t32) * sinl3 * (l1Pow2_sinl1 * l2_cosl2 + l1Pow2_cosl1 * l2_sinl2)
           - 2 * (t11 * t12 * t21) / (t31 * t31 * t32) * cosl3 * (l1Pow2_sinl1 * l2_sinl2 - l1Pow2_cosl1 * l2_cosl2)
           - 2 * (t11 * t12 * t22) / (t31 * t32 * t32) * sinl3 * (l2Pow2_sinl2 * l1_cosl1 + l2Pow2_cosl2 * l1_sinl1)
           - 2 * (t11 * t12 * t22) / (t31 * t32 * t32) * cosl3 * (l2Pow2_sinl2 * l1_sinl1 - l2Pow2_cosl2 * l1_cosl1)
           - 2 * (t11 * t12 * t23) / (t31 * t32 * t33) * l1_sinl1 * (l2_sinl2 * l3_cosl3 + l2_cosl2 * l3_sinl3)
           - 2 * (t11 * t12 * t23) / (t31 * t32 * t33) * l1_cosl1 * (l2_sinl2 * l3_sinl3 - l2_cosl2 * l3_cosl3)
           - 2 * (t11 * t13 * t21) / (t31 * t31 * t33) * sinl2 * (l1Pow2_sinl1 * l3_cosl3 + l1Pow2_cosl1 * l3_sinl3)
           - 2 * (t11 * t13 * t21) / (t31 * t31 * t33) * cosl2 * (l1Pow2_sinl1 * l3_sinl3 - l1Pow2_cosl1 * l3_cosl3)
           - 2 * (t11 * t13 * t22) / (t31 * t32 * t33) * l1_sinl1 * (l2_sinl2 * l3_cosl3 + l3_sinl3 * l2_cosl2)
           - 2 * (t11 * t13 * t22) / (t31 * t32 * t33) * l1_cosl1 * (l2_sinl2 * l3_sinl3 - l2_cosl2 * l3_cosl3)
           - 2 * (t11 * t13 * t23) / (t31 * t33 * t33) * sinl2 * (l1_sinl1 * l3Pow2_cosl3 + l1_cosl1 * l3Pow2_sinl3)
           - 2 * (t11 * t13 * t23) / (t31 * t33 * t33) * cosl2 * (l1_sinl1 * l3Pow2_sinl3 - l1_cosl1 * l3Pow2_cosl3)
           - 2 * (t12 * t13 * t21) / (t31 * t32 * t33) * l1_sinl1 * (l2_sinl2 * l3_cosl3 + l2_cosl2 * l3_sinl3)
           - 2 * (t12 * t13 * t21) / (t31 * t32 * t33) * l1_cosl1 * (l2_sinl2 * l3_sinl3 - l2_cosl2 * l3_cosl3)
           - 2 * (t12 * t13 * t22) / (t32 * t32 * t33) * sinl1 * (l2Pow2_sinl2 * l3_cosl3 + l2Pow2_cosl2 * l3_sinl3)
           - 2 * (t12 * t13 * t22) / (t32 * t32 * t33) * cosl1 * (l2Pow2_sinl2 * l3_sinl3 - l2Pow2_cosl2 * l3_cosl3)
           - 2 * (t12 * t13 * t23) / (t32 * t33 * t33) * sinl1 * (l3Pow2_sinl3 * l2_cosl2 + l3Pow2_cosl3 * l2_sinl2)
           - 2 * (t12 * t13 * t23) / (t32 * t33 * t33) * cosl1 * (l3Pow2_sinl3 * l2_sinl2 - l3Pow2_cosl3 * l2_cosl2);
}

double ThreeDimensionalNormalDistribution::calc_xy_cos_z_cos_z_moment()
{
    if (!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if (std::fabs(covariance_(0, 2)) < cov_threshold_ && std::fabs(covariance_(1, 2)) < cov_threshold_ && std::fabs(covariance_(0,1))<cov_threshold_) {
        NormalDistribution dist_x(mean_(0), covariance_(0, 0));
        NormalDistribution dist_y(mean_(1), covariance_(1, 1));
        NormalDistribution dist_z(mean_(2), covariance_(2, 2));

        return dist_x.calc_moment(1) * dist_y.calc_moment(1) * dist_z.calc_cos_moment(2);
    }

    const auto y_mean = T_.transpose() * mean_;
    const double t11 = T_(0, 0);
    const double t12 = T_(0, 1);
    const double t13 = T_(0, 2);
    const double t21 = T_(1, 0);
    const double t22 = T_(1, 1);
    const double t23 = T_(1, 2);
    const double t31 = T_(2, 0);
    const double t32 = T_(2, 1);
    const double t33 = T_(2, 2);

    const double l1_mean = t31 * y_mean(0);
    const double l1_cov = t31 * t31 / eigen_values_(0);
    NormalDistribution normal1(l1_mean, l1_cov);

    const double l2_mean = t32 * y_mean(1);
    const double l2_cov = t32 * t32 / eigen_values_(1);
    NormalDistribution normal2(l2_mean, l2_cov);

    const double l3_mean = t33 * y_mean(2);
    const double l3_cov = t33 * t33 / eigen_values_(2);
    NormalDistribution normal3(l3_mean, l3_cov);

    // compute E[xycos(theta)^2]
    const double sinl1Pow2 = normal1.calc_sin_moment(2);
    const double cosl1Pow2 = normal1.calc_cos_moment(2);
    const double sinl2Pow2 = normal2.calc_sin_moment(2);
    const double cosl2Pow2 = normal2.calc_cos_moment(2);
    const double sinl3Pow2 = normal3.calc_sin_moment(2);
    const double cosl3Pow2 = normal3.calc_cos_moment(2);
    const double cosl1_sinl1 = normal1.calc_cos_sin_moment(1, 1);
    const double cosl2_sinl2 = normal2.calc_cos_sin_moment(1, 1);
    const double cosl3_sinl3 = normal3.calc_cos_sin_moment(1, 1);

    const double l1_cosl1_sinl1 = normal1.calc_x_cos_sin_moment(1, 1, 1);
    const double l2_cosl2_sinl2 = normal2.calc_x_cos_sin_moment(1, 1, 1);
    const double l3_cosl3_sinl3 = normal3.calc_x_cos_sin_moment(1, 1, 1);
    const double l1Pow2_cosl1_sinl1 = normal1.calc_x_cos_sin_moment(2, 1, 1);
    const double l2Pow2_cosl2_sinl2 = normal2.calc_x_cos_sin_moment(2, 1, 1);
    const double l3Pow2_cosl3_sinl3 = normal3.calc_x_cos_sin_moment(2, 1, 1);
    const double l1_cosl1Pow2 = normal1.calc_x_cos_moment(1, 2);
    const double l1_sinl1Pow2 = normal1.calc_x_sin_moment(1, 2);
    const double l2_cosl2Pow2 = normal2.calc_x_cos_moment(1, 2);
    const double l2_sinl2Pow2 = normal2.calc_x_sin_moment(1, 2);
    const double l3_cosl3Pow2 = normal3.calc_x_cos_moment(1, 2);
    const double l3_sinl3Pow2 = normal3.calc_x_sin_moment(1, 2);
    const double l1Pow2_cosl1Pow2 = normal1.calc_x_cos_moment(2, 2);
    const double l1Pow2_sinl1Pow2 = normal1.calc_x_sin_moment(2, 2);
    const double l2Pow2_cosl2Pow2 = normal2.calc_x_cos_moment(2, 2);
    const double l2Pow2_sinl2Pow2 = normal2.calc_x_sin_moment(2, 2);
    const double l3Pow2_cosl3Pow2 = normal3.calc_x_cos_moment(2, 2);
    const double l3Pow2_sinl3Pow2 = normal3.calc_x_sin_moment(2, 2);

    return  t11*t21/(t31*t31)   * l1Pow2_sinl1Pow2*sinl2Pow2*cosl3Pow2
          + 2*t11*t21/(t31*t31) * l1Pow2_sinl1Pow2*cosl2_sinl2*cosl3_sinl3
          + t11*t21/(t31*t31)   * l1Pow2_sinl1Pow2*sinl3Pow2*cosl2Pow2
          + 2*t11*t21/(t31*t31) * l1Pow2_cosl1_sinl1 * sinl2Pow2 * cosl3_sinl3
          + 2*t11*t21/(t31*t31) * l1Pow2_cosl1_sinl1 * sinl3Pow2 * cosl2_sinl2
          - 2*t11*t21/(t31*t31) * l1Pow2_cosl1_sinl1 * cosl2_sinl2 * cosl3Pow2
          - 2*t11*t21/(t31*t31) * l1Pow2_cosl1_sinl1 * cosl2Pow2 * cosl3_sinl3
          + t11*t21/(t31*t31)   * l1Pow2_cosl1Pow2 * sinl2Pow2 * sinl3Pow2
          - 2*t11*t21/(t31*t31) * l1Pow2_cosl1Pow2 * cosl2_sinl2 * cosl3_sinl3
          + t11*t21/(t31*t31)   * l1Pow2_cosl1Pow2 * cosl2Pow2 * cosl3Pow2
          + t11*t22/(t31*t32)   * l1_sinl1Pow2 * l2_sinl2Pow2 * cosl3Pow2
          + 2*t11*t22/(t31*t32) * l1_sinl1Pow2 * l2_cosl2_sinl2 * cosl3_sinl3
          + t11*t22/(t31*t32)   * l1_sinl1Pow2 * sinl3Pow2 * l2_cosl2Pow2
          + 2*t11*t22/(t31*t32) * l1_cosl1_sinl1*l2_sinl2Pow2*cosl3_sinl3
          + 2*t11*t22/(t31*t32) * l1_cosl1_sinl1*l2_cosl2_sinl2*sinl3Pow2
          - 2*t11*t22/(t31*t32) * l1_cosl1_sinl1*l2_cosl2_sinl2*cosl3Pow2
          - 2*t11*t22/(t31*t32) * l1_cosl1_sinl1*l2_cosl2Pow2*cosl3_sinl3 //checked
          + t11*t22/(t31*t32)   * l1_cosl1Pow2 * l2_sinl2Pow2*sinl3Pow2
          - 2*t11*t22/(t31*t32) * l1_cosl1Pow2 * l2_cosl2_sinl2*cosl3_sinl3
          + t11*t22/(t31*t32)   * l1_cosl1Pow2 * l2_cosl2Pow2*cosl3Pow2
          + t11*t23/(t31*t33) * l1_sinl1Pow2*sinl2Pow2*l3_cosl3Pow2
          + 2*t11*t23/(t31*t33) * l1_sinl1Pow2*cosl2_sinl2*l3_cosl3_sinl3
          + t11*t23/(t31*t33) * l1_sinl1Pow2*l3_sinl3Pow2*cosl2Pow2
          + 2*t11*t23/(t31*t33) * sinl2Pow2*l1_cosl1_sinl1*l3_cosl3_sinl3
          + 2*t11*t23/(t31*t33) * l1_cosl1_sinl1*cosl2_sinl2*l3_sinl3Pow2
          - 2*t11*t23/(t31*t33) * l1_cosl1_sinl1*cosl2_sinl2*l3_cosl3Pow2
          - 2*t11*t23/(t31*t33) * l1_cosl1_sinl1*cosl2Pow2*l3_cosl3_sinl3
          + t11*t23/(t31*t33) * sinl2Pow2*l3_sinl3Pow2*l1_cosl1Pow2
          - 2*t11*t23/(t31*t33) * cosl2_sinl2*l3_cosl3_sinl3*l1_cosl1Pow2
          + t11*t23/(t31*t33) * l1_cosl1Pow2*cosl2Pow2*l3_cosl3Pow2
          + t12*t21/(t31*t32) * l1_sinl1Pow2*l2_sinl2Pow2*cosl3Pow2
          + 2*t12*t21/(t31*t32) * l1_sinl1Pow2*l2_cosl2_sinl2*cosl3_sinl3
          + t12*t21/(t31*t32) * l1_sinl1Pow2*sinl3Pow2*l2_cosl2Pow2
          + 2*t12*t21/(t31*t32) * l2_sinl2Pow2*l1_cosl1_sinl1*cosl3_sinl3
          + 2*t12*t21/(t31*t32) * l1_cosl1_sinl1*l2_cosl2_sinl2*sinl3Pow2
          - 2*t12*t21/(t31*t32) * l1_cosl1_sinl1*l2_cosl2_sinl2*cosl3Pow2
          - 2*t12*t21/(t31*t32) * l1_cosl1_sinl1*l2_cosl2Pow2*cosl3_sinl3
          + t12*t21/(t31*t32) * l2_sinl2Pow2*sinl3Pow2*l1_cosl1Pow2
          - 2*t12*t21/(t31*t32) * l2_cosl2_sinl2*cosl3_sinl3*l1_cosl1Pow2
          + t12*t21/(t31*t32) * l1_cosl1Pow2*l2_cosl2Pow2*cosl3Pow2
          + t12*t22/(t32*t32) * l2Pow2_sinl2Pow2*sinl1Pow2*cosl3Pow2
          + 2*t12*t22/(t32*t32) * l2Pow2_cosl2_sinl2*sinl1Pow2*cosl3_sinl3
          + t12*t22/(t32*t32) * l2Pow2_cosl2Pow2*sinl1Pow2*sinl3Pow2
          + 2*t12*t22/(t32*t32) * l2Pow2_sinl2Pow2*cosl1_sinl1*cosl3_sinl3
          + 2*t12*t22/(t32*t32) * l2Pow2_cosl2_sinl2*cosl1_sinl1*sinl3Pow2
          - 2*t12*t22/(t32*t32) * l2Pow2_cosl2_sinl2*cosl1_sinl1*cosl3Pow2
          - 2*t12*t22/(t32*t32) * l2Pow2_cosl2Pow2*cosl1_sinl1*cosl3_sinl3
          + t12*t22/(t32*t32) * l2Pow2_sinl2Pow2*sinl3Pow2*cosl1Pow2
          - 2*t12*t22/(t32*t32) * l2Pow2_cosl2_sinl2*cosl3_sinl3*cosl1Pow2
          + t12*t22/(t32*t32) * l2Pow2_cosl2Pow2*cosl1Pow2*cosl3Pow2
          + t12*t23/(t32*t33) * sinl1Pow2*l2_sinl2Pow2*l3_cosl3Pow2
          + 2*t12*t23/(t32*t33) * sinl1Pow2*l2_cosl2_sinl2*l3_cosl3_sinl3
          + t12*t23/(t32*t33) * sinl1Pow2*l3_sinl3Pow2*l2_cosl2Pow2
          + 2*t12*t23/(t32*t33) * l2_sinl2Pow2*cosl1_sinl1*l3_cosl3_sinl3
          + 2*t12*t23/(t32*t33) * cosl1_sinl1*l2_cosl2_sinl2*l3_sinl3Pow2
          - 2*t12*t23/(t32*t33) * cosl1_sinl1*l2_cosl2_sinl2*l3_cosl3Pow2
          - 2*t12*t23/(t32*t33) * cosl1_sinl1*l2_cosl2Pow2*l3_cosl3_sinl3
          + t12*t23/(t32*t33) * l2_sinl2Pow2*l3_sinl3Pow2*cosl1Pow2
          - 2*t12*t23/(t32*t33) * l2_cosl2_sinl2*l3_cosl3_sinl3*cosl1Pow2
          + t12*t23/(t32*t33) * cosl1Pow2*l2_cosl2Pow2*l3_cosl3Pow2
          + t13*t21/(t31*t33) * l1_sinl1Pow2*sinl2Pow2*l3_cosl3Pow2
          + 2*t13*t21/(t31*t33) * l1_sinl1Pow2*cosl2_sinl2*l3_cosl3_sinl3
          + t13*t21/(t31*t33) * l1_sinl1Pow2*l3_sinl3Pow2*cosl2Pow2
          + 2*t13*t21/(t31*t33) * sinl2Pow2*l1_cosl1_sinl1*l3_cosl3_sinl3
          + 2*t13*t21/(t31*t33) * l1_cosl1_sinl1*cosl2_sinl2*l3_sinl3Pow2
          - 2*t13*t21/(t31*t33) * l1_cosl1_sinl1*cosl2_sinl2*l3_cosl3Pow2
          - 2*t13*t21/(t31*t33) * l1_cosl1_sinl1*cosl2Pow2*l3_cosl3_sinl3
          + t13*t21/(t31*t33) * sinl2Pow2*l3_sinl3Pow2*l1_cosl1Pow2
          - 2*t13*t21/(t31*t33) * cosl2_sinl2*l3_cosl3_sinl3*l1_cosl1Pow2
          + t13*t21/(t31*t33) * l1_cosl1Pow2*cosl2Pow2*l3_cosl3Pow2
          + t13*t22/(t32*t33) * sinl1Pow2*l2_sinl2Pow2*l3_cosl3Pow2
          + 2*t13*t22/(t32*t33) * sinl1Pow2*l2_cosl2_sinl2*l3_cosl3_sinl3
          + t13*t22/(t32*t33) * sinl1Pow2*l3_sinl3Pow2*l2_cosl2Pow2
          + 2*t13*t22/(t32*t33) * l2_sinl2Pow2*cosl1_sinl1*l3_cosl3_sinl3
          + 2*t13*t22/(t32*t33) * cosl1_sinl1*l2_cosl2_sinl2*l3_sinl3Pow2
          - 2*t13*t22/(t32*t33) * cosl1_sinl1*l2_cosl2_sinl2*l3_cosl3Pow2
          - 2*t13*t22/(t32*t33) * cosl1_sinl1*l2_cosl2Pow2*l3_cosl3_sinl3
          + t13*t22/(t32*t33) * l2_sinl2Pow2*l3_sinl3Pow2*cosl1Pow2
          - 2*t13*t22/(t32*t33) * l2_cosl2_sinl2*l3_cosl3_sinl3*cosl1Pow2
          + t13*t22/(t32*t33) * cosl1Pow2*l2_cosl2Pow2*l3_cosl3Pow2
          + t13*t23/(t33*t33) * l3Pow2_cosl3Pow2*sinl1Pow2*sinl2Pow2
          + 2*t13*t23/(t33*t33) * l3Pow2_cosl3_sinl3*sinl1Pow2*cosl2_sinl2
          + t13*t23/(t33*t33) * l3Pow2_sinl3Pow2*sinl1Pow2*cosl2Pow2
          + 2*t13*t23/(t33*t33) * l3Pow2_cosl3_sinl3*sinl2Pow2*cosl1_sinl1
          + 2*t13*t23/(t33*t33) * l3Pow2_sinl3Pow2*cosl1_sinl1*cosl2_sinl2
          - 2*t13*t23/(t33*t33) * l3Pow2_cosl3Pow2*cosl1_sinl1*cosl2_sinl2
          - 2*t13*t23/(t33*t33) * l3Pow2_cosl3_sinl3*cosl1_sinl1*cosl2Pow2
          + t13*t23/(t33*t33) * l3Pow2_sinl3Pow2*sinl2Pow2*cosl1Pow2
          - 2*t13*t23/(t33*t33) * l3Pow2_cosl3_sinl3*cosl2_sinl2*cosl1Pow2
          + t13*t23/(t33*t33) * l3Pow2_cosl3Pow2*cosl1Pow2*cosl2Pow2;
}

double ThreeDimensionalNormalDistribution::calc_xy_sin_z_sin_z_moment()
{
    if (!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if (std::fabs(covariance_(0, 2)) < cov_threshold_ && std::fabs(covariance_(1, 2)) < cov_threshold_ && std::fabs(covariance_(0,1))<cov_threshold_) {
        NormalDistribution dist_x(mean_(0), covariance_(0, 0));
        NormalDistribution dist_y(mean_(1), covariance_(1, 1));
        NormalDistribution dist_z(mean_(2), covariance_(2, 2));

        return dist_x.calc_moment(1) * dist_y.calc_moment(1) * dist_z.calc_sin_moment(2);
    }

    const auto y_mean = T_.transpose() * mean_;
    const double t11 = T_(0, 0);
    const double t12 = T_(0, 1);
    const double t13 = T_(0, 2);
    const double t21 = T_(1, 0);
    const double t22 = T_(1, 1);
    const double t23 = T_(1, 2);
    const double t31 = T_(2, 0);
    const double t32 = T_(2, 1);
    const double t33 = T_(2, 2);

    const double l1_mean = t31 * y_mean(0);
    const double l1_cov = t31 * t31 / eigen_values_(0);
    NormalDistribution normal1(l1_mean, l1_cov);

    const double l2_mean = t32 * y_mean(1);
    const double l2_cov = t32 * t32 / eigen_values_(1);
    NormalDistribution normal2(l2_mean, l2_cov);

    const double l3_mean = t33 * y_mean(2);
    const double l3_cov = t33 * t33 / eigen_values_(2);
    NormalDistribution normal3(l3_mean, l3_cov);

    // compute E[xycos(theta)^2]
    const double sinl1Pow2 = normal1.calc_sin_moment(2);
    const double cosl1Pow2 = normal1.calc_cos_moment(2);
    const double sinl2Pow2 = normal2.calc_sin_moment(2);
    const double cosl2Pow2 = normal2.calc_cos_moment(2);
    const double sinl3Pow2 = normal3.calc_sin_moment(2);
    const double cosl3Pow2 = normal3.calc_cos_moment(2);
    const double cosl1_sinl1 = normal1.calc_cos_sin_moment(1, 1);
    const double cosl2_sinl2 = normal2.calc_cos_sin_moment(1, 1);
    const double cosl3_sinl3 = normal3.calc_cos_sin_moment(1, 1);

    const double l1_cosl1_sinl1 = normal1.calc_x_cos_sin_moment(1, 1, 1);
    const double l2_cosl2_sinl2 = normal2.calc_x_cos_sin_moment(1, 1, 1);
    const double l3_cosl3_sinl3 = normal3.calc_x_cos_sin_moment(1, 1, 1);
    const double l1Pow2_cosl1_sinl1 = normal1.calc_x_cos_sin_moment(2, 1, 1);
    const double l2Pow2_cosl2_sinl2 = normal2.calc_x_cos_sin_moment(2, 1, 1);
    const double l3Pow2_cosl3_sinl3 = normal3.calc_x_cos_sin_moment(2, 1, 1);
    const double l1_cosl1Pow2 = normal1.calc_x_cos_moment(1, 2);
    const double l1_sinl1Pow2 = normal1.calc_x_sin_moment(1, 2);
    const double l2_cosl2Pow2 = normal2.calc_x_cos_moment(1, 2);
    const double l2_sinl2Pow2 = normal2.calc_x_sin_moment(1, 2);
    const double l3_cosl3Pow2 = normal3.calc_x_cos_moment(1, 2);
    const double l3_sinl3Pow2 = normal3.calc_x_sin_moment(1, 2);
    const double l1Pow2_cosl1Pow2 = normal1.calc_x_cos_moment(2, 2);
    const double l1Pow2_sinl1Pow2 = normal1.calc_x_sin_moment(2, 2);
    const double l2Pow2_cosl2Pow2 = normal2.calc_x_cos_moment(2, 2);
    const double l2Pow2_sinl2Pow2 = normal2.calc_x_sin_moment(2, 2);
    const double l3Pow2_cosl3Pow2 = normal3.calc_x_cos_moment(2, 2);
    const double l3Pow2_sinl3Pow2 = normal3.calc_x_sin_moment(2, 2);

    return  t11*t21/(t31*t31) * l1Pow2_sinl1Pow2*sinl2Pow2*sinl3Pow2
          - 2*t11*t21/(t31*t31) * l1Pow2_sinl1Pow2*cosl2_sinl2*cosl3_sinl3
          + t11*t21/(t31*t31) * l1Pow2_sinl1Pow2*cosl2Pow2*cosl3Pow2
          - 2*t11*t21/(t31*t31) * l1Pow2_cosl1_sinl1*sinl2Pow2*cosl3_sinl3
          - 2*t11*t21/(t31*t31) * l1Pow2_cosl1_sinl1*cosl2_sinl2*sinl3Pow2
          + 2*t11*t21/(t31*t31) * l1Pow2_cosl1_sinl1*cosl2_sinl2*cosl3Pow2
          + 2*t11*t21/(t31*t31) * l1Pow2_cosl1_sinl1*cosl2Pow2*cosl3_sinl3
          + t11*t21/(t31*t31) * l1Pow2_cosl1Pow2*sinl2Pow2*cosl3Pow2
          + 2*t11*t21/(t31*t31) * l1Pow2_cosl1Pow2*cosl2_sinl2*cosl3_sinl3
          + t11*t21/(t31*t31) * l1Pow2_cosl1Pow2*sinl3Pow2*cosl2Pow2
          + t11*t22/(t31*t32) * l1_sinl1Pow2*l2_sinl2Pow2*sinl3Pow2
          - 2*t11*t22/(t31*t32) * l1_sinl1Pow2*l2_cosl2_sinl2*cosl3_sinl3
          + t11*t22/(t31*t32) * l1_sinl1Pow2*l2_cosl2Pow2*cosl3Pow2
          - 2*t11*t22/(t31*t32) * l1_cosl1_sinl1*l2_sinl2Pow2*cosl3_sinl3
          - 2*t11*t22/(t31*t32) * l1_cosl1_sinl1*l2_cosl2_sinl2*sinl3Pow2
          + 2*t11*t22/(t31*t32) * l1_cosl1_sinl1*l2_cosl2_sinl2*cosl3Pow2
          + 2*t11*t22/(t31*t32) * l1_cosl1_sinl1*l2_cosl2Pow2*cosl3_sinl3
          + t11*t22/(t31*t32) * l2_sinl2Pow2*l1_cosl1Pow2*cosl3Pow2
          + 2*t11*t22/(t31*t32) * l1_cosl1Pow2*l2_cosl2_sinl2*cosl3_sinl3
          + t11*t22/(t31*t32) * sinl3Pow2*l1_cosl1Pow2*l2_cosl2Pow2
          + t11*t23/(t31*t33) * l1_sinl1Pow2*sinl2Pow2*l3_sinl3Pow2
          - 2*t11*t23/(t31*t33) * l1_sinl1Pow2*cosl2_sinl2*l3_cosl3_sinl3
          + t11*t23/(t31*t33) * l1_sinl1Pow2*cosl2Pow2*l3_cosl3Pow2
          - 2*t11*t23/(t31*t33) * l1_cosl1_sinl1*sinl2Pow2*l3_cosl3_sinl3
          - 2*t11*t23/(t31*t33) * l1_cosl1_sinl1*cosl2_sinl2*l3_sinl3Pow2
          + 2*t11*t23/(t31*t33) * l1_cosl1_sinl1*cosl2_sinl2*l3_cosl3Pow2
          + 2*t11*t23/(t31*t33) * l1_cosl1_sinl1*cosl2Pow2*l3_cosl3_sinl3
          + t11*t23/(t31*t33) * sinl2Pow2*l1_cosl1Pow2*l3_cosl3Pow2
          + 2*t11*t23/(t31*t33) * l1_cosl1Pow2*cosl2_sinl2*l3_cosl3_sinl3
          + t11*t23/(t31*t33) * l3_sinl3Pow2*l1_cosl1Pow2*cosl2Pow2
          + t12*t21/(t31*t32) * l1_sinl1Pow2*l2_sinl2Pow2*sinl3Pow2
          - 2*t12*t21/(t31*t32) * l1_sinl1Pow2*l2_cosl2_sinl2*cosl3_sinl3
          + t12*t21/(t31*t32) * l1_sinl1Pow2*l2_cosl2Pow2*cosl3Pow2
          - 2*t12*t21/(t31*t32) * l1_cosl1_sinl1*l2_sinl2Pow2*cosl3_sinl3
          - 2*t12*t21/(t31*t32) * l1_cosl1_sinl1*l2_cosl2_sinl2*sinl3Pow2
          + 2*t12*t21/(t31*t32) * l1_cosl1_sinl1*l2_cosl2_sinl2*cosl3Pow2
          + 2*t12*t21/(t31*t32) * l1_cosl1_sinl1*l2_cosl2Pow2*cosl3_sinl3
          + t12*t21/(t31*t32) * l2_sinl2Pow2*l1_cosl1Pow2*cosl3Pow2
          + 2*t12*t21/(t31*t32) * l1_cosl1Pow2*l2_cosl2_sinl2*cosl3_sinl3
          + t12*t21/(t31*t32) * sinl3Pow2*l1_cosl1Pow2*l2_cosl2Pow2
          + t12*t22/(t32*t32) * l2Pow2_sinl2Pow2*sinl1Pow2*sinl3Pow2
          - 2*t12*t22/(t32*t32) * l2Pow2_cosl2_sinl2*sinl1Pow2*cosl3_sinl3
          + t12*t22/(t32*t32) * l2Pow2_cosl2Pow2*sinl1Pow2*cosl3Pow2
          - 2*t12*t22/(t32*t32) * l2Pow2_sinl2Pow2*cosl1_sinl1*cosl3_sinl3
          - 2*t12*t22/(t32*t32) * l2Pow2_cosl2_sinl2*cosl1_sinl1*sinl3Pow2
          + 2*t12*t22/(t32*t32) * l2Pow2_cosl2_sinl2*cosl1_sinl1*cosl3Pow2
          + 2*t12*t22/(t32*t32) * l2Pow2_cosl2Pow2*cosl1_sinl1*cosl3_sinl3
          + t12*t22/(t32*t32) * l2Pow2_sinl2Pow2*cosl1Pow2*cosl3Pow2
          + 2*t12*t22/(t32*t32) * l2Pow2_cosl2_sinl2*cosl1Pow2*cosl3_sinl3
          + t12*t22/(t32*t32) * l2Pow2_cosl2Pow2*sinl3Pow2*cosl1Pow2
          + t12*t23/(t32*t33) * sinl1Pow2*l2_sinl2Pow2*l3_sinl3Pow2
          - 2*t12*t23/(t32*t33) * sinl1Pow2*l2_cosl2_sinl2*l3_cosl3_sinl3
          + t12*t23/(t32*t33) * sinl1Pow2*l2_cosl2Pow2*l3_cosl3Pow2
          - 2*t12*t23/(t32*t33) * cosl1_sinl1*l2_sinl2Pow2*l3_cosl3_sinl3
          - 2*t12*t23/(t32*t33) * l3_sinl3Pow2*cosl1_sinl1*l2_cosl2_sinl2
          + 2*t12*t23/(t32*t33) * cosl1_sinl1*l2_cosl2_sinl2*l3_cosl3Pow2
          + 2*t12*t23/(t32*t33) * cosl1_sinl1*l2_cosl2Pow2*l3_cosl3_sinl3
          + t12*t23/(t32*t33) * l2_sinl2Pow2*cosl1Pow2*l3_cosl3Pow2
          + 2*t12*t23/(t32*t33) * cosl1Pow2*l2_cosl2_sinl2*l3_cosl3_sinl3
          + t12*t23/(t32*t33) * l3_sinl3Pow2*cosl1Pow2*l2_cosl2Pow2
          + t13*t21/(t31*t33) * l1_sinl1Pow2*sinl2Pow2*l3_sinl3Pow2
          - 2*t13*t21/(t31*t33) * l1_sinl1Pow2*cosl2_sinl2*l3_cosl3_sinl3
          + t13*t21/(t31*t33) * l1_sinl1Pow2*cosl2Pow2*l3_cosl3Pow2
          - 2*t13*t21/(t31*t33) * l1_cosl1_sinl1*sinl2Pow2*l3_cosl3_sinl3
          - 2*t13*t21/(t31*t33) * l3_sinl3Pow2*l1_cosl1_sinl1*cosl2_sinl2
          + 2*t13*t21/(t31*t33) * l1_cosl1_sinl1*cosl2_sinl2*l3_cosl3Pow2
          + 2*t13*t21/(t31*t33) * l1_cosl1_sinl1*cosl2Pow2*l3_cosl3_sinl3
          + t13*t21/(t31*t33) * sinl2Pow2*l1_cosl1Pow2*l3_cosl3Pow2
          + 2*t13*t21/(t31*t33) * l1_cosl1Pow2*cosl2_sinl2*l3_cosl3_sinl3
          + t13*t21/(t31*t33) * l3_sinl3Pow2*l1_cosl1Pow2*cosl2Pow2
          + t13*t22/(t32*t33) * sinl1Pow2*l2_sinl2Pow2*l3_sinl3Pow2
          - 2*t13*t22/(t32*t33) * sinl1Pow2*l2_cosl2_sinl2*l3_cosl3_sinl3
          + t13*t22/(t32*t33) * sinl1Pow2*l2_cosl2Pow2*l3_cosl3Pow2
          - 2*t13*t22/(t32*t33) * cosl1_sinl1*l2_sinl2Pow2*l3_cosl3_sinl3
          - 2*t13*t22/(t32*t33) * l3_sinl3Pow2*cosl1_sinl1*l2_cosl2_sinl2
          + 2*t13*t22/(t32*t33) * cosl1_sinl1*l2_cosl2_sinl2*l3_cosl3Pow2
          + 2*t13*t22/(t32*t33) * cosl1_sinl1*l2_cosl2Pow2*l3_cosl3_sinl3
          + t13*t22/(t32*t33) * l2_sinl2Pow2*cosl1Pow2*l3_cosl3Pow2
          + 2*t13*t22/(t32*t33) * cosl1Pow2*l2_cosl2_sinl2*l3_cosl3_sinl3
          + t13*t22/(t32*t33) * l3_sinl3Pow2*cosl1Pow2*l2_cosl2Pow2
          + t13*t23/(t33*t33) * l3Pow2_sinl3Pow2*sinl1Pow2*sinl2Pow2
          - 2*t13*t23/(t33*t33) * l3Pow2_cosl3_sinl3*sinl1Pow2*cosl2_sinl2
          + t13*t23/(t33*t33) * l3Pow2_cosl3Pow2*sinl1Pow2*cosl2Pow2
          - 2*t13*t23/(t33*t33) * l3Pow2_cosl3_sinl3*cosl1_sinl1*sinl2Pow2
          - 2*t13*t23/(t33*t33) * l3Pow2_sinl3Pow2*cosl1_sinl1*cosl2_sinl2
          + 2*t13*t23/(t33*t33) * l3Pow2_cosl3Pow2*cosl1_sinl1*cosl2_sinl2
          + 2*t13*t23/(t33*t33) * l3Pow2_cosl3_sinl3*cosl1_sinl1*cosl2Pow2
          + t13*t23/(t33*t33) * l3Pow2_cosl3Pow2*sinl2Pow2*cosl1Pow2
          + 2*t13*t23/(t33*t33) * l3Pow2_cosl3_sinl3*cosl1Pow2*cosl2_sinl2
          + t13*t23/(t33*t33) * l3Pow2_sinl3Pow2*cosl1Pow2*cosl2Pow2;
}

double ThreeDimensionalNormalDistribution::calc_xy_cos_z_sin_z_moment()
{
    if (!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if (std::fabs(covariance_(0, 2)) < cov_threshold_ && std::fabs(covariance_(1, 2)) < cov_threshold_ && std::fabs(covariance_(0,1))<cov_threshold_) {
        NormalDistribution dist_x(mean_(0), covariance_(0, 0));
        NormalDistribution dist_y(mean_(1), covariance_(1, 1));
        NormalDistribution dist_z(mean_(2), covariance_(2, 2));

        return dist_x.calc_moment(1) * dist_y.calc_moment(1) * dist_z.calc_cos_sin_moment(1, 1);
    }

    const auto y_mean = T_.transpose() * mean_;
    const double t11 = T_(0, 0);
    const double t12 = T_(0, 1);
    const double t13 = T_(0, 2);
    const double t21 = T_(1, 0);
    const double t22 = T_(1, 1);
    const double t23 = T_(1, 2);
    const double t31 = T_(2, 0);
    const double t32 = T_(2, 1);
    const double t33 = T_(2, 2);

    const double l1_mean = t31 * y_mean(0);
    const double l1_cov = t31 * t31 / eigen_values_(0);
    NormalDistribution normal1(l1_mean, l1_cov);

    const double l2_mean = t32 * y_mean(1);
    const double l2_cov = t32 * t32 / eigen_values_(1);
    NormalDistribution normal2(l2_mean, l2_cov);

    const double l3_mean = t33 * y_mean(2);
    const double l3_cov = t33 * t33 / eigen_values_(2);
    NormalDistribution normal3(l3_mean, l3_cov);

    // compute E[xycos(theta)^2]
    const double sinl1Pow2 = normal1.calc_sin_moment(2);
    const double cosl1Pow2 = normal1.calc_cos_moment(2);
    const double sinl2Pow2 = normal2.calc_sin_moment(2);
    const double cosl2Pow2 = normal2.calc_cos_moment(2);
    const double sinl3Pow2 = normal3.calc_sin_moment(2);
    const double cosl3Pow2 = normal3.calc_cos_moment(2);
    const double cosl1_sinl1 = normal1.calc_cos_sin_moment(1, 1);
    const double cosl2_sinl2 = normal2.calc_cos_sin_moment(1, 1);
    const double cosl3_sinl3 = normal3.calc_cos_sin_moment(1, 1);

    const double l1_cosl1_sinl1 = normal1.calc_x_cos_sin_moment(1, 1, 1);
    const double l2_cosl2_sinl2 = normal2.calc_x_cos_sin_moment(1, 1, 1);
    const double l3_cosl3_sinl3 = normal3.calc_x_cos_sin_moment(1, 1, 1);
    const double l1Pow2_cosl1_sinl1 = normal1.calc_x_cos_sin_moment(2, 1, 1);
    const double l2Pow2_cosl2_sinl2 = normal2.calc_x_cos_sin_moment(2, 1, 1);
    const double l3Pow2_cosl3_sinl3 = normal3.calc_x_cos_sin_moment(2, 1, 1);
    const double l1_cosl1Pow2 = normal1.calc_x_cos_moment(1, 2);
    const double l1_sinl1Pow2 = normal1.calc_x_sin_moment(1, 2);
    const double l2_cosl2Pow2 = normal2.calc_x_cos_moment(1, 2);
    const double l2_sinl2Pow2 = normal2.calc_x_sin_moment(1, 2);
    const double l3_cosl3Pow2 = normal3.calc_x_cos_moment(1, 2);
    const double l3_sinl3Pow2 = normal3.calc_x_sin_moment(1, 2);
    const double l1Pow2_cosl1Pow2 = normal1.calc_x_cos_moment(2, 2);
    const double l1Pow2_sinl1Pow2 = normal1.calc_x_sin_moment(2, 2);
    const double l2Pow2_cosl2Pow2 = normal2.calc_x_cos_moment(2, 2);
    const double l2Pow2_sinl2Pow2 = normal2.calc_x_sin_moment(2, 2);
    const double l3Pow2_cosl3Pow2 = normal3.calc_x_cos_moment(2, 2);
    const double l3Pow2_sinl3Pow2 = normal3.calc_x_sin_moment(2, 2);

    return  t11*t21/(t31*t31) * l1Pow2_sinl1Pow2*sinl2Pow2*cosl3_sinl3
          + t11*t21/(t31*t31) * l1Pow2_sinl1Pow2*cosl2_sinl2*sinl3Pow2
          - t11*t21/(t31*t31) * l1Pow2_sinl1Pow2*cosl2_sinl2*cosl3Pow2
          - t11*t21/(t31*t31) * l1Pow2_sinl1Pow2*cosl2Pow2*cosl3_sinl3
          + t11*t21/(t31*t31)   * l1Pow2_cosl1_sinl1*sinl2Pow2*sinl3Pow2
          - t11*t21/(t31*t31)   * l1Pow2_cosl1_sinl1*sinl2Pow2*cosl3Pow2
          - 4*t11*t21/(t31*t31) * l1Pow2_cosl1_sinl1*cosl2_sinl2*cosl3_sinl3
          - t11*t21/(t31*t31)   * l1Pow2_cosl1_sinl1*sinl3Pow2*cosl2Pow2
          + t11*t21/(t31*t31)   * l1Pow2_cosl1_sinl1*cosl2Pow2*cosl3Pow2
          - t11*t21/(t31*t31) * l1Pow2_cosl1Pow2*sinl2Pow2*cosl3_sinl3
          - t11*t21/(t31*t31) * l1Pow2_cosl1Pow2*cosl2_sinl2*sinl3Pow2
          + t11*t21/(t31*t31) * l1Pow2_cosl1Pow2*cosl2_sinl2*cosl3Pow2
          + t11*t21/(t31*t31) * l1Pow2_cosl1Pow2*cosl2Pow2*cosl3_sinl3
          + t11*t22/(t31*t32) * l1_sinl1Pow2*l2_sinl2Pow2*cosl3_sinl3
          + t11*t22/(t31*t32) * l1_sinl1Pow2*l2_cosl2_sinl2*sinl3Pow2
          - t11*t22/(t31*t32) * l1_sinl1Pow2*l2_cosl2_sinl2*cosl3Pow2
          - t11*t22/(t31*t32) * l1_sinl1Pow2*l2_cosl2Pow2*cosl3_sinl3
          + t11*t22/(t31*t32) * l1_cosl1_sinl1*l2_sinl2Pow2*sinl3Pow2
          - t11*t22/(t31*t32) * l1_cosl1_sinl1*l2_sinl2Pow2*cosl3Pow2
          - 4*t11*t22/(t31*t32) * l1_cosl1_sinl1*l2_cosl2_sinl2*cosl3_sinl3
          - t11*t22/(t31*t32) * sinl3Pow2*l1_cosl1_sinl1*l2_cosl2Pow2
          + t11*t22/(t31*t32) * l1_cosl1_sinl1*l2_cosl2Pow2*cosl3Pow2
          - t11*t22/(t31*t32) * l2_sinl2Pow2*l1_cosl1Pow2*cosl3_sinl3
          - t11*t22/(t31*t32) * l2_cosl2_sinl2*sinl3Pow2*l1_cosl1Pow2
          + t11*t22/(t31*t32) * l2_cosl2_sinl2*l1_cosl1Pow2*cosl3Pow2
          + t11*t22/(t31*t32) * l1_cosl1Pow2*l2_cosl2Pow2*cosl3_sinl3
          + t11*t23/(t31*t33)* l1_sinl1Pow2*sinl2Pow2*l3_cosl3_sinl3
          + t11*t23/(t31*t33)* l1_sinl1Pow2*cosl2_sinl2*l3_sinl3Pow2
          - t11*t23/(t31*t33)* l1_sinl1Pow2*cosl2_sinl2*l3_cosl3Pow2
          - t11*t23/(t31*t33)* l1_sinl1Pow2*cosl2Pow2*l3_cosl3_sinl3
          + t11*t23/(t31*t33)* l1_cosl1_sinl1*sinl2Pow2*l3_sinl3Pow2
          - t11*t23/(t31*t33)* l1_cosl1_sinl1*sinl2Pow2*l3_cosl3Pow2
          - 4*t11*t23/(t31*t33)* l1_cosl1_sinl1*cosl2_sinl2*l3_cosl3_sinl3
          - t11*t23/(t31*t33)* l3_sinl3Pow2*l1_cosl1_sinl1*cosl2Pow2
          + t11*t23/(t31*t33)* l1_cosl1_sinl1*cosl2Pow2*l3_cosl3Pow2
          - t11*t23/(t31*t33)* sinl2Pow2*l1_cosl1Pow2*l3_cosl3_sinl3
          - t11*t23/(t31*t33)* l3_sinl3Pow2*l1_cosl1Pow2*cosl2_sinl2
          + t11*t23/(t31*t33)* cosl2_sinl2*l1_cosl1Pow2*l3_cosl3Pow2
          + t11*t23/(t31*t33)* l3_cosl3_sinl3*l1_cosl1Pow2*cosl2Pow2
          + t12*t21/(t31*t32) * l1_sinl1Pow2*l2_sinl2Pow2*cosl3_sinl3
          + t12*t21/(t31*t32) * l1_sinl1Pow2*l2_cosl2_sinl2*sinl3Pow2
          - t12*t21/(t31*t32) * l1_sinl1Pow2*l2_cosl2_sinl2*cosl3Pow2
          - t12*t21/(t31*t32) * l1_sinl1Pow2*l2_cosl2Pow2*cosl3_sinl3
          + t12*t21/(t31*t32) * l1_cosl1_sinl1*l2_sinl2Pow2*sinl3Pow2
          - t12*t21/(t31*t32) * l1_cosl1_sinl1*l2_sinl2Pow2*cosl3Pow2
          - 4*t12*t21/(t31*t32) * l1_cosl1_sinl1*l2_cosl2_sinl2*cosl3_sinl3
          - t12*t21/(t31*t32) * sinl3Pow2*l1_cosl1_sinl1*l2_cosl2Pow2
          + t12*t21/(t31*t32) * l1_cosl1_sinl1*l2_cosl2Pow2*cosl3Pow2
          - t12*t21/(t31*t32) * l2_sinl2Pow2*l1_cosl1Pow2*cosl3_sinl3
          - t12*t21/(t31*t32) * l2_cosl2_sinl2*sinl3Pow2*l1_cosl1Pow2
          + t12*t21/(t31*t32) * l2_cosl2_sinl2*l1_cosl1Pow2*cosl3Pow2
          + t12*t21/(t31*t32) * l1_cosl1Pow2*l2_cosl2Pow2*cosl3_sinl3
          + t12*t22/(t32*t32) * l2Pow2_sinl2Pow2*sinl1Pow2*cosl3_sinl3
          + t12*t22/(t32*t32) * l2Pow2_cosl2_sinl2*sinl1Pow2*sinl3Pow2
          - t12*t22/(t32*t32) * l2Pow2_cosl2_sinl2*sinl1Pow2*cosl3Pow2
          - t12*t22/(t32*t32) * l2Pow2_cosl2Pow2*sinl1Pow2*cosl3_sinl3
          + t12*t22/(t32*t32) * l2Pow2_sinl2Pow2*sinl3Pow2*cosl1_sinl1
          - t12*t22/(t32*t32) * l2Pow2_sinl2Pow2*cosl1_sinl1*cosl3Pow2
          - 4*t12*t22/(t32*t32) * l2Pow2_cosl2_sinl2*cosl1_sinl1*cosl3_sinl3
          - t12*t22/(t32*t32) * l2Pow2_cosl2Pow2*sinl3Pow2*cosl1_sinl1
          + t12*t22/(t32*t32) * l2Pow2_cosl2Pow2*cosl1_sinl1*cosl3Pow2
          - t12*t22/(t32*t32) * l2Pow2_sinl2Pow2*cosl1Pow2*cosl3_sinl3
          - t12*t22/(t32*t32) * l2Pow2_cosl2_sinl2*sinl3Pow2*cosl1Pow2
          + t12*t22/(t32*t32) * l2Pow2_cosl2_sinl2*cosl1Pow2*cosl3Pow2
          + t12*t22/(t32*t32) * l2Pow2_cosl2Pow2*cosl1Pow2*cosl3_sinl3
          + t12*t23/(t32*t33) * sinl1Pow2*l2_sinl2Pow2*l3_cosl3_sinl3
          + t12*t23/(t32*t33) * sinl1Pow2*l2_cosl2_sinl2*l3_sinl3Pow2
          - t12*t23/(t32*t33) * sinl1Pow2*l2_cosl2_sinl2*l3_cosl3Pow2
          - t12*t23/(t32*t33) * sinl1Pow2*l2_cosl2Pow2*l3_cosl3_sinl3
          + t12*t23/(t32*t33) * l2_sinl2Pow2*l3_sinl3Pow2*cosl1_sinl1
          - t12*t23/(t32*t33) * l2_sinl2Pow2*cosl1_sinl1*l3_cosl3Pow2
          - 4*t12*t23/(t32*t33) * cosl1_sinl1*l2_cosl2_sinl2*l3_cosl3_sinl3
          - t12*t23/(t32*t33) * l3_sinl3Pow2*cosl1_sinl1*l2_cosl2Pow2
          + t12*t23/(t32*t33) * cosl1_sinl1*l2_cosl2Pow2*l3_cosl3Pow2
          - t12*t23/(t32*t33) * l2_sinl2Pow2*cosl1Pow2*l3_cosl3_sinl3
          - t12*t23/(t32*t33) * l2_cosl2_sinl2*l3_sinl3Pow2*cosl1Pow2
          + t12*t23/(t32*t33) * l2_cosl2_sinl2*cosl1Pow2*l3_cosl3Pow2
          + t12*t23/(t32*t33) * l3_cosl3_sinl3*cosl1Pow2*l2_cosl2Pow2
          + t13*t21/(t31*t33)* l1_sinl1Pow2*sinl2Pow2*l3_cosl3_sinl3
          + t13*t21/(t31*t33)* l1_sinl1Pow2*cosl2_sinl2*l3_sinl3Pow2
          - t13*t21/(t31*t33)* l1_sinl1Pow2*cosl2_sinl2*l3_cosl3Pow2
          - t13*t21/(t31*t33)* l1_sinl1Pow2*cosl2Pow2*l3_cosl3_sinl3
          + t13*t21/(t31*t33)* sinl2Pow2*l3_sinl3Pow2*l1_cosl1_sinl1
          - t13*t21/(t31*t33)* sinl2Pow2*l1_cosl1_sinl1*l3_cosl3Pow2
          - 4*t13*t21/(t31*t33)* l1_cosl1_sinl1*cosl2_sinl2*l3_cosl3_sinl3
          - t13*t21/(t31*t33)* l3_sinl3Pow2*l1_cosl1_sinl1*cosl2Pow2
          + t13*t21/(t31*t33)* l1_cosl1_sinl1*cosl2Pow2*l3_cosl3Pow2
          - t13*t21/(t31*t33)* sinl2Pow2*l1_cosl1Pow2*l3_cosl3_sinl3
          - t13*t21/(t31*t33)* l3_sinl3Pow2*l1_cosl1Pow2*cosl2_sinl2
          + t13*t21/(t31*t33)* cosl2_sinl2*l1_cosl1Pow2*l3_cosl3Pow2
          + t13*t21/(t31*t33)* l1_cosl1Pow2*cosl2Pow2*l3_cosl3_sinl3
          + t13*t22/(t32*t33) * sinl1Pow2*l2_sinl2Pow2*l3_cosl3_sinl3
          + t13*t22/(t32*t33) * sinl1Pow2*l2_cosl2_sinl2*l3_sinl3Pow2
          - t13*t22/(t32*t33) * sinl1Pow2*l2_cosl2_sinl2*l3_cosl3Pow2
          - t13*t22/(t32*t33) * sinl1Pow2*l2_cosl2Pow2*l3_cosl3_sinl3
          + t13*t22/(t32*t33) * l2_sinl2Pow2*l3_sinl3Pow2*cosl1_sinl1
          - t13*t22/(t32*t33) * l2_sinl2Pow2*l3_cosl3Pow2*cosl1_sinl1
          - 4*t13*t22/(t32*t33) * cosl1_sinl1*l2_cosl2_sinl2*l3_cosl3_sinl3
          - t13*t22/(t32*t33) * l3_sinl3Pow2*cosl1_sinl1*l2_cosl2Pow2
          + t13*t22/(t32*t33) * cosl1_sinl1*l2_cosl2Pow2*l3_cosl3Pow2
          - t13*t22/(t32*t33) * l2_sinl2Pow2*cosl1Pow2*l3_cosl3_sinl3
          - t13*t22/(t32*t33) * l3_sinl3Pow2*cosl1Pow2*l2_cosl2_sinl2
          + t13*t22/(t32*t33) * l2_cosl2_sinl2*cosl1Pow2*l3_cosl3Pow2
          + t13*t22/(t32*t33) * l3_cosl3_sinl3*cosl1Pow2*l2_cosl2Pow2
          + t13*t23/(t33*t33) * l3Pow2_cosl3_sinl3*sinl1Pow2*sinl2Pow2
          + t13*t23/(t33*t33) * l3Pow2_sinl3Pow2*sinl1Pow2*cosl2_sinl2
          - t13*t23/(t33*t33) * l3Pow2_cosl3Pow2*sinl1Pow2*cosl2_sinl2
          - t13*t23/(t33*t33) * l3Pow2_cosl3_sinl3*sinl1Pow2*cosl2Pow2
          + t13*t23/(t33*t33) * l3Pow2_sinl3Pow2*sinl2Pow2*cosl1_sinl1
          - t13*t23/(t33*t33) * l3Pow2_cosl3Pow2*sinl2Pow2*cosl1_sinl1
          - 4*t13*t23/(t33*t33) * l3Pow2_cosl3_sinl3*cosl1_sinl1*cosl2_sinl2
          - t13*t23/(t33*t33) * l3Pow2_sinl3Pow2*cosl1_sinl1*cosl2Pow2
          + t13*t23/(t33*t33) * l3Pow2_cosl3Pow2*cosl1_sinl1*cosl2Pow2
          - t13*t23/(t33*t33) * l3Pow2_cosl3_sinl3*sinl2Pow2*cosl1Pow2
          - t13*t23/(t33*t33) * l3Pow2_sinl3Pow2*cosl1Pow2*cosl2_sinl2
          + t13*t23/(t33*t33) * l3Pow2_cosl3Pow2*cosl1Pow2*cosl2_sinl2
          + t13*t23/(t33*t33) * l3Pow2_cosl3_sinl3*cosl1Pow2*cosl2Pow2;
}
