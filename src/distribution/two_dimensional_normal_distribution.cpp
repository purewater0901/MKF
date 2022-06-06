#include "distribution/two_dimensional_normal_distribution.h"

TwoDimensionalNormalDistribution::TwoDimensionalNormalDistribution(const Eigen::Vector2d& mean,
                                                                   const Eigen::Matrix2d& covariance)
                                                                   : mean_(mean), covariance_(covariance)
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

void TwoDimensionalNormalDistribution::setValues(const Eigen::Vector2d &mean, const Eigen::Matrix2d &covariance)
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

bool TwoDimensionalNormalDistribution::checkPositiveDefiniteness(const Eigen::Matrix2d& covariance)
{
    Eigen::LLT<Eigen::MatrixXd> lltOfA(covariance_); // compute the Cholesky decomposition of A
    if(lltOfA.info() == Eigen::NumericalIssue) {
        return false;
    }

    return true;
}

bool TwoDimensionalNormalDistribution::initializeData()
{
    if(std::fabs(covariance_(0, 1)) < 1e-10)
    {
        // Two Variables are independent
        eigen_values_ << 1.0/covariance_(0, 0), 1.0/covariance_(1, 1);
        T_ = Eigen::Matrix2d::Identity();
        independent_ = true;
    }
    else
    {
        // Inverse Matrix
        const Eigen::Matrix2d inv_covariance = covariance_.inverse();

        // Eigen Decomposition
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(inv_covariance);
        if(es.info() != Eigen::Success) {
            return false;
        }

        eigen_values_ = es.eigenvalues();
        T_ = es.eigenvectors();
        independent_ = false;
    }
    return true;
}

double TwoDimensionalNormalDistribution::calc_mean(const int dim)
{
    if(dim > 1) {
        throw std::invalid_argument("Dim is larger than the size of the distribution");
    }
    return mean_(dim);
}

double TwoDimensionalNormalDistribution::calc_covariance(const int dim)
{
    if(dim > 1) {
        throw std::invalid_argument("Dim is larger than the size of the distribution");
    }
    return covariance_(dim, dim);
}

double TwoDimensionalNormalDistribution::calc_moment(const int dim, const int moment)
{
    if(dim > 1) {
        throw std::invalid_argument("Dim is larger than the size of the distribution");
    }
    NormalDistribution dist(mean_(dim), covariance_(dim, dim));
    return dist.calc_moment(moment);
}

double TwoDimensionalNormalDistribution::calc_xy_moment()
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    return mean_(0) * mean_(1) + covariance_(0, 1);
}

double TwoDimensionalNormalDistribution::calc_x_cos_y_moment()
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(independent_) {
        NormalDistribution normal_x(mean_(0), covariance_(0, 0));
        NormalDistribution normal_y(mean_(1), covariance_(1, 1));

        return normal_x.calc_moment(1) * normal_y.calc_cos_moment(1);
    }

    const auto y_mean = T_.transpose()*mean_;
    const double t11 = T_(0, 0);
    const double t12 = T_(0, 1);
    const double t21 = T_(1, 0);
    const double t22 = T_(1, 1);

    const double l1_mean = t21*y_mean(0);
    const double l1_cov = t21*t21/eigen_values_(0);
    NormalDistribution normal1(l1_mean, l1_cov);

    const double l2_mean = t22*y_mean(1);
    const double l2_cov = t22*t22/eigen_values_(1);
    NormalDistribution normal2(l2_mean, l2_cov);

    // compute E[xcos(theta)]
    const double l1_cosl1 = normal1.calc_x_cos_moment(1, 1);
    const double l1_sinl1 = normal1.calc_x_sin_moment(1, 1);
    const double l2_cosl2 = normal2.calc_x_cos_moment(1, 1);
    const double l2_sinl2 = normal2.calc_x_sin_moment(1, 1);
    const double sin_l1 = normal1.calc_sin_moment(1);
    const double cos_l1 = normal1.calc_cos_moment(1);
    const double sin_l2 = normal2.calc_sin_moment(1);
    const double cos_l2 = normal2.calc_cos_moment(1);

    return (t11 / t21) * (l1_cosl1*cos_l2 - l1_sinl1*sin_l2) + (t12 / t22) * (l2_cosl2*cos_l1 - l2_sinl2*sin_l1);
}

double TwoDimensionalNormalDistribution::calc_x_sin_y_moment()
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(independent_) {
        NormalDistribution normal_x(mean_(0), covariance_(0, 0));
        NormalDistribution normal_y(mean_(1), covariance_(1, 1));

        return normal_x.calc_moment(1) * normal_y.calc_sin_moment(1);
    }

    const auto y_mean = T_.transpose()*mean_;
    const double t11 = T_(0, 0);
    const double t12 = T_(0, 1);
    const double t21 = T_(1, 0);
    const double t22 = T_(1, 1);

    const double l1_mean = t21*y_mean(0);
    const double l1_cov = t21*t21/eigen_values_(0);
    NormalDistribution normal1(l1_mean, l1_cov);

    const double l2_mean = t22*y_mean(1);
    const double l2_cov = t22*t22/eigen_values_(1);
    NormalDistribution normal2(l2_mean, l2_cov);

    // compute E[xsin(theta)]
    const double l1_cosl1 = normal1.calc_x_cos_moment(1, 1);
    const double l1_sinl1 = normal1.calc_x_sin_moment(1, 1);
    const double l2_cosl2 = normal2.calc_x_cos_moment(1, 1);
    const double l2_sinl2 = normal2.calc_x_sin_moment(1, 1);
    const double sin_l1 = normal1.calc_sin_moment(1);
    const double cos_l1 = normal1.calc_cos_moment(1);
    const double sin_l2 = normal2.calc_sin_moment(1);
    const double cos_l2 = normal2.calc_cos_moment(1);

    return (t11 / t21) * (l1_sinl1*cos_l2 + l1_cosl1*sin_l2) + (t12 / t22) * (l2_sinl2*cos_l1 + l2_cosl2*sin_l1);
}

double TwoDimensionalNormalDistribution::calc_third_moment(const int moment1, const int moment2)
{
   if(moment1 + moment2 != 3) {
       throw std::invalid_argument("dim1 + dim2 is not 3");
   }

   if(moment1 == 3) {
       return calc_moment(0, moment1);
   } else if(moment1 == 2) {
       return calc_xxy_moment();
   } else if (moment1 == 1) {
       return calc_xyy_moment();
   } else {
       return calc_moment(1, moment2);
   }
}

double TwoDimensionalNormalDistribution::calc_xxy_moment()
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    const auto y_mean = T_.transpose()*mean_;
    const double t11 = T_(0, 0);
    const double t12 = T_(0, 1);
    const double t21 = T_(1, 0);
    const double t22 = T_(1, 1);

    const double y1_mean = y_mean(0);
    const double y1_cov = 1.0/eigen_values_(0);
    NormalDistribution normal_y1(y1_mean, y1_cov);

    const double y2_mean = y_mean(1);
    const double y2_cov = 1.0/eigen_values_(1);
    NormalDistribution normal_y2(y2_mean, y2_cov);

    const double y1_first_moment = normal_y1.calc_moment(1);
    const double y2_first_moment = normal_y2.calc_moment(1);
    const double y1_second_moment = normal_y1.calc_moment(2);
    const double y2_second_moment = normal_y2.calc_moment(2);
    const double y1_third_moment = normal_y1.calc_moment(3);
    const double y2_third_moment = normal_y2.calc_moment(3);

    return t11*t11*t21*y1_third_moment + (t11*t11*t22 + 2.0*t11*t12*t21)*y1_second_moment*y2_first_moment
           + (2.0*t11*t12*t22+t12*t12*t21)*y1_first_moment*y2_second_moment + t12*t12*t22*y2_third_moment;
}

double TwoDimensionalNormalDistribution::TwoDimensionalNormalDistribution::calc_xyy_moment()
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    const auto y_mean = T_.transpose()*mean_;
    const double t11 = T_(0, 0);
    const double t12 = T_(0, 1);
    const double t21 = T_(1, 0);
    const double t22 = T_(1, 1);

    const double y1_mean = y_mean(0);
    const double y1_cov = 1.0/eigen_values_(0);
    NormalDistribution normal_y1(y1_mean, y1_cov);

    const double y2_mean = y_mean(1);
    const double y2_cov = 1.0/eigen_values_(1);
    NormalDistribution normal_y2(y2_mean, y2_cov);

    const double y1_first_moment = normal_y1.calc_moment(1);
    const double y2_first_moment = normal_y2.calc_moment(1);
    const double y1_second_moment = normal_y1.calc_moment(2);
    const double y2_second_moment = normal_y2.calc_moment(2);
    const double y1_third_moment = normal_y1.calc_moment(3);
    const double y2_third_moment = normal_y2.calc_moment(3);

    return t11*t21*t21*y1_third_moment + (2*t11*t21*t22 + t12*t21*t21)*y1_second_moment*y2_first_moment
           + (2*t12*t21*t22 + t11*t22*t22)*y1_first_moment*y2_second_moment + t12*t22*t22*y2_third_moment;
}

double TwoDimensionalNormalDistribution::calc_xx_sin_y_moment()
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(independent_) {
        NormalDistribution normal_x(mean_(0), covariance_(0, 0));
        NormalDistribution normal_y(mean_(1), covariance_(1, 1));

        return normal_x.calc_moment(2) * normal_y.calc_sin_moment(1);
    }

    const auto y_mean = T_.transpose()*mean_;
    const double t11 = T_(0, 0);
    const double t12 = T_(0, 1);
    const double t21 = T_(1, 0);
    const double t22 = T_(1, 1);

    const double l1_mean = t21*y_mean(0);
    const double l1_cov = t21*t21/eigen_values_(0);
    NormalDistribution l1(l1_mean, l1_cov);

    const double l2_mean = t22*y_mean(1);
    const double l2_cov = t22*t22/eigen_values_(1);
    NormalDistribution l2(l2_mean, l2_cov);

    // compute E[xxsin(theta)]
    const double l1_cosl1 = l1.calc_x_cos_moment(1, 1);
    const double l1_sinl1 = l1.calc_x_sin_moment(1, 1);
    const double l2_cosl2 = l2.calc_x_cos_moment(1, 1);
    const double l2_sinl2 = l2.calc_x_sin_moment(1, 1);
    const double l1l1_cosl1 = l1.calc_x_cos_moment(2, 1);
    const double l1l1_sinl1 = l1.calc_x_sin_moment(2, 1);
    const double l2l2_cosl2 = l2.calc_x_cos_moment(2, 1);
    const double l2l2_sinl2 = l2.calc_x_sin_moment(2, 1);
    const double sin_l1 = l1.calc_sin_moment(1);
    const double cos_l1 = l1.calc_cos_moment(1);
    const double sin_l2 = l2.calc_sin_moment(1);
    const double cos_l2 = l2.calc_cos_moment(1);

    return std::pow(t11/t21, 2) * (l1l1_sinl1*cos_l2 + l1l1_cosl1*sin_l2)
           + 2*t11*t12/(t21*t22) * (l1_sinl1*l2_cosl2 + l2_sinl2*l1_cosl1)
           + std::pow(t12/t22, 2) * (l2l2_cosl2*sin_l1 + l2l2_sinl2*cos_l1);
}

double TwoDimensionalNormalDistribution::calc_xx_cos_y_moment()
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(independent_) {
        NormalDistribution normal_x(mean_(0), covariance_(0, 0));
        NormalDistribution normal_y(mean_(1), covariance_(1, 1));

        return normal_x.calc_moment(2) * normal_y.calc_cos_moment(1);
    }

    const auto y_mean = T_.transpose()*mean_;
    const double t11 = T_(0, 0);
    const double t12 = T_(0, 1);
    const double t21 = T_(1, 0);
    const double t22 = T_(1, 1);

    const double l1_mean = t21*y_mean(0);
    const double l1_cov = t21*t21/eigen_values_(0);
    NormalDistribution l1(l1_mean, l1_cov);

    const double l2_mean = t22*y_mean(1);
    const double l2_cov = t22*t22/eigen_values_(1);
    NormalDistribution l2(l2_mean, l2_cov);

    // compute E[xxcos(theta)]
    const double l1_cosl1 = l1.calc_x_cos_moment(1, 1);
    const double l1_sinl1 = l1.calc_x_sin_moment(1, 1);
    const double l2_cosl2 = l2.calc_x_cos_moment(1, 1);
    const double l2_sinl2 = l2.calc_x_sin_moment(1, 1);
    const double l1l1_cosl1 = l1.calc_x_cos_moment(2, 1);
    const double l1l1_sinl1 = l1.calc_x_sin_moment(2, 1);
    const double l2l2_cosl2 = l2.calc_x_cos_moment(2, 1);
    const double l2l2_sinl2 = l2.calc_x_sin_moment(2, 1);
    const double sin_l1 = l1.calc_sin_moment(1);
    const double cos_l1 = l1.calc_cos_moment(1);
    const double sin_l2 = l2.calc_sin_moment(1);
    const double cos_l2 = l2.calc_cos_moment(1);

    return std::pow(t11/t21, 2) * (l1l1_cosl1*cos_l2 - l1l1_sinl1*sin_l2)
           + 2*t11*t12/(t21*t22) * (l1_cosl1*l2_cosl2 - l2_sinl2*l1_sinl1)
           + std::pow(t12/t22, 2) * (l2l2_cosl2*cos_l1 - l2l2_sinl2*sin_l1);
}

double TwoDimensionalNormalDistribution::calc_x_cos_y_cos_y_moment()
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(independent_) {
        NormalDistribution normal_x(mean_(0), covariance_(0, 0));
        NormalDistribution normal_y(mean_(1), covariance_(1, 1));

        return normal_x.calc_moment(1) * normal_y.calc_cos_moment(2);
    }

    const auto y_mean = T_.transpose()*mean_;
    const double t11 = T_(0, 0);
    const double t12 = T_(0, 1);
    const double t21 = T_(1, 0);
    const double t22 = T_(1, 1);

    const double l1_mean = t21*y_mean(0);
    const double l1_cov = t21*t21/eigen_values_(0);
    NormalDistribution l1(l1_mean, l1_cov);

    const double l2_mean = t22*y_mean(1);
    const double l2_cov = t22*t22/eigen_values_(1);
    NormalDistribution l2(l2_mean, l2_cov);

    const double l1Pow1_cPow2 = l1.calc_x_cos_moment(1, 2);
    const double l1Pow1_sPow2 = l1.calc_x_sin_moment(1, 2);
    const double l2Pow1_cPow2 = l2.calc_x_cos_moment(1, 2);
    const double l2Pow1_sPow2 = l2.calc_x_sin_moment(1, 2);
    const double l1cPow1_sPow1 = l1.calc_cos_sin_moment(1, 1);
    const double l2cPow1_sPow1 = l2.calc_cos_sin_moment(1, 1);
    const double l1Pow1_cPow1_sPow1 = l1.calc_x_cos_sin_moment(1, 1, 1);
    const double l2Pow1_cPow1_sPow1 = l2.calc_x_cos_sin_moment(1, 1, 1);
    const double sl1Pow2 = l1.calc_sin_moment(2);
    const double cl1Pow2 = l1.calc_cos_moment(2);
    const double sl2Pow2 = l2.calc_sin_moment(2);
    const double cl2Pow2 = l2.calc_cos_moment(2);

    return t11/t21 * l1Pow1_sPow2 * sl2Pow2
         - 2 * (t11/t21) * l1Pow1_cPow1_sPow1 * l2cPow1_sPow1
         + (t11/t21) * l1Pow1_cPow2 * cl2Pow2
         + (t12/t22) * l2Pow1_sPow2 * sl1Pow2
         - 2 * (t12/t22) * l2Pow1_cPow1_sPow1 * l1cPow1_sPow1
         + t12/t22 * l2Pow1_cPow2 * cl1Pow2;
}

double TwoDimensionalNormalDistribution::calc_x_sin_y_sin_y_moment()
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(independent_) {
        NormalDistribution normal_x(mean_(0), covariance_(0, 0));
        NormalDistribution normal_y(mean_(1), covariance_(1, 1));

        return normal_x.calc_moment(1) * normal_y.calc_sin_moment(2);
    }

    const auto y_mean = T_.transpose()*mean_;
    const double t11 = T_(0, 0);
    const double t12 = T_(0, 1);
    const double t21 = T_(1, 0);
    const double t22 = T_(1, 1);

    const double l1_mean = t21*y_mean(0);
    const double l1_cov = t21*t21/eigen_values_(0);
    NormalDistribution l1(l1_mean, l1_cov);

    const double l2_mean = t22*y_mean(1);
    const double l2_cov = t22*t22/eigen_values_(1);
    NormalDistribution l2(l2_mean, l2_cov);

    const double l1Pow1_cPow2 = l1.calc_x_cos_moment(1, 2);
    const double l1Pow1_sPow2 = l1.calc_x_sin_moment(1, 2);
    const double l2Pow1_cPow2 = l2.calc_x_cos_moment(1, 2);
    const double l2Pow1_sPow2 = l2.calc_x_sin_moment(1, 2);
    const double l1cPow1_sPow1 = l1.calc_cos_sin_moment(1, 1);
    const double l2cPow1_sPow1 = l2.calc_cos_sin_moment(1, 1);
    const double l1Pow1_cPow1_sPow1 = l1.calc_x_cos_sin_moment(1, 1, 1);
    const double l2Pow1_cPow1_sPow1 = l2.calc_x_cos_sin_moment(1, 1, 1);
    const double sl1Pow2 = l1.calc_sin_moment(2);
    const double cl1Pow2 = l1.calc_cos_moment(2);
    const double sl2Pow2 = l2.calc_sin_moment(2);
    const double cl2Pow2 = l2.calc_cos_moment(2);

    return  t11/t21 * l1Pow1_sPow2 * cl2Pow2
          + 2*t11/t21 * l1Pow1_cPow1_sPow1 * l2cPow1_sPow1
          + t11/t21 * l1Pow1_cPow2 * sl2Pow2
          + t12/t22 * l2Pow1_cPow2 * sl1Pow2
          + 2*t12/t22 * l2Pow1_cPow1_sPow1 * l1cPow1_sPow1
          + t12/t22 * l2Pow1_sPow2 * cl1Pow2;
}

double TwoDimensionalNormalDistribution::calc_x_cos_y_sin_y_moment()
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(independent_) {
        NormalDistribution normal_x(mean_(0), covariance_(0, 0));
        NormalDistribution normal_y(mean_(1), covariance_(1, 1));

        return normal_x.calc_moment(1) * normal_y.calc_cos_sin_moment(1, 1);
    }

    const auto y_mean = T_.transpose()*mean_;
    const double t11 = T_(0, 0);
    const double t12 = T_(0, 1);
    const double t21 = T_(1, 0);
    const double t22 = T_(1, 1);

    const double l1_mean = t21*y_mean(0);
    const double l1_cov = t21*t21/eigen_values_(0);
    NormalDistribution l1(l1_mean, l1_cov);

    const double l2_mean = t22*y_mean(1);
    const double l2_cov = t22*t22/eigen_values_(1);
    NormalDistribution l2(l2_mean, l2_cov);

    const double l1Pow1_cPow2 = l1.calc_x_cos_moment(1, 2);
    const double l1Pow1_sPow2 = l1.calc_x_sin_moment(1, 2);
    const double l2Pow1_cPow2 = l2.calc_x_cos_moment(1, 2);
    const double l2Pow1_sPow2 = l2.calc_x_sin_moment(1, 2);
    const double l1cPow1_sPow1 = l1.calc_cos_sin_moment(1, 1);
    const double l2cPow1_sPow1 = l2.calc_cos_sin_moment(1, 1);
    const double l1Pow1_cPow1_sPow1 = l1.calc_x_cos_sin_moment(1, 1, 1);
    const double l2Pow1_cPow1_sPow1 = l2.calc_x_cos_sin_moment(1, 1, 1);
    const double sl1Pow2 = l1.calc_sin_moment(2);
    const double cl1Pow2 = l1.calc_cos_moment(2);
    const double sl2Pow2 = l2.calc_sin_moment(2);
    const double cl2Pow2 = l2.calc_cos_moment(2);

    return - t11/t21 * l1Pow1_sPow2 * l2cPow1_sPow1
           - t11/t21 * l1Pow1_cPow1_sPow1 * sl2Pow2
           + t11/t21 * l1Pow1_cPow1_sPow1 * cl2Pow2
           + t11/t21 * l1Pow1_cPow2 * l2cPow1_sPow1
           - t12/t22 * l2Pow1_cPow1_sPow1 * sl1Pow2
           - t12/t22 * l2Pow1_sPow2 * l1cPow1_sPow1
           + t12/t22 * l2Pow1_cPow2 * l1cPow1_sPow1
           + t12/t22 * l2Pow1_cPow1_sPow1 * cl1Pow2;
}

double TwoDimensionalNormalDistribution::calc_x_y_sin_y_moment()
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(independent_) {
        NormalDistribution normal_x(mean_(0), covariance_(0, 0));
        NormalDistribution normal_y(mean_(1), covariance_(1, 1));

        return normal_x.calc_moment(1) * normal_y.calc_x_sin_moment(1, 1);
    }

    const auto y_mean = T_.transpose()*mean_;
    const double t11 = T_(0, 0);
    const double t12 = T_(0, 1);
    const double t21 = T_(1, 0);
    const double t22 = T_(1, 1);

    const double l1_mean = t21*y_mean(0);
    const double l1_cov = t21*t21/eigen_values_(0);
    NormalDistribution l1(l1_mean, l1_cov);

    const double l2_mean = t22*y_mean(1);
    const double l2_cov = t22*t22/eigen_values_(1);
    NormalDistribution l2(l2_mean, l2_cov);

    // compute E[xxsin(theta)]
    const double l1_cosl1 = l1.calc_x_cos_moment(1, 1);
    const double l1_sinl1 = l1.calc_x_sin_moment(1, 1);
    const double l2_cosl2 = l2.calc_x_cos_moment(1, 1);
    const double l2_sinl2 = l2.calc_x_sin_moment(1, 1);
    const double l1l1_cosl1 = l1.calc_x_cos_moment(2, 1);
    const double l1l1_sinl1 = l1.calc_x_sin_moment(2, 1);
    const double l2l2_cosl2 = l2.calc_x_cos_moment(2, 1);
    const double l2l2_sinl2 = l2.calc_x_sin_moment(2, 1);
    const double sin_l1 = l1.calc_sin_moment(1);
    const double cos_l1 = l1.calc_cos_moment(1);
    const double sin_l2 = l2.calc_sin_moment(1);
    const double cos_l2 = l2.calc_cos_moment(1);

    return t11/t21 * (l1l1_sinl1*cos_l2 + l1l1_cosl1*sin_l2)
           + (t11/t21 + t12/t22) * (l1_sinl1*l2_cosl2 + l2_sinl2*l1_cosl1)
           + t12/t22 * (l2l2_cosl2*sin_l1 + l2l2_sinl2*cos_l1);
}

double TwoDimensionalNormalDistribution::calc_x_y_cos_y_moment()
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(independent_) {
        NormalDistribution normal_x(mean_(0), covariance_(0, 0));
        NormalDistribution normal_y(mean_(1), covariance_(1, 1));

        return normal_x.calc_moment(1) * normal_y.calc_x_cos_moment(1, 1);
    }

    const auto y_mean = T_.transpose()*mean_;
    const double t11 = T_(0, 0);
    const double t12 = T_(0, 1);
    const double t21 = T_(1, 0);
    const double t22 = T_(1, 1);

    const double l1_mean = t21*y_mean(0);
    const double l1_cov = t21*t21/eigen_values_(0);
    NormalDistribution l1(l1_mean, l1_cov);

    const double l2_mean = t22*y_mean(1);
    const double l2_cov = t22*t22/eigen_values_(1);
    NormalDistribution l2(l2_mean, l2_cov);

    // compute E[xxsin(theta)]
    const double l1_cosl1 = l1.calc_x_cos_moment(1, 1);
    const double l1_sinl1 = l1.calc_x_sin_moment(1, 1);
    const double l2_cosl2 = l2.calc_x_cos_moment(1, 1);
    const double l2_sinl2 = l2.calc_x_sin_moment(1, 1);
    const double l1l1_cosl1 = l1.calc_x_cos_moment(2, 1);
    const double l1l1_sinl1 = l1.calc_x_sin_moment(2, 1);
    const double l2l2_cosl2 = l2.calc_x_cos_moment(2, 1);
    const double l2l2_sinl2 = l2.calc_x_sin_moment(2, 1);
    const double sin_l1 = l1.calc_sin_moment(1);
    const double cos_l1 = l1.calc_cos_moment(1);
    const double sin_l2 = l2.calc_sin_moment(1);
    const double cos_l2 = l2.calc_cos_moment(1);

    return t11/t21 * (l1l1_cosl1*cos_l2 - l1l1_sinl1*sin_l2)
           + (t11/t21 + t12/t22) * (l1_cosl1*l2_cosl2 - l2_sinl2*l1_sinl1)
           + t12/t22 * (l2l2_cosl2*cos_l1 - l2l2_sinl2*sin_l1);
}

double TwoDimensionalNormalDistribution::calc_xxyy_moment()
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    const auto y_mean = T_.transpose()*mean_;
    const double t11 = T_(0, 0);
    const double t12 = T_(0, 1);
    const double t21 = T_(1, 0);
    const double t22 = T_(1, 1);

    const double y1_mean = y_mean(0);
    const double y1_cov = 1.0/eigen_values_(0);
    NormalDistribution normal_y1(y1_mean, y1_cov);

    const double y2_mean = y_mean(1);
    const double y2_cov = 1.0/eigen_values_(1);
    NormalDistribution normal_y2(y2_mean, y2_cov);

    const double y1_first_moment = normal_y1.calc_moment(1);
    const double y2_first_moment = normal_y2.calc_moment(1);
    const double y1_second_moment = normal_y1.calc_moment(2);
    const double y2_second_moment = normal_y2.calc_moment(2);
    const double y1_third_moment = normal_y1.calc_moment(3);
    const double y2_third_moment = normal_y2.calc_moment(3);
    const double y1_fourth_moment = normal_y1.calc_moment(4);
    const double y2_fourth_moment = normal_y2.calc_moment(4);

    return std::pow(t11*t21, 2)*y1_fourth_moment + 2*(t11*t11*t21*t22 + t11*t12*t21*t21)*y1_third_moment*y2_first_moment
           + (std::pow(t11*t22, 2) + std::pow(t12*t21, 2) + 4*t11*t12*t21*t22) * y1_second_moment * y2_second_moment
           + 2*(t12*t12*t21*t22 + t11*t12*t22*t22)*y1_first_moment*y2_third_moment + std::pow(t12*t22, 2)*y2_fourth_moment;
}

double TwoDimensionalNormalDistribution::calc_xx_cos_y_cos_y_moment()
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(independent_) {
        NormalDistribution normal_x(mean_(0), covariance_(0, 0));
        NormalDistribution normal_y(mean_(1), covariance_(1, 1));

        return normal_x.calc_moment(2) * normal_y.calc_cos_moment(2);
    }

    const auto y_mean = T_.transpose()*mean_;
    const double t11 = T_(0, 0);
    const double t12 = T_(0, 1);
    const double t21 = T_(1, 0);
    const double t22 = T_(1, 1);

    const double l1_mean = t21*y_mean(0);
    const double l1_cov = t21*t21/eigen_values_(0);
    NormalDistribution l1(l1_mean, l1_cov);

    const double l2_mean = t22*y_mean(1);
    const double l2_cov = t22*t22/eigen_values_(1);
    NormalDistribution l2(l2_mean, l2_cov);

    const double l1Pow1_cPow2 = l1.calc_x_cos_moment(1, 2);
    const double l1Pow1_sPow2 = l1.calc_x_sin_moment(1, 2);
    const double l2Pow1_cPow2 = l2.calc_x_cos_moment(1, 2);
    const double l2Pow1_sPow2 = l2.calc_x_sin_moment(1, 2);
    const double l1Pow1_cPow1_sPow1 = l1.calc_x_cos_sin_moment(1, 1, 1);
    const double l2Pow1_cPow1_sPow1 = l2.calc_x_cos_sin_moment(1, 1, 1);
    const double l1Pow2_cPow2 = l1.calc_x_cos_moment(2, 2);
    const double l1Pow2_sPow2 = l1.calc_x_sin_moment(2, 2);
    const double l2Pow2_cPow2 = l2.calc_x_cos_moment(2, 2);
    const double l2Pow2_sPow2 = l2.calc_x_sin_moment(2, 2);
    const double l1cPow1_sPow1 = l1.calc_cos_sin_moment(1, 1);
    const double l2cPow1_sPow1 = l2.calc_cos_sin_moment(1, 1);
    const double l1Pow2_cPow1_sPow1 = l1.calc_x_cos_sin_moment(2, 1, 1);
    const double l2Pow2_cPow1_sPow1 = l2.calc_x_cos_sin_moment(2, 1, 1);
    const double sl1Pow2 = l1.calc_sin_moment(2);
    const double cl1Pow2 = l1.calc_cos_moment(2);
    const double sl2Pow2 = l2.calc_sin_moment(2);
    const double cl2Pow2 = l2.calc_cos_moment(2);

    return std::pow(t11/t21, 2) * l1Pow2_sPow2*sl2Pow2
           - 2*std::pow(t11/t21, 2) * l1Pow2_cPow1_sPow1 * l2cPow1_sPow1
           + std::pow(t11/t21, 2) * l1Pow2_cPow2 * cl2Pow2
           + 2*(t11*t12)/(t21*t22) * l1Pow1_sPow2 * l2Pow1_sPow2
           - 4*(t11*t12)/(t21*t22) * l1Pow1_cPow1_sPow1 * l2Pow1_cPow1_sPow1
           + 2*(t11*t12)/(t21*t22) * l1Pow1_cPow2 * l2Pow1_cPow2
           + std::pow(t12/t22, 2) * l2Pow2_sPow2 * sl1Pow2
           - 2*std::pow(t12/t22, 2) * l2Pow2_cPow1_sPow1 * l1cPow1_sPow1
           + std::pow(t12/t22, 2) * l2Pow2_cPow2*cl1Pow2;
}

double TwoDimensionalNormalDistribution::calc_xx_sin_y_sin_y_moment()
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(independent_) {
        NormalDistribution normal_x(mean_(0), covariance_(0, 0));
        NormalDistribution normal_y(mean_(1), covariance_(1, 1));

        return normal_x.calc_moment(2) * normal_y.calc_sin_moment(2);
    }

    const auto y_mean = T_.transpose()*mean_;
    const double t11 = T_(0, 0);
    const double t12 = T_(0, 1);
    const double t21 = T_(1, 0);
    const double t22 = T_(1, 1);

    const double l1_mean = t21*y_mean(0);
    const double l1_cov = t21*t21/eigen_values_(0);
    NormalDistribution l1(l1_mean, l1_cov);

    const double l2_mean = t22*y_mean(1);
    const double l2_cov = t22*t22/eigen_values_(1);
    NormalDistribution l2(l2_mean, l2_cov);

    const double l1Pow1_cPow2 = l1.calc_x_cos_moment(1, 2);
    const double l1Pow1_sPow2 = l1.calc_x_sin_moment(1, 2);
    const double l2Pow1_cPow2 = l2.calc_x_cos_moment(1, 2);
    const double l2Pow1_sPow2 = l2.calc_x_sin_moment(1, 2);
    const double l1Pow1_cPow1_sPow1 = l1.calc_x_cos_sin_moment(1, 1, 1);
    const double l2Pow1_cPow1_sPow1 = l2.calc_x_cos_sin_moment(1, 1, 1);
    const double l1Pow2_cPow2 = l1.calc_x_cos_moment(2, 2);
    const double l1Pow2_sPow2 = l1.calc_x_sin_moment(2, 2);
    const double l2Pow2_cPow2 = l2.calc_x_cos_moment(2, 2);
    const double l2Pow2_sPow2 = l2.calc_x_sin_moment(2, 2);
    const double l1cPow1_sPow1 = l1.calc_cos_sin_moment(1, 1);
    const double l2cPow1_sPow1 = l2.calc_cos_sin_moment(1, 1);
    const double l1Pow2_cPow1_sPow1 = l1.calc_x_cos_sin_moment(2, 1, 1);
    const double l2Pow2_cPow1_sPow1 = l2.calc_x_cos_sin_moment(2, 1, 1);
    const double sl1Pow2 = l1.calc_sin_moment(2);
    const double cl1Pow2 = l1.calc_cos_moment(2);
    const double sl2Pow2 = l2.calc_sin_moment(2);
    const double cl2Pow2 = l2.calc_cos_moment(2);

    return std::pow(t11/t21, 2) * l1Pow2_sPow2 * cl2Pow2
          + 2 * std::pow(t11/t21, 2) * l1Pow2_cPow1_sPow1 * l2cPow1_sPow1
          + std::pow(t11/t21, 2) * l1Pow2_cPow2 * sl2Pow2
          + 2*(t11*t12)/(t21*t22) * l1Pow1_sPow2 * l2Pow1_cPow2
          + 4*(t11*t12)/(t21*t22) * l1Pow1_cPow1_sPow1 * l2Pow1_cPow1_sPow1
          + 2*(t11*t12)/(t21*t22) * l1Pow1_cPow2 * l2Pow1_sPow2
          + std::pow(t12/t22, 2) * l2Pow2_cPow2 * sl1Pow2
          + 2 * std::pow(t12/t22, 2) * l2Pow2_cPow1_sPow1 * l1cPow1_sPow1
          + std::pow(t12/t22, 2) * l2Pow2_sPow2 * cl1Pow2;
}

double TwoDimensionalNormalDistribution::calc_xx_cos_y_sin_y_moment()
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(independent_) {
        NormalDistribution normal_x(mean_(0), covariance_(0, 0));
        NormalDistribution normal_y(mean_(1), covariance_(1, 1));

        return normal_x.calc_moment(2) * normal_y.calc_cos_sin_moment(1, 1);
    }

    const auto y_mean = T_.transpose()*mean_;
    const double t11 = T_(0, 0);
    const double t12 = T_(0, 1);
    const double t21 = T_(1, 0);
    const double t22 = T_(1, 1);

    const double l1_mean = t21*y_mean(0);
    const double l1_cov = t21*t21/eigen_values_(0);
    NormalDistribution l1(l1_mean, l1_cov);

    const double l2_mean = t22*y_mean(1);
    const double l2_cov = t22*t22/eigen_values_(1);
    NormalDistribution l2(l2_mean, l2_cov);

    const double l1Pow1_cPow2 = l1.calc_x_cos_moment(1, 2);
    const double l1Pow1_sPow2 = l1.calc_x_sin_moment(1, 2);
    const double l2Pow1_cPow2 = l2.calc_x_cos_moment(1, 2);
    const double l2Pow1_sPow2 = l2.calc_x_sin_moment(1, 2);
    const double l1Pow1_cPow1_sPow1 = l1.calc_x_cos_sin_moment(1, 1, 1);
    const double l2Pow1_cPow1_sPow1 = l2.calc_x_cos_sin_moment(1, 1, 1);
    const double l1Pow2_cPow2 = l1.calc_x_cos_moment(2, 2);
    const double l1Pow2_sPow2 = l1.calc_x_sin_moment(2, 2);
    const double l2Pow2_cPow2 = l2.calc_x_cos_moment(2, 2);
    const double l2Pow2_sPow2 = l2.calc_x_sin_moment(2, 2);
    const double l1cPow1_sPow1 = l1.calc_cos_sin_moment(1, 1);
    const double l2cPow1_sPow1 = l2.calc_cos_sin_moment(1, 1);
    const double l1Pow2_cPow1_sPow1 = l1.calc_x_cos_sin_moment(2, 1, 1);
    const double l2Pow2_cPow1_sPow1 = l2.calc_x_cos_sin_moment(2, 1, 1);
    const double sl1Pow2 = l1.calc_sin_moment(2);
    const double cl1Pow2 = l1.calc_cos_moment(2);
    const double sl2Pow2 = l2.calc_sin_moment(2);
    const double cl2Pow2 = l2.calc_cos_moment(2);

    return - std::pow(t11/t21, 2) * l1Pow2_sPow2 * l2cPow1_sPow1
           - std::pow(t11/t21, 2) * l1Pow2_cPow1_sPow1 * sl2Pow2
           + std::pow(t11/t21, 2) * l1Pow2_cPow1_sPow1 * cl2Pow2
           + std::pow(t11/t21, 2) * l1Pow2_cPow2 * l2cPow1_sPow1
           - 2 * (t11*t12)/(t21*t22) * l1Pow1_sPow2 * l2Pow1_cPow1_sPow1
           - 2 * (t11*t12)/(t21*t22) * l1Pow1_cPow1_sPow1 * l2Pow1_sPow2
           + 2 * (t11*t12)/(t21*t22) * l1Pow1_cPow1_sPow1 * l2Pow1_cPow2
           + 2 * (t11*t12)/(t21*t22) * l1Pow1_cPow2 * l2Pow1_cPow1_sPow1
           - std::pow(t12/t22, 2) * l2Pow2_cPow1_sPow1 * sl1Pow2
           - std::pow(t12/t22, 2) * l2Pow2_sPow2 * l1cPow1_sPow1
           + std::pow(t12/t22, 2) * l2Pow2_cPow2 * l1cPow1_sPow1
           + std::pow(t12/t22, 2) * l2Pow2_cPow1_sPow1 * cl1Pow2;
}
