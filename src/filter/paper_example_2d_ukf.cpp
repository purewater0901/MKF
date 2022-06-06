#include "filter/paper_example_2d_ukf.h"

double forward_xy(const Eigen::Vector2d& x) {
    return x(0) * x(1);
}

double forward_xcos(const Eigen::Vector2d& x){
    return x(0)*std::cos(x(1));
}

double forward_xcossin(const Eigen::Vector2d& x){
    return x(0)*std::cos(x(1))*std::sin(x(1));
}

double forward(const Eigen::Vector2d& x, const std::string& type)
{
    if(type == "xy") {
        return forward_xy(x);
    } else if (type == "xcos") {
        return forward_xcos(x);
    } else if(type == "xcossin") {
        return forward_xcossin(x);
    }

    throw std::invalid_argument("Invalid type.");
}

PaperExample2DUKF::PaperExample2DUKF(const Eigen::Vector2d& mean, const Eigen::Matrix2d& cov) : mean_(mean),
                                                                                                cov_(cov),
                                                                                                size_(2),
                                                                                                alpha_squared_(1.0),
                                                                                                beta_(0.0),
                                                                                                kappa_(3.0),
                                                                                                lambda_(alpha_squared_*(size_ + kappa_) - size_)
{
    Sigma_WM0_ = lambda_/(size_ + lambda_);
    Sigma_WC0_ = Sigma_WM0_ + (1.0 - alpha_squared_ + beta_);
    Sigma_WMI_ = 1.0 / (2.0 * (size_ + lambda_));
    Sigma_WCI_ = Sigma_WMI_;
}

double PaperExample2DUKF::predict(const std::string& type)
{
    Eigen::MatrixXd sigma_points = Eigen::MatrixXd::Zero(size_, 2*size_+1);

    assert((cov_*(size_ + lambda_)).llt().info() == Eigen::Success);
    const Eigen::MatrixXd cov_squared = (cov_ * (size_ + lambda_)).llt().matrixL();

    double processed_mean = 0.0;
    Eigen::MatrixXd processed_matrix = Eigen::MatrixXd::Zero(1, 2*size_+1);
    for(size_t i=0; i<size_; ++i) {
        sigma_points.col(i) = mean_ + cov_squared.col(i);
        processed_matrix(0, i) = forward(sigma_points.col(i), type);
        processed_mean += Sigma_WMI_ * processed_matrix(0, i);
    }
    for(size_t i=size_; i<2*size_; ++i) {
        sigma_points.col(i) = mean_ - cov_squared.col(i-size_);
        processed_matrix(0, i) = forward(sigma_points.col(i), type);
        processed_mean += Sigma_WMI_ * processed_matrix(0, i);
    }
    {
        sigma_points.col(2*size_) = mean_;
        processed_matrix(0, 2*size_) = forward(sigma_points.col(2*size_), type);
        processed_mean += Sigma_WM0_ * processed_matrix(0, 2*size_);
    }

    double processed_augmented_cov = 0.0;
    for(size_t i=0; i<2*size_; ++i) {
        const double delta_x = processed_matrix(0, i) - processed_mean;
        processed_augmented_cov += Sigma_WCI_ * (delta_x * delta_x);
    }
    {
        const double delta_x = processed_matrix(0, 2*size_) - processed_mean;
        processed_augmented_cov += Sigma_WC0_ * (delta_x * delta_x);
    }

    return processed_mean;
}