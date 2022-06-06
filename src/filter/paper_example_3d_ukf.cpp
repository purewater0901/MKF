#include "filter/paper_example_3d_ukf.h"

double forward_xysin(const Eigen::Vector3d& x) {
    return x(0) * x(1) * std::sin(x(2));
}

double forward_xxycos(const Eigen::Vector3d& x){
    return x(0)*x(0)*x(1)*std::cos(x(2));
}

double forward_xycossin(const Eigen::Vector3d& x){
    return x(0)*x(1)*std::cos(x(2))*std::sin(x(2));
}

double forward(const Eigen::Vector3d& x, const std::string& type)
{
    if(type == "xysin") {
        return forward_xysin(x);
    } else if (type == "xxycos") {
        return forward_xxycos(x);
    } else if(type == "xycossin") {
        return forward_xycossin(x);
    }
}

PaperExample3DUKF::PaperExample3DUKF(const Eigen::Vector3d& mean, const Eigen::Matrix3d& cov) : mean_(mean),
                                                                                                cov_(cov),
                                                                                                size_(3),
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

double PaperExample3DUKF::predict(const std::string& type)
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