#include "filter/normal_vehicle_ukf.h"

NormalVehicleUKF::NormalVehicleUKF() : augmented_size_(8),
                                       alpha_squared_(1.0),
                                       beta_(0.0),
                                       kappa_(3.0),
                                       lambda_(alpha_squared_*(augmented_size_ + kappa_) - augmented_size_)
{
    sigma_points_ = Eigen::MatrixXd::Zero(augmented_size_, 2*augmented_size_+1);

    Sigma_WM0_ = lambda_/(augmented_size_ + lambda_);
    Sigma_WC0_ = Sigma_WM0_ + (1.0 - alpha_squared_ + beta_);
    Sigma_WMI_ = 1.0 / (2.0 * (augmented_size_ + lambda_));
    Sigma_WCI_ = Sigma_WMI_;
}

StateInfo NormalVehicleUKF::predict(const StateInfo& state_info,
                                    const Eigen::Vector2d & control_inputs,
                                    const std::map<int, std::shared_ptr<BaseDistribution>>& system_noise_map,
                                    const std::map<int, std::shared_ptr<BaseDistribution>>& measurement_noise_map)
{
    const auto dist_wx = system_noise_map.at(SYSTEM_NOISE::WX);
    const auto dist_wy = system_noise_map.at(SYSTEM_NOISE::WY);
    const auto dist_wyaw = system_noise_map.at(SYSTEM_NOISE::WYAW);
    const auto dist_mr = measurement_noise_map.at(OBSERVATION_NOISE::WR);
    const auto dist_myaw = measurement_noise_map.at(OBSERVATION_NOISE::WYAW);

    Eigen::VectorXd augmented_mean = Eigen::VectorXd::Zero(augmented_size_);
    augmented_mean.head(3) = state_info.mean;
    augmented_mean(3) = dist_wx->calc_mean();
    augmented_mean(4) = dist_wy->calc_mean();
    augmented_mean(5) = dist_wyaw->calc_mean();
    augmented_mean(6) = dist_mr->calc_mean();
    augmented_mean(7) = dist_myaw->calc_mean();

    Eigen::MatrixXd augmented_cov = Eigen::MatrixXd::Zero(augmented_size_, augmented_size_);
    augmented_cov.block(0, 0, 3, 3) = state_info.covariance;
    augmented_cov(3, 3) = dist_wx->calc_variance();
    augmented_cov(4, 4) = dist_wy->calc_variance();
    augmented_cov(5, 5) = dist_wyaw->calc_variance();
    augmented_cov(6, 6) = dist_mr->calc_variance();
    augmented_cov(7, 7) = dist_myaw->calc_variance();

    assert((augmented_cov*(augmented_size_ + lambda_)).llt().info() == Eigen::Success);
    const Eigen::MatrixXd augmented_cov_squared = (augmented_cov * (augmented_size_ + lambda_)).llt().matrixL();

    Eigen::VectorXd processed_augmented_mean = Eigen::VectorXd::Zero(augmented_size_);
    for(size_t i=0; i<augmented_size_; ++i) {
        sigma_points_.col(i) = augmented_mean + augmented_cov_squared.col(i);
        const Eigen::Vector3d processed_state = model_.propagate(sigma_points_.col(i).head(3), control_inputs, sigma_points_.col(i).segment(3, 3));
        sigma_points_.col(i).head(3) = processed_state;
        processed_augmented_mean += Sigma_WMI_ * sigma_points_.col(i);
    }
    for(size_t i=augmented_size_; i<2*augmented_size_; ++i) {
        sigma_points_.col(i) = augmented_mean - augmented_cov_squared.col(i-augmented_size_);
        const Eigen::Vector3d processed_state = model_.propagate(sigma_points_.col(i).head(3), control_inputs, sigma_points_.col(i).segment(3, 3));
        sigma_points_.col(i).head(3) = processed_state;
        processed_augmented_mean += Sigma_WMI_ * sigma_points_.col(i);
    }
    {
        sigma_points_.col(2*augmented_size_) = augmented_mean;
        const Eigen::Vector3d processed_state = model_.propagate(sigma_points_.col(2*augmented_size_).head(3), control_inputs, sigma_points_.col(2*augmented_size_).segment(3, 3));
        sigma_points_.col(2*augmented_size_).head(3) = processed_state;
        processed_augmented_mean += Sigma_WM0_ * sigma_points_.col(2*augmented_size_);
    }

    Eigen::MatrixXd processed_augmented_cov = Eigen::MatrixXd::Zero(augmented_size_, augmented_size_);
    for(size_t i=0; i<2*augmented_size_; ++i) {
        const Eigen::VectorXd delta_x = sigma_points_.col(i) - processed_augmented_mean;
        processed_augmented_cov += Sigma_WCI_ * (delta_x * delta_x.transpose());
    }
    {
        const Eigen::VectorXd delta_x = sigma_points_.col(2*augmented_size_) - processed_augmented_mean;
        processed_augmented_cov += Sigma_WC0_ * (delta_x * delta_x.transpose());
    }

    StateInfo result;
    result.mean = processed_augmented_mean.head(3);
    result.covariance = processed_augmented_cov.block(0, 0, 3, 3);

    return result;
}

StateInfo NormalVehicleUKF::update(const NormalVehicle::StateInfo &state_info,
                                   const Eigen::Vector2d &observed_values,
                                   const std::map<int, std::shared_ptr<BaseDistribution>>& system_noise_map,
                                   const std::map<int, std::shared_ptr<BaseDistribution>>& measurement_noise_map)
{
    const auto dist_wx = system_noise_map.at(SYSTEM_NOISE::WX);
    const auto dist_wy = system_noise_map.at(SYSTEM_NOISE::WY);
    const auto dist_wyaw = system_noise_map.at(SYSTEM_NOISE::WYAW);
    const auto dist_mr = measurement_noise_map.at(OBSERVATION_NOISE::WR);
    const auto dist_myaw = measurement_noise_map.at(OBSERVATION_NOISE::WYAW);

    Eigen::VectorXd augmented_mean = Eigen::VectorXd::Zero(augmented_size_);
    augmented_mean.head(3) = state_info.mean;
    augmented_mean(3) = dist_wx->calc_mean();
    augmented_mean(4) = dist_wy->calc_mean();
    augmented_mean(5) = dist_wyaw->calc_mean();
    augmented_mean(6) = dist_mr->calc_mean();
    augmented_mean(7) = dist_myaw->calc_mean();

    Eigen::MatrixXd augmented_cov = Eigen::MatrixXd::Zero(augmented_size_, augmented_size_);
    augmented_cov.block(0, 0, 3, 3) = state_info.covariance;
    augmented_cov(3, 3) = dist_wx->calc_variance();
    augmented_cov(4, 4) = dist_wy->calc_variance();
    augmented_cov(5, 5) = dist_wyaw->calc_variance();
    augmented_cov(6, 6) = dist_mr->calc_variance();
    augmented_cov(7, 7) = dist_myaw->calc_variance();

    assert((augmented_cov*(augmented_size_ + lambda_)).llt().info() == Eigen::Success);
    const Eigen::MatrixXd augmented_cov_squared = (augmented_cov * (augmented_size_ + lambda_)).llt().matrixL();

    // Resample Sigma Points
    for(size_t i=0; i<augmented_size_; ++i) {
        sigma_points_.col(i) = augmented_mean + augmented_cov_squared.col(i);
    }
    for(size_t i=augmented_size_; i<2*augmented_size_; ++i) {
        sigma_points_.col(i) = augmented_mean - augmented_cov_squared.col(i-augmented_size_);
    }
    {
        sigma_points_.col(2*augmented_size_) = augmented_mean;
    }

    // Calculate mean y
    Eigen::MatrixXd observed_sigma_points = Eigen::MatrixXd::Zero(2, 2*augmented_size_+1);
    Eigen::Vector2d y_mean = Eigen::Vector2d::Zero();
    for(size_t i=0; i<2*augmented_size_+1; ++i) {
        const Eigen::Vector2d y = model_.observe(sigma_points_.col(i).head(3), sigma_points_.col(i).segment(6, 2));
        observed_sigma_points.col(i) = y;
        if(i==2*augmented_size_) {
            y_mean += Sigma_WM0_ * y;
        } else {
            y_mean += Sigma_WMI_ * y;
        }
    }

    Eigen::MatrixXd Pyy = Eigen::MatrixXd::Zero(2, 2);
    for(size_t i=0; i<2*augmented_size_+1; ++i) {
        const Eigen::Vector2d delta_y = observed_sigma_points.col(i) - y_mean;
        if(i==2*augmented_size_) {
            Pyy += Sigma_WC0_ * (delta_y * delta_y.transpose());
        } else {
            Pyy += Sigma_WCI_ * (delta_y * delta_y.transpose());
        }
    }

    Eigen::MatrixXd Pxy = Eigen::MatrixXd::Zero(3, 2);
    for(size_t i=0; i<2*augmented_size_+1; ++i) {
        const Eigen::Vector3d delta_x = sigma_points_.col(i).head(3) - state_info.mean;
        const Eigen::Vector2d delta_y = observed_sigma_points.col(i) - y_mean;
        if(i==2*augmented_size_) {
            Pxy += Sigma_WC0_ * (delta_x * delta_y.transpose());
        } else {
            Pxy += Sigma_WCI_ * (delta_x * delta_y.transpose());
        }
    }

    const Eigen::MatrixXd K = Pxy*Pyy.inverse();

    StateInfo result;
    result.mean = state_info.mean + K*(observed_values - y_mean);
    result.covariance = state_info.covariance - K * Pyy * K.transpose();

    return result;
}
