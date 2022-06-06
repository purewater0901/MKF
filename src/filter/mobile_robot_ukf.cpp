#include "filter/mobile_robot_ukf.h"

MobileRobotUKF::MobileRobotUKF() : augmented_size_(10),
                                   alpha_squared_(1.0),
                                   beta_(0.0),
                                   kappa_(3.0),
                                   lambda_(alpha_squared_*(augmented_size_ + kappa_) - augmented_size_)
{
    sigma_points_ = Eigen::MatrixXd::Zero(10, 21);

    Sigma_WM0_ = lambda_/(augmented_size_ + lambda_);
    Sigma_WC0_ = Sigma_WM0_ + (1.0 - alpha_squared_ + beta_);
    Sigma_WMI_ = 1.0 / (2.0 * (augmented_size_ + lambda_));
    Sigma_WCI_ = Sigma_WMI_;
}

StateInfo MobileRobotUKF::predict(const StateInfo& state_info,
                                  const Eigen::Vector2d & control_inputs,
                                  const double dt,
                                  const std::map<int, std::shared_ptr<BaseDistribution>>& system_noise_map,
                                  const std::map<int, std::shared_ptr<BaseDistribution>>& measurement_noise_map)
{
    const auto dist_wv = system_noise_map.at(SYSTEM_NOISE::WV);
    const auto dist_wyaw = system_noise_map.at(SYSTEM_NOISE::WYAW);
    const auto dist_mx = measurement_noise_map.at(OBSERVATION_NOISE::WX);
    const auto dist_my = measurement_noise_map.at(OBSERVATION_NOISE::WY);
    const auto dist_mvc = measurement_noise_map.at(OBSERVATION_NOISE::WVC);
    const auto dist_myaw = measurement_noise_map.at(OBSERVATION_NOISE::WYAW);

    Eigen::VectorXd augmented_mean = Eigen::VectorXd::Zero(10);
    augmented_mean.head(4) = state_info.mean;
    augmented_mean(4) = dist_wv->calc_mean();
    augmented_mean(5) = dist_wyaw->calc_mean();
    augmented_mean(6) = dist_mx->calc_mean();
    augmented_mean(7) = dist_my->calc_mean();
    augmented_mean(8) = dist_mvc->calc_mean();
    augmented_mean(9) = dist_myaw->calc_mean();

    Eigen::MatrixXd augmented_cov = Eigen::MatrixXd::Zero(10, 10);
    augmented_cov.block(0, 0, 4, 4) = state_info.covariance;
    augmented_cov(4, 4) = dist_wv->calc_variance();
    augmented_cov(5, 5) = dist_wyaw->calc_variance();
    augmented_cov(6, 6) = dist_mx->calc_variance();
    augmented_cov(7, 7) = dist_my->calc_variance();
    augmented_cov(8, 8) = dist_mvc->calc_variance();
    augmented_cov(9, 9) = dist_myaw->calc_variance();

    assert((augmented_cov*(augmented_size_ + lambda_)).llt().info() == Eigen::Success);
    const Eigen::MatrixXd augmented_cov_squared = (augmented_cov * (augmented_size_ + lambda_)).llt().matrixL();

    Eigen::VectorXd processed_augmented_mean = Eigen::VectorXd::Zero(10);
    for(size_t i=0; i<10; ++i) {
        sigma_points_.col(i) = augmented_mean + augmented_cov_squared.col(i);
        const Eigen::Vector4d processed_state = model_.propagate(sigma_points_.col(i).head(4), control_inputs, sigma_points_.col(i).segment(4, 2), dt);
        sigma_points_.col(i).head(4) = processed_state;
        processed_augmented_mean += Sigma_WMI_ * sigma_points_.col(i);
    }
    for(size_t i=10; i<20; ++i) {
        sigma_points_.col(i) = augmented_mean - augmented_cov_squared.col(i-10);
        const Eigen::Vector4d processed_state = model_.propagate(sigma_points_.col(i).head(4), control_inputs, sigma_points_.col(i).segment(4, 2), dt);
        sigma_points_.col(i).head(4) = processed_state;
        processed_augmented_mean += Sigma_WMI_ * sigma_points_.col(i);
    }
    {
        sigma_points_.col(20) = augmented_mean;
        const Eigen::Vector4d processed_state = model_.propagate(sigma_points_.col(20).head(4), control_inputs, sigma_points_.col(20).segment(4, 2), dt);
        sigma_points_.col(20).head(4) = processed_state;
        processed_augmented_mean += Sigma_WM0_ * sigma_points_.col(20);
    }

    Eigen::MatrixXd processed_augmented_cov = Eigen::MatrixXd::Zero(10, 10);
    for(size_t i=0; i<20; ++i) {
        const Eigen::VectorXd delta_x = sigma_points_.col(i) - processed_augmented_mean;
        processed_augmented_cov += Sigma_WCI_ * (delta_x * delta_x.transpose());
    }
    {
        const Eigen::VectorXd delta_x = sigma_points_.col(20) - processed_augmented_mean;
        processed_augmented_cov += Sigma_WC0_ * (delta_x * delta_x.transpose());
    }

    StateInfo result;
    result.mean = processed_augmented_mean.head(4);
    result.covariance = processed_augmented_cov.block(0, 0, 4, 4);

    return result;
}

StateInfo MobileRobotUKF::update(const MobileRobot::StateInfo &state_info,
                                 const Eigen::Vector3d &observed_values,
                                 const std::map<int, std::shared_ptr<BaseDistribution>>& system_noise_map,
                                 const std::map<int, std::shared_ptr<BaseDistribution>>& measurement_noise_map)
{
    const auto dist_wv = system_noise_map.at(SYSTEM_NOISE::WV);
    const auto dist_wyaw = system_noise_map.at(SYSTEM_NOISE::WYAW);
    const auto dist_mx = measurement_noise_map.at(OBSERVATION_NOISE::WX);
    const auto dist_my = measurement_noise_map.at(OBSERVATION_NOISE::WY);
    const auto dist_mvc = measurement_noise_map.at(OBSERVATION_NOISE::WVC);
    const auto dist_myaw = measurement_noise_map.at(OBSERVATION_NOISE::WYAW);

    Eigen::VectorXd augmented_mean = Eigen::VectorXd::Zero(10);
    augmented_mean.head(4) = state_info.mean;
    augmented_mean(4) = dist_wv->calc_mean();
    augmented_mean(5) = dist_wyaw->calc_mean();
    augmented_mean(6) = dist_mx->calc_mean();
    augmented_mean(7) = dist_my->calc_mean();
    augmented_mean(8) = dist_mvc->calc_mean();
    augmented_mean(9) = dist_myaw->calc_mean();

    Eigen::MatrixXd augmented_cov = Eigen::MatrixXd::Zero(10, 10);
    augmented_cov.block(0, 0, 4, 4) = state_info.covariance;
    augmented_cov(4, 4) = dist_wv->calc_variance();
    augmented_cov(5, 5) = dist_wyaw->calc_variance();
    augmented_cov(6, 6) = dist_mx->calc_variance();
    augmented_cov(7, 7) = dist_my->calc_variance();
    augmented_cov(8, 8) = dist_mvc->calc_variance();
    augmented_cov(9, 9) = dist_myaw->calc_variance();

    assert((augmented_cov*(augmented_size_ + lambda_)).llt().info() == Eigen::Success);
    const Eigen::MatrixXd augmented_cov_squared = (augmented_cov * (augmented_size_ + lambda_)).llt().matrixL();

    // Resample Sigma Points
    for(size_t i=0; i<10; ++i) {
        sigma_points_.col(i) = augmented_mean + augmented_cov_squared.col(i);
    }
    for(size_t i=10; i<20; ++i) {
        sigma_points_.col(i) = augmented_mean - augmented_cov_squared.col(i-10);
    }
    {
        sigma_points_.col(20) = augmented_mean;
    }

    // Calculate mean y
    Eigen::MatrixXd observed_sigma_points = Eigen::MatrixXd::Zero(3, 21);
    Eigen::Vector3d y_mean = Eigen::Vector3d::Zero();
    for(size_t i=0; i<21; ++i) {
        const Eigen::Vector3d y = model_.observe(sigma_points_.col(i).head(4), sigma_points_.col(i).segment(6, 4));
        observed_sigma_points.col(i) = y;
        if(i==20) {
            y_mean += Sigma_WM0_ * y;
        } else {
            y_mean += Sigma_WMI_ * y;
        }
    }

    Eigen::MatrixXd Pyy = Eigen::MatrixXd::Zero(3, 3);
    for(size_t i=0; i<21; ++i) {
        const Eigen::Vector3d delta_y = observed_sigma_points.col(i) - y_mean;
        if(i==20) {
            Pyy += Sigma_WC0_ * (delta_y * delta_y.transpose());
        } else {
            Pyy += Sigma_WCI_ * (delta_y * delta_y.transpose());
        }
    }

    Eigen::MatrixXd Pxy = Eigen::MatrixXd::Zero(4, 3);
    for(size_t i=0; i<21; ++i) {
        const Eigen::Vector4d delta_x = sigma_points_.col(i).head(4) - state_info.mean;
        const Eigen::Vector3d delta_y = observed_sigma_points.col(i) - y_mean;
        if(i==20) {
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
