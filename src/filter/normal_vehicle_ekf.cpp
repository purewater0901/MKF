#include "filter/normal_vehicle_ekf.h"

using namespace NormalVehicle;

StateInfo NormalVehicleEKF::predict(const StateInfo &state_info,
                                    const Eigen::Vector2d &inputs,
                                    const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    NormalVehicleModel vehicle_model_;
    const auto wx_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WX);
    const auto wy_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WY);
    const auto wyaw_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WYAW);

    // State mean prediction
    StateInfo predicted_info;
    predicted_info.mean = vehicle_model_.propagate(state_info.mean, inputs, {wx_dist_ptr->calc_mean(), wy_dist_ptr->calc_mean(), wyaw_dist_ptr->calc_mean()});

    // Covariance Prediction
    /*  == Nonlinear model ==
     *
     * x_{k+1}   = x_k + v_k * cos(yaw_k) * dt + wx
     * y_{k+1}   = y_k + v_k * sin(yaw_k) * dt + wy
     * yaw_{k+1} = yaw_k + u_k * dt + wyaw
     *
     */
    const double v_k = inputs(INPUT::IDX::V);
    Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
    A(STATE::IDX::X, STATE::IDX::YAW) =  -v_k * std::sin(state_info.mean(STATE::IDX::YAW));
    A(STATE::IDX::Y, STATE::IDX::YAW) =   v_k * std::cos(state_info.mean(STATE::IDX::YAW));
    Eigen::Matrix3d Q = Eigen::Matrix3d::Zero();
    Q(SYSTEM_NOISE::IDX::WX, SYSTEM_NOISE::IDX::WX) = wx_dist_ptr->calc_variance();
    Q(SYSTEM_NOISE::IDX::WY, SYSTEM_NOISE::IDX::WY) = wy_dist_ptr->calc_variance();
    Q(SYSTEM_NOISE::IDX::WYAW, SYSTEM_NOISE::IDX::WYAW) = wyaw_dist_ptr->calc_variance();
    predicted_info.covariance = A * state_info.covariance * A.transpose() + Q;

    return predicted_info;
}

StateInfo NormalVehicleEKF::update(const NormalVehicle::StateInfo& state_info,
                                   const Eigen::Vector2d& y,
                                   const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    NormalVehicleModel vehicle_model_;
    const auto wr_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WR);
    const auto wyaw_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WYAW);

    // Update state mean
    auto predicted_y = vehicle_model_.observe(state_info.mean, {wr_dist_ptr->calc_mean(), wyaw_dist_ptr->calc_mean()});

    // Covariance Update
    /*  == Nonlinear model ==
     *
     * r = x^2 + y^2 + wr
     * yaw_k = yaw_k + w_yaw
     *
     */
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 3);
    H(OBSERVATION::IDX::R, STATE::IDX::X) = 2.0 * state_info.mean(STATE::IDX::X);
    H(OBSERVATION::IDX::R, STATE::IDX::Y) = 2.0 * state_info.mean(STATE::IDX::Y);
    H(OBSERVATION::IDX::YAW, STATE::IDX::YAW) = 1.0;
    Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
    R(OBSERVATION_NOISE::IDX::WR, OBSERVATION_NOISE::IDX::WR) = wr_dist_ptr->calc_variance();
    R(OBSERVATION_NOISE::IDX::WYAW, OBSERVATION_NOISE::IDX::WYAW) = wyaw_dist_ptr->calc_variance();

    const Eigen::Matrix2d S = H*state_info.covariance*H.transpose() + R;
    const auto K = state_info.covariance * H.transpose() * S.inverse();

    StateInfo updated_info;
    updated_info.mean = state_info.mean + K * (y - predicted_y);
    updated_info.covariance = state_info.covariance - K * H * state_info.covariance;

    return updated_info;
}
