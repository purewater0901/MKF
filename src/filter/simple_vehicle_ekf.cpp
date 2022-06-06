#include "filter/simple_vehicle_ekf.h"

using namespace SimpleVehicle;

StateInfo SimpleVehicleEKF::predict(const StateInfo& state_info,
                                    const Eigen::Vector2d& inputs,
                                    const double dt,
                                    const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    SimpleVehicleModel vehicle_model_;
    const auto wv_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WV);
    const auto wu_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WU);

    // State mean prediction
    StateInfo predicted_info;
    predicted_info.mean = vehicle_model_.propagate(state_info.mean, inputs, {wv_dist_ptr->calc_mean(), wu_dist_ptr->calc_mean()});

    // Covariance Prediction
    /*  == Nonlinear model ==
     *
     * x_{k+1}   = x_k + (uv_k+wv) * cos(yaw_k) * dt
     * y_{k+1}   = y_k + (uv_k+wv) * sin(yaw_k) * dt
     * yaw_{k+1} = yaw_k + (uw_k+wu) * dt
     *
     */
    const double& x = state_info.mean(STATE::IDX::X);
    const double& y = state_info.mean(STATE::IDX::Y);
    const double& yaw = state_info.mean(STATE::IDX::YAW);
    const double& uv_k = inputs(0);
    const double& uw_k = inputs(1);
    const double wv = wv_dist_ptr->calc_mean();
    const double wu = wu_dist_ptr->calc_mean();

    Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
    A(STATE::IDX::X, STATE::IDX::YAW) =  -(uv_k + wv) * std::sin(yaw);
    A(STATE::IDX::Y, STATE::IDX::YAW) =   (uw_k + wu) * std::cos(yaw);

    Eigen::Matrix2d Q;
    Q << wv_dist_ptr->calc_variance(), 0.0,
         0.0, wu_dist_ptr->calc_variance();
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(3, 2);
    L(STATE::IDX::X, SYSTEM_NOISE::IDX::WV) = std::cos(yaw) * dt;
    L(STATE::IDX::Y, SYSTEM_NOISE::IDX::WV) = std::sin(yaw) * dt;
    L(STATE::IDX::YAW, SYSTEM_NOISE::IDX::WU) = dt;

    predicted_info.covariance = A * state_info.covariance * A.transpose() + L*Q*L.transpose();

    return predicted_info;
}

StateInfo SimpleVehicleEKF::update(const StateInfo& state_info,
                                   const Eigen::Vector2d& y_meas,
                                   const Eigen::Vector2d& landmark,
                                   const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    SimpleVehicleModel vehicle_model_;
    const auto wr_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WR);
    const auto wa_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WA);

    // Update state mean
    const double mean_wr = wr_dist_ptr->calc_mean();
    const double mean_wa = wa_dist_ptr->calc_mean();
    const auto predicted_y = vehicle_model_.observe(state_info.mean, {mean_wr, mean_wa}, landmark);

    // Covariance Update
    /*  == Nonlinear model ==
     *
     * rcos = (x_land - x) * cos(yaw) + (y_land - y) * sin(yaw) + mrcos
     * rsin = (y_land - y) * cos(yaw) - (x_land - x) * sin(yaw) + mrsin
     *
     */

    const double& x = state_info.mean(STATE::IDX::X);
    const double& y = state_info.mean(STATE::IDX::Y);
    const double& yaw = state_info.mean(STATE::IDX::YAW);
    const double& x_land = landmark(0);
    const double& y_land = landmark(1);
    const double rcos_bearing = (x_land - x) * std::cos(yaw) + (y_land - y) * std::sin(yaw);
    const double rsin_bearing = (y_land - y) * std::cos(yaw) - (x_land - x) * std::sin(yaw);

    const double drcos_bearing_dx = -std::cos(yaw);
    const double drcos_bearing_dy = -std::sin(yaw);
    const double drcos_bearing_dyaw = -(x_land - x) * std::sin(yaw) + (y_land - y) * std::cos(yaw);
    const double drsin_bearing_dx = std::sin(yaw);
    const double drsin_bearing_dy = -std::cos(yaw);
    const double drsin_bearing_dyaw = -(y_land - y) * std::sin(yaw) - (x_land - x) * std::cos(yaw);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 3);
    H(OBSERVATION::IDX::RCOS, STATE::IDX::X) = mean_wr * std::cos(mean_wa) * drcos_bearing_dx - mean_wr * std::sin(mean_wa) * drsin_bearing_dx;
    H(OBSERVATION::IDX::RCOS, STATE::IDX::Y) = mean_wr * std::cos(mean_wa) * drcos_bearing_dy - mean_wr * std::sin(mean_wa) * drsin_bearing_dy;
    H(OBSERVATION::IDX::RCOS, STATE::IDX::YAW) = mean_wr * std::cos(mean_wa) * drcos_bearing_dyaw - mean_wr * std::sin(mean_wa) * drsin_bearing_dyaw;
    H(OBSERVATION::IDX::RSIN, STATE::IDX::X) = mean_wr * std::cos(mean_wa) * drsin_bearing_dx + mean_wr * std::sin(mean_wa) * drcos_bearing_dx;
    H(OBSERVATION::IDX::RSIN, STATE::IDX::Y) = mean_wr * std::cos(mean_wa) * drsin_bearing_dy + mean_wr * std::sin(mean_wa) * drcos_bearing_dy;
    H(OBSERVATION::IDX::RSIN, STATE::IDX::YAW) = mean_wr * std::cos(mean_wa) * drsin_bearing_dyaw + mean_wr * std::sin(mean_wa) * drcos_bearing_dyaw;

    Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
    R(OBSERVATION_NOISE::IDX::WR, OBSERVATION_NOISE::IDX::WR) = wr_dist_ptr->calc_variance();
    R(OBSERVATION_NOISE::IDX::WA, OBSERVATION_NOISE::IDX::WA) = wa_dist_ptr->calc_variance();

    Eigen::Matrix2d L = Eigen::Matrix2d::Zero();
    L(OBSERVATION::IDX::RCOS, OBSERVATION_NOISE::IDX::WR) = std::cos(mean_wa) * rcos_bearing - std::sin(mean_wa) * rsin_bearing;
    L(OBSERVATION::IDX::RSIN, OBSERVATION_NOISE::IDX::WR) = std::cos(mean_wa) * rsin_bearing + std::sin(mean_wa) * rcos_bearing;
    L(OBSERVATION::IDX::RCOS, OBSERVATION_NOISE::IDX::WA) = -mean_wr * std::sin(mean_wa) * rcos_bearing - mean_wr * std::cos(mean_wa) * rsin_bearing;
    L(OBSERVATION::IDX::RSIN, OBSERVATION_NOISE::IDX::WA) = -mean_wr * std::sin(mean_wa) * rsin_bearing + mean_wr * std::cos(mean_wa) * rcos_bearing;

    const Eigen::Matrix2d S = H*state_info.covariance*H.transpose() + L*R*L.transpose();
    const auto K = state_info.covariance * H.transpose() * S.inverse();

    StateInfo updated_info;
    updated_info.mean = state_info.mean + K * (y_meas - predicted_y);
    updated_info.covariance = state_info.covariance - K * H * state_info.covariance;

    return updated_info;
}