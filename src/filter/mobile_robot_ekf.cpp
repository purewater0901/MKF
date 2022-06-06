#include "filter/mobile_robot_ekf.h"
#include "utilities.h"

using namespace MobileRobot;

StateInfo MobileRobotEKF::predict(const StateInfo &state_info,
                                  const Eigen::Vector2d &inputs,
                                  const double dt,
                                  const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    MobileRobotModel vehicle_model_;
    const auto wv_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WV);
    const auto wyaw_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WYAW);

    // State mean prediction
    StateInfo predicted_info;
    predicted_info.mean = vehicle_model_.propagate(state_info.mean, inputs, {wv_dist_ptr->calc_mean(), wyaw_dist_ptr->calc_mean()}, dt);

    // Covariance Prediction
    /*  == Nonlinear model ==
     *
     * x_{k+1}   = x_k + v_k * cos(yaw_k) * dt
     * y_{k+1}   = y_k + v_k * sin(yaw_k) * dt
     * v_{k+1}   = v_k + a_k * dt
     * yaw_{k+1} = yaw_k + u_k * dt
     *
     */
    const double v_k = state_info.mean(STATE::IDX::V);
    Eigen::Matrix4d A = Eigen::Matrix4d::Identity();
    A(STATE::IDX::X, STATE::IDX::V) = std::cos(state_info.mean(STATE::IDX::YAW)) * dt;
    A(STATE::IDX::X, STATE::IDX::YAW) =  -v_k * std::sin(state_info.mean(STATE::IDX::YAW)) * dt;
    A(STATE::IDX::Y, STATE::IDX::V) = std::sin(state_info.mean(STATE::IDX::YAW)) * dt;
    A(STATE::IDX::Y, STATE::IDX::YAW) =   v_k * std::cos(state_info.mean(STATE::IDX::YAW)) * dt;
    Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
    Q(STATE::IDX::V, STATE::IDX::V) = wv_dist_ptr->calc_variance();
    Q(STATE::IDX::YAW, STATE::IDX::YAW) = wyaw_dist_ptr->calc_variance();
    predicted_info.covariance = A * state_info.covariance * A.transpose() + Q;

    return predicted_info;
}

StateInfo MobileRobotEKF::update(const StateInfo& state_info,
                                 const Eigen::Vector3d& y,
                                 const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    MobileRobotModel vehicle_model_;
    const auto wx_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WX);
    const auto wy_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WY);
    const auto wvc_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WVC);
    const auto wyaw_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WYAW);

    // Update state mean
    const double mean_x = wx_dist_ptr->calc_mean();
    const double mean_y = wy_dist_ptr->calc_mean();
    const double mean_vc = wvc_dist_ptr->calc_mean();
    const double mean_yaw = wyaw_dist_ptr->calc_mean();
    const auto predicted_y = vehicle_model_.observe(state_info.mean, {mean_x, mean_y, mean_vc, mean_yaw});

    // Covariance Update
    /*  == Nonlinear model ==
     *
     * x = x + v * wx
     * y = y + v * wy
     * vc = (v + wv) * cos(yaw_k + wyaw)
     *
     */
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 4);
    H(OBSERVATION::IDX::X, STATE::IDX::X) = 1.0;
    H(OBSERVATION::IDX::X, STATE::IDX::V) = wx_dist_ptr->calc_mean();
    H(OBSERVATION::IDX::Y, STATE::IDX::Y) = 1.0;
    H(OBSERVATION::IDX::Y, STATE::IDX::V) = wy_dist_ptr->calc_mean();
    H(OBSERVATION::IDX::VC, STATE::IDX::V) = std::cos(state_info.mean(STATE::IDX::YAW) + wyaw_dist_ptr->calc_mean());
    H(OBSERVATION::IDX::VC, STATE::IDX::YAW) = -(state_info.mean(STATE::IDX::V) + wvc_dist_ptr->calc_mean()) * std::sin(state_info.mean(STATE::IDX::YAW) + wyaw_dist_ptr->calc_mean());

    Eigen::Matrix4d R = Eigen::Matrix4d::Zero();
    R(OBSERVATION_NOISE::IDX::WX, OBSERVATION_NOISE::IDX::WX) = wx_dist_ptr->calc_variance();
    R(OBSERVATION_NOISE::IDX::WY, OBSERVATION_NOISE::IDX::WY) = wy_dist_ptr->calc_variance();
    R(OBSERVATION_NOISE::IDX::WVC, OBSERVATION_NOISE::IDX::WVC) = wvc_dist_ptr->calc_variance();
    R(OBSERVATION_NOISE::IDX::WYAW, OBSERVATION_NOISE::IDX::WYAW) = wyaw_dist_ptr->calc_variance();

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(3, 4);
    M(OBSERVATION::IDX::X, OBSERVATION_NOISE::IDX::WX) = state_info.mean(STATE::IDX::V);
    M(OBSERVATION::IDX::Y, OBSERVATION_NOISE::IDX::WY) = state_info.mean(STATE::IDX::V);
    M(OBSERVATION::IDX::VC, OBSERVATION_NOISE::IDX::WVC) = std::cos(state_info.mean(STATE::IDX::YAW) + wyaw_dist_ptr->calc_mean());
    M(OBSERVATION::IDX::VC, OBSERVATION_NOISE::IDX::WYAW) = -(state_info.mean(STATE::IDX::V) + wvc_dist_ptr->calc_mean()) * std::sin(state_info.mean(STATE::IDX::YAW) + wyaw_dist_ptr->calc_mean());

    const Eigen::Matrix3d S = H*state_info.covariance*H.transpose() + M*R*M.transpose();
    const auto K = state_info.covariance * H.transpose() * S.inverse();

    StateInfo updated_info;
    updated_info.mean = state_info.mean + K * (y - predicted_y);
    updated_info.covariance = state_info.covariance - K * H * state_info.covariance;

    return updated_info;
}