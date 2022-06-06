#include "filter/kinematic_vehicle_ekf.h"
#include "utilities.h"

using namespace KinematicVehicle;

StateInfo KinematicVehicleEKF::predict(const StateInfo &state_info,
                                       const Eigen::Vector2d &inputs,
                                       const double dt,
                                       const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    KinematicVehicleModel vehicle_model_;
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

StateInfo KinematicVehicleEKF::update(const KinematicVehicle::StateInfo& state_info,
                                      const Eigen::Vector3d& y,
                                      const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    KinematicVehicleModel vehicle_model_;
    const auto wr_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WR);
    const auto wvc_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WVC);
    const auto wyaw_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WYAW);

    // Update state mean
    const double mean_r = wr_dist_ptr->calc_mean();
    const double mean_vc = wvc_dist_ptr->calc_mean();
    const double mean_yaw = wyaw_dist_ptr->calc_mean();
    const auto predicted_y = vehicle_model_.observe(state_info.mean, {mean_r, mean_vc, mean_yaw});

    // Covariance Update
    /*  == Nonlinear model ==
     *
     * r = x^2 + y^2
     * vc = v_k * cos(yaw_k)
     * yaw_k = yaw_k
     *
     */
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 4);
    H(OBSERVATION::IDX::R, STATE::IDX::X) = 2.0 * state_info.mean(STATE::IDX::X);
    H(OBSERVATION::IDX::R, STATE::IDX::Y) = 2.0 * state_info.mean(STATE::IDX::Y);
    H(OBSERVATION::IDX::VC, STATE::IDX::V) = std::cos(state_info.mean(STATE::IDX::YAW));
    H(OBSERVATION::IDX::VC, STATE::IDX::YAW) = -state_info.mean(STATE::IDX::V) * std::sin(state_info.mean(STATE::IDX::YAW));
    H(OBSERVATION::IDX::YAW, STATE::IDX::YAW) = 1.0;

    Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
    R(OBSERVATION_NOISE::IDX::WR, OBSERVATION_NOISE::IDX::WR) = wr_dist_ptr->calc_variance();
    R(OBSERVATION_NOISE::IDX::WVC, OBSERVATION_NOISE::IDX::WVC) = wvc_dist_ptr->calc_variance();
    R(OBSERVATION_NOISE::IDX::WYAW, OBSERVATION_NOISE::IDX::WYAW) = wyaw_dist_ptr->calc_variance();

    const Eigen::Matrix3d S = H*state_info.covariance*H.transpose() + R;
    const auto K = state_info.covariance * H.transpose() * S.inverse();

    StateInfo updated_info;
    updated_info.mean = state_info.mean + K * (y - predicted_y);
    updated_info.covariance = state_info.covariance - K * H * state_info.covariance;

    return updated_info;
}
