#include "filter/kinematic_vehilce_nkf.h"

#include "distribution/four_dimensional_normal_distribution.h"

using namespace KinematicVehicle;

StateInfo KinematicVehicleNKF::predict(const StateInfo &state_info,
                                       const Eigen::Vector2d &control_inputs,
                                       const double dt,
                                       const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map)
{
    // Step1. Approximate to Gaussian Distribution
    const auto state_mean = state_info.mean;
    const auto state_cov = state_info.covariance;
    FourDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);

    // Step2. State Moment
    KinematicVehicleModel::StateMoments moment;
    moment.xPow1 = dist.calc_moment(STATE::IDX::X, 1);
    moment.yPow1 = dist.calc_moment(STATE::IDX::Y, 1);
    moment.vPow1 = dist.calc_moment(STATE::IDX::V, 1);
    moment.yawPow1 = dist.calc_moment(STATE::IDX::YAW, 1);
    moment.cPow1 = dist.calc_cos_moment(STATE::IDX::YAW, 1);
    moment.sPow1 = dist.calc_sin_moment(STATE::IDX::YAW, 1);

    moment.xPow2 = dist.calc_moment(STATE::IDX::X, 2);
    moment.yPow2 = dist.calc_moment(STATE::IDX::Y, 2);
    moment.vPow2 = dist.calc_moment(STATE::IDX::V, 2);
    moment.yawPow2 = dist.calc_moment(STATE::IDX::YAW, 2);
    moment.cPow2 = dist.calc_cos_moment(STATE::IDX::YAW, 2);
    moment.sPow2 = dist.calc_sin_moment(STATE::IDX::YAW, 2);
    moment.xPow1_yPow1 = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::Y);
    moment.xPow1_yawPow1 = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::YAW);
    moment.yPow1_yawPow1 = dist.calc_cross_second_moment(STATE::IDX::Y, STATE::IDX::YAW);
    moment.vPow1_xPow1 = dist.calc_cross_second_moment(STATE::IDX::V, STATE::IDX::X);
    moment.vPow1_yPow1 = dist.calc_cross_second_moment(STATE::IDX::V, STATE::IDX::Y);
    moment.vPow1_yawPow1 = dist.calc_cross_second_moment(STATE::IDX::V, STATE::IDX::YAW);
    moment.vPow1_cPow1 = dist.calc_x_cos_z_moment(STATE::IDX::V, STATE::IDX::YAW);
    moment.vPow1_sPow1 = dist.calc_x_sin_z_moment(STATE::IDX::V, STATE::IDX::YAW);
    moment.cPow1_xPow1 = dist.calc_x_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    moment.sPow1_xPow1 = dist.calc_x_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    moment.cPow1_yPow1 = dist.calc_x_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    moment.sPow1_yPow1 = dist.calc_x_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    moment.cPow1_yawPow1 = dist.calc_x_cos_x_moment(STATE::IDX::YAW, 1, 1);
    moment.sPow1_yawPow1 = dist.calc_x_sin_x_moment(STATE::IDX::YAW, 1, 1);
    moment.cPow1_sPow1 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 1, 1);

    moment.vPow1_cPow2 = dist.calc_x_cos_y_cos_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    moment.vPow1_sPow2 = dist.calc_x_sin_y_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    moment.vPow2_cPow1 = dist.calc_xx_cos_z_moment(STATE::IDX::V, STATE::IDX::YAW);
    moment.vPow2_sPow1 = dist.calc_xx_sin_z_moment(STATE::IDX::V, STATE::IDX::YAW);
    moment.vPow1_cPow1_xPow1 = dist.calc_xy_cos_z_moment(STATE::IDX::X, STATE::IDX::V, STATE::IDX::YAW);
    moment.vPow1_cPow1_yPow1 = dist.calc_xy_cos_z_moment(STATE::IDX::Y, STATE::IDX::V, STATE::IDX::YAW);
    moment.vPow1_cPow1_yawPow1 = dist.calc_xy_cos_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    moment.vPow1_sPow1_xPow1 = dist.calc_xy_sin_z_moment(STATE::IDX::X, STATE::IDX::V, STATE::IDX::YAW);
    moment.vPow1_sPow1_yPow1 = dist.calc_xy_sin_z_moment(STATE::IDX::Y, STATE::IDX::V, STATE::IDX::YAW);
    moment.vPow1_sPow1_yawPow1 = dist.calc_xy_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    moment.vPow1_cPow1_sPow1 = dist.calc_x_cos_y_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW);

    moment.vPow2_cPow2 = dist.calc_xx_cos_y_cos_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    moment.vPow2_sPow2 = dist.calc_xx_sin_y_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    moment.vPow2_cPow1_sPow1 = dist.calc_xx_cos_y_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW);

    // Step3. Control Input
    KinematicVehicleModel::Controls controls;
    controls.a = control_inputs(INPUT::IDX::A);
    controls.u = control_inputs(INPUT::IDX::U);
    controls.cu = std::cos(controls.u);
    controls.su = std::sin(controls.u);

    // Step4. System Noise
    const auto wv_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WV);
    const auto wyaw_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WYAW);
    KinematicVehicleModel::SystemNoiseMoments system_noise_moments;
    system_noise_moments.wvPow1 = wv_dist_ptr->calc_moment(1);
    system_noise_moments.wyawPow1 = wyaw_dist_ptr->calc_moment(1);
    system_noise_moments.cyawPow1 = wyaw_dist_ptr->calc_cos_moment(1);
    system_noise_moments.syawPow1 = wyaw_dist_ptr->calc_sin_moment(1);

    system_noise_moments.wvPow2 = wv_dist_ptr->calc_moment(2);
    system_noise_moments.wyawPow2 = wyaw_dist_ptr->calc_moment(2);
    system_noise_moments.cyawPow2 = wyaw_dist_ptr->calc_cos_moment(2);
    system_noise_moments.syawPow2 = wyaw_dist_ptr->calc_sin_moment(2);
    system_noise_moments.cyawPow1_syawPow1 = wyaw_dist_ptr->calc_cos_sin_moment(1, 1);

    // Step5. Propagate
    const auto predicted_moment = vehicle_model_.propagateStateMoments(moment, system_noise_moments, controls, dt);

    StateInfo predicted_info;
    predicted_info.mean(STATE::IDX::X) = predicted_moment.xPow1;
    predicted_info.mean(STATE::IDX::Y) = predicted_moment.yPow1;
    predicted_info.mean(STATE::IDX::V) = predicted_moment.vPow1;
    predicted_info.mean(STATE::IDX::YAW)= predicted_moment.yawPow1;

    predicted_info.covariance(STATE::IDX::X, STATE::IDX::X) = predicted_moment.xPow2 - predicted_moment.xPow1 * predicted_moment.xPow1;
    predicted_info.covariance(STATE::IDX::Y, STATE::IDX::Y) = predicted_moment.yPow2 - predicted_moment.yPow1 * predicted_moment.yPow1;
    predicted_info.covariance(STATE::IDX::V, STATE::IDX::V) = predicted_moment.vPow2 - predicted_moment.vPow1 * predicted_moment.vPow1;
    predicted_info.covariance(STATE::IDX::YAW, STATE::IDX::YAW) = predicted_moment.yawPow2 - predicted_moment.yawPow1 * predicted_moment.yawPow1;

    predicted_info.covariance(STATE::IDX::X, STATE::IDX::Y) = predicted_moment.xPow1_yPow1 - predicted_moment.xPow1 * predicted_moment.yPow1;
    predicted_info.covariance(STATE::IDX::X, STATE::IDX::V) = predicted_moment.vPow1_xPow1 - predicted_moment.xPow1 * predicted_moment.vPow1;
    predicted_info.covariance(STATE::IDX::X, STATE::IDX::YAW) = predicted_moment.xPow1_yawPow1 - predicted_moment.xPow1 * predicted_moment.yawPow1;

    predicted_info.covariance(STATE::IDX::Y, STATE::IDX::X) = predicted_info.covariance(STATE::IDX::X, STATE::IDX::Y);
    predicted_info.covariance(STATE::IDX::Y, STATE::IDX::V) = predicted_moment.vPow1_yPow1 - predicted_moment.yPow1 * predicted_moment.vPow1;
    predicted_info.covariance(STATE::IDX::Y, STATE::IDX::YAW) = predicted_moment.yPow1_yawPow1 - predicted_moment.yPow1 * predicted_moment.yawPow1;

    predicted_info.covariance(STATE::IDX::V, STATE::IDX::X) = predicted_info.covariance(STATE::IDX::X, STATE::IDX::V);
    predicted_info.covariance(STATE::IDX::V, STATE::IDX::Y) = predicted_info.covariance(STATE::IDX::Y, STATE::IDX::V);
    predicted_info.covariance(STATE::IDX::V, STATE::IDX::YAW) = predicted_moment.vPow1_yawPow1 - predicted_moment.vPow1 * predicted_moment.yawPow1;

    predicted_info.covariance(STATE::IDX::YAW, STATE::IDX::X) = predicted_info.covariance(STATE::IDX::X, STATE::IDX::YAW);
    predicted_info.covariance(STATE::IDX::YAW, STATE::IDX::Y) = predicted_info.covariance(STATE::IDX::Y, STATE::IDX::YAW);
    predicted_info.covariance(STATE::IDX::YAW, STATE::IDX::V) = predicted_info.covariance(STATE::IDX::V, STATE::IDX::YAW);

    return predicted_info;
}

StateInfo KinematicVehicleNKF::update(const KinematicVehicle::StateInfo &state_info,
                                      const Eigen::Vector3d &observed_values,
                                      const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map) {
    const auto &predicted_mean = state_info.mean;
    const auto &predicted_cov = state_info.covariance;

    FourDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);
    KinematicVehicleModel::ReducedStateMoments reduced_moments;
    reduced_moments.yawPow1 = dist.calc_moment(STATE::IDX::YAW, 1);
    reduced_moments.xPow2 = dist.calc_moment(STATE::IDX::X, 2);
    reduced_moments.yPow2 = dist.calc_moment(STATE::IDX::Y, 2);
    reduced_moments.yawPow2 = dist.calc_moment(STATE::IDX::YAW, 2);
    reduced_moments.vPow1_cPow1 = dist.calc_x_cos_z_moment(STATE::IDX::V, STATE::IDX::YAW);

    reduced_moments.vPow1_yawPow1_cPow1 = dist.calc_xy_cos_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    reduced_moments.xPow2_yawPow1 = dist.calc_cross_third_moment(STATE::IDX::X, STATE::IDX::YAW, 2, 1);
    reduced_moments.yPow2_yawPow1 = dist.calc_cross_third_moment(STATE::IDX::Y, STATE::IDX::YAW, 2, 1);

    reduced_moments.xPow4 = dist.calc_moment(STATE::IDX::X, 4);
    reduced_moments.yPow4 = dist.calc_moment(STATE::IDX::Y, 4);
    reduced_moments.xPow2_yPow2 = dist.calc_xxyy_moment(STATE::IDX::X, STATE::IDX::Y);
    reduced_moments.vPow2_cPow2 = dist.calc_xx_cos_y_cos_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    reduced_moments.xPow2_vPow1_cPow1 = dist.calc_xxy_cos_z_moment(STATE::IDX::X, STATE::IDX::V, STATE::IDX::YAW);
    reduced_moments.yPow2_vPow1_cPow1 = dist.calc_xxy_cos_z_moment(STATE::IDX::Y, STATE::IDX::V, STATE::IDX::YAW);

    // Step2. Create Observation Noise
    const auto wr_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WR);
    const auto wv_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WVC);
    const auto wyaw_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WYAW);
    KinematicVehicleModel::ObservationNoiseMoments observation_noise;
    observation_noise.wrPow1 = wr_dist_ptr->calc_moment(1);
    observation_noise.wrPow2 = wr_dist_ptr->calc_moment(2);
    observation_noise.wvPow1 = wv_dist_ptr->calc_moment(1);
    observation_noise.wvPow2 = wv_dist_ptr->calc_moment(2);
    observation_noise.wyawPow1 = wyaw_dist_ptr->calc_moment(1);
    observation_noise.wyawPow2 = wyaw_dist_ptr->calc_moment(2);

    // Step3. Get Observation Moments
    const auto observation_moments = vehicle_model_.getObservationMoments(reduced_moments, observation_noise);

    ObservedInfo observed_info;
    observed_info.mean(OBSERVATION::IDX::R) = observation_moments.rPow1;
    observed_info.mean(OBSERVATION::IDX::VC) = observation_moments.vcPow1;
    observed_info.mean(OBSERVATION::IDX::YAW) = observation_moments.yawPow1;

    observed_info.covariance(OBSERVATION::IDX::R, OBSERVATION::IDX::R) =
            observation_moments.rPow2 - observation_moments.rPow1 * observation_moments.rPow1;
    observed_info.covariance(OBSERVATION::IDX::VC, OBSERVATION::IDX::VC) =
            observation_moments.vcPow2 - observation_moments.vcPow1 * observation_moments.vcPow1;
    observed_info.covariance(OBSERVATION::IDX::YAW, OBSERVATION::IDX::YAW) =
            observation_moments.yawPow2 - observation_moments.yawPow1 * observation_moments.yawPow1;

    observed_info.covariance(OBSERVATION::IDX::R, OBSERVATION::IDX::VC) =
            observation_moments.rPow1_vcPow1 - observation_moments.rPow1 * observation_moments.vcPow1;
    observed_info.covariance(OBSERVATION::IDX::R, OBSERVATION::IDX::YAW) =
            observation_moments.rPow1_yawPow1 - observation_moments.rPow1 * observation_moments.yawPow1;
    observed_info.covariance(OBSERVATION::IDX::VC, OBSERVATION::IDX::YAW) =
            observation_moments.vcPow1_yawPow1 - observation_moments.vcPow1 * observation_moments.yawPow1;

    observed_info.covariance(OBSERVATION::IDX::VC, OBSERVATION::IDX::R) = observed_info.covariance(OBSERVATION::IDX::R, OBSERVATION::IDX::VC);
    observed_info.covariance(OBSERVATION::IDX::YAW, OBSERVATION::IDX::R) = observed_info.covariance(OBSERVATION::IDX::R, OBSERVATION::IDX::YAW);
    observed_info.covariance(OBSERVATION::IDX::YAW, OBSERVATION::IDX::VC) = observed_info.covariance(OBSERVATION::IDX::VC, OBSERVATION::IDX::YAW);

    std::cout << "observed covariance" << std::endl;
    std::cout << observed_info.covariance << std::endl;

    const auto observation_mean = observed_info.mean;
    const auto observation_cov = observed_info.covariance;

    Eigen::MatrixXd state_observation_cov(4, 3); // sigma = E[XY^T] - E[X]E[Y]^T

    state_observation_cov(STATE::IDX::X, OBSERVATION::IDX::R)
            = dist.calc_moment(STATE::IDX::X, 3) + dist.calc_cross_third_moment(STATE::IDX::X, STATE::IDX::Y, 1, 2) +
              dist.calc_moment(STATE::IDX::X, 1) * observation_noise.wrPow1
              - predicted_mean(STATE::IDX::X) * observation_mean(OBSERVATION::IDX::R); // xp * (xp^2 + yp^2)
    state_observation_cov(STATE::IDX::X, OBSERVATION::IDX::VC)
            = dist.calc_xy_cos_z_moment(STATE::IDX::X, STATE::IDX::V, STATE::IDX::YAW)
              + dist.calc_moment(STATE::IDX::X, 1) * observation_noise.wvPow1
              - predicted_mean(STATE::IDX::X) * observation_mean(OBSERVATION::IDX::VC);
    state_observation_cov(STATE::IDX::X, OBSERVATION::IDX::YAW)
            = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::YAW) +
              dist.calc_moment(STATE::IDX::X, 1) * observation_noise.wyawPow1
              - predicted_mean(STATE::IDX::X) * observation_mean(OBSERVATION::IDX::YAW); // x_p * yaw

    state_observation_cov(STATE::IDX::Y, OBSERVATION::IDX::R)
            = dist.calc_cross_third_moment(STATE::IDX::X, STATE::IDX::Y, 2, 1) + dist.calc_moment(STATE::IDX::Y, 3) +
              dist.calc_moment(STATE::IDX::Y, 1) * observation_noise.wrPow1
              - predicted_mean(STATE::IDX::Y) * observation_mean(OBSERVATION::IDX::R); // yp * (xp^2 + yp^2)
    state_observation_cov(STATE::IDX::Y, OBSERVATION::IDX::VC)
            = dist.calc_xy_cos_z_moment(STATE::IDX::Y, STATE::IDX::V, STATE::IDX::YAW)
              + dist.calc_moment(STATE::IDX::Y, 1) * observation_noise.wvPow1
              - predicted_mean(STATE::IDX::Y) * observation_mean(OBSERVATION::IDX::VC);
    state_observation_cov(STATE::IDX::Y, OBSERVATION::IDX::YAW)
            = dist.calc_cross_second_moment(STATE::IDX::Y, STATE::IDX::YAW)
              + dist.calc_moment(STATE::IDX::Y, 1) * observation_noise.wyawPow1
              - predicted_mean(STATE::IDX::Y) * observation_mean(OBSERVATION::IDX::YAW); // y_p * yaw

    // v_k (x_k^2 + y_k^2 + w_r)
    state_observation_cov(STATE::IDX::V, OBSERVATION::IDX::R)
            = dist.calc_cross_third_moment(STATE::IDX::X, STATE::IDX::V, 2, 1)
              + dist.calc_cross_third_moment(STATE::IDX::Y, STATE::IDX::V, 2, 1)
              + observation_noise.wrPow1 * dist.calc_moment(STATE::IDX::V, 1)
              - predicted_mean(STATE::IDX::V) * observation_mean(OBSERVATION::IDX::R);
    // v_k (v_k cos(theta) + w_v)
    state_observation_cov(STATE::IDX::V, OBSERVATION::IDX::VC) =
              dist.calc_xx_cos_z_moment(STATE::IDX::V, STATE::IDX::YAW)
              + dist.calc_moment(STATE::IDX::V, 1) * observation_noise.wvPow1
              - predicted_mean(STATE::IDX::V) * observation_mean(OBSERVATION::IDX::VC);
    state_observation_cov(STATE::IDX::V, OBSERVATION::IDX::YAW) =
              dist.calc_cross_second_moment(STATE::IDX::V, STATE::IDX::YAW)
              + observation_noise.wyawPow1 * dist.calc_moment(STATE::IDX::V, 1)
              - predicted_mean(STATE::IDX::V) * observation_mean(OBSERVATION::IDX::YAW);

    state_observation_cov(STATE::IDX::YAW, OBSERVATION::IDX::R)
            = dist.calc_cross_third_moment(STATE::IDX::X, STATE::IDX::YAW, 2, 1) +
              dist.calc_cross_third_moment(STATE::IDX::Y, STATE::IDX::YAW, 2, 1) +
              dist.calc_moment(STATE::IDX::YAW, 1) * observation_noise.wrPow1
              - predicted_mean(STATE::IDX::YAW) * observation_mean(OBSERVATION::IDX::R); // yaw_p * (x_p^2 + y_p^2 + w_r)
    state_observation_cov(STATE::IDX::YAW, OBSERVATION::IDX::VC)
            = dist.calc_xy_cos_y_moment(STATE::IDX::V, STATE::IDX::YAW)
              + observation_noise.wvPow1 * dist.calc_moment(STATE::IDX::YAW, 1)
              - predicted_mean(STATE::IDX::YAW) * observation_mean(OBSERVATION::IDX::VC);
    state_observation_cov(STATE::IDX::YAW, OBSERVATION::IDX::YAW)
            = dist.calc_moment(STATE::IDX::YAW, 2) + dist.calc_moment(STATE::IDX::YAW, 1) * observation_noise.wyawPow1
              - predicted_mean(STATE::IDX::YAW) * observation_mean(OBSERVATION::IDX::YAW); // yaw_p * (yaw_p + w_yaw)

    // Kalman Gain
    const auto K = state_observation_cov * observation_cov.inverse();

    StateInfo updated_info;
    updated_info.mean = predicted_mean + K * (observed_values - observation_mean);
    updated_info.covariance = predicted_cov - K * observation_cov * K.transpose();

    return updated_info;
}