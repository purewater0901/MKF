#include "filter/mobile_robot_nkf.h"

#include "distribution/four_dimensional_normal_distribution.h"

using namespace MobileRobot;

MobileRobot::StateInfo MobileRobotNKF::predict(const MobileRobot::StateInfo & state_info,
                                               const Eigen::Vector2d & control_inputs,
                                               const double dt,
                                               const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    // Step1. Approximate to Gaussian Distribution
    const auto state_mean = state_info.mean;
    const auto state_cov = state_info.covariance;
    FourDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);

    // Step2. State Moment
    MobileRobotModel::StateMoments moment;
    moment.xPow1 = dist.calc_moment(STATE::IDX::X, 1);
    moment.yPow1 = dist.calc_moment(STATE::IDX::Y, 1);
    moment.vPow1 = dist.calc_moment(STATE::IDX::V, 1);
    moment.yawPow1 = dist.calc_moment(STATE::IDX::YAW, 1);

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
    MobileRobotModel::Controls controls;
    controls.a = control_inputs(INPUT::IDX::A);
    controls.u = control_inputs(INPUT::IDX::U);
    controls.cu = std::cos(controls.u);
    controls.su = std::sin(controls.u);

    // Step4. System Noise
    const auto wv_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WV);
    const auto wyaw_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WYAW);
    MobileRobotModel::SystemNoiseMoments system_noise_moments;
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

StateInfo MobileRobotNKF::update(const MobileRobot::StateInfo &state_info,
                                 const Eigen::Vector3d &observed_values,
                                 const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map) {
    const auto &predicted_mean = state_info.mean;
    const auto &predicted_cov = state_info.covariance;

    FourDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);
    MobileRobotModel::measurementInputStateMoments input_moments;
    input_moments.xPow1 = dist.calc_moment(STATE::IDX::X, 1);
    input_moments.yPow1 = dist.calc_moment(STATE::IDX::Y, 1);
    input_moments.vPow1 = dist.calc_moment(STATE::IDX::V, 1);
    input_moments.cyawPow1 = dist.calc_cos_moment(STATE::IDX::YAW, 1);
    input_moments.syawPow1 = dist.calc_sin_moment(STATE::IDX::YAW, 1);

    input_moments.xPow2 = dist.calc_moment(STATE::IDX::X, 2);
    input_moments.yPow2 = dist.calc_moment(STATE::IDX::Y, 2);
    input_moments.vPow2 = dist.calc_moment(STATE::IDX::V, 2);
    input_moments.cyawPow2 = dist.calc_cos_moment(STATE::IDX::YAW, 2);
    input_moments.syawPow2 = dist.calc_sin_moment(STATE::IDX::YAW, 2);
    input_moments.xPow1_cyawPow1 = dist.calc_x_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    input_moments.xPow1_syawPow1 = dist.calc_x_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    input_moments.yPow1_cyawPow1 = dist.calc_x_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    input_moments.yPow1_syawPow1 = dist.calc_x_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    input_moments.vPow1_cyawPow1 = dist.calc_x_cos_z_moment(STATE::IDX::V, STATE::IDX::YAW);
    input_moments.vPow1_syawPow1 = dist.calc_x_sin_z_moment(STATE::IDX::V, STATE::IDX::YAW);
    input_moments.xPow1_yPow1 = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::Y);
    input_moments.xPow1_vPow1 = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::V);
    input_moments.yPow1_vPow1 = dist.calc_cross_second_moment(STATE::IDX::Y, STATE::IDX::V);
    input_moments.cyawPow1_syawPow1 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 1, 1);

    input_moments.vPow2_cyawPow1 = dist.calc_xx_cos_z_moment(STATE::IDX::V, STATE::IDX::YAW);
    input_moments.vPow2_syawPow1 = dist.calc_xx_sin_z_moment(STATE::IDX::V, STATE::IDX::YAW);
    input_moments.vPow1_cyawPow2 = dist.calc_x_cos_y_cos_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    input_moments.vPow1_syawPow2 = dist.calc_x_sin_y_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    input_moments.xPow1_vPow1_syawPow1 = dist.calc_xy_sin_z_moment(STATE::IDX::X, STATE::IDX::V, STATE::IDX::YAW);
    input_moments.xPow1_vPow1_cyawPow1 = dist.calc_xy_cos_z_moment(STATE::IDX::X, STATE::IDX::V, STATE::IDX::YAW);
    input_moments.yPow1_vPow1_syawPow1 = dist.calc_xy_sin_z_moment(STATE::IDX::Y, STATE::IDX::V, STATE::IDX::YAW);
    input_moments.yPow1_vPow1_cyawPow1 = dist.calc_xy_cos_z_moment(STATE::IDX::Y, STATE::IDX::V, STATE::IDX::YAW);
    input_moments.vPow1_cyawPow1_syawPow1 = dist.calc_x_cos_y_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW);

    input_moments.vPow2_cyawPow2 = dist.calc_xx_cos_y_cos_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    input_moments.vPow2_syawPow2 = dist.calc_xx_sin_y_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    input_moments.vPow2_cyawPow1_syawPow1 = dist.calc_xx_cos_y_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW);

    // Step2. Create Observation Noise
    const auto wx_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WX);
    const auto wy_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WY);
    const auto wv_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WVC);
    const auto wyaw_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WYAW);
    MobileRobotModel::ObservationNoiseMoments observation_noise;
    observation_noise.wxPow1 = wx_dist_ptr->calc_moment(1);
    observation_noise.wyPow1 = wy_dist_ptr->calc_moment(1);
    observation_noise.wvPow1 = wv_dist_ptr->calc_moment(1);
    observation_noise.cwyawPow1 = wyaw_dist_ptr->calc_cos_moment(1);
    observation_noise.swyawPow1 = wyaw_dist_ptr->calc_sin_moment(1);
    observation_noise.wxPow2 = wx_dist_ptr->calc_moment(2);
    observation_noise.wyPow2 = wy_dist_ptr->calc_moment(2);
    observation_noise.wvPow2 = wv_dist_ptr->calc_moment(2);
    observation_noise.cwyawPow2 = wyaw_dist_ptr->calc_cos_moment(2);
    observation_noise.swyawPow2 = wyaw_dist_ptr->calc_sin_moment(2);
    observation_noise.cwyawPow1_swyawPow1 = wyaw_dist_ptr->calc_cos_sin_moment(1, 1);


    // Step3. Get Observation Moments
    const auto observation_moments = vehicle_model_.getObservationMoments(input_moments, observation_noise);

    ObservedInfo observed_info;
    observed_info.mean(OBSERVATION::IDX::X) = observation_moments.xPow1;
    observed_info.mean(OBSERVATION::IDX::Y) = observation_moments.yPow1;
    observed_info.mean(OBSERVATION::IDX::VC) = observation_moments.vcPow1;

    observed_info.covariance(OBSERVATION::IDX::X, OBSERVATION::IDX::X) =
            observation_moments.xPow2 - observation_moments.xPow1 * observation_moments.xPow1;
    observed_info.covariance(OBSERVATION::IDX::Y, OBSERVATION::IDX::Y) =
            observation_moments.yPow2 - observation_moments.yPow1 * observation_moments.yPow1;
    observed_info.covariance(OBSERVATION::IDX::VC, OBSERVATION::IDX::VC) =
            observation_moments.vcPow2 - observation_moments.vcPow1 * observation_moments.vcPow1;

    observed_info.covariance(OBSERVATION::IDX::X, OBSERVATION::IDX::Y) =
            observation_moments.xPow1_yPow1 - observation_moments.xPow1 * observation_moments.yPow1;
    observed_info.covariance(OBSERVATION::IDX::X, OBSERVATION::IDX::VC) =
            observation_moments.xPow1_vcPow1 - observation_moments.xPow1 * observation_moments.vcPow1;
    observed_info.covariance(OBSERVATION::IDX::Y, OBSERVATION::IDX::VC) =
            observation_moments.yPow1_vcPow1 - observation_moments.yPow1 * observation_moments.vcPow1;

    observed_info.covariance(OBSERVATION::IDX::Y, OBSERVATION::IDX::X) = observed_info.covariance(OBSERVATION::IDX::X, OBSERVATION::IDX::Y);
    observed_info.covariance(OBSERVATION::IDX::VC, OBSERVATION::IDX::X) = observed_info.covariance(OBSERVATION::IDX::X, OBSERVATION::IDX::VC);
    observed_info.covariance(OBSERVATION::IDX::VC, OBSERVATION::IDX::Y) = observed_info.covariance(OBSERVATION::IDX::Y, OBSERVATION::IDX::VC);

    const auto observation_mean = observed_info.mean;
    const auto observation_cov = observed_info.covariance;

    Eigen::MatrixXd state_observation_cov(4, 3); // sigma = E[XY^T] - E[X]E[Y]^T

    state_observation_cov(STATE::IDX::X, OBSERVATION::IDX::X)
            = dist.calc_moment(STATE::IDX::X, 2) + dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::V) * observation_noise.wxPow1 +
              - predicted_mean(STATE::IDX::X) * observation_mean(OBSERVATION::IDX::X); // xp * (xp + vp * wx)
    state_observation_cov(STATE::IDX::X, OBSERVATION::IDX::Y)
            = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::Y) + dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::V) * observation_noise.wyPow1 +
              - predicted_mean(STATE::IDX::X) * observation_mean(OBSERVATION::IDX::Y); // xp * (yp + vp * wy)
    state_observation_cov(STATE::IDX::X, OBSERVATION::IDX::VC)
            = -dist.calc_xy_sin_z_moment(STATE::IDX::X, STATE::IDX::V, STATE::IDX::YAW) * observation_noise.swyawPow1
              +dist.calc_xy_cos_z_moment(STATE::IDX::X, STATE::IDX::V, STATE::IDX::YAW) * observation_noise.cwyawPow1
              -dist.calc_x_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW) * observation_noise.wvPow1 * observation_noise.swyawPow1
              +dist.calc_x_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW) * observation_noise.wvPow1 * observation_noise.cwyawPow1
              - predicted_mean(STATE::IDX::X) * observation_mean(OBSERVATION::IDX::VC); // xp * ((vp + wv)*cos(theta + wtheta))

    state_observation_cov(STATE::IDX::Y, OBSERVATION::IDX::X)
            = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::Y) + dist.calc_cross_second_moment(STATE::IDX::Y, STATE::IDX::V) * observation_noise.wxPow1 +
              - predicted_mean(STATE::IDX::Y) * observation_mean(OBSERVATION::IDX::X); // yp * (xp + vp * wx)
    state_observation_cov(STATE::IDX::Y, OBSERVATION::IDX::Y)
            = dist.calc_moment(STATE::IDX::Y, 2) + dist.calc_cross_second_moment(STATE::IDX::Y, STATE::IDX::V) * observation_noise.wyPow1 +
              - predicted_mean(STATE::IDX::Y) * observation_mean(OBSERVATION::IDX::Y); // yp * (yp + vp * wy)
    state_observation_cov(STATE::IDX::Y, OBSERVATION::IDX::VC)
            = -dist.calc_xy_sin_z_moment(STATE::IDX::Y, STATE::IDX::V, STATE::IDX::YAW) * observation_noise.swyawPow1
              +dist.calc_xy_cos_z_moment(STATE::IDX::Y, STATE::IDX::V, STATE::IDX::YAW) * observation_noise.cwyawPow1
              -dist.calc_x_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW) * observation_noise.wvPow1 * observation_noise.swyawPow1
              +dist.calc_x_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW) * observation_noise.wvPow1 * observation_noise.cwyawPow1
              - predicted_mean(STATE::IDX::Y) * observation_mean(OBSERVATION::IDX::VC); // yp * ((vp + wv)*cos(theta + wtheta))


    state_observation_cov(STATE::IDX::V, OBSERVATION::IDX::X)
            = dist.calc_moment(STATE::IDX::V, 2) * observation_noise.wxPow1 + dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::V)
              - predicted_mean(STATE::IDX::V) * observation_mean(OBSERVATION::IDX::X);
    state_observation_cov(STATE::IDX::V, OBSERVATION::IDX::Y)
            = dist.calc_moment(STATE::IDX::V, 2) * observation_noise.wyPow1 + dist.calc_cross_second_moment(STATE::IDX::Y, STATE::IDX::V)
              - predicted_mean(STATE::IDX::V) * observation_mean(OBSERVATION::IDX::Y);
    state_observation_cov(STATE::IDX::V, OBSERVATION::IDX::VC)
            = - dist.calc_xx_sin_z_moment(STATE::IDX::V, STATE::IDX::YAW) * observation_noise.swyawPow1
            + dist.calc_xx_cos_z_moment(STATE::IDX::V, STATE::IDX::YAW) * observation_noise.cwyawPow1
            - dist.calc_x_sin_z_moment(STATE::IDX::V, STATE::IDX::YAW) * observation_noise.wvPow1 * observation_noise.swyawPow1
            + dist.calc_x_cos_z_moment(STATE::IDX::V, STATE::IDX::YAW) * observation_noise.wvPow1 * observation_noise.cwyawPow1
            - predicted_mean(STATE::IDX::V) * observation_mean(OBSERVATION::IDX::VC);

    state_observation_cov(STATE::IDX::YAW, OBSERVATION::IDX::X)
            = dist.calc_cross_second_moment(STATE::IDX::V, STATE::IDX::YAW) * observation_noise.wxPow1 +
              dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::YAW) +
              - predicted_mean(STATE::IDX::YAW) * observation_mean(OBSERVATION::IDX::X); // yaw_p * (x_p + wx*v_p)
    state_observation_cov(STATE::IDX::YAW, OBSERVATION::IDX::Y)
            = dist.calc_cross_second_moment(STATE::IDX::V, STATE::IDX::YAW) * observation_noise.wyPow1 +
              dist.calc_cross_second_moment(STATE::IDX::Y, STATE::IDX::YAW) +
              - predicted_mean(STATE::IDX::YAW) * observation_mean(OBSERVATION::IDX::Y); // yaw_p * (x_p + wx*v_p)
    state_observation_cov(STATE::IDX::YAW, OBSERVATION::IDX::VC)
              = -dist.calc_xy_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW) * observation_noise.swyawPow1
              + dist.calc_xy_cos_y_moment(STATE::IDX::V, STATE::IDX::YAW) * observation_noise.cwyawPow1
              - dist.calc_x_sin_x_moment(STATE::IDX::YAW, 1, 1) * observation_noise.wvPow1 * observation_noise.swyawPow1
              + dist.calc_x_cos_x_moment(STATE::IDX::YAW, 1, 1) * observation_noise.wvPow1 * observation_noise.cwyawPow1
              - predicted_mean(STATE::IDX::YAW) * observation_mean(OBSERVATION::IDX::VC);

    // Kalman Gain
    const auto K = state_observation_cov * observation_cov.inverse();

    StateInfo updated_info;
    updated_info.mean = predicted_mean + K * (observed_values - observation_mean);
    updated_info.covariance = predicted_cov - K * observation_cov * K.transpose();

    return updated_info;
}