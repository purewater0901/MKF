#include "filter/normal_vehicle_nkf.h"
#include "distribution/two_dimensional_normal_distribution.h"
#include "distribution/three_dimensional_normal_distribution.h"

using namespace NormalVehicle;

NormalVehicleNKF::NormalVehicleNKF()
{
    vehicle_model_ = NormalVehicleModel();
}

StateInfo NormalVehicleNKF::predict(const StateInfo & state_info,
                                    const Eigen::Vector2d & control_inputs,
                                    const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    // Step1. Approximate to Gaussian Distribution
    const auto state_mean = state_info.mean;
    const auto state_cov = state_info.covariance;
    ThreeDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);

    // Step2. State Moment
    NormalVehicleModel::StateMoments moment;
    moment.xPow1 = dist.calc_moment(STATE::IDX::X, 1); // x
    moment.yPow1 = dist.calc_moment(STATE::IDX::Y, 1); // y
    moment.cPow1 = dist.calc_cos_moment(STATE::IDX::YAW, 1); // cos(yaw)
    moment.sPow1 = dist.calc_sin_moment(STATE::IDX::YAW, 1); // sin(yaw)
    moment.yawPow1 = dist.calc_moment(STATE::IDX::YAW, 1); // yaw
    moment.xPow2 = dist.calc_moment(STATE::IDX::X, 2); // x^2
    moment.yPow2 = dist.calc_moment(STATE::IDX::Y, 2); // y^2
    moment.cPow2 = dist.calc_cos_moment(STATE::IDX::YAW, 2); // cos(yaw)^2
    moment.sPow2 = dist.calc_sin_moment(STATE::IDX::YAW, 2); // sin(yaw)^2
    moment.yawPow2 = dist.calc_moment(STATE::IDX::YAW, 2); // yaw^2
    moment.xPow1_yPow1 = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::Y); // xy
    moment.cPow1_xPow1 = dist.calc_x_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW); // x*cos(yaw)
    moment.sPow1_xPow1 = dist.calc_x_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW); // x*sin(yaw)
    moment.cPow1_yPow1 = dist.calc_x_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW); // y*cos(yaw)
    moment.sPow1_yPow1 = dist.calc_x_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW); // y*sin(yaw)
    moment.cPow1_sPow1 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 1, 1); // cos(yaw)*sin(yaw)
    moment.xPow1_yawPow1 = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::YAW); // x*yaw
    moment.yPow1_yawPow1 = dist.calc_cross_second_moment(STATE::IDX::Y, STATE::IDX::YAW); // y*yaw
    moment.cPow1_yawPow1 = dist.calc_x_cos_x_moment(STATE::IDX::YAW, 1, 1); // yaw*cos(yaw)
    moment.sPow1_yawPow1 = dist.calc_x_sin_x_moment(STATE::IDX::YAW, 1, 1); // yaw*sin(yaw)

    // Step3. Control Input
    NormalVehicleModel::Controls controls;
    controls.v = control_inputs(INPUT::IDX::V);
    controls.u = control_inputs(INPUT::IDX::U);
    controls.cu = std::cos(controls.u);
    controls.su = std::sin(controls.u);

    // Step4. System Noise
    const auto wx_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WX);
    const auto wy_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WY);
    const auto wyaw_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WYAW);
    NormalVehicleModel::SystemNoiseMoments system_noise_moments;
    system_noise_moments.wxPow1 = wx_dist_ptr->calc_moment(1);
    system_noise_moments.wyPow1 = wy_dist_ptr->calc_moment(1);
    system_noise_moments.wyawPow1 = wyaw_dist_ptr->calc_moment(1);
    system_noise_moments.wxPow2 = wx_dist_ptr->calc_moment(2);
    system_noise_moments.wyPow2 = wy_dist_ptr->calc_moment(2);
    system_noise_moments.wyawPow2 = wyaw_dist_ptr->calc_moment(2);
    system_noise_moments.cyawPow1 = wyaw_dist_ptr->calc_cos_moment(1);
    system_noise_moments.syawPow1 = wyaw_dist_ptr->calc_sin_moment(1);
    system_noise_moments.syawPow2 = wyaw_dist_ptr->calc_sin_moment(2);
    system_noise_moments.cyawPow2 = wyaw_dist_ptr->calc_cos_moment(2);
    system_noise_moments.cyawPow1_syawPow1 = wyaw_dist_ptr->calc_cos_sin_moment(1, 1);

    // Step5. Propagate
    const auto predicted_moment = vehicle_model_.propagateStateMoments(moment, system_noise_moments, controls);

    StateInfo predicted_info;
    predicted_info.mean(STATE::IDX::X) = predicted_moment.xPow1;
    predicted_info.mean(STATE::IDX::Y) = predicted_moment.yPow1;
    predicted_info.mean(STATE::IDX::YAW)= predicted_moment.yawPow1;
    predicted_info.covariance(STATE::IDX::X, STATE::IDX::X) = predicted_moment.xPow2 - predicted_moment.xPow1 * predicted_moment.xPow1;
    predicted_info.covariance(STATE::IDX::Y, STATE::IDX::Y) = predicted_moment.yPow2 - predicted_moment.yPow1 * predicted_moment.yPow1;
    predicted_info.covariance(STATE::IDX::YAW, STATE::IDX::YAW) = predicted_moment.yawPow2 - predicted_moment.yawPow1 * predicted_moment.yawPow1;
    predicted_info.covariance(STATE::IDX::X, STATE::IDX::Y) = predicted_moment.xPow1_yPow1 - predicted_moment.xPow1 * predicted_moment.yPow1;
    predicted_info.covariance(STATE::IDX::X, STATE::IDX::YAW) = predicted_moment.xPow1_yawPow1 - predicted_moment.xPow1 * predicted_moment.yawPow1;
    predicted_info.covariance(STATE::IDX::Y, STATE::IDX::YAW) = predicted_moment.yPow1_yawPow1 - predicted_moment.yPow1 * predicted_moment.yawPow1;
    predicted_info.covariance(STATE::IDX::Y, STATE::IDX::X) = predicted_info.covariance(STATE::IDX::X, STATE::IDX::Y);
    predicted_info.covariance(STATE::IDX::YAW, STATE::IDX::X) = predicted_info.covariance(STATE::IDX::X, STATE::IDX::YAW);
    predicted_info.covariance(STATE::IDX::YAW, STATE::IDX::Y) = predicted_info.covariance(STATE::IDX::Y, STATE::IDX::YAW);

    return predicted_info;
}

StateInfo NormalVehicleNKF::update(const StateInfo & state_info,
                                   const Eigen::Vector2d & observed_values,
                                   const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    const auto predicted_mean = state_info.mean;
    const auto predicted_cov = state_info.covariance;

    ThreeDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);
    NormalVehicleModel::ReducedStateMoments reduced_moments;
    reduced_moments.yawPow1 = dist.calc_moment(STATE::IDX::YAW, 1);
    reduced_moments.xPow2 = dist.calc_moment(STATE::IDX::X, 2);
    reduced_moments.yPow2 = dist.calc_moment(STATE::IDX::Y, 2);
    reduced_moments.yawPow2 = dist.calc_moment(STATE::IDX::YAW, 2);
    reduced_moments.xPow2_yawPow1 = dist.calc_cross_third_moment(STATE::IDX::X, STATE::IDX::YAW, 2, 1);
    reduced_moments.yPow2_yawPow1 = dist.calc_cross_third_moment(STATE::IDX::Y, STATE::IDX::YAW, 2, 1);
    reduced_moments.xPow3 = dist.calc_moment(STATE::IDX::X, 3);
    reduced_moments.yPow3 = dist.calc_moment(STATE::IDX::Y, 3);
    reduced_moments.xPow4 = dist.calc_moment(STATE::IDX::X, 4);
    reduced_moments.yPow4 = dist.calc_moment(STATE::IDX::Y, 4);
    reduced_moments.xPow2_yPow2 = dist.calc_xxyy_moment(STATE::IDX::X, STATE::IDX::Y);

    // Step2. Create Observation Noise
    const auto wr_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WR);
    const auto wyaw_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WYAW);
    NormalVehicleModel::ObservationNoiseMoments observation_noise;
    observation_noise.w_rPow1 = wr_dist_ptr->calc_moment(1);
    observation_noise.w_rPow2 = wr_dist_ptr->calc_moment(2);
    observation_noise.w_yawPow1 = wyaw_dist_ptr->calc_moment(1);
    observation_noise.w_yawPow2 = wyaw_dist_ptr->calc_moment(2);

    // Step3. Get Observation Moments
    const auto observation_moments = vehicle_model_.getObservationMoments(reduced_moments, observation_noise);

    ObservedInfo observed_info;
    observed_info.mean(OBSERVATION::IDX::R) = observation_moments.rPow1;
    observed_info.mean(OBSERVATION::IDX::YAW) = observation_moments.yawPow1;
    observed_info.covariance(OBSERVATION::IDX::R, OBSERVATION::IDX::R) = observation_moments.rPow2 - observation_moments.rPow1*observation_moments.rPow1;
    observed_info.covariance(OBSERVATION::IDX::YAW, OBSERVATION::IDX::YAW) = observation_moments.yawPow2 - observation_moments.yawPow1*observation_moments.yawPow1;
    observed_info.covariance(OBSERVATION::IDX::R, OBSERVATION::IDX::YAW) = observation_moments.rPow1_yawPow1 - observation_moments.rPow1*observation_moments.yawPow1;
    observed_info.covariance(OBSERVATION::IDX::YAW, OBSERVATION::IDX::R) = observed_info.covariance(OBSERVATION::IDX::R, OBSERVATION::IDX::YAW);

    const auto observation_mean = observed_info.mean;
    const auto observation_cov = observed_info.covariance;

    Eigen::MatrixXd state_observation_cov(3, 2); // sigma = E[XY^T] - E[X]E[Y]^T
    state_observation_cov(STATE::IDX::X, OBSERVATION::IDX::R)
            = dist.calc_moment(STATE::IDX::X, 3) + dist.calc_cross_third_moment(STATE::IDX::X, STATE::IDX::Y, 1, 2) + dist.calc_moment(STATE::IDX::X, 1) * observation_noise.w_rPow1
              - predicted_mean(STATE::IDX::X) * observation_mean(OBSERVATION::IDX::R); // xp * (xp^2 + yp^2)
    state_observation_cov(STATE::IDX::X, OBSERVATION::IDX::YAW)
            = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::YAW) + dist.calc_moment(STATE::IDX::X, 1) * observation_noise.w_yawPow1
              - predicted_mean(STATE::IDX::X) * observation_mean(OBSERVATION::IDX::YAW); // x_p * yaw
    state_observation_cov(STATE::IDX::Y, OBSERVATION::IDX::R)
            = dist.calc_cross_third_moment(STATE::IDX::X, STATE::IDX::Y, 2, 1) + dist.calc_moment(STATE::IDX::Y, 3)  + dist.calc_moment(STATE::IDX::Y, 1) * observation_noise.w_rPow1
              - predicted_mean(STATE::IDX::Y) * observation_mean(OBSERVATION::IDX::R); // yp * (xp^2 + yp^2)
    state_observation_cov(STATE::IDX::Y, OBSERVATION::IDX::YAW)
            = dist.calc_cross_second_moment(STATE::IDX::Y, STATE::IDX::YAW) + dist.calc_moment(STATE::IDX::Y, 1) * observation_noise.w_yawPow1
              - predicted_mean(STATE::IDX::Y) * observation_mean(OBSERVATION::IDX::YAW); // y_p * yaw
    state_observation_cov(STATE::IDX::YAW, OBSERVATION::IDX::R)
            = dist.calc_cross_third_moment(STATE::IDX::X, STATE::IDX::YAW, 2, 1) + dist.calc_cross_third_moment(STATE::IDX::Y, STATE::IDX::YAW, 2 ,1)  + dist.calc_moment(STATE::IDX::YAW, 1) * observation_noise.w_rPow1
              - predicted_mean(STATE::IDX::YAW) * observation_mean(OBSERVATION::IDX::R); // yaw_p * (x_p^2 + y_p^2 + w_r)
    state_observation_cov(STATE::IDX::YAW, OBSERVATION::IDX::YAW)
            = dist.calc_moment(STATE::IDX::YAW, 2) + dist.calc_moment(STATE::IDX::YAW, 1) * observation_noise.w_yawPow1
              - predicted_mean(STATE::IDX::YAW) * observation_mean(OBSERVATION::IDX::YAW); // yaw_p * (yaw_p + w_yaw)

    // Kalman Gain
    const auto K = state_observation_cov * observation_cov.inverse();

    StateInfo updated_info;
    updated_info.mean = predicted_mean + K * (observed_values - observation_mean);
    updated_info.covariance = predicted_cov - K*observation_cov*K.transpose();

    return updated_info;
}
