#include "filter/simple_vehicle_nkf.h"
#include "distribution/two_dimensional_normal_distribution.h"
#include "distribution/three_dimensional_normal_distribution.h"

using namespace SimpleVehicle;

SimpleVehicleNKF::SimpleVehicleNKF()
{
    vehicle_model_ = SimpleVehicleModel();
}

StateInfo SimpleVehicleNKF::predict(const StateInfo & state_info,
                                    const Eigen::Vector2d & control_inputs,
                                    const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    // Step1. Approximate to Gaussian Distribution
    const auto state_mean = state_info.mean;
    const auto state_cov = state_info.covariance;
    ThreeDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);

    // Step2. State Moment
    SimpleVehicleModel::StateMoments moment;
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
    SimpleVehicleModel::Controls controls;
    controls.v = control_inputs(INPUT::IDX::V);
    controls.u = control_inputs(INPUT::IDX::U);
    controls.cu = std::cos(controls.u);
    controls.su = std::sin(controls.u);

    // Step4. System Noise
    const auto wv_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WV);
    const auto wu_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WU);
    SimpleVehicleModel::SystemNoiseMoments system_noise_moments;
    system_noise_moments.wvPow1 = wv_dist_ptr->calc_moment(1);
    system_noise_moments.wvPow2 = wv_dist_ptr->calc_moment(2);
    system_noise_moments.wuPow1 = wu_dist_ptr->calc_moment(1);
    system_noise_moments.wuPow2 = wu_dist_ptr->calc_moment(2);
    system_noise_moments.cwuPow1 = wu_dist_ptr->calc_cos_moment(1);
    system_noise_moments.swuPow1 = wu_dist_ptr->calc_sin_moment(1);
    system_noise_moments.swuPow2 = wu_dist_ptr->calc_sin_moment(2);
    system_noise_moments.cwuPow2 = wu_dist_ptr->calc_cos_moment(2);
    system_noise_moments.cwuPow1_swuPow1 = wu_dist_ptr->calc_cos_sin_moment(1, 1);

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

StateInfo SimpleVehicleNKF::update(const StateInfo & state_info,
                                   const Eigen::Vector2d & observed_values,
                                   const Eigen::Vector2d & landmark,
                                   const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    const auto predicted_mean = state_info.mean;
    const auto predicted_cov = state_info.covariance;

    ThreeDimensionalNormalDistribution dist(predicted_mean, predicted_cov);
    SimpleVehicleModel::ReducedStateMoments reduced_moments;
    reduced_moments.cPow1= dist.calc_cos_moment(STATE::IDX::YAW, 1);
    reduced_moments.sPow1= dist.calc_sin_moment(STATE::IDX::YAW, 1);

    reduced_moments.cPow2= dist.calc_cos_moment(STATE::IDX::YAW, 2);
    reduced_moments.sPow2= dist.calc_sin_moment(STATE::IDX::YAW, 2);
    reduced_moments.xPow1_cPow1 = dist.calc_x_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow1_cPow1 = dist.calc_x_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    reduced_moments.xPow1_sPow1 = dist.calc_x_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow1_sPow1 = dist.calc_x_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    reduced_moments.cPow1_sPow1 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 1, 1);

    reduced_moments.xPow1_cPow2 = dist.calc_x_cos_z_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow1_cPow2 = dist.calc_x_cos_z_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    reduced_moments.xPow1_sPow2 = dist.calc_x_sin_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow1_sPow2 = dist.calc_x_sin_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    reduced_moments.xPow1_cPow1_sPow1 = dist.calc_x_cos_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow1_cPow1_sPow1 = dist.calc_x_cos_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);

    reduced_moments.xPow2_cPow2 = dist.calc_xx_cos_z_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow2_cPow2 = dist.calc_xx_cos_z_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    reduced_moments.xPow2_sPow2 = dist.calc_xx_sin_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow2_sPow2 = dist.calc_xx_sin_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    reduced_moments.xPow2_cPow1_sPow1 = dist.calc_xx_cos_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow2_cPow1_sPow1 = dist.calc_xx_cos_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    reduced_moments.xPow1_yPow1_cPow2 = dist.calc_xy_cos_z_cos_z_moment();
    reduced_moments.xPow1_yPow1_sPow2 = dist.calc_xy_sin_z_sin_z_moment();
    reduced_moments.xPow1_yPow1_cPow1_sPow1 = dist.calc_xy_cos_z_sin_z_moment();


    // Step2. Create Observation Noise
    const auto wr_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WR);
    const auto wa_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WA);
    SimpleVehicleModel::ObservationNoiseMoments observation_noise;
    observation_noise.wrPow1 = wr_dist_ptr->calc_moment(1);
    observation_noise.wrPow2 = wr_dist_ptr->calc_moment(2);
    observation_noise.cwaPow1 = wa_dist_ptr->calc_cos_moment(1);
    observation_noise.swaPow1 = wa_dist_ptr->calc_sin_moment(1);
    observation_noise.cwaPow2 = wa_dist_ptr->calc_cos_moment(2);
    observation_noise.swaPow2 = wa_dist_ptr->calc_sin_moment(2);
    observation_noise.cwaPow1_swaPow1 = wa_dist_ptr->calc_cos_sin_moment(1, 1);

    // Step3. Get Observation Moments
    const auto observation_moments = vehicle_model_.getObservationMoments(reduced_moments, observation_noise, landmark);

    ObservedInfo observed_info;
    observed_info.mean(OBSERVATION::IDX::RCOS) = observation_moments.rcosPow1;
    observed_info.mean(OBSERVATION::IDX::RSIN) = observation_moments.rsinPow1;
    observed_info.covariance(OBSERVATION::IDX::RCOS, OBSERVATION::IDX::RCOS) = observation_moments.rcosPow2 - observation_moments.rcosPow1*observation_moments.rcosPow1;
    observed_info.covariance(OBSERVATION::IDX::RSIN, OBSERVATION::IDX::RSIN) = observation_moments.rsinPow2 - observation_moments.rsinPow1*observation_moments.rsinPow1;
    observed_info.covariance(OBSERVATION::IDX::RCOS, OBSERVATION::IDX::RSIN) = observation_moments.rcosPow1_rsinPow1 - observation_moments.rcosPow1*observation_moments.rsinPow1;
    observed_info.covariance(OBSERVATION::IDX::RSIN, OBSERVATION::IDX::RCOS) = observed_info.covariance(OBSERVATION::IDX::RCOS, OBSERVATION::IDX::RSIN);

    const auto observation_mean = observed_info.mean;
    const auto observation_cov = observed_info.covariance;

    const double& x_land = landmark(0);
    const double& y_land = landmark(1);

    const double& wrPow1 = observation_noise.wrPow1;
    const double& cwaPow1 = observation_noise.cwaPow1;
    const double& swaPow1 = observation_noise.swaPow1;

    const double xPow1_caPow1 = x_land * dist.calc_x_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW) - dist.calc_xx_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW)
                                + y_land * dist.calc_x_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW) - dist.calc_xy_sin_z_moment();
    const double xPow1_saPow1 = y_land * dist.calc_x_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW) - dist.calc_xy_cos_z_moment()
                                - x_land * dist.calc_x_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW) + dist.calc_xx_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double yPow1_caPow1 =  x_land * dist.calc_x_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW) - dist.calc_xy_cos_z_moment()
                                 + y_land * dist.calc_x_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW) - dist.calc_xx_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double yPow1_saPow1 =  y_land * dist.calc_x_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW) - dist.calc_xx_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW)
                                  - x_land * dist.calc_x_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW) + dist.calc_xy_sin_z_moment();
    const double yawPow1_caPow1 = x_land * dist.calc_x_cos_x_moment(STATE::IDX::YAW, 1, 1) - dist.calc_xy_cos_y_moment(STATE::IDX::X, STATE::IDX::YAW)
                                  + y_land * dist.calc_x_sin_x_moment(STATE::IDX::YAW, 1, 1) - dist.calc_xy_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double yawPow1_saPow1 = y_land * dist.calc_x_cos_x_moment(STATE::IDX::YAW, 1, 1) - dist.calc_xy_cos_y_moment(STATE::IDX::Y, STATE::IDX::YAW)
                                  - x_land * dist.calc_x_sin_x_moment(STATE::IDX::YAW, 1, 1) + dist.calc_xy_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW);

    Eigen::MatrixXd state_observation_cov(3, 2); // sigma = E[XY^T] - E[X]E[Y]^T
    state_observation_cov(STATE::IDX::X, OBSERVATION::IDX::RCOS)
            = wrPow1 * cwaPow1 * xPow1_caPow1 - wrPow1 * swaPow1 * xPow1_saPow1
            - predicted_mean(STATE::IDX::X) * observation_mean(OBSERVATION::IDX::RCOS);
    state_observation_cov(STATE::IDX::X, OBSERVATION::IDX::RSIN)
            = wrPow1 * cwaPow1 * xPow1_saPow1 + wrPow1 * swaPow1 * xPow1_caPow1
            - predicted_mean(STATE::IDX::X) * observation_mean(OBSERVATION::IDX::RSIN); // x_p * yaw

    state_observation_cov(STATE::IDX::Y, OBSERVATION::IDX::RCOS)
            =  wrPow1 * cwaPow1 * yPow1_caPow1 - wrPow1 * swaPow1 * yPow1_saPow1
              - predicted_mean(STATE::IDX::Y) * observation_mean(OBSERVATION::IDX::RCOS); // yp * (xp^2 + yp^2)
    state_observation_cov(STATE::IDX::Y, OBSERVATION::IDX::RSIN)
            =  wrPow1 * cwaPow1 * yPow1_saPow1 + wrPow1 * swaPow1 * yPow1_caPow1
              - predicted_mean(STATE::IDX::Y) * observation_mean(OBSERVATION::IDX::RSIN); // y_p * yaw

    state_observation_cov(STATE::IDX::YAW, OBSERVATION::IDX::RCOS)
            =   wrPow1 * cwaPow1 * yawPow1_caPow1 - wrPow1 * swaPow1 * yawPow1_saPow1
              - predicted_mean(STATE::IDX::YAW) * observation_mean(OBSERVATION::IDX::RCOS);
    state_observation_cov(STATE::IDX::YAW, OBSERVATION::IDX::RSIN)
            =    wrPow1 * cwaPow1 * yawPow1_saPow1 + wrPow1 * swaPow1 * yawPow1_caPow1
              - predicted_mean(STATE::IDX::YAW) * observation_mean(OBSERVATION::IDX::RSIN);

    // Kalman Gain
    const auto K = state_observation_cov * observation_cov.inverse();

    StateInfo updated_info;
    updated_info.mean = predicted_mean + K * (observed_values - observation_mean);
    updated_info.covariance = predicted_cov - K*observation_cov*K.transpose();

    return updated_info;
}
