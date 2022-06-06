#include "model/simple_vehicle_model.h"

using namespace SimpleVehicle;

Eigen::Vector3d SimpleVehicleModel::propagate(const Eigen::Vector3d& x_curr,
                                              const Eigen::Vector2d& u_curr,
                                              const Eigen::Vector2d& system_noise)
{
    Eigen::Vector3d x_next;
    x_next(STATE::IDX::X) = x_curr(STATE::IDX::X) + (u_curr(INPUT::IDX::V) + system_noise(SYSTEM_NOISE::IDX::WV))* std::cos(x_curr(STATE::IDX::YAW));
    x_next(STATE::IDX::Y) = x_curr(STATE::IDX::Y) + (u_curr(INPUT::IDX::V) + system_noise(SYSTEM_NOISE::IDX::WV)) * std::sin(x_curr(STATE::IDX::YAW));
    x_next(STATE::IDX::YAW) = x_curr(STATE::IDX::YAW) + (u_curr(INPUT::IDX::U) + system_noise(SYSTEM_NOISE::IDX::WU));

    return x_next;
}

Eigen::Vector2d SimpleVehicleModel::observe(const Eigen::Vector3d& x_curr, const Eigen::Vector2d& observation_noise, const Eigen::Vector2d& landmark)
{
    const double& x = x_curr(STATE::IDX::X);
    const double& y = x_curr(STATE::IDX::Y);
    const double& yaw = x_curr(STATE::IDX::YAW);
    const double& x_land = landmark(0);
    const double& y_land = landmark(1);
    const double& wr = observation_noise(OBSERVATION_NOISE::IDX::WR); // length noise
    const double& wa = observation_noise(OBSERVATION_NOISE::IDX::WA); // bearing noise
    const double rcos_bearing = (x_land - x) * std::cos(yaw) + (y_land - y) * std::sin(yaw);
    const double rsin_bearing = (y_land - y) * std::cos(yaw) - (x_land - x) * std::sin(yaw);

    Eigen::Vector2d y_next;
    y_next(OBSERVATION::IDX::RCOS) =  wr * std::cos(wa) * rcos_bearing - wr * std::sin(wa) * rsin_bearing;
    y_next(OBSERVATION::IDX::RSIN) =  wr * std::cos(wa) * rsin_bearing + wr * std::sin(wa) * rcos_bearing;

    return y_next;
}

SimpleVehicleModel::StateMoments SimpleVehicleModel::propagateStateMoments(const StateMoments& prev_state_moments,
                                                                           const SystemNoiseMoments & system_noise_moments,
                                                                           const Controls & control_inputs)
{
    StateMoments state_moments;
    const double &xPow1 = prev_state_moments.xPow1;
    const double &cPow1 = prev_state_moments.cPow1;
    const double &sPow1 = prev_state_moments.sPow1;
    const double &yPow1 = prev_state_moments.yPow1;
    const double &yawPow1 = prev_state_moments.yawPow1;
    const double &xPow2 = prev_state_moments.xPow2;
    const double &cPow2 = prev_state_moments.cPow2;
    const double &sPow2 = prev_state_moments.sPow2;
    const double &yawPow2 = prev_state_moments.yawPow2;
    const double &cPow1_xPow1 = prev_state_moments.cPow1_xPow1;
    const double &sPow1_xPow1 = prev_state_moments.sPow1_xPow1;
    const double &yPow2 = prev_state_moments.yPow2;
    const double &sPow1_yPow1 = prev_state_moments.sPow1_yPow1;
    const double &cPow1_yPow1 = prev_state_moments.cPow1_yPow1;
    const double &xPow1_yPow1 = prev_state_moments.xPow1_yPow1;
    const double &cPow1_sPow1 = prev_state_moments.cPow1_sPow1;
    const double &xPow1_yawPow1= prev_state_moments.xPow1_yawPow1;
    const double &yPow1_yawPow1 = prev_state_moments.yPow1_yawPow1;
    const double &cPow1_yawPow1 = prev_state_moments.cPow1_yawPow1;
    const double &sPow1_yawPow1 = prev_state_moments.sPow1_yawPow1;


    const double &wvPow1 = system_noise_moments.wvPow1;
    const double &wvPow2 = system_noise_moments.wvPow2;
    const double &wuPow1 = system_noise_moments.wuPow1;
    const double &wuPow2 = system_noise_moments.wuPow2;
    const double &cwuPow2 = system_noise_moments.cwuPow2;
    const double &swuPow1 = system_noise_moments.swuPow1;
    const double &swuPow2 = system_noise_moments.swuPow2;
    const double &cwuPow1_swuPow1 = system_noise_moments.cwuPow1_swuPow1;
    const double &cwuPow1 = system_noise_moments.cwuPow1;


    const double &v = control_inputs.v;
    const double &u = control_inputs.u;
    const double &cu = control_inputs.cu;
    const double &su = control_inputs.su;

    // Dynamics updates.
    state_moments.xPow1 = v*cPow1 + cPow1*wvPow1 + xPow1;

    state_moments.cPow1 = cu*cPow1*cwuPow1 - cu*sPow1*swuPow1 - su*cPow1*swuPow1 - su*cwuPow1*sPow1;

    state_moments.sPow1 = cu*cPow1*swuPow1 + cu*cwuPow1*sPow1 + su*cPow1*cwuPow1 - su*sPow1*swuPow1;

    state_moments.yPow1 = v*sPow1 + sPow1*wvPow1 + yPow1;

    state_moments.yawPow1 = u + yawPow1 + wuPow1;

    state_moments.xPow2 = pow(v, 2)*cPow2 + 2*v*cPow1_xPow1 + 2*v*cPow2*wvPow1 + 2*cPow1_xPow1*wvPow1 + cPow2*wvPow2 + xPow2;

    state_moments.cPow2 = pow(cu, 2)*cPow2*cwuPow2 + pow(cu, 2)*sPow2*swuPow2 - 2*cu*su*cPow1_sPow1*cwuPow2 + 2*cu*su*cPow1_sPow1*swuPow2 - 2*cu*su*cPow2*cwuPow1_swuPow1 + 2*cu*su*cwuPow1_swuPow1*sPow2 + pow(su, 2)*cPow2*swuPow2 + pow(su, 2)*cwuPow2*sPow2 + cPow1_sPow1*cwuPow1_swuPow1*(-2*pow(cu, 2) + 2*pow(su, 2));

    state_moments.sPow2 = pow(cu, 2)*cPow2*swuPow2 + pow(cu, 2)*cwuPow2*sPow2 + 2*cu*su*cPow1_sPow1*cwuPow2 - 2*cu*su*cPow1_sPow1*swuPow2 + 2*cu*su*cPow2*cwuPow1_swuPow1 - 2*cu*su*cwuPow1_swuPow1*sPow2 + pow(su, 2)*cPow2*cwuPow2 + pow(su, 2)*sPow2*swuPow2 + cPow1_sPow1*cwuPow1_swuPow1*(2*pow(cu, 2) - 2*pow(su, 2));

    state_moments.cPow1_xPow1 = -cu*v*cPow1_sPow1*swuPow1 + cu*v*cPow2*cwuPow1 - cu*cPow1_sPow1*swuPow1*wvPow1 + cu*cPow1_xPow1*cwuPow1 + cu*cPow2*cwuPow1*wvPow1 - cu*sPow1_xPow1*swuPow1 - su*v*cPow1_sPow1*cwuPow1 - su*v*cPow2*swuPow1 - su*cPow1_sPow1*cwuPow1*wvPow1 - su*cPow1_xPow1*swuPow1 - su*cPow2*swuPow1*wvPow1 - su*cwuPow1*sPow1_xPow1;

    state_moments.sPow1_xPow1 = cu*v*cPow1_sPow1*cwuPow1 + cu*v*cPow2*swuPow1 + cu*cPow1_sPow1*cwuPow1*wvPow1 + cu*cPow1_xPow1*swuPow1 + cu*cPow2*swuPow1*wvPow1 + cu*cwuPow1*sPow1_xPow1 - su*v*cPow1_sPow1*swuPow1 + su*v*cPow2*cwuPow1 - su*cPow1_sPow1*swuPow1*wvPow1 + su*cPow1_xPow1*cwuPow1 + su*cPow2*cwuPow1*wvPow1 - su*sPow1_xPow1*swuPow1;

    state_moments.yPow2 = pow(v, 2)*sPow2 + 2*v*sPow1_yPow1 + 2*v*sPow2*wvPow1 + 2*sPow1_yPow1*wvPow1 + sPow2*wvPow2 + yPow2;

    state_moments.sPow1_yPow1 = cu*v*cPow1_sPow1*swuPow1 + cu*v*cwuPow1*sPow2 + cu*cPow1_sPow1*swuPow1*wvPow1 + cu*cPow1_yPow1*swuPow1 + cu*cwuPow1*sPow1_yPow1 + cu*cwuPow1*sPow2*wvPow1 + su*v*cPow1_sPow1*cwuPow1 - su*v*sPow2*swuPow1 + su*cPow1_sPow1*cwuPow1*wvPow1 + su*cPow1_yPow1*cwuPow1 - su*sPow1_yPow1*swuPow1 - su*sPow2*swuPow1*wvPow1;

    state_moments.cPow1_yPow1 = cu*v*cPow1_sPow1*cwuPow1 - cu*v*sPow2*swuPow1 + cu*cPow1_sPow1*cwuPow1*wvPow1 + cu*cPow1_yPow1*cwuPow1 - cu*sPow1_yPow1*swuPow1 - cu*sPow2*swuPow1*wvPow1 - su*v*cPow1_sPow1*swuPow1 - su*v*cwuPow1*sPow2 - su*cPow1_sPow1*swuPow1*wvPow1 - su*cPow1_yPow1*swuPow1 - su*cwuPow1*sPow1_yPow1 - su*cwuPow1*sPow2*wvPow1;

    state_moments.yawPow2 = pow(u, 2) + 2*u*yawPow1 + 2*u*wuPow1 + 2*yawPow1*wuPow1 + yawPow2 + wuPow2;

    state_moments.xPow1_yPow1 = pow(v, 2)*cPow1_sPow1 + 2*v*cPow1_sPow1*wvPow1 + v*cPow1_yPow1 + v*sPow1_xPow1 + cPow1_sPow1*wvPow2 + cPow1_yPow1*wvPow1 + sPow1_xPow1*wvPow1 + xPow1_yPow1;

    state_moments.cPow1_sPow1 = -4*cu*su*cPow1_sPow1*cwuPow1_swuPow1 + cu*su*cPow2*cwuPow2 - cu*su*cPow2*swuPow2 - cu*su*cwuPow2*sPow2 + cu*su*sPow2*swuPow2 + cPow1_sPow1*cwuPow2*(pow(cu, 2) - pow(su, 2)) + cPow1_sPow1*swuPow2*(-pow(cu, 2) + pow(su, 2)) + cPow2*cwuPow1_swuPow1*(pow(cu, 2) - pow(su, 2)) + cwuPow1_swuPow1*sPow2*(-pow(cu, 2) + pow(su, 2));

    state_moments.xPow1_yawPow1 = u*v*cPow1 + u*cPow1*wvPow1 + u*xPow1 + v*cPow1*wuPow1 + v*cPow1_yawPow1 + cPow1*wuPow1*wvPow1 + cPow1_yawPow1*wvPow1 + xPow1_yawPow1 + wuPow1*xPow1;

    state_moments.yPow1_yawPow1 = u*v*sPow1 + u*sPow1*wvPow1 + u*yPow1 + v*sPow1*wuPow1 + v*sPow1_yawPow1 + sPow1*wuPow1*wvPow1 + sPow1_yawPow1*wvPow1 + yPow1_yawPow1 + wuPow1*yPow1;

    state_moments.cPow1_yawPow1 = cu*u*cPow1*cwuPow1 - cu*u*sPow1*swuPow1 + cu*cPow1*cwuPow1*wuPow1 + cu*cPow1_yawPow1*cwuPow1 - cu*sPow1*swuPow1*wuPow1 - cu*sPow1_yawPow1*swuPow1 - su*u*cPow1*swuPow1 - su*u*cwuPow1*sPow1 - su*cPow1*swuPow1*wuPow1 - su*cPow1_yawPow1*swuPow1 - su*cwuPow1*sPow1*wuPow1 - su*cwuPow1*sPow1_yawPow1;

    state_moments.sPow1_yawPow1 = cu*u*cPow1*swuPow1 + cu*u*cwuPow1*sPow1 + cu*cPow1*swuPow1*wuPow1 + cu*cPow1_yawPow1*swuPow1 + cu*cwuPow1*sPow1*wuPow1 + cu*cwuPow1*sPow1_yawPow1 + su*u*cPow1*cwuPow1 - su*u*sPow1*swuPow1 + su*cPow1*cwuPow1*wuPow1 + su*cPow1_yawPow1*cwuPow1 - su*sPow1*swuPow1*wuPow1 - su*sPow1_yawPow1*swuPow1;

    return state_moments;
}

SimpleVehicleModel::ObservationMoments SimpleVehicleModel::getObservationMoments(const ReducedStateMoments & state_moments,
                                                                                 const ObservationNoiseMoments & observation_noise_moments,
                                                                                 const Eigen::Vector2d& landmark)
{
    // State Moments
    const double &cPow1 = state_moments.cPow1;
    const double &sPow1 = state_moments.sPow1;

    const double &cPow2 = state_moments.cPow2;
    const double &sPow2 = state_moments.sPow2;
    const double &xPow1_cPow1 = state_moments.xPow1_cPow1;
    const double &yPow1_cPow1 = state_moments.yPow1_cPow1;
    const double &xPow1_sPow1 = state_moments.xPow1_sPow1;
    const double &yPow1_sPow1 = state_moments.yPow1_sPow1;
    const double &cPow1_sPow1 = state_moments.cPow1_sPow1;

    const double &xPow1_cPow2 = state_moments.xPow1_cPow2;
    const double &yPow1_cPow2 = state_moments.yPow1_cPow2;
    const double &xPow1_sPow2 = state_moments.xPow1_sPow2;
    const double &yPow1_sPow2 = state_moments.yPow1_sPow2;
    const double &xPow1_cPow1_sPow1 = state_moments.xPow1_cPow1_sPow1;
    const double &yPow1_cPow1_sPow1 = state_moments.yPow1_cPow1_sPow1;

    const double &xPow2_cPow2 = state_moments.xPow2_cPow2;
    const double &yPow2_cPow2 = state_moments.yPow2_cPow2;
    const double &xPow2_sPow2 = state_moments.xPow2_sPow2;
    const double &yPow2_sPow2 = state_moments.yPow2_sPow2;
    const double &xPow1_yPow1_cPow2 = state_moments.xPow1_yPow1_cPow2;
    const double &xPow1_yPow1_sPow2 = state_moments.xPow1_yPow1_sPow2;
    const double &xPow2_cPow1_sPow1 = state_moments.xPow2_cPow1_sPow1;
    const double &yPow2_cPow1_sPow1 = state_moments.yPow2_cPow1_sPow1;
    const double &xPow1_yPow1_cPow1_sPow1 = state_moments.xPow1_yPow1_cPow1_sPow1;

    // Observation Noise
    const double &wrPow1 = observation_noise_moments.wrPow1;
    const double &cwaPow1 = observation_noise_moments.cwaPow1;
    const double &swaPow1 = observation_noise_moments.swaPow1;
    const double &wrPow2 = observation_noise_moments.wrPow2;
    const double &cwaPow2 = observation_noise_moments.cwaPow2;
    const double &swaPow2 = observation_noise_moments.swaPow2;
    const double &cwaPow1_swaPow1 = observation_noise_moments.cwaPow1_swaPow1;

    // Landmark
    const double &x_land = landmark(0);
    const double &y_land = landmark(1);

    const double caPow1 = x_land * cPow1 - xPow1_cPow1 + y_land * sPow1 - yPow1_sPow1;
    const double saPow1 = y_land * cPow1 - yPow1_cPow1 - x_land * sPow1 + xPow1_sPow1;
    const double caPow2 = std::pow(x_land, 2) * cPow2 + xPow2_cPow2 - 2.0 * x_land * xPow1_cPow2
                          + std::pow(y_land, 2) * sPow2 + yPow2_sPow2 - 2.0 * y_land * yPow1_sPow2
                          + 2.0 * x_land * y_land * cPow1_sPow1 - 2.0 * x_land * yPow1_cPow1_sPow1
                          - 2.0 * y_land * xPow1_cPow1_sPow1 + 2.0 * xPow1_yPow1_cPow1_sPow1;
    const double saPow2 = std::pow(y_land, 2) * cPow2 + yPow2_cPow2 - 2.0 * y_land * yPow1_cPow2
                          + std::pow(x_land, 2) * sPow2 + xPow2_sPow2 - 2.0 * x_land * xPow1_sPow2
                          - 2.0 * x_land * y_land * cPow1_sPow1 + 2.0 * x_land * yPow1_cPow1_sPow1
                          + 2.0 * y_land * xPow1_cPow1_sPow1 - 2.0 * xPow1_yPow1_cPow1_sPow1;
    const double caPow1_saPow1 =  x_land * y_land * cPow2 + xPow1_yPow1_cPow2 - x_land * yPow1_cPow2 - y_land * xPow1_cPow2
                                  - std::pow(x_land, 2) * cPow1_sPow1 - xPow2_cPow1_sPow1 + 2.0 * x_land * xPow1_cPow1_sPow1
                                  + std::pow(y_land, 2) * cPow1_sPow1 + yPow2_cPow1_sPow1 - 2.0 * y_land * yPow1_cPow1_sPow1
                                  - x_land * y_land * sPow2 - xPow1_yPow1_sPow2 + x_land * yPow1_sPow2 + y_land * xPow1_sPow2;

    ObservationMoments observation_moments;
    observation_moments.rcosPow1 = wrPow1 * cwaPow1 * caPow1 - wrPow1 * swaPow1 * saPow1;
    observation_moments.rsinPow1 = wrPow1 * cwaPow1 * saPow1 + wrPow1 * swaPow1 * caPow1;

    observation_moments.rcosPow2 = wrPow2 * cwaPow2 * caPow2 + wrPow2 * swaPow2 * saPow2 - 2.0 * wrPow2 * cwaPow1_swaPow1 * caPow1_saPow1;
    observation_moments.rsinPow2 = wrPow2 * cwaPow2 * saPow2 + wrPow2 * swaPow2 * caPow2 + 2.0 * wrPow2 * cwaPow1_swaPow1 * caPow1_saPow1;
    observation_moments.rcosPow1_rsinPow1 = wrPow2 * (cwaPow2 - swaPow2) * caPow1_saPow1 - wrPow2 * cwaPow1_swaPow1 * (caPow2 - saPow2);

    return observation_moments;
}
