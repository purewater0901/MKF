#include "model/normal_vehicle_model.h"

using namespace NormalVehicle;

Eigen::Vector3d NormalVehicleModel::propagate(const Eigen::Vector3d& x_curr,
                                              const Eigen::Vector2d& u_curr,
                                              const Eigen::Vector3d& system_noise)
{
    Eigen::Vector3d x_next;
    x_next(STATE::IDX::X) = x_curr(STATE::IDX::X) + u_curr(INPUT::IDX::V) * std::cos(x_curr(STATE::IDX::YAW)) + system_noise(SYSTEM_NOISE::IDX::WX);
    x_next(STATE::IDX::Y) = x_curr(STATE::IDX::Y) + u_curr(INPUT::IDX::V) * std::sin(x_curr(STATE::IDX::YAW)) + system_noise(SYSTEM_NOISE::IDX::WY);
    x_next(STATE::IDX::YAW) = x_curr(STATE::IDX::YAW) + u_curr(INPUT::IDX::U) + system_noise(SYSTEM_NOISE::IDX::WYAW);

    return x_next;
}

Eigen::Vector2d NormalVehicleModel::observe(const Eigen::Vector3d& x_curr, const Eigen::Vector2d& observation_noise)
{
    Eigen::Vector2d y_next;
    y_next(OBSERVATION::IDX::R) = x_curr(STATE::IDX::X)*x_curr(STATE::IDX::X) + x_curr(STATE::IDX::Y)*x_curr(STATE::IDX::Y) + observation_noise(OBSERVATION_NOISE::IDX::WR);
    y_next(OBSERVATION::IDX::YAW) = x_curr(STATE::IDX::YAW) + observation_noise(OBSERVATION_NOISE::IDX::WYAW);

    return y_next;
}

NormalVehicleModel::StateMoments NormalVehicleModel::propagateStateMoments(const StateMoments & prev_state_moments,
                                                                           const SystemNoiseMoments& system_noise_moments,
                                                                           const Controls& control_inputs)
{
    // Aliases for the required inputs.
    const double &xPow1 = prev_state_moments.xPow1;
    const double &cPow1 = prev_state_moments.cPow1;
    const double &sPow1 = prev_state_moments.sPow1;
    const double &yPow1 = prev_state_moments.yPow1;
    const double &yawPow1 = prev_state_moments.yawPow1;
    const double &xPow2 = prev_state_moments.xPow2;
    const double &cPow2 = prev_state_moments.cPow2;
    const double &sPow2 = prev_state_moments.sPow2;
    const double &cPow1_xPow1 = prev_state_moments.cPow1_xPow1;
    const double &sPow1_xPow1 = prev_state_moments.sPow1_xPow1;
    const double &yPow2 = prev_state_moments.yPow2;
    const double &sPow1_yPow1 = prev_state_moments.sPow1_yPow1;
    const double &cPow1_yPow1 = prev_state_moments.cPow1_yPow1;
    const double &yawPow2 = prev_state_moments.yawPow2;
    const double &xPow1_yPow1 = prev_state_moments.xPow1_yPow1;
    const double &cPow1_sPow1 = prev_state_moments.cPow1_sPow1;
    const double &xPow1_yawPow1 = prev_state_moments.xPow1_yawPow1;
    const double &cPow1_yawPow1 = prev_state_moments.cPow1_yawPow1;
    const double &yPow1_yawPow1 = prev_state_moments.yPow1_yawPow1;
    const double &sPow1_yawPow1 = prev_state_moments.sPow1_yawPow1;


    const double &wxPow1 = system_noise_moments.wxPow1;
    const double &wyPow1 = system_noise_moments.wyPow1;
    const double &wyawPow1 = system_noise_moments.wyawPow1;
    const double &wxPow2 = system_noise_moments.wxPow2;
    const double &wyPow2 = system_noise_moments.wyPow2;
    const double &wyawPow2 = system_noise_moments.wyawPow2;
    const double &syawPow1 = system_noise_moments.syawPow1;
    const double &cyawPow1 = system_noise_moments.cyawPow1;
    const double &syawPow2 = system_noise_moments.syawPow2;
    const double &cyawPow2 = system_noise_moments.cyawPow2;
    const double &cyawPow1_syawPow1 = system_noise_moments.cyawPow1_syawPow1;


    const double &v = control_inputs.v;
    const double &u = control_inputs.u;
    const double &cu = control_inputs.cu;
    const double &su = control_inputs.su;

    // Dynamics updates.
    StateMoments state_moments;

    state_moments.xPow1 = v*cPow1 + wxPow1 + xPow1;

    state_moments.cPow1 = cu*cPow1*cyawPow1 - cu*sPow1*syawPow1 - su*cPow1*syawPow1 - su*cyawPow1*sPow1;

    state_moments.sPow1 = cu*cPow1*syawPow1 + cu*cyawPow1*sPow1 + su*cPow1*cyawPow1 - su*sPow1*syawPow1;

    state_moments.yPow1 = v*sPow1 + wyPow1 + yPow1;

    state_moments.yawPow1 = 1.0*u + wyawPow1 + yawPow1;

    state_moments.xPow2 = pow(v, 2)*cPow2 + 2*v*cPow1*wxPow1 + 2*v*cPow1_xPow1 + 2*wxPow1*xPow1 + wxPow2 + xPow2;

    state_moments.cPow2 = pow(cu, 2)*cPow2*cyawPow2 + pow(cu, 2)*sPow2*syawPow2 - 2*cu*su*cPow1_sPow1*cyawPow2 + 2*cu*su*cPow1_sPow1*syawPow2 - 2*cu*su*cPow2*cyawPow1_syawPow1 + 2*cu*su*cyawPow1_syawPow1*sPow2 + pow(su, 2)*cPow2*syawPow2 + pow(su, 2)*cyawPow2*sPow2 + cPow1_sPow1*cyawPow1_syawPow1*(-2*pow(cu, 2) + 2*pow(su, 2));

    state_moments.sPow2 = pow(cu, 2)*cPow2*syawPow2 + pow(cu, 2)*cyawPow2*sPow2 + 2*cu*su*cPow1_sPow1*cyawPow2 - 2*cu*su*cPow1_sPow1*syawPow2 + 2*cu*su*cPow2*cyawPow1_syawPow1 - 2*cu*su*cyawPow1_syawPow1*sPow2 + pow(su, 2)*cPow2*cyawPow2 + pow(su, 2)*sPow2*syawPow2 + cPow1_sPow1*cyawPow1_syawPow1*(2*pow(cu, 2) - 2*pow(su, 2));

    state_moments.cPow1_xPow1 = -cu*v*cPow1_sPow1*syawPow1 + cu*v*cPow2*cyawPow1 + cu*cPow1*cyawPow1*wxPow1 + cu*cPow1_xPow1*cyawPow1 - cu*sPow1*syawPow1*wxPow1 - cu*sPow1_xPow1*syawPow1 - su*v*cPow1_sPow1*cyawPow1 - su*v*cPow2*syawPow1 - su*cPow1*syawPow1*wxPow1 - su*cPow1_xPow1*syawPow1 - su*cyawPow1*sPow1*wxPow1 - su*cyawPow1*sPow1_xPow1;

    state_moments.sPow1_xPow1 = cu*v*cPow1_sPow1*cyawPow1 + cu*v*cPow2*syawPow1 + cu*cPow1*syawPow1*wxPow1 + cu*cPow1_xPow1*syawPow1 + cu*cyawPow1*sPow1*wxPow1 + cu*cyawPow1*sPow1_xPow1 - su*v*cPow1_sPow1*syawPow1 + su*v*cPow2*cyawPow1 + su*cPow1*cyawPow1*wxPow1 + su*cPow1_xPow1*cyawPow1 - su*sPow1*syawPow1*wxPow1 - su*sPow1_xPow1*syawPow1;

    state_moments.yPow2 = pow(v, 2)*sPow2 + 2*v*sPow1*wyPow1 + 2*v*sPow1_yPow1 + 2*wyPow1*yPow1 + wyPow2 + yPow2;

    state_moments.sPow1_yPow1 = cu*v*cPow1_sPow1*syawPow1 + cu*v*cyawPow1*sPow2 + cu*cPow1*syawPow1*wyPow1 + cu*cPow1_yPow1*syawPow1 + cu*cyawPow1*sPow1*wyPow1 + cu*cyawPow1*sPow1_yPow1 + su*v*cPow1_sPow1*cyawPow1 - su*v*sPow2*syawPow1 + su*cPow1*cyawPow1*wyPow1 + su*cPow1_yPow1*cyawPow1 - su*sPow1*syawPow1*wyPow1 - su*sPow1_yPow1*syawPow1;

    state_moments.cPow1_yPow1 = cu*v*cPow1_sPow1*cyawPow1 - cu*v*sPow2*syawPow1 + cu*cPow1*cyawPow1*wyPow1 + cu*cPow1_yPow1*cyawPow1 - cu*sPow1*syawPow1*wyPow1 - cu*sPow1_yPow1*syawPow1 - su*v*cPow1_sPow1*syawPow1 - su*v*cyawPow1*sPow2 - su*cPow1*syawPow1*wyPow1 - su*cPow1_yPow1*syawPow1 - su*cyawPow1*sPow1*wyPow1 - su*cyawPow1*sPow1_yPow1;

    state_moments.yawPow2 = 1.0*pow(u, 2) + 2*u*wyawPow1 + 2*u*yawPow1 + 2*wyawPow1*yawPow1 + wyawPow2 + yawPow2;

    state_moments.xPow1_yPow1 = pow(v, 2)*cPow1_sPow1 + v*cPow1*wyPow1 + v*cPow1_yPow1 + v*sPow1*wxPow1 + v*sPow1_xPow1 + wxPow1*wyPow1 + wxPow1*yPow1 + wyPow1*xPow1 + xPow1_yPow1;

    state_moments.cPow1_sPow1 = -4*cu*su*cPow1_sPow1*cyawPow1_syawPow1 + cu*su*cPow2*cyawPow2 - cu*su*cPow2*syawPow2 - cu*su*cyawPow2*sPow2 + cu*su*sPow2*syawPow2 + cPow1_sPow1*cyawPow2*(pow(cu, 2) - pow(su, 2)) + cPow1_sPow1*syawPow2*(-pow(cu, 2) + pow(su, 2)) + cPow2*cyawPow1_syawPow1*(pow(cu, 2) - pow(su, 2)) + cyawPow1_syawPow1*sPow2*(-pow(cu, 2) + pow(su, 2));

    state_moments.xPow1_yawPow1 = u*v*cPow1 + u*wxPow1 + u*xPow1 + v*cPow1*wyawPow1 + v*cPow1_yawPow1 + wxPow1*wyawPow1 + wxPow1*yawPow1 + wyawPow1*xPow1 + xPow1_yawPow1;

    state_moments.cPow1_yawPow1 = cu*u*cPow1*cyawPow1 - cu*u*sPow1*syawPow1 + cu*cPow1*cyawPow1*wyawPow1 + cu*cPow1_yawPow1*cyawPow1 - cu*sPow1*syawPow1*wyawPow1 - cu*sPow1_yawPow1*syawPow1 - su*u*cPow1*syawPow1 - su*u*cyawPow1*sPow1 - su*cPow1*syawPow1*wyawPow1 - su*cPow1_yawPow1*syawPow1 - su*cyawPow1*sPow1*wyawPow1 - su*cyawPow1*sPow1_yawPow1;

    state_moments.yPow1_yawPow1 = u*v*sPow1 + u*wyPow1 + u*yPow1 + v*sPow1*wyawPow1 + v*sPow1_yawPow1 + wyPow1*wyawPow1 + wyPow1*yawPow1 + wyawPow1*yPow1 + yPow1_yawPow1;

    state_moments.sPow1_yawPow1 = cu*u*cPow1*syawPow1 + cu*u*cyawPow1*sPow1 + cu*cPow1*syawPow1*wyawPow1 + cu*cPow1_yawPow1*syawPow1 + cu*cyawPow1*sPow1*wyawPow1 + cu*cyawPow1*sPow1_yawPow1 + su*u*cPow1*cyawPow1 - su*u*sPow1*syawPow1 + su*cPow1*cyawPow1*wyawPow1 + su*cPow1_yawPow1*cyawPow1 - su*sPow1*syawPow1*wyawPow1 - su*sPow1_yawPow1*syawPow1;


    return state_moments;
}

NormalVehicleModel::ObservationMoments NormalVehicleModel::getObservationMoments(const ReducedStateMoments & state_moments,
                                                                                 const ObservationNoiseMoments & observation_noise_moments)
{
    // State Moments
    const double &yawPow1 = state_moments.yawPow1;
    const double &xPow2 = state_moments.xPow2;
    const double &yPow2 = state_moments.yPow2;
    const double &yawPow2 = state_moments.yawPow2;
    const double &xPow3 = state_moments.xPow3;
    const double &yPow3 = state_moments.yPow3;
    const double &xPow2_yawPow1 = state_moments.xPow2_yawPow1;
    const double &yPow2_yawPow1 = state_moments.yPow2_yawPow1;
    const double &xPow4 = state_moments.xPow4;
    const double &yPow4 = state_moments.yPow4;
    const double &xPow2_yPow2 = state_moments.xPow2_yPow2;


    // Observation Noise
    const double &w_rPow1 = observation_noise_moments.w_rPow1;
    const double &w_yawPow1 = observation_noise_moments.w_yawPow1;
    const double w_rPow2 = observation_noise_moments.w_rPow2;
    const double w_yawPow2 = observation_noise_moments.w_yawPow2;

    ObservationMoments observation_moments;
    observation_moments.rPow1 = xPow2 + yPow2 + w_rPow1;
    observation_moments.rPow2 = xPow4 + yPow4 + w_rPow2 + 2.0*xPow2_yPow2 + 2.0*xPow2*w_rPow1 + 2.0*yPow2*w_rPow1;
    observation_moments.yawPow1 = yawPow1 + w_yawPow1;
    observation_moments.yawPow2 = yawPow2 + w_yawPow2 + 2.0*yawPow1*w_yawPow1;
    observation_moments.rPow1_yawPow1 = xPow2_yawPow1 + yPow2_yawPow1 + yawPow1*w_rPow1 + xPow2*w_yawPow1 + yPow2*w_yawPow1 + w_rPow1*w_yawPow1;

    return observation_moments;
}
