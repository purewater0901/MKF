#include "model/mobile_robot_model.h"

using namespace MobileRobot;

Eigen::Vector4d MobileRobotModel::propagate(const Eigen::Vector4d& x_curr,
                                            const Eigen::Vector2d& u_curr,
                                            const Eigen::Vector2d& system_noise,
                                            const double dt)
{
    Eigen::Vector4d x_next;
    x_next(STATE::IDX::X) = x_curr(STATE::IDX::X) + x_curr(STATE::IDX::V) * std::cos(x_curr(STATE::IDX::YAW)) * dt;
    x_next(STATE::IDX::Y) = x_curr(STATE::IDX::Y) + x_curr(STATE::IDX::V) * std::sin(x_curr(STATE::IDX::YAW)) * dt;
    x_next(STATE::IDX::V) = x_curr(STATE::IDX::V) + u_curr(INPUT::IDX::A) + system_noise(SYSTEM_NOISE::IDX::WV);
    x_next(STATE::IDX::YAW) = x_curr(STATE::IDX::YAW) + u_curr(INPUT::IDX::U) + system_noise(SYSTEM_NOISE::IDX::WYAW);

    return x_next;
}

Eigen::Vector3d MobileRobotModel::observe(const Eigen::Vector4d& x_curr, const Eigen::Vector4d& observation_noise)
{

    Eigen::Vector3d y_next;
    y_next(OBSERVATION::IDX::X) = x_curr(STATE::IDX::X) + x_curr(STATE::IDX::V) * observation_noise(OBSERVATION_NOISE::IDX::WX);
    y_next(OBSERVATION::IDX::Y) = x_curr(STATE::IDX::Y) + x_curr(STATE::IDX::V) * observation_noise(OBSERVATION_NOISE::IDX::WY);
    y_next(OBSERVATION::IDX::VC) = (x_curr(STATE::IDX::V) + observation_noise(OBSERVATION_NOISE::IDX::WVC))
                                   * std::cos(x_curr(STATE::IDX::YAW) + observation_noise(OBSERVATION_NOISE::IDX::WYAW));

    return y_next;
}

MobileRobotModel::StateMoments MobileRobotModel::propagateStateMoments(const StateMoments & prev_state_moments,
                                                                                 const SystemNoiseMoments& system_noise_moments,
                                                                                 const Controls& control_inputs,
                                                                                 const double dt)
{
    // Aliases for the required inputs.
    const double &xPow1 = prev_state_moments.xPow1;
    const double &yPow1 = prev_state_moments.yPow1;
    const double &vPow1 = prev_state_moments.vPow1;
    const double &yawPow1 = prev_state_moments.yawPow1;
    const double &cPow1 = prev_state_moments.cPow1;
    const double &sPow1 = prev_state_moments.sPow1;

    const double &xPow2 = prev_state_moments.xPow2;
    const double &yPow2 = prev_state_moments.yPow2;
    const double &vPow2 = prev_state_moments.vPow2;
    const double &yawPow2 = prev_state_moments.yawPow2;
    const double &cPow2 = prev_state_moments.cPow2;
    const double &sPow2 = prev_state_moments.sPow2;
    const double &xPow1_yPow1 = prev_state_moments.xPow1_yPow1;
    const double &xPow1_yawPow1 = prev_state_moments.xPow1_yawPow1;
    const double &yPow1_yawPow1 = prev_state_moments.yPow1_yawPow1;
    const double &vPow1_xPow1 = prev_state_moments.vPow1_xPow1;
    const double &vPow1_yPow1 = prev_state_moments.vPow1_yPow1;
    const double &vPow1_yawPow1 = prev_state_moments.vPow1_yawPow1;
    const double &vPow1_cPow1 = prev_state_moments.vPow1_cPow1;
    const double &vPow1_sPow1 = prev_state_moments.vPow1_sPow1;
    const double &cPow1_xPow1 = prev_state_moments.cPow1_xPow1;
    const double &sPow1_xPow1 = prev_state_moments.sPow1_xPow1;
    const double &cPow1_yPow1 = prev_state_moments.cPow1_yPow1;
    const double &sPow1_yPow1 = prev_state_moments.sPow1_yPow1;
    const double &cPow1_yawPow1 = prev_state_moments.cPow1_yawPow1;
    const double &sPow1_yawPow1 = prev_state_moments.sPow1_yawPow1;
    const double &cPow1_sPow1 = prev_state_moments.cPow1_sPow1;

    const double &vPow1_cPow2 = prev_state_moments.vPow1_cPow2;
    const double &vPow1_sPow2 = prev_state_moments.vPow1_sPow2;
    const double &vPow2_cPow1 = prev_state_moments.vPow2_cPow1;
    const double &vPow2_sPow1 = prev_state_moments.vPow2_sPow1;
    const double &vPow1_cPow1_yawPow1 = prev_state_moments.vPow1_cPow1_yawPow1;
    const double &vPow1_sPow1_yawPow1 = prev_state_moments.vPow1_sPow1_yawPow1;
    const double &vPow1_cPow1_xPow1 = prev_state_moments.vPow1_cPow1_xPow1;
    const double &vPow1_sPow1_xPow1 = prev_state_moments.vPow1_sPow1_xPow1;
    const double &vPow1_sPow1_yPow1 = prev_state_moments.vPow1_sPow1_yPow1;
    const double &vPow1_cPow1_yPow1 = prev_state_moments.vPow1_cPow1_yPow1;
    const double &vPow1_cPow1_sPow1 = prev_state_moments.vPow1_cPow1_sPow1;

    const double &vPow2_cPow2 = prev_state_moments.vPow2_cPow2;
    const double &vPow2_sPow2 = prev_state_moments.vPow2_sPow2;
    const double &vPow2_cPow1_sPow1 = prev_state_moments.vPow2_cPow1_sPow1;

    const double &wvPow2 = system_noise_moments.wvPow2;
    const double &cyawPow1 = system_noise_moments.cyawPow1;
    const double &wyawPow1 = system_noise_moments.wyawPow1;
    const double &cyawPow2 = system_noise_moments.cyawPow2;
    const double &wvPow1 = system_noise_moments.wvPow1;
    const double &wyawPow2 = system_noise_moments.wyawPow2;
    const double &syawPow2 = system_noise_moments.syawPow2;
    const double &syawPow1 = system_noise_moments.syawPow1;
    const double &cyawPow1_syawPow1 = system_noise_moments.cyawPow1_syawPow1;


    const double &a = control_inputs.a;
    const double &u = control_inputs.u;
    const double &cu = control_inputs.cu;
    const double &su = control_inputs.su;

    // Dynamics updates.
    StateMoments state_moments;

    state_moments.xPow1 = dt*vPow1_cPow1 + xPow1;

    state_moments.yPow1 = dt*vPow1_sPow1 + yPow1;

    state_moments.vPow1 = 1.0*a + vPow1 + wvPow1;

    state_moments.yawPow1 = 1.0*u + wyawPow1 + yawPow1;

    state_moments.cPow1 = cu*cPow1*cyawPow1 - cu*sPow1*syawPow1 - su*cPow1*syawPow1 - su*cyawPow1*sPow1;

    state_moments.sPow1 = cu*cPow1*syawPow1 + cu*cyawPow1*sPow1 + su*cPow1*cyawPow1 - su*sPow1*syawPow1;

    state_moments.xPow2 = pow(dt, 2)*vPow2_cPow2 + 2*dt*vPow1_cPow1_xPow1 + xPow2;

    state_moments.yPow2 = pow(dt, 2)*vPow2_sPow2 + 2*dt*vPow1_sPow1_yPow1 + yPow2;

    state_moments.vPow2 = 1.0*pow(a, 2) + 2*a*vPow1 + 2*a*wvPow1 + 2*vPow1*wvPow1 + vPow2 + wvPow2;

    state_moments.yawPow2 = 1.0*pow(u, 2) + 2*u*wyawPow1 + 2*u*yawPow1 + 2*wyawPow1*yawPow1 + wyawPow2 + yawPow2;

    state_moments.cPow2 = pow(cu, 2)*cPow2*cyawPow2 + pow(cu, 2)*sPow2*syawPow2 - 2*cu*su*cPow1_sPow1*cyawPow2 + 2*cu*su*cPow1_sPow1*syawPow2 - 2*cu*su*cPow2*cyawPow1_syawPow1 + 2*cu*su*cyawPow1_syawPow1*sPow2 + pow(su, 2)*cPow2*syawPow2 + pow(su, 2)*cyawPow2*sPow2 + cPow1_sPow1*cyawPow1_syawPow1*(-2*pow(cu, 2) + 2*pow(su, 2));

    state_moments.sPow2 = pow(cu, 2)*cPow2*syawPow2 + pow(cu, 2)*cyawPow2*sPow2 + 2*cu*su*cPow1_sPow1*cyawPow2 - 2*cu*su*cPow1_sPow1*syawPow2 + 2*cu*su*cPow2*cyawPow1_syawPow1 - 2*cu*su*cyawPow1_syawPow1*sPow2 + pow(su, 2)*cPow2*cyawPow2 + pow(su, 2)*sPow2*syawPow2 + cPow1_sPow1*cyawPow1_syawPow1*(2*pow(cu, 2) - 2*pow(su, 2));

    state_moments.vPow1_cPow1 = a*cu*cPow1*cyawPow1 - a*cu*sPow1*syawPow1 - a*su*cPow1*syawPow1 - a*su*cyawPow1*sPow1 + cu*cPow1*cyawPow1*wvPow1 + cu*cyawPow1*vPow1_cPow1 - cu*sPow1*syawPow1*wvPow1 - cu*syawPow1*vPow1_sPow1 - su*cPow1*syawPow1*wvPow1 - su*cyawPow1*sPow1*wvPow1 - su*cyawPow1*vPow1_sPow1 - su*syawPow1*vPow1_cPow1;

    state_moments.vPow1_sPow1 = a*cu*cPow1*syawPow1 + a*cu*cyawPow1*sPow1 + a*su*cPow1*cyawPow1 - a*su*sPow1*syawPow1 + cu*cPow1*syawPow1*wvPow1 + cu*cyawPow1*sPow1*wvPow1 + cu*cyawPow1*vPow1_sPow1 + cu*syawPow1*vPow1_cPow1 + su*cPow1*cyawPow1*wvPow1 + su*cyawPow1*vPow1_cPow1 - su*sPow1*syawPow1*wvPow1 - su*syawPow1*vPow1_sPow1;

    state_moments.vPow2_cPow2 = pow(a, 2)*pow(cu, 2)*cPow2*cyawPow2 + pow(a, 2)*pow(cu, 2)*sPow2*syawPow2 - 2*pow(a, 2)*cu*su*cPow1_sPow1*cyawPow2 + 2*pow(a, 2)*cu*su*cPow1_sPow1*syawPow2 - 2*pow(a, 2)*cu*su*cPow2*cyawPow1_syawPow1 + 2*pow(a, 2)*cu*su*cyawPow1_syawPow1*sPow2 + pow(a, 2)*pow(su, 2)*cPow2*syawPow2 + pow(a, 2)*pow(su, 2)*cyawPow2*sPow2 + 2*a*pow(cu, 2)*vPow1_cPow2*cyawPow2 + 2*a*pow(cu, 2)*cPow2*cyawPow2*wvPow1 + 2*a*pow(cu, 2)*vPow1_sPow2*syawPow2 + 2*a*pow(cu, 2)*sPow2*syawPow2*wvPow1 - 4*a*cu*su*cPow1_sPow1*cyawPow2*wvPow1 + 4*a*cu*su*cPow1_sPow1*syawPow2*wvPow1 - 4*a*cu*su*vPow1_cPow2*cyawPow1_syawPow1 - 2*a*cu*su*vPow1_cPow1_sPow1*cyawPow2 + 2*a*cu*su*vPow1_cPow1_sPow1*syawPow2 - 4*a*cu*su*cPow2*cyawPow1_syawPow1*wvPow1 + 4*a*cu*su*cyawPow1_syawPow1*vPow1_sPow2 + 4*a*cu*su*cyawPow1_syawPow1*sPow2*wvPow1 - 2*a*cu*su*cyawPow2*vPow1_cPow1_sPow1 + 2*a*cu*su*vPow1_cPow1_sPow1*syawPow2 + 2*a*pow(su, 2)*vPow1_cPow2*syawPow2 + 2*a*pow(su, 2)*cPow2*syawPow2*wvPow1 + 2*a*pow(su, 2)*cyawPow2*vPow1_sPow2 + 2*a*pow(su, 2)*cyawPow2*sPow2*wvPow1 + 2*pow(cu, 2)*vPow1_cPow2*cyawPow2*wvPow1 + pow(cu, 2)*cPow2*cyawPow2*wvPow2 + pow(cu, 2)*cyawPow2*vPow2_cPow2 + 2*pow(cu, 2)*vPow1_sPow2*syawPow2*wvPow1 + pow(cu, 2)*sPow2*syawPow2*wvPow2 + pow(cu, 2)*syawPow2*vPow2_sPow2 - 2*cu*su*cPow1_sPow1*cyawPow2*wvPow2 + 2*cu*su*cPow1_sPow1*syawPow2*wvPow2 - 4*cu*su*vPow1_cPow2*cyawPow1_syawPow1*wvPow1 - 2*cu*su*vPow1_cPow1_sPow1*cyawPow2*wvPow1 + 2*cu*su*vPow1_cPow1_sPow1*syawPow2*wvPow1 - 2*cu*su*cPow2*cyawPow1_syawPow1*wvPow2 + 4*cu*su*cyawPow1_syawPow1*vPow1_sPow2*wvPow1 + 2*cu*su*cyawPow1_syawPow1*sPow2*wvPow2 - 2*cu*su*cyawPow1_syawPow1*vPow2_cPow2 + 2*cu*su*cyawPow1_syawPow1*vPow2_sPow2 - 2*cu*su*cyawPow2*vPow1_cPow1_sPow1*wvPow1 - 2*cu*su*cyawPow2*vPow2_cPow1_sPow1 + 2*cu*su*vPow1_cPow1_sPow1*syawPow2*wvPow1 + 2*cu*su*syawPow2*vPow2_cPow1_sPow1 + 2*pow(su, 2)*vPow1_cPow2*syawPow2*wvPow1 + pow(su, 2)*cPow2*syawPow2*wvPow2 + 2*pow(su, 2)*cyawPow2*vPow1_sPow2*wvPow1 + pow(su, 2)*cyawPow2*sPow2*wvPow2 + pow(su, 2)*cyawPow2*vPow2_sPow2 + pow(su, 2)*syawPow2*vPow2_cPow2 + cPow1_sPow1*cyawPow1_syawPow1*wvPow1*(-4*a*pow(cu, 2) + 4*a*pow(su, 2)) + cPow1_sPow1*cyawPow1_syawPow1*wvPow2*(-2*pow(cu, 2) + 2*pow(su, 2)) + cPow1_sPow1*cyawPow1_syawPow1*(-2*pow(a, 2)*pow(cu, 2) + 2*pow(a, 2)*pow(su, 2)) + vPow1_cPow1_sPow1*cyawPow1_syawPow1*wvPow1*(-2*pow(cu, 2) + 2*pow(su, 2)) + vPow1_cPow1_sPow1*cyawPow1_syawPow1*(-2*a*pow(cu, 2) + 2*a*pow(su, 2)) + cyawPow1_syawPow1*vPow1_cPow1_sPow1*wvPow1*(-2*pow(cu, 2) + 2*pow(su, 2)) + cyawPow1_syawPow1*vPow1_cPow1_sPow1*(-2*a*pow(cu, 2) + 2*a*pow(su, 2)) + cyawPow1_syawPow1*vPow2_cPow1_sPow1*(-2*pow(cu, 2) + 2*pow(su, 2));

    state_moments.vPow2_sPow2 = pow(a, 2)*pow(cu, 2)*cPow2*syawPow2 + pow(a, 2)*pow(cu, 2)*cyawPow2*sPow2 + 2*pow(a, 2)*cu*su*cPow1_sPow1*cyawPow2 - 2*pow(a, 2)*cu*su*cPow1_sPow1*syawPow2 + 2*pow(a, 2)*cu*su*cPow2*cyawPow1_syawPow1 - 2*pow(a, 2)*cu*su*cyawPow1_syawPow1*sPow2 + pow(a, 2)*pow(su, 2)*cPow2*cyawPow2 + pow(a, 2)*pow(su, 2)*sPow2*syawPow2 + 2*a*pow(cu, 2)*vPow1_cPow2*syawPow2 + 2*a*pow(cu, 2)*cPow2*syawPow2*wvPow1 + 2*a*pow(cu, 2)*cyawPow2*vPow1_sPow2 + 2*a*pow(cu, 2)*cyawPow2*sPow2*wvPow1 + 4*a*cu*su*cPow1_sPow1*cyawPow2*wvPow1 - 4*a*cu*su*cPow1_sPow1*syawPow2*wvPow1 + 4*a*cu*su*vPow1_cPow2*cyawPow1_syawPow1 + 2*a*cu*su*vPow1_cPow1_sPow1*cyawPow2 - 2*a*cu*su*vPow1_cPow1_sPow1*syawPow2 + 4*a*cu*su*cPow2*cyawPow1_syawPow1*wvPow1 - 4*a*cu*su*cyawPow1_syawPow1*vPow1_sPow2 - 4*a*cu*su*cyawPow1_syawPow1*sPow2*wvPow1 + 2*a*cu*su*cyawPow2*vPow1_cPow1_sPow1 - 2*a*cu*su*vPow1_cPow1_sPow1*syawPow2 + 2*a*pow(su, 2)*vPow1_cPow2*cyawPow2 + 2*a*pow(su, 2)*cPow2*cyawPow2*wvPow1 + 2*a*pow(su, 2)*vPow1_sPow2*syawPow2 + 2*a*pow(su, 2)*sPow2*syawPow2*wvPow1 + 2*pow(cu, 2)*vPow1_cPow2*syawPow2*wvPow1 + pow(cu, 2)*cPow2*syawPow2*wvPow2 + 2*pow(cu, 2)*cyawPow2*vPow1_sPow2*wvPow1 + pow(cu, 2)*cyawPow2*sPow2*wvPow2 + pow(cu, 2)*cyawPow2*vPow2_sPow2 + pow(cu, 2)*syawPow2*vPow2_cPow2 + 2*cu*su*cPow1_sPow1*cyawPow2*wvPow2 - 2*cu*su*cPow1_sPow1*syawPow2*wvPow2 + 4*cu*su*vPow1_cPow2*cyawPow1_syawPow1*wvPow1 + 2*cu*su*vPow1_cPow1_sPow1*cyawPow2*wvPow1 - 2*cu*su*vPow1_cPow1_sPow1*syawPow2*wvPow1 + 2*cu*su*cPow2*cyawPow1_syawPow1*wvPow2 - 4*cu*su*cyawPow1_syawPow1*vPow1_sPow2*wvPow1 - 2*cu*su*cyawPow1_syawPow1*sPow2*wvPow2 + 2*cu*su*cyawPow1_syawPow1*vPow2_cPow2 - 2*cu*su*cyawPow1_syawPow1*vPow2_sPow2 + 2*cu*su*cyawPow2*vPow1_cPow1_sPow1*wvPow1 + 2*cu*su*cyawPow2*vPow2_cPow1_sPow1 - 2*cu*su*vPow1_cPow1_sPow1*syawPow2*wvPow1 - 2*cu*su*syawPow2*vPow2_cPow1_sPow1 + 2*pow(su, 2)*vPow1_cPow2*cyawPow2*wvPow1 + pow(su, 2)*cPow2*cyawPow2*wvPow2 + pow(su, 2)*cyawPow2*vPow2_cPow2 + 2*pow(su, 2)*vPow1_sPow2*syawPow2*wvPow1 + pow(su, 2)*sPow2*syawPow2*wvPow2 + pow(su, 2)*syawPow2*vPow2_sPow2 + cPow1_sPow1*cyawPow1_syawPow1*wvPow1*(4*a*pow(cu, 2) - 4*a*pow(su, 2)) + cPow1_sPow1*cyawPow1_syawPow1*wvPow2*(2*pow(cu, 2) - 2*pow(su, 2)) + cPow1_sPow1*cyawPow1_syawPow1*(2*pow(a, 2)*pow(cu, 2) - 2*pow(a, 2)*pow(su, 2)) + vPow1_cPow1_sPow1*cyawPow1_syawPow1*wvPow1*(2*pow(cu, 2) - 2*pow(su, 2)) + vPow1_cPow1_sPow1*cyawPow1_syawPow1*(2*a*pow(cu, 2) - 2*a*pow(su, 2)) + cyawPow1_syawPow1*vPow1_cPow1_sPow1*wvPow1*(2*pow(cu, 2) - 2*pow(su, 2)) + cyawPow1_syawPow1*vPow1_cPow1_sPow1*(2*a*pow(cu, 2) - 2*a*pow(su, 2)) + cyawPow1_syawPow1*vPow2_cPow1_sPow1*(2*pow(cu, 2) - 2*pow(su, 2));

    state_moments.vPow1_cPow1_xPow1 = a*cu*dt*vPow1_cPow2*cyawPow1 - a*cu*dt*vPow1_cPow1_sPow1*syawPow1 + a*cu*cPow1_xPow1*cyawPow1 - a*cu*sPow1_xPow1*syawPow1 - a*dt*su*vPow1_cPow2*syawPow1 - a*dt*su*cyawPow1*vPow1_cPow1_sPow1 - a*su*cPow1_xPow1*syawPow1 - a*su*cyawPow1*sPow1_xPow1 + cu*dt*vPow1_cPow2*cyawPow1*wvPow1 + cu*dt*cyawPow1*vPow2_cPow2 - cu*dt*vPow1_cPow1_sPow1*syawPow1*wvPow1 - cu*dt*syawPow1*vPow2_cPow1_sPow1 + cu*cPow1_xPow1*cyawPow1*wvPow1 + cu*cyawPow1*vPow1_cPow1_xPow1 - cu*sPow1_xPow1*syawPow1*wvPow1 - cu*syawPow1*vPow1_sPow1_xPow1 - dt*su*vPow1_cPow2*syawPow1*wvPow1 - dt*su*cyawPow1*vPow1_cPow1_sPow1*wvPow1 - dt*su*cyawPow1*vPow2_cPow1_sPow1 - dt*su*syawPow1*vPow2_cPow2 - su*cPow1_xPow1*syawPow1*wvPow1 - su*cyawPow1*sPow1_xPow1*wvPow1 - su*cyawPow1*vPow1_sPow1_xPow1 - su*syawPow1*vPow1_cPow1_xPow1;

    state_moments.cPow1_xPow1 = cu*dt*vPow1_cPow2*cyawPow1 - cu*dt*vPow1_cPow1_sPow1*syawPow1 + cu*cPow1_xPow1*cyawPow1 - cu*sPow1_xPow1*syawPow1 - dt*su*vPow1_cPow2*syawPow1 - dt*su*cyawPow1*vPow1_cPow1_sPow1 - su*cPow1_xPow1*syawPow1 - su*cyawPow1*sPow1_xPow1;

    state_moments.sPow1_xPow1 = cu*dt*vPow1_cPow2*syawPow1 + cu*dt*cyawPow1*vPow1_cPow1_sPow1 + cu*cPow1_xPow1*syawPow1 + cu*cyawPow1*sPow1_xPow1 + dt*su*vPow1_cPow2*cyawPow1 - dt*su*vPow1_cPow1_sPow1*syawPow1 + su*cPow1_xPow1*cyawPow1 - su*sPow1_xPow1*syawPow1;

    state_moments.vPow1_sPow1_xPow1 = a*cu*dt*vPow1_cPow2*syawPow1 + a*cu*dt*cyawPow1*vPow1_cPow1_sPow1 + a*cu*cPow1_xPow1*syawPow1 + a*cu*cyawPow1*sPow1_xPow1 + a*dt*su*vPow1_cPow2*cyawPow1 - a*dt*su*vPow1_cPow1_sPow1*syawPow1 + a*su*cPow1_xPow1*cyawPow1 - a*su*sPow1_xPow1*syawPow1 + cu*dt*vPow1_cPow2*syawPow1*wvPow1 + cu*dt*cyawPow1*vPow1_cPow1_sPow1*wvPow1 + cu*dt*cyawPow1*vPow2_cPow1_sPow1 + cu*dt*syawPow1*vPow2_cPow2 + cu*cPow1_xPow1*syawPow1*wvPow1 + cu*cyawPow1*sPow1_xPow1*wvPow1 + cu*cyawPow1*vPow1_sPow1_xPow1 + cu*syawPow1*vPow1_cPow1_xPow1 + dt*su*vPow1_cPow2*cyawPow1*wvPow1 + dt*su*cyawPow1*vPow2_cPow2 - dt*su*vPow1_cPow1_sPow1*syawPow1*wvPow1 - dt*su*syawPow1*vPow2_cPow1_sPow1 + su*cPow1_xPow1*cyawPow1*wvPow1 + su*cyawPow1*vPow1_cPow1_xPow1 - su*sPow1_xPow1*syawPow1*wvPow1 - su*syawPow1*vPow1_sPow1_xPow1;

    state_moments.vPow1_sPow1_yPow1 = a*cu*dt*vPow1_cPow1_sPow1*syawPow1 + a*cu*dt*cyawPow1*vPow1_sPow2 + a*cu*cPow1_yPow1*syawPow1 + a*cu*cyawPow1*sPow1_yPow1 + a*dt*su*vPow1_cPow1_sPow1*cyawPow1 - a*dt*su*vPow1_sPow2*syawPow1 + a*su*cPow1_yPow1*cyawPow1 - a*su*sPow1_yPow1*syawPow1 + cu*dt*vPow1_cPow1_sPow1*syawPow1*wvPow1 + cu*dt*cyawPow1*vPow1_sPow2*wvPow1 + cu*dt*cyawPow1*vPow2_sPow2 + cu*dt*syawPow1*vPow2_cPow1_sPow1 + cu*cPow1_yPow1*syawPow1*wvPow1 + cu*cyawPow1*sPow1_yPow1*wvPow1 + cu*cyawPow1*vPow1_sPow1_yPow1 + cu*syawPow1*vPow1_cPow1_yPow1 + dt*su*vPow1_cPow1_sPow1*cyawPow1*wvPow1 + dt*su*cyawPow1*vPow2_cPow1_sPow1 - dt*su*vPow1_sPow2*syawPow1*wvPow1 - dt*su*syawPow1*vPow2_sPow2 + su*cPow1_yPow1*cyawPow1*wvPow1 + su*cyawPow1*vPow1_cPow1_yPow1 - su*sPow1_yPow1*syawPow1*wvPow1 - su*syawPow1*vPow1_sPow1_yPow1;

    state_moments.cPow1_yPow1 = cu*dt*vPow1_cPow1_sPow1*cyawPow1 - cu*dt*vPow1_sPow2*syawPow1 + cu*cPow1_yPow1*cyawPow1 - cu*sPow1_yPow1*syawPow1 - dt*su*vPow1_cPow1_sPow1*syawPow1 - dt*su*cyawPow1*vPow1_sPow2 - su*cPow1_yPow1*syawPow1 - su*cyawPow1*sPow1_yPow1;

    state_moments.sPow1_yPow1 = cu*dt*vPow1_cPow1_sPow1*syawPow1 + cu*dt*cyawPow1*vPow1_sPow2 + cu*cPow1_yPow1*syawPow1 + cu*cyawPow1*sPow1_yPow1 + dt*su*vPow1_cPow1_sPow1*cyawPow1 - dt*su*vPow1_sPow2*syawPow1 + su*cPow1_yPow1*cyawPow1 - su*sPow1_yPow1*syawPow1;

    state_moments.vPow1_cPow1_yPow1 = a*cu*dt*vPow1_cPow1_sPow1*cyawPow1 - a*cu*dt*vPow1_sPow2*syawPow1 + a*cu*cPow1_yPow1*cyawPow1 - a*cu*sPow1_yPow1*syawPow1 - a*dt*su*vPow1_cPow1_sPow1*syawPow1 - a*dt*su*cyawPow1*vPow1_sPow2 - a*su*cPow1_yPow1*syawPow1 - a*su*cyawPow1*sPow1_yPow1 + cu*dt*vPow1_cPow1_sPow1*cyawPow1*wvPow1 + cu*dt*cyawPow1*vPow2_cPow1_sPow1 - cu*dt*vPow1_sPow2*syawPow1*wvPow1 - cu*dt*syawPow1*vPow2_sPow2 + cu*cPow1_yPow1*cyawPow1*wvPow1 + cu*cyawPow1*vPow1_cPow1_yPow1 - cu*sPow1_yPow1*syawPow1*wvPow1 - cu*syawPow1*vPow1_sPow1_yPow1 - dt*su*vPow1_cPow1_sPow1*syawPow1*wvPow1 - dt*su*cyawPow1*vPow1_sPow2*wvPow1 - dt*su*cyawPow1*vPow2_sPow2 - dt*su*syawPow1*vPow2_cPow1_sPow1 - su*cPow1_yPow1*syawPow1*wvPow1 - su*cyawPow1*sPow1_yPow1*wvPow1 - su*cyawPow1*vPow1_sPow1_yPow1 - su*syawPow1*vPow1_cPow1_yPow1;

    state_moments.xPow1_yPow1 = pow(dt, 2)*vPow2_cPow1_sPow1 + dt*vPow1_cPow1_yPow1 + dt*vPow1_sPow1_xPow1 + xPow1_yPow1;

    state_moments.vPow2_cPow1_sPow1 = -4*pow(a, 2)*cu*su*cPow1_sPow1*cyawPow1_syawPow1 + pow(a, 2)*cu*su*cPow2*cyawPow2 - pow(a, 2)*cu*su*cPow2*syawPow2 - pow(a, 2)*cu*su*cyawPow2*sPow2 + pow(a, 2)*cu*su*sPow2*syawPow2 - 8*a*cu*su*cPow1_sPow1*cyawPow1_syawPow1*wvPow1 + 2*a*cu*su*vPow1_cPow2*cyawPow2 - 2*a*cu*su*vPow1_cPow2*syawPow2 - 4*a*cu*su*vPow1_cPow1_sPow1*cyawPow1_syawPow1 + 2*a*cu*su*cPow2*cyawPow2*wvPow1 - 2*a*cu*su*cPow2*syawPow2*wvPow1 - 4*a*cu*su*cyawPow1_syawPow1*vPow1_cPow1_sPow1 - 2*a*cu*su*cyawPow2*vPow1_sPow2 - 2*a*cu*su*cyawPow2*sPow2*wvPow1 + 2*a*cu*su*vPow1_sPow2*syawPow2 + 2*a*cu*su*sPow2*syawPow2*wvPow1 - 4*cu*su*cPow1_sPow1*cyawPow1_syawPow1*wvPow2 + 2*cu*su*vPow1_cPow2*cyawPow2*wvPow1 - 2*cu*su*vPow1_cPow2*syawPow2*wvPow1 - 4*cu*su*vPow1_cPow1_sPow1*cyawPow1_syawPow1*wvPow1 + cu*su*cPow2*cyawPow2*wvPow2 - cu*su*cPow2*syawPow2*wvPow2 - 4*cu*su*cyawPow1_syawPow1*vPow1_cPow1_sPow1*wvPow1 - 4*cu*su*cyawPow1_syawPow1*vPow2_cPow1_sPow1 - 2*cu*su*cyawPow2*vPow1_sPow2*wvPow1 - cu*su*cyawPow2*sPow2*wvPow2 + cu*su*cyawPow2*vPow2_cPow2 - cu*su*cyawPow2*vPow2_sPow2 + 2*cu*su*vPow1_sPow2*syawPow2*wvPow1 + cu*su*sPow2*syawPow2*wvPow2 - cu*su*syawPow2*vPow2_cPow2 + cu*su*syawPow2*vPow2_sPow2 + cPow1_sPow1*cyawPow2*wvPow1*(2*a*pow(cu, 2) - 2*a*pow(su, 2)) + cPow1_sPow1*cyawPow2*wvPow2*(pow(cu, 2) - pow(su, 2)) + cPow1_sPow1*cyawPow2*(pow(a, 2)*pow(cu, 2) - pow(a, 2)*pow(su, 2)) + cPow1_sPow1*syawPow2*wvPow1*(-2*a*pow(cu, 2) + 2*a*pow(su, 2)) + cPow1_sPow1*syawPow2*wvPow2*(-pow(cu, 2) + pow(su, 2)) + cPow1_sPow1*syawPow2*(-pow(a, 2)*pow(cu, 2) + pow(a, 2)*pow(su, 2)) + vPow1_cPow2*cyawPow1_syawPow1*wvPow1*(2*pow(cu, 2) - 2*pow(su, 2)) + vPow1_cPow2*cyawPow1_syawPow1*(2*a*pow(cu, 2) - 2*a*pow(su, 2)) + vPow1_cPow1_sPow1*cyawPow2*wvPow1*(pow(cu, 2) - pow(su, 2)) + vPow1_cPow1_sPow1*cyawPow2*(a*pow(cu, 2) - a*pow(su, 2)) + vPow1_cPow1_sPow1*syawPow2*wvPow1*(-pow(cu, 2) + pow(su, 2)) + vPow1_cPow1_sPow1*syawPow2*(-a*pow(cu, 2) + a*pow(su, 2)) + cPow2*cyawPow1_syawPow1*wvPow1*(2*a*pow(cu, 2) - 2*a*pow(su, 2)) + cPow2*cyawPow1_syawPow1*wvPow2*(pow(cu, 2) - pow(su, 2)) + cPow2*cyawPow1_syawPow1*(pow(a, 2)*pow(cu, 2) - pow(a, 2)*pow(su, 2)) + cyawPow1_syawPow1*vPow1_sPow2*wvPow1*(-2*pow(cu, 2) + 2*pow(su, 2)) + cyawPow1_syawPow1*vPow1_sPow2*(-2*a*pow(cu, 2) + 2*a*pow(su, 2)) + cyawPow1_syawPow1*sPow2*wvPow1*(-2*a*pow(cu, 2) + 2*a*pow(su, 2)) + cyawPow1_syawPow1*sPow2*wvPow2*(-pow(cu, 2) + pow(su, 2)) + cyawPow1_syawPow1*sPow2*(-pow(a, 2)*pow(cu, 2) + pow(a, 2)*pow(su, 2)) + cyawPow1_syawPow1*vPow2_cPow2*(pow(cu, 2) - pow(su, 2)) + cyawPow1_syawPow1*vPow2_sPow2*(-pow(cu, 2) + pow(su, 2)) + cyawPow2*vPow1_cPow1_sPow1*wvPow1*(pow(cu, 2) - pow(su, 2)) + cyawPow2*vPow1_cPow1_sPow1*(a*pow(cu, 2) - a*pow(su, 2)) + cyawPow2*vPow2_cPow1_sPow1*(pow(cu, 2) - pow(su, 2)) + vPow1_cPow1_sPow1*syawPow2*wvPow1*(-pow(cu, 2) + pow(su, 2)) + vPow1_cPow1_sPow1*syawPow2*(-a*pow(cu, 2) + a*pow(su, 2)) + syawPow2*vPow2_cPow1_sPow1*(-pow(cu, 2) + pow(su, 2));

    state_moments.vPow1_xPow1 = a*dt*vPow1_cPow1 + a*xPow1 + dt*vPow2_cPow1 + dt*vPow1_cPow1*wvPow1 + vPow1_xPow1 + wvPow1*xPow1;

    state_moments.vPow2_cPow1 = pow(a, 2)*cu*cPow1*cyawPow1 - pow(a, 2)*cu*sPow1*syawPow1 - pow(a, 2)*su*cPow1*syawPow1 - pow(a, 2)*su*cyawPow1*sPow1 + 2*a*cu*cPow1*cyawPow1*wvPow1 + a*cu*vPow1_cPow1*cyawPow1 + a*cu*cyawPow1*vPow1_cPow1 - 2*a*cu*sPow1*syawPow1*wvPow1 - a*cu*vPow1_sPow1*syawPow1 - a*cu*syawPow1*vPow1_sPow1 - 2*a*su*cPow1*syawPow1*wvPow1 - a*su*vPow1_cPow1*syawPow1 - 2*a*su*cyawPow1*sPow1*wvPow1 - a*su*cyawPow1*vPow1_sPow1 - a*su*cyawPow1*vPow1_sPow1 - a*su*syawPow1*vPow1_cPow1 + cu*cPow1*cyawPow1*wvPow2 + cu*vPow1_cPow1*cyawPow1*wvPow1 + cu*cyawPow1*vPow2_cPow1 + cu*cyawPow1*vPow1_cPow1*wvPow1 - cu*sPow1*syawPow1*wvPow2 - cu*vPow1_sPow1*syawPow1*wvPow1 - cu*syawPow1*vPow2_sPow1 - cu*syawPow1*vPow1_sPow1*wvPow1 - su*cPow1*syawPow1*wvPow2 - su*vPow1_cPow1*syawPow1*wvPow1 - su*cyawPow1*sPow1*wvPow2 - su*cyawPow1*vPow1_sPow1*wvPow1 - su*cyawPow1*vPow2_sPow1 - su*cyawPow1*vPow1_sPow1*wvPow1 - su*syawPow1*vPow2_cPow1 - su*syawPow1*vPow1_cPow1*wvPow1;

    state_moments.xPow1_yawPow1 = dt*u*vPow1_cPow1 + dt*vPow1_cPow1*wyawPow1 + dt*vPow1_cPow1_yawPow1 + u*xPow1 + wyawPow1*xPow1 + xPow1_yawPow1;

    state_moments.vPow1_cPow1_yawPow1 = a*cu*u*cPow1*cyawPow1 - a*cu*u*sPow1*syawPow1 + a*cu*cPow1*cyawPow1*wyawPow1 + a*cu*cPow1_yawPow1*cyawPow1 - a*cu*sPow1*syawPow1*wyawPow1 - a*cu*sPow1_yawPow1*syawPow1 - a*su*u*cPow1*syawPow1 - a*su*u*cyawPow1*sPow1 - a*su*cPow1*syawPow1*wyawPow1 - a*su*cPow1_yawPow1*syawPow1 - a*su*cyawPow1*sPow1*wyawPow1 - a*su*cyawPow1*sPow1_yawPow1 + cu*u*cPow1*cyawPow1*wvPow1 + cu*u*cyawPow1*vPow1_cPow1 - cu*u*sPow1*syawPow1*wvPow1 - cu*u*syawPow1*vPow1_sPow1 + cu*cPow1*cyawPow1*wvPow1*wyawPow1 + cu*cPow1_yawPow1*cyawPow1*wvPow1 + cu*cyawPow1*vPow1_cPow1*wyawPow1 + cu*cyawPow1*vPow1_cPow1_yawPow1 - cu*sPow1*syawPow1*wvPow1*wyawPow1 - cu*sPow1_yawPow1*syawPow1*wvPow1 - cu*syawPow1*vPow1_sPow1*wyawPow1 - cu*syawPow1*vPow1_sPow1_yawPow1 - su*u*cPow1*syawPow1*wvPow1 - su*u*cyawPow1*sPow1*wvPow1 - su*u*cyawPow1*vPow1_sPow1 - su*u*syawPow1*vPow1_cPow1 - su*cPow1*syawPow1*wvPow1*wyawPow1 - su*cPow1_yawPow1*syawPow1*wvPow1 - su*cyawPow1*sPow1*wvPow1*wyawPow1 - su*cyawPow1*sPow1_yawPow1*wvPow1 - su*cyawPow1*vPow1_sPow1*wyawPow1 - su*cyawPow1*vPow1_sPow1_yawPow1 - su*syawPow1*vPow1_cPow1*wyawPow1 - su*syawPow1*vPow1_cPow1_yawPow1;

    state_moments.vPow1_yPow1 = a*dt*vPow1_sPow1 + a*yPow1 + dt*vPow2_sPow1 + dt*vPow1_sPow1*wvPow1 + vPow1_yPow1 + wvPow1*yPow1;

    state_moments.vPow2_sPow1 = pow(a, 2)*cu*cPow1*syawPow1 + pow(a, 2)*cu*cyawPow1*sPow1 + pow(a, 2)*su*cPow1*cyawPow1 - pow(a, 2)*su*sPow1*syawPow1 + 2*a*cu*cPow1*syawPow1*wvPow1 + a*cu*vPow1_cPow1*syawPow1 + 2*a*cu*cyawPow1*sPow1*wvPow1 + a*cu*cyawPow1*vPow1_sPow1 + a*cu*cyawPow1*vPow1_sPow1 + a*cu*syawPow1*vPow1_cPow1 + 2*a*su*cPow1*cyawPow1*wvPow1 + a*su*vPow1_cPow1*cyawPow1 + a*su*cyawPow1*vPow1_cPow1 - 2*a*su*sPow1*syawPow1*wvPow1 - a*su*vPow1_sPow1*syawPow1 - a*su*syawPow1*vPow1_sPow1 + cu*cPow1*syawPow1*wvPow2 + cu*vPow1_cPow1*syawPow1*wvPow1 + cu*cyawPow1*sPow1*wvPow2 + cu*cyawPow1*vPow1_sPow1*wvPow1 + cu*cyawPow1*vPow2_sPow1 + cu*cyawPow1*vPow1_sPow1*wvPow1 + cu*syawPow1*vPow2_cPow1 + cu*syawPow1*vPow1_cPow1*wvPow1 + su*cPow1*cyawPow1*wvPow2 + su*vPow1_cPow1*cyawPow1*wvPow1 + su*cyawPow1*vPow2_cPow1 + su*cyawPow1*vPow1_cPow1*wvPow1 - su*sPow1*syawPow1*wvPow2 - su*vPow1_sPow1*syawPow1*wvPow1 - su*syawPow1*vPow2_sPow1 - su*syawPow1*vPow1_sPow1*wvPow1;

    state_moments.yPow1_yawPow1 = dt*u*vPow1_sPow1 + dt*vPow1_sPow1*wyawPow1 + dt*vPow1_sPow1_yawPow1 + u*yPow1 + wyawPow1*yPow1 + yPow1_yawPow1;

    state_moments.vPow1_sPow1_yawPow1 = a*cu*u*cPow1*syawPow1 + a*cu*u*cyawPow1*sPow1 + a*cu*cPow1*syawPow1*wyawPow1 + a*cu*cPow1_yawPow1*syawPow1 + a*cu*cyawPow1*sPow1*wyawPow1 + a*cu*cyawPow1*sPow1_yawPow1 + a*su*u*cPow1*cyawPow1 - a*su*u*sPow1*syawPow1 + a*su*cPow1*cyawPow1*wyawPow1 + a*su*cPow1_yawPow1*cyawPow1 - a*su*sPow1*syawPow1*wyawPow1 - a*su*sPow1_yawPow1*syawPow1 + cu*u*cPow1*syawPow1*wvPow1 + cu*u*cyawPow1*sPow1*wvPow1 + cu*u*cyawPow1*vPow1_sPow1 + cu*u*syawPow1*vPow1_cPow1 + cu*cPow1*syawPow1*wvPow1*wyawPow1 + cu*cPow1_yawPow1*syawPow1*wvPow1 + cu*cyawPow1*sPow1*wvPow1*wyawPow1 + cu*cyawPow1*sPow1_yawPow1*wvPow1 + cu*cyawPow1*vPow1_sPow1*wyawPow1 + cu*cyawPow1*vPow1_sPow1_yawPow1 + cu*syawPow1*vPow1_cPow1*wyawPow1 + cu*syawPow1*vPow1_cPow1_yawPow1 + su*u*cPow1*cyawPow1*wvPow1 + su*u*cyawPow1*vPow1_cPow1 - su*u*sPow1*syawPow1*wvPow1 - su*u*syawPow1*vPow1_sPow1 + su*cPow1*cyawPow1*wvPow1*wyawPow1 + su*cPow1_yawPow1*cyawPow1*wvPow1 + su*cyawPow1*vPow1_cPow1*wyawPow1 + su*cyawPow1*vPow1_cPow1_yawPow1 - su*sPow1*syawPow1*wvPow1*wyawPow1 - su*sPow1_yawPow1*syawPow1*wvPow1 - su*syawPow1*vPow1_sPow1*wyawPow1 - su*syawPow1*vPow1_sPow1_yawPow1;

    state_moments.vPow1_yawPow1 = 1.0*a*u + a*wyawPow1 + a*yawPow1 + u*vPow1 + u*wvPow1 + vPow1*wyawPow1 + vPow1_yawPow1 + wvPow1*wyawPow1 + wvPow1*yawPow1;

    state_moments.cPow1_sPow1 = -4*cu*su*cPow1_sPow1*cyawPow1_syawPow1 + cu*su*cPow2*cyawPow2 - cu*su*cPow2*syawPow2 - cu*su*cyawPow2*sPow2 + cu*su*sPow2*syawPow2 + cPow1_sPow1*cyawPow2*(pow(cu, 2) - pow(su, 2)) + cPow1_sPow1*syawPow2*(-pow(cu, 2) + pow(su, 2)) + cPow2*cyawPow1_syawPow1*(pow(cu, 2) - pow(su, 2)) + cyawPow1_syawPow1*sPow2*(-pow(cu, 2) + pow(su, 2));

    state_moments.vPow1_cPow2 = a*pow(cu, 2)*cPow2*cyawPow2 + a*pow(cu, 2)*sPow2*syawPow2 - 2*a*cu*su*cPow1_sPow1*cyawPow2 + 2*a*cu*su*cPow1_sPow1*syawPow2 - 2*a*cu*su*cPow2*cyawPow1_syawPow1 + 2*a*cu*su*cyawPow1_syawPow1*sPow2 + a*pow(su, 2)*cPow2*syawPow2 + a*pow(su, 2)*cyawPow2*sPow2 + pow(cu, 2)*vPow1_cPow2*cyawPow2 + pow(cu, 2)*cPow2*cyawPow2*wvPow1 + pow(cu, 2)*vPow1_sPow2*syawPow2 + pow(cu, 2)*sPow2*syawPow2*wvPow1 - 2*cu*su*cPow1_sPow1*cyawPow2*wvPow1 + 2*cu*su*cPow1_sPow1*syawPow2*wvPow1 - 2*cu*su*vPow1_cPow2*cyawPow1_syawPow1 - cu*su*vPow1_cPow1_sPow1*cyawPow2 + cu*su*vPow1_cPow1_sPow1*syawPow2 - 2*cu*su*cPow2*cyawPow1_syawPow1*wvPow1 + 2*cu*su*cyawPow1_syawPow1*vPow1_sPow2 + 2*cu*su*cyawPow1_syawPow1*sPow2*wvPow1 - cu*su*cyawPow2*vPow1_cPow1_sPow1 + cu*su*vPow1_cPow1_sPow1*syawPow2 + pow(su, 2)*vPow1_cPow2*syawPow2 + pow(su, 2)*cPow2*syawPow2*wvPow1 + pow(su, 2)*cyawPow2*vPow1_sPow2 + pow(su, 2)*cyawPow2*sPow2*wvPow1 + cPow1_sPow1*cyawPow1_syawPow1*wvPow1*(-2*pow(cu, 2) + 2*pow(su, 2)) + cPow1_sPow1*cyawPow1_syawPow1*(-2*a*pow(cu, 2) + 2*a*pow(su, 2)) + vPow1_cPow1_sPow1*cyawPow1_syawPow1*(-pow(cu, 2) + pow(su, 2)) + cyawPow1_syawPow1*vPow1_cPow1_sPow1*(-pow(cu, 2) + pow(su, 2));

    state_moments.vPow1_cPow1_sPow1 = -4*a*cu*su*cPow1_sPow1*cyawPow1_syawPow1 + a*cu*su*cPow2*cyawPow2 - a*cu*su*cPow2*syawPow2 - a*cu*su*cyawPow2*sPow2 + a*cu*su*sPow2*syawPow2 + pow(cu, 2)*vPow1_cPow1_sPow1*cyawPow2 - pow(cu, 2)*vPow1_cPow1_sPow1*syawPow2 - 4*cu*su*cPow1_sPow1*cyawPow1_syawPow1*wvPow1 + cu*su*vPow1_cPow2*cyawPow2 - cu*su*vPow1_cPow2*syawPow2 - 2*cu*su*vPow1_cPow1_sPow1*cyawPow1_syawPow1 + cu*su*cPow2*cyawPow2*wvPow1 - cu*su*cPow2*syawPow2*wvPow1 - 2*cu*su*cyawPow1_syawPow1*vPow1_cPow1_sPow1 - cu*su*cyawPow2*vPow1_sPow2 - cu*su*cyawPow2*sPow2*wvPow1 + cu*su*vPow1_sPow2*syawPow2 + cu*su*sPow2*syawPow2*wvPow1 + pow(su, 2)*vPow1_cPow1_sPow1*syawPow2 - pow(su, 2)*cyawPow2*vPow1_cPow1_sPow1 + cPow1_sPow1*cyawPow2*wvPow1*(pow(cu, 2) - pow(su, 2)) + cPow1_sPow1*cyawPow2*(a*pow(cu, 2) - a*pow(su, 2)) + cPow1_sPow1*syawPow2*wvPow1*(-pow(cu, 2) + pow(su, 2)) + cPow1_sPow1*syawPow2*(-a*pow(cu, 2) + a*pow(su, 2)) + vPow1_cPow2*cyawPow1_syawPow1*(pow(cu, 2) - pow(su, 2)) + cPow2*cyawPow1_syawPow1*wvPow1*(pow(cu, 2) - pow(su, 2)) + cPow2*cyawPow1_syawPow1*(a*pow(cu, 2) - a*pow(su, 2)) + cyawPow1_syawPow1*vPow1_sPow2*(-pow(cu, 2) + pow(su, 2)) + cyawPow1_syawPow1*sPow2*wvPow1*(-pow(cu, 2) + pow(su, 2)) + cyawPow1_syawPow1*sPow2*(-a*pow(cu, 2) + a*pow(su, 2));

    state_moments.vPow1_cPow1_sPow1 = -4*a*cu*su*cPow1_sPow1*cyawPow1_syawPow1 + a*cu*su*cPow2*cyawPow2 - a*cu*su*cPow2*syawPow2 - a*cu*su*cyawPow2*sPow2 + a*cu*su*sPow2*syawPow2 - pow(cu, 2)*vPow1_cPow1_sPow1*syawPow2 + pow(cu, 2)*cyawPow2*vPow1_cPow1_sPow1 - 4*cu*su*cPow1_sPow1*cyawPow1_syawPow1*wvPow1 + cu*su*vPow1_cPow2*cyawPow2 - cu*su*vPow1_cPow2*syawPow2 - 2*cu*su*vPow1_cPow1_sPow1*cyawPow1_syawPow1 + cu*su*cPow2*cyawPow2*wvPow1 - cu*su*cPow2*syawPow2*wvPow1 - 2*cu*su*cyawPow1_syawPow1*vPow1_cPow1_sPow1 - cu*su*cyawPow2*vPow1_sPow2 - cu*su*cyawPow2*sPow2*wvPow1 + cu*su*vPow1_sPow2*syawPow2 + cu*su*sPow2*syawPow2*wvPow1 - pow(su, 2)*vPow1_cPow1_sPow1*cyawPow2 + pow(su, 2)*vPow1_cPow1_sPow1*syawPow2 + cPow1_sPow1*cyawPow2*wvPow1*(pow(cu, 2) - pow(su, 2)) + cPow1_sPow1*cyawPow2*(a*pow(cu, 2) - a*pow(su, 2)) + cPow1_sPow1*syawPow2*wvPow1*(-pow(cu, 2) + pow(su, 2)) + cPow1_sPow1*syawPow2*(-a*pow(cu, 2) + a*pow(su, 2)) + vPow1_cPow2*cyawPow1_syawPow1*(pow(cu, 2) - pow(su, 2)) + cPow2*cyawPow1_syawPow1*wvPow1*(pow(cu, 2) - pow(su, 2)) + cPow2*cyawPow1_syawPow1*(a*pow(cu, 2) - a*pow(su, 2)) + cyawPow1_syawPow1*vPow1_sPow2*(-pow(cu, 2) + pow(su, 2)) + cyawPow1_syawPow1*sPow2*wvPow1*(-pow(cu, 2) + pow(su, 2)) + cyawPow1_syawPow1*sPow2*(-a*pow(cu, 2) + a*pow(su, 2));

    state_moments.vPow1_sPow2 = a*pow(cu, 2)*cPow2*syawPow2 + a*pow(cu, 2)*cyawPow2*sPow2 + 2*a*cu*su*cPow1_sPow1*cyawPow2 - 2*a*cu*su*cPow1_sPow1*syawPow2 + 2*a*cu*su*cPow2*cyawPow1_syawPow1 - 2*a*cu*su*cyawPow1_syawPow1*sPow2 + a*pow(su, 2)*cPow2*cyawPow2 + a*pow(su, 2)*sPow2*syawPow2 + pow(cu, 2)*vPow1_cPow2*syawPow2 + pow(cu, 2)*cPow2*syawPow2*wvPow1 + pow(cu, 2)*cyawPow2*vPow1_sPow2 + pow(cu, 2)*cyawPow2*sPow2*wvPow1 + 2*cu*su*cPow1_sPow1*cyawPow2*wvPow1 - 2*cu*su*cPow1_sPow1*syawPow2*wvPow1 + 2*cu*su*vPow1_cPow2*cyawPow1_syawPow1 + cu*su*vPow1_cPow1_sPow1*cyawPow2 - cu*su*vPow1_cPow1_sPow1*syawPow2 + 2*cu*su*cPow2*cyawPow1_syawPow1*wvPow1 - 2*cu*su*cyawPow1_syawPow1*vPow1_sPow2 - 2*cu*su*cyawPow1_syawPow1*sPow2*wvPow1 + cu*su*cyawPow2*vPow1_cPow1_sPow1 - cu*su*vPow1_cPow1_sPow1*syawPow2 + pow(su, 2)*vPow1_cPow2*cyawPow2 + pow(su, 2)*cPow2*cyawPow2*wvPow1 + pow(su, 2)*vPow1_sPow2*syawPow2 + pow(su, 2)*sPow2*syawPow2*wvPow1 + cPow1_sPow1*cyawPow1_syawPow1*wvPow1*(2*pow(cu, 2) - 2*pow(su, 2)) + cPow1_sPow1*cyawPow1_syawPow1*(2*a*pow(cu, 2) - 2*a*pow(su, 2)) + vPow1_cPow1_sPow1*cyawPow1_syawPow1*(pow(cu, 2) - pow(su, 2)) + cyawPow1_syawPow1*vPow1_cPow1_sPow1*(pow(cu, 2) - pow(su, 2));

    state_moments.cPow1_yawPow1 = cu*u*cPow1*cyawPow1 - cu*u*sPow1*syawPow1 + cu*cPow1*cyawPow1*wyawPow1 + cu*cPow1_yawPow1*cyawPow1 - cu*sPow1*syawPow1*wyawPow1 - cu*sPow1_yawPow1*syawPow1 - su*u*cPow1*syawPow1 - su*u*cyawPow1*sPow1 - su*cPow1*syawPow1*wyawPow1 - su*cPow1_yawPow1*syawPow1 - su*cyawPow1*sPow1*wyawPow1 - su*cyawPow1*sPow1_yawPow1;

    state_moments.sPow1_yawPow1 = cu*u*cPow1*syawPow1 + cu*u*cyawPow1*sPow1 + cu*cPow1*syawPow1*wyawPow1 + cu*cPow1_yawPow1*syawPow1 + cu*cyawPow1*sPow1*wyawPow1 + cu*cyawPow1*sPow1_yawPow1 + su*u*cPow1*cyawPow1 - su*u*sPow1*syawPow1 + su*cPow1*cyawPow1*wyawPow1 + su*cPow1_yawPow1*cyawPow1 - su*sPow1*syawPow1*wyawPow1 - su*sPow1_yawPow1*syawPow1;

    return state_moments;
}

MobileRobotModel::ObservationMoments MobileRobotModel::getObservationMoments(const measurementInputStateMoments& state_moments,
                                                                             const ObservationNoiseMoments & observation_noise_moments)
{
    // State Moments
    const double & xPow1 = state_moments.xPow1;
    const double & yPow1 = state_moments.yPow1;
    const double & vPow1 = state_moments.vPow1;
    const double & cyawPow1 = state_moments.cyawPow1;
    const double & syawPow1 = state_moments.syawPow1;

    const double & xPow2 = state_moments.xPow2;
    const double & yPow2 = state_moments.yPow2;
    const double & vPow2 = state_moments.vPow2;
    const double & cyawPow2 = state_moments.cyawPow2;
    const double & syawPow2 = state_moments.syawPow2;
    const double & xPow1_cyawPow1 = state_moments.xPow1_cyawPow1;
    const double & xPow1_syawPow1 = state_moments.xPow1_syawPow1;
    const double & yPow1_cyawPow1 = state_moments.yPow1_cyawPow1;
    const double & yPow1_syawPow1 = state_moments.yPow1_syawPow1;
    const double & vPow1_cyawPow1 = state_moments.vPow1_cyawPow1;
    const double & vPow1_syawPow1 = state_moments.vPow1_syawPow1;
    const double & xPow1_yPow1 = state_moments.xPow1_yPow1;
    const double & xPow1_vPow1 = state_moments.xPow1_vPow1;
    const double & yPow1_vPow1 = state_moments.yPow1_vPow1;
    const double & cyawPow1_syawPow1 = state_moments.cyawPow1_syawPow1;

    const double & vPow2_cyawPow1 = state_moments.vPow2_cyawPow1;
    const double & vPow2_syawPow1 = state_moments.vPow2_syawPow1;
    const double & vPow1_cyawPow2 = state_moments.vPow1_cyawPow2;
    const double & vPow1_syawPow2 = state_moments.vPow1_syawPow2;
    const double & xPow1_vPow1_cyawPow1 = state_moments.xPow1_vPow1_cyawPow1;
    const double & xPow1_vPow1_syawPow1 = state_moments.xPow1_vPow1_syawPow1;
    const double & yPow1_vPow1_cyawPow1 = state_moments.yPow1_vPow1_cyawPow1;
    const double & yPow1_vPow1_syawPow1 = state_moments.yPow1_vPow1_syawPow1;
    const double & vPow1_cyawPow1_syawPow1 = state_moments.vPow1_cyawPow1_syawPow1;

    const double & vPow2_cyawPow2 = state_moments.vPow2_cyawPow2;
    const double & vPow2_syawPow2 = state_moments.vPow2_syawPow2;
    const double & vPow2_cyawPow1_syawPow1 = state_moments.vPow2_cyawPow1_syawPow1;

    // Observation Noise
    const double & wxPow1 = observation_noise_moments.wxPow1;
    const double & wyPow1 = observation_noise_moments.wyPow1;
    const double & wvPow1 = observation_noise_moments.wvPow1;
    const double & cwyawPow1 = observation_noise_moments.cwyawPow1;
    const double & swyawPow1 = observation_noise_moments.swyawPow1;

    const double & wxPow2 = observation_noise_moments.wxPow2;
    const double & wyPow2 = observation_noise_moments.wyPow2;
    const double & wvPow2 = observation_noise_moments.wvPow2;
    const double & cwyawPow2 = observation_noise_moments.cwyawPow2;
    const double & swyawPow2 = observation_noise_moments.swyawPow2;
    const double & cwyawPow1_swyawPow1 = observation_noise_moments.cwyawPow1_swyawPow1;

    // Calculate Observation Moments
    ObservationMoments observation_moments;
    observation_moments.xPow1 = xPow1 + vPow1 * wxPow1;
    observation_moments.yPow1 = yPow1 + vPow1 * wyPow1;
    observation_moments.vcPow1 = vPow1_cyawPow1 * cwyawPow1 - vPow1_syawPow1 * swyawPow1
                                 + wvPow1 * cyawPow1 * cwyawPow1 - wvPow1 * syawPow1 * swyawPow1;
    observation_moments.xPow2 = xPow2 + vPow2 * wxPow2 + 2.0 * wxPow1 * xPow1_vPow1;
    observation_moments.yPow2 = yPow2 + vPow2 * wyPow2 + 2.0 * wyPow1 * yPow1_vPow1;
    observation_moments.vcPow2 = vPow2_syawPow2 * swyawPow2
                                - 2 * vPow2_cyawPow1_syawPow1 * cwyawPow1_swyawPow1
                                + vPow2_cyawPow2 * cwyawPow2
                                + 2 * vPow1_syawPow2 * wvPow1 * swyawPow2
                                - 4 * vPow1_cyawPow1_syawPow1 * cwyawPow1_swyawPow1 * wvPow1
                                + 2 * vPow1_cyawPow2 * wvPow1 * cwyawPow2
                                + wvPow2 * syawPow2 * swyawPow2
                                - 2 * wvPow2 * cyawPow1_syawPow1 * cwyawPow1_swyawPow1
                                + wvPow2 * cyawPow2 * cwyawPow2;
    observation_moments.xPow1_yPow1 = vPow2 * wxPow1 * wyPow1 + yPow1_vPow1 * wxPow1 + xPow1_vPow1 * wyPow1 + xPow1_yPow1;
    observation_moments.xPow1_vcPow1 = - vPow2_syawPow1 * swyawPow1 * wxPow1
                                       + vPow2_cyawPow1 * cwyawPow1 * wxPow1
                                       - vPow1_syawPow1 * swyawPow1 * wxPow1 * wvPow1
                                       + vPow1_cyawPow1 * cwyawPow1 * wxPow1 * wvPow1
                                       - xPow1_vPow1_syawPow1 * swyawPow1
                                       + xPow1_vPow1_cyawPow1 * cwyawPow1
                                       - xPow1_syawPow1 * swyawPow1 * wvPow1
                                       + xPow1_cyawPow1 * cwyawPow1 * wvPow1;
    observation_moments.yPow1_vcPow1 = -vPow2_syawPow1 * swyawPow1 * wyPow1 +
                                        vPow2_cyawPow1 * cwyawPow1 * wyPow1 -
                                        vPow1_syawPow1 * swyawPow1 * wyPow1 * wvPow1 +
                                        vPow1_cyawPow1 * cwyawPow1 * wyPow1 * wvPow1 -
                                        yPow1_vPow1_syawPow1 * swyawPow1 +
                                        yPow1_vPow1_cyawPow1 * cwyawPow1 -
                                        yPow1_syawPow1 * swyawPow1 * wvPow1 +
                                        yPow1_cyawPow1 * cwyawPow1 * wvPow1;

    return observation_moments;
}