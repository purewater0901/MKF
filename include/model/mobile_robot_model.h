#ifndef UNCERTAINTY_PROPAGATION_MOBILE_ROBOT_MODEL_H
#define UNCERTAINTY_PROPAGATION_MOBILE_ROBOT_MODEL_H

#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Eigen>

namespace MobileRobot
{
    struct StateInfo {
        Eigen::Vector4d mean;
        Eigen::Matrix4d covariance;
    };

    struct ObservedInfo {
        Eigen::Vector3d mean;
        Eigen::Matrix3d covariance;
    };

    namespace STATE {
        enum IDX {
            X = 0,
            Y = 1,
            V = 2,
            YAW = 3,
        };
    }

    namespace OBSERVATION {
        enum IDX {
            X = 0,
            Y = 1,
            VC = 2
        };
    }

    namespace INPUT {
        enum IDX {
            A = 0,
            U = 1,
        };
    }

    namespace SYSTEM_NOISE {
        enum IDX {
            WV = 0,
            WYAW = 1,
        };
    }

    namespace OBSERVATION_NOISE {
        enum IDX {
            WX = 0,
            WY = 1,
            WVC = 2,
            WYAW = 3,
        };
    }
}

class MobileRobotModel
{
public:
    struct StateMoments {
        double xPow1;
        double yPow1;
        double vPow1;
        double yawPow1;
        double cPow1;
        double sPow1;

        double xPow2;
        double yPow2;
        double vPow2;
        double yawPow2;
        double cPow2;
        double sPow2;
        double xPow1_yPow1;
        double xPow1_yawPow1;
        double yPow1_yawPow1;
        double vPow1_xPow1;
        double vPow1_yPow1;
        double vPow1_yawPow1;
        double vPow1_cPow1;
        double vPow1_sPow1;
        double cPow1_xPow1;
        double sPow1_xPow1;
        double cPow1_yPow1;
        double sPow1_yPow1;
        double cPow1_yawPow1;
        double sPow1_yawPow1;
        double cPow1_sPow1;

        double vPow1_cPow2;
        double vPow1_sPow2;
        double vPow2_cPow1;
        double vPow2_sPow1;
        double vPow1_cPow1_xPow1;
        double vPow1_cPow1_yPow1;
        double vPow1_cPow1_yawPow1;
        double vPow1_sPow1_xPow1;
        double vPow1_sPow1_yPow1;
        double vPow1_sPow1_yawPow1;
        double vPow1_cPow1_sPow1;

        double vPow2_cPow2;
        double vPow2_sPow2;
        double vPow2_cPow1_sPow1;
    };

    struct measurementInputStateMoments {
        double xPow1;
        double yPow1;
        double vPow1;
        double cyawPow1;
        double syawPow1;

        double xPow2;
        double yPow2;
        double vPow2;
        double cyawPow2;
        double syawPow2;
        double xPow1_cyawPow1;
        double xPow1_syawPow1;
        double yPow1_cyawPow1;
        double yPow1_syawPow1;
        double vPow1_cyawPow1;
        double vPow1_syawPow1;
        double xPow1_yPow1;
        double xPow1_vPow1;
        double yPow1_vPow1;
        double cyawPow1_syawPow1;

        double vPow2_cyawPow1;
        double vPow2_syawPow1;
        double vPow1_cyawPow2;
        double vPow1_syawPow2;
        double xPow1_vPow1_cyawPow1;
        double xPow1_vPow1_syawPow1;
        double yPow1_vPow1_cyawPow1;
        double yPow1_vPow1_syawPow1;
        double vPow1_cyawPow1_syawPow1;

        double vPow2_cyawPow2;
        double vPow2_syawPow2;
        double vPow2_cyawPow1_syawPow1;
    };

    struct ObservationMoments {
        double xPow1;
        double yPow1;
        double vcPow1;

        double xPow2;
        double yPow2;
        double vcPow2;
        double xPow1_yPow1;
        double xPow1_vcPow1;
        double yPow1_vcPow1;
    };

    struct SystemNoiseMoments {
        double wvPow1;
        double wyawPow1;
        double cyawPow1;
        double syawPow1;

        double wvPow2;
        double wyawPow2;
        double cyawPow2;
        double syawPow2;
        double cyawPow1_syawPow1;
    };

    struct ObservationNoiseMoments {
        double wxPow1;
        double wyPow1;
        double wvPow1;
        double cwyawPow1;
        double swyawPow1;

        double wxPow2;
        double wyPow2;
        double wvPow2;
        double cwyawPow2;
        double swyawPow2;
        double cwyawPow1_swyawPow1;
    };

    struct Controls{
        double a;
        double u;
        double cu;
        double su;
    };

    MobileRobotModel() = default;

    Eigen::Vector4d propagate(const Eigen::Vector4d& x_curr,
                              const Eigen::Vector2d& u_curr,
                              const Eigen::Vector2d& system_noise,
                              const double dt);
    Eigen::Vector3d observe(const Eigen::Vector4d& x_curr, const Eigen::Vector4d& observation_noise);

    StateMoments propagateStateMoments(const StateMoments & prev_state_moments,
                                       const SystemNoiseMoments& system_noise_moments,
                                       const Controls& control_inputs,
                                       const double dt);

    ObservationMoments getObservationMoments(const measurementInputStateMoments & state_moments,
                                             const ObservationNoiseMoments & observation_noise_moments);
};

#endif //UNCERTAINTY_PROPAGATION_MOBILE_ROBOT_MODEL_H
