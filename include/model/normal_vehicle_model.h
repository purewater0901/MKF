#ifndef UNCERTAINTY_PROPAGATION_NORMAL_VEHICLE_MODEL_H
#define UNCERTAINTY_PROPAGATION_NORMAL_VEHICLE_MODEL_H

#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Eigen>

namespace NormalVehicle
{
    struct StateInfo {
        Eigen::Vector3d mean;
        Eigen::Matrix3d covariance;
    };

    struct ObservedInfo {
        Eigen::Vector2d mean;
        Eigen::Matrix2d covariance;
    };

    namespace STATE {
        enum IDX {
            X = 0,
            Y = 1,
            YAW = 2,
        };
    }

    namespace OBSERVATION {
        enum IDX {
            R = 0,
            YAW = 1,
        };
    }

    namespace INPUT {
        enum IDX {
            V = 0,
            U = 1,
        };
    }

    namespace SYSTEM_NOISE {
        enum IDX {
            WX = 0,
            WY = 1,
            WYAW = 2,
        };
    }

    namespace OBSERVATION_NOISE {
        enum IDX {
            WR = 0,
            WYAW= 1,
        };
    }
}

class NormalVehicleModel
{
public:
    struct StateMoments {
        double xPow1;
        double cPow1;
        double sPow1;
        double yPow1;
        double yawPow1;
        double xPow2;
        double cPow2;
        double sPow2;
        double cPow1_xPow1;
        double sPow1_xPow1;
        double yPow2;
        double sPow1_yPow1;
        double cPow1_yPow1;
        double yawPow2;
        double xPow1_yPow1;
        double cPow1_sPow1;
        double xPow1_yawPow1;
        double cPow1_yawPow1;
        double yPow1_yawPow1;
        double sPow1_yawPow1;
    };

    struct ReducedStateMoments {
        double yawPow1;

        double xPow2;
        double yPow2;
        double yawPow2;

        double xPow3;
        double yPow3;
        double xPow2_yawPow1;
        double yPow2_yawPow1;

        double xPow4;
        double yPow4;
        double xPow2_yPow2;
    };

    struct ObservationMoments {
        double rPow1;
        double yawPow1;

        double rPow2;
        double yawPow2;
        double rPow1_yawPow1;
    };

    struct SystemNoiseMoments {
        double wxPow1;
        double wyPow1;
        double wyawPow1;
        double syawPow1;
        double cyawPow1;

        double wxPow2;
        double wyPow2;
        double wyawPow2;

        double syawPow2;
        double cyawPow2;
        double cyawPow1_syawPow1;
    };

    struct ObservationNoiseMoments {
        double w_rPow1;
        double w_yawPow1;

        double w_rPow2;
        double w_yawPow2;
    };

    struct Controls{
        double v;
        double u;
        double cu;
        double su;
    };

    NormalVehicleModel() = default;

    Eigen::Vector3d propagate(const Eigen::Vector3d& x_curr, const Eigen::Vector2d& u_curr, const Eigen::Vector3d& system_noise);
    Eigen::Vector2d observe(const Eigen::Vector3d& x_curr, const Eigen::Vector2d& observation_noise);

    StateMoments propagateStateMoments(const StateMoments & prev_state_moments,
                                       const SystemNoiseMoments& system_noise_moments,
                                       const Controls& control_inputs);

    ObservationMoments getObservationMoments(const ReducedStateMoments & state_moments,
                                             const ObservationNoiseMoments & observation_noise_moments);

};

#endif //UNCERTAINTY_PROPAGATION_NORMAL_VEHICLE_MODEL_H
