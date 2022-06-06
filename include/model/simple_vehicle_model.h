#ifndef UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_MODEL_H
#define UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_MODEL_H

#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Eigen>

namespace SimpleVehicle
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
            RCOS = 0,
            RSIN = 1,
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
            WV = 0,
            WU = 1,
        };
    }

    namespace OBSERVATION_NOISE {
        enum IDX {
            WR = 0,
            WA = 1,
        };
    }
}

class SimpleVehicleModel
{
public:
    struct StateMoments {
        double xPow1{0.0};
        double cPow1{0.0};
        double sPow1{0.0};
        double yPow1{0.0};
        double yawPow1{0.0};

        double xPow2{0.0};
        double yPow2{0.0};
        double cPow2{0.0};
        double sPow2{0.0};
        double yawPow2{0.0};
        double cPow1_xPow1{0.0};
        double sPow1_xPow1{0.0};
        double sPow1_yPow1{0.0};
        double cPow1_yPow1{0.0};
        double xPow1_yPow1{0.0};
        double cPow1_sPow1{0.0};
        double xPow1_yawPow1{0.0};
        double yPow1_yawPow1{0.0};
        double cPow1_yawPow1{0.0};
        double sPow1_yawPow1{0.0};
    };

    struct ReducedStateMoments {
        double cPow1{0.0};
        double sPow1{0.0};

        double cPow2{0.0};
        double sPow2{0.0};
        double xPow1_cPow1{0.0};
        double yPow1_cPow1{0.0};
        double xPow1_sPow1{0.0};
        double yPow1_sPow1{0.0};
        double cPow1_sPow1{0.0};

        double xPow1_cPow2{0.0};
        double yPow1_cPow2{0.0};
        double xPow1_sPow2{0.0};
        double yPow1_sPow2{0.0};
        double xPow1_cPow1_sPow1{0.0};
        double yPow1_cPow1_sPow1{0.0};

        double xPow2_cPow2{0.0};
        double yPow2_cPow2{0.0};
        double xPow2_sPow2{0.0};
        double yPow2_sPow2{0.0};
        double xPow1_yPow1_cPow2{0.0};
        double xPow1_yPow1_sPow2{0.0};
        double xPow2_cPow1_sPow1{0.0};
        double yPow2_cPow1_sPow1{0.0};
        double xPow1_yPow1_cPow1_sPow1{0.0};
    };

    struct ObservationMoments {
        double rcosPow1{0.0};
        double rsinPow1{0.0};

        double rcosPow2{0.0};
        double rsinPow2{0.0};
        double rcosPow1_rsinPow1{0.0};
    };

    struct SystemNoiseMoments {
        double wvPow1{0.0};
        double wuPow1{0.0};
        double cwuPow1{0.0};
        double swuPow1{0.0};

        double wvPow2{0.0};
        double wuPow2{0.0};
        double swuPow2{0.0};
        double cwuPow2{0.0};
        double cwuPow1_swuPow1{0.0};
    };

    struct ObservationNoiseMoments {
        double wrPow1{0.0};
        double cwaPow1{0.0};
        double swaPow1{0.0};

        double wrPow2{0.0};
        double cwaPow2{0.0};
        double swaPow2{0.0};
        double cwaPow1_swaPow1{0.0};
    };

    struct Controls{
        double v;
        double u;
        double cu;
        double su;
    };

    SimpleVehicleModel() = default;

    Eigen::Vector3d propagate(const Eigen::Vector3d& x_curr, const Eigen::Vector2d& u_curr, const Eigen::Vector2d& system_noise);
    Eigen::Vector2d observe(const Eigen::Vector3d& x_curr, const Eigen::Vector2d& observation_noise, const Eigen::Vector2d& landmark);

    StateMoments propagateStateMoments(const StateMoments & prev_state_moments,
                                       const SystemNoiseMoments& system_noise_moments,
                                       const Controls& control_inputs);

    ObservationMoments getObservationMoments(const ReducedStateMoments & state_moments,
                                             const ObservationNoiseMoments & observation_noise_moments,
                                             const Eigen::Vector2d& landmark);
};

#endif //UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_MODEL_H
