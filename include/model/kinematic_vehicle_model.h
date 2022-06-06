#ifndef UNCERTAINTY_PROPAGATION_KINEMATIC_VEHICLE_MODEL_H
#define UNCERTAINTY_PROPAGATION_KINEMATIC_VEHICLE_MODEL_H

#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Eigen>

namespace KinematicVehicle
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
            R = 0,
            VC = 1,
            YAW = 2
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
            WR = 0,
            WVC = 1,
            WYAW = 2,
        };
    }
}

class KinematicVehicleModel
{
public:
    struct StateMoments {
        double xPow1{0.0};
        double yPow1{0.0};
        double vPow1{0.0};
        double yawPow1{0.0};
        double cPow1{0.0};
        double sPow1{0.0};

        double xPow2{0.0};
        double yPow2{0.0};
        double vPow2{0.0};
        double yawPow2{0.0};
        double cPow2{0.0};
        double sPow2{0.0};
        double xPow1_yPow1{0.0};
        double xPow1_yawPow1{0.0};
        double yPow1_yawPow1{0.0};
        double vPow1_xPow1{0.0};
        double vPow1_yPow1{0.0};
        double vPow1_yawPow1{0.0};
        double vPow1_cPow1{0.0};
        double vPow1_sPow1{0.0};
        double cPow1_xPow1{0.0};
        double sPow1_xPow1{0.0};
        double cPow1_yPow1{0.0};
        double sPow1_yPow1{0.0};
        double cPow1_yawPow1{0.0};
        double sPow1_yawPow1{0.0};
        double cPow1_sPow1{0.0};

        double vPow1_cPow2{0.0};
        double vPow1_sPow2{0.0};
        double vPow2_cPow1{0.0};
        double vPow2_sPow1{0.0};
        double vPow1_cPow1_xPow1{0.0};
        double vPow1_cPow1_yPow1{0.0};
        double vPow1_cPow1_yawPow1{0.0};
        double vPow1_sPow1_xPow1{0.0};
        double vPow1_sPow1_yPow1{0.0};
        double vPow1_sPow1_yawPow1{0.0};
        double vPow1_cPow1_sPow1{0.0};

        double vPow2_cPow2{0.0};
        double vPow2_sPow2{0.0};
        double vPow2_cPow1_sPow1{0.0};
    };

    struct ReducedStateMoments {
        double yawPow1{0.0};

        double xPow2{0.0};
        double yPow2{0.0};
        double yawPow2{0.0};
        double vPow1_cPow1{0.0};

        double vPow1_yawPow1_cPow1{0.0};
        double xPow2_yawPow1{0.0};
        double yPow2_yawPow1{0.0};

        double xPow4{0.0};
        double yPow4{0.0};
        double xPow2_yPow2{0.0};
        double vPow2_cPow2{0.0};
        double xPow2_vPow1_cPow1{0.0};
        double yPow2_vPow1_cPow1{0.0};
    };

    struct ObservationMoments {
        double rPow1{0.0};
        double vcPow1{0.0};
        double yawPow1{0.0};

        double rPow2{0.0};
        double vcPow2{0.0};
        double yawPow2{0.0};
        double rPow1_vcPow1{0.0};
        double rPow1_yawPow1{0.0};
        double vcPow1_yawPow1{0.0};
    };

    struct SystemNoiseMoments {
        double wvPow1{0.0};
        double wyawPow1{0.0};
        double cyawPow1{0.0};
        double syawPow1{0.0};

        double wvPow2{0.0};
        double wyawPow2{0.0};
        double cyawPow2{0.0};
        double syawPow2{0.0};
        double cyawPow1_syawPow1{0.0};
    };

    struct ObservationNoiseMoments {
        double wrPow1{0.0};
        double wvPow1{0.0};
        double wyawPow1{0.0};

        double wrPow2{0.0};
        double wvPow2{0.0};
        double wyawPow2{0.0};
    };

    struct Controls{
        double a{0.0};
        double u{0.0};
        double cu{0.0};
        double su{0.0};
    };

    KinematicVehicleModel() = default;

    Eigen::Vector4d propagate(const Eigen::Vector4d& x_curr,
                              const Eigen::Vector2d& u_curr,
                              const Eigen::Vector2d& system_noise,
                              const double dt);
    Eigen::Vector3d observe(const Eigen::Vector4d& x_curr, const Eigen::Vector3d& observation_noise);

    StateMoments propagateStateMoments(const StateMoments & prev_state_moments,
                                       const SystemNoiseMoments& system_noise_moments,
                                       const Controls& control_inputs,
                                       const double dt);

    ObservationMoments getObservationMoments(const ReducedStateMoments & state_moments,
                                             const ObservationNoiseMoments & observation_noise_moments);
};

#endif //UNCERTAINTY_PROPAGATION_KINEMATIC_VEHICLE_MODEL_H
