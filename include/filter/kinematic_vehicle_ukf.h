#ifndef UNCERTAINTY_PROPAGATION_KINEMATIC_VEHICLE_UKF_H
#define UNCERTAINTY_PROPAGATION_KINEMATIC_VEHICLE_UKF_H

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <Eigen/Eigen>

#include "distribution/base_distribution.h"
#include "model/kinematic_vehicle_model.h"

using namespace KinematicVehicle;

class KinematicVehicleUKF
{
public:
    KinematicVehicleUKF();

    StateInfo predict(const StateInfo& state_info,
                      const Eigen::Vector2d & control_inputs,
                      const double dt,
                      const std::map<int, std::shared_ptr<BaseDistribution>>& system_noise_map,
                      const std::map<int, std::shared_ptr<BaseDistribution>>& measurement_noise_map);

    StateInfo update(const KinematicVehicle::StateInfo &state_info,
                     const Eigen::Vector3d &observed_values,
                     const std::map<int, std::shared_ptr<BaseDistribution>>& system_noise_map,
                     const std::map<int, std::shared_ptr<BaseDistribution>>& measurement_noise_map);

    Eigen::MatrixXd sigma_points_;
    const int augmented_size_{9};
    const double alpha_squared_{1.0};
    const double beta_{0.0};
    const double kappa_{0.0};
    const double lambda_;

    double Sigma_WM0_;
    double Sigma_WC0_;
    double Sigma_WMI_;
    double Sigma_WCI_;

    KinematicVehicleModel model_;
};

#endif //UNCERTAINTY_PROPAGATION_KINEMATIC_VEHICLE_UKF_H
