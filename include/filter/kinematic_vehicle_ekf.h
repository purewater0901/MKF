#ifndef UNCERTAINTY_PROPAGATION_KINEMATIC_VEHICLE_EKF_H
#define UNCERTAINTY_PROPAGATION_KINEMATIC_VEHICLE_EKF_H

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <Eigen/Eigen>

#include "model/kinematic_vehicle_model.h"
#include "distribution/base_distribution.h"

class KinematicVehicleEKF
{
public:
    KinematicVehicleEKF() = default;

    KinematicVehicle::StateInfo predict(const KinematicVehicle::StateInfo& state_info,
                                        const Eigen::Vector2d& inputs,
                                        const double dt,
                                        const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    KinematicVehicle::StateInfo update(const KinematicVehicle::StateInfo& state_info,
                                       const Eigen::Vector3d& y,
                                       const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);
};

#endif //UNCERTAINTY_PROPAGATION_KINEMATIC_VEHICLE_EKF_H
