#ifndef UNCERTAINTY_PROPAGATION_KINEMATIC_VEHILCE_NKF_H
#define UNCERTAINTY_PROPAGATION_KINEMATIC_VEHILCE_NKF_H

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <Eigen/Eigen>

#include "model/kinematic_vehicle_model.h"
#include "distribution/base_distribution.h"

class KinematicVehicleNKF
{
public:
    KinematicVehicleNKF() = default;

    KinematicVehicle::StateInfo predict(const KinematicVehicle::StateInfo & state_info,
                                        const Eigen::Vector2d & control_inputs,
                                        const double dt,
                                        const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    KinematicVehicle::StateInfo update(const KinematicVehicle::StateInfo & state_info,
                                       const Eigen::Vector3d & observed_values,
                                       const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    KinematicVehicleModel vehicle_model_;

};

#endif //UNCERTAINTY_PROPAGATION_KINEMATIC_VEHILCE_NKF_H
