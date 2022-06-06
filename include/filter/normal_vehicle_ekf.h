#ifndef UNCERTAINTY_PROPAGATION_NORMAL_VEHICLE_EKF_H
#define UNCERTAINTY_PROPAGATION_NORMAL_VEHICLE_EKF_H

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <Eigen/Eigen>

#include "model/normal_vehicle_model.h"
#include "distribution/base_distribution.h"

class NormalVehicleEKF
{
public:
    NormalVehicleEKF() = default;

    NormalVehicle::StateInfo predict(const NormalVehicle::StateInfo& state_info,
                                     const Eigen::Vector2d& inputs,
                                     const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);
    NormalVehicle::StateInfo update(const NormalVehicle::StateInfo& state_info,
                                    const Eigen::Vector2d& y,
                                    const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);
};

#endif //UNCERTAINTY_PROPAGATION_NORMAL_VEHICLE_EKF_H
