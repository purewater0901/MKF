#ifndef UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_EKF_H
#define UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_EKF_H

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <Eigen/Eigen>

#include "model/simple_vehicle_model.h"
#include "distribution/base_distribution.h"

class SimpleVehicleEKF
{
public:
    SimpleVehicleEKF() = default;

    SimpleVehicle::StateInfo predict(const SimpleVehicle::StateInfo& state_info,
                                     const Eigen::Vector2d& inputs,
                                     const double dt,
                                     const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    SimpleVehicle::StateInfo update(const SimpleVehicle::StateInfo& state_info,
                                    const Eigen::Vector2d& y_meas,
                                    const Eigen::Vector2d& landmark,
                                    const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);
};

#endif //UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_EKF_H
