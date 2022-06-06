#ifndef UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_NKF_H
#define UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_NKF_H

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <Eigen/Eigen>

#include "model/simple_vehicle_model.h"
#include "distribution/base_distribution.h"

class SimpleVehicleNKF {
public:

    SimpleVehicleNKF();

    SimpleVehicle::StateInfo predict(const SimpleVehicle::StateInfo & state_info,
                                     const Eigen::Vector2d & control_inputs,
                                     const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    SimpleVehicle::StateInfo update(const SimpleVehicle::StateInfo & state_info,
                                    const Eigen::Vector2d & observed_values,
                                    const Eigen::Vector2d & landmark,
                                    const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    SimpleVehicleModel vehicle_model_;
};

#endif //UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_NKF_H
