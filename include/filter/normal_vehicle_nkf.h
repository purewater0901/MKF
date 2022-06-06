#ifndef UNCERTAINTY_PROPAGATION_NORMAL_VEHICLE_NKF_H
#define UNCERTAINTY_PROPAGATION_NORMAL_VEHICLE_NKF_H

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <Eigen/Eigen>

#include "model/normal_vehicle_model.h"
#include "distribution/base_distribution.h"

class NormalVehicleNKF {
public:

    NormalVehicleNKF();

    NormalVehicle::StateInfo predict(const NormalVehicle::StateInfo & state_info,
                                     const Eigen::Vector2d & control_inputs,
                                     const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    NormalVehicle::StateInfo update(const NormalVehicle::StateInfo & state_info,
                                    const Eigen::Vector2d & observed_values,
                                    const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    NormalVehicleModel vehicle_model_;
};

#endif //UNCERTAINTY_PROPAGATION_NORMAL_VEHICLE_NKF_H
