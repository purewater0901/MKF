#ifndef UNCERTAINTY_PROPAGATION_MOBILE_ROBOT_NKF_H
#define UNCERTAINTY_PROPAGATION_MOBILE_ROBOT_NKF_H

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <Eigen/Eigen>

#include "model/mobile_robot_model.h"
#include "distribution/base_distribution.h"

class MobileRobotNKF
{
public:
    MobileRobotNKF() = default;

    MobileRobot::StateInfo predict(const MobileRobot::StateInfo & state_info,
                                   const Eigen::Vector2d & control_inputs,
                                   const double dt,
                                   const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    MobileRobot::StateInfo update(const MobileRobot::StateInfo & state_info,
                                  const Eigen::Vector3d & observed_values,
                                  const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    MobileRobotModel vehicle_model_;

};

#endif //UNCERTAINTY_PROPAGATION_MOBILE_ROBOT_NKF_H
