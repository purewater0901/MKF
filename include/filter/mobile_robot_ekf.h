#ifndef UNCERTAINTY_PROPAGATION_MOBILE_ROBOT_EKF_H
#define UNCERTAINTY_PROPAGATION_MOBILE_ROBOT_EKF_H

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <Eigen/Eigen>

#include "model/mobile_robot_model.h"
#include "distribution/base_distribution.h"

class MobileRobotEKF
{
public:
    MobileRobotEKF() = default;

    MobileRobot::StateInfo predict(const MobileRobot::StateInfo& state_info,
                                   const Eigen::Vector2d& inputs,
                                   const double dt,
                                   const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    MobileRobot::StateInfo update(const MobileRobot::StateInfo& state_info,
                                  const Eigen::Vector3d& y,
                                  const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);
};

#endif //UNCERTAINTY_PROPAGATION_MOBILE_ROBOT_EKF_H
