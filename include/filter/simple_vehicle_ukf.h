#ifndef UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_UKF_H
#define UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_UKF_H

#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Eigen>
#include <memory>

#include "distribution/base_distribution.h"
#include "model/simple_vehicle_model.h"

class SimpleVehicleUKF
{
public:
    SimpleVehicleUKF();

    SimpleVehicle::StateInfo predict(const SimpleVehicle::StateInfo& state_info,
                                     const Eigen::Vector2d & control_inputs,
                                     const std::map<int, std::shared_ptr<BaseDistribution>>& system_noise_map,
                                     const std::map<int, std::shared_ptr<BaseDistribution>>& measurement_noise_map);

    SimpleVehicle::StateInfo update(const SimpleVehicle::StateInfo &state_info,
                                    const Eigen::Vector2d &observed_values,
                                    const Eigen::Vector2d &landmark,
                                    const std::map<int, std::shared_ptr<BaseDistribution>>& system_noise_map,
                                    const std::map<int, std::shared_ptr<BaseDistribution>>& measurement_noise_map);

    Eigen::MatrixXd sigma_points_;
    const int augmented_size_{10};
    const double alpha_squared_{1.0};
    const double beta_{0.0};
    const double kappa_{0.0};
    const double lambda_;

    double Sigma_WM0_;
    double Sigma_WC0_;
    double Sigma_WMI_;
    double Sigma_WCI_;

    SimpleVehicleModel model_;
};

#endif //UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_UKF_H
