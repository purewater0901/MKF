#include <iostream>
#include <random>
#include <vector>
#include <Eigen/Eigen>
#include <map>
#include <memory>
#include <gtest/gtest.h>

#include "filter/simple_vehicle_nkf.h"
#include "distribution/uniform_distribution.h"

using namespace SimpleVehicle;

TEST(SimpleVehicleNKF, Update)
{
    SimpleVehicleNKF filter;
    StateInfo state_info;
    state_info.mean = {80.4759, 58.3819, 0.539531};
    state_info.covariance << 0.198053, -0.273244, -0.00354857,
                            -0.273244, 0.377295,  0.00497959,
                            -0.00354857,  0.00497959, 0.000928337;

    // Observation
    Eigen::Vector2d y{std::sqrt(6400+3600), 0.54000};

    // Observation Noise
    const double upper_wr = 10.0;
    const double lower_wr = 0.0;
    const double upper_wtheta = (M_PI/10.0);
    const double lower_wtheta = -(M_PI/10.0);
    std::map<int, std::shared_ptr<BaseDistribution>> observation_noise_map{
            {OBSERVATION_NOISE::IDX::WR, std::make_shared<UniformDistribution>(lower_wr, upper_wr)},
            {OBSERVATION_NOISE::IDX::WTHETA, std::make_shared<UniformDistribution>(lower_wtheta, upper_wtheta)}};

    const auto updated_info = filter.update(state_info, y, observation_noise_map);

    std::cout << updated_info.covariance << std::endl;

}