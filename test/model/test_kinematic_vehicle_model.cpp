#include <iostream>
#include <random>
#include <vector>
#include <Eigen/Eigen>
#include <map>
#include <memory>
#include <gtest/gtest.h>

#include "model/kinematic_vehicle_model.h"
#include "distribution/uniform_distribution.h"
#include "distribution/normal_distribution.h"
#include "distribution/four_dimensional_normal_distribution.h"

using namespace KinematicVehicle;

TEST(KinematicVehicleModel, propagate)
{
    const double epsilon = 1e-6;

    KinematicVehicleModel model;
    const double dt = 0.1;

    // Input a
    {
        const Eigen::Vector4d x_curr = Eigen::Vector4d::Zero();
        const Eigen::Vector2d input{1.0*dt, 0.0*dt};
        const Eigen::Vector2d system_noise{0.0, 0.0};
        const auto x_next = model.propagate(x_curr, input, system_noise, dt);

        EXPECT_NEAR(x_next(STATE::IDX::X), 0.0, epsilon);
        EXPECT_NEAR(x_next(STATE::IDX::Y), 0.0, epsilon);
        EXPECT_NEAR(x_next(STATE::IDX::V), 0.1, epsilon);
        EXPECT_NEAR(x_next(STATE::IDX::YAW), 0.0, epsilon);
    }

    // Input u
    {
        const Eigen::Vector4d x_curr = Eigen::Vector4d::Zero();
        const Eigen::Vector2d input{0.0*dt, 1.0*dt};
        const Eigen::Vector2d system_noise{0.0, 0.0};
        const auto x_next = model.propagate(x_curr, input, system_noise, dt);

        EXPECT_NEAR(x_next(STATE::IDX::X), 0.0, epsilon);
        EXPECT_NEAR(x_next(STATE::IDX::Y), 0.0, epsilon);
        EXPECT_NEAR(x_next(STATE::IDX::V), 0.0, epsilon);
        EXPECT_NEAR(x_next(STATE::IDX::YAW), 0.1, epsilon);
    }

    // Input a, u
    {
        const Eigen::Vector4d x_curr = Eigen::Vector4d::Zero();
        const Eigen::Vector2d input{2.0*dt, 3.0*dt};
        const Eigen::Vector2d system_noise{0.0, 0.0};
        const auto x_next = model.propagate(x_curr, input, system_noise, dt);

        EXPECT_NEAR(x_next(STATE::IDX::X), 0.0, epsilon);
        EXPECT_NEAR(x_next(STATE::IDX::Y), 0.0, epsilon);
        EXPECT_NEAR(x_next(STATE::IDX::V), 0.2, epsilon);
        EXPECT_NEAR(x_next(STATE::IDX::YAW), 0.3, epsilon);
    }

    // Input random start
    {
        const Eigen::Vector4d x_curr{0.0, 0.0, 3.0, M_PI/6.0};
        const Eigen::Vector2d input{2.0*dt, 3.0*dt};
        const Eigen::Vector2d system_noise{0.0, 0.0};
        const auto x_next = model.propagate(x_curr, input, system_noise, dt);

        EXPECT_NEAR(x_next(STATE::IDX::X), 3.0*std::cos(M_PI/6.0)*dt, epsilon);
        EXPECT_NEAR(x_next(STATE::IDX::Y), 3.0*std::sin(M_PI/6.0)*dt, epsilon);
        EXPECT_NEAR(x_next(STATE::IDX::V), 3.2, epsilon);
        EXPECT_NEAR(x_next(STATE::IDX::YAW), M_PI/6.0 + 0.3, epsilon);
    }

    // Input random start with noise
    {
        const Eigen::Vector4d x_curr{0.0, 0.0, 3.0, M_PI/6.0};
        const Eigen::Vector2d input{2.0*dt, 3.0*dt};
        const Eigen::Vector2d system_noise{0.5, 0.1};
        const auto x_next = model.propagate(x_curr, input, system_noise, dt);

        EXPECT_NEAR(x_next(STATE::IDX::X), 3.0*std::cos(M_PI/6.0)*dt, epsilon);
        EXPECT_NEAR(x_next(STATE::IDX::Y), 3.0*std::sin(M_PI/6.0)*dt, epsilon);
        EXPECT_NEAR(x_next(STATE::IDX::V), 3.7, epsilon);
        EXPECT_NEAR(x_next(STATE::IDX::YAW), M_PI/6.0 + 0.4, epsilon);
    }
}

TEST(KinematicVehicleModel, observe)
{
    const double epsilon = 1e-6;

    KinematicVehicleModel model;

    {
        const Eigen::Vector4d x_curr = Eigen::Vector4d::Zero();
        const Eigen::Vector3d observation_noise{0.0, 0.0, 0.0};
        const auto y = model.observe(x_curr, observation_noise);

        EXPECT_NEAR(y(OBSERVATION::IDX::R), 0.0, epsilon);
        EXPECT_NEAR(y(OBSERVATION::IDX::VC), 0.0, epsilon);
    }

    {
        const Eigen::Vector4d x_curr = Eigen::Vector4d::Zero();
        const Eigen::Vector3d observation_noise{10.2, 0.1, 0.0};
        const auto y = model.observe(x_curr, observation_noise);

        EXPECT_NEAR(y(OBSERVATION::IDX::R), 10.2, epsilon);
        EXPECT_NEAR(y(OBSERVATION::IDX::VC), 0.1, epsilon);
    }

    {
        const Eigen::Vector4d x_curr{10.0, 10.0, 5.0, M_PI/4.0};
        const Eigen::Vector3d observation_noise{10.2, 0.1, 0.0};
        const auto y = model.observe(x_curr, observation_noise);

        EXPECT_NEAR(y(OBSERVATION::IDX::R), 210.2, epsilon);
        EXPECT_NEAR(y(OBSERVATION::IDX::VC), 5.0*std::cos(M_PI/4.0)+0.1, epsilon);
    }
}

/*
TEST(KinematicVehicleModel, propagateStateMoments)
{
    const double epsilon = 0.001;

    KinematicVehicleModel model;
    const double dt = 0.1;
    // Normal Distribution
    {
        // state
        NormalDistribution x_dist(0.0, 1.0);
        NormalDistribution y_dist(0.0, 1.0);
        NormalDistribution v_dist(5.0, 0.5*0.5);
        NormalDistribution yaw_dist(0.0, (M_PI/10.0)*(M_PI/10.0));

        // system noise
        NormalDistribution wv_dist(0.0, std::pow(1.0*dt, 2));
        NormalDistribution wyaw_dist(0.0, std::pow(0.1*dt, 2));

        // Input
        KinematicVehicleModel::Controls inputs;
        inputs.a = 0.0;
        inputs.u = 0.0;
        inputs.cu = std::cos(inputs.u);
        inputs.su = std::sin(inputs.u);

        KinematicVehicleModel::StateMoments state_moments;
        state_moments.xPow1 = x_dist.calc_moment(1);
        state_moments.yPow1 = y_dist.calc_moment(1);
        state_moments.vPow1 = v_dist.calc_moment(1);
        state_moments.yawPow1 = yaw_dist.calc_moment(1);
        state_moments.cPow1 = yaw_dist.calc_cos_moment(1);
        state_moments.sPow1 = yaw_dist.calc_sin_moment(1);

        state_moments.xPow2 = x_dist.calc_moment(2);
        state_moments.yPow2 = y_dist.calc_moment(2);
        state_moments.vPow2 = v_dist.calc_moment(2);
        state_moments.yawPow2 = yaw_dist.calc_moment(2);
        state_moments.cPow2 = yaw_dist.calc_cos_moment(2);
        state_moments.sPow2 = yaw_dist.calc_sin_moment(2);
        state_moments.xPow1_yPow1 = x_dist.calc_moment(1) * y_dist.calc_moment(1);
        state_moments.xPow1_yawPow1 = x_dist.calc_moment(1) * yaw_dist.calc_moment(1);
        state_moments.yPow1_yawPow1 = y_dist.calc_moment(1) * yaw_dist.calc_moment(1);
        state_moments.vPow1_xPow1 = v_dist.calc_moment(1) * x_dist.calc_moment(1);
        state_moments.vPow1_yPow1 = v_dist.calc_moment(1) * y_dist.calc_moment(1);
        state_moments.vPow1_yawPow1 = v_dist.calc_moment(1) * yaw_dist.calc_moment(1);
        state_moments.vPow1_cPow1 = v_dist.calc_moment(1) * yaw_dist.calc_cos_moment(1);
        state_moments.vPow1_sPow1 = v_dist.calc_moment(1) * yaw_dist.calc_sin_moment(1);
        state_moments.cPow1_xPow1 = yaw_dist.calc_cos_moment(1) * x_dist.calc_moment(1);
        state_moments.sPow1_xPow1 = yaw_dist.calc_sin_moment(1) * x_dist.calc_moment(1);
        state_moments.cPow1_yPow1 = yaw_dist.calc_cos_moment(1) * y_dist.calc_moment(1);
        state_moments.sPow1_yPow1 = yaw_dist.calc_sin_moment(1) * y_dist.calc_moment(1);
        state_moments.cPow1_yawPow1 = yaw_dist.calc_x_cos_moment(1, 1);
        state_moments.sPow1_yawPow1 = yaw_dist.calc_x_sin_moment(1, 1);
        state_moments.cPow1_sPow1 = yaw_dist.calc_cos_sin_moment(1, 1);

        state_moments.vPow1_cPow2 = v_dist.calc_moment(1) * yaw_dist.calc_cos_moment(2);
        state_moments.vPow1_sPow2 = v_dist.calc_moment(1) * yaw_dist.calc_sin_moment(2);
        state_moments.vPow2_cPow1 = v_dist.calc_moment(2) * yaw_dist.calc_cos_moment(1);
        state_moments.vPow2_sPow1 = v_dist.calc_moment(2) * yaw_dist.calc_sin_moment(1);
        state_moments.vPow1_cPow1_xPow1 = v_dist.calc_moment(1) * yaw_dist.calc_cos_moment(1) * x_dist.calc_moment(1);
        state_moments.vPow1_cPow1_yPow1 = v_dist.calc_moment(1) * yaw_dist.calc_cos_moment(1) * y_dist.calc_moment(1);
        state_moments.vPow1_cPow1_yawPow1 = v_dist.calc_moment(1) * yaw_dist.calc_x_cos_moment(1, 1);
        state_moments.vPow1_sPow1_xPow1 = v_dist.calc_moment(1) * yaw_dist.calc_sin_moment(1) * x_dist.calc_moment(1);
        state_moments.vPow1_sPow1_yPow1 = v_dist.calc_moment(1) * yaw_dist.calc_sin_moment(1) * y_dist.calc_moment(1);
        state_moments.vPow1_sPow1_yawPow1 = v_dist.calc_moment(1) * yaw_dist.calc_x_sin_moment(1, 1);
        state_moments.vPow1_cPow1_sPow1 = v_dist.calc_moment(1) * yaw_dist.calc_cos_sin_moment(1, 1);

        state_moments.vPow2_cPow2 = v_dist.calc_moment(2)*yaw_dist.calc_cos_moment(2);
        state_moments.vPow2_sPow2 = v_dist.calc_moment(2)*yaw_dist.calc_sin_moment(2);
        state_moments.vPow2_cPow1_sPow1 = v_dist.calc_moment(2) * yaw_dist.calc_cos_sin_moment(1, 1);

        KinematicVehicleModel::SystemNoiseMoments system_noise_moments;
        system_noise_moments.wvPow1 = wv_dist.calc_moment(1);
        system_noise_moments.wyawPow1 = wyaw_dist.calc_moment(1);
        system_noise_moments.cyawPow1 = wyaw_dist.calc_cos_moment(1);
        system_noise_moments.syawPow1 = wyaw_dist.calc_sin_moment(1);

        system_noise_moments.wvPow2 = wv_dist.calc_moment(2);
        system_noise_moments.wyawPow2 = wyaw_dist.calc_moment(2);
        system_noise_moments.cyawPow2 = wyaw_dist.calc_cos_moment(2);
        system_noise_moments.syawPow2 = wyaw_dist.calc_sin_moment(2);
        system_noise_moments.cyawPow1_syawPow1 = wyaw_dist.calc_cos_sin_moment(1, 1);

        const auto next_state_moments = model.propagateStateMoments(state_moments, system_noise_moments, inputs, dt);

        // Monte Carlo
        KinematicVehicleModel::StateMoments montecarlo_next_state_moments;
        {
            std::random_device seed_gen;
            std::default_random_engine engine(seed_gen());

            const size_t num_of_samples = 10000 * 10000;
            std::normal_distribution<double> x_dist_monte(0.0, 1.0);
            std::normal_distribution<double> y_dist_monte(0.0, 1.0);
            std::normal_distribution<double> v_dist_monte(5.0, 0.5);
            std::normal_distribution<double> yaw_dist_monte(0.0, M_PI/10.0);
            std::normal_distribution<double> wv_dist_monte(0.0, 1.0*dt);
            std::normal_distribution<double> wyaw_dist_monte(0.0, 0.1*dt);

            double sum_cPowsPow = 0.0;
            double sum_cPow2 = 0.0;
            double sum_sPow2 = 0.0;
            double sum_cyawPow2 = 0.0;
            double sum_syawPow2 = 0.0;
            double sum_cYawPow1sYawPow1 = 0.0;

            for(size_t i=0; i<num_of_samples; ++i) {
                const auto x = x_dist_monte(engine);
                const auto y = y_dist_monte(engine);
                const auto v = v_dist_monte(engine);
                const auto yaw = yaw_dist_monte(engine);
                const auto wv = wv_dist_monte(engine);
                const auto wyaw = wyaw_dist_monte(engine);

                sum_cPowsPow += std::cos(yaw) * std::sin(yaw);
                sum_cPow2 += std::cos(yaw) * std::cos(yaw);
                sum_sPow2 += std::sin(yaw) * std::sin(yaw);
                sum_cyawPow2 += std::cos(wyaw) * std::cos(wyaw);
                sum_syawPow2 += std::sin(wyaw) * std::sin(wyaw);
                sum_cYawPow1sYawPow1 += std::cos(wyaw) * std::sin(wyaw);

                const auto x_next = model.propagate({x,y,v,yaw}, {inputs.a*dt, inputs.u*dt}, {wv, wyaw}, dt);
                montecarlo_next_state_moments.xPow1 += x_next[0];
                montecarlo_next_state_moments.yPow1 += x_next[1];
                montecarlo_next_state_moments.vPow1 += x_next[2];
                montecarlo_next_state_moments.yawPow1 += x_next[3];
                montecarlo_next_state_moments.cPow1 += std::cos(x_next[3]);
                montecarlo_next_state_moments.sPow1 += std::sin(x_next[3]);

                montecarlo_next_state_moments.xPow2 += x_next[0] * x_next[0];
                montecarlo_next_state_moments.yPow2 += x_next[1] * x_next[1];
                montecarlo_next_state_moments.vPow2 += x_next[2] * x_next[2];
                montecarlo_next_state_moments.yawPow2 += x_next[3] * x_next[3];
                montecarlo_next_state_moments.cPow2 += std::cos(x_next[3]) * std::cos(x_next[3]);
                montecarlo_next_state_moments.sPow2 += std::sin(x_next[3]) * std::sin(x_next[3]);
                montecarlo_next_state_moments.xPow1_yPow1 = x_next[0] * x_next[1];
                montecarlo_next_state_moments.xPow1_yawPow1 += x_next[0] * x_next[3];
                montecarlo_next_state_moments.yPow1_yawPow1 += x_next[1] * x_next[3];
                montecarlo_next_state_moments.vPow1_xPow1 += x_next[0] * x_next[2];
                montecarlo_next_state_moments.vPow1_yPow1 += x_next[1] * x_next[2];
                montecarlo_next_state_moments.vPow1_yawPow1 += x_next[2] * x_next[3];
                montecarlo_next_state_moments.vPow1_cPow1 += x_next[2] * std::cos(x_next[3]);
                montecarlo_next_state_moments.vPow1_sPow1 += x_next[2] * std::sin(x_next[3]);
                montecarlo_next_state_moments.cPow1_xPow1 += x_next[0] * std::cos(x_next[3]);
                montecarlo_next_state_moments.sPow1_xPow1 += x_next[0] * std::sin(x_next[3]);
                montecarlo_next_state_moments.cPow1_yPow1 += x_next[1] * std::cos(x_next[3]);
                montecarlo_next_state_moments.sPow1_yPow1 += x_next[1] * std::sin(x_next[3]);
                montecarlo_next_state_moments.cPow1_yawPow1 += x_next[3] * std::cos(x_next[3]);
                montecarlo_next_state_moments.sPow1_yawPow1 += x_next[3] * std::sin(x_next[3]);
                montecarlo_next_state_moments.cPow1_sPow1 += std::cos(x_next[3]) * std::sin(x_next[3]);

                montecarlo_next_state_moments.vPow1_cPow2 += x_next[2] * std::pow(std::cos(x_next[3]), 2);
                montecarlo_next_state_moments.vPow1_sPow2 += x_next[2] * std::pow(std::sin(x_next[3]), 2);
                montecarlo_next_state_moments.vPow2_cPow1 += x_next[2] * x_next[2] * std::cos(x_next[3]);
                montecarlo_next_state_moments.vPow2_sPow1 += x_next[2] * x_next[2] * std::cos(x_next[3]);
                montecarlo_next_state_moments.vPow1_cPow1_xPow1 += x_next[0] * x_next[2] * std::cos(x_next[3]);
                montecarlo_next_state_moments.vPow1_cPow1_yPow1 += x_next[1] * x_next[2] * std::cos(x_next[3]);
                montecarlo_next_state_moments.vPow1_cPow1_yawPow1 += x_next[3] * x_next[2] * std::cos(x_next[3]);
                montecarlo_next_state_moments.vPow1_sPow1_xPow1 += x_next[0] * x_next[2] * std::sin(x_next[3]);
                montecarlo_next_state_moments.vPow1_sPow1_yPow1 += x_next[1] * x_next[2] * std::sin(x_next[3]);
                montecarlo_next_state_moments.vPow1_sPow1_yawPow1 += x_next[3] * x_next[2] * std::sin(x_next[3]);
                montecarlo_next_state_moments.vPow1_cPow1_sPow1 += x_next[2] * std::cos(x_next[3]) * std::sin(x_next[3]);

                montecarlo_next_state_moments.vPow2_cPow2 += x_next[2] * x_next[2] * std::cos(x_next[3]) * std::cos(x_next[3]);
                montecarlo_next_state_moments.vPow2_sPow2 += x_next[2] * x_next[2] * std::sin(x_next[3]) * std::sin(x_next[3]);
                montecarlo_next_state_moments.vPow2_cPow1_sPow1 += x_next[2] * x_next[2] * std::cos(x_next[3]) * std::sin(x_next[3]);
            }

            std::cout << "E[cPowsPow]: " << sum_cPowsPow / num_of_samples << "  True: " << state_moments.cPow1_sPow1 << std::endl;
            std::cout << "E[cPow2]: " << sum_cPow2 / num_of_samples << " True: " << state_moments.cPow2 << std::endl;
            std::cout << "E[sPow2]: " << sum_sPow2 / num_of_samples << " True: " << state_moments.sPow2 << std::endl;
            std::cout << "E[cyawPow2]: " << sum_cyawPow2 / num_of_samples << " True: " << system_noise_moments.cyawPow2 << std::endl;
            std::cout << "E[syawPow2]: " << sum_syawPow2 / num_of_samples << " True: " << system_noise_moments.syawPow2 << std::endl;
            std::cout << "E[cyawPow1_syawPow1]: " << sum_cYawPow1sYawPow1 / num_of_samples << " True: " << system_noise_moments.cyawPow1_syawPow1<< std::endl;

            montecarlo_next_state_moments.xPow1 /= num_of_samples;
            montecarlo_next_state_moments.yPow1 /= num_of_samples;
            montecarlo_next_state_moments.vPow1 /= num_of_samples;
            montecarlo_next_state_moments.yawPow1 /= num_of_samples;
            montecarlo_next_state_moments.cPow1 /= num_of_samples;
            montecarlo_next_state_moments.sPow1 /= num_of_samples;

            montecarlo_next_state_moments.xPow2 /= num_of_samples;
            montecarlo_next_state_moments.yPow2 /= num_of_samples;
            montecarlo_next_state_moments.vPow2 /= num_of_samples;
            montecarlo_next_state_moments.yawPow2 /= num_of_samples;
            montecarlo_next_state_moments.cPow2 /= num_of_samples;
            montecarlo_next_state_moments.sPow2 /= num_of_samples;
            montecarlo_next_state_moments.xPow1_yPow1 /= num_of_samples;
            montecarlo_next_state_moments.xPow1_yawPow1 /= num_of_samples;
            montecarlo_next_state_moments.yPow1_yawPow1 /= num_of_samples;
            montecarlo_next_state_moments.vPow1_xPow1 /= num_of_samples;
            montecarlo_next_state_moments.vPow1_yPow1 /= num_of_samples;
            montecarlo_next_state_moments.vPow1_yawPow1 /= num_of_samples;
            montecarlo_next_state_moments.vPow1_cPow1 /= num_of_samples;
            montecarlo_next_state_moments.vPow1_sPow1 /= num_of_samples;
            montecarlo_next_state_moments.cPow1_xPow1 /= num_of_samples;
            montecarlo_next_state_moments.sPow1_xPow1 /= num_of_samples;
            montecarlo_next_state_moments.cPow1_yPow1 /= num_of_samples;
            montecarlo_next_state_moments.sPow1_yPow1 /= num_of_samples;
            montecarlo_next_state_moments.cPow1_yawPow1 /= num_of_samples;
            montecarlo_next_state_moments.sPow1_yawPow1 /= num_of_samples;
            montecarlo_next_state_moments.cPow1_sPow1 /= num_of_samples;

            montecarlo_next_state_moments.vPow1_cPow2 /= num_of_samples;
            montecarlo_next_state_moments.vPow1_sPow2 /= num_of_samples;
            montecarlo_next_state_moments.vPow2_cPow1 /= num_of_samples;
            montecarlo_next_state_moments.vPow2_sPow1 /= num_of_samples;
            montecarlo_next_state_moments.vPow1_cPow1_xPow1 /= num_of_samples;
            montecarlo_next_state_moments.vPow1_cPow1_yPow1 /= num_of_samples;
            montecarlo_next_state_moments.vPow1_cPow1_yawPow1 /= num_of_samples;
            montecarlo_next_state_moments.vPow1_sPow1_xPow1 /= num_of_samples;
            montecarlo_next_state_moments.vPow1_sPow1_yPow1 /= num_of_samples;
            montecarlo_next_state_moments.vPow1_sPow1_yawPow1 /= num_of_samples;
            montecarlo_next_state_moments.vPow1_cPow1_sPow1 /= num_of_samples;

            montecarlo_next_state_moments.vPow2_cPow2 /= num_of_samples;
            montecarlo_next_state_moments.vPow2_sPow2 /= num_of_samples;
            montecarlo_next_state_moments.vPow2_cPow1_sPow1 /= num_of_samples;
        }

        EXPECT_NEAR(next_state_moments.xPow1, montecarlo_next_state_moments.xPow1, epsilon);
        EXPECT_NEAR(next_state_moments.yPow1, montecarlo_next_state_moments.yPow1, epsilon);
        EXPECT_NEAR(next_state_moments.vPow1, montecarlo_next_state_moments.vPow1, epsilon);
        EXPECT_NEAR(next_state_moments.yawPow1, montecarlo_next_state_moments.yawPow1, epsilon);
        EXPECT_NEAR(next_state_moments.cPow1, montecarlo_next_state_moments.cPow1, epsilon);
        EXPECT_NEAR(next_state_moments.sPow1, montecarlo_next_state_moments.sPow1, epsilon);

        EXPECT_NEAR(next_state_moments.xPow2, montecarlo_next_state_moments.xPow2, epsilon);
        EXPECT_NEAR(next_state_moments.yPow2, montecarlo_next_state_moments.yPow2, epsilon);
        EXPECT_NEAR(next_state_moments.vPow2, montecarlo_next_state_moments.vPow2, epsilon);
        EXPECT_NEAR(next_state_moments.yawPow2, montecarlo_next_state_moments.yawPow2, epsilon);
        EXPECT_NEAR(next_state_moments.cPow2, montecarlo_next_state_moments.cPow2, epsilon);
        EXPECT_NEAR(next_state_moments.sPow2, montecarlo_next_state_moments.sPow2, epsilon);
        EXPECT_NEAR(next_state_moments.xPow1_yPow1, montecarlo_next_state_moments.xPow1_yPow1, epsilon);
        EXPECT_NEAR(next_state_moments.xPow1_yawPow1, montecarlo_next_state_moments.xPow1_yawPow1, epsilon);
        EXPECT_NEAR(next_state_moments.yPow1_yawPow1, montecarlo_next_state_moments.yPow1_yawPow1, epsilon);
        EXPECT_NEAR(next_state_moments.vPow1_xPow1, montecarlo_next_state_moments.vPow1_xPow1, epsilon);
        EXPECT_NEAR(next_state_moments.vPow1_yPow1, montecarlo_next_state_moments.vPow1_yPow1, epsilon);
        EXPECT_NEAR(next_state_moments.vPow1_yawPow1, montecarlo_next_state_moments.vPow1_yawPow1, epsilon);
        EXPECT_NEAR(next_state_moments.vPow1_cPow1, montecarlo_next_state_moments.vPow1_cPow1, epsilon);
        EXPECT_NEAR(next_state_moments.vPow1_sPow1, montecarlo_next_state_moments.vPow1_sPow1, epsilon);
        EXPECT_NEAR(next_state_moments.cPow1_xPow1, montecarlo_next_state_moments.cPow1_xPow1, epsilon);
        EXPECT_NEAR(next_state_moments.sPow1_xPow1, montecarlo_next_state_moments.sPow1_xPow1, epsilon);
        EXPECT_NEAR(next_state_moments.cPow1_yPow1, montecarlo_next_state_moments.cPow1_yPow1, epsilon);
        EXPECT_NEAR(next_state_moments.sPow1_yPow1, montecarlo_next_state_moments.sPow1_yPow1, epsilon);
        EXPECT_NEAR(next_state_moments.cPow1_yawPow1, montecarlo_next_state_moments.cPow1_yawPow1, epsilon);
        EXPECT_NEAR(next_state_moments.sPow1_yawPow1, montecarlo_next_state_moments.sPow1_yawPow1, epsilon);
        EXPECT_NEAR(next_state_moments.cPow1_sPow1, montecarlo_next_state_moments.cPow1_sPow1, epsilon);

        EXPECT_NEAR(next_state_moments.vPow1_cPow2, montecarlo_next_state_moments.vPow1_cPow2, epsilon);
        EXPECT_NEAR(next_state_moments.vPow1_sPow2, montecarlo_next_state_moments.vPow1_sPow2, epsilon);
        EXPECT_NEAR(next_state_moments.vPow2_cPow1, montecarlo_next_state_moments.vPow2_cPow1, epsilon);
        EXPECT_NEAR(next_state_moments.vPow2_sPow1, montecarlo_next_state_moments.vPow2_sPow1, epsilon);
        EXPECT_NEAR(next_state_moments.vPow1_cPow1_xPow1, montecarlo_next_state_moments.vPow1_cPow1_xPow1, epsilon);
        EXPECT_NEAR(next_state_moments.vPow1_cPow1_yPow1, montecarlo_next_state_moments.vPow1_cPow1_yPow1, epsilon);
        EXPECT_NEAR(next_state_moments.vPow1_cPow1_yawPow1, montecarlo_next_state_moments.vPow1_cPow1_yawPow1, epsilon);
        EXPECT_NEAR(next_state_moments.vPow1_sPow1_xPow1, montecarlo_next_state_moments.vPow1_sPow1_xPow1, epsilon);
        EXPECT_NEAR(next_state_moments.vPow1_sPow1_yPow1, montecarlo_next_state_moments.vPow1_sPow1_yPow1, epsilon);
        EXPECT_NEAR(next_state_moments.vPow1_sPow1_yawPow1, montecarlo_next_state_moments.vPow1_sPow1_yawPow1, epsilon);
        EXPECT_NEAR(next_state_moments.vPow1_cPow1_sPow1, montecarlo_next_state_moments.vPow1_cPow1_sPow1, epsilon);

        EXPECT_NEAR(next_state_moments.vPow2_cPow2,  montecarlo_next_state_moments.vPow2_cPow2, epsilon);
        EXPECT_NEAR(next_state_moments.vPow2_sPow2,  montecarlo_next_state_moments.vPow2_sPow2, epsilon);
        EXPECT_NEAR(next_state_moments.vPow2_cPow1_sPow1, montecarlo_next_state_moments.vPow2_cPow1_sPow1, epsilon);
    }
}
 */

TEST(KinematicVehicleModel, getObservationMoments_indpendent)
{
    const double epsilon = 0.01;
    KinematicVehicleModel model;

    const Eigen::Vector4d mean = {3.5, 5.5, 3.0, 0.8};
    Eigen::Matrix4d cov;
    cov <<0.5, 0.0, 0.0, 0.0,
          0.0, 0.5, 0.0, 0.0,
          0.0, 0.0, 0.05, 0.0,
          0.0, 0.0, 0.0, 0.1;

    FourDimensionalNormalDistribution dist(mean, cov);
    KinematicVehicleModel::ReducedStateMoments reduced_moments;
    reduced_moments.yawPow1 = dist.calc_moment(STATE::IDX::YAW, 1);
    reduced_moments.xPow2 = dist.calc_moment(STATE::IDX::X, 2);
    reduced_moments.yPow2 = dist.calc_moment(STATE::IDX::Y, 2);
    reduced_moments.yawPow2 = dist.calc_moment(STATE::IDX::YAW, 2);
    reduced_moments.vPow1_cPow1 = dist.calc_x_cos_z_moment(STATE::IDX::V, STATE::IDX::YAW);

    reduced_moments.vPow1_yawPow1_cPow1 = dist.calc_xy_cos_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    reduced_moments.xPow2_yawPow1 = dist.calc_cross_third_moment(STATE::IDX::X, STATE::IDX::YAW, 2, 1);
    reduced_moments.yPow2_yawPow1 = dist.calc_cross_third_moment(STATE::IDX::Y, STATE::IDX::YAW, 2, 1);

    reduced_moments.xPow4 = dist.calc_moment(STATE::IDX::X, 4);
    reduced_moments.yPow4 = dist.calc_moment(STATE::IDX::Y, 4);
    reduced_moments.xPow2_yPow2 = dist.calc_xxyy_moment(STATE::IDX::X, STATE::IDX::Y);
    reduced_moments.vPow2_cPow2 = dist.calc_xx_cos_y_cos_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    reduced_moments.xPow2_vPow1_cPow1 = dist.calc_xxy_cos_z_moment(STATE::IDX::X, STATE::IDX::V, STATE::IDX::YAW);
    reduced_moments.yPow2_vPow1_cPow1 = dist.calc_xxy_cos_z_moment(STATE::IDX::Y, STATE::IDX::V, STATE::IDX::YAW);

    {
        std::cout << "Exact E[vcos(theta)]: " << reduced_moments.vPow1_cPow1 << std::endl;
        NormalDistribution tmp_yaw_dist(0.8, 0.1);
        std::cout << "E[cos(theta)]: " << tmp_yaw_dist.calc_cos_moment(1) << std::endl;
        NormalDistribution tmp_x_dist(5.5, 0.5);
        std::cout << "E[y^2]: " << tmp_x_dist.calc_moment(2) << std::endl;
        NormalDistribution tmp_v_dist(3.0, 0.05);
        std::cout << "E[v]: " << tmp_v_dist.calc_moment(1) << std::endl;
    }

    // Step2. Create Observation Noise
    auto wr_dist = NormalDistribution(100, 10.5*10.5);
    auto wv_dist = NormalDistribution(0.0, 0.3*0.3);
    auto wyaw_dist = NormalDistribution(0.0, M_PI*M_PI/100);
    KinematicVehicleModel::ObservationNoiseMoments observation_noise;
    observation_noise.wrPow1 = wr_dist.calc_moment(1);
    observation_noise.wrPow2 = wr_dist.calc_moment(2);
    observation_noise.wvPow1 = wv_dist.calc_moment(1);
    observation_noise.wvPow2 = wv_dist.calc_moment(2);
    observation_noise.wyawPow1 = wyaw_dist.calc_moment(1);
    observation_noise.wyawPow2 = wyaw_dist.calc_moment(2);

    // Step3. Get Observation Moments
    const auto observation_moments = model.getObservationMoments(reduced_moments, observation_noise);

    // Monte Carlo
    KinematicVehicleModel::ObservationMoments montecarlo_observation_moments;
    {
        std::random_device seed_gen;
        std::default_random_engine engine(seed_gen());

        const size_t num_of_samples = 10000 * 10000;
        std::normal_distribution<double> x_dist_monte(mean(0), std::sqrt(cov(0,0)));
        std::normal_distribution<double> y_dist_monte(mean(1), std::sqrt(cov(1,1)));
        std::normal_distribution<double> v_dist_monte(mean(2), std::sqrt(cov(2,2)));
        std::normal_distribution<double> yaw_dist_monte(mean(3), std::sqrt(cov(3,3)));
        std::normal_distribution<double> wr_dist_monte(100.0, 10.5);
        std::normal_distribution<double> wvc_dist_monte(0.0, 0.3);
        std::normal_distribution<double> wyaw_dist_monte(0.0, M_PI/10.0);

        double sum_xPow2_vcPow1 = 0.0;
        double sum_yPow2_vcPow1 = 0.0;
        double sum_vcPow1 = 0.0;
        for(size_t i=0; i<num_of_samples; ++i) {
            const auto x = x_dist_monte(engine);
            const auto y = y_dist_monte(engine);
            const auto v = v_dist_monte(engine);
            const auto yaw = yaw_dist_monte(engine);
            const auto wr = wr_dist_monte(engine);
            const auto wvc = wvc_dist_monte(engine);
            const auto wyaw = wyaw_dist_monte(engine);

            sum_xPow2_vcPow1 += x*x*v*std::cos(yaw);
            sum_yPow2_vcPow1 += y*y*v*std::cos(yaw);
            sum_vcPow1 += v*std::cos(yaw);

            const auto y_observe = model.observe({x,y,v,yaw}, {wr, wvc, wyaw});

            montecarlo_observation_moments.rPow1 += y_observe[0];
            montecarlo_observation_moments.vcPow1 += y_observe[1];
            montecarlo_observation_moments.yawPow1 += y_observe[2];
            montecarlo_observation_moments.rPow2 += y_observe[0]*y_observe[0];
            montecarlo_observation_moments.vcPow2 += y_observe[1]*y_observe[1];
            montecarlo_observation_moments.yawPow2 += y_observe[2]*y_observe[2];
            montecarlo_observation_moments.rPow1_vcPow1 += y_observe[0] * y_observe[1];
            montecarlo_observation_moments.rPow1_yawPow1 += y_observe[0] * y_observe[2];
            montecarlo_observation_moments.vcPow1_yawPow1 += y_observe[1] * y_observe[2];
        }

        std::cout << "Montecarlo E[x^2vcos(theta)]: " << sum_xPow2_vcPow1/num_of_samples << std::endl;
        std::cout << "Exact E[x^2vcos(theta)]: " << reduced_moments.xPow2_vPow1_cPow1 << std::endl;
        std::cout << "Montecarlo E[y^2vcos(theta)]: " << sum_yPow2_vcPow1/num_of_samples << std::endl;
        std::cout << "Exact E[y^2vcos(theta)]: " << reduced_moments.yPow2_vPow1_cPow1 << std::endl;
        std::cout << "Montecarlo E[vcos(theta)]: " << sum_vcPow1/num_of_samples << std::endl;
        std::cout << "Exact E[vcos(theta)]: " << reduced_moments.vPow1_cPow1 << std::endl;

        montecarlo_observation_moments.rPow1 /= num_of_samples;
        montecarlo_observation_moments.vcPow1 /= num_of_samples;
        montecarlo_observation_moments.yawPow1 /= num_of_samples;
        montecarlo_observation_moments.rPow2 /= num_of_samples;
        montecarlo_observation_moments.vcPow2 /= num_of_samples;
        montecarlo_observation_moments.yawPow2 /= num_of_samples;
        montecarlo_observation_moments.rPow1_vcPow1/= num_of_samples;
        montecarlo_observation_moments.rPow1_yawPow1/= num_of_samples;
        montecarlo_observation_moments.vcPow1_yawPow1/= num_of_samples;
    }

    EXPECT_NEAR(observation_moments.rPow1, montecarlo_observation_moments.rPow1, epsilon);
    EXPECT_NEAR(observation_moments.vcPow1, montecarlo_observation_moments.vcPow1, epsilon);
    EXPECT_NEAR(observation_moments.yawPow1, montecarlo_observation_moments.yawPow1, epsilon);
    EXPECT_NEAR(observation_moments.rPow2, montecarlo_observation_moments.rPow2, 1.0);
    EXPECT_NEAR(observation_moments.vcPow2, montecarlo_observation_moments.vcPow2, epsilon);
    EXPECT_NEAR(observation_moments.yawPow2, montecarlo_observation_moments.yawPow2, epsilon);
    EXPECT_NEAR(observation_moments.rPow1_vcPow1, montecarlo_observation_moments.rPow1_vcPow1, epsilon);
    EXPECT_NEAR(observation_moments.rPow1_yawPow1, montecarlo_observation_moments.rPow1_yawPow1, epsilon);
    EXPECT_NEAR(observation_moments.vcPow1_yawPow1, montecarlo_observation_moments.vcPow1_yawPow1, epsilon);
}

TEST(KinematicVehicleModel, getObservationMoments)
{
    const double epsilon = 0.01;

    KinematicVehicleModel model;
    // Normal Distribution
    {
        const Eigen::Vector4d mean = {0.395004, 0.407541, 3.00258, 0.811033};
        Eigen::Matrix4d cov;
        cov <<0.011991, -0.00157717,  0.00140716, -0.00422148,
              -0.00157717, 0.011991,  0.00140716,  0.00422148,
              0.00140716,  0.00140716,      0.02,           0,
              -0.00422148,  0.00422148,        0,      0.0109;

        FourDimensionalNormalDistribution dist(mean, cov);
        KinematicVehicleModel::ReducedStateMoments reduced_moments;
        reduced_moments.yawPow1 = dist.calc_moment(STATE::IDX::YAW, 1);
        reduced_moments.xPow2 = dist.calc_moment(STATE::IDX::X, 2);
        reduced_moments.yPow2 = dist.calc_moment(STATE::IDX::Y, 2);
        reduced_moments.yawPow2 = dist.calc_moment(STATE::IDX::YAW, 2);
        reduced_moments.vPow1_cPow1 = dist.calc_x_cos_z_moment(STATE::IDX::V, STATE::IDX::YAW);

        reduced_moments.vPow1_yawPow1_cPow1 = dist.calc_xy_cos_y_moment(STATE::IDX::V, STATE::IDX::YAW);
        reduced_moments.xPow2_yawPow1 = dist.calc_cross_third_moment(STATE::IDX::X, STATE::IDX::YAW, 2, 1);
        reduced_moments.yPow2_yawPow1 = dist.calc_cross_third_moment(STATE::IDX::Y, STATE::IDX::YAW, 2, 1);

        reduced_moments.xPow4 = dist.calc_moment(STATE::IDX::X, 4);
        reduced_moments.yPow4 = dist.calc_moment(STATE::IDX::Y, 4);
        reduced_moments.xPow2_yPow2 = dist.calc_xxyy_moment(STATE::IDX::X, STATE::IDX::Y);
        reduced_moments.vPow2_cPow2 = dist.calc_xx_cos_y_cos_y_moment(STATE::IDX::V, STATE::IDX::YAW);
        reduced_moments.xPow2_vPow1_cPow1 = dist.calc_xxy_cos_z_moment(STATE::IDX::X, STATE::IDX::V, STATE::IDX::YAW);
        reduced_moments.yPow2_vPow1_cPow1 = dist.calc_xxy_cos_z_moment(STATE::IDX::Y, STATE::IDX::V, STATE::IDX::YAW);

        // Step2. Create Observation Noise
        auto wr_dist = NormalDistribution(100, 10.5*10.5);
        auto wv_dist = NormalDistribution(0.0, 0.3*0.3);
        auto wyaw_dist = NormalDistribution(0.0, M_PI*M_PI/100);
        KinematicVehicleModel::ObservationNoiseMoments observation_noise;
        observation_noise.wrPow1 = wr_dist.calc_moment(1);
        observation_noise.wrPow2 = wr_dist.calc_moment(2);
        observation_noise.wvPow1 = wv_dist.calc_moment(1);
        observation_noise.wvPow2 = wv_dist.calc_moment(2);
        observation_noise.wyawPow1 = wyaw_dist.calc_moment(1);
        observation_noise.wyawPow2 = wyaw_dist.calc_moment(2);

        // Step3. Get Observation Moments
        const auto observation_moments = model.getObservationMoments(reduced_moments, observation_noise);

        EXPECT_NEAR(observation_moments.rPow1, 100.34659406906736, epsilon);
        EXPECT_NEAR(observation_moments.vcPow1, 2.0567334733086207, epsilon);
        EXPECT_NEAR(observation_moments.yawPow1, 0.8110468279982124, epsilon);
        EXPECT_NEAR(observation_moments.rPow2, 10179.699987902486, 1.0);
        EXPECT_NEAR(observation_moments.vcPow2, 4.380944133686176, epsilon);
        EXPECT_NEAR(observation_moments.yawPow2, 0.7674096866230029, epsilon);
        EXPECT_NEAR(observation_moments.rPow1_vcPow1, 206.388111310666, epsilon);
        EXPECT_NEAR(observation_moments.rPow1_yawPow1, 81.3861019975689, epsilon);
        EXPECT_NEAR(observation_moments.vcPow1_yawPow1, 1.6445076647831747, epsilon);
    }

}
