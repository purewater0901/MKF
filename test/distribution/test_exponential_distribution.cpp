#include <gtest/gtest.h>
#include <random>
#include <iostream>

#include "distribution/exponential_distribution.h"

const double epsilon = 0.01;

namespace {
    std::vector<double> getExponentialDistributionSamples(const double lambda,
                                                     const int num_sample)
    {
        std::default_random_engine generator;
        std::exponential_distribution<double> distribution(lambda);

        std::vector<double> samples(num_sample);
        for(int i=0; i<num_sample; ++i) {
            samples.at(i) = distribution(generator);
        }

        return samples;
    }
} // namespace

const double lambda = 1.0;

// Exact Distribution
ExponentialDistribution dist(lambda);

// Monte Carlo Simulation
const int num_sample = 10000*100000;
const auto samples = getExponentialDistributionSamples(lambda, num_sample);

TEST(ExponentialDistribution, X_MOMENT)
{
    // First Order
    {
        // exact
        const auto exact_moment = dist.calc_moment(1);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<samples.size(); ++i) {
            sum += samples.at(i);
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }

    // Second Order
    {
        // exact
        const auto exact_moment = dist.calc_moment(2);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<samples.size(); ++i) {
            const double x = samples.at(i);
            sum += x*x;
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }

    // Third Order
    {
        // exact
        const auto exact_moment = dist.calc_moment(3);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<samples.size(); ++i) {
            const double x = samples.at(i);
            sum += x*x*x;
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }

    // Fourth Order
    {
        // exact
        const auto exact_moment = dist.calc_moment(4);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<num_sample; ++i) {
            const double x = samples.at(i);
            sum += x*x*x*x;
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, 0.01);
    }
}

TEST(ExponentialDistribution, TRIGONOMETRIC_MOMENT)
{
    // First Order
    {
        {
            // exact
            const auto exact_moment = dist.calc_sin_moment(1);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_cos_moment(1);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::cos(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }
    }

    // Second Order
    {
        {
            // exact
            const auto exact_moment = dist.calc_sin_moment(2);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::sin(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_cos_moment(2);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::cos(x) * std::cos(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_cos_sin_moment(1, 1);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::cos(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }
    }

    // Third Order
    {
        {
            // exact
            const auto exact_moment = dist.calc_sin_moment(3);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::sin(x) * std::sin(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_cos_moment(3);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::cos(x) * std::cos(x) * std::cos(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_cos_sin_moment(2, 1);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::cos(x) * std::cos(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }


        // exact
        {
            const auto exact_moment = dist.calc_cos_sin_moment(1, 2);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::cos(x) * std::sin(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }
    }

    // Fourth Order
    // Third Order
    {
        {
            // exact
            const auto exact_moment = dist.calc_sin_moment(4);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::sin(x) * std::sin(x) * std::sin(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_cos_moment(4);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::cos(x) * std::cos(x) * std::cos(x) * std::cos(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_cos_sin_moment(2, 2);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::cos(x) * std::cos(x) * std::sin(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }


        {
            // exact
            const auto exact_moment = dist.calc_cos_sin_moment(1, 3);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::cos(x) * std::sin(x) * std::sin(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_cos_sin_moment(3, 1);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::cos(x) * std::cos(x) * std::cos(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }
    }
}

TEST(ExponentialDistribution, MIXED_TRIGONOMETRIC_MOMENT)
{
    // Second Order
    {
        {
            // exact
            const auto exact_moment = dist.calc_x_sin_moment(1, 1);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x*std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_x_cos_moment(1, 1);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x * std::cos(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }
    }

    // Third Order
    {
        {
            // exact
            const auto exact_moment = dist.calc_x_sin_moment(2, 1);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x * x * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }


        {
            // exact
            const auto exact_moment = dist.calc_x_sin_moment(1, 2);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x * std::sin(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_x_cos_moment(2, 1);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x * x * std::cos(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }


        {
            // exact
            const auto exact_moment = dist.calc_x_cos_moment(1, 2);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x * std::cos(x) * std::cos(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }
    }

    // Fourth Order
    {
        {
            // exact
            const auto exact_moment = dist.calc_x_sin_moment(3, 1);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x * x * x *std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }


        {
            // exact
            const auto exact_moment = dist.calc_x_sin_moment(1, 3);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x * std::sin(x) * std::sin(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_x_cos_moment(3, 1);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x * x * x * std::cos(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_x_cos_moment(1, 3);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x * std::cos(x) * std::cos(x) * std::cos(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_x_sin_moment(2, 2);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x * x * std::sin(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_x_cos_moment(2, 2);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x * x * std::cos(x) * std::cos(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }
    }
}
