#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#include "distribution/normal_distribution.h"
#include "distribution/two_dimensional_normal_distribution.h"
#include "distribution/three_dimensional_normal_distribution.h"
#include "distribution/exponential_distribution.h"
#include "distribution/uniform_distribution.h"

#include "filter/paper_example_2d_ukf.h"
#include "filter/paper_example_3d_ukf.h"

int main()
{
    // Two Dimensional
    {
        const double x_lambda = 1.0;
        const double lower_theta = -M_PI/3.0;
        const double upper_theta = M_PI/6.0;
        auto x_dist = ExponentialDistribution(x_lambda);
        auto theta_dist = UniformDistribution(lower_theta, upper_theta);
        const Eigen::Vector2d mean = {x_dist.calc_mean(), theta_dist.calc_mean()};
        const Eigen::Matrix2d cov = (Eigen::Matrix2d() << x_dist.calc_variance(), 0.0, 0.0, theta_dist.calc_variance()).finished();

        // E[xy]
        {
            // Exact Moment
            const double exact_moment = x_dist.calc_moment(1) * theta_dist.calc_moment(1);

            // Linear(E[x]E[y)])
            const double linear_moment = x_dist.calc_moment(1) * theta_dist.calc_moment(1);

            // E[x]E[y]
            const double non_correlation_exact = x_dist.calc_moment(1) * theta_dist.calc_moment(1);

            // Unscented Transform
            PaperExample2DUKF ukf(mean, cov);
            const double ukf_moment = ukf.predict("xy");

            std::cout << "E[xy]" << std::endl;
            std::cout << "Exact Moment: " << exact_moment << std::endl;
            std::cout << "Linear: " << linear_moment << std::endl;
            std::cout << "E[x]E[y]: " << non_correlation_exact << std::endl;
            std::cout << "UKF Moment: " << ukf_moment << std::endl;
            std::cout << "MonteCarlo: " << -0.2618191494201287 << std::endl;
            std::cout << "-----------------------------" << std::endl;
        }

        // E[xcos(theta)]
        {
            // Exact Moment
            const double exact_moment = x_dist.calc_moment(1) * theta_dist.calc_cos_moment(1);

            // Linear(E[x]E[cos(theta)])
            const double linear_moment = x_dist.calc_moment(1) * std::cos(theta_dist.calc_moment(1));

            // E[x]E[cos(theta)]
            const double non_correlation_exact = x_dist.calc_moment(1) * theta_dist.calc_cos_moment(1);

            // Unscented Transform
            PaperExample2DUKF ukf(mean, cov);
            const double ukf_moment = ukf.predict("xcos");

            std::cout << "E[xcos(theta)]" << std::endl;
            std::cout << "Exact Moment: " << exact_moment << std::endl;
            std::cout << "Linear: " << linear_moment << std::endl;
            std::cout << "E[x]E[cos(theta)]: " << non_correlation_exact << std::endl;
            std::cout << "UKF Moment: " << ukf_moment << std::endl;
            std::cout << "MonteCarlo: " << 0.8696105959403237 << std::endl;
            std::cout << "----------------------" << std::endl;
        }

        // E[xcos(theta)sin(theta)]
        {
            // Exact Moment
            const double exact_moment = x_dist.calc_moment(1) * theta_dist.calc_cos_sin_moment(1, 1);

            // Linear(E[x]E[cos(theta)]E[sin(theta)])
            const double linear_moment = x_dist.calc_moment(0) * std::cos(theta_dist.calc_moment(1)) * std::sin(theta_dist.calc_moment(1));

            // E[x]E[cos(theta)sin(theta)]
            const double non_correlation_exact = x_dist.calc_moment(1) * theta_dist.calc_cos_sin_moment(1, 1);

            // Unscented Transform
            PaperExample2DUKF ukf(mean, cov);
            const double ukf_moment = ukf.predict("xcossin");

            std::cout << "E[xcos(theta)sin(theta)]" << std::endl;
            std::cout << "Exact Moment: " << exact_moment << std::endl;
            std::cout << "Linear: " << linear_moment << std::endl;
            std::cout << "E[x]E[cos(theta)sin(theta)]: " << non_correlation_exact << std::endl;
            std::cout << "UKF Moment: " << ukf_moment << std::endl;
            std::cout << "MonteCarlo: " << -0.15915291917551003 << std::endl;
        }
    }

    std::cout << "------------------------------" << std::endl;
    std::cout << "------------------------------" << std::endl;
    std::cout << "------------------------------" << std::endl;
    std::cout << "Non-Gaussian Example Finished" << std::endl;


    // Two Dimensional
    {
        const Eigen::Vector2d mean = {10.0, M_PI/3.0};
        const Eigen::Matrix2d cov = (Eigen::Matrix2d() << 5.0, 0.1, 0.1, M_PI/6.0).finished();

        // E[xy]
        {
            // Exact Moment
            TwoDimensionalNormalDistribution two_dim_normal_dist(mean, cov);
            const double exact_moment = two_dim_normal_dist.calc_xy_moment();

            // Linear(E[x]E[y)])
            NormalDistribution x_dist(mean(0), cov(0, 0));
            NormalDistribution theta_dist(mean(1), cov(1, 1));
            const double linear_moment = x_dist.calc_moment(1) * theta_dist.calc_moment(1);

            // E[x]E[y]
            const double non_correlation_exact = x_dist.calc_moment(1) * theta_dist.calc_moment(1);

            // Unscented Transform
            PaperExample2DUKF ukf(mean, cov);
            const double ukf_moment = ukf.predict("xy");

            std::cout << "E[xy]" << std::endl;
            std::cout << "Exact Moment: " << exact_moment << std::endl;
            std::cout << "Linear: " << linear_moment << std::endl;
            std::cout << "E[x]E[y]: " << non_correlation_exact << std::endl;
            std::cout << "UKF Moment: " << ukf_moment << std::endl;
            std::cout << "MonteCarlo: " << 2.1774640733069437 << std::endl;
            std::cout << "-----------------------------" << std::endl;
        }

        // E[xcos(theta)]
        {
            // Exact Moment
            TwoDimensionalNormalDistribution two_dim_normal_dist(mean, cov);
            const double exact_moment = two_dim_normal_dist.calc_x_cos_y_moment();

            // Linear(E[x]E[cos(theta)])
            NormalDistribution x_dist(mean(0), cov(0, 0));
            NormalDistribution theta_dist(mean(1), cov(1, 1));
            const double linear_moment = x_dist.calc_moment(1) * std::cos(theta_dist.calc_moment(1));

            // E[x]E[cos(theta)]
            const double non_correlation_exact = x_dist.calc_moment(1) * theta_dist.calc_cos_moment(1);

            // Unscented Transform
            PaperExample2DUKF ukf(mean, cov);
            const double ukf_moment = ukf.predict("xcos");

            std::cout << "E[xcos(theta)]" << std::endl;
            std::cout << "Exact Moment: " << exact_moment << std::endl;
            std::cout << "Linear: " << linear_moment << std::endl;
            std::cout << "E[x]E[cos(theta)]: " << non_correlation_exact << std::endl;
            std::cout << "UKF Moment: " << ukf_moment << std::endl;
            std::cout << "MonteCarlo: " << 2.1774640733069437 << std::endl;
            std::cout << "----------------------" << std::endl;
        }

        // E[xcos(theta)sin(theta)]
        {
            // Exact Moment
            TwoDimensionalNormalDistribution two_dim_normal_dist(mean, cov);
            const double exact_moment = two_dim_normal_dist.calc_x_cos_y_sin_y_moment();

            // Linear(E[x]E[cos(theta)]E[sin(theta)])
            NormalDistribution x_dist(mean(0), cov(0, 0));
            NormalDistribution theta_dist(mean(1), cov(1, 1));
            const double linear_moment = x_dist.calc_moment(0) * std::cos(theta_dist.calc_moment(1)) * std::sin(theta_dist.calc_moment(1));

            // E[x]E[cos(theta)sin(theta)]
            const double non_correlation_exact = x_dist.calc_moment(1) * theta_dist.calc_cos_sin_moment(1, 1);

            // Unscented Transform
            PaperExample2DUKF ukf(mean, cov);
            const double ukf_moment = ukf.predict("xcossin");

            std::cout << "E[xcos(theta)sin(theta)]" << std::endl;
            std::cout << "Exact Moment: " << exact_moment << std::endl;
            std::cout << "Linear: " << linear_moment << std::endl;
            std::cout << "E[x]E[cos(theta)sin(theta)]: " << non_correlation_exact << std::endl;
            std::cout << "UKF Moment: " << ukf_moment << std::endl;
            std::cout << "MonteCarlo: " << 2.1774640733069437 << std::endl;
        }
    }

    std::cout << "------------------------------" << std::endl;
    std::cout << "------------------------------" << std::endl;

    {
        const Eigen::Vector2d mean = {10.0, M_PI/3.0};
        const Eigen::Matrix2d cov = (Eigen::Matrix2d() << 5.0, 1.5, 1.5, M_PI/6.0).finished();

        // E[xy]
        {
            // Exact Moment
            TwoDimensionalNormalDistribution two_dim_normal_dist(mean, cov);
            const double exact_moment = two_dim_normal_dist.calc_xy_moment();

            // Linear(E[x]E[y)])
            NormalDistribution x_dist(mean(0), cov(0, 0));
            NormalDistribution theta_dist(mean(1), cov(1, 1));
            const double linear_moment = x_dist.calc_moment(1) * theta_dist.calc_moment(1);

            // E[x]E[y]
            const double non_correlation_exact = x_dist.calc_moment(1) * theta_dist.calc_moment(1);

            // Unscented Transform
            PaperExample2DUKF ukf(mean, cov);
            const double ukf_moment = ukf.predict("xy");

            std::cout << "E[xy]" << std::endl;
            std::cout << "Exact Moment: " << exact_moment << std::endl;
            std::cout << "Linear: " << linear_moment << std::endl;
            std::cout << "E[x]E[y]: " << non_correlation_exact << std::endl;
            std::cout << "UKF Moment: " << ukf_moment << std::endl;
            std::cout << "MonteCarlo: " << 2.1774640733069437 << std::endl;
            std::cout << "-----------------------------" << std::endl;
        }

        // E[xcos(theta)]
        {
            // Exact Moment
            TwoDimensionalNormalDistribution two_dim_normal_dist(mean, cov);
            const double exact_moment = two_dim_normal_dist.calc_x_cos_y_moment();

            // Linear
            NormalDistribution x_dist(mean(0), cov(0, 0));
            NormalDistribution theta_dist(mean(1), cov(1, 1));
            const double linear_moment = x_dist.calc_moment(1) * std::cos(theta_dist.calc_moment(1));

            // E[x]E[cos(theta)]
            const double non_correlation_exact = x_dist.calc_moment(1) * theta_dist.calc_cos_moment(1);

            // Unscented Transform
            PaperExample2DUKF ukf(mean, cov);
            const double ukf_moment = ukf.predict("xcos");

            std::cout << "E[xcos(theta)]" << std::endl;
            std::cout << "Exact Moment: " << exact_moment << std::endl;
            std::cout << "Linear: " << linear_moment << std::endl;
            std::cout << "E[x]E[cos(theta)]: " << non_correlation_exact << std::endl;
            std::cout << "UKF Moment: " << ukf_moment << std::endl;
            std::cout << "MonteCarlo: " << 2.8476949077323734 << std::endl;
            std::cout << "------------------------------" << std::endl;
        }

        // E[xcos(theta)sin(theta)]
        {
            // Exact Moment
            TwoDimensionalNormalDistribution two_dim_normal_dist(mean, cov);
            const double exact_moment = two_dim_normal_dist.calc_x_cos_y_sin_y_moment();

            // Linear(E[x]E[cos(theta)]E[sin(theta)])
            NormalDistribution x_dist(mean(0), cov(0, 0));
            NormalDistribution theta_dist(mean(1), cov(1, 1));
            const double linear_moment = x_dist.calc_moment(0) * std::cos(theta_dist.calc_moment(1)) * std::sin(theta_dist.calc_moment(1));

            // E[x]E[cos(theta)sin(theta)]
            const double non_correlation_exact = x_dist.calc_moment(1) * theta_dist.calc_cos_sin_moment(1, 1);

            // Unscented Transform
            PaperExample2DUKF ukf(mean, cov);
            const double ukf_moment = ukf.predict("xcossin");

            std::cout << "E[xcos(theta)sin(theta)]" << std::endl;
            std::cout << "Exact Moment: " << exact_moment << std::endl;
            std::cout << "Linear: " << linear_moment << std::endl;
            std::cout << "E[x]E[cos(theta)sin(theta)]: " << non_correlation_exact << std::endl;
            std::cout << "UKF Moment: " << ukf_moment << std::endl;
            std::cout << "MonteCarlo: " << 2.1774640733069437 << std::endl;
        }
    }

    std::cout << "------------------------------" << std::endl;
    std::cout << "------------------------------" << std::endl;

    {
        const Eigen::Vector3d mean = {10.0, 5.0, M_PI/3.0};
        const Eigen::Matrix3d cov = (Eigen::Matrix3d() << 3.0, 0.5, 0.5,
                                                          0.5, 2.0, 0.3,
                                                          0.5, 0.3, M_PI/10.0).finished();

        // E[xysin(theta)]
        {
            // Exact Moment
            ThreeDimensionalNormalDistribution two_dim_normal_dist(mean, cov);
            const double exact_moment = two_dim_normal_dist.calc_xy_sin_z_moment();

            // Linear
            NormalDistribution x_dist(mean(0), cov(0, 0));
            NormalDistribution y_dist(mean(1), cov(1, 1));
            NormalDistribution theta_dist(mean(2), cov(2, 2));
            const double linear_moment = x_dist.calc_moment(1) * y_dist.calc_moment(1) * std::sin(theta_dist.calc_moment(1));

            // E[x^2]E[y]E[cos(theta)]
            const double non_correlation_exact = x_dist.calc_moment(1) * y_dist.calc_moment(1) * theta_dist.calc_sin_moment(1);

            // Unscented Transform
            PaperExample3DUKF ukf(mean, cov);
            const double ukf_moment = ukf.predict("xysin");

            std::cout << "E[xysin(theta)]" << std::endl;
            std::cout << "Exact Moment: " << exact_moment << std::endl;
            std::cout << "Linear: " << linear_moment << std::endl;
            std::cout << "E[x]E[y]E[sin(theta)]: " << non_correlation_exact << std::endl;
            std::cout << "UKF Moment: " << ukf_moment << std::endl;
            std::cout << "MonteCarlo: " << 162.29415844773192 << std::endl;
            std::cout << "------------" << std::endl;
        }

        // E[xxycos(theta)]
        {
            // Exact Moment
            ThreeDimensionalNormalDistribution two_dim_normal_dist(mean, cov);
            const double exact_moment = two_dim_normal_dist.calc_xxy_cos_z_moment();

            // Linear
            NormalDistribution x_dist(mean(0), cov(0, 0));
            NormalDistribution y_dist(mean(1), cov(1, 1));
            NormalDistribution theta_dist(mean(2), cov(2, 2));
            const double separate_moment = x_dist.calc_moment(2) * y_dist.calc_moment(1) * std::cos(theta_dist.calc_moment(1));

            // E[x^2]E[y]E[cos(theta)]
            const double non_correlation_exact = x_dist.calc_moment(2) * y_dist.calc_moment(1) * theta_dist.calc_cos_moment(1);

            // Unscented Transform
            PaperExample3DUKF ukf(mean, cov);
            const double ukf_moment = ukf.predict("xxycos");

            std::cout << "E[xxycos(theta)]" << std::endl;
            std::cout << "Exact Moment: " << exact_moment << std::endl;
            std::cout << "Linear: " << separate_moment << std::endl;
            std::cout << "E[x^2]E[cos(theta)]: " << non_correlation_exact << std::endl;
            std::cout << "UKF Moment: " << ukf_moment << std::endl;
            std::cout << "MonteCarlo: " << 162.29415844773192 << std::endl;
        }
    }

    return 0;
}