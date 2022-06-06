#include <gtest/gtest.h>
#include <random>
#include <iostream>

#include "distribution/two_dimensional_normal_distribution.h"
#include "distribution/three_dimensional_normal_distribution.h"

TEST(ThreeDimensionalNormalDistribution, FIRST_ORDER)
{
    const double epsilon = 0.1;
    const Eigen::Vector3d mean{1.59378, 1.3832, 0.680927};
    Eigen::Matrix3d covariance;
    covariance << 0.00179042,  -0.00146877, -5.51844e-05,
                  -0.00146877, 0.00201589, 4.10447e-05,
                  -5.51844e-05,  4.10447e-05,  0.000123276;
    ThreeDimensionalNormalDistribution dist(mean, covariance);

    // mean and covariance
    for(int i=0; i<3; ++i) {
        const auto exact_mean = dist.calc_mean(i);
        const auto exact_cov = dist.calc_covariance(i);
        EXPECT_NEAR(exact_mean, mean(i), epsilon);
        EXPECT_NEAR(exact_cov, covariance(i, i), epsilon);
    }

    // moment
    for(int i=0; i<3; ++i) {
        NormalDistribution ans_dist(mean(i), covariance(i, i));
        for(int moment=1; moment<5; ++moment) {
            const auto exact_moment = dist.calc_moment(i, moment);
            EXPECT_NEAR(exact_moment, ans_dist.calc_moment(moment), epsilon);
        }
    }
}

TEST(ThreeDimensionalNormalDistribution, SECOND_ORDER)
{
    const double epsilon = 0.1;
    {
        const Eigen::Vector3d mean{1.59378, 1.3832, 0.680927};
        Eigen::Matrix3d covariance;
        covariance << 0.00179042,  -0.00146877, -5.51844e-05,
                -0.00146877, 0.00201589, 4.10447e-05,
                -5.51844e-05,  4.10447e-05,  0.000123276;
        ThreeDimensionalNormalDistribution dist(mean, covariance);

        // E[XY]
        {
            const auto exact_moment = dist.calc_cross_second_moment(0, 1);
            EXPECT_NEAR(exact_moment, 2.203032708965769, epsilon);
        }

        // E[XZ]
        {
            const auto exact_moment = dist.calc_cross_second_moment(0, 2);
            EXPECT_NEAR(exact_moment, 1.0851907459761667, epsilon);
        }

        // E[YZ]
        {
            const auto exact_moment = dist.calc_cross_second_moment(1, 2);
            EXPECT_NEAR(exact_moment, 0.9418837963208465, epsilon);
        }

        // E[XsinZ]
        {
            const auto exact_moment = dist.calc_x_sin_z_moment(0, 2);
            EXPECT_NEAR(exact_moment, 1.0031835126980233, epsilon);
        }

        // E[YsinZ]
        {
            const auto exact_moment = dist.calc_x_sin_z_moment(1, 2);
            EXPECT_NEAR(exact_moment, 0.8707247528942638, epsilon);
        }

        // E[XcosZ]
        {
            const auto exact_moment = dist.calc_x_cos_z_moment(0, 2);
            EXPECT_NEAR(exact_moment, 1.2383025610644065, epsilon);
        }

        // E[YcosZ]
        {
            const auto exact_moment = dist.calc_x_cos_z_moment(1, 2);
            EXPECT_NEAR(exact_moment, 1.0746573620514024, epsilon);
        }
    }

    {
        const Eigen::Vector3d mean{10.0, 10.0, M_PI/3.0};
        Eigen::Matrix3d covariance;
        covariance << 1.0,  1.0, 0.05,
                1.0,  4.0, 0.2,
                0.05, 0.2, M_PI*M_PI/100;
        ThreeDimensionalNormalDistribution dist(mean, covariance);

        // E[XY]
        {
            const auto exact_moment = dist.calc_cross_second_moment(0, 1);
            EXPECT_NEAR(exact_moment, 100.99996042907365, epsilon);
        }

        // E[XZ]
        {
            const auto exact_moment = dist.calc_cross_second_moment(0, 2);
            EXPECT_NEAR(exact_moment, 10.521336060847299, epsilon);
        }

        // E[YZ]
        {
            const auto exact_moment = dist.calc_cross_second_moment(1, 2);
            EXPECT_NEAR(exact_moment, 10.670843788192368, epsilon);
        }

        // E[XsinZ]
        {
            const auto exact_moment = dist.calc_x_sin_z_moment(0, 2);
            EXPECT_NEAR(exact_moment, 8.266846513763234, epsilon);
        }

        // E[YsinZ]
        {
            const auto exact_moment = dist.calc_x_sin_z_moment(1, 2);
            EXPECT_NEAR(exact_moment, 8.337759405477527, epsilon);
        }

        // E[XcosZ]
        {
            const auto exact_moment = dist.calc_x_cos_z_moment(0, 2);
            EXPECT_NEAR(exact_moment, 4.719056827999589, epsilon);
        }

        // E[YcosZ]
        {
            const auto exact_moment = dist.calc_x_cos_z_moment(1, 2);
            EXPECT_NEAR(exact_moment, 4.594809380237797, epsilon);
        }
    }

}

TEST(ThreeDimensionalNormalDistribution, SECOND_ORDER_INDEPENDENT)
{
    const double epsilon = 0.1;
    const Eigen::Vector3d mean{1.59378, 1.3832, 0.680927};
    Eigen::Matrix3d covariance;
    covariance << 0.00179042,  0.0, 0.0,
                  0.0, 0.00201589, 0.0,
                  0.0,  0.0,  0.000123276;
    ThreeDimensionalNormalDistribution dist(mean, covariance);

    // E[XY]
    {
        const auto exact_moment = dist.calc_cross_second_moment(0, 1);
        EXPECT_NEAR(exact_moment, 2.2045317285644783, epsilon);
    }

    // E[XZ]
    {
        const auto exact_moment = dist.calc_cross_second_moment(0, 2);
        EXPECT_NEAR(exact_moment, 1.085263158467877, epsilon);
    }

    // E[YZ]
    {
        const auto exact_moment = dist.calc_cross_second_moment(1, 2);
        EXPECT_NEAR(exact_moment, 0.9418660917734089, epsilon);
    }

    // E[XsinZ]
    {
        const auto exact_moment = dist.calc_x_sin_z_moment(0, 2);
        EXPECT_NEAR(exact_moment, 1.0032572411497331, epsilon);
    }

    // E[YsinZ]
    {
        const auto exact_moment = dist.calc_x_sin_z_moment(1, 2);
        EXPECT_NEAR(exact_moment, 0.870695712669271, epsilon);
    }

    // E[XcosZ]
    {
        const auto exact_moment = dist.calc_x_cos_z_moment(0, 2);
        EXPECT_NEAR(exact_moment, 1.2382766676773642, epsilon);
    }

    // E[YcosZ]
    {
        const auto exact_moment = dist.calc_x_cos_z_moment(1, 2);
        EXPECT_NEAR(exact_moment, 1.074661768382129, epsilon);
    }
}

TEST(ThreeDimensionalNormalDistribution, THREE_ORDER)
{
    const double epsilon = 0.1;
    {
        const Eigen::Vector3d mean{1.59378, 1.3832, 0.680927};
        Eigen::Matrix3d covariance;
        covariance << 0.00179042,  -0.00146877, -5.51844e-05,
                -0.00146877, 0.00201589, 4.10447e-05,
                -5.51844e-05,  4.10447e-05,  0.000123276;
        ThreeDimensionalNormalDistribution dist(mean, covariance);

        // E[XXsinZ]
        {
            const auto exact_moment = dist.calc_xx_sin_z_moment(0, 2);
            EXPECT_NEAR(exact_moment, 1.5999219017662898, epsilon);
        }

        // E[YYsinZ]
        {
            const auto exact_moment = dist.calc_xx_sin_z_moment(1, 2);
            EXPECT_NEAR(exact_moment, 1.2057279331745845, epsilon);
        }

        // E[XXcosZ]
        {
            const auto exact_moment = dist.calc_xx_cos_z_moment(0, 2);
            EXPECT_NEAR(exact_moment, 1.9750118193977413, epsilon);
        }

        // E[YYcosZ]
        {
            const auto exact_moment = dist.calc_xx_cos_z_moment(1, 2);
            EXPECT_NEAR(exact_moment,  1.4880104219569756, epsilon);
        }
    }

    {
        const Eigen::Vector3d mean{10.0, 10.0, M_PI/3.0};
        Eigen::Matrix3d covariance;
        covariance << 1.0,  1.0, 0.05,
                      1.0,  4.0, 0.2,
                      0.05, 0.2, M_PI*M_PI/100;
        ThreeDimensionalNormalDistribution dist(mean, covariance);

        // E[XXsinZ]
        {
            const auto exact_moment = dist.calc_xx_sin_z_moment(0, 2);
            EXPECT_NEAR(exact_moment, 83.73166675390627, epsilon);
        }

        // E[YYsinZ]
        {
            const auto exact_moment = dist.calc_xx_sin_z_moment(1, 2);
            EXPECT_NEAR(exact_moment,  87.59306538840139, epsilon);
        }

        // E[XXcosZ]
        {
            const auto exact_moment = dist.calc_xx_cos_z_moment(0, 2);
            EXPECT_NEAR(exact_moment, 47.253427437643346, epsilon);
        }

        // E[YYcosZ]
        {
            const auto exact_moment = dist.calc_xx_cos_z_moment(1, 2);
            EXPECT_NEAR(exact_moment,  46.179867786188986, epsilon);
        }

        // E[XYcosZ]
        {
            const auto exact_moment = dist.calc_xy_cos_z_moment();
            EXPECT_NEAR(exact_moment, 46.002933072079756, epsilon);
        }

        // E[XYsinZ]
        {
            const auto exact_moment = dist.calc_xy_sin_z_moment();
            EXPECT_NEAR(exact_moment, 84.43723044965331, epsilon);
        }

        // E[XXYcosZ]
        {
            const auto exact_moment = dist.calc_xxy_cos_z_moment();
            EXPECT_NEAR(exact_moment, 465.1488547888971, epsilon);
        }
    }

}

TEST(ThreeDimensionalNormalDistribution, THREE_ORDER_INDEPENDENT)
{
    const double epsilon = 0.001;
    {
        const Eigen::Vector3d mean{1.59378, 1.3832, 0.680927};
        Eigen::Matrix3d covariance;
        covariance << 0.00179042,  0.0, 0.0,
                      0.0, 0.00201589, 0.0,
                      0.0,  0.0,  0.000123276;
        ThreeDimensionalNormalDistribution dist(mean, covariance);

        // E[XXsinZ]
        {
            const auto exact_moment = dist.calc_xx_sin_z_moment(0, 2);
            EXPECT_NEAR(exact_moment, 1.6001081135776964, epsilon);
        }

        // E[YYsinZ]
        {
            const auto exact_moment = dist.calc_xx_sin_z_moment(1, 2);
            EXPECT_NEAR(exact_moment, 1.2056155207462655, epsilon);
        }

        // E[XXcosZ]
        {
            const auto exact_moment = dist.calc_xx_cos_z_moment(0, 2);
            EXPECT_NEAR(exact_moment, 1.9749439225285936, epsilon);
        }

        // E[YYcosZ]
        {
            const auto exact_moment = dist.calc_xx_cos_z_moment(1, 2);
            EXPECT_NEAR(exact_moment, 1.488038872588126, epsilon);
        }
    }

    {
        const Eigen::Vector3d mean{1.0, 1.0, M_PI/4.0};
        Eigen::Matrix3d cov;
        cov << 1.0, 0.05, 0.0,
               0.05, 1.0, 0.2,
               0.0, 0.2,  0.1;
        ThreeDimensionalNormalDistribution dist(mean, cov);

        // E[XYcos(z)]
        {
            const double exact_moment = dist.calc_xy_cos_z_moment();
            EXPECT_NEAR(exact_moment, 0.5717082269602641, epsilon);
        }

        // E[XYsin(z)]
        {
            const double exact_moment = dist.calc_xy_sin_z_moment();
            EXPECT_NEAR(exact_moment, 0.8408374011650688, epsilon);
        }
    }

    {

        const Eigen::Vector3d mean{1.0, 1.0, M_PI/4.0};
        Eigen::Matrix3d cov;
        cov << 1.0, 0.0, 0.05,
               0.0, 1.0, 0.2,
               0.05, 0.2,  0.1;
        ThreeDimensionalNormalDistribution dist(mean, cov);

        // E[XYcos(z)]
        {
            const double exact_moment = dist.calc_xy_cos_z_moment();
            EXPECT_NEAR(exact_moment,  0.49767157402591145, epsilon);
        }

        // E[XYsin(z)]
        {
            const double exact_moment = dist.calc_xy_sin_z_moment();
            EXPECT_NEAR(exact_moment, 0.8340922355209054, epsilon);
        }

        // E[XXYcos(Z)]
        {
            const double exact_moment = dist.calc_xxy_cos_z_moment();
            EXPECT_NEAR(exact_moment, 0.9941102018163359, epsilon);
        }
    }

    {
        const Eigen::Vector3d mean{1.0, 1.0, M_PI/4.0};
        Eigen::Matrix3d cov;
        cov << 1.0, 0.05, 0.1,
               0.05, 1.0, 0.0,
               0.1, 0.0,  0.1;
        ThreeDimensionalNormalDistribution dist(mean, cov);

        // E[XYcos(z)]
        {
            const double exact_moment = dist.calc_xy_cos_z_moment();
            EXPECT_NEAR(exact_moment, 0.6391623949442164, epsilon);
        }

        // E[XYsin(z)]
        {
            const double exact_moment = dist.calc_xy_sin_z_moment();
            EXPECT_NEAR(exact_moment, 0.7737224521226675, epsilon);
        }
    }

    {
        const Eigen::Vector3d mean{1.0, 1.0, M_PI/4.0};
        Eigen::Matrix3d cov;
        cov << 1.0, 0.0, 0.0,
               0.0, 1.0, 0.0,
               0.0, 0.0,  0.1;
        ThreeDimensionalNormalDistribution dist(mean, cov);

        // E[XYcos(z)]
        {
            const double exact_moment = dist.calc_xy_cos_z_moment();
            EXPECT_NEAR(exact_moment, 0.6727703373675349, epsilon);
        }

        // E[XYsin(z)]
        {
            const double exact_moment = dist.calc_xy_sin_z_moment();
            EXPECT_NEAR(exact_moment, 0.6726921174920212, epsilon);
        }
    }

}

TEST(ThreeDimensionalNormalDistribution, FOURTH_ORDER)
{
    const double epsilon = 0.01;
    {
        const Eigen::Vector3d mean{1.59378, 1.3832, 0.680927};
        Eigen::Matrix3d covariance;
        covariance << 0.00179042,  -0.00146877, -5.51844e-05,
                     -0.00146877, 0.00201589, 4.10447e-05,
                     -5.51844e-05,  4.10447e-05,  0.000123276;
        ThreeDimensionalNormalDistribution dist(mean, covariance);

        // E[XXYY]
        {
            const auto exact_moment = dist.calc_xxyy_moment(0, 1);
            EXPECT_NEAR(exact_moment, 4.8555, epsilon);
        }

        // E[XXZZ]
        {
            const auto exact_moment = dist.calc_xxyy_moment(0, 2);
            EXPECT_NEAR(exact_moment, 1.1786588207179274, epsilon);
        }

        // E[YYZZ]
        {
            const auto exact_moment = dist.calc_xxyy_moment(1, 2);
            EXPECT_NEAR(exact_moment, 0.8884422989077401, epsilon);
        }
    }

    {
        const Eigen::Vector3d mean{10.0, 10.0, M_PI/3.0};
        Eigen::Matrix3d covariance;
        covariance << 1.0,  1.0, 0.05,
                1.0,  4.0, 0.2,
                0.05, 0.2, M_PI*M_PI/100;
        ThreeDimensionalNormalDistribution dist(mean, covariance);

        // E[XXYY]
        {
            const auto exact_moment = dist.calc_xxyy_moment(0, 1);
            EXPECT_NEAR(exact_moment, 10906.299426620859, 0.5);
        }

        // E[XXZZ]
        {
            const auto exact_moment = dist.calc_xxyy_moment(0, 2);
            EXPECT_NEAR(exact_moment, 122.81734769970474, epsilon);
        }

        // E[YYZZ]
        {
            const auto exact_moment = dist.calc_xxyy_moment(1, 2);
            EXPECT_NEAR(exact_moment, 132.75721057249802, 0.02);
        }
    }

    {
        const Eigen::Vector2d mean{1.59378, 1.3832};
        Eigen::Matrix2d covariance;
        covariance << 0.00179042,  -0.00146877,
                     -0.00146877, 0.00201589;
        TwoDimensionalNormalDistribution dist(mean, covariance);
        const auto exact_moment = dist.calc_xxyy_moment();
        EXPECT_NEAR(exact_moment, 4.8555, epsilon);
    }
}

TEST(ThreeDimensionalNormalDistribution, FOURTH_ORDER_INDEPENDENT)
{
    const double epsilon = 0.01;
    {
        const Eigen::Vector3d mean{1.59378, 1.3832, 0.680927};
        Eigen::Matrix3d covariance;
        covariance << 0.00179042,  0.0, 0.0,
                      0.0, 0.00201589, 0.0,
                      0.0,  0.0,  0.000123276;
        ThreeDimensionalNormalDistribution dist(mean, covariance);

        // E[XXYY]
        {
            const auto exact_moment = dist.calc_xxyy_moment(0, 1);
            EXPECT_NEAR(exact_moment, 4.8685058886827735, epsilon);
        }

        // E[XXZZ]
        {
            const auto exact_moment = dist.calc_xxyy_moment(0, 2);
            EXPECT_NEAR(exact_moment, 1.1789390639445279, epsilon);
        }

        // E[YYZZ]
        {
            const auto exact_moment = dist.calc_xxyy_moment(1, 2);
            EXPECT_NEAR(exact_moment, 0.8882819814996061, epsilon);
        }
    }

    {
        const Eigen::Vector3d mean{1.0, 1.0, M_PI/4.0};
        Eigen::Matrix3d cov;
        cov << 1.0, 0.05, 0.0,
               0.05, 1.0, 0.1,
               0.0, 0.1,  0.1;
        ThreeDimensionalNormalDistribution dist(mean, cov);

        // E[XXYcos(z)]
        {
            const double exact_moment = dist.calc_xxy_cos_z_moment();
            EXPECT_NEAR(exact_moment, 1.2776448771058118, epsilon);
        }
    }

    {
        const Eigen::Vector3d mean{1.0, 1.0, M_PI/4.0};
        Eigen::Matrix3d cov;
        cov << 1.0, 0.05, 0.1,
               0.05, 1.0, 0.0,
               0.1, 0.0,  0.1;
        ThreeDimensionalNormalDistribution dist(mean, cov);

        // E[XXYcos(z)]
        {
            const double exact_moment = dist.calc_xxy_cos_z_moment();
            EXPECT_NEAR(exact_moment, 1.2642921359975434, epsilon);
        }
    }

    {
        const Eigen::Vector3d mean{1.0, 1.0, M_PI/4.0};
        Eigen::Matrix3d cov;
        cov << 1.0, 0.0, 0.0,
               0.0, 1.0, 0.0,
               0.0, 0.0,  0.1;
        ThreeDimensionalNormalDistribution dist(mean, cov);

        // E[XXYcos(z)]
        {
            const double exact_moment = dist.calc_xxy_cos_z_moment();
            EXPECT_NEAR(exact_moment, 1.345424226717228, epsilon);
        }
    }

    {
        const Eigen::Vector3d mean = {3.0, 2.0, M_PI/3.0};
        Eigen::Matrix3d cov;
        cov << 1.0,  0.1, 0.05,
               0.1, 1.0, 0.2,
               0.05, 0.2, 0.1;
        ThreeDimensionalNormalDistribution dist(mean, cov);

        // E[XYcos(Z)cos(Z)]
        {
            const double exact_moment = dist.calc_xy_cos_z_cos_z_moment();
            EXPECT_NEAR(exact_moment, 1.3134247613800036, 0.001);
            std::cout << "E[xycoszcosz]: " << exact_moment << std::endl;
        }

        // E[XYsin(Z)sin(Z)]
        {
            const double exact_moment = dist.calc_xy_sin_z_sin_z_moment();
            EXPECT_NEAR(exact_moment,  4.786631915606325, 0.001);
            std::cout << "E[xysinzsinz]: " << exact_moment << std::endl;
        }

        // E[XYcos(Z)sin(Z)]
        {
            const double exact_moment = dist.calc_xy_cos_z_sin_z_moment();
            EXPECT_NEAR(exact_moment, 1.861773707968873, 0.001);
            std::cout << "E[xycoszsinz]: " << exact_moment << std::endl;
        }
    }
}
