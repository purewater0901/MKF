import math
import numpy as np

def update(x_current, v_current):
    vr = v_current[0]
    wtheta = v_current[1]

    y_next = np.zeros(2)
    y_next[0] = x_current[0]**2 + x_current[1]**2 + vr
    y_next[1] = x_current[2] + wtheta

    return y_next

dt = 0.1
num = 1000 * 10000

predicted_mean = np.array([1.59378, 1.3832, 0.680927])
predicted_cov = np.array([[0.00179042,  -0.00146877, -5.51844e-05],
                          [-0.00146877, 0.00201589, 4.10447e-05],
                          [-5.51844e-05,  4.10447e-05,  0.000123276]])

values = np.random.multivariate_normal(predicted_mean, predicted_cov, num)
vr_samples = np.random.uniform(0.0, 1.0, num)
vtheta_samples = np.random.uniform(-math.pi/100, math.pi/100, num)

or_sum = 0.0
otheta_sum = 0.0
or_square_sum = 0.0
otheta_square_sum = 0.0
rtheta_sum = 0.0
xpr_sum = 0.0
ypr_sum = 0.0
thetapr_sum = 0.0
xptheta_sum = 0.0
yptheta_sum = 0.0
thetaptheta_sum = 0.0
xy_square_sum = 0.0
xy_sum = 0.0
xz_sum = 0.0
yz_sum = 0.0
for i in range(num):
    x_current = values[i]
    v_current = np.array([vr_samples[i], vtheta_samples[i]])
    y_next = update(x_current, v_current)
    or_sum = or_sum + y_next[0]
    otheta_sum = otheta_sum + y_next[1]
    or_square_sum = or_square_sum + y_next[0]**2
    otheta_square_sum = otheta_square_sum + y_next[1]*y_next[1]
    rtheta_sum = rtheta_sum + y_next[0]*y_next[1]
    xpr_sum = xpr_sum + values[i][0]*y_next[0]
    ypr_sum = ypr_sum + values[i][1]*y_next[0]
    thetapr_sum = thetapr_sum + values[i][2] * y_next[0]
    xptheta_sum = xptheta_sum + values[i][0] * y_next[1]
    yptheta_sum = yptheta_sum + values[i][1] * y_next[1]
    thetaptheta_sum = thetaptheta_sum + values[i][2]*y_next[1]
    xy_square_sum += values[i][0]*values[i][0]*values[i][1]*values[i][1]
    xy_sum += values[i][0]*values[i][1]
    xz_sum += values[i][0]*values[i][2]
    yz_sum += values[i][1]*values[i][2]

or_mean = or_sum/num
otheta_mean = otheta_sum/num
or_square_mean = or_square_sum / num
otheta_square_mean = otheta_square_sum / num
rtheta_mean = rtheta_sum / num
xy_square_mean = xy_square_sum / num
xy_mean = xy_sum / num
xz_mean = xz_sum / num
yz_mean = yz_sum / num

xpr_mean = xpr_sum / num
ypr_mean = ypr_sum / num
thetapr_mean = thetapr_sum / num
xptheta_mean = xptheta_sum / num
yptheta_mean = yptheta_sum / num
thetaptheta_mean = thetaptheta_sum / num

print("E[r]: ", or_mean)
print("E[theta]: ", otheta_mean)
print("E[r^2]: ", or_square_mean)
print("E[theta^2]: ", otheta_square_mean)
print("E[rtheta]: ", rtheta_mean)
print("r_cov: ", or_square_mean - or_mean*or_mean)
print("theta_cov: ", otheta_square_mean - otheta_mean*otheta_mean)
print("rtheta: ", rtheta_mean - or_mean*otheta_mean)
print("xy_square_mean: ", xy_square_mean)
print("xy_mean: ", xy_mean)
print("xz_mean: ", xz_mean)
print("yz_mean: ", yz_mean)

print("E[XR]: ", xpr_mean)
print("E[YR]: ", ypr_mean)
print("E[THETAR]: ", thetapr_mean)
print("E[XTHETA]: ", xptheta_mean)
print("E[YTHETA]: ", yptheta_mean)
print("E[THETATHETA]: ", thetaptheta_mean)

print("sigma[XR]: ", xpr_mean - predicted_mean[0] * or_mean)
print("sigma[XTHETA]: ", xptheta_mean - predicted_mean[0] * otheta_mean)
print("sigma[YR]: ", ypr_mean - predicted_mean[1] * or_mean)
print("sigma[YTHETA]: ", yptheta_mean - predicted_mean[1] * otheta_mean)
print("sigma[THETAR]: ", thetapr_mean - predicted_mean[2] * or_mean)
print("sigma[THETATHETA]: ", thetaptheta_mean - predicted_mean[2] * otheta_mean)

Sigma_yy = np.array([[or_square_mean - or_mean*or_mean, rtheta_mean - or_mean*otheta_mean],
                     [rtheta_mean - or_mean*otheta_mean, otheta_square_mean - otheta_mean*otheta_mean]])

Sigma_xy = np.array([[xpr_mean - predicted_mean[0] * or_mean, xptheta_mean - predicted_mean[0] * otheta_mean],
                     [ypr_mean - predicted_mean[1] * or_mean, yptheta_mean - predicted_mean[1] * otheta_mean],
                     [thetapr_mean - predicted_mean[2] * or_mean, thetaptheta_mean - predicted_mean[2] * otheta_mean]])

K = np.dot(Sigma_xy, np.linalg.inv(Sigma_yy))

updated_cov = predicted_cov - np.dot(np.dot(K, Sigma_yy), K.transpose())

print("K: ", K)
print("updated_cov: ", updated_cov)
