import math
import numpy as np

def update(x_current, w_current):
    wr = w_current[0]
    wv = w_current[1]
    wyaw = w_current[2]

    y = np.zeros(3)
    y[0] = x_current[0]**2 + x_current[1]**2 + wr
    y[1] = x_current[2] * math.cos(x_current[3]) + wv
    y[2] = x_current[3] + wyaw

    return y

dt = 0.1
num = 10000 * 10000

mean = np.array([0.395004, 0.407541, 3.00258, 0.811033])
cov = np.array([[0.011991, -0.00157717,  0.00140716, -0.00422148],
                [-0.00157717, 0.011991,  0.00140716,  0.00422148],
                [0.00140716,  0.00140716,      0.02,           0],
                [-0.00422148,  0.00422148,        0,      0.0109]])

samples = np.random.multivariate_normal(mean, cov, num)
wr_samples = np.random.normal(100.0, 10.5, num)
wv_samples = np.random.normal(0.0, 0.3, num)
wyaw_samples = np.random.normal(0.0, math.pi/10.0, num)

r_sum = 0.0
vc_sum = 0.0
yaw_sum = 0.0
rPow2_sum = 0.0
vcPow2_sum = 0.0
yawPow2_sum = 0.0
rvc_sum = 0.0
ryaw_sum = 0.0
vcyaw_sum = 0.0

for i in range(0, num):
    x_current = samples[i]
    noise = np.array([wr_samples[i], wv_samples[i], wyaw_samples[i]])
    y = update(x_current, noise)
    r_sum += y[0]
    vc_sum += y[1]
    yaw_sum += y[2]
    rPow2_sum += y[0]**2
    vcPow2_sum += y[1]**2
    yawPow2_sum += y[2]**2
    rvc_sum += y[0] * y[1]
    ryaw_sum += y[0] * y[2]
    vcyaw_sum += y[1] * y[2]

r_mean = r_sum / num
vc_mean = vc_sum / num
yaw_mean = yaw_sum / num
rPow2_mean = rPow2_sum / num
vcPow2_mean = vcPow2_sum / num
yawPow2_mean = yawPow2_sum / num
rvc_mean = rvc_sum / num
ryaw_mean = ryaw_sum / num
vcyaw_mean = vcyaw_sum / num

print("E[r]: ", r_mean)
print("E[vc]: ", vc_mean)
print("E[yaw]: ", yaw_mean)
print("E[r^2]: ", rPow2_mean)
print("E[vc^2]: ", vcPow2_mean)
print("E[yaw^2]: ", yawPow2_mean)
print("E[rvc]: ", rvc_mean)
print("E[ryaw]: ", ryaw_mean)
print("E[vcyaw]: ", vcyaw_mean)
