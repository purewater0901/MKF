import math
import numpy as np

def update(x_current, w_current):
    wx = w_current[0]
    wy = w_current[1]
    wv = w_current[2]
    wyaw = w_current[3]

    y = np.zeros(3)
    y[0] = x_current[0] + x_current[2] * wx
    y[1] = x_current[1] + x_current[2] * wy
    y[2] = (x_current[2] + wv) * math.cos(x_current[3] + wyaw)

    return y

dt = 0.1
num = 10000 * 1000

mean = np.array([0.211074, 0.211074, 3.003, 0.790398])
cov = np.array([[0.0104978, -0.000394292,   0.00070358,  -0.00211074],
                [-0.000394292, 0.0104978,   0.00070358,   0.00211074],
                [0.00070358,   0.00070358,    0.0133333, -4.44089e-16],
                [-0.00211074,   0.00211074, -4.44089e-16,     0.010075]])

samples = np.random.multivariate_normal(mean, cov, num)
wx_samples = np.random.uniform(-1.5, 2.0, num)
wy_samples = np.random.uniform(-2.0, 1.5, num)
wv_samples = np.random.exponential(2.0, num)
wyaw_samples = np.random.uniform(-math.pi/20.0, math.pi/10.0, num)

cyawPow2_sum = 0.0
syawPow2_sum = 0.0
xPow1_cyawPow1_sum = 0.0
xPow1_syawPow1_sum = 0.0
yPow1_cyawPow1_sum = 0.0
yPow1_syawPow1_sum = 0.0
vPow1_cyawPow1_sum = 0.0
vPow1_syawPow1_sum = 0.0
vPow2_cyawPow1_sum = 0.0
vPow2_syawPow1_sum = 0.0
cyawPow1_syawPow1_sum = 0.0
xPow1_vPow1_syawPow1_sum = 0.0
xPow1_vPow1_cyawPow1_sum = 0.0
wxPow1_sum = 0.0
wyPow1_sum = 0.0
wvPow1_sum = 0.0
swyawPow1_sum = 0.0
cwyawPow1_sum = 0.0
wxPow2_sum = 0.0
wyPow2_sum = 0.0
wvPow2_sum = 0.0

for i in range(0, num):
    x_current = samples[i]
    '''
    cyawPow2_sum += math.cos(x_current[3])**2
    syawPow2_sum += math.sin(x_current[3])**2
    xPow1_cyawPow1_sum += x_current[0] * math.cos(x_current[3])
    xPow1_syawPow1_sum += x_current[0] * math.sin(x_current[3])
    yPow1_cyawPow1_sum += x_current[1] * math.cos(x_current[3])
    yPow1_syawPow1_sum += x_current[1] * math.sin(x_current[3])
    vPow1_cyawPow1_sum += x_current[2] * math.cos(x_current[3])
    vPow1_syawPow1_sum += x_current[2] * math.sin(x_current[3])
    cyawPow1_syawPow1_sum = math.cos(x_current[3]) * math.sin(x_current[3])
    xPow1_vPow1_syawPow1_sum += x_current[0] * x_current[2] * math.sin(x_current[3])
    xPow1_vPow1_cyawPow1_sum += x_current[0] * x_current[2] * math.cos(x_current[3])
    '''
    vPow2_cyawPow1_sum += x_current[2]**2 * math.cos(x_current[3])
    vPow2_syawPow1_sum += x_current[2]**2 * math.sin(x_current[3])

    '''
    wxPow1_sum += wx_samples[i]
    wyPow1_sum += wy_samples[i]
    wvPow1_sum += wv_samples[i]
    swyawPow1_sum += math.sin(wyaw_samples[i])
    cwyawPow1_sum += math.cos(wyaw_samples[i])
    wxPow2_sum += wx_samples[i]**2
    wyPow2_sum += wy_samples[i]**2
    wvPow2_sum += wv_samples[i]**2
    '''

print("E[cos(theta)^2]: ", cyawPow2_sum/num)
print("E[sin(theta)^2]: ", syawPow2_sum/num)
print("E[xcos(theta)]: ", xPow1_cyawPow1_sum/num)
print("E[xsin(theta)]: ", xPow1_syawPow1_sum/num)
print("E[ycos(theta)]: ", yPow1_cyawPow1_sum/num)
print("E[ysin(theta)]: ", yPow1_syawPow1_sum/num)
print("E[vcos(theta)]: ", vPow1_cyawPow1_sum/num)
print("E[vsin(theta)]: ", vPow1_syawPow1_sum/num)
print("E[v^2cos(theta)]: ", vPow2_cyawPow1_sum/num)
print("E[v^2sin(theta)]: ", vPow2_syawPow1_sum/num)
print("E[cos(theta)sin(theta)]: ", cyawPow1_syawPow1_sum/num)
print("E[xvsin(theta)]: ", xPow1_vPow1_syawPow1_sum/num)
print("E[xvcos(theta)]: ", xPow1_vPow1_cyawPow1_sum/num)

print("E[wx]: ", wxPow1_sum/num)
print("E[wy]: ", wyPow1_sum/num)
print("E[wv]: ", wvPow1_sum/num)
print("E[sin(wyaw)]: ", swyawPow1_sum/num)
print("E[cos(wyaw)]: ", cwyawPow1_sum/num)
print("E[wx^2]: ", wxPow2_sum/num)
print("E[wy^2]: ", wyPow2_sum/num)
print("E[wv^2]: ", wvPow2_sum/num)

x_sum = 0.0
y_sum = 0.0
vc_sum = 0.0
xPow2_sum = 0.0
yPow2_sum = 0.0
vcPow2_sum = 0.0
xy_sum = 0.0
xvc_sum = 0.0
yvc_sum = 0.0

xp_xm_sum = 0.0
xp_ym_sum = 0.0
xp_vcm_sum = 0.0
yp_xm_sum = 0.0
yp_ym_sum = 0.0
yp_vcm_sum = 0.0
vp_xm_sum = 0.0
vp_ym_sum = 0.0
vp_vcm_sum = 0.0
yawp_xm_sum = 0.0
yawp_ym_sum = 0.0
yawp_vcm_sum = 0.0

for i in range(0, num):
    x_current = samples[i]
    noise = np.array([wx_samples[i], wy_samples[i], wv_samples[i], wyaw_samples[i]])
    y = update(x_current, noise)
    x_sum += y[0]
    y_sum += y[1]
    vc_sum += y[2]
    xPow2_sum += y[0]**2
    yPow2_sum += y[1]**2
    vcPow2_sum += y[2]**2
    xy_sum += y[0] * y[1]
    xvc_sum += y[0] * y[2]
    yvc_sum += y[1] * y[2]

    xp = samples[i][0]
    yp = samples[i][1]
    vp = samples[i][2]
    yawp = samples[i][3]
    xp_xm_sum += xp * y[0]
    xp_ym_sum += xp * y[1]
    xp_vcm_sum += xp * y[2]
    yp_xm_sum += yp * y[0]
    yp_ym_sum += yp * y[1]
    yp_vcm_sum += yp * y[2]
    vp_xm_sum += vp * y[0]
    vp_ym_sum += vp * y[1]
    vp_vcm_sum += vp * y[2]
    yawp_xm_sum += yawp * y[0]
    yawp_ym_sum += yawp * y[1]
    yawp_vcm_sum += yawp * y[2]


x_mean = x_sum / num
y_mean = y_sum / num
vc_mean = vc_sum / num
xPow2_mean = xPow2_sum / num
yPow2_mean = yPow2_sum / num
vcPow2_mean = vcPow2_sum / num
xy_mean = xy_sum / num
xvc_mean = xvc_sum / num
yvc_mean = yvc_sum / num

xp_xm_mean = xp_xm_sum / num
xp_ym_mean = xp_ym_sum / num
xp_vcm_mean = xp_vcm_sum / num
yp_xm_mean = yp_xm_sum / num
yp_ym_mean = yp_ym_sum / num
yp_vcm_mean = yp_vcm_sum / num
vp_xm_mean = vp_xm_sum / num
vp_ym_mean = vp_ym_sum / num
vp_vcm_mean = vp_vcm_sum / num
yawp_xm_mean = yawp_xm_sum / num
yawp_ym_mean = yawp_ym_sum / num
yawp_vcm_mean = yawp_vcm_sum / num

print("E[x]: ", x_mean)
print("E[y]: ", y_mean)
print("E[vc]: ", vc_mean)
print("E[x^2]: ", xPow2_mean)
print("E[y^2]: ", yPow2_mean)
print("E[vc^2]: ", vcPow2_mean)
print("E[xy]: ", xy_mean)
print("E[xvc]: ", xvc_mean)
print("E[yvc]: ", yvc_mean)

print("Sigma[x_p_x_m]: ", xp_xm_mean - x_mean * mean[0])
print("Sigma[x_p_y_m]: ", xp_ym_mean - y_mean * mean[0])
print("Sigma[x_p_vc_m]: ", xp_vcm_mean - vc_mean * mean[0])
print("Sigma[y_p_x_m]: ", yp_xm_mean - x_mean * mean[1])
print("Sigma[y_p_y_m]: ", yp_ym_mean - y_mean * mean[1])
print("Sigma[y_p_vc_m]: ", yp_vcm_mean - vc_mean * mean[1])
print("Sigma[v_p_x_m]: ", vp_xm_mean - x_mean * mean[2])
print("Sigma[v_p_y_m]: ", vp_ym_mean - y_mean * mean[2])
print("Sigma[v_p_vc_m]: ", vp_vcm_mean - vc_mean * mean[2])
print("Sigma[yaw_p_x_m]: ", yawp_xm_mean - x_mean * mean[3])
print("Sigma[yaw_p_y_m]: ", yawp_ym_mean - y_mean * mean[3])
print("Sigma[yaw_p_vc_m]: ", yawp_vcm_mean - vc_mean * mean[3])
