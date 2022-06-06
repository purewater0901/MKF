import math
import numpy as np

def predict(x_current, u_current ,w_current, dt):
    v = u_current[0]
    u = u_current[1]
    wx = w_current[0]
    wy = w_current[1]
    wyaw = w_current[2]

    x_next = np.zeros(x_current.size)
    x_next[0] = x_current[0] + v * math.cos(x_current[2])*dt + wx
    x_next[1] = x_current[1] + v * math.sin(x_current[2])*dt + wy
    x_next[2] = x_current[2] + u * dt + wyaw

    return x_next

def update(x_current, v_current):
    vr = v_current[0]
    wyaw = v_current[1]

    y_next = np.zeros(2)
    y_next[0] = x_current[0]**2 + x_current[1]**2 + vr
    y_next[1] = x_current[2] + wyaw

    return y_next


dt = 0.1
num = 1000 * 10000

x_samples = np.random.normal(0.0, 0.1, num)
y_samples = np.random.normal(0.0, 0.1, num)
yaw_samples = np.random.normal(math.pi/4.0, 0.1, num)
wx_samples = np.random.normal(0.0, 0.1, num)
wy_samples = np.random.normal(0.0, 0.1, num)
wyaw_samples = np.random.normal(0.0, 0.05, num)

x_sum = 0.0
y_sum = 0.0
yaw_sum = 0.0
x_square_sum = 0.0
y_square_sum = 0.0
yaw_square_sum = 0.0
xy_sum = 0.0
xyaw_sum = 0.0
yyaw_sum = 0.0
x_sin_yaw_sum = 0.0
y_sin_yaw_sum = 0.0
x_cos_yaw_sum = 0.0
y_cos_yaw_sum = 0.0
sin_yaw_cos_yaw_sum = 0.0

for i in range(0, num):
    x_current = np.array([x_samples[i], y_samples[i], yaw_samples[i]])
    u_current = np.array([2.0, 0.0])
    w_current = np.array([wx_samples[i], wy_samples[i], wyaw_samples[i]])
    x_next = predict(x_current, u_current, w_current, dt)
    x_sum += x_next[0]
    y_sum += x_next[1]
    yaw_sum += x_next[2]
    x_square_sum += x_next[0] * x_next[0]
    y_square_sum += x_next[1] * x_next[1]
    yaw_square_sum += x_next[2] * x_next[2]
    xy_sum += x_next[0]*x_next[1]
    xyaw_sum += x_next[0]*x_next[2]
    yyaw_sum += x_next[1]*x_next[2]
    x_sin_yaw_sum += x_next[0] * math.sin(x_next[2])
    y_sin_yaw_sum += x_next[1] * math.sin(x_next[2])
    x_cos_yaw_sum += x_next[0] * math.cos(x_next[2])
    y_cos_yaw_sum += x_next[1] * math.cos(x_next[2])
    sin_yaw_cos_yaw_sum += math.sin(x_next[2]) * math.cos(x_next[2])

x_mean = x_sum / num
y_mean = y_sum / num
yaw_mean = yaw_sum / num
x_square_mean = x_square_sum / num
y_square_mean = y_square_sum / num
yaw_square_mean = yaw_square_sum / num
xy_mean = xy_sum / num
xyaw_mean = xyaw_sum / num
yyaw_mean = yyaw_sum / num
x_sin_yaw_mean = x_sin_yaw_sum / num
y_sin_yaw_mean = y_sin_yaw_sum / num
x_cos_yaw_mean = x_cos_yaw_sum / num
y_cos_yaw_mean = y_cos_yaw_sum / num
sin_yaw_cos_yaw_mean = sin_yaw_cos_yaw_sum / num

print("E[x]: ", x_mean)
print("E[y]: ", y_mean)
print("E[yaw]: ", yaw_mean)
print("E[x^2]: ", x_square_mean)
print("E[y^2]: ", y_square_mean)
print("E[yaw^2]: ", yaw_square_mean)
print("E[xy]: ", xy_mean)
print("E[xyaw]: ", xyaw_mean)
print("E[yyaw]: ", yyaw_mean)
print("E[xsin(yaw)]: ", x_sin_yaw_mean)
print("E[ysin(yaw)]: ", y_sin_yaw_mean)
print("E[xcos(yaw)]: ", x_cos_yaw_mean)
print("E[ycos(yaw)]: ", y_cos_yaw_mean)
print("E[sin(yaw)cos(yaw)]: ", sin_yaw_cos_yaw_mean)
print("sigma_x: ", x_square_mean-x_mean*x_mean)
print("sigma_y: ", y_square_mean-y_mean*y_mean)
print("sigma_yaw: ", yaw_square_mean-yaw_mean*yaw_mean)
print("sigma_xy: ", xy_mean -x_mean*y_mean)
print("sigma_xyaw: ", xyaw_mean -x_mean*yaw_mean)
print("sigma_yyaw: ", yyaw_mean -y_mean*yaw_mean)
