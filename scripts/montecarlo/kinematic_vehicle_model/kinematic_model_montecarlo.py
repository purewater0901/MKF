import math
import numpy as np

def predict(x_current, u_current ,w_current, dt):
    a = u_current[0]
    u = u_current[1]
    wv = w_current[0]
    wyaw = w_current[1]

    x_next = np.zeros(x_current.size)
    x_next[0] = x_current[0] + x_current[2] * math.cos(x_current[3])*dt
    x_next[1] = x_current[1] + x_current[2] * math.sin(x_current[3])*dt
    x_next[2] = x_current[2] + (a + wv) * dt
    x_next[3] = x_current[3] + (u + wyaw) * dt

    return x_next

def update(x_current, w_current):
    wr = w_current[0]
    wv = w_current[1]
    wyaw = w_current[2]

    y = np.zeros(3)
    y[0] = x_current[0]**2 + x_current[1]**2 + wr
    y[1] = x_current[2] * math.cos(x_current[3]) * wv
    y[2] = x_current[3] + wyaw

    return y

dt = 0.1
num = 1000 * 10000

mean = np.array([0.0, 0.0, 5.0, 0.0])
cov = np.array([[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.5**2, 0.0],
                [0.0, 0.0, 0.0, (math.pi/10.0)**2]])

samples = np.random.multivariate_normal(mean, cov, num)
wv_samples = np.random.normal(0.0, 1.0*dt, num)
wyaw_samples = np.random.normal(0.0, 0.1*dt, num)

x_sum = 0.0
y_sum = 0.0
v_sum = 0.0
yaw_sum = 0.0
x_square_sum = 0.0
y_square_sum = 0.0
v_square_sum = 0.0
yaw_square_sum = 0.0
cos_yaw_square_sum = 0.0
sin_yaw_square_sum = 0.0
v_cos_yaw_square_sum = 0.0
v_sin_yaw_square_sum = 0.0
v_square_cos_yaw_sum = 0.0
v_square_sin_yaw_sum = 0.0
xy_sum = 0.0
xv_sum = 0.0
yv_sum = 0.0
xyaw_sum = 0.0
yyaw_sum = 0.0
vyaw_sum = 0.0
x_sin_yaw_sum = 0.0
y_sin_yaw_sum = 0.0
v_sin_yaw_sum = 0.0
x_cos_yaw_sum = 0.0
y_cos_yaw_sum = 0.0
v_cos_yaw_sum = 0.0
sin_yaw_cos_yaw_sum = 0.0

v1_c2_sum = 0.0
v1_s2_sum = 0.0
v1_c1_x1_sum = 0.0
v1_c1_y1_sum = 0.0
v1_c1_yaw1_sum = 0.0
v1_s1_x1_sum = 0.0
v1_s1_y1_sum = 0.0
v1_s1_yaw1_sum = 0.0
v1_c1_s1_sum = 0.0

v2_c1_s1_sum = 0.0

for i in range(0, num):
    x_current = samples[i]
    u_current = np.array([0.0, 0.0])
    w_current = np.array([wv_samples[i], wyaw_samples[i]])
    x_next = predict(x_current, u_current, w_current, dt)
    x_sum += x_next[0]
    y_sum += x_next[1]
    v_sum += x_next[2]
    yaw_sum += x_next[3]
    x_square_sum += x_next[0]**2
    y_square_sum += x_next[1]**2
    v_square_sum += x_next[2]**2
    yaw_square_sum += x_next[3]**2
    cos_yaw_square_sum += np.cos(x_next[3])**2
    sin_yaw_square_sum += np.sin(x_next[3])**2
    v_cos_yaw_square_sum += x_next[2]**2 * np.cos(x_next[3])**2
    v_sin_yaw_square_sum += x_next[2]**2 * np.sin(x_next[3])**2
    v_square_cos_yaw_sum += x_next[2]**2 * np.cos(x_next[3])
    v_square_sin_yaw_sum += x_next[2]**2 * np.sin(x_next[3])
    xy_sum += x_next[0]*x_next[1]
    xv_sum += x_next[0]*x_next[2]
    yv_sum += x_next[1]*x_next[2]
    xyaw_sum += x_next[0]*x_next[3]
    yyaw_sum += x_next[1]*x_next[3]
    vyaw_sum += x_next[2]*x_next[3]
    x_sin_yaw_sum += x_next[0] * math.sin(x_next[3])
    y_sin_yaw_sum += x_next[1] * math.sin(x_next[3])
    v_sin_yaw_sum += x_next[2] * math.sin(x_next[3])
    x_cos_yaw_sum += x_next[0] * math.cos(x_next[3])
    y_cos_yaw_sum += x_next[1] * math.cos(x_next[3])
    v_cos_yaw_sum += x_next[2] * math.cos(x_next[3])
    sin_yaw_cos_yaw_sum += math.sin(x_next[3]) * math.cos(x_next[3])

    v1_c2_sum += x_next[2] * math.cos(x_next[3])**2
    v1_s2_sum += x_next[2] * math.sin(x_next[3])**2
    v1_c1_x1_sum += x_next[2] * math.cos(x_next[3]) * x_next[0]
    v1_c1_y1_sum += x_next[2] * math.cos(x_next[3]) * x_next[1]
    v1_c1_yaw1_sum += x_next[2] * math.cos(x_next[3]) * x_next[3]
    v1_s1_x1_sum += x_next[2] * math.sin(x_next[3]) * x_next[0]
    v1_s1_y1_sum += x_next[2] * math.sin(x_next[3]) * x_next[1]
    v1_s1_yaw1_sum += x_next[2] * math.sin(x_next[3]) * x_next[3]
    v1_c1_s1_sum += x_next[2] * math.sin(x_next[3]) * math.cos(x_next[3])

    v2_c1_s1_sum += x_next[2] * math.cos(x_next[3]) * math.sin(x_next[3])

x_mean = x_sum / num
y_mean = y_sum / num
v_mean = v_sum / num
yaw_mean = yaw_sum / num
x_square_mean = x_square_sum / num
y_square_mean = y_square_sum / num
v_square_mean = v_square_sum / num
yaw_square_mean = yaw_square_sum / num
cos_yaw_square_mean = cos_yaw_square_sum / num
sin_yaw_square_mean = sin_yaw_square_sum / num
v_cos_yaw_square_mean = v_cos_yaw_square_sum / num
v_sin_yaw_square_mean = v_sin_yaw_square_sum / num
v_square_cos_yaw_mean = v_square_cos_yaw_sum / num
v_square_sin_yaw_mean = v_square_sin_yaw_sum / num
xy_mean = xy_sum / num
xv_mean = xv_sum / num
yv_mean = yv_sum / num
xyaw_mean = xyaw_sum / num
yyaw_mean = yyaw_sum / num
vyaw_mean = vyaw_sum / num
x_sin_yaw_mean = x_sin_yaw_sum / num
y_sin_yaw_mean = y_sin_yaw_sum / num
v_sin_yaw_mean = v_sin_yaw_sum / num
x_cos_yaw_mean = x_cos_yaw_sum / num
y_cos_yaw_mean = y_cos_yaw_sum / num
v_cos_yaw_mean = v_cos_yaw_sum / num
sin_yaw_cos_yaw_mean = sin_yaw_cos_yaw_sum / num

v1_c2_mean = v1_c2_sum / num
v1_s2_mean = v1_s2_sum / num
v1_c1_x1_mean = v1_c1_x1_sum / num
v1_c1_y1_mean = v1_c1_y1_sum / num
v1_c1_yaw1_mean = v1_c1_yaw1_sum / num
v1_s1_x1_mean = v1_s1_x1_sum / num
v1_s1_y1_mean = v1_s1_y1_sum / num
v1_s1_yaw1_mean = v1_s1_yaw1_sum / num
v1_c1_s1_mean = v1_c1_s1_sum / num

v2_c1_s1_mean = v2_c1_s1_sum / num

print("E[x]: ", x_mean)
print("E[y]: ", y_mean)
print("E[v]: ", v_mean)
print("E[yaw]: ", yaw_mean)
print("E[x^2]: ", x_square_mean)
print("E[y^2]: ", y_square_mean)
print("E[v^2]: ", v_square_mean)
print("E[yaw^2]: ", yaw_square_mean)
print("E[cos(yaw)^2]: ", cos_yaw_square_mean)
print("E[sin(yaw)^2]: ", sin_yaw_square_mean)
print("E[(vcos(yaw))^2]: ", v_cos_yaw_square_mean)
print("E[(vsin(yaw))^2]: ", v_sin_yaw_square_mean)
print("E[(v^2cos(yaw))]: ", v_square_cos_yaw_mean)
print("E[(v^2sin(yaw))]: ", v_square_sin_yaw_mean)
print("E[xy]: ", xy_mean)
print("E[xv]: ", xv_mean)
print("E[yv]: ", yv_mean)
print("E[xyaw]: ", xyaw_mean)
print("E[yyaw]: ", yyaw_mean)
print("E[vyaw]: ", vyaw_mean)
print("E[xsin(yaw)]: ", x_sin_yaw_mean)
print("E[ysin(yaw)]: ", y_sin_yaw_mean)
print("E[vsin(yaw)]: ", v_sin_yaw_mean)
print("E[xcos(yaw)]: ", x_cos_yaw_mean)
print("E[ycos(yaw)]: ", y_cos_yaw_mean)
print("E[vcos(yaw)]: ", v_cos_yaw_mean)
print("E[sin(yaw)cos(yaw)]: ", sin_yaw_cos_yaw_mean)

print("E[v1_c2]: ", v1_c2_mean)
print("E[v1_s2]: ", v1_s2_mean)
print("E[v1_c1_x1_mean]: ", v1_c1_x1_mean)
print("E[v1_c1_y1_mean]: ", v1_c1_y1_mean)
print("E[v1_c1_yaw1_mean]: ", v1_c1_yaw1_mean)
print("E[v1_s1_x1_mean]: ", v1_s1_x1_mean)
print("E[v1_s1_y1_mean]: ", v1_s1_y1_mean)
print("E[v1_s1_yaw1_mean]: ", v1_s1_yaw1_mean)
print("E[v1_c1_s1_mean]: ", v1_c1_s1_mean)

print("E[v2_c1_s1]: ", v2_c1_s1_mean)

print("sigma_x: ", x_square_mean-x_mean*x_mean)
print("sigma_y: ", y_square_mean-y_mean*y_mean)
print("sigma_yaw: ", yaw_square_mean-yaw_mean*yaw_mean)
print("sigma_xy: ", xy_mean -x_mean*y_mean)
print("sigma_xyaw: ", xyaw_mean -x_mean*yaw_mean)
print("sigma_yyaw: ", yyaw_mean -y_mean*yaw_mean)