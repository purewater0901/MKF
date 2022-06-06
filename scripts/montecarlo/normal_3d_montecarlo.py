import math
import numpy as np

num = 10000 * 10000
"""
mean = np.array([1.59378, 1.3832, 0.680927])
cov = np.array([[0.00179042,  -0.00146877, -5.51844e-05],
                [-0.00146877, 0.00201589, 4.10447e-05],
                [-5.51844e-05,  4.10447e-05,  0.000123276]])
mean = np.array([10, 10, np.pi/3.0])
cov = np.array([[1.0**2,  1.0, 0.05],
                [1.0, 2.0**2, 0.2],
                [0.05,  0.2,  np.pi**2/100]])
"""
mean = np.array([3.0, 2.0, np.pi/3.0])
cov = np.array([[1.0,  0.1, 0.05],
                [0.1, 1.0, 0.2],
                [0.05,  0.2,  0.1]])
values = np.random.multivariate_normal(mean, cov, num)

xy_square_sum = 0.0
xz_square_sum = 0.0
yz_square_sum = 0.0
xy_sum = 0.0
xz_sum = 0.0
yz_sum = 0.0
x_sin_z_sum = 0.0
y_sin_z_sum = 0.0
x_cos_z_sum = 0.0
y_cos_z_sum = 0.0
xx_sin_z_sum = 0.0
yy_sin_z_sum = 0.0
xx_cos_z_sum = 0.0
yy_cos_z_sum = 0.0
xy_sin_z_sum = 0.0
xy_cos_z_sum = 0.0
xxy_cos_z_sum = 0.0
xy_cos_z_cos_z_sum = 0.0;
xy_sin_z_sin_z_sum = 0.0;
xy_cos_z_sin_z_sum = 0.0;

for i in range(num):
    '''
    xy_square_sum += values[i][0]*values[i][0]*values[i][1]*values[i][1]
    xz_square_sum += values[i][0]*values[i][0]*values[i][2]*values[i][2]
    yz_square_sum += values[i][1]*values[i][1]*values[i][2]*values[i][2]
    xy_sum += values[i][0]*values[i][1]
    xz_sum += values[i][0]*values[i][2]
    yz_sum += values[i][1]*values[i][2]
    x_sin_z_sum += values[i][0] * math.sin(values[i][2])
    y_sin_z_sum += values[i][1] * math.sin(values[i][2])
    x_cos_z_sum += values[i][0] * math.cos(values[i][2])
    y_cos_z_sum += values[i][1] * math.cos(values[i][2])
    xx_sin_z_sum += values[i][0]**2 * math.sin(values[i][2])
    yy_sin_z_sum += values[i][1]**2 * math.sin(values[i][2])
    xx_cos_z_sum += values[i][0]**2 * math.cos(values[i][2])
    yy_cos_z_sum += values[i][1]**2 * math.cos(values[i][2])
    xy_sin_z_sum += values[i][0] * values[i][1] * math.sin(values[i][2])
    xy_cos_z_sum += values[i][0] * values[i][1] * math.cos(values[i][2])
    xxy_cos_z_sum += values[i][0]**2 * values[i][1] * math.cos(values[i][2])
    '''
    xy_cos_z_cos_z_sum += values[i][0] * values[i][1] * math.cos(values[i][2])**2
    xy_sin_z_sin_z_sum += values[i][0] * values[i][1] * math.sin(values[i][2])**2
    xy_cos_z_sin_z_sum += values[i][0] * values[i][1] * math.cos(values[i][2]) * math.sin(values[i][2])

xy_square_mean = xy_square_sum / num
xz_square_mean = xz_square_sum / num
yz_square_mean = yz_square_sum / num
xy_mean = xy_sum / num
xz_mean = xz_sum / num
yz_mean = yz_sum / num
x_sin_z_mean = x_sin_z_sum / num
y_sin_z_mean = y_sin_z_sum / num
x_cos_z_mean = x_cos_z_sum / num
y_cos_z_mean = y_cos_z_sum / num
xx_sin_z_mean = xx_sin_z_sum / num
yy_sin_z_mean = yy_sin_z_sum / num
xx_cos_z_mean = xx_cos_z_sum / num
yy_cos_z_mean = yy_cos_z_sum / num
xy_sin_z_mean = xy_sin_z_sum / num
xy_cos_z_mean = xy_cos_z_sum / num
xxy_cos_z_mean = xxy_cos_z_sum / num
xy_cos_z_cos_z_mean = xy_cos_z_cos_z_sum / num
xy_sin_z_sin_z_mean = xy_sin_z_sin_z_sum / num
xy_cos_z_sin_z_mean = xy_cos_z_sin_z_sum / num

# E[XY]
print("E[XY]: ", xy_mean)

# E[XZ]
print("E[XZ]: ", xz_mean)

# E[YZ]
print("E[YZ]: ", yz_mean)

# E[Xsin(Z)]
print("E[Xsin(Z)]: ", x_sin_z_mean)

# E[Ysin(Z)]
print("E[Ysin(Z)]: ", y_sin_z_mean)

# E[Xcos(Z)]
print("E[Xcos(Z)]: ", x_cos_z_mean)

# E[Ycos(Z)]
print("E[Ycos(Z)]: ", y_cos_z_mean)

# E[XXsin(Z)]
print("E[XXsin(Z)]: ", xx_sin_z_mean)

# E[YYsin(Z)]
print("E[YYsin(Z)]: ", yy_sin_z_mean)

# E[XXcos(Z)]
print("E[XXcos(Z)]: ", xx_cos_z_mean)

# E[YYcos(Z)]
print("E[YYcos(Z)]: ", yy_cos_z_mean)

# E[XYsin(Z)]
print("E[XYsin(Z)]: ", xy_sin_z_mean)

# E[XYcos(Z)]
print("E[XYcos(Z)]: ", xy_cos_z_mean)

# E[XXYcos(Z)]
print("E[XXYcos(Z)]: ", xxy_cos_z_mean)

# E[XXYY]
print("E[XXYY]: ", xy_square_mean)

# E[XXZZ]
print("E[XXZZ]: ", xz_square_mean)

# E[YYZZ]
print("E[YYZZ]: ", yz_square_mean)

# E[XYcos(Z)^2]
print("E[XYcos(Z)^2]: ", xy_cos_z_cos_z_mean)
print("E[XYsin(Z)^2]: ", xy_sin_z_sin_z_mean)
print("E[XYcos(Z)sin(Z)]: ", xy_cos_z_sin_z_mean)
