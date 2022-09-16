import numpy as np
import matplotlib as plt
import math

num = 10000*10000

x_lambda = 1.0
lower_theta = -np.pi/3.0
upper_theta = np.pi/6.0
x_value = np.random.exponential(x_lambda, num)
theta_value = np.random.uniform(lower_theta, upper_theta, num)

# E[XY]
estimated_sum = 0.0
for (x, theta) in zip(x_value, theta_value):
    estimated_sum += x*theta
estimated_mean = estimated_sum / num
print("E[XY]: ", estimated_mean)


# E[Xcos(Y)]
estimated_sum = 0.0
for (x, theta) in zip(x_value, theta_value):
    estimated_sum += x*math.cos(theta)
estimated_mean = estimated_sum / num
print("E[Xcos(Y)]: ", estimated_mean)

# E[X*sin(Y)*cos(Y)]
estimated_sum = 0.0
for (x, theta) in zip(x_value, theta_value):
    estimated_sum += x*math.sin(theta)*math.cos(theta)
estimated_mean = estimated_sum / num
print("E[X*sin(Y)*cos(Y)]: ", estimated_mean)

'''
# E[Xsin(Y)]
estimated_sum = 0.0
for value in values:
    x = value[0]
    theta = value[1]
    estimated_sum += x*math.sin(theta)
estimated_mean = estimated_sum / num
print("E[Xsin(Y)]: ", estimated_mean)

# E[X*X*Y]
estimated_sum = 0.0
for value in values:
    x = value[0]
    theta = value[1]
    estimated_sum += x*x*theta
estimated_mean = estimated_sum / num
print("E[X*X*Y)]: ", estimated_mean)

# E[X*Y*Y]
estimated_sum = 0.0
for value in values:
    x = value[0]
    theta = value[1]
    estimated_sum += x*theta*theta
estimated_mean = estimated_sum / num
print("E[X*Y*Y]: ", estimated_mean)

# E[X*X*sin(Y)]
estimated_sum = 0.0
for value in values:
    x = value[0]
    theta = value[1]
    estimated_sum += x*x*math.sin(theta)
estimated_mean = estimated_sum / num
print("E[X*X*sin(Y)]: ", estimated_mean)

# E[X*X*cos(Y)]
estimated_sum = 0.0
for value in values:
    x = value[0]
    theta = value[1]
    estimated_sum += x*x*math.cos(theta)
estimated_mean = estimated_sum / num
print("E[X*X*cos(Y)]: ", estimated_mean)

# E[X*sin(Y)*cos(Y)]
estimated_sum = 0.0
for value in values:
    x = value[0]
    theta = value[1]
    estimated_sum += x*math.sin(theta)*math.cos(theta)
estimated_mean = estimated_sum / num
print("E[X*sin(Y)*cos(Y)]: ", estimated_mean)

# E[X*Y*sin(Y)]
estimated_sum = 0.0
for value in values:
    x = value[0]
    theta = value[1]
    estimated_sum += x*theta*math.sin(theta)
estimated_mean = estimated_sum / num
print("E[X*Y*sin(Y)]: ", estimated_mean)

# E[X*Y*cos(Y)]
estimated_sum = 0.0
for value in values:
    x = value[0]
    theta = value[1]
    estimated_sum += x*theta*math.cos(theta)
estimated_mean = estimated_sum / num
print("E[X*Y*cos(Y)]: ", estimated_mean)

# E[X*X*Y*Y]
estimated_sum = 0.0
for value in values:
    x = value[0]
    theta = value[1]
    estimated_sum += x*x*theta*theta
estimated_mean = estimated_sum / num
print("E[X*X*Y*Y]: ", estimated_mean)
'''