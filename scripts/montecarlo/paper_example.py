import numpy as np
import matplotlib as plt
import math

num = 10000*10000

mean = [10.0, math.pi/3.0]
cov = [[5.0, 0.1], [0.1, (math.pi/6.0)]]
values = np.random.multivariate_normal(mean, cov, num)

# E[xy]
estimated_sum = 0.0
for value in values:
    x = value[0]
    theta = value[1]
    estimated_sum += x*theta
estimated_mean = estimated_sum / num
print("E[xy]: ", estimated_mean)

# E[Xcos(theta)]
estimated_sum = 0.0
for value in values:
    x = value[0]
    theta = value[1]
    estimated_sum += x*math.cos(theta)
estimated_mean = estimated_sum / num
print("E[Xcos(theta)]: ", estimated_mean)

# E[Xcos(theta)sin(theta)]
estimated_sum = 0.0
for value in values:
    x = value[0]
    theta = value[1]
    estimated_sum += x*math.cos(theta)*math.sin(theta)
estimated_mean = estimated_sum / num
print("E[Xcos(theta)sin(theta)]: ", estimated_mean)
'''
mean = [3.0, math.pi/6.0]
cov = [[2.0, 0.1], [0.1, (math.pi/10.0)]]
values = np.random.multivariate_normal(mean, cov, num)

# E[Xcos(theta)]
estimated_sum = 0.0
for value in values:
    x = value[0]
    theta = value[1]
    estimated_sum += x*math.cos(theta)
estimated_mean = estimated_sum / num
print("E[Xcos(theta)]: ", estimated_mean)

mean = [10.0, 5.0, math.pi/3.0]
cov = [[3.0, 0.5, 0.5],
       [0.5, 2.0, 0.3],
       [0.5, 0.3, math.pi/10.0]]
values = np.random.multivariate_normal(mean, cov, num)

# E[XXYcos(theta)]
estimated_sum = 0.0
for value in values:
    x = value[0]
    y = value[1]
    theta = value[2]
    estimated_sum += x**2*y*math.cos(theta)
estimated_mean = estimated_sum / num
print("E[XXYcos(theta)]: ", estimated_mean)
'''
