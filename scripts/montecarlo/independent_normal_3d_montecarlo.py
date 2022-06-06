import math
import numpy as np

num = 10000*10000

mean = [1.0, 1.0, math.pi/4.0]
cov = [[1.0, 0.05, 0.0],
       [0.05, 1.0, 0.2],
       [0.0, 0.2, 0.1]]
#cov = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.1]]
values = np.random.multivariate_normal(mean, cov, num)

# E[XXYcos(z)]
'''
estimated_sum = 0.0
for value in values:
    x = value[0]
    y = value[1]
    theta = value[2]
    estimated_sum += x*x*y*math.cos(theta)
estimated_mean = estimated_sum / num
print("E[XXYcos(z)]: ", estimated_mean)
'''

# E[XYcos(z)]
estimated_sum = 0.0
for value in values:
    x = value[0]
    y = value[1]
    theta = value[2]
    estimated_sum += x*y*math.cos(theta)
estimated_mean = estimated_sum / num
print("E[XYcos(z)]: ", estimated_mean)


# E[XYsin(z)]
estimated_sum = 0.0
for value in values:
    x = value[0]
    y = value[1]
    theta = value[2]
    estimated_sum += x*y*math.sin(theta)
estimated_mean = estimated_sum / num
print("E[XYsin(z)]: ", estimated_mean)