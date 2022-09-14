import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from scipy import stats
import platform
import os
from scipy import stats

if __name__ == '__main__':
    filename = "distribution_approximation"
    os.chdir('../../')
    path = os.getcwd()

    x = np.linspace(-7, 8, 500)
    pi_k = np.array([0.4, 0.3, 0.5])
    means = np.array([-3, 4, 1])
    norm1 = stats.norm.pdf(x, loc=means[0], scale=1)
    norm2 = stats.norm.pdf(x, loc=means[1], scale=1)
    norm3 = stats.norm.pdf(x, loc=means[2], scale=2)
    mixed_pdf = pi_k[0]*norm1 + pi_k[1]*norm2 + pi_k[2]*norm3
    mean = np.dot(pi_k, means)
    app_norm = stats.norm.pdf(x, loc=mean, scale=1.5)
    app_norm = stats.norm.pdf(x, loc=mean, scale=3)
    extended_x = np.linspace(-7, 9, 600)
    ukf_app_norm = stats.norm.pdf(extended_x, loc=mean + 1.0, scale=2)
    print(mean)

    fig = plt.figure(figsize=(8.5, 8.5))
    plt.plot(x, mixed_pdf, color="black", label=r"$x_k$", linewidth=4)
    plt.axvline(mean + 0.7, 0.0, 1.0, color="blue", linestyle="dashed", linewidth=3)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.axis("off")

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.axis("off")
    #plt.rc('legend', fontsize=20)
    #plt.legend()

    plt.savefig(path + "/result/picture/" + filename + ".eps", bbox_inches="tight", pad_inches=0.05)
    plt.show()
