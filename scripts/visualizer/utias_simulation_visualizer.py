import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from scipy import stats
import platform
import os

if __name__ == '__main__':
    robot_num = 2
    gaussian_filename = "/simple_vehicle_gaussian"
    non_gaussian_filename = "/simple_vehicle_non_gaussian"
    output_filename = "/utlas_simulation_result"
    os.chdir('../../')
    path = os.getcwd()
    gaussian_data = pd.read_csv(path + "/result/data/robot" + str(robot_num) + gaussian_filename + ".csv")
    non_gaussian_data = pd.read_csv(path + "/result/data/robot" + str(robot_num) + non_gaussian_filename + ".csv")
    flag_visualize_trajectory = True

    fig = plt.figure(figsize=(15.5,8.5))
    plt.subplots_adjust(wspace=0.4, hspace=0.8)
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams["font.size"] = 28
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['ytick.labelsize'] = 30
    plt.rcParams['xtick.direction'] = 'in' # x axis in
    plt.rcParams['ytick.direction'] = 'in' # y axis in
    plt.rcParams['font.family'] = 'Times New Roman'

    if flag_visualize_trajectory:
        ax1 = fig.add_subplot(121)
        ax1.plot(gaussian_data["x_true"], gaussian_data["y_true"], color="black", linewidth=2.0, label="True")
        ax1.plot(gaussian_data["nkf_x"], gaussian_data["nkf_y"], color="red", linewidth=2.0, label="MKF")
        ax1.plot(gaussian_data["ekf_x"], gaussian_data["ekf_y"], color="blue", linewidth=1.5, label="EKF", linestyle="dashed")
        ax1.plot(gaussian_data["ukf_x"], gaussian_data["ukf_y"], color="green", linewidth=1.5, label="UKF", linestyle="dashdot")
        ax1.set_xlabel(r"x[m]", fontsize=40)
        ax1.set_ylabel(r"y[m]", fontsize=40)
        ax1.legend(fontsize=25)
        ax1.set_title("Gaussian Measurements")

        ax2 = fig.add_subplot(122)
        ax2.plot(non_gaussian_data["x_true"], non_gaussian_data["y_true"], color="black", linewidth=2.0, label="True")
        ax2.plot(non_gaussian_data["nkf_x"], non_gaussian_data["nkf_y"], color="red", linewidth=2.0, label="MKF")
        ax2.plot(non_gaussian_data["ekf_x"], non_gaussian_data["ekf_y"], color="blue", linewidth=1.5, label="EKF", linestyle="dashed")
        ax2.plot(non_gaussian_data["ukf_x"], non_gaussian_data["ukf_y"], color="green", linewidth=1.5, label="UKF", linestyle="dashdot")
        ax2.set_xlabel(r"x[m]", fontsize=40)
        ax2.set_ylabel(r"y[m]", fontsize=40)
        ax2.legend(fontsize=25)
        ax2.set_title("Non-Gaussian Measurements")

        output_filename += "_trajectory"
    else:
        ax1 = fig.add_subplot(211)
        ax1.plot(gaussian_data["time"], gaussian_data["nkf_xy_error"], color="red", label="MKF")
        ax1.plot(gaussian_data["time"], gaussian_data["ekf_xy_error"], color="blue", label="EKF")
        ax1.plot(gaussian_data["time"], gaussian_data["ukf_xy_error"], color="green", label="UKF")
        ax1.set_xlabel(r'time[$s$]', fontsize=30)
        ax1.set_ylabel(r'position[$m$]', fontsize=30)
        ax1.legend(loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=4, fontsize=30)

        ax2 = fig.add_subplot(212)
        ax2.plot(gaussian_data["time"], gaussian_data["nkf_yaw_error"], color="red")
        ax2.plot(gaussian_data["time"], gaussian_data["ekf_yaw_error"], color="blue")
        ax2.plot(gaussian_data["time"], gaussian_data["ukf_yaw_error"], color="green")
        ax2.set_xlabel(r'time[$s$]', fontsize=30)
        ax2.set_ylabel(r'yaw[$rad$]', fontsize=30)

    plt.savefig(path + "/result/picture/robot" + str(robot_num) + output_filename + ".png")
    plt.show()