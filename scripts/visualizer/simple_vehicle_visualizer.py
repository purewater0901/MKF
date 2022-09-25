import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from scipy import stats
import platform
import os
import matplotlib

if __name__ == '__main__':
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    robot_num = 2
    filename = "/simple_vehicle_non_gaussian"
    os.chdir('../../')
    path = os.getcwd()
    data = pd.read_csv(path + "/result/data/robot" + str(robot_num) + filename + ".csv")
    flag_visualize_trajectory = False

    fig = plt.figure(figsize=(8.5,11))
    plt.subplots_adjust(wspace=0.4, hspace=0.8)
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams["font.size"] = 28
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['ytick.labelsize'] = 30
    plt.rcParams['xtick.direction'] = 'in' # x axis in
    plt.rcParams['ytick.direction'] = 'in' # y axis in
    plt.rcParams['font.family'] = 'Times New Roman'

    nkf_xy_error_mean = data['nkf_xy_error'].mean()
    ekf_xy_error_mean = data['ekf_xy_error'].mean()
    ukf_xy_error_mean = data['ukf_xy_error'].mean()
    nkf_xy_error_max = data['nkf_xy_error'].max()
    ekf_xy_error_max = data['ekf_xy_error'].max()
    ukf_xy_error_max = data['ukf_xy_error'].max()
    nkf_yaw_error_mean = data['nkf_yaw_error'].mean()
    ekf_yaw_error_mean = data['ekf_yaw_error'].mean()
    ukf_yaw_error_mean = data['ukf_yaw_error'].mean()
    nkf_yaw_error_max = data['nkf_yaw_error'].max()
    ekf_yaw_error_max = data['ekf_yaw_error'].max()
    ukf_yaw_error_max = data['ukf_yaw_error'].max()
    print("nkf_xy_error_mean: ", nkf_xy_error_mean)
    print("ekf_xy_error_mean: ", ekf_xy_error_mean)
    print("ukf_xy_error_mean: ", ukf_xy_error_mean)
    print("nkf_xy_error_max: ", nkf_xy_error_max)
    print("ekf_xy_error_max: ", ekf_xy_error_max)
    print("ukf_xy_error_max: ", ukf_xy_error_max)
    print("nkf_yaw_error_mean: ", nkf_yaw_error_mean)
    print("ekf_yaw_error_mean: ", ekf_yaw_error_mean)
    print("ukf_yaw_error_mean: ", ukf_yaw_error_mean)
    print("nkf_yaw_error_max: ", nkf_yaw_error_max)
    print("ekf_yaw_error_max: ", ekf_yaw_error_max)
    print("ukf_yaw_error_max: ", ukf_yaw_error_max)

    if flag_visualize_trajectory:
        plt.plot(data["x_true"], data["y_true"], color="black", linewidth=3.5, label="True")
        plt.plot(data["nkf_x"], data["nkf_y"], color="red", linewidth=4.0, label="MKF")
        plt.plot(data["ekf_x"], data["ekf_y"], color="blue", linewidth=3.5, label="EKF", linestyle="dotted")
        plt.plot(data["ukf_x"], data["ukf_y"], color="green", linewidth=3.5, label="UKF", linestyle="dashed")
        plt.xlabel(r"x[m]", fontsize=40)
        plt.ylabel(r"y[m]", fontsize=40)
        plt.legend(fontsize=35)
        filename += "_trajectory"
    else:
        ax1 = fig.add_subplot(211)
        ax1.plot(data["time"], data["nkf_xy_error"], color="red", label="MKF")
        ax1.plot(data["time"], data["ekf_xy_error"], color="blue", label="EKF")
        ax1.plot(data["time"], data["ukf_xy_error"], color="green", label="UKF")
        ax1.set_xlabel(r'time[$s$]', fontsize=30)
        ax1.set_ylabel(r'position[$m$]', fontsize=30)
        ax1.legend(loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=4, fontsize=30)

        ax2 = fig.add_subplot(212)
        ax2.plot(data["time"], data["nkf_yaw_error"], color="red")
        ax2.plot(data["time"], data["ekf_yaw_error"], color="blue")
        ax2.plot(data["time"], data["ukf_yaw_error"], color="green")
        ax2.set_xlabel(r'time[$s$]', fontsize=30)
        ax2.set_ylabel(r'yaw[$rad$]', fontsize=30)

    plt.savefig(path + "/result/picture/robot" + str(robot_num) + filename + ".png")
    plt.savefig(path + "/result/picture/robot" + str(robot_num) + filename + ".eps", bbox_inches="tight", pad_inches=0.05)
    plt.show()