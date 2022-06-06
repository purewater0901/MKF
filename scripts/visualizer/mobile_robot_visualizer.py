import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from scipy import stats
import platform
import os

if __name__ == '__main__':
    filename = "kinematic_vehicle_gaussian"
    os.chdir('../../')
    path = os.getcwd()
    data = pd.read_csv(path + "/result/data/" + filename + ".csv")
    flag_visualize_trajectory = True

    cm = 1 / 2.54
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
    nkf_v_error_mean = data['nkf_v_error'].mean()
    ekf_v_error_mean = data['ekf_v_error'].mean()
    ukf_v_error_mean = data['ukf_v_error'].mean()
    nkf_yaw_error_mean = data['nkf_yaw_error'].mean()
    ekf_yaw_error_mean = data['ekf_yaw_error'].mean()
    ukf_yaw_error_mean = data['ukf_yaw_error'].mean()
    print("nkf_xy_error_mean: ", nkf_xy_error_mean)
    print("ekf_xy_error_mean: ", ekf_xy_error_mean)
    print("ukf_xy_error_mean: ", ukf_xy_error_mean)
    print("nkf_v_error_mean: ", nkf_v_error_mean)
    print("ekf_v_error_mean: ", ekf_v_error_mean)
    print("ukf_v_error_mean: ", ukf_v_error_mean)
    print("nkf_yaw_error_mean: ", nkf_yaw_error_mean)
    print("ekf_yaw_error_mean: ", ekf_yaw_error_mean)
    print("ukf_yaw_error_mean: ", ukf_yaw_error_mean)

    if flag_visualize_trajectory:
        plt.plot(data["x_true"], data["y_true"], color="black", linewidth=2.0, label="True")
        plt.plot(data["nkf_x"], data["nkf_y"], color="red", linewidth=2.0, label="MKF")
        plt.plot(data["ekf_x"], data["ekf_y"], color="blue", linewidth=1.5, label="EKF")
        plt.plot(data["ukf_x"], data["ukf_y"], color="green", linewidth=1.5, label="UKF")
        plt.xlabel(r"x[m]", fontsize=40)
        plt.ylabel(r"y[m]", fontsize=40)
        plt.legend(fontsize=25)
        filename += "_trajectory"
    else:
        ax1 = fig.add_subplot(311)
        ax1.plot(data["time"], data["nkf_xy_error"], color="red", label="MKF")
        ax1.plot(data["time"], data["ekf_xy_error"], color="blue", label="EKF")
        ax1.plot(data["time"], data["ukf_xy_error"], color="green", label="UKF")
        ax1.set_xlabel(r'time[$s$]', fontsize=30)
        ax1.set_ylabel(r'position[$m^2$]', fontsize=30)
        ax1.legend(loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=4, fontsize=30)

        ax2 = fig.add_subplot(312)
        ax2.plot(data["time"], data["nkf_yaw_error"], color="red")
        ax2.plot(data["time"], data["ekf_yaw_error"], color="blue")
        ax2.plot(data["time"], data["ukf_yaw_error"], color="green")
        ax2.set_xlabel(r'time[$s$]', fontsize=30)
        ax2.set_ylabel(r'yaw[$rad$]', fontsize=30)

        ax3 = fig.add_subplot(313)
        ax3.plot(data["time"], data["nkf_v_error"], color="red")
        ax3.plot(data["time"], data["ekf_v_error"], color="blue")
        ax3.plot(data["time"], data["ukf_v_error"], color="green")
        ax3.set_xlabel(r'time[$s$]', fontsize=30)
        ax3.set_ylabel(r'velocity[$m/s$]', fontsize=30)

    plt.savefig(path + "/result/picture/" + filename + ".png")
    plt.savefig(path + "/result/picture/" + filename + ".eps", bbox_inches="tight", pad_inches=0.05)
    plt.show()
