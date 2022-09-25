import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from scipy import stats
import platform
import os

if __name__ == '__main__':
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    filename = "simple_vehicle_non_gaussian"
    os.chdir('../../')
    path = os.getcwd()
    print_pos = False

    fig = plt.figure(figsize=(8.5,11))
    plt.subplots_adjust(wspace=0.4, hspace=0.8)
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams["font.size"] = 28
    plt.rcParams['xtick.labelsize'] = 35
    plt.rcParams['ytick.labelsize'] = 35
    plt.rcParams['xtick.direction'] = 'in' # x axis in
    plt.rcParams['ytick.direction'] = 'in' # y axis in
    plt.rcParams['font.family'] = 'Times New Roman'

    phi = [np.pi/2, np.pi/2.1, np.pi/2.3, np.pi/2.4, np.pi/2.5, np.pi/2.7, np.pi/ 2.8, np.pi/3, np.pi/4, np.pi/5, np.pi/6, np.pi/7, np.pi/8, np.pi/9]
    for i in range(10, 90, 10):
        phi.append(np.pi/i)

    ekf_pos_error = [0.678911, 0.653025, 0.604626, 0.582253, 0.561124, 0.522538, 0.505022, 0.473369, 0.370337, 0.321377,
                     0.293626, 0.275508, 0.262721, 0.253132,0.245386, 0.197286, 0.167771, 0.150334, 0.139681, 0.132707, 0.127794, 0.124101]
    ukf_pos_error = [0.73617, 0.530213, 0.406161, 0.375781, 0.353037, 0.318908, 0.305039, 0.28227, 0.225088, 0.199009,
                     0.183783, 0.173801, 0.166696, 0.161334, 0.157114, 0.137406, 0.125916, 0.115141, 0.106267, 0.0996791, 0.0951945, 0.0926149]
    mkf_pos_error = [0.331895, 0.31686, 0.292308, 0.282037, 0.272773, 0.256674, 0.249623, 0.237127, 0.195376, 0.172233,
                     0.157879, 0.148182, 0.141236, 0.136029, 0.131907, 0.110128, 0.0993425, 0.0927156, 0.0882086, 0.0849647, 0.0825318, 0.0806486]
    ekf_yaw_error = [0.410777, 0.394987, 0.366171, 0.35312, 0.340925, 0.318907, 0.308985, 0.291274, 0.233217, 0.199168,
                     0.176464, 0.160228, 0.147967, 0.138302, 0.130438, 0.0908151, 0.0741617, 0.0639321, 0.0572538, 0.0525557, 0.0493913, 0.0469328]
    ukf_yaw_error = [0.565068, 0.465822, 0.372367, 0.342032, 0.317812, 0.281156, 0.266858, 0.243818, 0.184736, 0.157713,
                     0.141109, 0.12966, 0.121093, 0.114217, 0.108506, 0.0772117, 0.064255, 0.0577283, 0.0540155, 0.05144, 0.049556, 0.0481044]
    mkf_yaw_error = [0.18233, 0.177734, 0.17, 0.166561, 0.163374, 0.157552, 0.154826, 0.149754, 0.130344, 0.116703,
                     0.106655, 0.0991898, 0.0931556, 0.0880097, 0.0835732, 0.0593777, 0.0490457, 0.0433009, 0.0396737, 0.0372307, 0.0355028, 0.0342396]

    if print_pos:
        plt.plot(phi, mkf_pos_error, color="red", linewidth=4.0, label="MKF")
        plt.plot(phi, ekf_pos_error, color="blue", linewidth=4.0, label="EKF")
        plt.plot(phi, ukf_pos_error, color="green", linewidth=4.0, label="UKF")
        plt.ylabel(r"position error[m]", fontsize=40)
        filename += "_position_progression"
    else:
        plt.plot(phi, mkf_yaw_error, color="red", linewidth=4.0, label="MKF")
        plt.plot(phi, ekf_yaw_error, color="blue", linewidth=4.0, label="EKF")
        plt.plot(phi, ukf_yaw_error, color="green", linewidth=4.0, label="UKF")
        plt.ylabel(r"yaw error[rad]", fontsize=40)
        filename += "_yaw_progression"

    plt.xlabel(r"$v_{\varphi, limit}[rad]$", fontsize=40)
    plt.legend(fontsize=35)

    plt.savefig(path + "/result/picture/robot1/" + filename + ".png")
    plt.savefig(path + "/result/picture/robot1/" + filename + ".eps", format="eps", bbox_inches="tight", pad_inches=0.05)
    plt.show()
