# Import plot library
import matplotlib.pyplot as plt
import matplotlib

# numpy library
import numpy as np

import sys


def plot_diff_cap(*filename):
    "Numerically differentiating and plotting capacitance"
    # General Plotting Settings
    font = {'weight': 'bold',
            'size': 19}
    matplotlib.rc('font', **font)

    plt.figure(num=10, figsize=(16, 12))

    for names in filename:
        f2 = open(names, 'r')

        lines = f2.readlines()
        f2.close()

        x1 = []
        y1 = []

        for line in lines:
            p = line.split()
            x1.append(float(p[0]))
            y1.append(float(p[1]))

        npV = np.array(x1)
        npQ = np.array(y1)
        dQdV = np.zeros(npV.shape, np.float)
        dQdV[0:-1] = -np.diff(npQ) / np.diff(npV)
        dQdV[-1] = -(npQ[-1] - npQ[-2]) / (npV[-1] - npV[-2])
        plt.plot(npV, dQdV, linewidth=3, label=names)
        f2.close()

    plt.xlabel('Bias Voltage (V)')
    plt.ylabel('Differential Capacitance [F]')
    plt.axhline(y=1.5, color='r', linewidth=4, linestyle='--', label='Value of series capacitance')
    plt.legend()
    plt.ylim(ymax=5)
    plt.ylim(ymin=1)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    "Read Arguments from Terminal"
    first_arg = tuple(sys.argv[1:])
    plot_diff_cap(*first_arg)
