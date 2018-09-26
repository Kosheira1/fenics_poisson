# Import plot library
import matplotlib.pyplot as plt
import matplotlib

# numpy library
import numpy as np
import pandas as pd

import sys

from S_curve_solver import read_hysteresis, plot_routine


def plot_S_traj(start_in, final_in, *filename):
    # General Plotting Settings
    font = {'weight': 'bold',
            'size': 25}
    matplotlib.rc('font', **font)

    idx = pd.IndexSlice

    for names in filename:
        df = pd.read_csv(names, index_col=[0, 1, 2])

    print(df)
    volt_list = df.index.unique(level=0).values
    FE_list = df.index.unique(level=2).values

    for FE_num in FE_list:
        plot_routine(FE_num)

        for i in range(start_in, final_in + 1):

            bias_df = df.loc[idx[volt_list[i]], :, FE_num]
            print(bias_df)
            iteration = bias_df.index.unique(level=1).values

            num_f = 3 * len(iteration)
            rgb = (np.arange(float(num_f)) / num_f).reshape(len(iteration), 3)
            plt.scatter(bias_df['E'], bias_df['P'], s=500, facecolors=rgb)
            plt.annotate("{0:.2f}".format(volt_list[i]), (bias_df['E'].values[-1] - 0.2, bias_df['P'].values[-1]))
            plt.title('Convergence for FE number: ' + str(FE_num), y=1.02)
            plt.gcf().subplots_adjust(bottom=0.15)

    plt.show()


if __name__ == "__main__":
    "Read Arguments from Terminal"
    start_in = int(sys.argv[1])
    final_in = int(sys.argv[2])
    first_arg = tuple(sys.argv[3:])
    plot_S_traj(start_in, final_in, *first_arg)
