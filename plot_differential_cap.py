# Import plot library
import matplotlib.pyplot as plt

# numpy library
import numpy as np

import sys

def plot_diff_cap(filename):
	f2 = open(filename, 'r')

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
	dQdV[0:-1] = -np.diff(npQ)/np.diff(npV)
	dQdV[-1] = -(npQ[-1] - npQ[-2])/(npV[-1] - npV[-2])
	plt.figure(num=10, figsize=(16,12))
	plt.plot(dQdV)
	plt.grid(True)

if __name__ == "__main__":

	first_arg = sys.argv[1]
	plot_diff_cap(first_arg)
	plt.show()
