import numpy as np

import matplotlib.pyplot as plt
import matplotlib

# Create a saturation curve with constant material polarizability between -5, 5 kv/cm and saturation beyond
x_lower = np.linspace(-6, -4, 10)
x_middle = np.linspace(-2.5, 2.5, 20)
x_upper = np.linspace(4, 6, 10)

y_lower = [-6 for x in range(0, 10, 1)]
y_middle = 2 * x_middle
y_upper = [6 for x in range(0, 10, 1)]

x = np.concatenate((x_lower, x_middle, x_upper))
y = np.concatenate((y_lower, y_middle, y_upper))

x = np.array(x)
y = np.array(y)

data = np.array([x, y])
data = data.T
np.savetxt('saturation_PE.dat', data, fmt='%.3f')

xvals = np.linspace(-7, 7, 100)
yinterp = np.interp(xvals, x, y)


# General Plotting Settings
font = {'weight': 'bold',
        'size': 28}
matplotlib.rc('font', **font)

# Plot routine
plt.figure(num=0, figsize=(16, 12))
plt.plot(x, y, 'o', linewidth=3, label='E-P data')
plt.plot(xvals, yinterp, '-x', linewidth=2.5, label='Interpolation')
plt.xlabel('Electric Field (kV/cm)')
plt.ylabel('Polarization ' + r"$\frac{fC}{\mu m^2}$")
plt.grid(True)
plt.show()
