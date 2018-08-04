import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from S_curve_solver import read_hysteresis

# Create a S-curve P-E dependency look-up table with remnant Polarization = 10 uC/cm^2 and Coercive Field = 1 MV/cm

# Hafnium Dioxide Material

P_r = 10  # (uC/cm^2)
E_c = 1  # (MV/cm)

# E_ext = 2\alpha*P + 4\beta*P^3

alph = -3 * np.sqrt(3.0) * E_c / (4 * P_r)
bet = 3 * np.sqrt(3.0) * E_c / (8 * P_r ** 3)

P_vals = np.linspace(-20, 20, 100)
E_vals = 2 * alph * P_vals + 4 * bet * P_vals ** 3

x = np.array(E_vals)
y = np.array(P_vals)

data = np.array([x, y])
data = data.T
np.savetxt('s_curve_PE.dat', data, fmt='%.3f')

# index = np.array(np.where(abs(P_vals - 0) < 1))
# print(index[0, 0])

(E_vals, P_vals) = read_hysteresis('s_curve_PE.dat')

primordial_value = 1.5
E_pol_tho = np.interp(primordial_value, P_vals, E_vals)
print(E_pol_tho)

# General Plotting Settings
font = {'weight': 'bold',
        'size': 28}
matplotlib.rc('font', **font)

# Plot routine
plt.figure(num=0, figsize=(16, 12))
plt.plot(E_vals, P_vals, 'o', linewidth=3, label='E-P data')
# plt.plot(xvals, yinterp, '-x', linewidth=2.5, label='Interpolation')
plt.xlabel('Electric Field (MV/cm)')
plt.ylabel('Polarization ' + r"$\frac{\mu C}{cm^2}$")
plt.grid(True)
plt.legend(loc='best')
plt.show()
