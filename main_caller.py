"""
Main File. Invokes all solver functionality to solve the tensor-weighted Poisson equation. The displacement-field dependent Permittivity can be defined and the solver is ran.
"""

from __future__ import print_function
from refactor_solver import *
import numpy as np
from plot_differential_cap import *
from plot_results import *

# Import plot library
import matplotlib.pyplot as plt
import matplotlib

font = {'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

"""
User Input Part:

1. Define Geometry and Domains
2. Create Mesh or Read Mesh from File
3. Select Material and Carrier Models
4. Create a List of Bias Points
"""

# Define Device Variables
sem_width = 2.7 # [um] Width of the semiconductor channel
sem_relperm = 1 # Relative permittivity of semiconductor channel
doping = -15.2   # [C/um^2] Acceptor doping concentration
epsilon_FE = -3.0 # [] Initial Guess for out-of-plane FE permittivity

# Create a custom mesh and save it to a file or read it from a file
mesh = RectangleMesh(Point(0,0), Point(1,3), 20, 360)
File('saved_mesh.xml') << mesh

#Select supported carrier model: 'Depletion'
c_model = 'Depletion'

#Select supported Ferroelectric model: 'const_negative', 'saturation', 'S_curve'
FE_model = 'const_negative'

#Bias points
volt_list_low = [float(x)/10 for x in range(2, 20, 4)] # [V]
volt_list_high = [float(x)/10 for x in range(25, 80, 7)] # [V]
volt_list = volt_list_low+volt_list_high

# Main Function: Solve Problem for all defined bias points
Solution_points = []
Permittivity_points = []
TotalCharge_points = []

for idx, bias in enumerate(volt_list):

	print("Bias Point: " + str(bias)  +  " The index, at which the bias point is extracted: " + str(idx))	
	(u_v, C_v, Q_v) = run_solver(mesh, sem_width, sem_relperm, doping, bias)
	Solution_points.append(u_v)
	Permittivity_points.append(C_v)
	TotalCharge_points.append(Q_v)
	print("")

# Write Charge and voltage value in file
cap_dat = np.array([volt_list, TotalCharge_points])
cap_dat = cap_dat.T
print(cap_dat)
np.savetxt('capdataV1.dat', cap_dat, fmt='%.3f')
plot_diff_cap('capdataV1.dat')

# Plot the first (or any) bias point
u = Solution_points[0]
File('saved_u.xml') << u

plot_solution('saved_mesh.xml', 'saved_u.xml')
