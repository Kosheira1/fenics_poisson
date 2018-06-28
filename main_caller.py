"""
Main File. Invokes all solver functionality to solve the tensor-weighted Poisson equation. The displacement-field dependent Permittivity can be defined and the solver is ran.
"""
from __future__ import print_function

import numpy as np

from plot_differential_cap import *
from plot_results import *
from refactor_saturation import *
from refactor_solver import *
from S_curve_solver import *
from setup_domains import *

# Import plot library
import matplotlib
import matplotlib.pyplot as plt

import os
folder = 'Scurve_two_layers'
if not os.path.exists(folder):
    os.mkdir(folder)
os.chdir(folder)

font = {'weight': 'bold',
        'size': 22}

matplotlib.rc('font', **font)

"""
User Input Part:

1. Define Geometry and Domains
2. Create Mesh or Read Mesh from File
3. Select Material and Carrier Models
4. Create a List of Bias Points
"""

# Define Device Variables
sem_width = 1.0  # [um] Width of the semiconductor channel
ins_width = 0.0  # [um] Width of the insulator layer
FE_width = 1.0  # [um] Width of the ferroelectric layer
sem_relperm = 1  # Relative permittivity of semiconductor channel
doping = -35.2   # [C/um^2] Acceptor doping concentration
epsilon_FE = -5.0  # [] Initial Guess for out-of-plane FE permittivity
epsilon_0 = 3.0  # [F*um^-1]
z_thick = 1.0  # [um]

# Define Domains and assign material identifier
(SC, FE) = setup_domains(sem_width, FE_width)

# Create a custom mesh or read it from a file
# 420
mesh = RectangleMesh(Point(0, 0), Point(1, sem_width + FE_width), 20, 420)

# Define Interface markers
# edge_markers = MeshFunction('bool', mesh, 1, False)
# LayerBoundary(sem_width).mark(edge_markers, True)

# refine mesh
# adapt(mesh, edge_markers)
# mesh = mesh.child()
mesh_name = 'saved_mesh_FEW_' + str(FE_width) + '_SEMW_' + str(sem_width) + '.xml'
File(mesh_name) << mesh

# Define a Mesh Function which stores material labels. It is used in the solver routine to identify
materials = MeshFunction('size_t', mesh, 2)

FE.mark(materials, 0)
SC.mark(materials, 1)

# Store material dimensions
dimensions = [sem_width, FE_width]

# Define a Mesh Function with initial permittivity values
permi = MeshFunction('double', mesh, 2)

FE.mark(permi, epsilon_FE)
SC.mark(permi, sem_relperm)

# Select supported carrier model: 'Depletion'
c_model = 'Depletion'

# Select supported Ferroelectric model: 'const_negative', 'saturation', 'S_curve'
FE_model = 'S_curve'

# Bias points
volt_list_ultra = [0.12]  # [V]
volt_list_low = [float(x) / 10 for x in range(2, 63, 9)]  # [V]
volt_list_high = [float(x) / 10 for x in range(73, 110, 25)]  # [V]
volt_list = volt_list_ultra + volt_list_low + volt_list_high

# volt_list = [float(x) for x in np.linspace(-6, 6, 10)]
volt_list = [0.8]

# Main Function: Solve Problem for all defined bias points
Solution_points = []
Permittivity_points = []
TotalCharge_points = []

for idx, bias in enumerate(volt_list):
    print("Bias Point: " + str(bias) + " The index, at which the bias point is extracted: " + str(idx))
    if (FE_model == 'const_negative'):

        (u_v, C_v, Q_v) = run_solver_const(mesh, dimensions, materials, permi, doping, bias)
        Solution_points.append(u_v)
        Permittivity_points.append(C_v)
        TotalCharge_points.append(Q_v)

    elif (FE_model == 'saturation'):
        (u_v, C_v, Q_v) = run_solver_sat(mesh, dimensions, materials, permi, doping, bias)
        Permittivity_points.append(C_v)
        TotalCharge_points.append(Q_v)
        Solution_points.append(u_v)

    elif (FE_model == 'S_curve'):
        (u_v, C_v, Q_v) = run_solver_S(mesh, dimensions, materials, permi, doping, bias)
        Permittivity_points.append(C_v)
        TotalCharge_points.append(Q_v)
        Solution_points.append(u_v)

    else:

        print("")

'''
# Write Charge and voltage value in file and create capacitance plot
cap_dat = np.array([volt_list, TotalCharge_points])
cap_dat = cap_dat.T
print(cap_dat)
np.savetxt('capdata' + 'Model:_' + FE_model + '_' + str(epsilon_FE) + '.dat', cap_dat, fmt='%.3f')
plot_diff_cap('capdata' + 'Model:_' + FE_model + '_' + str(epsilon_FE) + '.dat')
'''

# Plot the first (or any) bias point
solution_name = 'Model:_' + FE_model + '_' + str(epsilon_FE) + '_.xml'
u = Solution_points[0]
File(solution_name) << u


# plot_solution(mesh_name, solution_name)
os.chdir('../')
plt.show()
