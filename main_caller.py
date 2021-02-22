"""
Main File. Invokes all solver functionality to solve the tensor-weighted Poisson equation. The displacement-field dependent Permittivity can be defined on all Ferroelectric material points between solver runs.
"""
from __future__ import print_function

import numpy as np
import pandas as pd

from plot_differential_cap import *
from plot_results import *
from refactor_saturation import *
from refactor_solver import *
from S_curve_solver import *
from S_curve_newton import *
from S_curve_gradientdescent import *
from setup_domains import *
from Parameter_class import Device
from coordinate_class import coordinate_data

# Import plot library
import matplotlib as mpl
import matplotlib.pyplot as plt

import time
import os


# mpl.rcParams["savefig.directory"] = os.chdir(os.path.dirname(__file__))


folder = 'MulFE_Scurv'
if not os.path.exists(folder):
    os.mkdir(folder)
os.chdir(folder)

font = {'weight': 'bold',
        'size': 22}

matplotlib.rc('font', **font)

"""
User Input Part:

1. Define Geometry and Domains as well as boundary conditions and save list of coordinates for each material
2. Pass coordinate object to setup_domains.py to create domains
3. Create Mesh or Read Mesh from File
4. Create a List of Bias Points
"""

# Define Device Variables
sem_width = 1.0  # [um] Width of the semiconductor channel
ins_width = 0.0  # [um] Width of the insulator layer
FE_width = 1.0  # [um] Width of the ferroelectric layer
sem_relperm = 19.4  # Relative permittivity of semiconductor channel
doping = -35.2   # [C/um^2] Acceptor doping concentration

epsilon_0 = 8.85E-18  # [F*um^-1]
z_thick = 1.0  # [um]
P_r = 10 * 1E-6 * 1E-8   # [C/um^2]
epsilon_FE = -44.0  # [] Initial Guess for out-of-plane FE permittivity
number_FE = 2  # [] Number of single-domain FE materials

# Initialize rectangle coordinates
FE_coords = []
DE_coords = []
SEM_coords = []
ME_coords = []

# Build device from rectangles
FE_coords.append([0.0, 2.0, 0.5, 1.0])
FE_coords.append([0.5, 2.0, 1.0, 1.0])
# FE_coords.append([0.5, 2.0, 0.75, 1.0])
# FE_coords.append([0.75, 2.0, 1.0, 1.0])
DE_coords.append([0.0, 1.0, 1.0, 0.0])

# Create coordinate class.
geom_device = coordinate_data(FE_coords, DE_coords, SEM_coords, ME_coords)

# Initialize out-of plane FE permittivity dictionary
FE_dict = dict([(key, epsilon_FE) for key in range(number_FE)])

# Define Domains and assign material identifier
domains = setup_domains(geom_device)

mesh = Mesh('newtest.xml')
V = FunctionSpace(mesh, 'P', 1)

# Define Interface markers
# edge_markers = MeshFunction('bool', mesh, 1, False)
# LayerBoundary(sem_width).mark(edge_markers, True)
# refine mesh
# adapt(mesh, edge_markers)
# mesh = mesh.child()

# mesh_name = 'saved_mesh_FEW_' + str(FE_width) + '_SEMW_' + str(sem_width) + '.xml'
# File(mesh_name) << mesh

# Create device object and assign material, permittivity and remnant polarization Mesh Function
NCFET = Device(domains, geom_device, mesh)
NCFET.assign_labels()
NCFET.assign_permittivity(FE_dict, sem_relperm, [])
NCFET.assign_remnantpol(P_r)
NCFET.compute_FE_midpointlist()

file = File('subdomains.pvd')
file << NCFET.materials

# Store material dimensions
dimensions = [sem_width, FE_width]

# Select supported carrier model: 'Depletion'
c_model = 'Depletion'

# Select supported Ferroelectric model: 'const_negative', 'saturation', 'S_curve'
FE_model = 'S_curve'

# volt_list = np.linspace(0.10, 0.5, 2)
volt_list = np.linspace(50, 500, 9)
volt_list = [110, 120, 130, 140]

# Main Function: Solve Problem for all defined bias points
Solution_points = []
Permittivity_points = []
TotalCharge_points = []
P_it = []
E_it = []
max_it = 9  # Defines the maximum allowed iteration number for the Ferroelectric permittivity update routin
rem_flag_dict = dict([(key, 0) for key in range(number_FE)])  # Store for each single-domain FE the segment of the Polarization state. 0 for neg-cap region, 1 for upper part, 2 for lower part.

start = time.time()

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
        (u_v, C_v, Q_v, P, E) = gradient_solver_S(V, NCFET, FE_dict, dimensions, bias, max_it, rem_flag_dict)

        Permittivity_points.append(C_v)
        TotalCharge_points.append(Q_v)
        Solution_points.append(u_v)

        P.extend([P[-1]] * (max_it - len(P)))
        E.extend([E[-1]] * (max_it - len(E)))
        P_it.append(P)
        E_it.append(E)

        point_list = NCFET.FE_midpointlist
        # Routine to create initial guess for all single-domain FE materials.
        EXP = C_v
        F = project(EXP[2], V)

        for vals in FE_dict.keys():
            # We create a data structure that stores the convergence of E vs. P for each bias point

            eval_point = Point(point_list[vals][0], point_list[vals][1])
            print(F(eval_point))

            # Switch of linearization point, permittivities have to be calibrated with new FE material
            if (F(eval_point) < (-90.0) and rem_flag_dict[vals] == 0):
                rem_flag_dict[vals] = 1
                FE_dict[vals] = 25.0
                NCFET.update_permittivity(FE_dict)

            else:
                FE_dict[vals] = F(eval_point)
                NCFET.update_permittivity(FE_dict)

        # FE_dict = dict([(key, FE_dict[0]) for key in range(number_FE)])
        # NCFET.update_permittivity(FE_dict)

        # plt.figure(num=120)
        # plot(F)
        # plt.show()

    else:

        print("")

end = time.time()

print('The elapsed time is: ' + '{0:.2f}'.format(end - start) + ' seconds.')
# Using a pandas data frame to store convergence of P-E
iterables = [volt_list, list(range(max_it + 1)), list(range(number_FE))]
m_index = pd.MultiIndex.from_product(iterables, names=['Bias', 'Iter', 'FE_num'])
E_R, P_R = np.array(E_it), np.array(P_it)

space_dat = np.array([E_R.reshape((max_it + 1) * len(volt_list) * number_FE, 1), P_R.reshape((max_it + 1) * len(volt_list) * number_FE, 1)])
df = pd.DataFrame(space_dat.T.reshape((max_it + 1) * len(volt_list) * number_FE, 2), index=m_index)
df. columns = ['E', 'P']

print(df)
df.to_csv(FE_model + '_' + 'max_it_' + str(max_it) + '_biasn_' + str(len(volt_list)) + '.csv')

'''
# Write Charge and voltage value in file and create capacitance plot
cap_dat = np.array([volt_list, TotalCharge_points])
cap_dat = cap_dat.T
print(cap_dat)
np.savetxt('capdata' + 'Model:_' + FE_model + '_' + str(epsilon_FE) + '.dat', cap_dat, fmt='%.3f')
plot_diff_cap('capdata' + 'Model:_' + FE_model + '_' + str(epsilon_FE) + '.dat')
'''

# Plot the first (or any) bias point to xml and vtu file
solution_name = 'Model:_' + FE_model + '_' + str(epsilon_FE) + '_.xml'
u = Solution_points[0]
File(solution_name) << u
solution_name = 'Model:_' + FE_model + '_' + str(epsilon_FE) + '_.pvd'
File(solution_name) << u
# Plot mesh
plt.figure(num='mesh')
plot(mesh)


# plot_solution(mesh_name, solution_name)
os.chdir('../')
plt.show()
