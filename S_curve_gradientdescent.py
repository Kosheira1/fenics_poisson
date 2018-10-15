from fenics import *
import numpy as np
from setup_domains import setup_boundaries
from Expressions import *
from solver import solver
from auxiliary_solverfunctions import read_hysteresis, compute_fields, plot_routine, intermediate_S_plot, FEM_solverexpressions, initial_solve, single_step_solve, newton_step

# Import plot library
import matplotlib.pyplot as plt


def gradient_solver_S(V, NCFET, FE_dict, dimensions, volt_bias, max_it, rem_flag_dict):
    '''
    Run solver with a S-curve PE-model, see gen_S_lookup.py for details about physical quantities.
    '''
    # Read P-E curve
    filename = 's_curve_PE.dat'
    (E_values, P_values) = read_hysteresis(filename)

    # Define vaccuum permittivity
    epsilon_0 = 8.85 * 1E-18  # (F/um)

    # Transform to np arrays
    E_values = np.array(E_values)
    P_values = np.array(P_values)

    # Setup boundary conditions
    bcs = setup_boundaries(V, volt_bias, 2.0)

    # Initialize Charge Expression
    f = Constant(-0.0)

   # For each bias point we want to track the trajectory of the solution in the P-E space while it converges onto a point on the S-Curve, used for plotting the trajectory and observing convergence!
    P_space = []
    E_space = []

    error = initial_solve(V, f, bcs, NCFET, rem_flag_dict, E_values, P_values, FE_dict, P_space, E_space)

    # Initialize Loop Variables, state_init is set to 0 because for zero electric external field, the FE should be in the negative capacitance regime!
    eta = 5E-4
    counter = 0

    point_list = NCFET.FE_midpointlist

    # Solving Poisson's equation and updating relative permittivity in a loop. This approach is used because the field dependence of polarization is not explicitly used in the PDE formulation!!
