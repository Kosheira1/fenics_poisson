from fenics import *
import numpy as np
from setup_domains import setup_boundaries
from Expressions import *
from solver import solver

# Import plot library
import matplotlib.pyplot as plt


def read_hysteresis(filename):
    # Read saturation curve values
    f2 = open(filename, 'r')
    lines = f2.readlines()
    f2.close()

    E_values = []
    P_values = []

    for line in lines:
        p = line.split()
        E_values.append(float(p[0]))
        P_values.append(float(p[1]))

    return (E_values, P_values)


def compute_fields(V, mesh, C, u, Pol, e_0):
    # Compute D_field, E_field and P_field
    degree = V.ufl_element().degree()
    W = VectorFunctionSpace(mesh, 'P', degree)

    disp_field = project(e_0 * C * grad(u) + Pol, W)
    flux_x, flux_y = disp_field.split(deepcopy=True)  # extract components.

    electric_field = project(grad(u), W)
    elec_x, elec_y = electric_field.split(deepcopy=True)  # extract components.

    pol_y = flux_y - e_0 * elec_y

    return (flux_y, elec_y, pol_y)


def plot_routine(number):
    # Read P-E curve
    filename = 's_curve_PE.dat'
    (E_values, P_values) = read_hysteresis(filename)
    # Plotting
    plt.figure(num=number, figsize=(16, 12))
    # xvals = np.linspace(-7, 7, 100)
    # yinterp = np.interp(xvals, E_values, P_values)
    plt.plot(np.asarray(E_values) * 1E-2, np.asarray(P_values) * 1E+6 * 1E+8, 'o', linewidth=3, label='P-E data')
    # plt.plot(xvals, yinterp, '-x', linewidth=2.5, label='Interpolation')
    plt.xlabel('Electric Field (kV/cm)')
    plt.ylabel('Polarization ' + r"$(\frac{fC}{\mu m^2})$")

    plt.grid(True)


def intermediate_S_plot(FE_index, max_it, P_space, E_space, volt_bias, perm_plot):

    P_space.extend([P_space[-1]] * (max_it - len(P_space)))
    E_space.extend([E_space[-1]] * (max_it - len(E_space)))
    plot_P = list(P_space[i][FE_index] for i in range(max_it))
    plot_E = list(E_space[i][FE_index] for i in range(max_it))

    num_f = 3 * len(P_space)

    rgb = (np.arange(float(num_f)) / num_f).reshape(len(plot_P), 3)
    plt.scatter(plot_E, plot_P, s=500, facecolors=rgb, label='Location in P-E space')
    plt.annotate("{0:.2e}".format(volt_bias), (plot_E[-1], plot_P[-1]))

    if (perm_plot):
        plt.show()


def FEM_solverexpressions(V, NCFET, rem_flag_dict):
    # Create the Permittivity Tensor for the anisotropic Poisson equation, FE material only exhibits field dependent permittivity in confinement direction!
    Con_M = Permittivity_Tensor_M(NCFET.materials, NCFET.permi, NCFET.domains, degree=2)
    C = as_matrix(((Con_M[0], Con_M[1]), (Con_M[1], Con_M[2])))

    # Create the Vector Expression for the remnant Polarization, only non-zero in FE material and only exhibited in confinement direction
    P_r = Remnant_Pol(NCFET.materials, rem_flag_dict, NCFET.Pol_r, NCFET.domains, degree=2)
    P = as_vector((P_r[0], P_r[1]))

    degree = V.ufl_element().degree()
    W = VectorFunctionSpace(NCFET.mesh, 'P', degree)
    Po = project(P, W)
    return (C, Po, P)


def initial_solve():
    # This function solves the electrostatic problem given the initial guess for the FE-permittivity from either the previous bias point

    # Po is projected on vector function space, P is vector Expression
    (C, Po, P) = FEM_solverexpressions(V, NCFET, rem_flag_dict)

    # Solve Variational Problem
    u = solver(f, C, Po, V, bcs, 2)
    (flux_y, elec_y, pol_y) = compute_fields(V, NCFET.mesh, C, u, P, epsilon_0)

    # Compute midpoints of all single-domain FE materials
    point_list = NCFET.FE_midpointlist
    error = []

    # Compute error
    for vals in FE_dict.keys():
        point = point_list[vals]
        error.append(abs(np.interp(pol_y(point), P_values, E_values) * 1E-2 - elec_y(point) * 1E-2))
    print('\n'.join('{0:{0}d}  :  {1:.3e}'.format(k, vals) for k, vals in enumerate(error)))
