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

    P_space.extend([P_space[-1]] * (max_it + 1 - len(P_space)))
    E_space.extend([E_space[-1]] * (max_it + 1 - len(E_space)))
    plot_P = list(P_space[i][FE_index] for i in range(max_it + 1))
    plot_E = list(E_space[i][FE_index] for i in range(max_it + 1))

    num_f = 3 * len(plot_P)

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


def initial_solve(V, f, bcs, NCFET, rem_flag_dict, E_values, P_values, FE_dict, P_space, E_space):
    # This function solves the electrostatic problem given the initial guess for the FE-permittivity from either the previous bias point or after a single update has been performed

    # Po is projected on vector function space, P is vector Expression
    (C, Po, P) = FEM_solverexpressions(V, NCFET, rem_flag_dict)

    # Solve Variational Problem
    u = solver(f, C, Po, V, bcs, 2)
    (flux_y, elec_y, pol_y) = compute_fields(V, NCFET.mesh, C, u, P, NCFET.epsilon_0)

    # Compute midpoints of all single-domain FE materials
    point_list = NCFET.FE_midpointlist
    error = []
    P_sing_iter = []
    E_sing_iter = []

    # Compute error and save P and E pairs for all FE materials
    for vals in FE_dict.keys():
        point = point_list[vals]
        error.append(abs(np.interp(pol_y(point), P_values, E_values) * 1E-2 - elec_y(point) * 1E-2))
        P_sing_iter.append(pol_y(point) * 1E+6 * 1E+8)
        E_sing_iter.append(elec_y(point) * 1E-2
                           )

    print('\n'.join('{0:{0}d}  :  {1:.3e}'.format(k, vals) for k, vals in enumerate(error)))
    P_space.append(P_sing_iter)
    E_space.append(E_sing_iter)

    return error


def single_step_solve(V, f, NCFET, rem_flag_dict, bcs, E_values, P_values):
    '''
    This function solves the electrostatic problem and returns the error, used in the context of filling the Jacobian matrix for the newton update step
    '''
    # Po is projected on vector function space, P is vector Expression
    (C, Po, P) = FEM_solverexpressions(V, NCFET, rem_flag_dict)

    # Solve Variational Problem
    u = solver(f, C, Po, V, bcs, 2)
    (flux_y, elec_y, pol_y) = compute_fields(V, NCFET.mesh, C, u, P, NCFET.epsilon_0)

    error = []
    point_list = NCFET.FE_midpointlist
    # Compute error
    for point in point_list:
        error.append(abs(np.interp(pol_y(point), P_values, E_values) * 1E-2 - elec_y(point) * 1E-2))

    return error


def newton_step(V, f, NCFET, FE_dict, error, rem_flag_dict, bcs, E_values, P_values, P_space, E_space):
    '''
    This function analyses the error locally and uses a newton method to update the NCFET permittivity matrix
    '''
    # Original permittivity values
    primordial_vals = list(FE_dict.values())

    # System size
    N = len(FE_dict.keys())

    # Initializing Jacobi Matrix
    J_mat = np.zeros(shape=(N, N))

    # Stores the differential change in permittivity that should be applied to compute partial derivatives
    h_vect = np.array(primordial_vals) / 1E+4
    print(h_vect)

    for i in FE_dict.keys():
        error_temp = []

        FE_dict[i] += h_vect[i]
        NCFET.update_permittivity(FE_dict)

        error_temp = single_step_solve(V, f, NCFET, rem_flag_dict, bcs, E_values, P_values)

        for k in error_temp:
            print(k)

        # Reset original permittivity
        FE_dict = dict([(key, primordial_vals[key]) for key in range(N)])
        NCFET.update_permittivity(FE_dict)

    # What up bitch

    FE_dict[0] = FE_dict[0] - 1.0 * 0.206
    FE_dict[1] = FE_dict[1] + 1.0 * 0.127
    NCFET.update_permittivity(FE_dict)
