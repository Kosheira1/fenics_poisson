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
    Con_M = Permittivity_Tensor_M(NCFET.materials, NCFET.permi, NCFET.domains)
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
    print('\n'.join('{0:.2f} '.format(k) for k in primordial_vals))
    # print('Initial guess switching direction permittivities for each FE:' + str(primordial_vals) + "{0:.3f}".format(primordial_vals))

    # Damping coefficient
    damp = 1.0

    # System size
    N = len(FE_dict.keys())

    # Initializing Jacobi Matrix
    J_mat = np.zeros(shape=(N, N))

    # Stores the differential change in permittivity that should be applied to compute partial derivatives
    h_vect = np.array(primordial_vals) / 1E+4

    for i in FE_dict.keys():
        error_temp = []

        FE_dict[i] += h_vect[i]
        NCFET.update_permittivity(FE_dict)

        error_temp = single_step_solve(V, f, NCFET, rem_flag_dict, bcs, E_values, P_values)

        for k, vals in enumerate(error_temp):
            # Numerical differentiation and filling of J_mat, fill it up
            dfkdfi = (vals - error[k]) / h_vect[i]
            J_mat[k][i] = dfkdfi

        # Reset original permittivity
        FE_dict = dict([(key, primordial_vals[key]) for key in range(N)])
        NCFET.update_permittivity(FE_dict)

    # Inverting Jacobian and performing update step.
    x_n = np.array(list(FE_dict.values()))
    f_x_n = np.array(error)
    J_inv = np.linalg.inv(J_mat)
    dist = np.matmul(J_inv, f_x_n)

    print(dist)

    x_n -= damp * dist

    FE_dict = dict([(key, x_n[key]) for key in range(N)])
    NCFET.update_permittivity(FE_dict)

    return FE_dict


def gradient_step(V, f, NCFET, FE_dict, error, rem_flag_dict, bcs, E_values, P_values, P_space, E_space, pre_eps, pre_grad, counter, volt_bias):
    '''
    This function analyses the error locally and uses a gradient descent method to update the NCFET permittivity matrix
    '''
    # Determine System size
    N = len(FE_dict.keys())

    # Original permittivity values
    primordial_vals = list(FE_dict.values())
    # print('\n'.join('{0:.2f} '.format(k) for k in primordial_vals))

    # Original gradient values
    prim_gradient = [pre_grad[x] for x in range(len(pre_grad))]

    # Stores the differential change in permittivity that should be applied to compute partial derivatives
    h_vect = np.array(primordial_vals) / 1E+4

    # Compute norm of previous error
    n_error = np.linalg.norm(error)

    # Initialize gradient of error norm
    gradient = [0 for key in range(len(FE_dict.keys()))]

    for i in FE_dict.keys():
        error_temp = []

        FE_dict[i] += h_vect[i]
        NCFET.update_permittivity(FE_dict)

        error_temp = single_step_solve(V, f, NCFET, rem_flag_dict, bcs, E_values, P_values)

        n_error_temp = np.linalg.norm(error_temp)
        gradient[i] = (n_error_temp - n_error) / (h_vect[i])

        # Reset original permittivity
        FE_dict = dict([(key, primordial_vals[key]) for key in range(N)])
        NCFET.update_permittivity(FE_dict)

    if counter == 0:
        # Initial step size
        step_size = 100.0
        if volt_bias > 135:
            step_size = 60.0
    else:
        # Implementing secant equation approximation for step_size
        difference = np.array(primordial_vals) - np.array(pre_eps)
        grad_dif = np.array(gradient) - np.array(prim_gradient)
        numerator = np.dot(difference, grad_dif)
        normalization = np.linalg.norm(grad_dif) ** 2
        # print('The eps difference is' + repr(difference))
        step_size = numerator / normalization

        if counter < 2:
            step_size = 45.0
        elif counter >= 2 and counter < 5:
            step_size = 22
        elif counter >= 5 and counter < 7:
            step_size = 7.5
        elif counter >= 7:
            step_size = 3.2
        print('The step size is: ' + '{0:.2f}'.format(step_size))

    x_n = np.array(list(FE_dict.values()))
    dist = step_size * np.array(gradient)
    x_n -= dist

    FE_dict = dict([(key, x_n[key]) for key in range(N)])
    NCFET.update_permittivity(FE_dict)

    print('Updated state of FE permittivity:')
    print('\n'.join('{0:.2f} '.format(k) for k in list(FE_dict.values())))

    print('Current state of FE permittivity:')
    print('\n'.join('{0:.2f} '.format(k) for k in primordial_vals))

    print('Previous state of FE permittivity:')
    print('\n'.join('{0:.2f} '.format(k) for k in pre_eps))

    print('Gradient n:')
    print('\n'.join('{0:.3f} '.format(k) for k in gradient))

    print('Gradient n-1:')
    print('\n'.join('{0:.3f} '.format(k) for k in prim_gradient))

    # returning eps_{n+1}, eps_{n} and grad_{n}
    return(FE_dict, primordial_vals, gradient)
