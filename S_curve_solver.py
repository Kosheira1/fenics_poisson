from fenics import *
import numpy as np
from setup_domains import setup_boundaries
from Expressions import *
from solver import solver

# Import plot library
import matplotlib.pyplot as plt


def run_solver_S(V, NCFET, FE_dict, dimensions, volt_bias, max_it, rem_flag_dict):
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

    # Initialize Loop Variables, state_init is set to 0 because for zero electric external field, the FE should be in the negative capacitance regime!
    eta = 1E-3
    error = 1E+4
    counter = 0
    ratio = 0.8
    if (volt_bias > 1.5):
        ratio = -0.4
    # For each bias point we want to track the trajectory of the solution in the P-E space while it converges onto a point on the S-Curve, used for plotting the trajectory and observing convergence!
    P_space = []
    E_space = []

    remnant_pol = NCFET.P_G

    # Solving Poisson's equation and updating relative permittivity in a loop. This approach is used because the field dependence of polarization is not explicitly used in the PDE formulation!!
    while (abs(error) > eta and counter < max_it):
        # Prepare Expressions for solver
        (C, Po, P) = FEM_solverexpressions(V, NCFET, rem_flag_dict)
        # Solve Variational Problem
        u = solver(f, C, Po, V, bcs, 2)
        (flux_y, elec_y, pol_y) = compute_fields(V, NCFET.mesh, C, u, P, epsilon_0)

        # Defining the maximum error over all cells
        error_max = 0

        # Get a list of points that consitute the midpoints of all single-domain FE layers
        point_list = NCFET.FE_midpointlist

        # Each single domain FE has its own trajectory
        P_sing_iter = []
        E_sing_iter = []

        # Only iterate over FE material points as only they change permittivity as a function of applied external electric field!
        # Update routine
        # Loop over all single-domain FE materials
        for vals in FE_dict.keys():
            # Keep track of state of Ferroelectric, could be different for each single domain material.
            big_index = NCFET.material_cellindex(vals)
            point = point_list[vals]
            P_sing_iter.append(pol_y(point) * 1E+6 * 1E+8)
            E_sing_iter.append(elec_y(point) * 1E-2)

            # Compare actual state of Polarization vs       Electric field to S_curve data
            error_temp = abs(np.interp(pol_y(point), P_values, E_values) * 1E-2 - elec_y(point) * 1E-2)
            if (error_temp > error_max):
                error_max = error_temp

            if (rem_flag_dict[vals] == 0):

                if (error_temp > 1E-3):
                    # Define susceptibility as ratio P/E and update permittivity locally
                    chi_1 = pol_y(point) / np.interp(pol_y(point), P_values, E_values) / epsilon_0
                    FE_dict[vals] = (1 + chi_1) * ratio + NCFET.permi[big_index] * (1 - ratio)

                    NCFET.update_permittivity(FE_dict)

                # TO-DO: State transition condition

            if (rem_flag_dict[vals] == 1):

                if (error_temp > 1E-3):
                    # Define susceptibility as ratio P/E and update permittivity locally
                    chi_1 = (pol_y(point) - remnant_pol) / np.interp(pol_y(point), P_values, E_values) / epsilon_0
                    FE_dict[vals] = (1 + chi_1) * ratio + NCFET.permi[big_index] * (1 - ratio)

                    NCFET.update_permittivity(FE_dict)

        print('Maximum Error over all cells: ' + "{0:.5e}".format(error_max))

        # Update Loop variables
        counter += 1
        error = error_max

        # Update list of trajectories
        P_space.append(P_sing_iter)
        E_space.append(E_sing_iter)

    point_out = (0.5, 0.5)

    class InnerBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 1.5)

    boundary_parts = MeshFunction('size_t', NCFET.mesh, NCFET.mesh.topology().dim() - 1, 2)
    inner_bndr = InnerBoundary()
    inner_bndr.mark(boundary_parts, 1)

    dS = Measure('dS')(subdomain_data=boundary_parts)
    n = FacetNormal(NCFET.mesh)
    middle = dot(epsilon_0 * C * grad(u) + P, n)
    flux_l = middle('+') * dS(1)

    flux_int = assemble(flux_l)

    print('Value of D-Field computed through flux integral at point: ' + "{0:.3e}".format(flux_int) + ' (C*m^-2)')
    print('Value of D-Field at point: ' + "{0:.3e}".format(flux_y(point)) + ' (C*m^-2)')
    print('Value of D-Field at point_out: ' + "{0:.3e}".format(flux_y(point_out)) + ' (C*m^-2)')
    print('Value of E-Field at point: ' + "{0:.3e}".format(elec_y(point)) + ' (V*m^-1)')
    print('Value of P-Field at point: ' + "{0:.3e}".format(pol_y(point)) + ' (C*m^-2)')

    plot_routine(2 * int(volt_bias))
    intermediate_S_plot(0, max_it, P_space, E_space, volt_bias, perm_plot=False)

    plot_routine(2 * int(volt_bias) + 1)
    intermediate_S_plot(1, max_it, P_space, E_space, volt_bias, perm_plot=True)

    Con_M = Permittivity_Tensor_M(NCFET.materials, NCFET.permi, NCFET.domains, degree=2)

    return(u, Con_M, -flux_int, P_space, E_space)


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
