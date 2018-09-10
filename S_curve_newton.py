from fenics import *
import numpy as np
from setup_domains import setup_boundaries
from Expressions import *
from solver import solver
from auxiliary_solverfunctions import read_hysteresis, compute_fields, plot_routine, intermediate_S_plot, FEM_solverexpressions

# Import plot library
import matplotlib.pyplot as plt


def newton_solver_S(V, NCFET, FE_dict, dimensions, volt_bias, max_it, rem_flag_dict):
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
    '''
    big_index = NCFET.material_cellindex(0)
    chi_1 = NCFET.permi[big_index] - 1
    FE_dict[0] = (1 + chi_1) - 1.0 * 0.206
    FE_dict[1] = FE_dict[1] + 1.0 * 0.127
    NCFET.update_permittivity(FE_dict)
    '''
    # Initial solve
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

    # Initialize Loop Variables, state_init is set to 0 because for zero electric external field, the FE should be in the negative capacitance regime!
    eta = 1E-3
    counter = 0
    ratio = 0.8
    if (volt_bias > 1.5):
        ratio = -0.4
    # For each bias point we want to track the trajectory of the solution in the P-E space while it converges onto a point on the S-Curve, used for plotting the trajectory and observing convergence!
    P_space = []
    E_space = []

    remnant_pol = NCFET.P_G

    # Solving Poisson's equation and updating relative permittivity in a loop. This approach is used because the field dependence of polarization is not explicitly used in the PDE formulation!!
    while (np.linalg.norm(error) > eta and counter < max_it or counter == 0):
        # Each single domain FE has its own trajectory
        P_sing_iter = []
        E_sing_iter = []
        error_temp = []

        # Only iterate over FE material points as only they change permittivity as a function of applied external electric field!
        # Update routine
        # Loop over all single-domain FE materials
        for vals in FE_dict.keys():
            # Keep track of state of Ferroelectric, could be different for each single domain material.
            big_index = NCFET.material_cellindex(vals)
            point = point_list[vals]
            P_sing_iter.append(pol_y(point) * 1E+6 * 1E+8)
            E_sing_iter.append(elec_y(point) * 1E-2)

            if (rem_flag_dict[vals] == 0):
                # Define susceptibility as ratio P/E and update permittivity locally

                if (vals == 1):
                    chi_1 = NCFET.permi[big_index] - 1
                    FE_dict[vals] = (1 + chi_1) - 0.01

                NCFET.update_permittivity(FE_dict)

            if (rem_flag_dict[vals] == 1):
                # Define susceptibility as ratio P/E and update permittivity locally
                chi_1 = (pol_y(point) - remnant_pol) / np.interp(pol_y(point), P_values, E_values) / epsilon_0
                FE_dict[vals] = (1 + chi_1) * ratio + NCFET.permi[big_index] * (1 - ratio)

                NCFET.update_permittivity(FE_dict)

        # Prepare Expressions for solver
        (C, Po, P) = FEM_solverexpressions(V, NCFET, rem_flag_dict)
        # Solve Variational Problem
        u = solver(f, C, Po, V, bcs, 2)
        (flux_y, elec_y, pol_y) = compute_fields(V, NCFET.mesh, C, u, P, epsilon_0)

        # Update Loop variables
        counter += 1
        error = error_temp

        for vals in FE_dict.keys():
            # Compare actual state of Polarization vs Electric field to S_curve data
            point = point_list[vals]
            error_temp.append(abs(np.interp(pol_y(point), P_values, E_values) * 1E-2 - elec_y(point) * 1E-2))

        print('\n'.join('{0:{0}d}  :  {1:.3e}'.format(k, vals) for k, vals in enumerate(error_temp)))

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

