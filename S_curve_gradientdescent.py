from fenics import *
import numpy as np
from setup_domains import setup_boundaries
from Expressions import *
from solver import solver
from auxiliary_solverfunctions import read_hysteresis, compute_fields, plot_routine, intermediate_S_plot, FEM_solverexpressions, initial_solve, single_step_solve, newton_step, gradient_step

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
    # Print Error Norm
    print('Initial_solver:' + ': Error norm: ' + '{0:.3e}'.format(np.linalg.norm(error)))

    # Initializing step size for gradient method
    previous_perm = dict([(0, FE_dict[key]) for key in range(len(FE_dict.keys()))])
    previous_gradient = [0 for key in range(len(FE_dict.keys()))]

    # Initialize Loop Variables, state_init is set to 0 because for zero electric external field, the FE should be in the negative capacitance regime!
    eta = 5E-4
    counter = 0

    point_list = NCFET.FE_midpointlist

    # Solving Poisson's equation and updating relative permittivity in a loop. This approach is used because the field dependence of polarization is not explicitly used in the PDE formulation!!
    while (np.linalg.norm(error) > eta and counter < max_it or counter == 0):
        # Each single domain FE has its own trajectory
        P_sing_iter = []
        E_sing_iter = []
        error_temp = []
        # Update permittivity for all single-domain FE materials using a newton Iteration
        # FE_dict = newton_step(V, f, NCFET, FE_dict, error, rem_flag_dict, bcs, E_values, P_values, P_space, E_space)
        (FE_dict, previous_perm, previous_gradient) = gradient_step(V, f, NCFET, FE_dict, error, rem_flag_dict, bcs, E_values, P_values, P_space, E_space, previous_perm, previous_gradient, counter, volt_bias)

        # Prepare Expressions for solver
        (C, Po, P) = FEM_solverexpressions(V, NCFET, rem_flag_dict)
        # Solve Variational Problem
        u = solver(f, C, Po, V, bcs, 2)
        (flux_y, elec_y, pol_y) = compute_fields(V, NCFET.mesh, C, u, P, epsilon_0)

        for vals in FE_dict.keys():
            # Compare actual state of Polarization vs Electric field to S_curve data
            point = point_list[vals]
            error_temp.append(abs(np.interp(pol_y(point), P_values, E_values) * 1E-2 - elec_y(point) * 1E-2))
            P_sing_iter.append(pol_y(point) * 1E+6 * 1E+8)
            E_sing_iter.append(elec_y(point) * 1E-2)

        print('\n'.join('{0:{0}d}  :  {1:.3e}'.format(k, vals) for k, vals in enumerate(error_temp)))

        # Print Error Norm
        print('Iteration ' + str(counter) + ': Error norm: ' + '{0:.3e}'.format(np.linalg.norm(error_temp)))

        # Update Loop variables
        counter += 1
        error = error_temp

        # Update list of trajectories
        P_space.append(P_sing_iter)
        E_space.append(E_sing_iter)

    point_out = (0.5, 0.5)
# Create a Gaussian surface that encloses the gate metal and compute total gate charge

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

    # plot_routine(2 * int(volt_bias))
    intermediate_S_plot(0, max_it, P_space, E_space, volt_bias, perm_plot=False)

    # plot_routine(2 * int(volt_bias) + 1)
    intermediate_S_plot(1, max_it, P_space, E_space, volt_bias, perm_plot=False)

    Con_M = Permittivity_Tensor_M(NCFET.materials, NCFET.permi, NCFET.domains, degree=2)

    return(u, Con_M, -flux_int, P_space, E_space)
