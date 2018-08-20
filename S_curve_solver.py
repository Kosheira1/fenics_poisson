from fenics import *
import numpy as np
from setup_domains import setup_boundaries
from Expressions import *
from solver import solver

# Import plot library
import matplotlib.pyplot as plt


def run_solver_S(V, mesh, dimensions, materials, FE, Pol_r, permi, doping, volt_bias, max_it, rem_flag):
    '''
    Run solver with a S-curve PE-model, see gen_S_lookup.py for details about physical quantities.
    '''
    # Read P-E curve
    filename = 's_curve_PE.dat'
    (E_values, P_values) = read_hysteresis(filename)

    # Transform to np arrays
    E_values = np.array(E_values)
    P_values = np.array(P_values)

    # Extract the relevant parts of the S-curve, three parts are defined because each of them represents a different physical state of the ferroelectric, i.e. negative or positive capacitance regime!
    center_P = P_values[np.where(abs(P_values) < 6.0)]
    center_E = E_values[np.where(abs(P_values) < 6.0)]

    low_P = P_values[np.where(P_values < -6.0)]
    low_E = E_values[np.where(P_values < -6.0)]

    up_P = P_values[np.where(P_values > 6.0)]
    up_E = E_values[np.where(P_values > 6.0)]

    # Setup boundary conditions
    bcs = setup_boundaries(V, volt_bias, dimensions[0] + dimensions[1])

    # Initialize Charge Expression
    f = Constant(-0.0)

    # Setting Permittivity over Mesh function material indicators, we want to separate ferroelectric materials from everything else because the permittivity update loop only affects the former type
    marked_cells = SubsetIterator(materials, 0)
    for cells in marked_cells:
        big_index = cells.index()
        continue

    remnant_pol = Pol_r[big_index]
    # Initialize Loop Variables, state_init is set to 0 because for zero electric external field, the FE should be in the negative capacitance regime!
    eta = 1E-3
    error = 1E+4
    counter = 0
    ratio = 0.8
    if (volt_bias > 1.5):
        ratio = 0.2
    # For each bias point we want to track the trajectory of the solution in the P-E space while it converges onto a point on the S-Curve, used for plotting the trajectory and observing convergence!
    P_space = []
    E_space = []

    # Solving Poisson's equation and updating relative permittivity in a loop. This approach is used because the field dependence of polarization is not explicitly used in the PDE formulation!!
    while (abs(error) > eta and counter < max_it):
        # Create the Permittivity Tensor for the anisotropic Poisson equation, FE material only exhibits field dependent permittivity in confinement direction!
        Con_M = Permittivity_Tensor_M(materials, permi, 0.0, degree=2)
        C = as_matrix(((Con_M[0], Con_M[1]), (Con_M[1], Con_M[2])))

        # Create the Vector Expression for the remnant Polarization, only non-zero in FE material and only exhibited in confinement direction
        P_r = Remnant_Pol(materials, rem_flag, Pol_r, degree=2)
        P = as_vector((P_r[0], P_r[1]))

        degree = V.ufl_element().degree()
        W = VectorFunctionSpace(mesh, 'P', degree)
        Po = project(P, W)

        # Solve Variational Problem
        u = solver(f, C, Po, V, bcs, 2)
        (flux_y, elec_y, pol_y) = compute_fields(V, mesh, C, u, P)

        # Defining the maximum error over all cells
        error_max = 0

        # Plot state of Ferroelectric, could be different for each single domain material.
        point = (0.5, 1.5)
        P_space.append(pol_y(point))
        E_space.append(elec_y(point))

        # Compare actual state of Polarization vs       Electric field to S_curve data
        error_temp = abs(np.interp(pol_y(point), P_values, E_values) - elec_y(point))
        print(error_temp)

        # Only iterate over FE material points as only they change permittivity as a function of applied external electric field!
        # Update routine
        # Loop over all single-domain FE materials

        if (rem_flag == 0):

            if (error_temp > 1E-3):
                # Define susceptibility as ratio P/E and update permittivity locally
                chi_1 = pol_y(point) / np.interp(pol_y(point), P_values, E_values)

                FE.mark(permi, (1 + chi_1) * ratio + permi[big_index] * (1 - ratio))

            if (error_temp > error_max):
                error_max = error_temp

            # TO-DO: State transition condition

        if (rem_flag == 1):

            if (error_temp > 1E-3):
                # Define susceptibility as ratio P/E and update permittivity locally
                chi_1 = (pol_y(point) - remnant_pol) / np.interp(pol_y(point), P_values, E_values)

                FE.mark(permi, (1 + chi_1) * ratio + permi[big_index] * (1 - ratio))

            if (error_temp > error_max):
                error_max = error_temp

        print('Maximum Error over all cells: ' + "{0:.5f}".format(error_max))

        # Update Loop variables
        counter += 1
        error = error_max

    point_out = (0.5, 0.5)

    class InnerBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 1.5)

    boundary_parts = MeshFunction('size_t', mesh, mesh.topology().dim() - 1, 2)
    inner_bndr = InnerBoundary()
    inner_bndr.mark(boundary_parts, 1)

    dS = Measure('dS')(subdomain_data=boundary_parts)
    n = FacetNormal(mesh)
    middle = dot(C * grad(u) + P, n)
    flux_l = middle('+') * dS(1)

    flux_int = assemble(flux_l)

    print('Value of D-Field computed through flux integral at point: ' + "{0:.3f}".format(flux_int) + ' (C*m^-2')
    print('Value of D-Field at point: ' + "{0:.3f}".format(flux_y(point)) + ' (C*m^-2)')
    print('Value of D-Field at point_out: ' + "{0:.3f}".format(flux_y(point_out)) + ' (C*m^-2)')
    print('Value of E-Field at point: ' + "{0:.3f}".format(elec_y(point)) + ' (V*m^-1)')
    print('Value of P-Field at point: ' + "{0:.3f}".format(pol_y(point)) + ' (C*m^-2)')

    plot_routine(int(volt_bias))

    num_f = 3 * len(P_space)
    rgb = (np.arange(float(num_f)) / num_f).reshape(len(P_space), 3)
    plt.scatter(E_space, P_space, s=500, facecolors=rgb, label='Location in P-E space')
    plt.annotate("{0:.2f}".format(volt_bias), (E_space[-1], P_space[-1]))
    # plt.show()

    Con_M = Permittivity_Tensor_M(materials, permi, 0.0, degree=2)

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


def compute_fields(V, mesh, C, u, Pol):
    # Compute D_field, E_field and P_field
    degree = V.ufl_element().degree()
    W = VectorFunctionSpace(mesh, 'P', degree)

    disp_field = project(C * grad(u) + Pol, W)
    flux_x, flux_y = disp_field.split(deepcopy=True)  # extract components.

    electric_field = project(grad(u), W)
    elec_x, elec_y = electric_field.split(deepcopy=True)  # extract components.

    pol_y = flux_y - elec_y

    return (flux_y, elec_y, pol_y)


def plot_routine(num):
    # Read P-E curve
    filename = 's_curve_PE.dat'
    (E_values, P_values) = read_hysteresis(filename)
    # Plotting
    plt.figure(num=12, figsize=(16, 12))
    # xvals = np.linspace(-7, 7, 100)
    # yinterp = np.interp(xvals, E_values, P_values)
    plt.plot(E_values, P_values, 'o', linewidth=3, label='P-E data')
    # plt.plot(xvals, yinterp, '-x', linewidth=2.5, label='Interpolation')
    plt.xlabel('Electric Field (kV/cm)')
    plt.ylabel('Polarization ' + r"$(\frac{fC}{\mu m^2})$")

    plt.grid(True)
