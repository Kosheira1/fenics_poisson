from fenics import *
import numpy as np
from setup_domains import setup_boundaries
from Expressions import *
from solver import solver

# Import plot library
import matplotlib.pyplot as plt


def run_solver_S(V, mesh, dimensions, materials, permi, doping, volt_bias, max_it):
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

    # Initialize Loop Variables, state_init is set to 0 because for zero electric external field, the FE should be in the negative capacitance regime!
    eta = 1E-2
    error = 1E+4
    counter = 0
    state_init = 0  # 0 for neg-cap region, 1 for lower part, 2 for upper part. The lower

    # For each bias point we want to track the trajectory of the solution in the P-E space while it converges onto a point on the S-Curve, used for plotting the trajectory and observing convergence!
    P_space = []
    E_space = []

    # Solving Poisson's equation and updating relative permittivity in a loop. This approach is used because the field dependence of polarization is not explicitly used in the PDE formulation!!
    while (abs(error) > eta and counter < max_it):
        # Create the Permittivity Tensor for the anisotropic Poisson equation, FE material only exhibits field dependent permittivity in confinement direction!
        Con_M = Permittivity_Tensor_M(materials, permi, 0.0, degree=2)
        C = as_matrix(((Con_M[0], Con_M[1]), (Con_M[1], Con_M[2])))

        # Solve Variational Problem
        u = solver(f, C, V, bcs, 2)
        (flux_y, elec_y, pol_y) = compute_fields(V, mesh, C, u)

        # Defining the maximum error over all cells
        error_max = 0

        # Plot state of Ferroelectric, could be different for each point.
        point = (0.5, 1.5)
        P_space.append(pol_y(point))
        E_space.append(elec_y(point))

        # Setting Permittivity over Mesh function material indicators, we want to separate ferroelectric materials from everything else because the permittivity update loop only affects the former type
        marked_cells = SubsetIterator(materials, 0)

        # Only iterate over FE material points as only they change permittivity as a function of applied external electric field!
        for cells in marked_cells:
            x1 = cells.midpoint().x()
            y1 = cells.midpoint().y()
            point_temp = (x1, y1)

            # Update routine
            if (state_init == 0):
                # Compare actual state of Polarization vs Electric field to S_curve data
                error_temp = abs(np.interp(elec_y(point_temp), center_E[::-1], center_P[::-1]) - pol_y(point_temp))

                if(abs(x1 - 0.5) < 2E-2 and abs(y1 - 1.5) < 2E-2):
                    print(error_temp)
                    # print(cells.index())
                if (error_temp > 1E-2):
                    # Define susceptibility as ratio P/E and update permittivity locally
                    chi_1 = np.interp(elec_y(point_temp), center_E[::-1], center_P[::-1]) / elec_y(point_temp)

                    permi[cells.index()] = (1 + chi_1) * 0.8 + permi[cells.index()] * 0.2

                if (error_temp > error_max):
                    error_max = error_temp
                    print('y-coordinate of max_value: ' + "{0:.5f}".format(y1))
                if(error_temp > 8E-1 and counter >= 2):
                    print('y-coordinate of high_value: ' + "{0:.5f}".format(y1))

                # TO-DO: State transition condition

        print('Maximum Error over all cells: ' + "{0:.5f}".format(error_max))

        # Update Loop variables
        counter += 1
        error = error_max

    print('Value of D-Field at point: ' + "{0:.3f}".format(flux_y(point)) + ' (C*m^-2)')
    print('Value of E-Field at point: ' + "{0:.3f}".format(elec_y(point)) + ' (V*m^-1)')
    print('Value of P-Field at point: ' + "{0:.3f}".format(pol_y(point)) + ' (C*m^-2)')

    point = (0.5, 1.5)

    plot_routine(int(volt_bias))

    num_f = 3 * len(P_space)
    rgb = (np.arange(float(num_f)) / num_f).reshape(len(P_space), 3)
    plt.scatter(E_space, P_space, s=500, facecolors=rgb, label='Location in P-E space')
    plt.annotate("{0:.2f}".format(volt_bias), (E_space[-1], P_space[-1]))
    # plt.show()

    Con_M = Permittivity_Tensor_M(materials, permi, 0.0, degree=2)

    return(u, Con_M, -flux_y(point), P_space, E_space)


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


def compute_fields(V, mesh, C, u):
    # Compute D_field, E_field and P_field
    degree = V.ufl_element().degree()
    W = VectorFunctionSpace(mesh, 'P', degree)

    disp_field = project(C * grad(u), W)
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
