from fenics import *
import matplotlib.pyplot as plt
import matplotlib
import sys
import numpy as np

"""
Basic 3D plotting functionality

mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(10., 4., 2.), 10, 10, 10)
print("Plotting a BoxMesh")
plot(mesh, title="Box")
plt.show()
"""


def plot_solution(filename_mesh, *filename_solution):
    # Load Mesh and Solution Data
    mesh_old = Mesh(filename_mesh)
    num = len(filename_solution)
    V_old = FunctionSpace(mesh_old, 'P', 1)
    u = Function(V_old, filename_solution[0])  # could theoretically load more solution files here.

    # General Plotting Settings
    font = {'weight': 'bold',
            'size': 17}
    matplotlib.rc('font', **font)

    # Surface Plot of Solution
    plt.figure(num=0, figsize=(16, 12))
    plot(u)
    plt.title('Surface Plot of Potential V', y=1.03)
    plt.grid(True)
    plt.xlabel('x-coordinate [nm]')
    plt.ylabel('y-coordinate [nm]')

    # Compute and plot flux field
    V = u.function_space()
    mesh = V.mesh()
    degree = V.ufl_element().degree()
    W = VectorFunctionSpace(mesh, 'P', degree)
    grad_u = project(grad(u), W)
    point = (0.4, 0.61)
    valueuu = np.array(grad_u(point))
    print("The y-component of the electric field at x=0.4 and y=" + "{0:.3f}".format(point[1]) + " is " + "{0:.2f}".format(valueuu[1] / 100) + "MV/cm")
    print("The x-component of the electric field at x=0.4 and y=" + "{0:.3f}".format(point[1]) + " is " + "{0:.2f}".format(valueuu[0] / 100) + "MV/cm")
    value2 = u(point)
    print("The Potential at x=0.4 and y=" + "{0:.2f}".format(point[1]) + " is " + "{0:.2f}".format(value2) + "V")
    plt.figure(num=1, figsize=(16, 12))
    plot(grad_u, title='Electric Field')
    plt.xlabel('x-coordinate [nm]')
    plt.ylabel('y-coordinate [nm]')

    # Curve Plot along x=0.5
    plt.figure(num=2, figsize=(16, 12))
    tolerance = 0.01
    y = np.linspace(0 + tolerance, 2 - tolerance, 1001)
    points = [(0.5, y_prime) for y_prime in y]  # Create tuples of 2D points
    pot_line = np.array([u(point) for point in points])
    plt.plot(y, pot_line, 'k', linewidth=2)
    plt.title('Plot of potential along x=0.5')
    plt.grid(True)
    plt.xlabel('y-coordinate [nm]')
    plt.ylabel('Potential [V]')

    # Plot y-coordinate of electric field.
    pot_line = np.array([grad_u(point) for point in points])
    plt.figure(num=3, figsize=(16, 12))
    plt.plot(y, pot_line[:, 1] / 100, linewidth=3)
    plt.title('y-Component of Electric Field along x=0.5')
    plt.grid(True)
    plt.xlabel('y-coordinate [nm]')
    plt.ylabel('Electric Field [MV/cm]')

    # Display plots
    plt.show()


if __name__ == "__main__":
    try:
        first_arg = sys.argv[1]
        second_arg = sys.argv[2]
    except:
        print('Error: You failed to provide the correct number of input arguments: Function takes mesh file and solution file in .xml format')
        sys.exit(1)  # abort

    plot_solution(first_arg, second_arg)
