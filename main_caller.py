"""
In this file, the displacement-field dependent Permittivity can be defined and the solver is ran. It also plots simulation results. 

"""
from __future__ import print_function
from refactor_solver import *
import numpy as np
from plot_differential_cap import *

# Import plot library
import matplotlib.pyplot as plt
import matplotlib


font = {'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

# Define Device Variables
sem_width = 2.7 # [um] Width of the semiconductor channel
sem_relperm = 1 # Relative permittivity of semiconductor channel
doping = -15.2   # [C/um^2] Acceptor doping concentration

# Create a custom mesh or read it from a file

mesh = RectangleMesh(Point(0,0), Point(1,3), 20, 360)
File('saved_mesh.xml') << mesh

#Bias points
volt_list_low = [float(x)/10 for x in range(2, 20, 4)] # [V]
volt_list_high = [float(x)/10 for x in range(25, 80, 7)] # [V]
volt_list = volt_list_low+volt_list_high

# Main Function: Solve Problem for all defined bias points

Solution_points = []
Permittivity_points = []
TotalCharge_points = []

for idx, bias in enumerate(volt_list):

	print("Bias Point: " + str(bias)  +  " The index, at which the bias point is extracted: " + str(idx))	
	(u_v, C_v, Q_v) = run_solver(mesh, sem_width, sem_relperm, doping, bias)
	Solution_points.append(u_v)
	Permittivity_points.append(C_v)
	TotalCharge_points.append(Q_v)
	print("")

# Write Charge and voltage value in file


cap_dat = np.array([volt_list, TotalCharge_points])
cap_dat = cap_dat.T
print(cap_dat)
np.savetxt('capdataV1.dat', cap_dat, fmt='%.3f')
plot_diff_cap('capdataV1.dat')

# Plot the first (or any) bias point

u = Solution_points[0]
File('saved_u.xml') << u

#Plot result
plt.figure(num=0, figsize=(16,12))
plot(u, title='Surface Plot of Potential V')
plt.grid(True)
plt.xlabel('x-coordinate [nm]')
plt.ylabel('y-coordinate [nm]')
#plot(u.function_space().mesh())

# Compute and plot flux field

V = u.function_space()
mesh = V.mesh()
degree = V.ufl_element().degree()
W = VectorFunctionSpace(mesh, 'P', degree)

grad_u=project(grad(u), W)

point = (0.4, 0.61)
valueuu = np.array(grad_u(point))

print("The y-component of the electric field at x=0.4 and y=" +"{0:.2f}".format(point[1]) +" is " + "{0:.2f}".format(valueuu[1]/100) + "MV/cm")

print("The x-component of the electric field at x=0.4 and y=" +"{0:.2f}".format(point[1]) +" is " + "{0:.2f}".format(valueuu[0]/100) + "MV/cm")

value2 = u(point)

print("The Potential at x=0.4 and y=" +"{0:.2f}".format(point[1]) +" is " + "{0:.2f}".format(value2) + "V")


plt.figure(num=1, figsize=(16,12))
plot(grad_u, title='Electric Field')
plt.xlabel('x-coordinate [nm]')
plt.ylabel('y-coordinate [nm]')

#Curve Plot along y=0.5

plt.figure(num=2, figsize=(16,12))
tolerance = 0.001 #  
y = np.linspace(0 + tolerance, 3 - tolerance, 101)
points = [(0.5, y_prime) for y_prime in y] #Create tuples of 2D points
pot_line = np.array([u(point) for point in points])
plt.plot(y, pot_line, 'k', linewidth=2)
plt.title('Plot of potential along x=0.5')
plt.grid(True)
plt.xlabel('y-coordinate [nm]')
plt.ylabel('Potential [V]')

#Plot y-coordinate of electric field.

pot_line = np.array([grad_u(point) for point in points])

plt.figure(num=3, figsize=(16,12))
plt.plot(y, pot_line[:,1]/100, linewidth=3)
plt.title('x-Component of Electric Field along x=0.5')
plt.grid(True)
plt.xlabel('y-coordinate [nm]')
plt.ylabel('Electric Field [MV/cm]')

# Display plots
plt.show()
