from fenics import *
import numpy as np
from setup_domains import *
from Expressions import *
from solver import *

# Import plot library
import matplotlib.pyplot as plt

def run_solver_sat(mesh, dimensions, materials, permi, doping, volt_bias):
	"Run solver with a saturation PE-model, see gen_saturation_lookup.py for details"
	#Read P-E curve
	filename = 'saturation_PE.dat'
	(E_values, P_values) = read_hysteresis('saturation_PE.dat')

	#Setting Permittivity over Mesh function material indicators
	marked_cells = SubsetIterator(materials, 0)
	
	# Create the function space and setup boundary conditions
	V = FunctionSpace(mesh, 'P', 2) #maybe re-define degree
	bcs = setup_boundaries(V, volt_bias, dimensions[0]+dimensions[1])

	# Run initial solution
	#Create the Permittivity Tensor
	Con_M = Permittivity_Tensor_M(materials, permi, 0.0, degree=2)
	C = as_matrix(((Con_M[0], Con_M[1]), (Con_M[1], Con_M[2])))

	#Initialize Charge Expression
	f = Constant(-0.0)

	#Solve Variational Problem
	u = solver(f, C, V, bcs, 2)

	(flux_y, elec_y, pol_y) = compute_fields(V, mesh, C, u)
	
	#Run Solution loop and update permittivity until convergence is reached.
	eta = 1E-2
	error = 1E+4
	counter = 1

	while (abs(error)>eta):
		# Defining the maximum error over all cells
		error_max = 0
	
		#Plot state of Ferroelectric
		plot_routine(counter)
		point = (0.5, 0.5)
		plt.scatter(elec_y(point), pol_y(point), s=500, c='red', label='Location in P-E space')
		plt.legend(loc=1, bbox_to_anchor=(1, 0.5))
		plt.show()

		for cells in marked_cells:
			x1 = cells.midpoint().x()
			y1 = cells.midpoint().y()
			point_temp = (x1, y1)
			
			# Update routine
			error_temp = abs(np.interp(elec_y(point_temp), E_values, P_values)-pol_y(point_temp))
			if (error_temp > 1E-2):
				chi_1 = np.interp(elec_y(point_temp), E_values, P_values)/elec_y(point_temp)
				permi[cells.index()]=1+chi_1
			
			if (error_temp > error_max):
				error_max = error_temp	

		print('Maximum Error over all cells: ' + "{0:.3f}".format(error_max))
			
		#Create the Permittivity Tensor
		Con_M = Permittivity_Tensor_M(materials, permi, 0.0, degree=2)
		C = as_matrix(((Con_M[0], Con_M[1]), (Con_M[1], Con_M[2])))	
		
		#Solve Variational Problem
		u = solver(f, C, V, bcs, 2)
		(flux_y, elec_y, pol_y) = compute_fields(V, mesh, C, u)
		
		#Update Loop variables
		counter += 1
		error = error_max
	
	print('Value of D-Field at point: ' + "{0:.3f}".format(flux_y(point)) + ' (C*m^-2)')
	print('Value of E-Field at point: ' + "{0:.3f}".format(elec_y(point)) + ' (V*m^-1)')
	print('Value of P-Field at point: ' + "{0:.3f}".format(pol_y(point)) + ' (C*m^-2)')
	print('Error: ' + "{0:.3f}".format(np.interp(elec_y(point), E_values, P_values)-flux_y(point)+elec_y(point)))

	return(u, Con_M, 0.0)

def read_hysteresis(filename):
	#Read saturation curve values
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
	#Compute D_field, E_field and P_field
	degree = V.ufl_element().degree()
	W = VectorFunctionSpace(mesh, 'P', degree)
	
	disp_field = project(C*grad(u), W)
	flux_x, flux_y = disp_field.split(deepcopy=True)  # extract components.

	electric_field = project(grad(u), W)
	elec_x, elec_y = electric_field.split(deepcopy=True) # extract components.

	pol_y = flux_y - elec_y

	return (flux_y, elec_y, pol_y)	

def plot_routine(num):
	#Read P-E curve
	filename = 'saturation_PE.dat'
	(E_values, P_values) = read_hysteresis('saturation_PE.dat')
	#Plotting
	plt.figure(num=12, figsize=(16,12))
	xvals = np.linspace(-7, 7, 100)
	yinterp = np.interp(xvals, E_values, P_values)
	plt.plot(E_values, P_values, 'o', linewidth=3, label='P-E data')
	plt.plot(xvals, yinterp, '-x', linewidth=2.5, label='Interpolation')
	plt.xlabel('Electric Field (kV/cm)')
	plt.ylabel('Polarization ' + r"$\frac{fC}{\mu m^2}$")	
	
	plt.grid(True)
