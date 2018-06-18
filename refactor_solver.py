from fenics import *
import numpy as np
from setup_domains import *
from Expressions import *
from solver import *

# Import plot library
import matplotlib.pyplot as plt

def run_solver_const(mesh, dimensions, materials, permi, doping, volt_bias):
	"Run solver to compute and post-process solution"

	# Create the function space
	V = FunctionSpace(mesh, 'P', 2) #maybe re-define degree
	bcs = setup_boundaries(V, volt_bias, dimensions[0]+dimensions[1])

	#Create the Permittivity Tensor
	Con_M = Permittivity_Tensor_M(materials, permi, 0.0, degree=2)
	C = as_matrix(((Con_M[0], Con_M[1]), (Con_M[1], Con_M[2])))

	#Initialize Charge Expression
	f = Charge(degree=2)

	sem_width = dimensions[0] # Initialize sem_width

	#Run an Initial solution
	#Find actual depletion width using a simple bisection method. Will be replaced with Fermi-level adaptation in the future.
	eta = 1E-2
	error = 1E+4
	d_error = 1E+4
	a_init = 0.0
	b_init = sem_width

	deplwidth_init = 0.5 # [um]

	#Evaluate Potential at lower bound of domain
	point  = (0.5, 1E-14)

	while (abs(error)>eta and abs(d_error)>1E-4):

		f.set_param(sem_width, doping, deplwidth_init)
		u = solver(f, C,  V, bcs, 2)
		
		if (u(point)) < 0:
			b_init = deplwidth_init
			deplwidth_init = 0.5 * (a_init + b_init)

		else:
			a_init = deplwidth_init
			deplwidth_init = 0.5 * (a_init + b_init)

		print(str(deplwidth_init))
		d_error = error - u(point) 	# Compute differential Error
		error = u(point)		# Error
		print(str(error))
		
	#Compute D_field
	degree = V.ufl_element().degree()
	W = VectorFunctionSpace(mesh, 'P', degree)
	
	disp_field = project(C*grad(u), W)
	flux_x, flux_y = disp_field.split(deepcopy=True)  # extract components.

	new_point = (0.5, 2.9)
	d_value = flux_y(new_point)

	print('The depletion width is ' + str(deplwidth_init))
	print('Total Charge through depletion is ' + str(deplwidth_init*doping))
	print('Total gate charge is ' + str(d_value))

	#Initialize Permittivity based on previous results
	#To-DO mah guy

	Con_trial = Permittivity_Tensor_M(materials, permi, flux_y, degree=2)
	return (u, Con_M, deplwidth_init*doping)
