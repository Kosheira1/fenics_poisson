from fenics import *
import numpy as np
from setup_domains import *

# Import plot library
import matplotlib.pyplot as plt


def solver(f, Permi, V, bcs, degree=2, u_prev=Constant(-0.0)):
	"""
	Solves non-linear Poisson equation on [0,1]x[0,3] with Langrange elements
	"""
	# Define variational problem
	u = TrialFunction(V)
	v = TestFunction(V)
	g = Constant(-0.0)
	a = inner(Permi*grad(u), grad(v))*dx
	L = f*v*dx + g*v*ds

	# Compute solution
	u = Function(V)
	solve(a==L, u, bcs)

	return u

def run_solver(sem_width, sem_relperm, doping, volt_bias):
	"Run solver to compute and post-process solution"

	#Material Parameters

	epsilon_0 = 1.0 #[F*um^-1]
	z_thick = 1.0   #[um]

	#Create mesh and define function space
	# 108, 60, 48
	mesh = RectangleMesh(Point(0,0), Point(1,3), 20, 360)
	V = FunctionSpace(mesh, 'P', 2) #maybe re-define degree
	bcs = setup_boundaries(V, volt_bias)

	# Define a Mesh Function which
	materials = MeshFunction('size_t', mesh, 2)

	# Define Domains and assign material identifier
	(FE, SC) = setup_domains(sem_width)
	FE.mark(materials, 0)
	SC.mark(materials, 1)

	Con_M = Permittivity_Tensor_M(materials, 0.0, degree=2)

	marked_cells = SubsetIterator(materials, 0)

	"""
	for cell in marked_cells:
		print('%2d %3f' % (cell.index(), cell.midpoint().x()))
	"""

	C = as_matrix(((Con_M[0], Con_M[1]), (Con_M[1], Con_M[2])))

	#Initialize Charge Expression
	f = Charge(degree=2)

	#Run an Initial solution

	#Find actual depletion width using a simple bisection method. Will be replaced with Fermi-level adaptation in the future.

	eta = 1E-2
	error = 1E+4
	d_error = 1E+4
	a_init = 0.0
	b_init = sem_width

	deplwidth_init = 1.0 # [um]

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

	Con_trial = Permittivity_Tensor_M(materials, flux_y, degree=2)
	return (u, Con_M, deplwidth_init*doping)


#Define charge function based on doping and depletion width (depletion approx. only)
# Input arguments: thickness of channel, doping concentration, depletion width

class Charge(Expression):
	def set_param(self, thick1, doping, width):
		self.thick1, self.doping, self.width = thick1, doping, width

	def eval(self, value, x):
		tol = 1E-14
		if x[1] >= (self.thick1 - self.width)+tol and x[1] <= (self.thick1+tol):
			value[0] = self.doping
		else:
			value[0] = 0.0

#Defining the elements of the two-dimensional dielectric tensor through the material mesh function, values might depend on previously computed polarization fields.

class Permittivity_Tensor_M(Expression):
	def __init__(self, materials, flux, **kwargs):
		self.materials, self.flux = materials, flux

	def eval_cell(self, values, x, cell):

		#Iterate over ferroelectric material points
		if self.materials[cell.index] == 0:
			values[0]=1.0  #e_xx
			values[1]=0.0  #e_xy = e_yx
			values[2]=-3.0 #e_yy

		#Iterate over semiconductor channel material points
		elif self.materials[cell.index] == 1:
			values[0]=1.0	#e_xx
			values[1]=0.0	#e_xy = e_yx
			values[2]=1.0	#e_yy
		#Rest should be vaccuum
		else:
			values[0]=1.0	#e_xx
			values[1]=0.0	#e_xy = e_yx
			values[2]=1.0	#e_yy			

	def value_shape(self):
		return (3,)

