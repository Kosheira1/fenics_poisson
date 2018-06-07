from fenics import *

tol = 1E-12


# Method to define domains

def setup_domains(width):
	
	FE = Ferroelectric(width)
	SC = Sem_channel(width)

	return (FE, SC)

class Sem_channel(SubDomain):
	def __init__(self, width):
		self.width = width
		SubDomain.__init__(self) # Call base class constructor
	def inside(self, x, on_boundary):
		return x[1] <= self.width + tol

class Ferroelectric(SubDomain):
	def __init__(self, width):
		self.width = width
		SubDomain.__init__(self) # Call base class constructor
	def inside(self, x, on_boundary):
		return x[1] >= self.width - tol




# Method to define essential boundary conditions

def setup_boundaries(V, volt_bias):

	tol = 1E-14

	u_L = Expression('0', degree=2)

	class U_Bound(SubDomain):
		def inside(self, x, on_boundary):
			tol = 1E-14
			return on_boundary and near(x[1], 0, tol)

	boundary_L = U_Bound()

	bc_L = DirichletBC(V, u_L, boundary_L)

	u_R = Expression(str(volt_bias), degree=2)

	def boundary_R(x, on_boundary):
		tol = 1E-14
		return on_boundary and near(x[1], 3, tol)

	bc_R = DirichletBC(V, u_R, boundary_R)

	bcs = [bc_R]
	
	return bcs



