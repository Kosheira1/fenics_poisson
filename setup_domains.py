from fenics import *

# Method to define domains and interfaces


def setup_domains(width, FE_width):
    "Setup the layer structure in the order from depth-coordinate=0 to highest. Make sure to provide lower and upper bound"
    SC = ep_layer(0.0, width)
    FE = Ferroelectric(width, width + FE_width)

    return (SC, FE)

# Semiconductor Channel


class ep_layer(SubDomain):
    def __init__(self, lower, upper):
        self.lower, self.upper = lower, upper
        self.tol = 1E-12
        SubDomain.__init__(self)  # Call base class constructor

    def inside(self, x, on_boundary):
        return (x[1] <= self.upper + self.tol) and (x[1] >= self.lower - self.tol)

# Ferroelectric Insulator


class Ferroelectric(SubDomain):
    def __init__(self, lower, upper):
        self.lower, self.upper = lower, upper
        self.tol = 1E-12
        SubDomain.__init__(self)  # Call base class constructor

    def inside(self, x, on_boundary):
        return (x[1] <= self.upper + self.tol) and (x[1] >= self.lower - self.tol)

# Boundary between Layers


class LayerBoundary(SubDomain):
    def __init__(self, loc):
        self.loc = loc
        SubDomain.__init__(self)  # Call base class constructor

    def inside(self, x, on_boundary):
        tol = 0.2
        return near(x[1], self.loc, tol)

# Method to define essential boundary conditions


def setup_boundaries(V, volt_bias, top_coord):
    ""
    tol = 1E-14
    # Lower Bound Boundary Domain

    class L_Bound(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-14
            return on_boundary and near(x[1], 0, tol)
    # Upper Bound Boundary Domain

    class U_Bound(SubDomain):
        def __init__(self, top):
            self.top = top
            SubDomain.__init__(self)  # Call base class constructor

        def inside(self, x, on_boundary):
            tol = 1E-14
            return on_boundary and near(x[1], self.top, tol)

    class S_bound(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-14
            return on_boundary and near(x[0], 0, tol) and (x[1] > 0.2) and (x[1] < 0.4)

    class D_bound(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-14
            return on_boundary and near(x[0], 1.0, tol) and (x[1] > 0.2) and (x[1] < 0.4)

    u_L = Expression('0', degree=2)
    u_U = Expression(str(volt_bias), degree=2)

    u_S = Expression('0.4', degree=2)
    u_D = Expression('0.2', degree=2)

    boundary_L = L_Bound()
    boundary_U = U_Bound(top_coord)
    boundary_S = S_bound()
    boundary_D = D_bound()

    bc_L = DirichletBC(V, u_L, boundary_L)
    bc_U = DirichletBC(V, u_U, boundary_U)
    bc_S = DirichletBC(V, u_S, boundary_S)
    bc_D = DirichletBC(V, u_D, boundary_D)

    bcs = [bc_L, bc_U]

    return bcs
