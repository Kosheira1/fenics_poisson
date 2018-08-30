from fenics import *

# Method to define domains and interfaces


def setup_domains(geometry):
    "Setup the 2-D structure given a coordinate data class"

    keys = ['Ferroelectric', 'Dielectric', 'Semiconductor', 'Metal']
    domain_dict = dict([(key, []) for key in keys])

    for FE_coords in geometry.data_dict['Ferroelectric']:
        domain_dict['Ferroelectric'].append(Ferroelectric(FE_coords))
    for DE_coords in geometry.data_dict['Dielectric']:
        domain_dict['Dielectric'].append(ep_layer(DE_coords))
    for SE_coords in geometry.data_dict['Semiconductor']:
        domain_dict['Semicondcutor'].append(ep_layer(SE_coords))
    for ME_coords in geometry.data_dict['Metal']:
        domain_dict['Metal'].append(ep_layer(ME_coords))

    return domain_dict

# Semiconductor Channel


class ep_layer(SubDomain):
    def __init__(self, coords_list):
        self.coords_list = coords_list
        self.tol = 1E-10
        SubDomain.__init__(self)  # Call base class constructor

    def inside(self, x, on_boundary):
        return (x[1] <= self.coords_list[1] + self.tol) and (x[1] >= self.coords_list[3] - self.tol) and(x[0] <= self.coords_list[2] + self.tol) and (x[0] >= self.coords_list[0] - self.tol)

# Ferroelectric Insulator


class Ferroelectric(SubDomain):
    def __init__(self, coords_list):
        self.coords_list = coords_list
        self.tol = 1E-10
        SubDomain.__init__(self)  # Call base class constructor

    def inside(self, x, on_boundary):
        return (x[1] <= self.coords_list[1] + self.tol) and (x[1] >= self.coords_list[3] - self.tol) and(x[0] <= self.coords_list[2] + self.tol) and (x[0] >= self.coords_list[0] - self.tol)

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
            return on_boundary and near(x[0], 0, tol) and (x[1] > 0.2 - tol) and (x[1] < 0.3 + tol)

    class D_bound(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-14
            return on_boundary and near(x[0], 1.0, tol) and (x[1] > 0.2 - tol) and (x[1] < 0.3 + tol)

    u_L = Expression('0+10*x[0]', degree=2)
    u_U = Expression(str(volt_bias), degree=2)

    u_S = Expression('20.0', degree=2)
    u_D = Expression('20.0', degree=2)

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
