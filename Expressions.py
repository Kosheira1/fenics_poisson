from fenics import *
from Parameter_class import Device
# Define charge function based on doping and depletion width (depletion approx. only)
# Input arguments: thickness of channel, doping concentration, depletion width


class Charge(Expression):
    def set_param(self, thick1, doping, width):
        self.thick1, self.doping, self.width = thick1, doping, width

    def eval(self, value, x):
        tol = 1E-14
        if x[1] >= (self.thick1 - self.width) + tol and x[1] <= (self.thick1 + tol):
            value[0] = self.doping
        else:
            value[0] = 0.0

# Defining the elements of the two-dimensional dielectric tensor through the material mesh function, values depend on previously computed polarization fields.


class Permittivity_Tensor_M(Expression):
    def __init__(self, materials, permi, domains, **kwargs):
        self.materials, self.permi, self.domains = materials, permi, domains
        self.FE_layers = self.domains['Ferroelectric']
        self.length = len(self.FE_layers)

    def eval_cell(self, values, x, cell):
        # Iterate over ferroelectric material points
        if self.materials[cell.index] < self.length:
            values[0] = 18.3  # e_xx
            values[1] = 0.0  # e_xy = e_yx
            values[2] = self.permi[cell.index]  # e_yy

        # Iterate over other material points
        else:
            values[0] = self.permi[cell.index]  # e_xx
            values[1] = 0.0  # e_xy = e_yx
            values[2] = self.permi[cell.index]  # e_yy

    def value_shape(self):
        return (3,)

# Defining the elements of the two-dimensional remnant polarization for different operating points.


class Remnant_Pol(Expression):
    def __init__(self, materials, flag, P_R, domains, ** kwargs):
        self.materials, self.flag, self.P_R, self.domains = materials, flag, P_R, domains
        self.FE_layers = self.domains['Ferroelectric']
        self.length = len(self.FE_layers)

    def eval_cell(self, values, x, cell):
        # Iterate over ferroelectric material points
        values[0] = 0.0  # P_r,x
        identifier = self.materials[cell.index]
        if (identifier < self.length) and (self.flag[identifier] == 1):
            values[1] = self.P_R[cell.index]

        else:
            values[1] = 0.0

    def value_shape(self):
        return(2,)
