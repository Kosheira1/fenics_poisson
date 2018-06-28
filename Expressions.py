from fenics import *

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
    def __init__(self, materials, permi, flux, **kwargs):
        self.materials, self.permi, self.flux = materials, permi, flux

    def eval_cell(self, values, x, cell):
        # Iterate over ferroelectric material points
        if self.materials[cell.index] == 0:
            values[0] = 1.0  # e_xx
            values[1] = 0.0  # e_xy = e_yx
            values[2] = self.permi[cell.index]  # e_yy

        # Iterate over semiconductor channel material points
        elif self.materials[cell.index] == 1:
            values[0] = 1.0  # e_xx
            values[1] = 0.0  # e_xy = e_yx
            values[2] = 1.0  # e_yy
        # Rest should be vaccuum
        else:
            values[0] = 1.0  # e_xx
            values[1] = 0.0  # e_xy = e_yx
            values[2] = 1.0  # e_yy

    def value_shape(self):
        return (3,)
