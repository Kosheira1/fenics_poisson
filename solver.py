from fenics import *


def solver(f, Permi, V, bcs, degree=2, u_prev=Constant(-0.0)):
    """
    Solves non-linear, tensor-weighted Poisson equation on given function space and given boundary conditions.
    """
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    g = Constant(-0.0)
    a = inner(Permi * grad(u), grad(v)) * dx
    L = f * v * dx + g * v * ds

    # Compute solution
    u = Function(V)
    solve(a == L, u, bcs)

    return u
