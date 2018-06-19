from fenics import *


def solver(f, Permi, V, bcs, degree=2, u_prev=Constant(-0.0)):
    """
    Solves non-linear Poisson equation on [0,1]x[0,3] with Langrange elements
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
