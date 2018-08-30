from fenics import *


def solver(f, Permi, Pol, V, bcs, degree=2, u_prev=Constant(-0.0)):
    """
    Solves non-linear, tensor-weighted Poisson equation on given function space and given boundary conditions.
    """
    # Define variational problem
    epsilon_0 = 8.85E-18  # (F*um^-1)
    u = TrialFunction(V)
    v = TestFunction(V)
    g = Constant(-0.0)
    a = inner(epsilon_0 * Permi * grad(u), grad(v)) * dx
    g_cj = div(Pol)

    L = f * v * dx + g * v * ds + g_cj * v * dx

    # Compute solution
    u = Function(V)
    solve(a == L, u, bcs, solver_parameters=dict(linear_solver='default', preconditioner='default'))  # THe dictionary argument allows for the use of different Krylov solvers

    return u
