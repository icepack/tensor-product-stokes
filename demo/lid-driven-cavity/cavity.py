import argparse
import firedrake
from firedrake import inner, sym, grad, div, dx

parser = argparse.ArgumentParser()
parser.add_argument('--horizontal-resolution', type=int, default=32)
parser.add_argument('--vertical-degree', type=int, default=4)
parser.add_argument('--viscosity', type=float, default=1e3)
# TODO: add options for Taylor-Hood, MINI, etc.
args = parser.parse_args()

# Make a mesh
N = args.horizontal_resolution
mesh2d = firedrake.UnitSquareMesh(N, N, diagonal='crossed')
mesh = firedrake.ExtrudedMesh(mesh2d, 1)

# Make function spaces
k = args.vertical_degree
Q = firedrake.FunctionSpace(mesh, 'CG', 1, vfamily='GL', vdegree=k)
V = firedrake.VectorFunctionSpace(mesh, 'CG', 2, vfamily='GL', vdegree=k + 1)

# Set up the weak form of the problem
Z = V * Q
z = firedrake.Function(Z)

u, p = firedrake.split(z)
ε = sym(grad(u))
μ = firedrake.Constant(args.viscosity)
τ = 2 * μ * ε
J = (0.5 * inner(τ, ε) - p * div(u)) * dx
F = firedrake.derivative(z)

# Set up BCs...

# Let the solver know about null spaces
basis = firedrake.VectorSpaceBasis(constant=True)
nullspace = firedrake.MixedVectorSpaceBasis(Z, [Z.sub(0), basis])

# Solve the thing
parameters = {
    'solver_parameters': {
        'mat_type': 'aij',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps'
    }
}
firedrake.solve(F == 0, z, bcs=bcs, nullspace=nullspace, **parameters)

# Check for horrors
u, p = z.split()
