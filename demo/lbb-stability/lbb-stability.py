import argparse
import firedrake
from firedrake import assemble, inner, sym, grad, div, dx
import matplotlib.pyplot as plt
import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc
import tqdm


def compute_inf_sup_constant(spaces, b, inner_products):
    V, Q = spaces
    m, n = inner_products

    Z = V * Q
    u, p = firedrake.TrialFunctions(Z)
    v, q = firedrake.TestFunctions(Z)

    A = assemble(-(m(u, v) + b(v, p) + b(u, q)), mat_type='aij').M.handle
    M = assemble(n(p, q), mat_type='aij').M.handle

    opts = PETSc.Options()
    opts.setValue('eps_gen_hermitian', None)
    opts.setValue('eps_target_real', None)
    opts.setValue('eps_smallest_magnitude', None)
    opts.setValue('st_type', 'sinvert')
    opts.setValue('st_ksp_type', 'preonly')
    opts.setValue('st_pc_type', 'lu')
    opts.setValue('st_pc_factor_mat_solver_type', 'mumps')
    opts.setValue('eps_tol', 1e-8)

    num_values = 1
    eigensolver = SLEPc.EPS().create(comm=firedrake.COMM_WORLD)
    eigensolver.setDimensions(num_values)
    eigensolver.setOperators(A, M)
    eigensolver.setFromOptions()
    eigensolver.solve()

    Vr, Vi = A.getVecs()
    λ = eigensolver.getEigenpair(0, Vr, Vi)
    return λ


parser = argparse.ArgumentParser()
parser.add_argument('--dimension', type=int, choices=[2, 3])
parser.add_argument('--log-dx-min', type=int, default=2)
parser.add_argument('--log-dx-max', type=int, default=5)
parser.add_argument('--zdegree-min', type=int, default=1)
parser.add_argument('--zdegree-max', type=int, default=6)
parser.add_argument('--output')
args = parser.parse_args()


if args.dimension == 2:
    stability_constants = []
    for N in tqdm.trange(2**args.log_dx_min, 2**args.log_dx_max + 1):
        mesh = firedrake.UnitSquareMesh(N, N)
        d = mesh.topological_dimension()
        volume = assemble(firedrake.Constant(1) * dx(mesh))
        l = firedrake.Constant(volume ** (1 / d))

        V = firedrake.VectorFunctionSpace(mesh, 'CG', 2)
        Q = firedrake.FunctionSpace(mesh, 'CG', 1)
        try:
            b = lambda v, q: -q * div(v) * dx
            m = lambda u, v: (inner(u, v) + l**2 * inner(grad(u), grad(v))) * dx
            n = lambda p, q: p * q * dx
            λ = compute_inf_sup_constant((V, Q), b, (m, n))
        except PETSc.Error:
            λ = np.nan
        stability_constants.append(λ)

    fig, axes = plt.subplots()
    axes.plot(abs(np.array(stability_constants)))
    fig.savefig(args.output)

elif args.dimension == 3:
    kmin, kmax = args.zdegree_min, args.zdegree_max + 1
    fig, axes = plt.subplots()
    for k in tqdm.trange(kmin, kmax):
        stability_constants = []
        for N in tqdm.trange(2**args.log_dx_min, 2**args.log_dx_max + 1):
            mesh2d = firedrake.UnitSquareMesh(N, N, diagonal='crossed')
            mesh = firedrake.ExtrudedMesh(mesh2d, 1)
            d = mesh.topological_dimension()
            volume = assemble(firedrake.Constant(1) * dx(mesh))
            l = firedrake.Constant(volume ** (1 / d))

            V = firedrake.VectorFunctionSpace(mesh, 'CG', 2, vfamily='GL', vdegree=k + 1)
            Q = firedrake.FunctionSpace(mesh, 'CG', 1, vfamily='GL', vdegree=k)
            try:
                b = lambda v, q: -q * div(v) * dx
                m = lambda u, v: (inner(u, v) + l**2 * inner(grad(u), grad(v))) * dx
                n = lambda p, q: p * q * dx
                λ = compute_inf_sup_constant((V, Q), b, (m, n))
            except PETSc.Error:
                λ = np.nan
            stability_constants.append(λ)

        axes.plot(abs(np.array(stability_constants)), label=str(k))

    axes.legend()
    fig.savefig(args.output)

else:
    raise ValueError("Dimension must be 2 or 3!")
