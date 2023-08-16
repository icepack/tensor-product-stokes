import json
import argparse
import pathlib
import firedrake
from firedrake import assemble, inner, sym, grad, div, dx
import matplotlib.pyplot as plt
import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc


def compute_inf_sup_constant(spaces, b, inner_products):
    V, Q = spaces
    m, n = inner_products

    Z = V * Q
    u, p = firedrake.TrialFunctions(Z)
    v, q = firedrake.TestFunctions(Z)

    A = assemble(-(m(u, v) + b(v, p) + b(u, q)), mat_type="aij").M.handle
    M = assemble(n(p, q), mat_type="aij").M.handle

    opts = PETSc.Options()
    opts.setValue("eps_gen_hermitian", None)
    opts.setValue("eps_target_real", None)
    opts.setValue("eps_smallest_magnitude", None)
    opts.setValue("st_type", "sinvert")
    opts.setValue("st_ksp_type", "preonly")
    opts.setValue("st_pc_type", "lu")
    opts.setValue("st_pc_factor_mat_solver_type", "mumps")
    opts.setValue("eps_tol", 1e-8)

    num_values = 1
    eigensolver = SLEPc.EPS().create(comm=firedrake.COMM_WORLD)
    eigensolver.setDimensions(num_values)
    eigensolver.setOperators(A, M)
    eigensolver.setFromOptions()
    eigensolver.solve()

    Vr, Vi = A.getVecs()
    λ = eigensolver.getEigenpair(0, Vr, Vi)
    return λ


def run(*, dimension, num_cells, element, num_layers=None, vdegree=None):
    mesh = firedrake.UnitSquareMesh(num_cells, num_cells, diagonal="crossed")
    h = mesh.cell_sizes.dat.data_ro[:].min()

    cg_1 = firedrake.FiniteElement("CG", "triangle", 1)
    cg_2 = firedrake.FiniteElement("CG", "triangle", 2)
    b_3 = firedrake.FiniteElement("Bubble", "triangle", 3)
    if element == "taylor-hood":
        velocity_element = cg_2
        pressure_element = cg_1
    elif element == "mini":
        velocity_element = cg_1 + b_3
        pressure_element = cg_1
    elif element == "crouzeix-raviart":
        velocity_element = cg_2 + b_3
        pressure_element = firedrake.FiniteElement("DG", "triangle", 1)

    if dimension == 3:
        cg_z = firedrake.FiniteElement("CG", "interval", vdegree)
        dg_z = firedrake.FiniteElement("DG", "interval", vdegree - 1)
        velocity_element = firedrake.TensorProductElement(velocity_element, cg_z)
        pressure_element = firedrake.TensorProductElement(pressure_element, dg_z)
        mesh = firedrake.ExtrudedMesh(mesh, num_layers)

    V = firedrake.VectorFunctionSpace(mesh, velocity_element)
    Q = firedrake.FunctionSpace(mesh, pressure_element)

    # This is always just going to be equal to 1 in our case, but in general
    # you should compute a characteristic length scale for the domain when
    # defining the H1 norm.
    d = mesh.topological_dimension()
    volume = assemble(firedrake.Constant(1) * dx(mesh))
    l = firedrake.Constant(volume ** (1 / d))

    b = lambda v, q: -q * div(v) * dx
    m = lambda u, v: (inner(u, v) + l**2 * inner(grad(u), grad(v))) * dx
    n = lambda p, q: p * q * dx
    try:
        λ = compute_inf_sup_constant((V, Q), b, (m, n)).real
    except PETSc.Error:
        λ = np.nan
    return h, λ


parser = argparse.ArgumentParser()
parser.add_argument("--dimension", type=int, choices=[2, 3])
elements = ["taylor-hood", "crouzeix-raviart", "mini"]
parser.add_argument("--element", choices=elements, default="taylor-hood")
parser.add_argument("--num-cells-min", type=int, default=4)
parser.add_argument("--num-cells-max", type=int, default=32)
parser.add_argument("--num-cells-step", type=int, default=1)
parser.add_argument("--num-layers", type=int, default=1)
parser.add_argument("--vdegree", type=int, default=1)
parser.add_argument("--output")
args = parser.parse_args()

stability_constants = []
nmin, nmax, nstep = args.num_cells_min, args.num_cells_max, args.num_cells_step
print(f"Element: {args.element}")
print(f"#layers: {args.num_layers}")
print(f"vdegree: {args.vdegree}")
for num_cells in range(nmin, nmax + 1, nstep):
    h, λ = run(
        dimension=args.dimension,
        num_cells=num_cells,
        element=args.element,
        num_layers=args.num_layers,
        vdegree=args.vdegree
    )
    stability_constants.append((h, λ))
    print(".", flush=True, end="")

data = []
if pathlib.Path(args.output).is_file():
    with open(args.output, "r") as output_file:
        data = json.load(output_file)

result = {
    "dimension": args.dimension,
    "element": args.element,
    "results": stability_constants,
}
if args.dimension == 3:
    result.update({"num_layers": args.num_layers, "vdegree": args.vdegree})
data.append(result)
with open(args.output, "w") as output_file:
    json.dump(data, output_file)
