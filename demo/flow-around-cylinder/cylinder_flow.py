import argparse
import subprocess
import numpy as np
from numpy import pi as π
import matplotlib.pyplot as plt
import pygmsh
import firedrake
from firedrake import sqrt, sym, inner, grad, div, dx, ds, ds_v, Constant


parser = argparse.ArgumentParser()
parser.add_argument("--coordinate-degree", type=int, default=1)
parser.add_argument("--mesh-spacing", type=float, default=0.05)
parser.add_argument(
    "--element", choices=["mini", "taylor-hood", "crouzeix-raviart"]
)
parser.add_argument("--output")
args = parser.parse_args()


# TODO: Figure out how to make it work with pygmsh-7.0.
if not hasattr(pygmsh, "built_in"):
    raise RuntimeError("Must run using pygmsh < 7.0, new version is broke")


def create_geometry(width, height, radius, position, lcar):
    geometry = pygmsh.built_in.Geometry()
    points = [
        geometry.add_point((-Lx / 2, -Ly / 2, 0), lcar=lcar),
        geometry.add_point((+Lx / 2, -Ly / 2, 0), lcar=lcar),
        geometry.add_point((+Lx / 2, +Ly / 2, 0), lcar=lcar),
        geometry.add_point((-Lx / 2, +Ly / 2, 0), lcar=lcar),
    ]
    lines = [
        geometry.add_line(p1, p2) for p1, p2 in
        zip(points, points[1:] + [points[0]])
    ]
    [geometry.add_physical(line) for line in lines]
    outer_loop = geometry.add_line_loop(lines)

    r = 0.25
    cpoints = [
        geometry.add_point(
            (radius * np.cos(θ), position + radius * np.sin(θ), 0), lcar=lcar
        )
        for θ in np.linspace(0, 3 * π / 2, 4)
    ]
    center = geometry.add_point((0, position, 0), lcar=lcar)
    arcs = [
        geometry.add_circle_arc(q1, center, q2)
        for q1, q2 in zip(cpoints, cpoints[1:] + [cpoints[0]])
    ]
    geometry.add_physical(arcs)
    inner_loop = geometry.add_line_loop(arcs)

    surface = geometry.add_plane_surface(outer_loop, [inner_loop])
    geometry.add_physical(surface)
    with open("domain.geo", "w") as geo_file:
        geo_file.write(geometry.get_code())
    subprocess.run("gmsh -v 0 -2 -o domain.msh domain.geo".split())


# Create an initial piecewise linear geometry of the domain.
Lx, Ly = 1.0, 2.0
position = -Ly / 8
radius = Lx / 4
create_geometry(
    width=Lx, height=Ly, radius=radius, position=position, lcar=args.mesh_spacing
)
initial_mesh = firedrake.Mesh("domain.msh")

# Create a higher-order coordinate function space, a Dirichlet BC that will fit
# the coordinates to the cylinder in the center of the domain, and a new high-
# order coordinate field.
cdegree = args.coordinate_degree
cylinder_ids = (5,)
if cdegree > 1:
    Vc = firedrake.VectorFunctionSpace(initial_mesh, "CG", cdegree)
    ξ = Constant((0, position))
    r = Constant(radius)
    def fixup(x):
        distance = sqrt(inner(x - ξ, x - ξ))
        return ξ + r * (x - ξ) / distance

    x = firedrake.SpatialCoordinate(initial_mesh)
    coordinate_bc = firedrake.DirichletBC(Vc, fixup(x), cylinder_ids)
    X0 = firedrake.interpolate(initial_mesh.coordinates, Vc)
    X = X0.copy(deepcopy=True)
    coordinate_bc.apply(X)

    base_mesh = firedrake.Mesh(X)
else:
    base_mesh = initial_mesh

# Extrude the base mesh into 3D, create the MINI element in 2D, and extrude that
# element into 3D
# TODO: Pick the vertical degree and the number of layers at the command line
num_layers = 2
mesh = firedrake.ExtrudedMesh(base_mesh, num_layers, name="mesh")
vdegree = 4
cg_z = firedrake.FiniteElement("CG", "interval", vdegree)
dg_z = firedrake.FiniteElement("DG", "interval", vdegree - 1)

if args.element == "mini":
    degree = 1
    cg_x = firedrake.FiniteElement("CG", "triangle", degree)
    b_x = firedrake.FiniteElement("Bubble", "triangle", degree + 2)
    velocity_element = firedrake.TensorProductElement(cg_x + b_x, cg_z)
    pressure_element = firedrake.TensorProductElement(cg_x, dg_z)
elif args.element == "taylor-hood":
    cg_x2 = firedrake.FiniteElement("CG", "triangle", 2)
    cg_x1 = firedrake.FiniteElement("CG", "triangle", 1)
    velocity_element = firedrake.TensorProductElement(cg_x2, cg_z)
    pressure_element = firedrake.TensorProductElement(cg_x1, dg_z)
elif args.element == "crouzeix-raviart":
    cg_x = firedrake.FiniteElement("CG", "triangle", 2)
    b_x = firedrake.FiniteElement("Bubble", "triangle", 3)
    dg_x = firedrake.FiniteElement("DG", "triangle", 1)
    velocity_element = firedrake.TensorProductElement(cg_x + b_x, cg_z)
    pressure_element = firedrake.TensorProductElement(dg_x, dg_z)

V = firedrake.VectorFunctionSpace(mesh, velocity_element)
Q = firedrake.FunctionSpace(mesh, pressure_element)
Z = V * Q

z = firedrake.Function(Z)
u, p = firedrake.split(z)

# Make the velocity field equal to 0 on the top and bottom surfaces and on the
# cylinder in the middle of the domain. Then make the pressure equal to 1 on
# the inflow boundary and 0 on the outflow boundary.
bc_u_top = firedrake.DirichletBC(Z.sub(0), Constant((0, 0, 0)), "top")
bc_u_bot = firedrake.DirichletBC(Z.sub(0), Constant((0, 0, 0)), "bottom")
side_wall_ids = (2, 4)
dirichlet_ids = side_wall_ids + cylinder_ids
bc_u_sides = firedrake.DirichletBC(Z.sub(0), Constant((0, 0, 0)), dirichlet_ids)

inflow_ids = (1,)
outflow_ids = (3,)
p_in = Constant(1.0)
p_out = Constant(0.0)
bc_p_in = firedrake.DirichletBC(Z.sub(1), p_in, inflow_ids)
bc_p_out = firedrake.DirichletBC(Z.sub(1), p_out, outflow_ids)

bcs = [bc_u_top, bc_u_bot, bc_u_sides, bc_p_in, bc_p_out]

# Form the Lagrangian for the Stokes equations. This consists of the viscous
# energy dissipation, the constraint that the divergence of the velocity field
# is 0, and the stress forcing on the inflow and outflow boundaries.
μ = Constant(1.0)
ε = sym(grad(u))
τ = 2 * μ * ε
L_interior = (0.5 * inner(τ, ε) - p * div(u)) * dx

n = firedrake.FacetNormal(mesh)
L_inflow = p_in * inner(u, n) * ds_v(inflow_ids)
L_outflow = p_out * inner(u, n) * ds_v(outflow_ids)
L = L_interior + L_inflow + L_outflow
F = firedrake.derivative(L, z)

# Solve the PDE. As a smoke text, calculate the total flux into and out of the
# domain and check that they're roughly equal.
problem = firedrake.NonlinearVariationalProblem(F, z, bcs)
solver = firedrake.NonlinearVariationalSolver(problem)
solver.solve()

u, p = z.subfunctions
volume = firedrake.assemble(Constant(1) * dx(mesh))
rms_velocity = np.sqrt(firedrake.assemble(u**2 * dx) / volume)

n = firedrake.FacetNormal(mesh)
inflow_area = firedrake.assemble(Constant(1) * ds_v(domain=mesh, subdomain_id=inflow_ids))
outflow_area = firedrake.assemble(Constant(1) * ds_v(domain=mesh, subdomain_id=outflow_ids))
influx = firedrake.assemble(inner(u, n) * ds_v(inflow_ids)) / inflow_area
outflux = firedrake.assemble(inner(u, n) * ds_v(outflow_ids)) / outflow_area

print(f"RMS velocity: {rms_velocity}")
print(f"Influx:       {influx}")
print(f"Outflux:      {outflux}")
print(f"Flux error:   {(influx + outflux) / outflux}")

# Save the result to disk
with firedrake.CheckpointFile(args.output, "w") as chk:
    chk.save_mesh(mesh)
    chk.save_function(u, name="velocity")
    chk.save_function(p, name="pressure")
