import argparse
import matplotlib.pyplot as plt
import firedrake

parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--output")
args = parser.parse_args()

with firedrake.CheckpointFile(args.input, "r") as chk:
    mesh = chk.load_mesh("mesh")
    u = chk.load_function(mesh, name="velocity")
    p = chk.load_function(mesh, name="pressure")

base_mesh = mesh._base_mesh

degree = 1
cg_x = firedrake.FiniteElement("CG", "triangle", degree)
b_x = firedrake.FiniteElement("Bubble", "triangle", degree + 2)

# Compute the depth-averaged horizontal velocity field and plot it.
r = firedrake.FiniteElement("R", "interval", 0)
avg_pressure_element = firedrake.TensorProductElement(cg_x, r)
Q_avg = firedrake.FunctionSpace(mesh, avg_pressure_element)
p_avg = firedrake.project(p, Q_avg)
Q_2 = firedrake.FunctionSpace(base_mesh, cg_x)
p_2 = firedrake.Function(Q_2)
p_2.dat.data[:] = p_avg.dat.data_ro[:]

avg_velocity_element = firedrake.TensorProductElement(cg_x, r)
V_avg = firedrake.VectorFunctionSpace(mesh, avg_velocity_element, dim=2)
u_avg = firedrake.project(firedrake.as_vector((u[0], u[1])), V_avg)
V_2 = firedrake.VectorFunctionSpace(base_mesh, cg_x)
u_2 = firedrake.Function(V_2)
u_2.dat.data[:] = u_avg.dat.data_ro[:]

fig, ax = plt.subplots()
ax.set_aspect("equal")
colors = firedrake.quiver(u_2, axes=ax, headwidth=1)
fig.colorbar(colors)
fig.savefig(args.output, bbox_inches="tight")
