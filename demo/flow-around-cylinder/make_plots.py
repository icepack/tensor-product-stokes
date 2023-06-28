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

# Get the horizontal element from the velocity space and create spaces both on
# the extruded space and the base mesh for the depth-averaged velocity
V = u.function_space()
elt_x = V.ufl_element().sub_elements()[0].sub_elements()[0]
r = firedrake.FiniteElement("R", "interval", 0)
avg_velocity_element = firedrake.TensorProductElement(elt_x, r)
V_avg = firedrake.VectorFunctionSpace(mesh, avg_velocity_element, dim=2)

base_mesh = mesh._base_mesh
V_2 = firedrake.VectorFunctionSpace(base_mesh, elt_x)

# Compute the depth-averaged horizontal velocity field and plot it.
u_avg = firedrake.project(firedrake.as_vector((u[0], u[1])), V_avg)
u_2 = firedrake.Function(V_2)
u_2.dat.data[:] = u_avg.dat.data_ro[:]

fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
for ax in axes:
    ax.set_aspect("equal")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

kw = {
    "interior_kw": {"linewidth": 0.25},
    "boundary_kw": {
        "linewidth": 2,
        "colors": [
            "tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:grey"
        ],
    },
}
firedrake.triplot(base_mesh, axes=axes[0], **kw)
axes[0].legend()
firedrake.quiver(u_2, axes=axes[1], headwidth=2)

fig.savefig(args.output, bbox_inches="tight")
