"""
load_check.py  -  read truss.msh in DOLFINx and check it before solving.
Run inside your FEniCSx Docker container:   python3 load_check.py truss.msh
"""
import sys
import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem

path = sys.argv[1] if len(sys.argv) > 1 else "truss.msh"

# DOLFINx >= 0.10 renamed dolfinx.io.gmshio -> dolfinx.io.gmsh
try:
    from dolfinx.io import gmsh as gmshio
    data = gmshio.read_from_msh(path, MPI.COMM_WORLD, 0, gdim=2)
    domain, cell_tags, facet_tags = data.mesh, data.cell_tags, data.facet_tags
except (ImportError, AttributeError):
    from dolfinx.io import gmshio
    domain, cell_tags, facet_tags = gmshio.read_from_msh(path, MPI.COMM_WORLD, 0, gdim=2)

TAGS = {"support_left": 11, "support_right": 12, "roof_left": 21, "roof_right": 22, "other": 30}
b = 0.076                                                   # m

dx = ufl.Measure("dx", domain=domain)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
n = ufl.FacetNormal(domain)
I = lambda form: domain.comm.allreduce(fem.assemble_scalar(fem.form(form)), op=MPI.SUM)

area = I(1.0 * dx)
print(f"cells: {domain.topology.index_map(2).size_global}, "
      f"coordinate element degree: {domain.geometry.cmap.degree}")
print(f"area = {area:.5f} m^2   V = b*A = {b*area:.5f} m^3   self-weight = {500*9.81*b*area:.1f} N")
for k in ("support_left", "support_right"):
    print(f"{k:14s} length = {1e3*I(1.0*ds(TAGS[k])):.2f} mm  (expect 100)")
for k in ("roof_left", "roof_right"):
    # |t_x| = |n_y| on a 2D boundary -> integral is the horizontal projection
    print(f"{k:14s} length = {I(1.0*ds(TAGS[k])):.4f} m, "
          f"horizontal projection = {I(abs(n[1])*ds(TAGS[k])):.4f} m")
untagged = domain.topology.index_map(1).size_local  # informational
print("facet tags present:", sorted(set(np.unique(facet_tags.values).tolist())))
