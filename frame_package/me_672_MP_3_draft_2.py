# %% [markdown]
# # ME672 A3: general plane-stress analysis of a timber roof frame
# 
# Works for **any** frame profile exported from FreeCAD (2D face BREP or the extruded solid BREP).
# Each design is described by a small JSON file (written by `rebuild_truss.py`):
# 
# ```json
# {"name": "baseline", "brep": "baseline_profile2d.brep", "units": "mm",
#  "nodes":   {"B0": [0, 0], "B1": [4000, 0], ...},
#  "members": {"B0-B1": 286, "A-R": 286, ...}}
# ```
# 
# Everything else is detected from the geometry: lower support face, support segments,
# exposed upper (loaded) boundary, joint fillets for mesh refinement, subdivision nodes
# (collinear, degree-2) that are not real joints, and clean member cross-sections for the
# axial-resultant and buckling checks.
# 
# Units inside the notebook: **m, N, Pa**.
# 
# Run everything headless (single reproduce command):
# `jupyter nbconvert --to notebook --execute frame_general.ipynb --output frame_general_run.ipynb`

# %% [markdown]
# ## 1. Settings

# %%
DESIGNS = [
    "designs/baseline.json",
    "designs/kingpost.json",
    "designs/Design_2.json",
    "designs/Design_3.json",
    "designs/Design_4.json",
    "designs/Design_4_40mm_Fillet.json",
    "designs/Design_5.json",
    "designs/Design_6.json",
]  # analysed in order
N_FRAMES = 5  # 12.0 m house length / 3.0 m spacing + 1 = 5 frames total
CONVERGENCE_DESIGNS = []                 # mesh study (slow); [] to skip
CONV_SCALES = [2.0, 1.0, 0.5]
OUTDIR = "results"

# mesh (sizes in m, multiplied by the scale)
ELEMENT_ORDER = 2
H_MAX = 0.030          # global element size cap
N_ACROSS = 6           # at least this many elements across the narrowest member
H_JOINT = 0.008        # near fillets / joint arcs
H_SUPPORT = 0.005      # near the support-segment ends
LOAD_SPAN_ONLY = True  # True: roof load only on 0 <= x <= 8 m (= truss projected area)

# frame data from the brief
b, s = 0.076, 3.0
E, nu, rho, g = 10e9, 0.30, 500.0, 9.81
f_t, f_c, f_s = 5e6, 5e6, 1e6
SUP_HALF = 0.050       # half-length of each support segment
EXCL_R = 0.050         # stress-exclusion radius around the support-segment ends
U_LIM = 0.0333         # displacement limit
BUCK_MIN = 2.0         # required Pcr/|Nc|

# load cases: (p_left, p_right) in kPa, positive downward, incl. 0.35 kPa panel weight
CASES = {
    "G":   (0.35 + 0.75, 0.35 + 0.75),
    "A":   (0.35 + 0.75, 0.35),
    "A_m": (0.35, 0.35 + 0.75),
    "U":   (0.35 - 1.50, 0.35 - 1.50),
    "W":   (0.35 - 1.50, 0.35 + 0.50),
    "W_m": (0.35 + 0.50, 0.35 - 1.50),
}

# %%
import json, os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpi4py import MPI
from petsc4py import PETSc
import gmsh, basix, ufl, dolfinx
from dolfinx import default_scalar_type, mesh
from dolfinx.io import gmsh as gmshio
from dolfinx.fem import (Constant, Expression, Function, functionspace, form,
                         assemble_scalar, locate_dofs_topological, dirichletbc)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from ufl import (TestFunction, TrialFunction, Measure, FacetNormal, SpatialCoordinate,
                 inner, dot, grad, sym, tr, Identity, as_vector, conditional, gt, lt, ge, le, And)

os.makedirs(OUTDIR, exist_ok=True)
VERSIONS = {"dolfinx": dolfinx.__version__, "basix": basix.__version__, "ufl": ufl.__version__,
            "petsc": ".".join(map(str, PETSc.Sys.getVersion())), "gmsh": gmsh.__version__}
print(VERSIONS)
with open(os.path.join(OUTDIR, "versions.json"), "w") as fh:
    json.dump(VERSIONS, fh, indent=1)

# %% [markdown]
# ## 2. Design description and member graph
# Nodes with exactly two collinear members (e.g. A in a king-post frame without a web at A)
# are subdivisions, not joints: they are merged into one physical member for the buckling
# length and for the pin-jointed truss.

# %%
def load_design(path):
    with open(path) as fh:
        D = json.load(fh)
    sc = 1e-3 if D.get("units", "mm") == "mm" else 1.0
    D["path"] = path
    D["brep"] = os.path.join(os.path.dirname(path), D["brep"])
    D["b"] = float(D.get("b", 76.0)) * sc
    D["nodes"] = {k: np.array(v, float) * sc for k, v in D["nodes"].items()}
    D["members"] = {k: float(v) * sc for k, v in D["members"].items()}
    N = D["nodes"]
    base = [k for k, p in N.items() if abs(p[1]) < 1e-9]
    D["left"] = min(base, key=lambda k: N[k][0])
    D["right"] = max(base, key=lambda k: N[k][0])
    D["ridge"] = max(N, key=lambda k: N[k][1])
    D["chains"] = member_chains(D)
    return D

def ends(m):
    return m.split("-")


def roofline(D, x):
    xl, xr, xR = D["nodes"][D["left"]][0], D["nodes"][D["right"]][0], D["nodes"][D["ridge"]][0]
    H = D["nodes"][D["ridge"]][1]
    return np.where(x <= xR, H * (x - xl) / (xR - xl), H * (xr - x) / (xr - xR))


def member_chains(D):
    """Merge chains of collinear members through degree-2 subdivision nodes."""
    N, M = D["nodes"], D["members"]
    adj = {n: [] for n in N}
    for m in M:
        a, c = ends(m)
        adj[a].append((m, c)); adj[c].append((m, a))

    def unit(a, c):
        v = N[c] - N[a]; return v / np.linalg.norm(v)

    def is_pass(n):
        if n in (D["left"], D["right"]) or len(adj[n]) != 2:
            return False
        (m1, o1), (m2, o2) = adj[n]
        u1, u2 = unit(n, o1), unit(n, o2)
        return abs(u1[0] * u2[1] - u1[1] * u2[0]) < 1e-6

    used, chains = set(), []
    for start in N:
        if is_pass(start):
            continue
        for m, nxt in adj[start]:
            if m in used:
                continue
            path, mems = [start, nxt], [m]
            used.add(m)
            while is_pass(path[-1]):
                (ma, oa), (mb, ob) = adj[path[-1]]
                m2, o2 = (mb, ob) if ma == mems[-1] else (ma, oa)
                path.append(o2); mems.append(m2); used.add(m2)
            chains.append({"name": mems[0] if len(mems) == 1 else f"{path[0]}-{path[-1]}", "nodes": path, "members": mems,
                           "length": float(np.linalg.norm(N[path[-1]] - N[path[0]])),
                           "d_min": min(M[x] for x in mems),
                           "upper": bool(np.all(np.abs(roofline(D, np.array([N[p][0] for p in path]))
                                                      - np.array([N[p][1] for p in path])) < 1e-6))})
    return chains

# %% [markdown]
# ## 3. Geometry import, boundary detection and mesh
# * 3D solid BREP: the flat face at the lowest z is used; 2D BREP: the face itself.
# * mm → m is detected from the bounding box.
# * The face is split by vertical lines at the ridge (and at the support centres if
#   `LOAD_SPAN_ONLY`) so the load halves change exactly at element edges, and by points at
#   the support-segment ends. These internal curves are conforming; they do not change physics.
# * Loaded boundary = boundary curves whose upward ray never re-enters material
#   (exposed top surface); hole boundaries are therefore excluded automatically.

# %%
TAG_SURF, TAG_SUP_L, TAG_SUP_R, TAG_ROOF, TAG_OTHER = 1, 11, 12, 20, 30


def build_mesh(D, scale=1.0, verbose=True):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(D["name"])
    occ = gmsh.model.occ
    occ.importShapes(D["brep"])
    occ.synchronize()

    flat = []
    for _, t in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(2, t)
        if abs(bb[5] - bb[2]) < 1e-6 * max(1.0, abs(bb[3] - bb[0])):
            flat.append((round(bb[2], 6), -occ.getMass(2, t), t))
    if not flat:
        raise RuntimeError("no planar face found in " + D["brep"])
    z0, _, face = sorted(flat)[0]
    keep = occ.copy([(2, face)])
    occ.remove(gmsh.model.getEntities(3), recursive=True)
    occ.remove([e for e in gmsh.model.getEntities(2) if e not in keep], recursive=True)
    occ.synchronize()
    occ.translate(keep, 0, 0, -z0)
    occ.synchronize()
    bb = gmsh.model.getBoundingBox(2, keep[0][1])
    unit = 1e-3 if (bb[3] - bb[0]) > 100 else 1.0     # model unit -> m (mesh scaled after meshing)
    k = 1.0 / unit                                     # m -> model unit
    vy = [gmsh.model.getValue(0, t, [])[1] for _, t in gmsh.model.getEntities(0)]
    ybot, ytop = min(vy), bb[4]
    tol = 1e-6 * (bb[3] - bb[0])

    xs_m = [D["nodes"][D["left"]][0], D["nodes"][D["right"]][0]]
    xs, xR = [k * x for x in xs_m], k * D["nodes"][D["ridge"]][0]
    sh = k * SUP_HALF
    cut_x = [xR] + (xs if LOAD_SPAN_ONLY else [])
    tools = [(1, occ.addLine(occ.addPoint(x, ybot - 0.1 * k, 0), occ.addPoint(x, ytop + 0.1 * k, 0)))
             for x in cut_x]
    tools += [(0, occ.addPoint(x + dx, ybot, 0)) for x in xs for dx in (-sh, sh)]
    occ.fragment(keep, tools)
    occ.synchronize()
    # drop line pieces / points outside the material
    for dim in (1, 0):
        orphan = [(dim, t) for _, t in gmsh.model.getEntities(dim)
                  if len(gmsh.model.getAdjacencies(dim, t)[0]) == 0]
        if orphan:
            occ.remove(orphan, recursive=True)
    occ.synchronize()
    surfs = [t for _, t in gmsh.model.getEntities(2)]
    bnd = sorted({abs(t) for _, t in gmsh.model.getBoundary([(2, t) for t in surfs],
                                                            combined=True, oriented=False)})

    def inside(x, y):
        return any(gmsh.model.isInside(2, t, [x, y, 0]) for t in surfs)

    sup = {TAG_SUP_L: [], TAG_SUP_R: []}
    roof, other, arcs = [], [], []
    for c in bnd:
        cx0, cy0, _, cx1, cy1, _ = gmsh.model.getBoundingBox(1, c)
        if gmsh.model.getType(1, c) == "Circle":
            arcs.append(c)
        on_bot = abs(cy0 - ybot) < 10 * tol and abs(cy1 - ybot) < 10 * tol
        tagged = False
        for tag, xc in zip((TAG_SUP_L, TAG_SUP_R), xs):
            if on_bot and cx0 >= xc - sh - 10 * tol and cx1 <= xc + sh + 10 * tol:
                sup[tag].append(c); tagged = True
        if tagged:
            continue
        t0, t1 = gmsh.model.getParametrizationBounds(1, c)
        xm, ym, _ = gmsh.model.getValue(1, c, [0.5 * (t0[0] + t1[0])])
        ys = np.arange(ym + 0.003 * k, ytop + 0.01 * k, 0.004 * k)
        (other if (on_bot or any(inside(xm, y) for y in ys)) else roof).append(c)

    gmsh.model.addPhysicalGroup(2, surfs, TAG_SURF)
    gmsh.model.addPhysicalGroup(1, sup[TAG_SUP_L], TAG_SUP_L)
    gmsh.model.addPhysicalGroup(1, sup[TAG_SUP_R], TAG_SUP_R)
    gmsh.model.addPhysicalGroup(1, roof, TAG_ROOF)
    gmsh.model.addPhysicalGroup(1, other, TAG_OTHER)

    # size field
    h = scale * min(H_MAX, min(D["members"].values()) / N_ACROSS)
    hj, hs = scale * H_JOINT, scale * H_SUPPORT          # in m; converted with k below
    sup_pts = []
    for x in xs:
        for dx in (-sh, sh):
            sup_pts += [t for _, t in gmsh.model.getEntitiesInBoundingBox(
                x + dx - 10 * tol, ybot - 10 * tol, -1, x + dx + 10 * tol, ybot + 10 * tol, 1, dim=0)]
    assert len(sup_pts) == 4, f"support-end points found: {len(sup_pts)} (expected 4)"
    F = gmsh.model.mesh.field
    fields = []
    for ents, key, hmin, dmin in ((arcs, "CurvesList", hj, hj), (sup_pts, "PointsList", hs, 1.3 * EXCL_R)):
        if not ents:
            continue
        fd = F.add("Distance"); F.setNumbers(fd, key, ents); F.setNumber(fd, "Sampling", 200)
        ft = F.add("Threshold"); F.setNumber(ft, "InField", fd)
        F.setNumber(ft, "SizeMin", k * hmin); F.setNumber(ft, "SizeMax", k * h)
        F.setNumber(ft, "DistMin", k * dmin); F.setNumber(ft, "DistMax", k * (dmin + 0.15))
        fields.append(ft)
    fm = F.add("Min"); F.setNumbers(fm, "FieldsList", fields); F.setAsBackgroundMesh(fm)
    for opt in ("Mesh.MeshSizeFromPoints", "Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromCurvature"):
        gmsh.option.setNumber(opt, 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", k * h)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(ELEMENT_ORDER)
    md = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    md.mesh.geometry.x[:, :2] *= unit                  # model unit -> m
    info = {"unit_scale": unit, "y_bot": ybot * unit, "x_sup": xs_m, "x_ridge": xR * unit, "h": h, "h_joint": hj,
            "h_support": hs, "n_arcs": len(arcs), "n_roof_curves": len(roof)}
    if verbose:
        print(f"[{D['name']}] h={h*1e3:.1f} mm, joint {hj*1e3:.1f} mm, support {hs*1e3:.2f} mm, "
              f"cells {md.mesh.topology.index_map(2).size_global}, arcs refined {len(arcs)}")
    return md.mesh, md.facet_tags, info

# %% [markdown]
# ## 4. Plane-stress model, loads, solve
# $\int_\Omega b\,\varepsilon(v):\sigma(u)\,dA = \int_\Omega b\,f_v\cdot v\,dA + \int_{\Gamma_q} q\cdot v\,ds$,
# with $q=(0,-p\,s\,|t_x|)$. On the exposed upper boundary $|t_x| = n_y$ (outward normal).
# $\sigma = 2\mu\varepsilon + \lambda_{ps}\,\mathrm{tr}(\varepsilon)I$, $\lambda_{ps}=E\nu/(1-\nu^2)$.

# %%
def I(domain, expr):
    return domain.comm.allreduce(assemble_scalar(form(expr)), op=MPI.SUM)


class FrameModel:
    def __init__(self, D, scale=1.0, verbose=True):
        t0 = time.perf_counter()
        self.D = D
        b = D["b"]
        self.domain, self.ft, self.info = build_mesh(D, scale, verbose)
        dom = self.domain
        self.fdim = 1
        dom.topology.create_connectivity(1, 2)
        self.dx = Measure("dx", domain=dom)
        self.ds = Measure("ds", domain=dom, subdomain_data=self.ft)
        n, x = FacetNormal(dom), SpatialCoordinate(dom)
        self.x = x
        self.V = functionspace(dom, ("Lagrange", ELEMENT_ORDER, (2,)))
        self.mu, self.lam = E / (2 * (1 + nu)), E * nu / (1 - nu ** 2)
        u, v = TrialFunction(self.V), TestFunction(self.V)
        self.f_v = Constant(dom, default_scalar_type((0.0, -rho * g)))
        self.pL = Constant(dom, default_scalar_type(0.0))
        self.pR = Constant(dom, default_scalar_type(0.0))
        xs, xR = self.info["x_sup"], self.info["x_ridge"]
        up = conditional(gt(n[1], 0.0), n[1], 0.0)
        span = conditional(And(ge(x[0], xs[0] - 1e-9), le(x[0], xs[1] + 1e-9)), 1.0, 0.0) if LOAD_SPAN_ONLY else 1.0
        left = conditional(lt(x[0], xR), 1.0, 0.0)
        self.proj_w = up * span                       # |t_x| weight on the loaded boundary
        self.left_w = left
        self.q = as_vector((0.0, -(self.pL * left + self.pR * (1 - left)) * s * self.proj_w))
        self.a = b * inner(self.sigma(u), self.eps(v)) * self.dx
        self.L = b * dot(self.f_v, v) * self.dx + dot(self.q, v) * self.ds(TAG_ROOF)

        fl, fr = self.ft.find(TAG_SUP_L), self.ft.find(TAG_SUP_R)
        self.nodes_L = locate_dofs_topological(self.V, 1, fl)
        self.nodes_R = locate_dofs_topological(self.V, 1, fr)
        bcs = [dirichletbc(np.zeros(2, dtype=default_scalar_type), self.nodes_L, self.V),
               dirichletbc(default_scalar_type(0.0), locate_dofs_topological(self.V.sub(1), 1, fr), self.V.sub(1))]
        self.problem = LinearProblem(self.a, self.L, bcs=bcs, petsc_options_prefix=f"frame_{id(self)}_",
                                     petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        self.ndofs = self.V.dofmap.index_map.size_global * 2
        self.area = I(dom, 1.0 * self.dx)
        self.xbar = I(dom, x[0] * self.dx) / self.area
        self.proj = (I(dom, self.proj_w * left * self.ds(TAG_ROOF)),
                     I(dom, self.proj_w * (1 - left) * self.ds(TAG_ROOF)))
        self.len_sup = (I(dom, 1.0 * self.ds(TAG_SUP_L)), I(dom, 1.0 * self.ds(TAG_SUP_R)))
        # stress sampling space (unaveraged, discontinuous)
        self.W0 = functionspace(dom, ("DG", ELEMENT_ORDER))
        self.WT = functionspace(dom, ("DG", ELEMENT_ORDER, (2, 2)))
        xy = self.W0.tabulate_dof_coordinates()[:, :2]
        yb = self.info["y_bot"]
        self.excl_pts = np.array([[x0 + dx, yb] for x0 in xs for dx in (-SUP_HALF, SUP_HALF)])
        self.excluded = np.min(np.linalg.norm(xy[:, None, :] - self.excl_pts[None], axis=2), axis=1) < EXCL_R
        self.xy0 = xy
        self.tree = bb_tree(dom, 2)
        self.t_setup = time.perf_counter() - t0

    def eps(self, u):
        return sym(grad(u))

    def sigma(self, u):
        return self.lam * tr(self.eps(u)) * Identity(2) + 2 * self.mu * self.eps(u)

    def solve(self, pl_kpa, pr_kpa):
        self.pL.value, self.pR.value = pl_kpa * 1e3, pr_kpa * 1e3
        t0 = time.perf_counter()
        out = self.problem.solve()
        self.t_solve = time.perf_counter() - t0
        uh = out[0] if isinstance(out, (tuple, list)) else out
        self.uh = uh
        return uh

    # --- reactions from the residual action on a unit field supported on the support dofs
    def reaction(self, uh, nodes, comp):
        w = Function(self.V)
        w.x.array[2 * nodes + comp] = 1.0
        L_w = ufl.replace(self.L, {self.L.arguments()[0]: w})
        return I(self.domain, self.D["b"] * inner(self.sigma(uh), self.eps(w)) * self.dx) - I(self.domain, L_w)

    def roof_resultants(self):
        qy = self.q[1]
        Fy = I(self.domain, qy * self.ds(TAG_ROOF))
        My = I(self.domain, self.x[0] * qy * self.ds(TAG_ROOF))
        return Fy, My            # N (positive up), N*m about x = 0

    # --- point evaluation
    def cells_at(self, pts):
        p3 = np.zeros((len(pts), 3)); p3[:, :2] = pts
        coll = compute_colliding_cells(self.domain, compute_collisions_points(self.tree, p3), p3)
        cells = np.array([coll.links(i)[0] if len(coll.links(i)) else -1 for i in range(len(pts))], np.int32)
        return p3, cells

    def stress_tensor(self, uh):
        S = Function(self.WT)
        S.interpolate(Expression(self.sigma(uh), self.WT.element.interpolation_points))
        return S

# %% [markdown]
# ## 5. Stress, displacement and section checks

# %%
GP, GW = np.polynomial.legendre.leggauss(24)


def section_resultants(Mdl, S, chain):
    """Axial force N and bending moment M on clean cuts across each sub-member.
    A cut is 'clean' if the strip edges +/- (d/2 + 4 mm) lie outside the material
    (i.e. the cut is not inside a joint or fillet region)."""
    D, out = Mdl.D, []
    b = Mdl.D["b"]
    for m in chain["members"]:
        a, c = ends(m)
        pa, pc = D["nodes"][a], D["nodes"][c]
        d = D["members"][m]
        L = np.linalg.norm(pc - pa); t = (pc - pa) / L; nrm = np.array([-t[1], t[0]])
        for f in np.linspace(0.05, 0.95, 37):
            p0 = pa + f * L * t
            edge = np.array([p0 + (d / 2 + 0.004) * nrm, p0 - (d / 2 + 0.004) * nrm])
            _, ce = Mdl.cells_at(edge)
            if np.any(ce >= 0):
                continue
            eta = 0.5 * d * GP
            pts = p0[None] + eta[:, None] * nrm[None]
            p3, cells = Mdl.cells_at(pts)
            if np.any(cells < 0):
                continue
            sig = S.eval(p3, cells).reshape(-1, 2, 2)
            stt = np.einsum("i,nij,j->n", t, sig, t)
            N = b * 0.5 * d * np.sum(GW * stt)
            Mb = b * 0.5 * d * np.sum(GW * stt * eta)
            s_along = np.linalg.norm(p0 - D["nodes"][chain["nodes"][0]])
            out.append((m, f, s_along, N, Mb))
    return pd.DataFrame(out, columns=["member", "frac", "s", "N", "M"])


def check_case(Mdl, uh):
    S = Mdl.stress_tensor(uh)
    sig = Function(Mdl.W0)
    comps = []
    for i, j in ((0, 0), (1, 1), (0, 1)):
        sig.interpolate(Expression(Mdl.sigma(uh)[i, j], Mdl.W0.element.interpolation_points))
        comps.append(sig.x.array.copy())
    sxx, syy, sxy = comps
    c, R = (sxx + syy) / 2, np.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2)
    z = np.zeros_like(c)
    smax = np.maximum.reduce([z, c + R, c - R]); smin = np.minimum.reduce([z, c + R, c - R])
    ratios = {"tension": smax / f_t, "compression": -smin / f_c, "shear": (smax - smin) / 2 / f_s}
    keep = ~Mdl.excluded
    u = uh.x.array.reshape(-1, 2)
    gu = Function(Mdl.W0)
    gu.interpolate(Expression(ufl.sqrt(inner(grad(uh), grad(uh))), Mdl.W0.element.interpolation_points))
    row = {"max_ux_mm": 1e3 * np.abs(u[:, 0]).max(), "max_uy_mm": 1e3 * np.abs(u[:, 1]).max(),
           "max_grad_u": gu.x.array.max()}
    for k, r in ratios.items():
        i = np.argmax(np.where(keep, r, -np.inf))
        row[k] = r[i]; row[k + "_x"], row[k + "_y"] = Mdl.xy0[i]
        row[k + "_excl"] = r[Mdl.excluded].max() if Mdl.excluded.any() else np.nan
    return row, S, ratios

# %% [markdown]
# ## 6. Matching pin-jointed truss (screening model)
# Real joints only (subdivision nodes merged, series axial stiffness). Roof panel resultants
# go half to each end node; self-weight $\rho b d \ell g$ is lumped the same way, then scaled
# by $w_i(a + c x_i)$ so the total weight and first moment equal the continuum frame's.

# %%
def truss_model(D, Mdl, pl, pr):
    b = D["b"]
    N, ch = D["nodes"], D["chains"]
    joints = sorted({n for c in ch for n in (c["nodes"][0], c["nodes"][-1])})
    idx = {n: i for i, n in enumerate(joints)}
    nd = 2 * len(joints)

    def lump(F, n, fy):            # lever-rule transfer of a load on a subdivision node
        if n in idx:
            F[2 * idx[n] + 1] += fy; return
        for c in ch:
            if n in c["nodes"]:
                a, e = N[c["nodes"][0]], N[c["nodes"][-1]]
                lam = np.linalg.norm(N[n] - a) / np.linalg.norm(e - a)
                F[2 * idx[c["nodes"][0]] + 1] += (1 - lam) * fy
                F[2 * idx[c["nodes"][-1]] + 1] += lam * fy
                return

    # roof
    Froof = np.zeros(nd)
    xR = N[D["ridge"]][0]
    for c in ch:
        if not c["upper"]:
            continue
        for m in c["members"]:
            a, e = ends(m)
            dxp = abs(N[e][0] - N[a][0]); xm = 0.5 * (N[e][0] + N[a][0])
            p = (pl if xm < xR else pr) * 1e3
            for n in (a, e):
                lump(Froof, n, -0.5 * p * s * dxp)
    # self-weight, then match continuum total weight and first moment
    w = np.zeros(nd)
    for m, d in D["members"].items():
        a, e = ends(m)
        Wm = rho * g * b * d * np.linalg.norm(N[e] - N[a])
        for n in (a, e):
            lump(w, n, 0.5 * Wm)
    wy, xj = w[1::2], np.array([N[n][0] for n in joints])
    W, Mx = rho * g * b * Mdl.area, rho * g * b * Mdl.area * Mdl.xbar
    A = np.array([[wy.sum(), (wy * xj).sum()], [(wy * xj).sum(), (wy * xj ** 2).sum()]])
    aa, cc = np.linalg.solve(A, [W, Mx])
    Fsw = np.zeros(nd); Fsw[1::2] = -wy * (aa + cc * xj)
    F = Froof + Fsw

    K = np.zeros((nd, nd)); els = []
    for c in ch:
        a, e = c["nodes"][0], c["nodes"][-1]
        L = np.linalg.norm(N[e] - N[a]); t = (N[e] - N[a]) / L
        k = 1.0 / sum(np.linalg.norm(N[ends(m)[1]] - N[ends(m)[0]]) / (E * b * D["members"][m]) for m in c["members"])
        dof = [2 * idx[a], 2 * idx[a] + 1, 2 * idx[e], 2 * idx[e] + 1]
        B = np.array([-t[0], -t[1], t[0], t[1]])
        K[np.ix_(dof, dof)] += k * np.outer(B, B)
        els.append((c["name"], dof, B, k))
    fixed = [2 * idx[D["left"]], 2 * idx[D["left"]] + 1, 2 * idx[D["right"]] + 1]
    free = [i for i in range(nd) if i not in fixed]
    Kr = K[np.ix_(free, free)]
    ok = np.linalg.matrix_rank(Kr) == len(free) and np.linalg.cond(Kr) < 1e12
    res = {"applicable": bool(ok), "F_total": F[1::2].sum(),
           "M_total": (F[1::2] * xj).sum(), "roof_total": Froof[1::2].sum(), "roof_M": (Froof[1::2] * xj).sum()}
    if not ok:
        res["note"] = "truss is a mechanism: approximation inapplicable"
        return res
    u = np.zeros(nd); u[free] = np.linalg.solve(Kr, F[free])
    R = K @ u - F
    res.update({"N": {nm: k * B @ u[dof] for nm, dof, B, k in els},
                "Ry_L": R[2 * idx[D["left"]] + 1], "Ry_R": R[2 * idx[D["right"]] + 1],
                "Rx_L": R[2 * idx[D["left"]]],
                "max_ux_mm": 1e3 * np.abs(u[0::2]).max(), "max_uy_mm": 1e3 * np.abs(u[1::2]).max(),
                "ndofs": len(free)})
    return res

# %% [markdown]
# ## 7. Full analysis of one design

# %%


def analyse(path, scale=1.0, cases=CASES, sections=True, verbose=True):
    D = load_design(path)
    Mdl = FrameModel(D, scale, verbose)
    dom = Mdl.domain
    b = D["b"]
    V = b * Mdl.area
    if verbose:
        cl = sum(b * d * np.linalg.norm(D["nodes"][ends(m)[1]] - D["nodes"][ends(m)[0]]) for m, d in D["members"].items())
        print(f"  area {Mdl.area:.5f} m^2, V = {V:.5f} m^3/frame (centreline sum {cl:.4f}), mass {rho*V:.1f} kg")
        print(f"  supports {Mdl.len_sup[0]:.4f}, {Mdl.len_sup[1]:.4f} m; loaded projection "
              f"{Mdl.proj[0]:.4f} + {Mdl.proj[1]:.4f} m; DOFs {Mdl.ndofs}")
        print("  real members:", ", ".join(f"{c['name']} (L={c['length']:.3f} m)" for c in D["chains"]))
    # every node and member midpoint must be inside the meshed material
    pts = np.array(list(D["nodes"].values()) +
                   [0.5 * (D["nodes"][ends(m)[0]] + D["nodes"][ends(m)[1]]) for m in D["members"]])
    assert np.all(Mdl.cells_at(pts)[1] >= 0), "design JSON does not match the BREP (node/member outside material)"
    assert abs(sum(Mdl.len_sup) - 4 * SUP_HALF) < 1e-6, "support segments not found correctly"
    span_target = (D["nodes"][D["right"]][0] - D["nodes"][D["left"]][0]) if LOAD_SPAN_ONLY else None
    if span_target is not None:
        assert abs(sum(Mdl.proj) - span_target) < 1e-6, f"loaded projection {sum(Mdl.proj)} != {span_target}"

    rows, members, stash = [], [], {}
    for case, (pl, pr) in cases.items():
        uh = Mdl.solve(pl, pr)
        Fy_roof, M_roof = Mdl.roof_resultants()
        W_sw = rho * g * b * Mdl.area
        Ry_L, Ry_R = Mdl.reaction(uh, Mdl.nodes_L, 1), Mdl.reaction(uh, Mdl.nodes_R, 1)
        Rx_L = Mdl.reaction(uh, Mdl.nodes_L, 0)
        U = 0.5 * I(dom, b * inner(Mdl.sigma(uh), Mdl.eps(uh)) * Mdl.dx)
        L_u = ufl.replace(Mdl.L, {Mdl.L.arguments()[0]: uh})
        fu = I(dom, L_u)
        row, S, ratios = check_case(Mdl, uh)
        load_down = W_sw - Fy_roof
        row.update({"case": case, "load_down_kN": load_down / 1e3,
                    "Ry_L_kN": Ry_L / 1e3, "Ry_R_kN": Ry_R / 1e3, "Rx_L_N": Rx_L,
                    "equil_err_N": abs(Ry_L + Ry_R - load_down),
                    "U_J": U, "energy_rel_err": abs(2 * U - fu) / abs(fu),
                    "attach_L_kN": max(0.0, -Ry_L / 1e3), "attach_R_kN": max(0.0, -Ry_R / 1e3),
                    "solve_s": Mdl.t_solve})
        T = truss_model(D, Mdl, pl, pr)
        row["truss_applicable"] = T["applicable"]
        row["load_match_N"] = abs(-T["F_total"] - load_down)
        row["moment_match_Nm"] = abs(-T["M_total"] - (W_sw * Mdl.xbar - M_roof))
        if T["applicable"]:
            row.update({"truss_max_uy_mm": T["max_uy_mm"], "truss_Ry_L_kN": T["Ry_L"] / 1e3,
                        "truss_Ry_R_kN": T["Ry_R"] / 1e3})
        # sections: axial force, bending, buckling
        buck_min = np.inf
        if sections:
            for c in D["chains"]:
                sec = section_resultants(Mdl, S, c)
                Imin = min(b * c["d_min"] ** 3, c["d_min"] * b ** 3) / 12
                Pcr = np.pi ** 2 * E * Imin / c["length"] ** 2
                if len(sec):
                    mid = sec.iloc[np.argmin(np.abs(sec["s"] - c["length"] / 2))]
                    Nc = sec["N"].min()
                    br = Pcr / abs(Nc) if Nc < 0 else np.inf
                    rec = {"case": case, "member": c["name"], "L_m": c["length"], "d_min_mm": 1e3 * c["d_min"],
                           "n_cuts": len(sec), "N_mid_kN": mid["N"] / 1e3, "N_min_kN": Nc / 1e3,
                           "N_max_kN": sec["N"].max() / 1e3, "M_absmax_kNm": sec["M"].abs().max() / 1e3,
                           "Pcr_kN": Pcr / 1e3, "buckling_ratio": br}
                else:
                    rec = {"case": case, "member": c["name"], "L_m": c["length"], "n_cuts": 0,
                           "buckling_ratio": np.nan, "Pcr_kN": Pcr / 1e3}
                    warnings.warn(f"{D['name']} {c['name']}: no clean cross-section (member inside joints)")
                rec["truss_N_kN"] = T["N"][c["name"]] / 1e3 if T["applicable"] else np.nan
                members.append(rec)
                if np.isfinite(rec.get("buckling_ratio", np.nan)):
                    buck_min = min(buck_min, rec["buckling_ratio"])
        row["buckling_min"] = buck_min
        rows.append(row)
        stash[case] = (uh.copy(), ratios)
    df = pd.DataFrame(rows).set_index("case")
    dm = pd.DataFrame(members)
    return D, Mdl, df, dm, stash


def summarise(D, Mdl, df):
    b = D["b"]
    util = {"tension": df["tension"], "compression": df["compression"], "shear": df["shear"],
            "disp": np.maximum(df["max_ux_mm"], df["max_uy_mm"]) / (1e3 * U_LIM),
            "buckling": BUCK_MIN / df["buckling_min"]}
    out = {"design": D["name"], "V_frame_m3": b * Mdl.area, "V_roof_21_m3": 21 * b * Mdl.area,
           "mass_frame_kg": rho * b * Mdl.area, "mass_roof_21_kg": 21 * rho * b * Mdl.area,
           "dofs": Mdl.ndofs, "max_u_mm": max(df["max_ux_mm"].max(), df["max_uy_mm"].max())}
    for k, u in util.items():
        out[k + "_util"] = float(u.max()); out[k + "_case"] = u.idxmax()
    out["buckling_min_ratio"] = float(df["buckling_min"].min())
    out["attach_force_kN"] = float(max(df["attach_L_kN"].max(), df["attach_R_kN"].max()))
    out["feasible"] = bool(all(out[k + "_util"] <= 1.0 for k in util))
    out["governing"] = max(util, key=lambda k: out[k + "_util"])
    return out

# %% [markdown]
# ## 8. Plots

# %%
def tri_of(Vspace):
    """Linear triangulation on the vertex DOFs of a (P2 or DG) space for matplotlib."""
    dm = Vspace.dofmap
    nc = Vspace.mesh.topology.index_map(2).size_local
    cells = np.array([dm.cell_dofs(c)[:3] for c in range(nc)])
    xy = Vspace.tabulate_dof_coordinates()[:, :2]
    return xy, cells


def plot_design(D, Mdl, df, stash, case=None):
    case = case or df["shear"].idxmax()
    uh, ratios = stash[case]
    # mesh + tags
    dom = Mdl.domain
    xg = dom.geometry.x[:, :2]
    cg = dom.geometry.dofmap[:, :3]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.triplot(Triangulation(xg[:, 0], xg[:, 1], cg), lw=0.15, color="0.6")
    for tag, col, lab in ((TAG_ROOF, "tab:blue", "loaded boundary"), (TAG_SUP_L, "tab:red", "supports"),
                          (TAG_SUP_R, "tab:red", None)):
        f = Mdl.ft.find(tag)
        first = True
        for fi in f:
            pv = dolfinx.mesh.entities_to_geometry(dom, 1, np.array([fi], np.int32))[0][:2]
            ax.plot(xg[pv, 0], xg[pv, 1], color=col, lw=2, label=lab if (first and lab) else None)
            first = False
    for p in Mdl.excl_pts:
        ax.add_patch(plt.Circle(p, EXCL_R, fill=False, color="k", lw=1))
    for n, p in D["nodes"].items():
        ax.annotate(n, p, fontsize=8, ha="center", va="center", color="darkgreen")
    for m, d in D["members"].items():
        pm = 0.5 * (D["nodes"][ends(m)[0]] + D["nodes"][ends(m)[1]])
        ax.annotate(f"{d*1e3:.0f}", pm, fontsize=7, color="purple", ha="center")
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"{D['name']}: mesh ({Mdl.ndofs} DOFs), widths [mm], support exclusion zones")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, f"{D['name']}_mesh.png"), dpi=200); plt.close(fig)

    # deformed shape + |u|
    xy, cells = tri_of(Mdl.V)
    u = uh.x.array.reshape(-1, 2)
    mag = 1e3 * np.linalg.norm(u, axis=1)
    k = 0.15 / max(np.abs(u).max(), 1e-12)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.triplot(Triangulation(xy[:, 0], xy[:, 1], cells), lw=0.1, color="0.8")
    tc = ax.tripcolor(Triangulation(xy[:, 0] + k * u[:, 0], xy[:, 1] + k * u[:, 1], cells), mag,
                      shading="gouraud", cmap="viridis")
    fig.colorbar(tc, ax=ax, label="|u| [mm]", shrink=0.8)
    ax.set_aspect("equal"); ax.set_title(f"{D['name']} case {case}: deformed shape x{k:.0f}")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, f"{D['name']}_{case}_deformed.png"), dpi=200); plt.close(fig)

    # stress ratio maps (cell-wise max of the unaveraged DG field)
    W0 = Mdl.W0
    dm = W0.dofmap
    nc = dom.topology.index_map(2).size_local
    cd = np.array([dm.cell_dofs(c) for c in range(nc)])
    for name in ("shear", "tension", "compression"):
        r = ratios[name][cd].max(axis=1)
        fig, ax = plt.subplots(figsize=(11, 3.2))
        tp = ax.tripcolor(Triangulation(xg[:, 0], xg[:, 1], cg), facecolors=r, cmap="turbo", vmin=0, vmax=1)
        tp.cmap.set_over("magenta")
        fig.colorbar(tp, ax=ax, label=f"{name} utilisation [-]", shrink=0.8, extend="max")
        for p in Mdl.excl_pts:
            ax.add_patch(plt.Circle(p, EXCL_R, fill=False, color="k", lw=1))
        rw = df.loc[case]
        ax.plot(rw[name + "_x"], rw[name + "_y"], "kx", ms=8)
        ax.set_aspect("equal")
        ax.set_title(f"{D['name']} case {case}: {name} utilisation (max {rw[name]:.2f} at x, magenta > 1)")
        fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, f"{D['name']}_{case}_{name}.png"), dpi=200); plt.close(fig)
    return case

# %% [markdown]
# ## 9. Run all designs

# %%
import hashlib, pickle

ANALYSIS_VERSION = "v2"   # bump this whenever you edit the analysis code in sections 2-7
FORCE_RERUN = []          # e.g. ["Design_4"] to recompute specific designs, or True for all
CACHE_DIR = os.path.join(OUTDIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

SETTINGS = dict(ELEMENT_ORDER=ELEMENT_ORDER, H_MAX=H_MAX, N_ACROSS=N_ACROSS, H_JOINT=H_JOINT,
                H_SUPPORT=H_SUPPORT, LOAD_SPAN_ONLY=LOAD_SPAN_ONLY, s=s, E=E, nu=nu, rho=rho, g=g,
                f_t=f_t, f_c=f_c, f_s=f_s, SUP_HALF=SUP_HALF, EXCL_R=EXCL_R, U_LIM=U_LIM,
                BUCK_MIN=BUCK_MIN, CASES=CASES, version=ANALYSIS_VERSION)


def design_key(path):
    """Fingerprint of the JSON, its BREP and all settings: changes when any of them changes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        js = fh.read()
    h.update(js)
    with open(os.path.join(os.path.dirname(path), json.loads(js)["brep"]), "rb") as fh:
        h.update(fh.read())
    h.update(json.dumps(SETTINGS, sort_keys=True, default=str).encode())
    return h.hexdigest()


def run_design(path):
    with open(path) as fh:
        name = json.load(fh)["name"]
    key, fn = design_key(path), os.path.join(CACHE_DIR, f"{name}.pkl")
    force = FORCE_RERUN is True or name in FORCE_RERUN
    if not force and os.path.exists(fn):
        with open(fn, "rb") as fh:
            C = pickle.load(fh)
        if C["key"] == key:
            print(f"[{name}] loaded from cache")
            return C
    D, Mdl, df, dm, stash = analyse(path)
    S = summarise(D, Mdl, df)
    S["b_mm"] = 1e3 * D["b"]
    S["s1_max_MPa"] = S["tension_util"] * f_t / 1e6
    plot_design(D, Mdl, df, stash)                      # PNGs are written once, here
    df.to_csv(os.path.join(OUTDIR, f"{name}_cases.csv"))
    dm.to_csv(os.path.join(OUTDIR, f"{name}_members.csv"), index=False)
    C = {"key": key, "D": D, "df": df, "dm": dm, "summary": S}
    with open(fn, "wb") as fh:
        pickle.dump(C, fh)
    return C


rows = []
for name, df in all_cases.items():
    D = DESIGN_INFO[name]
    yb = build_mesh(D, verbose=False)[2]["y_bot"]
    xs = np.array([D["nodes"][D["left"]][0], D["nodes"][D["right"]][0]])
    for c, r in df.iterrows():
        j = np.argmin(np.abs(xs - r.shear_x))                       # nearest support
        e = np.array([[xs[j] - SUP_HALF, yb], [xs[j] + SUP_HALF, yb]])
        R = abs(r.Ry_L_kN if j == 0 else r.Ry_R_kN) * 1e3
        rows.append({"design": name, "case": c, "shear": r.shear,
                     "dx_from_centre_mm": 1e3 * (r.shear_x - xs[j]),
                     "height_above_face_mm": 1e3 * (r.shear_y - yb),
                     "dist_to_end_mm": 1e3 * np.min(np.linalg.norm(e - [r.shear_x, r.shear_y], axis=1)),
                     "bearing_floor": R / (2 * SUP_HALF * D["b"]) / 2 / f_s})
diag = pd.DataFrame(rows)
diag.sort_values("shear", ascending=False).groupby("design").head(2).round(2)

def diagnose_support_failure(Mdl, uh, case_name):
    """Inspects the exact stress state and distance to support singularities for peak shear/compression."""
    _, _, ratios = check_case(Mdl, uh)
    
    # Extract raw stress components at DG2 nodes
    comps = {}
    for name, idx in (("sxx", (0, 0)), ("syy", (1, 1)), ("sxy", (0, 1))):
        f = Function(Mdl.W0)
        f.interpolate(Expression(Mdl.sigma(uh)[idx[0], idx[1]], Mdl.W0.element.interpolation_points))
        comps[name] = f.x.array.copy()

    keep = ~Mdl.excluded
    yb = Mdl.info["y_bot"]
    xs = Mdl.info["x_sup"]

    print(f"\n--- Support Failure Diagnostics [{Mdl.D['name']} | Case {case_name}] ---")
    for mode in ("shear", "compression"):
        r_masked = np.where(keep, ratios[mode], -np.inf)
        i = np.argmax(r_masked)
        px, py = Mdl.xy0[i]
        
        # Distance to nearest support center and nearest support endpoint
        dist_center = min(np.hypot(px - x0, py - yb) for x0 in xs)
        dist_end = np.min(np.linalg.norm(Mdl.excl_pts - np.array([px, py]), axis=1))
        side = "LEFT (pinned ux=uy=0)" if abs(px - xs[0]) < abs(px - xs[1]) else "RIGHT (roller uy=0)"

        print(f"Peak {mode.upper()} util = {ratios[mode][i]:.3f} at (x={px:.4f} m, y={py:.4f} m) [{side}]")
        print(f"  -> Height above bottom face: {(py - yb)*1e3:.1f} mm | Dist to support end: {dist_end*1e3:.1f} mm (EXCL_R={EXCL_R*1e3:.1f} mm)")
        print(f"  -> Stress tensor [MPa]: sxx = {comps['sxx'][i]/1e6:.2f}, syy = {comps['syy'][i]/1e6:.2f}, sxy = {comps['sxy'][i]/1e6:.2f}")
        
        # Check if Poisson clamping (sxx) or pure vertical bearing (syy) is causing Mohr's circle to blow up
        if abs(comps['sxy'][i]) > 0.7 * abs(comps['syy'][i]):
            print("  -> CAUSE: High shear sxy (corner singularity / Poisson horizontal restraint on pinned support).")
        elif abs(comps['syy'][i]) > 2 * abs(comps['sxx'][i]):
            print("  -> CAUSE: Pure vertical bearing stress syy dominating Mohr's circle (tau_max ~ |syy|/2).")
        else:
            print("  -> CAUSE: Biaxial compression/tension jump near support-to-web transition.")

# %% [markdown]
# ## 10. Frame vs truss: member axial forces (case G)

# %%
for name, dm in all_members.items():
    print(name)
    print(dm[dm.case == "G"][["member", "L_m", "n_cuts", "N_mid_kN", "N_min_kN", "N_max_kN",
                              "truss_N_kN", "M_absmax_kNm", "Pcr_kN", "buckling_ratio"]].round(3).to_string(index=False))

# %% [markdown]
# ## 11. Mesh convergence
# All mesh sizes are multiplied by each scale. Target: < 2 % change between the last two meshes.

# %%
conv = []
for path in CONVERGENCE_DESIGNS:
    name = load_design(path)["name"]
    gov = summary_df.loc[name]
    for sc in CONV_SCALES:
        D, Mdl, df, dm, _ = analyse(path, scale=sc, sections=False, verbose=False)
        conv.append({"design": name, "scale": sc, "dofs": Mdl.ndofs, "h_mm": 1e3 * Mdl.info["h"],
                     "max_u_mm": max(df["max_ux_mm"].max(), df["max_uy_mm"].max()),
                     "U_G_J": df.loc["G", "U_J"],
                     "shear_max": df["shear"].max(), "tension_max": df["tension"].max(),
                     "compression_max": df["compression"].max(),
                     "shear_excl_max": df["shear_excl"].max()})
        print(conv[-1])
conv_df = pd.DataFrame(conv)
if len(conv_df):
    for k in ("max_u_mm", "U_G_J", "shear_max", "tension_max", "compression_max", "shear_excl_max"):
        conv_df[k + "_chg%"] = conv_df.groupby("design")[k].pct_change() * 100
    conv_df.to_csv(os.path.join(OUTDIR, "convergence.csv"), index=False)
conv_df.round(4)

# %%
from matplotlib.transforms import Bbox

DODGE_MM = 2.5   # sideways spread (mm on the x-axis) for designs sharing the same b


def place_labels(ax, xs, ys, labels, fontsize=8.5):
    """Greedy label placement: try offsets at growing radii, keep the first that overlaps
    no other label or marker and stays inside the axes."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ax_box = ax.get_window_extent(r)
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    taken = [Bbox.from_extents(px - 7, py - 7, px + 7, py + 7) for px, py in pts]
    cands = [(rad * np.cos(np.radians(a)), rad * np.sin(np.radians(a)))
             for rad in (14, 34, 56, 80, 108, 140)
             for a in (20, 160, -20, -160, 45, 135, -45, -135, 0, 180)]
    box = dict(boxstyle="round,pad=0.25", fc="white", ec="lightgray", alpha=0.9)

    def overlap(bb):
        return sum(max(0, min(bb.x1, t.x1) - max(bb.x0, t.x0)) *
                   max(0, min(bb.y1, t.y1) - max(bb.y0, t.y0)) for t in taken)

    for i in np.argsort(-ys):
        best = None
        for dx, dy in cands:
            ha = "left" if dx > 1 else "right" if dx < -1 else "center"
            va = "bottom" if dy > 1 else "top" if dy < -1 else "center"
            ann = ax.annotate(labels[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points",
                              ha=ha, va=va, fontsize=fontsize, fontweight="bold", bbox=box)
            bb = ann.get_window_extent(r).expanded(1.08, 1.15)
            inside = (ax_box.x0 <= bb.x0 and bb.x1 <= ax_box.x1 and
                      ax_box.y0 <= bb.y0 and bb.y1 <= ax_box.y1)
            score = overlap(bb) + (0 if inside else 1e9)
            ann.remove()
            if best is None or score < best[0]:
                best = (score, dx, dy, ha, va, bb)
            if score == 0:
                break
        _, dx, dy, ha, va, bb = best
        ax.annotate(labels[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=fontsize, fontweight="bold", bbox=box, zorder=5,
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color="0.45", lw=0.7, shrinkA=0, shrinkB=4))
        taken.append(bb)


def plot_vs_b(sdf, col, limit, unit, ylabel, title, fname):
    names = list(sdf.index)
    bx = sdf["b_mm"].to_numpy(float)
    y = sdf[col].to_numpy(float)
    bp = bx.copy()                                     # dodge markers that share the same b
    for bv in np.unique(bx):
        idx = np.where(bx == bv)[0][np.argsort(y[bx == bv])]
        if len(idx) > 1:
            bp[idx] = bv + DODGE_MM * (np.arange(len(idx)) - (len(idx) - 1) / 2)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(bp, y, c=["#2b7bba" if v <= limit else "#d9534f" for v in y],
               s=90, edgecolors="black", linewidth=1.0, zorder=4)
    ax.axhline(limit, color="#d9534f", ls="--", lw=1.5, label=f"Limit ({limit:.1f} {unit})", zorder=2)
    ax.set_xlabel(r"Frame thickness $b$ (mm)", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    span = bx.max() - bx.min() if bx.max() > bx.min() else 40.0
    ax.set_xlim(bx.min() - 0.6 * span, bx.max() + 0.6 * span)
    ax.set_ylim(0, max(y.max(), limit) * 1.45)
    ax.set_xticks(sorted(set(bx)))
    ax.grid(True, ls=":", alpha=0.6, zorder=1)
    ax.legend(loc="upper right")
    fig.tight_layout()
    place_labels(ax, bp, y, [f"{n}\n({v:.2f} {unit})" for n, v in zip(names, y)])
    fig.savefig(os.path.join(OUTDIR, fname), dpi=200, bbox_inches="tight")
    plt.show()


plot_vs_b(summary_df, "max_u_mm", 1e3 * U_LIM, "mm",
          r"Maximum displacement $u_{\max}$ (mm)",
          r"Maximum displacement vs. frame thickness $b$", "disp_vs_b.png")
plot_vs_b(summary_df, "s1_max_MPa", f_t / 1e6, "MPa",
          r"Max principal stress $\sigma_1$ (MPa)",
          r"Maximum principal stress vs. frame thickness $b$", "stress_vs_b.png")


