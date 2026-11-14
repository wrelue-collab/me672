"""
mesh_truss.py  -  gmsh mesh of the ME672 A3 timber frame for DOLFINx

    python3 mesh_truss.py truss_profile_2d.brep            # default size
    python3 mesh_truss.py truss_profile_2d.brep --n 8      # finer (8 elements across d_min)
    python3 mesh_truss.py truss_profile_2d.brep --order 1  # linear triangles

Input : 2D BREP written by rebuild_truss.py (millimetres)
Output: truss.msh in METRES (SI: m, N, Pa) with these physical groups

    surface 1  "frame"
    curve  11  "support_left"    100 mm under B0  -> ux = uy = 0
    curve  12  "support_right"   100 mm under B2  -> uy = 0
    curve  21  "roof_left"       exposed upper boundary, x < 4 m
    curve  22  "roof_right"      exposed upper boundary, x > 4 m
    curve  30  "other"           every other boundary edge (traction free)
"""
import argparse, math, sys
import gmsh

ap = argparse.ArgumentParser()
ap.add_argument("brep")
ap.add_argument("-o", "--out", default="truss.msh")
ap.add_argument("--n", type=float, default=5, help="elements across the narrowest member")
ap.add_argument("--dmin", type=float, default=286.0, help="narrowest member width, mm")
ap.add_argument("--refine", type=float, default=3.0, help="size divisor near fillets/supports")
ap.add_argument("--order", type=int, default=2, help="1 = linear, 2 = quadratic triangles")
ap.add_argument("--span", type=float, default=8000.0)
ap.add_argument("--ext", type=float, default=50.0, help="support half-length / chord extension, mm")
ap.add_argument("--gui", action="store_true")
a = ap.parse_args()

TOL = 1e-3          # mm
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("truss")
occ = gmsh.model.occ

ents = occ.importShapes(a.brep)
occ.synchronize()
surfs = [t for d, t in ents if d == 2] or [t for d, t in gmsh.model.getEntities(2)]
assert len(surfs) == 1, f"expected one surface, got {surfs}"
s = surfs[0]

# ---- split boundary: support ends at x = +-ext, and the ridge at x = span/2
xmin, ybot, _, xmax, ytop, _ = gmsh.model.getBoundingBox(2, s)
cuts = [(a.ext, ybot), (a.span - a.ext, ybot), (a.span / 2, ytop)]
pts = [occ.addPoint(x, y, 0) for x, y in cuts]
occ.fragment([(2, s)], [(0, p) for p in pts])
occ.synchronize()
s = gmsh.model.getEntities(2)[0][1]
for d, t in gmsh.model.getEntities(0):           # drop any point not on the boundary
    if not gmsh.model.getAdjacencies(0, t)[0].size:
        occ.remove([(0, t)])
occ.synchronize()

# ---- classify boundary curves (in mm, before scaling)
H_SPAN = a.span / 2
def centreline_y(x):                    # straight upper centreline B0-R-B2 (ridge 4/3 m)
    return (4000.0 / 3) * (1 - abs(x - H_SPAN) / H_SPAN)

groups = {"support_left": [], "support_right": [], "roof_left": [], "roof_right": [], "other": []}
for d, c in gmsh.model.getBoundary([(2, s)], oriented=False):
    c = abs(c)
    x, y, _ = occ.getCenterOfMass(1, c)
    L = occ.getMass(1, c)
    on_bottom = abs(y - ybot) < TOL
    if on_bottom and abs(x - 0.0) < TOL and abs(L - 2 * a.ext) < 1e-2:
        groups["support_left"].append(c)
    elif on_bottom and abs(x - a.span) < TOL and abs(L - 2 * a.ext) < 1e-2:
        groups["support_right"].append(c)
    elif y > centreline_y(min(max(x, 0), a.span)) + 0.25 * a.dmin:   # outer upper edge
        groups["roof_left" if x < H_SPAN else "roof_right"].append(c)
    else:
        groups["other"].append(c)

for k in ("support_left", "support_right", "roof_left", "roof_right"):
    if not groups[k]:
        sys.exit(f"ERROR: no curves found for {k}; check the BREP")

# ---- scale mm -> m (tags are kept)
occ.dilate(gmsh.model.getEntities(), 0, 0, 0, 1e-3, 1e-3, 1e-3)
occ.synchronize()

# ---- mesh size: uniform across members, finer at fillets and support ends
h = a.dmin / a.n            # mm
m = 1e-3
arcs = [abs(c) for _, c in gmsh.model.getBoundary([(2, s)], oriented=False)
        if gmsh.model.getType(1, abs(c)) == "Circle"]
sup_pts = [p for c in groups["support_left"] + groups["support_right"]
           for _, p in gmsh.model.getBoundary([(1, c)], oriented=False)]
f1 = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(f1, "CurvesList", arcs)
gmsh.model.mesh.field.setNumbers(f1, "PointsList", [abs(p) for p in sup_pts])
gmsh.model.mesh.field.setNumber(f1, "Sampling", 200)
f2 = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(f2, "InField", f1)
gmsh.model.mesh.field.setNumber(f2, "SizeMin", m * h / a.refine)
gmsh.model.mesh.field.setNumber(f2, "SizeMax", m * h)
gmsh.model.mesh.field.setNumber(f2, "DistMin", m * 0.25 * a.dmin)
gmsh.model.mesh.field.setNumber(f2, "DistMax", m * 1.0 * a.dmin)
gmsh.model.mesh.field.setAsBackgroundMesh(f2)
for opt in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
    gmsh.option.setNumber("Mesh." + opt, 0)

# ---- physical groups, mesh
gmsh.model.addPhysicalGroup(2, [s], 1, "frame")
tags = {"support_left": 11, "support_right": 12, "roof_left": 21, "roof_right": 22, "other": 30}
for k, t in tags.items():
    gmsh.model.addPhysicalGroup(1, groups[k], t, k)

gmsh.option.setNumber("Mesh.Algorithm", 6)          # Frontal-Delaunay
gmsh.model.mesh.generate(2)
if a.order > 1:
    gmsh.model.mesh.setOrder(a.order)
gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
gmsh.write(a.out)

# ---- report
area = occ.getMass(2, s)
lens = {k: sum(occ.getMass(1, c) for c in groups[k]) for k in tags}
proj = {}
for k in ("roof_left", "roof_right"):
    xs = []
    for c in groups[k]:
        b0, b1, *_ = gmsh.model.getBoundingBox(1, c)[:1] + gmsh.model.getBoundingBox(1, c)[3:4]
        xs.append((b0, b1))
    proj[k] = sum(x1 - x0 for x0, x1 in xs)   # edges are x-monotone on the roof
_, et, _ = gmsh.model.mesh.getElements(2)
nel = sum(len(e) for e in et)
nn = len(gmsh.model.mesh.getNodes()[0])
print("\n---------- mesh_truss report (SI units) ----------")
print(f"area = {area:.5f} m^2  ->  V = 0.076*A = {0.076*area:.5f} m^3")
print(f"support_left  length = {lens['support_left']*1e3:.2f} mm   (expect 100)")
print(f"support_right length = {lens['support_right']*1e3:.2f} mm   (expect 100)")
print(f"roof_left  horizontal projection = {proj['roof_left']:.4f} m  (curves {groups['roof_left']})")
print(f"roof_right horizontal projection = {proj['roof_right']:.4f} m  (curves {groups['roof_right']})")
print(f"h = {h:.1f} mm, near fillets/supports {h/a.refine:.1f} mm, order {a.order}")
print(f"triangles = {nel}, nodes = {nn}  ->  written {a.out}")
if a.gui:
    gmsh.fltk.run()
gmsh.finalize()
