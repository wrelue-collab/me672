# rebuild_truss.py  -  FreeCAD macro (Macro > Macros... > Execute)
# Rebuilds the ME672 A3 timber frame profile from scratch: exact nodes,
# per-member widths, circular joint nodes, rounded internal corners,
# 76 mm solid, volume printout, and a BREP of the 2D face for gmsh.
# Change only the PARAMETERS block, then re-run.

import os, math, re, json
import FreeCAD as App
import Part

# ======================= PARAMETERS (mm) ==========================
b        = 76.0     # out-of-plane thickness
EXT      = 50.0     # bottom chord extension past B0 and B2 (support segments)
R_FILLET = 40.0     # radius of every internal corner (>= 20 mm)

NODES = {
    "B0": (0.0, 0.0),      "B1": (4000.0, 0.0),        "B2": (8000.0, 0.0),
    "A":  (2000.0, 2000.0/3), "R": (4000.0, 4000.0/3), "C":  (6000.0, 2000.0/3),
}
# member width d, choose from {89, 140, 184, 235, 286}
WIDTHS = {
    "B0-B1": 184, "B1-B2": 184,                              # bottom chord
    "B0-A": 286, "A-R": 286, "R-C": 286, "C-B2": 286,        # upper members
    "A-B1":89, "R-B1": 89, "C-B1": 89,                   # webs
}
DOC_NAME = "Design_6"
# ==================================================================

V = App.Vector


def strip(p, q, d, e0=0.0, e1=0.0):
    """Rectangle of width d centred on segment p->q, extended e0/e1 past the ends."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    nx, ny = -uy * d / 2, ux * d / 2
    a = (p[0] - e0 * ux, p[1] - e0 * uy)
    c = (q[0] + e1 * ux, q[1] + e1 * uy)
    pts = [V(a[0] + nx, a[1] + ny, 0), V(a[0] - nx, a[1] - ny, 0),
           V(c[0] - nx, c[1] - ny, 0), V(c[0] + nx, c[1] + ny, 0)]
    return Part.Face(Part.makePolygon(pts + [pts[0]]))


def as_face(shape):
    if shape.ShapeType == "Face":
        return shape
    if shape.Faces:
        return max(shape.Faces, key=lambda f: f.Area)
    return Part.Face(Part.Wire(shape.Edges))


def offset(face, r):
    return as_face(face.makeOffset2D(r, 0))   # join=0 -> arc joints


# ---- 1. members: strips node-to-node ------------------------------
# supports = left/rightmost nodes on y = 0; lower-chord members = both ends on y = 0
base = [n for n, p in NODES.items() if abs(p[1]) < 1e-9]
SUP_L = min(base, key=lambda n: NODES[n][0])
SUP_R = max(base, key=lambda n: NODES[n][0])
bottom = [m for m in WIDTHS if all(abs(NODES[n][1]) < 1e-9 for n in m.split("-"))]

shapes = []
for name, d in WIDTHS.items():
    n0, n1 = name.split("-")
    on_bot = name in bottom
    e0 = EXT if on_bot and n0 in (SUP_L, SUP_R) else 0.0
    e1 = EXT if on_bot and n1 in (SUP_L, SUP_R) else 0.0
    shapes.append(strip(NODES[n0], NODES[n1], d, e0, e1))

# ---- 2. joints: circular node, diameter = widest member at that node
#         (covers every square member end, so nothing pokes out)
for n, c in NODES.items():
    r = max(d for m, d in WIDTHS.items() if n in m.split("-")) / 2.0
    shapes.append(Part.Face(Part.Wire(Part.makeCircle(r, V(c[0], c[1], 0), V(0, 0, 1)))))

body = shapes[0].fuse(shapes[1:])

# ---- 3. trim: each lower-chord member's own lower face, vertical eave cuts
xL, xR = NODES[SUP_L][0] - EXT, NODES[SUP_R][0] + EXT
segs = sorted((min(NODES[n][0] for n in m.split("-")), max(NODES[n][0] for n in m.split("-")),
               -WIDTHS[m] / 2.0) for m in bottom)
clip = []
for i, (x0, x1, yb) in enumerate(segs):
    x0 = xL if i == 0 else x0
    x1 = xR if i == len(segs) - 1 else x1
    clip += [V(x0, yb, 0), V(x1, yb, 0)]            # steps appear where widths change
clip += [V(xR, 5000, 0), V(xL, 5000, 0)]
clip = [p for i, p in enumerate(clip) if i == 0 or (p - clip[i - 1]).Length > 1e-6]   # no duplicates
body = as_face(body.common(Part.Face(Part.makePolygon(clip + [clip[0]]))).removeSplitter())

# ---- 4. round internal corners --------------------------------------
outer = Part.Face(body.OuterWire)
outer = offset(offset(outer, R_FILLET), -R_FILLET)      # re-entrant corners of the outline
profile = outer
n_holes = 0
for w in body.Wires:
    if w.isSame(body.OuterWire):
        continue
    hole = offset(offset(Part.Face(w), -R_FILLET), R_FILLET)   # corners of each opening
    profile = profile.cut(hole)
    n_holes += 1
profile = as_face(profile.removeSplitter())

# ---- 5. checks --------------------------------------------------------
area = profile.Area                       # mm^2
vol = area * b / 1e9                      # m^3
mass = 500.0 * vol                        # kg, rho = 500 kg/m^3
cl = sum(WIDTHS[m] * math.dist(NODES[m.split('-')[0]], NODES[m.split('-')[1]])
         for m in WIDTHS) * b / 1e9
ok_closed = len(profile.Wires) == 1 + n_holes and profile.isValid()
print("-------- Timber frame rebuild --------")
print(f"valid face: {ok_closed}   openings: {n_holes}   internal corner radius: {R_FILLET} mm")
print(f"area = {area/1e6:.4f} m^2   V = b*A = {vol:.4f} m^3 per frame   (centreline sum {cl:.4f})")
print(f"mass = {mass:.1f} kg per frame   21V = {21*vol:.3f} m^3   21*rho*V = {21*mass:.0f} kg")
print(f"bounding box: {profile.BoundBox}")

# ---- 6. into a document -----------------------------------------------
doc = App.newDocument(DOC_NAME)
f2d = doc.addObject("Part::Feature", "Profile_2D")
f2d.Shape = profile
solid = doc.addObject("Part::Feature", "Frame_Solid")
solid.Shape = profile.extrude(V(0, 0, b))
f2d.Visibility = False
doc.recompute()
print(f"solid valid: {solid.Shape.isValid()}   solid volume = {solid.Shape.Volume/1e9:.4f} m^3")

# save next to this macro (or in the home folder)
try:
    folder = os.path.dirname(os.path.abspath(__file__))
except NameError:
    folder = os.path.expanduser("~")
doc.saveAs(os.path.join(folder, DOC_NAME + ".FCStd"))
brep_name = DOC_NAME + "_profile2d.brep"
profile.exportBrep(os.path.join(folder, brep_name))   # 2D face for gmsh / frame_general.ipynb
design = {"name": DOC_NAME, "brep": brep_name, "units": "mm", "b": b, "ext": EXT,
          "r_fillet": R_FILLET, "nodes": {k: list(v) for k, v in NODES.items()},
          "members": WIDTHS, "V_frame_m3": vol}
with open(os.path.join(folder, DOC_NAME + ".json"), "w") as fh:
    json.dump(design, fh, indent=1)
print("saved:", DOC_NAME + ".FCStd,", brep_name, "and", DOC_NAME + ".json in", folder)

if App.GuiUp:
    import FreeCADGui as Gui
    Gui.SendMsgToActiveView("ViewFit")
    Gui.activeDocument().activeView().viewIsometric()
