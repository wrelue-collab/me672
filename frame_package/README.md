# ME672 A3 frame analysis package

## Files
- `rebuild_truss.py`: FreeCAD macro. Edit its PARAMETERS block and run it. It writes
  `<DOC_NAME>_profile2d.brep` and `<DOC_NAME>.json` (nodes, widths, units).
- `designs/`: one `.json` + `.brep` per design (baseline and king post included).
- `frame_general.ipynb`: plane-stress FEniCSx analysis of every design listed in `DESIGNS`,
  plus a pin-jointed truss model, member axial forces and buckling, a mesh study, plots and CSVs.

## Add a design
1. Set NODES / WIDTHS / DOC_NAME in `rebuild_truss.py`, then run it
   (GUI, or headless: `FreeCADCmd rebuild_truss.py`).
2. Copy the `.json` and `.brep` it wrote into `designs/`.
3. Add `"designs/<DOC_NAME>.json"` to `DESIGNS` in the notebook's first code cell.
   A BREP of the extruded solid works too: its z = 0 face is used.

## Reproduce everything (one command, inside the DOLFINx container)
    jupyter nbconvert --to notebook --execute frame_general.ipynb --output frame_general_run.ipynb

## Outputs (`results/`)
- `summary.csv`: one row per design (volume, mass, utilisations, governing case, feasibility)
- `<design>_cases.csv`: per load case (displacements, stress ratios and their locations,
  excluded support peaks, reactions, equilibrium and energy checks, frame-vs-truss load match)
- `<design>_members.csv`: axial force and moment on clean sections, truss force, buckling ratio
- `convergence.csv`, `versions.json`, and PNG plots (mesh, deformed shape, stress utilisation)
