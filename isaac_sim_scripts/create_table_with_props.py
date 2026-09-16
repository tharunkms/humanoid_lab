"""
create_table_with_props.py -- paste into Isaac Sim's Script Editor (Window > Script Editor).

Creates a round table with a scattered set of props on top: a cup, a mug
(with a simple handle), a bottle, a cuboid (box), and a plain cylinder --
good variety of shapes for testing segmentation/classification and
grasp planning across different geometries.

Built from basic USD primitives (Cylinder, Cube) with physics enabled
(rigid body + collider), so props can be grasped/pushed once the
simulation plays. No external asset files needed -- runs standalone.

Usage: paste the whole thing into Script Editor, Ctrl+Enter once. Re-run
to rebuild the scene (it clears any existing prims at TABLE_ROOT first).
"""
import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics, Sdf

stage = omni.usd.get_context().get_stage()

# ---- EDIT THESE TO POSITION THE SCENE -----------------------------------
TABLE_ROOT = "/World/PropTable"
TABLE_CENTER = Gf.Vec3d(1.5, 1.5, 0.0)   # world x, y, z of the table's base (floor contact point)
TABLE_RADIUS = 0.45                       # metres
TABLE_HEIGHT = 0.42                       # metres, floor to tabletop surface
TABLE_THICKNESS = 0.03                    # metres
ENABLE_PHYSICS = True                     # rigid body + collider on table and all props
# ---------------------------------------------------------------------------

# Clear any previous run's prims first, so re-running rebuilds cleanly.
existing = stage.GetPrimAtPath(TABLE_ROOT)
if existing.IsValid():
    stage.RemovePrim(Sdf.Path(TABLE_ROOT))

root_xform = UsdGeom.Xform.Define(stage, Sdf.Path(TABLE_ROOT))


def _set_color(prim, rgb):
    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])


def _add_physics(prim, mass=None, static=False):
    if not ENABLE_PHYSICS:
        return
    UsdPhysics.CollisionAPI.Apply(prim)
    if not static:
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        if mass is not None:
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(mass)


def _make_cylinder(path, radius, height, position, rgb, mass=None, static=False):
    cyl = UsdGeom.Cylinder.Define(stage, Sdf.Path(path))
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr("Z")
    xf = UsdGeom.Xformable(cyl)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(position)
    _set_color(cyl.GetPrim(), rgb)
    _add_physics(cyl.GetPrim(), mass=mass, static=static)
    return cyl


def _make_cube(path, size_xyz, position, rgb, mass=None, static=False):
    cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
    cube.CreateSizeAttr(1.0)  # base unit cube, scaled via xform below
    xf = UsdGeom.Xformable(cube)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(position)
    xf.AddScaleOp().Set(Gf.Vec3f(size_xyz[0], size_xyz[1], size_xyz[2]))
    _set_color(cube.GetPrim(), rgb)
    _add_physics(cube.GetPrim(), mass=mass, static=static)
    return cube


# ---- Table -----------------------------------------------------------
# Tabletop: a wide, flat cylinder.
table_z = TABLE_CENTER[2] + TABLE_HEIGHT - TABLE_THICKNESS / 2
_make_cylinder(
    f"{TABLE_ROOT}/Tabletop", TABLE_RADIUS, TABLE_THICKNESS,
    Gf.Vec3d(TABLE_CENTER[0], TABLE_CENTER[1], table_z),
    rgb=(0.55, 0.4, 0.25), static=True)

# Central pedestal leg.
leg_radius = 0.06
leg_height = TABLE_HEIGHT - TABLE_THICKNESS
_make_cylinder(
    f"{TABLE_ROOT}/Leg", leg_radius, leg_height,
    Gf.Vec3d(TABLE_CENTER[0], TABLE_CENTER[1], TABLE_CENTER[2] + leg_height / 2),
    rgb=(0.3, 0.3, 0.3), static=True)

# Base plate for stability (visual only, matches a typical pedestal table).
_make_cylinder(
    f"{TABLE_ROOT}/Base", TABLE_RADIUS * 0.5, 0.02,
    Gf.Vec3d(TABLE_CENTER[0], TABLE_CENTER[1], TABLE_CENTER[2] + 0.01),
    rgb=(0.3, 0.3, 0.3), static=True)

surface_z = TABLE_CENTER[2] + TABLE_HEIGHT  # top surface of the tabletop

# ---- Props, scattered around the tabletop -----------------------------
# Each prop is placed a bit off-center so they don't overlap, at a height
# that sits its base exactly on the table surface.

# 1. Cup -- short, narrow cylinder.
cup_r, cup_h = 0.035, 0.09
_make_cylinder(
    f"{TABLE_ROOT}/Cup", cup_r, cup_h,
    Gf.Vec3d(TABLE_CENTER[0] + 0.18, TABLE_CENTER[1] + 0.10, surface_z + cup_h / 2),
    rgb=(0.9, 0.9, 0.95), mass=0.15)

# 2. Mug -- wider/shorter cylinder body + a simple handle (thin curved
# cube approximation, since USD has no native torus primitive).
mug_r, mug_h = 0.04, 0.10
mug_pos = Gf.Vec3d(TABLE_CENTER[0] - 0.12, TABLE_CENTER[1] + 0.15, surface_z + mug_h / 2)
_make_cylinder(f"{TABLE_ROOT}/Mug_Body", mug_r, mug_h, mug_pos, rgb=(0.15, 0.45, 0.2), mass=0.2)
# Handle: a thin flattened cube offset to the side of the mug body, at
# mid-height -- a simple stand-in for a curved handle.
handle_pos = Gf.Vec3d(mug_pos[0] + mug_r + 0.012, mug_pos[1], mug_pos[2])
_make_cube(
    f"{TABLE_ROOT}/Mug_Handle", (0.008, 0.025, mug_h * 0.5),
    handle_pos, rgb=(0.15, 0.45, 0.2), mass=0.02)

# 3. Bottle -- tall, narrow cylinder with a smaller "neck" cylinder on top.
bottle_r, bottle_h = 0.03, 0.18
bottle_pos = Gf.Vec3d(TABLE_CENTER[0] + 0.05, TABLE_CENTER[1] - 0.18, surface_z + bottle_h / 2)
_make_cylinder(f"{TABLE_ROOT}/Bottle_Body", bottle_r, bottle_h, bottle_pos,
               rgb=(0.1, 0.5, 0.7), mass=0.3)
neck_r, neck_h = 0.012, 0.04
neck_pos = Gf.Vec3d(bottle_pos[0], bottle_pos[1], bottle_pos[2] + bottle_h / 2 + neck_h / 2)
_make_cylinder(f"{TABLE_ROOT}/Bottle_Neck", neck_r, neck_h, neck_pos,
               rgb=(0.1, 0.5, 0.7), mass=0.05)

# 4. Cuboid -- a plain rectangular box.
cuboid_size = (0.07, 0.05, 0.06)  # width, depth, height
_make_cube(
    f"{TABLE_ROOT}/Cuboid", cuboid_size,
    Gf.Vec3d(TABLE_CENTER[0] - 0.15, TABLE_CENTER[1] - 0.12, surface_z + cuboid_size[2] / 2),
    rgb=(0.75, 0.2, 0.2), mass=0.25)

# 5. Plain cylinder -- distinct from the cup/mug/bottle, a squat can-like shape.
plain_r, plain_h = 0.045, 0.07
_make_cylinder(
    f"{TABLE_ROOT}/Cylinder", plain_r, plain_h,
    Gf.Vec3d(TABLE_CENTER[0] + 0.02, TABLE_CENTER[1] + 0.02, surface_z + plain_h / 2),
    rgb=(0.85, 0.65, 0.1), mass=0.2)

print(f"Created table + 5 props at {TABLE_ROOT}, centered at world "
      f"({TABLE_CENTER[0]}, {TABLE_CENTER[1]}, {TABLE_CENTER[2]}).")
print("Props: Cup, Mug (body+handle), Bottle (body+neck), Cuboid, Cylinder.")
if ENABLE_PHYSICS:
    print("Physics enabled -- press Play to let props settle onto the table surface.")
