"""
robot_view_poser.py  --  Go1 + OpenManipulator-X + D435i "puppet" for the perception pipeline
=============================================================================================

Replaces the old "teleport the free-floating camera rig" step. The camera is now
mounted on the arm's gripper link (link5), and the arm is mounted on top of the
Go1 trunk. To reach a requested view the poser:

  1. solves Go1 base (x, y, yaw) + arm joints (joint1..joint4) so the D435i
     optical centre lands on the requested position and looks at the target
     (damped least squares, body kept outside the table edge),
  2. writes every Go1 link, every arm link and the camera rig to USD
     (kinematic puppet: physics on the robot is switched off, so nothing
     falls or fights the pose),
  3. returns the achieved pose and the error vs. the request.

The camera rig stays at RIG_PATH (/World/d435i_camera). The GUI keeps reading
Camera_RGB's world transform from USD exactly as before, so capture and fusion
are unchanged. The mount is enforced by the poser (rig = link5 * MOUNT) on every
call, not by USD parenting.

Scope: perception only. This is a stand-in for Imanish's locomotion and
Sanditya's arm control, just like the old teleport was.
"""

import math
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf

# ------------------------------------------------------------------ config --
GO1_PATH = "/World/go1"
ARM_PATH = "/World/open_manipulator_x"
RIG_PATH = "/World/d435i_camera"
CAM_PRIM_NAME = "Camera_RGB"
TABLE_PATH = "/World/PropTable"
TABLETOP_PATH = "/World/PropTable/Tabletop"

CAMERA_LINK = "link5"                  # OMX gripper link the D435i is mounted on
ARM_IK_JOINTS = ["joint1", "joint2", "joint3", "joint4"]
ARM_OTHER_JOINTS = {"gripper": 0.0, "gripper_sub": 0.0}   # held fixed

# Go1 standing pose (matched by joint-name suffix), radians
GO1_STAND_JOINTS = {"hip_joint": 0.0, "thigh_joint": 0.8, "calf_joint": -1.5}
GO1_FOOT_HINT = "foot"                 # link-name substring used to find the feet
FOOT_RADIUS_M = 0.02
GO1_STAND_HEIGHT_FALLBACK_M = 0.30     # used only if no foot links are found
FLOOR_Z = None                         # None -> bottom of TABLE_PATH bounding box

# Mounts -- PLACEHOLDERS, refine on hardware
ARM_MOUNT_XYZ = (0.0, 0.0, 0.06)       # arm root in Go1 trunk frame (top centre)
ARM_MOUNT_YAW_DEG = 0.0
CAM_MOUNT_XYZ = (-0.02, 0.0, 0.05)     # D435i optical centre in link5 frame (above/behind gripper)
CAM_MOUNT_PITCH_DEG = 0.0              # + tilts the camera down relative to the gripper

# Body / table constraint (keep in sync with the GUI planner constants)
BODY_HALF_LENGTH_M = 0.19
BODY_HALF_WIDTH_M = 0.10
BODY_EDGE_CLEARANCE_M = 0.25

# Solver weights
W_DIR = 0.3          # 1 rad of look-direction error ~ 0.3 m of position error
W_CLEAR = 10.0       # body-in-table is penalised hard
REG_BASE = 0.02
REG_JOINT = 0.005
BASE_SEED_DISTS = (0.25, 0.35, 0.45)   # trunk centre behind the camera, metres
ARM_SEEDS = ((0.0, 0.0, 0.0, 0.0),
             (0.0, -0.5, 0.2, 0.6),
             (0.0, 0.3, 0.3, 1.0),
             (0.0, -1.0, 0.8, 0.8))
POS_TOL_M = 0.01
ANG_TOL_DEG = 5.0

ARM_HOME = (0.0, -0.6, 0.3, 0.3)       # used by set_base()


# ------------------------------------------------------------ math helpers --
def _T(R=None, t=None):
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R
    if t is not None:
        M[:3, 3] = t
    return M


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_axis(u, a):
    u = np.asarray(u, float)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def _inv(M):
    R, t = M[:3, :3], M[:3, 3]
    return _T(R.T, -R.T @ t)


def _quat_R(q):
    if q is None:
        return np.eye(3)
    w = q.GetReal()
    x, y, z = q.GetImaginary()
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _from_gf(m):   # Gf is row-vector (translation in last row) -> column convention
    return np.array([[m[i][j] for j in range(4)] for i in range(4)], float).T


def _to_gf(M):
    return Gf.Matrix4d(*[float(v) for v in np.asarray(M, float).T.flatten()])


_AXES = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}


class _Joint:
    __slots__ = ("name", "parent", "child", "J0", "J1inv", "kind", "axis", "lo", "hi")


def _motion(j, v):
    if j.kind == "rev":
        return _T(R=_rot_axis(j.axis, v))
    if j.kind == "pri":
        return _T(t=j.axis * v)
    return np.eye(4)


class _Tree:
    """Kinematic tree read from the USD physics joints of one robot."""

    def __init__(self, stage, root_path):
        top = stage.GetPrimAtPath(root_path)
        if not top.IsValid():
            raise RuntimeError(f"robot prim not found: {root_path}")
        self.links, self.joints = set(), []
        for prim in Usd.PrimRange(top):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                self.links.add(str(prim.GetPath()))
            if not prim.IsA(UsdPhysics.Joint):
                continue
            uj = UsdPhysics.Joint(prim)
            b0 = uj.GetBody0Rel().GetTargets()
            b1 = uj.GetBody1Rel().GetTargets()
            if not b0 or not b1:
                continue                       # world anchor (root_joint etc.)
            j = _Joint()
            j.name = prim.GetName()
            j.parent, j.child = str(b0[0]), str(b1[0])
            p0 = uj.GetLocalPos0Attr().Get()
            p1 = uj.GetLocalPos1Attr().Get()
            j.J0 = _T(_quat_R(uj.GetLocalRot0Attr().Get()),
                      np.array(p0, float) if p0 is not None else None)
            J1 = _T(_quat_R(uj.GetLocalRot1Attr().Get()),
                    np.array(p1, float) if p1 is not None else None)
            j.J1inv = _inv(J1)
            j.lo, j.hi = -math.inf, math.inf
            if prim.IsA(UsdPhysics.RevoluteJoint):
                rj = UsdPhysics.RevoluteJoint(prim)
                j.kind, j.axis = "rev", _AXES[rj.GetAxisAttr().Get() or "X"]
                lo, hi = rj.GetLowerLimitAttr().Get(), rj.GetUpperLimitAttr().Get()
                if lo is not None and hi is not None and abs(lo) < 1e6 and abs(hi) < 1e6:
                    j.lo, j.hi = math.radians(lo), math.radians(hi)
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                pj = UsdPhysics.PrismaticJoint(prim)
                j.kind, j.axis = "pri", _AXES[pj.GetAxisAttr().Get() or "X"]
                lo, hi = pj.GetLowerLimitAttr().Get(), pj.GetUpperLimitAttr().Get()
                if lo is not None and hi is not None and abs(lo) < 1e6 and abs(hi) < 1e6:
                    j.lo, j.hi = lo, hi
            else:
                j.kind, j.axis = "fix", _AXES["X"]
            self.joints.append(j)
            self.links.update((j.parent, j.child))

        self.children = {}
        for j in self.joints:
            self.children.setdefault(j.parent, []).append(j)
        childs = {j.child for j in self.joints}
        roots = [l for l in self.links if l not in childs]
        if not roots:
            raise RuntimeError(f"no root link found under {root_path}")
        roots.sort(key=lambda r: -self._count(r))
        self.root = roots[0]
        self.by_name = {j.name: j for j in self.joints}

    def _count(self, link):
        n, stack = 0, [link]
        while stack:
            for j in self.children.get(stack.pop(), []):
                n += 1
                stack.append(j.child)
        return n

    def find_link(self, name):
        for l in self.links:
            if l.rsplit("/", 1)[-1] == name:
                return l
        raise RuntimeError(f"link '{name}' not found; links: {sorted(self.links)}")

    def chain_to(self, link):
        parent_joint = {j.child: j for j in self.joints}
        chain, cur = [], link
        while cur != self.root:
            if cur not in parent_joint:
                raise RuntimeError(f"{link} is not connected to root {self.root}")
            j = parent_joint[cur]
            chain.append(j)
            cur = j.parent
        return chain[::-1]

    def fk(self, root_world, q):
        W, stack = {self.root: root_world}, [self.root]
        while stack:
            p = stack.pop()
            for j in self.children.get(p, []):
                W[j.child] = W[p] @ j.J0 @ _motion(j, q.get(j.name, 0.0)) @ j.J1inv
                stack.append(j.child)
        return W


# ------------------------------------------------------------- USD helpers --
def _world(prim):
    return _from_gf(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))


def _local(prim):
    if not prim.IsA(UsdGeom.Xformable):
        return np.eye(4)
    return _from_gf(UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode.Default()))


def _set_local(prim, M):
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    if len(ops) == 1 and ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform:
        ops[0].Set(_to_gf(M))
        return
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(_to_gf(M))


def _write_worlds(stage, W):
    """Write world matrices; parents first so nested links stay consistent."""
    for path in sorted(W, key=lambda s: s.count("/")):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        acc, par = np.eye(4), prim.GetParent()
        parent_world = None
        while par and par.IsValid() and not par.IsPseudoRoot():
            ps = str(par.GetPath())
            if ps in W:
                parent_world = W[ps] @ acc
                break
            acc = _local(par) @ acc
            par = par.GetParent()
        if parent_world is None:
            parent_world = acc
        _set_local(prim, _inv(parent_world) @ W[path])


def _deinstance(stage, root_path):
    changed = True
    while changed:
        changed = False
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
            if prim.IsInstance():
                prim.SetInstanceable(False)
                changed = True
                break


def make_puppet(stage, root_path):
    """Switch physics off on a robot so poses written to USD are final."""
    _deinstance(stage, root_path)
    n = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            n += 1
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            n += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            n += 1
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
            n += 1
    print(f"[robot_view_poser] puppet mode on {root_path}: {n} physics attributes disabled")


def _bbox(stage, path):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    r = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    return np.array(r.GetMin(), float), np.array(r.GetMax(), float)


# ------------------------------------------------------------------ poser --
class RobotViewPoser:
    def __init__(self, stage=None, verbose=True):
        import omni.usd
        self.stage = stage or omni.usd.get_context().get_stage()
        s = self.stage
        self.go1 = _Tree(s, GO1_PATH)
        self.arm = _Tree(s, ARM_PATH)
        self.cam_link = self.arm.find_link(CAMERA_LINK)
        self.chain = self.arm.chain_to(self.cam_link)

        names = [j.name for j in self.chain]
        missing = [n for n in ARM_IK_JOINTS if n not in names]
        if missing:
            raise RuntimeError(f"IK joints {missing} not on chain {names}")
        self.ik_names = list(ARM_IK_JOINTS)
        self.lo = np.array([self.arm.by_name[n].lo for n in self.ik_names])
        self.hi = np.array([self.arm.by_name[n].hi for n in self.ik_names])

        self.go1_q = {}
        for j in self.go1.joints:
            for suf, v in GO1_STAND_JOINTS.items():
                if j.name.endswith(suf):
                    self.go1_q[j.name] = v

        # camera rig: Camera_RGB pose inside the rig, then the mount in link5
        rig = s.GetPrimAtPath(RIG_PATH)
        cam = s.GetPrimAtPath(f"{RIG_PATH}/{CAM_PRIM_NAME}")
        if not rig.IsValid() or not cam.IsValid():
            raise RuntimeError(f"camera rig not found: {RIG_PATH}/{CAM_PRIM_NAME}")
        self.rig_prim, self.cam_prim = rig, cam
        L_cam = _inv(_world(rig)) @ _world(cam)
        L_cam[:3, :3] /= np.linalg.norm(L_cam[:3, :3], axis=0)  # guard against leftover scale
        f = L_cam[:3, :3] @ np.array([0, 0, -1.0])      # USD camera looks along -Z
        u = L_cam[:3, :3] @ np.array([0, 1.0, 0])
        Bs = np.column_stack([f, u, np.cross(f, u)])
        x, z = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
        Bt = np.column_stack([x, z, np.cross(x, z)])
        R_m = _ry(math.radians(CAM_MOUNT_PITCH_DEG)) @ Bt @ Bs.T
        t_m = np.array(CAM_MOUNT_XYZ, float) - R_m @ L_cam[:3, 3]
        self.rig_in_link = _T(R_m, t_m)
        self.cam_in_link = self.rig_in_link @ L_cam

        self.arm_mount = _T(_rz(math.radians(ARM_MOUNT_YAW_DEG)), np.array(ARM_MOUNT_XYZ, float))

        # standing height
        bmin, _ = _bbox(s, TABLE_PATH)
        self.floor_z = FLOOR_Z if FLOOR_Z is not None else float(bmin[2])
        Wg = self.go1.fk(np.eye(4), self.go1_q)
        feet = [W[2, 3] for l, W in Wg.items() if GO1_FOOT_HINT in l.rsplit("/", 1)[-1].lower()]
        if feet:
            self.base_z = self.floor_z - min(feet) + FOOT_RADIUS_M
        else:
            self.base_z = self.floor_z + GO1_STAND_HEIGHT_FALLBACK_M
            print("[robot_view_poser] WARNING: no foot links found, using fallback stand height")

        tmin, tmax = _bbox(s, TABLETOP_PATH)
        self.table_c = 0.5 * (tmin[:2] + tmax[:2])
        self.table_r = 0.5 * (tmax[0] - tmin[0])
        self.table_top = float(tmax[2])

        self.reg_w = np.array([REG_BASE] * 3 + [REG_JOINT] * len(self.ik_names))
        self.last = None
        if verbose:
            self.report()

    # ---------------------------------------------------------- kinematics --
    def _arm_q(self, q):
        d = dict(ARM_OTHER_JOINTS)
        d.update(zip(self.ik_names, q))
        return d

    def _base(self, bx, by, yaw):
        return _T(_rz(yaw), np.array([bx, by, self.base_z]))

    def _link_world(self, x):
        W = self._base(*x[:3]) @ self.arm_mount
        qd = self._arm_q(x[3:])
        for j in self.chain:
            W = W @ j.J0 @ _motion(j, qd.get(j.name, 0.0)) @ j.J1inv
        return W

    def _cam_world(self, x):
        return self._link_world(x) @ self.cam_in_link

    # -------------------------------------------------------------- solver --
    def _resid(self, x, p, d, seed):
        C = self._cam_world(x)
        r = [C[:3, 3] - p, (-C[:3, 2] - d) * W_DIR, (x - seed) * self.reg_w]
        h = np.array([math.cos(x[2]), math.sin(x[2])])
        n = np.array([-h[1], h[0]])
        c = x[:2]
        viol = []
        for pt in (c, c + h * BODY_HALF_LENGTH_M,
                   c + h * BODY_HALF_LENGTH_M + n * BODY_HALF_WIDTH_M,
                   c + h * BODY_HALF_LENGTH_M - n * BODY_HALF_WIDTH_M):
            dist = np.linalg.norm(pt - self.table_c)
            viol.append(max(0.0, self.table_r + BODY_EDGE_CLEARANCE_M - dist) * W_CLEAR)
        r.append(np.array(viol))
        return np.concatenate(r)

    def _clamp(self, x):
        x = x.copy()
        x[2] = (x[2] + math.pi) % (2 * math.pi) - math.pi
        x[3:] = np.clip(x[3:], self.lo, self.hi)
        return x

    def _lm(self, x, p, d, seed, iters):
        x = self._clamp(x)
        r = self._resid(x, p, d, seed)
        cost, lam, eps = r @ r, 1e-2, 1e-6
        for _ in range(iters):
            J = np.empty((r.size, x.size))
            for k in range(x.size):
                xk = x.copy()
                xk[k] += eps
                J[:, k] = (self._resid(xk, p, d, seed) - r) / eps
            A, g = J.T @ J, J.T @ r
            improved = False
            while lam < 1e6:
                dx = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
                xn = self._clamp(x + dx)
                rn = self._resid(xn, p, d, seed)
                cn = rn @ rn
                if cn < cost:
                    x, r, lam, improved = xn, rn, max(lam / 3, 1e-7), True
                    step, cost = cost - cn, cn
                    break
                lam *= 4
            if not improved or step < 1e-12:
                break
        return x, cost

    def solve(self, cam_pos, look_at):
        p = np.asarray(cam_pos, float)
        t = np.asarray(look_at, float)
        d = t - p
        d /= np.linalg.norm(d)
        u = p[:2] - t[:2]
        if np.linalg.norm(u) < 1e-3:
            u = -d[:2] if np.linalg.norm(d[:2]) > 1e-3 else p[:2] - self.table_c
        u /= np.linalg.norm(u)
        yaw = math.atan2(-u[1], -u[0])          # body faces the target
        best = None
        for s in BASE_SEED_DISTS:
            bxy = p[:2] + u * s
            for q0 in ARM_SEEDS:
                seed = self._clamp(np.array([bxy[0], bxy[1], yaw, *q0]))
                x, c = self._lm(seed, p, d, seed, 20)
                if best is None or c < best[1]:
                    best = (x, c, seed)
        x, _ = self._lm(best[0], p, d, best[2], 100)
        return x, p, d

    # -------------------------------------------------------------- output --
    def _apply(self, x):
        W = dict(self.go1.fk(self._base(*x[:3]), self.go1_q))
        W.update(self.arm.fk(self._base(*x[:3]) @ self.arm_mount, self._arm_q(x[3:])))
        _write_worlds(self.stage, W)
        rig_world = W[self.cam_link] @ self.rig_in_link
        _write_worlds(self.stage, {RIG_PATH: rig_world})
        self.last = x.copy()

    def set_view(self, cam_pos, look_at):
        """Pose the robot so the D435i sits at cam_pos looking at look_at."""
        x, p, d = self.solve(cam_pos, look_at)
        self._apply(x)
        C = self._cam_world(x)
        pos_err = float(np.linalg.norm(C[:3, 3] - p))
        ang_err = math.degrees(math.acos(float(np.clip((-C[:3, 2]) @ d, -1, 1))))
        clear = float(np.max(self._resid(x, p, d, x)[-4:]) / W_CLEAR)
        ok = pos_err <= POS_TOL_M and ang_err <= ANG_TOL_DEG and clear < 1e-3
        info = {
            "ok": ok,
            "requested_pos": p.round(3).tolist(),
            "achieved_pos": C[:3, 3].round(3).tolist(),
            "pos_err_m": round(pos_err, 4),
            "ang_err_deg": round(ang_err, 2),
            "body_in_clearance_m": round(clear, 3),
            "base_xy_yaw": [round(float(x[0]), 3), round(float(x[1]), 3),
                            round(math.degrees(x[2]), 1)],
            "arm_deg": {n: round(math.degrees(v), 1) for n, v in zip(self.ik_names, x[3:])},
            "cam_to_world": C,
        }
        tag = "OK " if ok else "OFF"
        print(f"[robot_view_poser] {tag} pos_err={info['pos_err_m']*1000:.1f} mm "
              f"ang_err={info['ang_err_deg']:.1f} deg clearance_viol={clear*100:.1f} cm "
              f"base={info['base_xy_yaw']} arm={info['arm_deg']}")
        return info

    def set_base(self, bx, by, yaw_rad, arm_q=ARM_HOME):
        """Locomotion stand-in: place the body, arm in a fixed pose."""
        x = self._clamp(np.array([bx, by, yaw_rad, *arm_q], float))
        self._apply(x)
        return self._cam_world(x)

    def camera_world(self):
        return _world(self.cam_prim)

    def report(self):
        shoulder = (self._base(0, 0, 0) @ self.arm_mount)[2, 3]
        reach = sum(np.linalg.norm(j.J0[:3, 3]) for j in self.chain) + np.linalg.norm(CAM_MOUNT_XYZ)
        print("[robot_view_poser] ---------------------------------------------")
        print(f"  Go1 root link   : {self.go1.root}   ({len(self.go1_q)} stand joints set)")
        print(f"  arm root link   : {self.arm.root}")
        print(f"  camera link     : {self.cam_link}   chain={[j.name for j in self.chain]}")
        print(f"  floor z         : {self.floor_z:.3f}")
        print(f"  trunk z         : {self.base_z:.3f}  ({self.base_z - self.floor_z:.3f} above floor)")
        print(f"  arm root z      : {shoulder:.3f}")
        print(f"  tabletop z      : {self.table_top:.3f}  ({self.table_top - self.floor_z:.3f} above floor)")
        print(f"  rough max cam z : {shoulder + reach:.3f}  "
              f"({shoulder + reach - self.table_top:+.3f} vs tabletop)")
        if shoulder + reach - self.table_top < 0.30:
            print("  WARNING: the camera cannot get ~0.3 m above the tabletop from the floor.")
            print("           Top-down views will be off. Set FLOOR_Z / lower the table, or")
            print("           lower the planner's view heights.")
        print("[robot_view_poser] ---------------------------------------------")
