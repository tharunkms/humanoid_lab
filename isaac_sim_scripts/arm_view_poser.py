"""
arm_view_poser.py -- OpenManipulator-X-only view poser (no Go1 base)

robot_view_poser.py solves Go1 base + arm together. That's the eventual
target, but right now there is only the OpenManipulator-X in the scene, and
it is only meant to be used for the CLOSE-IN views (the plan's "arm_topdown"
pose and the NBV "arm_i" candidates) -- the wide "body_*"/"approach" views
are a stand-in for Go1 walking and stay on the free-camera teleport in
isaac_sim_native_gui.py until Go1 is wired in.

This solves ONLY the arm's own joint1..joint4 so the D435i (mounted on
link5, same mount convention as robot_view_poser) lands at a requested
cam_pos/look_at. The arm's base is NOT moved -- it stays wherever it
currently sits in the stage (a stand, a table bracket, whatever mounts it in
reality), read once at construction time.

One-time setup before using this (Script Editor, timeline stopped):
    import robot_view_poser as rvp
    rvp.make_puppet(<stage>, "/World/open_manipulator_x")
so physics doesn't fight the poses this class writes.
"""

import math
import numpy as np

from robot_view_poser import (
    ARM_PATH, RIG_PATH, CAM_PRIM_NAME, CAMERA_LINK, ARM_IK_JOINTS, ARM_OTHER_JOINTS,
    CAM_MOUNT_XYZ, CAM_MOUNT_PITCH_DEG, ARM_SEEDS, W_DIR, REG_JOINT,
    POS_TOL_M, ANG_TOL_DEG,
    _Tree, _T, _ry, _inv, _world, _local, _write_worlds, _motion,
)

REG_ARM = np.array([REG_JOINT] * len(ARM_IK_JOINTS))


class ArmViewPoser:
    """Fixed-base OpenManipulator-X: solves joint1..joint4 only."""

    def __init__(self, stage=None, verbose=True):
        import omni.usd
        self.stage = stage or omni.usd.get_context().get_stage()
        s = self.stage
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

        # fixed base: wherever the arm root sits right now (a stand, a table
        # bracket -- whatever mounts it in reality). Never moved by this class.
        self.base_world = _world(s.GetPrimAtPath(self.arm.root))

        # camera rig / camera-in-link5 mount.
        rig = s.GetPrimAtPath(RIG_PATH)
        cam = s.GetPrimAtPath(f"{RIG_PATH}/{CAM_PRIM_NAME}")
        if not rig.IsValid() or not cam.IsValid():
            raise RuntimeError(f"camera rig not found: {RIG_PATH}/{CAM_PRIM_NAME}")
        self.rig_prim, self.cam_prim = rig, cam
        L_cam = _inv(_world(rig)) @ _world(cam)
        L_cam[:3, :3] /= np.linalg.norm(L_cam[:3, :3], axis=0)   # guard against leftover scale

        # Physically mounted: the rig is a REAL child of link5 (see
        # mount_arm_camera.py). Its local xformOp IS the mount -- read it
        # straight from USD instead of the CAM_MOUNT_XYZ/PITCH guess, and
        # skip re-writing the rig's world pose every frame in _apply(); USD's
        # own parent/child inheritance carries it along whenever link5 moves.
        self.physically_mounted = str(rig.GetParent().GetPath()) == self.cam_link
        if self.physically_mounted:
            self.rig_in_link = _local(rig)
        else:
            f = L_cam[:3, :3] @ np.array([0, 0, -1.0])   # USD camera looks along -Z
            u = L_cam[:3, :3] @ np.array([0, 1.0, 0])
            Bs = np.column_stack([f, u, np.cross(f, u)])
            x, z = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
            Bt = np.column_stack([x, z, np.cross(x, z)])
            R_m = _ry(math.radians(CAM_MOUNT_PITCH_DEG)) @ Bt @ Bs.T
            t_m = np.array(CAM_MOUNT_XYZ, float) - R_m @ L_cam[:3, 3]
            self.rig_in_link = _T(R_m, t_m)
        self.cam_in_link = self.rig_in_link @ L_cam

        self.last = None
        if verbose:
            reach = sum(np.linalg.norm(j.J0[:3, 3]) for j in self.chain) + np.linalg.norm(CAM_MOUNT_XYZ)
            mount = "physically mounted (USD child of link5)" if self.physically_mounted else "software mount (CAM_MOUNT_XYZ/PITCH placeholder)"
            print(f"[arm_view_poser] arm root {self.arm.root} @ z={self.base_world[2, 3]:.3f}  "
                  f"camera link {self.cam_link}  chain={names}  rough reach {reach:.3f} m  [{mount}]")

    # ---------------------------------------------------------- kinematics --
    def _arm_q(self, q):
        d = dict(ARM_OTHER_JOINTS)
        d.update(zip(self.ik_names, q))
        return d

    def _link_world(self, q):
        W = self.base_world
        qd = self._arm_q(q)
        for j in self.chain:
            W = W @ j.J0 @ _motion(j, qd.get(j.name, 0.0)) @ j.J1inv
        return W

    def _cam_world(self, q):
        return self._link_world(q) @ self.cam_in_link

    # -------------------------------------------------------------- solver --
    def _resid(self, q, p, d, seed):
        C = self._cam_world(q)
        return np.concatenate([C[:3, 3] - p, (-C[:3, 2] - d) * W_DIR, (q - seed) * REG_ARM])

    def _clamp(self, q):
        return np.clip(q, self.lo, self.hi)

    def _lm(self, q, p, d, seed, iters):
        q = self._clamp(q)
        r = self._resid(q, p, d, seed)
        cost, lam, eps = r @ r, 1e-2, 1e-6
        for _ in range(iters):
            J = np.empty((r.size, q.size))
            for k in range(q.size):
                qk = q.copy()
                qk[k] += eps
                J[:, k] = (self._resid(qk, p, d, seed) - r) / eps
            A, g = J.T @ J, J.T @ r
            improved = False
            while lam < 1e6:
                dq = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
                qn = self._clamp(q + dq)
                rn = self._resid(qn, p, d, seed)
                cn = rn @ rn
                if cn < cost:
                    q, r, lam, improved = qn, rn, max(lam / 3, 1e-7), True
                    step, cost = cost - cn, cn
                    break
                lam *= 4
            if not improved or step < 1e-12:
                break
        return q, cost

    def solve(self, cam_pos, look_at):
        p = np.asarray(cam_pos, float)
        t = np.asarray(look_at, float)
        d = t - p
        d /= np.linalg.norm(d)
        best = None
        for q0 in ARM_SEEDS:
            seed = self._clamp(np.array(q0, float))
            q, c = self._lm(seed, p, d, seed, 30)
            if best is None or c < best[1]:
                best = (q, c, seed)
        q, _ = self._lm(best[0], p, d, best[2], 150)
        return q, p, d

    # -------------------------------------------------------------- output --
    def _apply(self, q):
        W = dict(self.arm.fk(self.base_world, self._arm_q(q)))
        _write_worlds(self.stage, W)
        if not self.physically_mounted:
            # free-floating rig: still has to be moved by hand every solve.
            # A real child of link5 (physically_mounted) rides along for
            # free through USD's own parent/child inheritance -- writing it
            # here too would just be redundant duplicate work.
            rig_world = W[self.cam_link] @ self.rig_in_link
            _write_worlds(self.stage, {RIG_PATH: rig_world})
        self.last = q.copy()

    def set_view(self, cam_pos, look_at):
        """Pose the arm so the D435i sits at cam_pos looking at look_at.
        Only 4 joints for 5 constraints (3 position + 2 direction) -- may not
        land exactly; check info['ok'] and fall back to the free camera if not."""
        q, p, d = self.solve(cam_pos, look_at)
        self._apply(q)
        C = self._cam_world(q)
        pos_err = float(np.linalg.norm(C[:3, 3] - p))
        ang_err = math.degrees(math.acos(float(np.clip((-C[:3, 2]) @ d, -1, 1))))
        ok = pos_err <= POS_TOL_M and ang_err <= ANG_TOL_DEG
        info = {
            "ok": ok,
            "requested_pos": p.round(3).tolist(),
            "achieved_pos": C[:3, 3].round(3).tolist(),
            "pos_err_m": round(pos_err, 4),
            "ang_err_deg": round(ang_err, 2),
            "arm_deg": {n: round(math.degrees(v), 1) for n, v in zip(self.ik_names, q)},
            "cam_to_world": C,
        }
        tag = "OK " if ok else "OFF"
        print(f"[arm_view_poser] {tag} pos_err={info['pos_err_m']*1000:.1f} mm "
              f"ang_err={info['ang_err_deg']:.1f} deg arm={info['arm_deg']}")
        return info

    def camera_world(self):
        return _world(self.cam_prim)
