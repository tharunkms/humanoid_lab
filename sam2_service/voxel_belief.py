"""
voxel_belief.py
---------------
Log-odds occupancy grid over the tabletop and the space above it, with
free / occupied / UNKNOWN per voxel -- the belief a next-best-view planner
reads ("which voxels near the target have I never seen?").

Pure numpy, no new dependencies. Same convention as the rest of the
pipeline: points are unprojected in the OpenCV optical frame
(x right, y down, z forward) and moved to world with the known
cam_to_world matrix, so integration needs nothing the capture step does
not already have.

Per voxel:
  log_odds  float32   >0 occupied, <0 free, 0 = never touched by a ray
  observed  bool      any ray has passed through or ended in this voxel

Log-odds update (OctoMap defaults, Hornung et al. 2013): a ray from the
camera to a depth hit marks every voxel it passes through as a MISS
(more free) and the endpoint voxel as a HIT (more occupied). Clamping
keeps voxels correctable instead of saturating.

Typical use:

    g = VoxelBelief.around_table(table_ctr, table_radius, table_top_z)
    g.integrate_depth(depth, cam_to_world, fx, fy, cx, cy)   # per view
    print(g.stats())
    g.save_ply("belief.ply")          # occupied voxels, for the viewer
    unk = g.unknown_near(target_xyz, radius=0.25)            # for the planner
"""
import numpy as np

# OctoMap-style log-odds constants
L_HIT = 0.85                 # per ray that ENDS in the voxel (surface hit)
L_MISS = -0.85               # per frame whose rays pass THROUGH the voxel (free space)
L_MIN, L_MAX = -2.0, 3.5
L_OCC_THRESH = 0.85          # log-odds above  this = occupied (p ~ 0.7)
L_FREE_THRESH = -0.85        # log-odds below this = free; one clean pass-through is enough

DEFAULT_RES = 0.005          # 5 mm voxels
DEFAULT_HEIGHT = 0.5         # how far above the tabletop the grid extends (m)
DEFAULT_MARGIN = 0.10        # grid extends this far beyond the table edge (m)
RAY_PIXEL_STRIDE = 2         # use every Nth depth pixel for ray-casting
MAX_RAY_M = 3.0              # ignore depth beyond this


class VoxelBelief:
    def __init__(self, origin, dims, res=DEFAULT_RES):
        """origin: world xyz of voxel (0,0,0)'s corner. dims: (nx, ny, nz)."""
        self.origin = np.asarray(origin, dtype=np.float64)
        self.dims = tuple(int(d) for d in dims)
        self.res = float(res)
        self.log_odds = np.zeros(self.dims, dtype=np.float32)
        self.observed = np.zeros(self.dims, dtype=bool)
        self.n_integrated = 0

    # ---------------------------------------------------------------- setup
    @classmethod
    def around_table(cls, table_ctr, table_radius, table_top_z,
                     res=DEFAULT_RES, height=DEFAULT_HEIGHT, margin=DEFAULT_MARGIN,
                     below=0.02):
        """Grid covering the tabletop disc (+margin) and `height` above it.
        Starts `below` metres under the top surface so the table surface
        itself is inside the grid and shows up as occupied."""
        ctr = np.asarray(table_ctr, dtype=np.float64)
        r = float(table_radius) + margin
        origin = np.array([ctr[0] - r, ctr[1] - r, float(table_top_z) - below])
        nx = ny = int(np.ceil(2 * r / res))
        nz = int(np.ceil((height + below) / res))
        g = cls(origin, (nx, ny, nz), res)
        print(f"[voxel] grid {nx}x{ny}x{nz} = {nx*ny*nz/1e6:.1f}M voxels @ {res*1000:.0f} mm, "
              f"origin {np.round(origin, 3)}, {g.nbytes()/1e6:.0f} MB")
        return g

    def nbytes(self):
        return self.log_odds.nbytes + self.observed.nbytes

    # ---------------------------------------------------------------- indexing
    def world_to_idx(self, pts):
        """(N,3) world points -> (N,3) int voxel indices (may be out of range)."""
        return np.floor((np.asarray(pts, dtype=np.float64) - self.origin) / self.res).astype(np.int32)

    def idx_to_world(self, idx):
        """(N,3) indices -> world xyz of voxel centres."""
        return self.origin + (np.asarray(idx, dtype=np.float64) + 0.5) * self.res

    def in_bounds(self, idx):
        idx = np.asarray(idx)
        return np.all((idx >= 0) & (idx < np.array(self.dims)), axis=-1)

    def _flat(self, idx):
        nx, ny, nz = self.dims
        return (idx[:, 0].astype(np.int64) * ny + idx[:, 1]) * nz + idx[:, 2]

    # ---------------------------------------------------------------- integration
    def integrate_depth(self, depth, cam_to_world, fx, fy, cx, cy,
                        stride=RAY_PIXEL_STRIDE, max_range=MAX_RAY_M, mask=None):
        """Fuse one depth frame.

        depth        HxW float32 metres, 0/nan = invalid
        cam_to_world 4x4, OpenCV optical frame -> world (the matrix the
                     capture step already stores per view)
        mask         optional HxW bool: only these pixels cast rays
        """
        h, w = depth.shape
        ys, xs = np.mgrid[0:h:stride, 0:w:stride]
        ys, xs = ys.ravel(), xs.ravel()
        if mask is not None:
            keep = mask[ys, xs]
            ys, xs = ys[keep], xs[keep]
        z = depth[ys, xs]
        ok = np.isfinite(z) & (z > 0.05) & (z < max_range)
        xs, ys, z = xs[ok], ys[ok], z[ok]
        if len(z) == 0:
            return 0
        # camera-frame endpoints, then world
        pc = np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], axis=1)
        T = np.asarray(cam_to_world, dtype=np.float64)
        ends = pc @ T[:3, :3].T + T[:3, 3]
        cam = T[:3, 3]
        self._cast(cam, ends)
        self.n_integrated += 1
        return len(z)

    def _cast(self, cam_pos, ends):
        """Mark free space along each ray and occupied at each endpoint.

        Samples along each ray at ~0.9 voxel steps (small enough not to
        skip a voxel on axis-aligned runs, ~2x fewer samples than a
        half-voxel step) and deduplicates the resulting voxel indices
        before the log-odds update, so a voxel crossed by many rays costs
        one update instead of hundreds. That dedup is where most of the
        speed comes from: np.add.at on raw samples is the slow path.
        """
        cam_pos = np.asarray(cam_pos, dtype=np.float64)
        dirs = ends - cam_pos
        lengths = np.linalg.norm(dirs, axis=1)
        # skip rays whose endpoint is outside the grid AND whose whole
        # segment stays outside its bounding box (cheap slab test): those
        # contribute nothing and dominate the cost on a wide-FOV frame
        gmin = self.origin
        gmax = self.origin + np.array(self.dims) * self.res
        seg_lo = np.minimum(cam_pos[None, :], ends)
        seg_hi = np.maximum(cam_pos[None, :], ends)
        overlaps = np.all((seg_hi >= gmin[None, :]) & (seg_lo <= gmax[None, :]), axis=1)
        good = (lengths > 1e-6) & overlaps
        dirs, lengths, ends = dirs[good], lengths[good], ends[good]
        if len(ends) == 0:
            return
        unit = dirs / lengths[:, None]
        step = self.res * 0.9
        n_steps = int(np.ceil(lengths.max() / step))
        stop = np.maximum(lengths - self.res, 0.0)      # stop short so the surface stays occupied
        dims = np.array(self.dims)

        free_parts = []
        chunk = max(1, 3_000_000 // max(n_steps, 1))
        t_all = np.arange(n_steps, dtype=np.float32) * step
        for s0 in range(0, len(ends), chunk):
            sl = slice(s0, s0 + chunk)
            valid = t_all[None, :] < stop[sl, None]
            pts = cam_pos[None, None, :] + unit[sl][:, None, :] * t_all[None, :, None]
            idx = np.floor((pts - self.origin) / self.res).astype(np.int32)
            inb = np.all((idx >= 0) & (idx < dims), axis=-1) & valid
            f = self._flat(idx[inb])
            free_parts.append(np.unique(f))             # dedup within the chunk
            del pts, idx, inb, valid, f
        free_flat = np.unique(np.concatenate(free_parts)) if free_parts else np.empty(0, np.int64)

        hit_idx = np.floor((ends - self.origin) / self.res).astype(np.int32)
        hit_in = np.all((hit_idx >= 0) & (hit_idx < dims), axis=-1)
        hit_flat, hit_counts = np.unique(self._flat(hit_idx[hit_in]), return_counts=True)

        lo = self.log_odds.reshape(-1)
        obs = self.observed.reshape(-1)
        # one miss per voxel per frame (not per ray): a voxel crossed by
        # 300 rays of the same frame is one observation of free space,
        # otherwise near voxels saturate instantly and can never be
        # corrected when something moves into them.
        if len(free_flat):
            lo[free_flat] += L_MISS
            obs[free_flat] = True
        if len(hit_flat):
            lo[hit_flat] += L_HIT * np.minimum(hit_counts, 3)   # a surface hit by several rays is stronger evidence
            obs[hit_flat] = True
        np.clip(lo, L_MIN, L_MAX, out=lo)

    # ---------------------------------------------------------------- queries
    def occupied_mask(self):
        return self.log_odds > L_OCC_THRESH

    def free_mask(self):
        return (self.log_odds <= L_FREE_THRESH) & self.observed

    def unknown_mask(self):
        return ~self.observed

    def stats(self):
        total = self.log_odds.size
        occ = int(self.occupied_mask().sum()); free = int(self.free_mask().sum())
        unk = int(self.unknown_mask().sum())
        return {"views": self.n_integrated, "voxels": total,
                "occupied": occ, "free": free, "unknown": unk,
                "observed_frac": round(1 - unk / total, 3)}

    def occupied_points(self):
        return self.idx_to_world(np.argwhere(self.occupied_mask()))

    def frontier_near(self, center_xyz, radius=0.25, exclude_below_z=None):
        """UNKNOWN voxels that are ADJACENT TO KNOWN FREE SPACE -- i.e. the
        ones a camera could actually reach and resolve.

        Plain unknown_near() also returns unknown voxels sealed inside an
        object's solid interior or under the table, which no viewpoint can
        ever observe. Those never clear, so they inflate every candidate's
        predicted information gain equally and make a gain-based stopping
        rule useless (observed: predicted gain stuck at ~36% while actual
        newly-observed volume had fallen to 1% per view)."""
        pts = self.unknown_near(center_xyz, radius, exclude_below_z)
        if len(pts) == 0:
            return pts
        idx = self.world_to_idx(pts)
        free = self.free_mask()
        dims = np.array(self.dims)
        touching = np.zeros(len(idx), dtype=bool)
        for off in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nb = idx + np.array(off)
            ok = np.all((nb >= 0) & (nb < dims), axis=1)
            sel = np.where(ok)[0]
            if len(sel):
                touching[sel] |= free[nb[sel, 0], nb[sel, 1], nb[sel, 2]]
        return pts[touching]

    def unknown_near(self, center_xyz, radius=0.25, exclude_below_z=None):
        """World xyz of UNKNOWN voxel centres within `radius` of a point --
        the frontier a next-best-view step should aim at. exclude_below_z
        drops voxels under the tabletop (never observable from above)."""
        c = np.asarray(center_xyz, dtype=np.float64)
        r_vox = int(np.ceil(radius / self.res))
        ci = self.world_to_idx(c[None, :])[0]
        lo = np.maximum(ci - r_vox, 0)
        hi = np.minimum(ci + r_vox + 1, np.array(self.dims))
        if np.any(lo >= hi):
            return np.empty((0, 3))
        sub = self.observed[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        idx = np.argwhere(~sub) + lo
        pts = self.idx_to_world(idx)
        d = np.linalg.norm(pts - c, axis=1)
        keep = d <= radius
        if exclude_below_z is not None:
            keep &= pts[:, 2] >= exclude_below_z
        return pts[keep]

    def visible_from(self, cam_pos, targets, look_at=None, half_fov_deg=32.0, max_range=0.9,
                     occ_thresh=None, surface_normals=None, min_incidence_cos=0.34):
        """Which of `targets` (N,3 world points) this camera would actually
        observe: inside the FOV cone, within range, and with a clear line of
        sight through the grid (no OCCUPIED voxel between camera and point).

        Marches each ray through the grid at ~0.9 voxel steps and checks the
        occupancy array directly, instead of testing distance to a sparse
        set of surface points -- a shell of occupied voxels has gaps that a
        perpendicular-distance test slips through, which made every pose
        look equally unoccluded."""
        cam = np.asarray(cam_pos, dtype=np.float64)
        tgt = np.asarray(targets, dtype=np.float64)
        if len(tgt) == 0:
            return np.zeros(0, dtype=bool)
        # Aim: the pose the planner will actually execute points the camera at
        # the TARGET, so visibility must be judged from that aim. Aiming at the
        # frontier's own centroid instead makes every candidate "see" nearly
        # all of it (observed: predicted gain stuck at 92-96% for every pose).
        if look_at is None:
            look = np.asarray(self._look_dir(cam_pos, targets), dtype=np.float64)
        else:
            d = np.asarray(look_at, dtype=np.float64) - cam
            look = d / max(np.linalg.norm(d), 1e-9)
        to_t = tgt - cam[None, :]
        rng = np.linalg.norm(to_t, axis=1)
        rng_safe = np.maximum(rng, 1e-9)
        dirn = to_t / rng_safe[:, None]
        vis = (rng > 0.05) & (rng < max_range) & ((dirn @ look) > np.cos(np.deg2rad(half_fov_deg)))
        if not vis.any():
            return vis
        occ = self.occupied_mask() if occ_thresh is None else (self.log_odds > occ_thresh)
        # Grazing-angle rejection: a frontier voxel can have clear line of
        # sight yet sit on a surface the camera meets almost edge-on, where
        # real depth is unreliable or dropped entirely. Counting those as
        # observable is why predicted gain stayed optimistic and degraded
        # view after view (verify: 75% -> 31% as the easy voxels ran out).
        # Approximate each frontier voxel's surface normal by the direction
        # from the nearest occupied neighbourhood, and require the viewing
        # ray to meet it within ~70 deg.
        if min_incidence_cos is not None and vis.any():
            nrm = surface_normals if surface_normals is not None else self._frontier_normals(tgt)
            if nrm is not None:
                cosinc = np.abs(np.sum(-dirn * nrm, axis=1))
                undefined = ~np.isfinite(cosinc)
                vis &= (cosinc >= min_incidence_cos) | undefined
        dims = np.array(self.dims)
        step = self.res * 0.9
        idxs = np.where(vis)[0]
        n_steps = int(np.ceil((rng[idxs].max()) / step))
        t_all = np.arange(1, n_steps + 1, dtype=np.float32) * step
        chunk = max(1, 2_000_000 // max(n_steps, 1))
        for i in range(0, len(idxs), chunk):
            sel = idxs[i:i + chunk]
            # stop one voxel short of the target so the target voxel itself doesn't block
            stop = (rng[sel] - self.res)[:, None]
            pts = cam[None, None, :] + dirn[sel][:, None, :] * t_all[None, :, None]
            vidx = np.floor((pts - self.origin) / self.res).astype(np.int32)
            inb = np.all((vidx >= 0) & (vidx < dims), axis=-1) & (t_all[None, :] < stop)
            hit = np.zeros(len(sel), dtype=bool)
            fl = np.where(inb)
            if len(fl[0]):
                occ_hit = occ[vidx[fl[0], fl[1], 0], vidx[fl[0], fl[1], 1], vidx[fl[0], fl[1], 2]]
                np.logical_or.at(hit, fl[0], occ_hit)
            vis[sel[hit]] = False
            del pts, vidx, inb
        return vis

    def _frontier_normals(self, pts, radius_vox=3):
        """Approximate outward normal at each point from the local occupied
        mass: the direction pointing AWAY from nearby occupied voxels. Rows
        are NaN where no occupied voxel is close enough to define one."""
        idx = self.world_to_idx(pts)
        dims = np.array(self.dims)
        occ = self.occupied_mask()
        acc = np.zeros((len(pts), 3), dtype=np.float64)
        cnt = np.zeros(len(pts), dtype=np.int32)
        r = int(radius_vox)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if dx == dy == dz == 0:
                        continue
                    nb = idx + np.array([dx, dy, dz])
                    ok = np.all((nb >= 0) & (nb < dims), axis=1)
                    sel = np.where(ok)[0]
                    if not len(sel):
                        continue
                    hit = occ[nb[sel, 0], nb[sel, 1], nb[sel, 2]]
                    hs = sel[hit]
                    if len(hs):
                        acc[hs] -= np.array([dx, dy, dz], dtype=np.float64)
                        cnt[hs] += 1
        n = np.linalg.norm(acc, axis=1)
        good = (cnt > 0) & (n > 1e-9)
        out = np.full((len(pts), 3), np.nan)
        out[good] = acc[good] / n[good][:, None]
        return out

    @staticmethod
    def _look_dir(cam_pos, targets):
        c = np.asarray(cam_pos, dtype=np.float64)
        t = np.asarray(targets, dtype=np.float64)
        d = t.mean(axis=0) - c
        return d / max(np.linalg.norm(d), 1e-9)

    # ---------------------------------------------------------------- io
    def save_ply(self, path, what="occupied"):
        """Voxel centres as a coloured .ply for the Open3D viewer.
        what: 'occupied' | 'unknown' | 'both'."""
        parts = []
        if what in ("occupied", "both"):
            p = self.occupied_points()
            parts.append((p, np.tile(np.array([[200, 60, 60]], np.uint8), (len(p), 1))))
        if what in ("unknown", "both"):
            p = self.idx_to_world(np.argwhere(self.unknown_mask()))
            parts.append((p, np.tile(np.array([[90, 90, 90]], np.uint8), (len(p), 1))))
        pts = np.vstack([p for p, _ in parts]) if parts else np.empty((0, 3))
        cols = np.vstack([c for _, c in parts]) if parts else np.empty((0, 3), np.uint8)
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for (x, y, z), (r, g, b) in zip(pts, cols):
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")
        return len(pts)

    def save_npz(self, path):
        np.savez_compressed(path, log_odds=self.log_odds, observed=self.observed,
                            origin=self.origin, dims=np.array(self.dims), res=self.res,
                            n_integrated=self.n_integrated)

    @classmethod
    def load_npz(cls, path):
        d = np.load(path)
        g = cls(d["origin"], tuple(d["dims"]), float(d["res"]))
        g.log_odds = d["log_odds"]; g.observed = d["observed"]
        g.n_integrated = int(d["n_integrated"])
        return g
