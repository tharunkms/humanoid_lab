"""
depth_noise.py
--------------
Turn Isaac Sim's ideal pinhole depth into something that behaves like a
real Intel RealSense D435i. Pure numpy + OpenCV, so it runs both inside
Isaac Sim's Python (Process A, per frame) and on the host (offline tests).

What a real stereo depth camera does that the simulator doesn't:

1. Range-dependent noise. Stereo depth comes from disparity d = f*b/z.
   A fixed sub-pixel disparity error sigma_d therefore becomes a depth
   error sigma_z = z^2 / (f*b) * sigma_d -- it grows with the SQUARE of
   distance. (D435: baseline b = 50 mm, f ~ 674 px at 1280x720.)
2. Angle-dependent noise. On surfaces seen at a grazing angle the stereo
   match is smeared along the surface, so noise grows roughly with
   1/cos(incidence angle), and matching fails entirely beyond ~70-80 deg.
3. Spatially correlated noise. Real depth noise is blotchy, not white per
   pixel, because block matching shares neighbourhoods.
4. Disparity quantisation. Disparity is stored with finite sub-pixel
   resolution, which shows as depth "banding" at range.
5. Invalid pixels. Matching fails and the sensor reports 0 on:
     - depth discontinuities (occlusion edges) -- plus "flying pixels":
       edge pixels that get a depth somewhere BETWEEN foreground and
       background,
     - dark / IR-absorbing surfaces (too little return signal),
     - saturated / specular spots,
     - the left image border strip the right imager cannot see
       (width = disparity = f*b/z pixels),
     - anything closer than the minimum range (~0.28 m at 720p) or beyond
       the useful maximum,
     - a sprinkle of random holes.

Output convention: float32 metres, 0.0 = invalid (same as RealSense and
as the 16-bit PNGs the pipeline already saves). Everything downstream
already treats z <= 0.05 as invalid, so no other code has to change.

All strengths live in DEFAULT_CFG. `severity` scales the whole thing
(0 = off, 1 = nominal D435i, 2 = pessimistic) without touching individual
knobs.
"""
import numpy as np
import cv2

DEFAULT_CFG = dict(
    enabled=True,
    severity=1.0,               # global multiplier on noise std and dropout probabilities
    baseline_m=0.050,           # D435 stereo baseline
    disp_sigma_px=0.08,         # sub-pixel disparity noise std (0.05 good unit, 0.15 poor)
    disp_quant_px=1.0 / 32.0,   # disparity resolution (RealSense reports 1/32 px)
    angle_gain=1.0,             # noise multiplier = 1 + gain * (1/cos(theta) - 1)
    max_angle_mult=4.0,         # cap on that multiplier
    angle_drop_start_deg=65.0,  # dropout probability ramps from 0 here ...
    angle_drop_full_deg=85.0,   # ... to angle_drop_p here
    angle_drop_p=0.9,
    min_z=0.28, max_z=4.0,      # valid range (720p min-Z per datasheet)
    edge_thresh_m=0.03,         # neighbouring-pixel depth jump that counts as an occlusion edge
    edge_band_px=2,             # how far the failure band extends from the edge
    edge_drop_p=0.6,            # dropout probability inside the band
    flying_p=0.15,              # surviving band pixels that get an in-between ("flying") depth
    dark_lum_zero=90.0,         # luminance ABOVE which darkness causes no extra dropout
    dark_lum_full=10.0,         # luminance AT/BELOW which dropout reaches its max (near-true-black only)
    dark_drop_p=0.55,           # dropout probability for a near-true-black surface (was 0.7, saturating
                                 # for anything <=40 -- that treated an ordinary dark-grey/black plastic
                                 # object the same as a zero-albedo IR absorber, which real D435i does not)
    max_drop_p=0.85,            # HARD CAP on total per-pixel dropout probability, after combining every
                                 # cause and applying `severity`. Real D435i depth on even near-black
                                 # (near-IR-absorbing) surfaces is sparse, not always 100% empty -- and a
                                 # detector that gets exactly 0 valid points can't tell "sensor failure"
                                 # from "nothing is there", so a real object should never fully vanish
                                 # from a single noise pass. Observed: an all-dark mug hit 0/N valid
                                 # points at severity=1.0 because dark (0.7) stacked with edge-band (0.6)
                                 # and random-hole dropout on most of its pixels; capped runs keep a
                                 # sparse-but-nonzero cloud on the same object instead.
    spec_lum=245.0,             # near-saturated pixels
    spec_drop_p=0.5,
    left_occlusion=True,        # invalid strip on the left border, width f*b/z px
    coarse_frac=0.5,            # share of noise variance that is spatially correlated
    coarse_sigma_px=6.0,        # correlation length of that share
    random_drop_p=0.002,        # sparse random holes
    hole_grow_px=1,             # dilate holes so they look like blobs, not single pixels
)


def _cfg(cfg):
    c = dict(DEFAULT_CFG)
    if cfg:
        c.update(cfg)
    return c


def incidence_cos(depth, fx, fy, cx, cy, scale=4):
    """cos(angle between viewing ray and surface normal), per pixel, in [0,1].
    Computed on a `scale`x downsampled image (the angle map is smooth, and
    the full-resolution version costs ~450 ms) and upsampled back.
    Normals from central differences of the unprojected surface; invalid
    pixels get 1 (treated as head-on)."""
    h, w = depth.shape
    z = depth.astype(np.float32)
    ok_full = np.isfinite(z) & (z > 0)
    if scale > 1:
        zs = cv2.resize(np.where(ok_full, z, 0.0).astype(np.float32), (w // scale, h // scale), interpolation=cv2.INTER_NEAREST)
        fx_s, fy_s, cx_s, cy_s = fx / scale, fy / scale, cx / scale, cy / scale
    else:
        zs, fx_s, fy_s, cx_s, cy_s = z, fx, fy, cx, cy
    hs, ws = zs.shape
    ok = zs > 0
    u, v = np.meshgrid(np.arange(ws, dtype=np.float32), np.arange(hs, dtype=np.float32))
    X = (u - cx_s) * zs / fx_s
    Y = (v - cy_s) * zs / fy_s
    # central differences via shifted slices (much cheaper than np.gradient on a 3-channel array)
    def dd(a, axis):
        g = np.empty_like(a)
        if axis == 1:
            g[:, 1:-1] = 0.5 * (a[:, 2:] - a[:, :-2]); g[:, 0] = a[:, 1] - a[:, 0]; g[:, -1] = a[:, -1] - a[:, -2]
        else:
            g[1:-1, :] = 0.5 * (a[2:, :] - a[:-2, :]); g[0, :] = a[1, :] - a[0, :]; g[-1, :] = a[-1, :] - a[-2, :]
        return g
    Xu, Yu, Zu = dd(X, 1), dd(Y, 1), dd(zs, 1)
    Xv, Yv, Zv = dd(X, 0), dd(Y, 0), dd(zs, 0)
    nx = Yu * Zv - Zu * Yv
    ny = Zu * Xv - Xu * Zv
    nz = Xu * Yv - Yu * Xv
    nn = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    pn = np.sqrt(X * X + Y * Y + zs * zs) + 1e-9
    cos = np.abs(-(nx * X + ny * Y + nz * zs)) / (nn * pn)     # |n . (-P/|P|)|
    cos = np.where(ok & np.isfinite(cos), cos, 1.0).astype(np.float32)
    # a downsampled pixel next to an invalid one has a bogus normal; treat it as head-on
    if scale > 1:
        cos = cv2.resize(cos, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(cos, 0.0, 1.0)


def _correlated_noise(shape, rng, coarse_frac, coarse_sigma_px):
    """Unit-variance Gaussian field: (1-frac) white + frac low-pass."""
    white = rng.standard_normal(shape, dtype=np.float32)
    if coarse_frac <= 0:
        return white
    coarse = rng.standard_normal(shape, dtype=np.float32)
    k = int(2 * round(3 * coarse_sigma_px) + 1)
    coarse = cv2.GaussianBlur(coarse, (k, k), coarse_sigma_px)
    coarse /= max(float(coarse.std()), 1e-6)      # blur shrinks the std; restore unit variance
    return np.sqrt(1.0 - coarse_frac) * white + np.sqrt(coarse_frac) * coarse


def apply_d435i_noise(depth, rgb_bgr, fx, fy, cx, cy, cfg=None, rng=None, return_info=False):
    """
    depth   : HxW float32 metres, ideal (0/nan/inf = invalid)
    rgb_bgr : HxWx3 uint8 aligned colour image (used for dark/specular dropout); may be None
    fx, fy, cx, cy : intrinsics of THIS depth image
    Returns HxW float32 metres with 0.0 = invalid (and an info dict if return_info).
    """
    c = _cfg(cfg)
    if not c["enabled"] or c["severity"] <= 0:
        out = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return (out, {}) if return_info else out
    rng = rng or np.random.default_rng()
    sev = float(c["severity"])
    h, w = depth.shape
    z = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    valid = z > 0
    fb = fx * c["baseline_m"]
    info = {}

    # ---- 1-3: range- and angle-dependent, spatially correlated noise ----
    cos_t = incidence_cos(z, fx, fy, cx, cy)
    sigma_z = (z * z) / fb * c["disp_sigma_px"] * sev
    ang_mult = np.minimum(1.0 + c["angle_gain"] * (1.0 / np.maximum(cos_t, 1e-3) - 1.0), c["max_angle_mult"])
    n = _correlated_noise((h, w), rng, c["coarse_frac"], c["coarse_sigma_px"])
    zn = z + sigma_z * ang_mult * n
    info["sigma_z_mm_at_1m"] = 1000.0 * (1.0 / fb) * c["disp_sigma_px"] * sev
    info["sigma_z_mm_at_2m"] = 4.0 * info["sigma_z_mm_at_1m"]

    # ---- 4: disparity quantisation ----
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(zn > 0, fb / np.maximum(zn, 1e-6), 0.0)
        q = c["disp_quant_px"]
        dq = np.round(d / q) * q
        zn = np.where(dq > 0, fb / np.maximum(dq, 1e-9), 0.0).astype(np.float32)

    # ---- 5: dropout probability map ----
    p = np.zeros((h, w), dtype=np.float32)
    # grazing angles
    th = np.degrees(np.arccos(np.clip(cos_t, 0, 1)))
    ramp = np.clip((th - c["angle_drop_start_deg"]) / max(c["angle_drop_full_deg"] - c["angle_drop_start_deg"], 1e-3), 0, 1)
    p = np.maximum(p, ramp * c["angle_drop_p"])
    # occlusion edges (+ flying pixels)
    gx = np.abs(np.diff(z, axis=1, prepend=z[:, :1]))
    gy = np.abs(np.diff(z, axis=0, prepend=z[:1, :]))
    edge = ((gx > c["edge_thresh_m"]) | (gy > c["edge_thresh_m"])) & valid
    if c["edge_band_px"] > 0:
        kb = 2 * int(c["edge_band_px"]) + 1
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((kb, kb), np.uint8)).astype(bool)
    p = np.where(edge, np.maximum(p, c["edge_drop_p"]), p)
    # IR-dark and specular surfaces (from the aligned colour image)
    if rgb_bgr is not None and rgb_bgr.shape[:2] == (h, w):
        lum = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        span = max(c["dark_lum_zero"] - c["dark_lum_full"], 1e-3)
        dark = np.clip((c["dark_lum_zero"] - lum) / span, 0, 1)   # 0 at dark_lum_zero, 1 at dark_lum_full and below
        p = np.maximum(p, dark * c["dark_drop_p"])
        p = np.where(lum >= c["spec_lum"], np.maximum(p, c["spec_drop_p"]), p)
    # random holes
    p = np.maximum(p, c["random_drop_p"])
    p = np.clip(p * sev, 0, c["max_drop_p"])
    drop = (rng.random((h, w), dtype=np.float32) < p) & valid
    # flying pixels: edge-band survivors take a value between local min and max
    if c["flying_p"] > 0:
        fly = edge & ~drop & valid & (rng.random((h, w), dtype=np.float32) < c["flying_p"] * sev)
        if fly.any():
            k5 = np.ones((5, 5), np.uint8)
            zpos = np.where(valid, z, np.inf).astype(np.float32)
            zmin = cv2.erode(zpos, k5)
            zmax = cv2.dilate(np.where(valid, z, 0).astype(np.float32), k5)
            a = rng.random((h, w), dtype=np.float32)
            ok_mix = np.isfinite(zmin) & (zmax > 0)
            mix = zmin + a * np.where(ok_mix, zmax - zmin, 0.0)   # avoid inf-inf/nan feeding the multiply
            zn = np.where(fly & ok_mix, mix, zn)
        info["flying_pixels"] = int(fly.sum())
    if c["hole_grow_px"] > 0:
        kg = 2 * int(c["hole_grow_px"]) + 1
        drop = cv2.dilate(drop.astype(np.uint8), np.ones((kg, kg), np.uint8)).astype(bool)
    # range limits + left occlusion strip
    drop |= valid & ((zn < c["min_z"]) | (zn > c["max_z"]))
    if c["left_occlusion"]:
        u = np.arange(w, dtype=np.float32)[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            strip = valid & (u < np.where(z > 0, fb / np.maximum(z, 1e-6), 0.0))
        drop |= strip
    zn = np.where(valid & ~drop, zn, 0.0).astype(np.float32)

    info.update({
        "valid_in": int(valid.sum()), "valid_out": int((zn > 0).sum()),
        "dropped_frac": float(1.0 - (zn > 0).sum() / max(valid.sum(), 1)),
        "edge_px": int(edge.sum()),
    })
    return (zn, info) if return_info else zn


# ------------------------------------------------------------------ CLI test
def _main():
    import sys, os, json, argparse
    ap = argparse.ArgumentParser(description="Apply the D435i noise model to a saved session view and report/visualise.")
    ap.add_argument("session_dir")
    ap.add_argument("--pose", type=int, default=0)
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.session_dir, "session_meta.json")))
    K = meta["intrinsics"]
    depth_png = cv2.imread(os.path.join(a.session_dir, f"pose_{a.pose:02d}_depth.png"), cv2.IMREAD_UNCHANGED)
    rgb = cv2.imread(os.path.join(a.session_dir, f"pose_{a.pose:02d}_rgb.png"), cv2.IMREAD_COLOR)
    mask = cv2.imread(os.path.join(a.session_dir, f"pose_{a.pose:02d}_mask.png"), cv2.IMREAD_GRAYSCALE)
    depth = depth_png.astype(np.float32) / 1000.0
    noisy, info = apply_d435i_noise(depth, rgb, K["fx"], K["fy"], K["cx"], K["cy"],
                                    cfg={"severity": a.severity}, rng=np.random.default_rng(a.seed), return_info=True)
    print("model:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()})
    # per-depth-bin noise on pixels that survived
    both = (depth > 0) & (noisy > 0)
    err = noisy - depth
    for lo, hi in [(0.2, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0)]:
        sel = both & (depth >= lo) & (depth < hi)
        if sel.sum() > 100:
            print(f"  z in [{lo:.1f},{hi:.1f}) m: n={int(sel.sum()):7d}  err std={1000*err[sel].std():6.1f} mm  "
                  f"|err| p95={1000*np.percentile(np.abs(err[sel]), 95):6.1f} mm")
    if mask is not None:
        m = mask > 127
        print(f"  object mask: {int(m.sum())} px, valid depth before={int((depth[m] > 0).sum())} after={int((noisy[m] > 0).sum())} "
              f"({100 * (noisy[m] > 0).sum() / max((depth[m] > 0).sum(), 1):.0f}% kept)")
    # visualisation: clean | noisy | error, all colour-mapped over the same range
    zmax = float(np.percentile(depth[depth > 0], 99)) if (depth > 0).any() else 1.0
    def cmap(d, vmin=0.0, vmax=zmax):
        x = np.clip((d - vmin) / max(vmax - vmin, 1e-6), 0, 1)
        img = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        img[d <= 0] = 0
        return img
    e = np.zeros_like(depth); e[both] = err[both]
    emax = max(0.02, float(np.percentile(np.abs(e[both]), 99))) if both.any() else 0.02
    ev = cv2.applyColorMap(np.clip((e / emax + 1) / 2 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_COOLWARM)
    ev[~both] = 0
    out = np.concatenate([cmap(depth), cmap(noisy), ev], axis=1)
    cv2.putText(out, "ideal depth", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(out, f"D435i model (severity {a.severity})", (depth.shape[1] + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(out, f"error, +-{1000 * emax:.0f} mm", (2 * depth.shape[1] + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    out_path = os.path.join(a.session_dir, f"pose_{a.pose:02d}_depth_noise_test.png")
    cv2.imwrite(out_path, out)
    cv2.imwrite(os.path.join(a.session_dir, f"pose_{a.pose:02d}_depth_noisy.png"),
                np.clip(noisy * 1000.0, 0, 65535).astype(np.uint16))
    print("  wrote", out_path)


if __name__ == "__main__":
    _main()
