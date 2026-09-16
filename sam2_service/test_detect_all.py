#!/usr/bin/env python3
"""
test_detect_all.py -- exercise REQ_DETECT_ALL + REQ_SELECT_MASK against a
running sam2_service.py, using a saved RGB frame instead of the GUI.

Run on the HOST, inside the sam2_service venv, with sam2_service.py already
listening:

    python3 test_detect_all.py /path/to/pose_00_rgb.png [--target bottle]

--target <category> additionally scores every candidate against that CLIP
category (this is what the GUI's "Search for" mode uses).

Writes <input>_detect_overlay.png next to the input: every candidate mask
tinted a different colour, with "id:label (shape conf)" drawn at its bbox.
Then selects candidate 1 and reports the mask it got back.
"""
import sys
import os
import time
import cv2
import numpy as np
import zmq

from ipc_common import (
    build_request, parse_response,
    REQ_DETECT_ALL, REQ_SELECT_MASK, REQ_RESET,
    STATUS_OK, PAYLOAD_LABEL_MAP,
)

ZMQ_ADDR = "tcp://localhost:5555"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    img_path = sys.argv[1]
    target = None
    if "--target" in sys.argv:
        target = sys.argv[sys.argv.index("--target") + 1]
    frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if frame is None:
        sys.exit(f"could not read {img_path}")
    print(f"frame {frame.shape[1]}x{frame.shape[0]}")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 120_000)   # first call loads a second model -- allow time
    sock.connect(ZMQ_ADDR)

    # ---- detect_all ------------------------------------------------------
    t0 = time.time()
    extra = {"target_label": target} if target else {}
    sock.send_multipart(list(build_request(REQ_DETECT_ALL, frame_bgr=frame, **extra)))
    meta, label_map = parse_response(*sock.recv_multipart())
    dt = time.time() - t0
    print(f"detect_all -> status={meta['status']} payload={meta.get('payload')} ({dt:.1f}s)")
    print(f"  message: {meta.get('message')}")
    if meta["status"] != STATUS_OK:
        sys.exit(1)
    assert meta.get("payload") == PAYLOAD_LABEL_MAP, "expected a label map payload"
    cands = meta["candidates"]
    print(f"  {len(cands)} candidates:")
    for c in cands:
        tp = f" target_prob({target})={c['target_prob']:.2f}" if "target_prob" in c else ""
        print(f"    id={c['id']:>2}  {c['label']:<14} shape={c['shape']:<9} conf={c['confidence']:.2f} "
              f"bbox={c['bbox']} area={c['area']} centroid={c['centroid']}{tp}")
    if target and cands:
        if "target_prob" not in cands[0]:
            print(f"  NOTE: '{target}' is not in the CLIP vocabulary -- no target_prob returned")
        else:
            best = max(cands, key=lambda c: c["target_prob"])
            print(f"  best match for '{target}': id={best['id']} ({best['label']}, target_prob={best['target_prob']:.2f})")
    ids_in_map = sorted(int(v) for v in np.unique(label_map) if v != 0)
    print(f"  ids present in label map: {ids_in_map}")

    # ---- overlay -----------------------------------------------------------
    rng = np.random.RandomState(7)
    vis = frame.copy()
    for c in cands:
        colour = tuple(int(v) for v in rng.randint(60, 255, size=3))
        m = label_map == c["id"]
        vis[m] = (0.45 * np.array(colour) + 0.55 * vis[m]).astype(np.uint8)
        x, y, w, h = c["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), colour, 2)
        cv2.putText(vis, f"{c['id']}:{c['label']} ({c['shape']} {c['confidence']:.2f})",
                    (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    out = os.path.splitext(img_path)[0] + "_detect_overlay.png"
    cv2.imwrite(out, vis)
    print(f"  overlay -> {out}")

    if not cands:
        print("no candidates -- nothing to select. Check the overlay/thresholds.")
        return

    # ---- select_mask -------------------------------------------------------
    cid = cands[0]["id"]
    sock.send_multipart(list(build_request(REQ_SELECT_MASK, frame_bgr=frame, candidate_id=cid)))
    meta, mask = parse_response(*sock.recv_multipart())
    print(f"select_mask id={cid} -> status={meta['status']} payload={meta.get('payload')}")
    if meta["status"] == STATUS_OK:
        print(f"  mask pixels={int(mask.sum())} (detect_all reported area={cands[0]['area']})")
    else:
        print(f"  message: {meta.get('message')}")

    # leave the service idle again
    sock.send_multipart(list(build_request(REQ_RESET)))
    sock.recv_multipart()


if __name__ == "__main__":
    main()
