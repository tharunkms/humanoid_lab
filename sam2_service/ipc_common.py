"""
ipc_common.py
--------------
Shared wire-protocol helpers for the two-process SAM2 click-to-track pipeline.

Process A (rgbd_segment_gui.py)  <---- ZeroMQ REQ/REP, tcp://localhost:5555 ---->  Process B (sam2_service.py)

Message shape (both directions) is a 2-part ZMQ multipart message:
    part 0: JSON metadata (utf-8 encoded)
    part 1: binary payload (JPEG frame bytes, or PNG mask bytes, or empty)

Keeping this in one file means both processes agree on the exact same framing
without needing to share a virtualenv (this file has zero heavy deps -- only
numpy/cv2/json/zmq, all of which are trivial to install in either env).
"""

import json
import numpy as np
import cv2

# ---- request "type" values sent by Process A -----------------------------
REQ_CLICK = "click"       # new object selection (positive or negative point)
REQ_TRACK = "track"       # propagate tracking to a new frame, no new prompt
REQ_RESET = "reset"       # clear tracking state, go back to idle
REQ_CLASSIFY = "classify"  # identify object category + primary geometric shape from a crop
REQ_GENERATE_PC = "generate_pointcloud"  # fuse a captured pose session into one point cloud
REQ_VIEW_PC = "view_pointcloud"          # open an interactive native viewer for a fused cloud
REQ_DETECT_ALL = "detect_all"     # auto-segment + classify every prop in the frame (no prompt)
REQ_SELECT_MASK = "select_mask"   # start tracking one of the candidates returned by detect_all

# ---- response "status" values sent by Process B ---------------------------
STATUS_OK = "ok"
STATUS_NO_OBJECT = "no_object"     # idle, nothing being tracked, no mask
STATUS_ERROR = "error"             # inference failed; message has details

# ---- what the binary part of a response contains --------------------------
PAYLOAD_MASK = "mask"              # single bool mask PNG (0/255)  -- the default
PAYLOAD_LABEL_MAP = "label_map"    # uint8 PNG, pixel value = candidate id (0 = none)


def encode_frame(frame_bgr, jpeg_quality=85):
    """numpy BGR uint8 image -> JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise ValueError("JPEG encode failed")
    return buf.tobytes()


def decode_frame(jpeg_bytes):
    """JPEG bytes -> numpy BGR uint8 image."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("JPEG decode failed")
    return frame


def encode_mask(mask_bool):
    """bool/0-1 HxW mask -> PNG bytes (single channel, 0/255)."""
    mask_u8 = (mask_bool.astype(np.uint8)) * 255
    ok, buf = cv2.imencode(".png", mask_u8)
    if not ok:
        raise ValueError("PNG encode failed")
    return buf.tobytes()


def decode_mask(png_bytes):
    """PNG bytes -> bool HxW mask."""
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    mask_u8 = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask_u8 is None:
        raise ValueError("PNG decode failed")
    return mask_u8 > 127


def encode_label_map(label_u8):
    """HxW uint8 array (pixel value = candidate id, 0 = none) -> PNG bytes.
    PNG is lossless, so ids survive the round trip exactly."""
    ok, buf = cv2.imencode(".png", np.asarray(label_u8, dtype=np.uint8))
    if not ok:
        raise ValueError("PNG encode failed")
    return buf.tobytes()


def decode_label_map(png_bytes):
    """PNG bytes -> HxW uint8 label map."""
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    lm = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if lm is None:
        raise ValueError("PNG decode failed")
    return lm


def build_request(req_type, frame_bgr=None, x=None, y=None, label=None, jpeg_quality=85, **extra_meta):
    """Pack a request into a (meta_bytes, payload_bytes) tuple for zmq send_multipart.
    extra_meta lets path-based requests (generate/view pointcloud) carry a
    session directory string etc. without needing frame bytes at all."""
    meta = {"type": req_type}
    if x is not None:
        meta["x"] = int(x)
    if y is not None:
        meta["y"] = int(y)
    if label is not None:
        meta["label"] = int(label)  # 1 = positive point, 0 = negative point
    meta.update(extra_meta)
    payload = encode_frame(frame_bgr, jpeg_quality) if frame_bgr is not None else b""
    return json.dumps(meta).encode("utf-8"), payload


def parse_request(meta_bytes, payload_bytes):
    meta = json.loads(meta_bytes.decode("utf-8"))
    frame = decode_frame(payload_bytes) if payload_bytes else None
    return meta, frame


def build_response(status, mask_bool=None, message="", label_map=None, **extra_meta):
    """extra_meta lets classify responses carry label/shape/confidence
    alongside status/message, without needing a whole separate payload
    format. The binary part is EITHER a single bool mask (default) OR a
    uint8 label map (detect_all) -- meta["payload"] says which, so an
    older Process A that never sends detect_all sees no difference."""
    meta = {"status": status, "message": message}
    meta.update(extra_meta)
    if label_map is not None:
        meta["payload"] = PAYLOAD_LABEL_MAP
        payload = encode_label_map(label_map)
    else:
        meta["payload"] = PAYLOAD_MASK
        payload = encode_mask(mask_bool) if mask_bool is not None else b""
    return json.dumps(meta).encode("utf-8"), payload


def parse_response(meta_bytes, payload_bytes):
    """Returns (meta, payload_array). payload_array is a bool mask for the
    default payload kind, or a uint8 label map when meta["payload"] ==
    PAYLOAD_LABEL_MAP, or None when the binary part is empty."""
    meta = json.loads(meta_bytes.decode("utf-8"))
    if not payload_bytes:
        return meta, None
    if meta.get("payload") == PAYLOAD_LABEL_MAP:
        return meta, decode_label_map(payload_bytes)
    return meta, decode_mask(payload_bytes)
