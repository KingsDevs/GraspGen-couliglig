import base64
import json
import os
import time

import cv2
import msgpack
import msgpack_numpy
import numpy as np
import torch
import zmq
from PIL import Image
from sklearn.cluster import DBSCAN
from transformers import Sam3Model, Sam3Processor
import zlib

msgpack_numpy.patch()

# ---------------------------------------------------------------------------
# SAM3 — loaded once at startup
# ---------------------------------------------------------------------------
#
# Unlike the SAM2 server (rpc_server.py), SAM3 is prompted by a TEXT string
# describing the object to pick (e.g. "the red mug") instead of a 2D click
# point or a bounding box. SAM3 returns every instance matching the concept;
# we keep the single highest-scoring instance and feed its mask downstream.
#
# Setup note: SAM3 (released Nov 2025) needs a newer `transformers` than the
# repo's pinned 4.48.3, plus a one-time `huggingface-cli login` to accept the
# gated `facebook/sam3` model terms. That newer `transformers` pulls a newer
# `huggingface_hub` that is INCOMPATIBLE with the ancient `diffusers` GraspGen
# depends on, so this server does NOT load GraspGen in-process. Instead it runs
# in its own venv and FORWARDS the cleaned point cloud to the existing GraspGen
# ZMQ server (client-server/graspgen_server.py) over ZMQ. See requirements.txt.

SAM3_MODEL_ID = os.getenv("SAM3_MODEL_ID", "facebook/sam3")
SAM3_THRESHOLD = float(os.getenv("SAM3_THRESHOLD", "0.5"))
SAM3_MASK_THRESHOLD = float(os.getenv("SAM3_MASK_THRESHOLD", "0.5"))
# Half precision halves the SAM3 weight/activation memory so it has a chance of
# fitting alongside GraspGen on a small (6 GB) GPU. Set SAM3_FP16=0 to load fp32.
SAM3_FP16 = os.getenv("SAM3_FP16", "1").lower() not in {"0", "false", "no"}
_sam3_dtype = torch.float16 if SAM3_FP16 else torch.float32
RPC_PORT = int(os.getenv("RPC_PORT", "5557"))
RPC_VISUALIZE = os.getenv("RPC_VISUALIZE", "1").lower() not in {
    "0",
    "false",
    "no",
}
RPC_MAX_MARKERS = int(os.getenv("RPC_MAX_MARKERS", "1000"))

# GraspGen runs in a SEPARATE process (its own venv). We forward the segmented +
# cleaned cloud to it over ZMQ. Start it with:
#   python client-server/graspgen_server.py \
#       --gripper_config weights/graspgen_robotiq_2f_140.yml --port 5556
GRASPGEN_HOST = os.getenv("GRASPGEN_HOST", "localhost")
GRASPGEN_PORT = int(os.getenv("GRASPGEN_PORT", "5556"))
GRASPGEN_TIMEOUT_MS = int(os.getenv("GRASPGEN_TIMEOUT_MS", "60000"))
GRASPGEN_NUM_GRASPS = int(os.getenv("GRASPGEN_NUM_GRASPS", "100"))
GRASPGEN_TOPK = int(os.getenv("GRASPGEN_TOPK", "20"))
GRASPGEN_MIN_GRASPS = int(os.getenv("GRASPGEN_MIN_GRASPS", "10"))
GRASPGEN_MAX_TRIES = int(os.getenv("GRASPGEN_MAX_TRIES", "3"))
GRASPGEN_REMOVE_OUTLIERS = os.getenv("GRASPGEN_REMOVE_OUTLIERS", "1").lower() not in {
    "0",
    "false",
    "no",
}
GRASPGEN_MIN_POINTS = int(os.getenv("GRASPGEN_MIN_POINTS", "32"))
GRASPGEN_SELECTION = os.getenv("GRASPGEN_SELECTION", "spatial_median").lower()

# 3D cleanup of the masked cloud: SAM3 gives a 2D silhouette, so mask-edge bleed
# and stereo flying pixels leave table/background points attached. DBSCAN keeps
# only the dominant 3D cluster before the cloud reaches GraspGen.
GRASPGEN_DBSCAN = os.getenv("GRASPGEN_DBSCAN", "1").lower() not in {
    "0",
    "false",
    "no",
}
DBSCAN_EPS = float(os.getenv("DBSCAN_EPS", "0.01"))  # meters
DBSCAN_MIN_SAMPLES = int(os.getenv("DBSCAN_MIN_SAMPLES", "10"))

# Support-plane removal: DBSCAN can't split an object from the surface it rests
# on (they're spatially connected through the contact region). RANSAC fits the
# dominant plane (table/floor) and drops it before clustering. Runs only when a
# plane explains >= PLANE_MIN_RATIO of the points and the seed isn't on it.
GRASPGEN_PLANE_REMOVAL = os.getenv("GRASPGEN_PLANE_REMOVAL", "1").lower() not in {
    "0",
    "false",
    "no",
}
PLANE_DIST_THRESH = float(os.getenv("PLANE_DIST_THRESH", "0.005"))  # meters
PLANE_MIN_RATIO = float(os.getenv("PLANE_MIN_RATIO", "0.25"))
PLANE_RANSAC_ITERS = int(os.getenv("PLANE_RANSAC_ITERS", "100"))

print(f"Loading SAM3 from {SAM3_MODEL_ID} (fp16={SAM3_FP16}) ...")
_sam3_model = (
    Sam3Model.from_pretrained(SAM3_MODEL_ID, torch_dtype=_sam3_dtype)
    .to("cuda")
    .eval()
)
_sam3_processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID)
print("SAM3 loaded.")


# ---------------------------------------------------------------------------
# GraspGen forwarding client
# ---------------------------------------------------------------------------
#
# Minimal inline ZMQ REQ client (mirrors grasp_gen.serving.zmq_client) so this
# server needs ZERO grasp_gen / CUDA-heavy imports in its venv. Sends the cleaned
# (N, 3) cloud to the GraspGen server and returns (grasps, confidences).

class _GraspGenForwarder:
    def __init__(self, host: str, port: int, timeout_ms: int) -> None:
        self._addr = f"tcp://{host}:{port}"
        self._timeout_ms = timeout_ms
        self._ctx = zmq.Context.instance()
        self._socket = None

    def _connect(self):
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self._addr)
        return sock

    def infer(self, point_cloud: np.ndarray, **params):
        """Returns (grasps (M,4,4) float32, confidences (M,) float32)."""
        if self._socket is None:
            self._socket = self._connect()
        payload = {"action": "infer", "point_cloud": point_cloud, **params}
        try:
            self._socket.send(msgpack.packb(payload, use_bin_type=True))
            raw = self._socket.recv()
        except zmq.error.ZMQError:
            # REQ sockets wedge after a timeout/error — drop and rebuild it so
            # the next request can recover.
            self._socket.close()
            self._socket = None
            raise
        response = msgpack.unpackb(raw, raw=False)
        if "error" in response:
            raise RuntimeError(f"GraspGen server error: {response['error']}")
        grasps = np.asarray(response["grasps"], dtype=np.float32)
        confidences = np.asarray(response["confidences"], dtype=np.float32)
        return grasps, confidences


_grasp_forwarder = _GraspGenForwarder(
    GRASPGEN_HOST, GRASPGEN_PORT, GRASPGEN_TIMEOUT_MS
)
print(
    f"GraspGen forwarding -> tcp://{GRASPGEN_HOST}:{GRASPGEN_PORT} "
    f"(num_grasps={GRASPGEN_NUM_GRASPS}, topk={GRASPGEN_TOPK}, "
    f"min_grasps={GRASPGEN_MIN_GRASPS}, max_tries={GRASPGEN_MAX_TRIES}, "
    f"remove_outliers={GRASPGEN_REMOVE_OUTLIERS}, "
    f"selection={GRASPGEN_SELECTION})"
)

# ---------------------------------------------------------------------------
# ZMQ
# ---------------------------------------------------------------------------

context = zmq.Context.instance()
socket = context.socket(zmq.REP)
socket.bind(f"tcp://0.0.0.0:{RPC_PORT}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_image_payload(image_payload) -> np.ndarray:
    """Decode an image payload and return an RGB (H, W, 3) uint8 array."""
    if isinstance(image_payload, dict):
        if "data" in image_payload:
            image = np.asarray(
                image_payload["data"],
                dtype=image_payload.get("dtype", np.uint8),
            )
            if "shape" in image_payload:
                image = image.reshape(image_payload["shape"])
            return image  # assumed RGB from caller

        if "base64" in image_payload:
            encoded_image = image_payload["base64"]
        else:
            raise ValueError("image dict must include 'data' or 'base64'")
    elif isinstance(image_payload, str):
        encoded_image = image_payload
    else:
        return np.asarray(image_payload, dtype=np.uint8)  # assumed RGB

    image_bytes = base64.b64decode(encoded_image)
    image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image payload")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _decode_point_cloud(pc_payload) -> np.ndarray:
    """Decode point cloud payload to (H, W, 3) or (N, 3) float32 array."""
    return np.asarray(pc_payload, dtype=np.float32)


def _decode_message(raw: bytes) -> tuple[dict, str]:
    """Decode either the legacy JSON request or the faster msgpack request."""
    try:
        return json.loads(raw.decode("utf-8")), "json"
    except (UnicodeDecodeError, json.JSONDecodeError):
        return msgpack.unpackb(raw, raw=False), "msgpack"


def _json_ready(value):
    """Convert NumPy values to JSON-compatible values for legacy clients."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _send_response(response: dict, protocol: str) -> None:
    if protocol == "json":
        socket.send_json(_json_ready(response))
    else:
        socket.send(msgpack.packb(response, use_bin_type=True))


def _empty_grasp_result(reason: str) -> dict:
    return {
        "best_grasp": None,
        "num_grasps": 0,
        "num_candidates": 0,
        "message": reason,
    }


def _select_grasp_index(grasps: np.ndarray, confidences: np.ndarray) -> int:
    """Pick a single grasp index from the candidates.

    "max_confidence" returns the top-scored grasp (jittery across diffusion runs).
    "spatial_median" picks the grasp closest to the median translation of the
    top-K, which is more stable across stochastic samples.
    """
    if GRASPGEN_SELECTION == "max_confidence" or len(grasps) <= 1:
        return int(np.argmax(confidences))

    positions = grasps[:, :3, 3]
    median_position = np.median(positions, axis=0)
    distances = np.linalg.norm(positions - median_position, axis=1)
    return int(np.argmin(distances))


def _format_best_grasp(grasps: np.ndarray, confidences: np.ndarray) -> dict:
    best_idx = _select_grasp_index(grasps, confidences)
    pose = grasps[best_idx].astype(np.float32, copy=False)
    confidence = float(confidences[best_idx])
    return {
        "best_grasp": {
            "pose": pose,
            "position": pose[:3, 3].astype(np.float32, copy=False),
            "rotation_matrix": pose[:3, :3].astype(np.float32, copy=False),
            "confidence": confidence,
            "candidate_index": best_idx,
            "selection": GRASPGEN_SELECTION,
            "frame": "zed_camera",
            "units": "meters",
        },
        "num_grasps": 1,
        "num_candidates": int(len(grasps)),
        "best_confidence": confidence,
    }


def _draw_overlay(
    rgb: np.ndarray, mask: np.ndarray, u: int, v: int, score: float, n_pts: int,
    text: str,
) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr[mask == 1] = (
        bgr[mask == 1] * 0.6 + np.array([0, 200, 0]) * 0.4
    ).astype(np.uint8)
    # u, v is the mask centroid (no click in the text-prompt flow).
    if u >= 0 and v >= 0:
        cv2.circle(bgr, (u, v), 6, (0, 0, 255), -1)
        cv2.circle(bgr, (u, v), 8, (255, 255, 255), 2)
    label = f"'{text}'  score={score:.3f}  pts={n_pts}"
    cv2.putText(
        bgr,
        label,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        bgr,
        label,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return bgr


# ---------------------------------------------------------------------------
# RPC functions
# ---------------------------------------------------------------------------

def _remove_support_plane(pts, seed_xyz=None):
    """RANSAC-fit the dominant plane and drop its inliers (the floor/table).

    DBSCAN can't separate an object from the surface it touches because they're
    one connected component. Removing the support plane cuts that link so the
    object survives clustering on its own.

    Skips removal (returns pts unchanged) when no plane explains at least
    PLANE_MIN_RATIO of the points, when the seed point lies on the best plane
    (the user is grasping a flat object, not the floor), or when removal would
    leave too few points to grasp.
    """
    n = len(pts)
    if not GRASPGEN_PLANE_REMOVAL or n < 3 * DBSCAN_MIN_SAMPLES:
        return pts

    rng = np.random.default_rng()
    best_inliers, best_count, best_plane = None, 0, None
    for _ in range(PLANE_RANSAC_ITERS):
        p0, p1, p2 = pts[rng.choice(n, 3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(normal)
        if nn < 1e-9:
            continue
        normal = normal / nn
        inliers = np.abs((pts - p0) @ normal) < PLANE_DIST_THRESH
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers, best_plane = count, inliers, (normal, p0)

    if best_inliers is None or best_count < PLANE_MIN_RATIO * n:
        return pts  # no dominant plane to remove

    # Abort if the seed sits on the plane — that's the object, not the floor.
    if seed_xyz is not None and np.all(np.isfinite(seed_xyz)):
        normal, p0 = best_plane
        if abs(float((seed_xyz - p0) @ normal)) < PLANE_DIST_THRESH:
            return pts

    keep = ~best_inliers
    if int(keep.sum()) < DBSCAN_MIN_SAMPLES:
        return pts
    return pts[keep]


def _largest_object_cluster(pts, seed_xyz=None):
    """Keep the dominant 3D cluster of the masked points.

    A 2D SAM3 mask projected into 3D drags in table/background points along the
    silhouette edge and stereo flying pixels at depth discontinuities. DBSCAN
    separates those into distinct clusters; we keep one and drop the rest.

    Args:
        pts: (N, 3) finite points.
        seed_xyz: optional (3,) anchor — the 3D point under the mask centroid.
            When given, the cluster whose nearest member is closest to the seed
            is kept, so a small object beside a large background blob still wins.
            When None, the largest cluster wins.

    Returns:
        (M, 3) filtered points. Falls back to the input unchanged when clustering
        is disabled, the cloud is too small, or every point is labelled noise.
    """
    if not GRASPGEN_DBSCAN or len(pts) < DBSCAN_MIN_SAMPLES:
        return pts

    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(pts).labels_
    valid = labels >= 0
    if not valid.any():
        return pts  # all noise — don't throw the whole cloud away

    if seed_xyz is not None and np.all(np.isfinite(seed_xyz)):
        best_label, best_d = None, np.inf
        for lab in np.unique(labels[valid]):
            d = np.linalg.norm(pts[labels == lab] - seed_xyz, axis=1).min()
            if d < best_d:
                best_d, best_label = d, lab
        keep = labels == best_label
    else:
        counts = np.bincount(labels[valid])
        keep = labels == int(counts.argmax())

    return pts[keep]


def view_image_point_cloud_text(image, point_cloud, text):
    t_start = time.perf_counter()
    rgb = _decode_image_payload(image)
    client_frame = image.get("frame") if isinstance(image, dict) else None
    client_checksum = image.get("checksum") if isinstance(image, dict) else None
    server_checksum = zlib.crc32(np.ascontiguousarray(rgb).tobytes()) & 0xFFFFFFFF

    print(
        f"server image client_frame={client_frame} "
        f"client_crc={client_checksum} server_crc={server_checksum} "
        f"text={text!r}"
    )

    pc = _decode_point_cloud(point_cloud)  # (H_c, W_c, 3) organized
    t_decode = time.perf_counter()

    finite_map = (
        np.isfinite(pc).all(axis=-1)
        if pc.ndim == 3
        else np.isfinite(pc).all(axis=1)
    )

    # SAM3: text ("concept") prompt -> one mask per matching instance.
    with torch.inference_mode():
        pil = Image.fromarray(rgb)
        inputs = _sam3_processor(
            images=pil, text=text, return_tensors="pt"
        ).to("cuda")
        # Match float tensors (e.g. pixel_values) to the model dtype; leave
        # integer tensors (input_ids, attention masks) as-is.
        if SAM3_FP16:
            inputs = inputs.to(_sam3_dtype)
        outputs = _sam3_model(**inputs)
        results = _sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=SAM3_THRESHOLD,
            mask_threshold=SAM3_MASK_THRESHOLD,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
    t_sam = time.perf_counter()

    if len(results["scores"]) == 0:
        print(f"SAM3 found no instance for {text!r}")
        return _empty_grasp_result(f"SAM3 found no instance for '{text}'")

    # Keep the single highest-scoring instance.
    best = int(results["scores"].argmax())
    mask = results["masks"][best].cpu().numpy().astype(np.uint8)  # (H_img, W_img)
    score = float(results["scores"][best])
    n_instances = int(len(results["scores"]))
    print(
        f"SAM3 found {n_instances} instance(s) for {text!r}; "
        f"best score={score:.3f}, mask coverage={mask.mean() * 100:.1f}%"
    )

    # Mask centroid (image coords) — stands in for the SAM2 click point: drives
    # both the overlay marker and the 3D seed for cluster selection.
    ys, xs = np.where(mask == 1)
    if len(xs) > 0:
        u = int(round(xs.mean()))
        v = int(round(ys.mean()))
    else:
        u, v = -1, -1

    # Apply mask to organized cloud — resize mask to match cloud's strided dims
    seed_xyz = None  # 3D anchor under the mask centroid, for cluster selection
    if pc.ndim == 3:
        H_c, W_c = pc.shape[:2]
        mask_resized = cv2.resize(mask, (W_c, H_c), interpolation=cv2.INTER_NEAREST)
        masked_pixels = int(mask_resized.sum())
        pts = pc[mask_resized == 1]  # (M, 3)
        finite_mask = np.isfinite(pts).all(axis=1)
        print(
            f"Cloud ({H_c}x{W_c}), image ({rgb.shape[0]}x{rgb.shape[1]}), "
            f"masked_pixels={masked_pixels}, finite={finite_mask.sum()}"
        )
        pts = pts[finite_mask]

        # 3D point under the mask centroid (image coords -> cloud coords).
        if u >= 0 and v >= 0:
            u_c = max(0, min(int(round(u * W_c / rgb.shape[1])), W_c - 1))
            v_c = max(0, min(int(round(v * H_c / rgb.shape[0])), H_c - 1))
            cand = pc[v_c, u_c]
            if np.all(np.isfinite(cand)):
                seed_xyz = cand

        # Textureless surfaces (e.g. cubes) cause ZED depth NaN inside the mask.
        # Fall back to dilated mask to capture valid boundary/edge points.
        if len(pts) == 0 and masked_pixels > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask_dilated = cv2.dilate(mask_resized, kernel)
            pts_d = pc[mask_dilated == 1]
            fm = np.isfinite(pts_d).all(axis=1)
            pts = pts_d[fm]
            print(f"Dilation fallback: {len(pts)} boundary points")
    else:
        pts = pc  # flat cloud — return as-is

    # Drop table/background contamination left by the 2D mask: first remove the
    # support plane the object rests on, then keep the dominant 3D cluster.
    if len(pts) >= DBSCAN_MIN_SAMPLES:
        n_before = len(pts)
        pts = _remove_support_plane(pts, seed_xyz)
        n_plane = len(pts)
        pts = _largest_object_cluster(pts, seed_xyz)
        print(
            f"3D cleanup: {n_before} -> {n_plane} (plane) -> {len(pts)} (cluster) "
            f"points (seed={'yes' if seed_xyz is not None else 'no'})"
        )

    t_mask = time.perf_counter()

    if pts.ndim != 2 or pts.shape[1] != 3:
        grasp_result = _empty_grasp_result(
            f"Segmented point cloud must be (N, 3), got {pts.shape}"
        )
    elif len(pts) < GRASPGEN_MIN_POINTS:
        grasp_result = _empty_grasp_result(
            f"Only {len(pts)} segmented points; need at least {GRASPGEN_MIN_POINTS}"
        )
    else:
        # Forward the cleaned cloud to the GraspGen server (separate process).
        try:
            grasps_np, confidences_np = _grasp_forwarder.infer(
                pts.astype(np.float32, copy=False),
                grasp_threshold=-1.0,
                num_grasps=GRASPGEN_NUM_GRASPS,
                topk_num_grasps=GRASPGEN_TOPK,
                min_grasps=GRASPGEN_MIN_GRASPS,
                max_tries=GRASPGEN_MAX_TRIES,
                remove_outliers=GRASPGEN_REMOVE_OUTLIERS,
            )
        except (zmq.error.ZMQError, RuntimeError) as exc:
            grasp_result = _empty_grasp_result(f"GraspGen forward failed: {exc}")
        else:
            if len(grasps_np) == 0:
                grasp_result = _empty_grasp_result("GraspGen returned no grasps")
            else:
                grasps_np = np.asarray(grasps_np, dtype=np.float32)
                confidences_np = np.asarray(confidences_np, dtype=np.float32)
                grasps_np[:, 3, 3] = 1
                grasp_result = _format_best_grasp(grasps_np, confidences_np)
    t_graspgen = time.perf_counter()

    if RPC_VISUALIZE:
        vis = _draw_overlay(rgb, mask, u, v, score, len(pts), text)

        # Drawing every masked pixel dominates latency for large masks.
        if pc.ndim == 3 and len(pts) > 0 and RPC_MAX_MARKERS > 0:
            H_c, W_c = pc.shape[:2]
            scale_r = rgb.shape[0] / H_c
            scale_c = rgb.shape[1] / W_c
            seg_map = (mask_resized == 1) & finite_map
            coords = np.column_stack(np.where(seg_map))
            if len(coords) > RPC_MAX_MARKERS:
                step = int(np.ceil(len(coords) / RPC_MAX_MARKERS))
                coords = coords[::step][:RPC_MAX_MARKERS]
            for r, c in coords:
                px = (int(c * scale_c), int(r * scale_r))
                cv2.drawMarker(vis, px, (255, 0, 255), cv2.MARKER_STAR, 12, 2)

        cv2.imshow("SAM3 Segmentation", vis)
        cv2.waitKey(1)

    t_vis = time.perf_counter()
    print(
        f"Segmented {len(pts)} 3D points | "
        f"decode={(t_decode - t_start) * 1000:.1f}ms, "
        f"sam={(t_sam - t_decode) * 1000:.1f}ms, "
        f"mask={(t_mask - t_sam) * 1000:.1f}ms, "
        f"graspgen={(t_graspgen - t_mask) * 1000:.1f}ms, "
        f"vis={(t_vis - t_graspgen) * 1000:.1f}ms"
    )

    grasp_result["timing"] = {
        "decode_ms": (t_decode - t_start) * 1000,
        "sam_ms": (t_sam - t_decode) * 1000,
        "mask_ms": (t_mask - t_sam) * 1000,
        "graspgen_ms": (t_graspgen - t_mask) * 1000,
        "visualize_ms": (t_vis - t_graspgen) * 1000,
        "total_handler_ms": (t_vis - t_start) * 1000,
    }
    return grasp_result


FUNCTIONS = {
    "view_image_point_cloud_text": view_image_point_cloud_text,
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

print(
    f"RPC Server (SAM3) started on port {RPC_PORT} "
    f"(visualize={RPC_VISUALIZE}, max_markers={RPC_MAX_MARKERS})"
)

while True:
    raw = socket.recv()

    try:
        message, protocol = _decode_message(raw)
        func_name = message["function"]
        args = message.get("args", [])

        print(f"Received: {func_name} ({protocol})")

        if func_name in FUNCTIONS:
            t_handle = time.perf_counter()
            result = FUNCTIONS[func_name](*args)
            t_result = time.perf_counter()
            if isinstance(result, dict):
                response = {"success": True, **result}
            else:
                response = {"success": True, "result": result}
            response.setdefault("timing", {})["handler_ms"] = (
                t_result - t_handle
            ) * 1000
            _send_response(response, protocol)
            t_sent = time.perf_counter()
            print(
                f"Replied ({protocol}) | "
                f"handler={(t_result - t_handle) * 1000:.1f}ms, "
                f"encode_send={(t_sent - t_result) * 1000:.1f}ms"
            )
        else:
            _send_response(
                {"success": False, "error": "Function not found"}, protocol
            )
    except Exception as exc:
        print("RPC Error:", exc)
        protocol = locals().get("protocol", "json")
        _send_response({"success": False, "error": str(exc)}, protocol)
