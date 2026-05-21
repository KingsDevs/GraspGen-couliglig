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
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

msgpack_numpy.patch()

# ---------------------------------------------------------------------------
# SAM2 — loaded once at startup
# ---------------------------------------------------------------------------

SAM2_CHECKPOINT = "weights/sam2.1_hiera_small.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s"
RPC_PORT = int(os.getenv("RPC_PORT", "5555"))
RPC_VISUALIZE = os.getenv("RPC_VISUALIZE", "1").lower() not in {
    "0",
    "false",
    "no",
}
RPC_MAX_MARKERS = int(os.getenv("RPC_MAX_MARKERS", "1000"))
GRASPGEN_CONFIG = os.getenv(
    "GRASPGEN_CONFIG", "weights/graspgen_robotiq_2f_140.yml"
)
GRASPGEN_NUM_GRASPS = int(os.getenv("GRASPGEN_NUM_GRASPS", "100"))
GRASPGEN_TOPK = int(os.getenv("GRASPGEN_TOPK", "20"))
GRASPGEN_MIN_GRASPS = int(os.getenv("GRASPGEN_MIN_GRASPS", "1"))
GRASPGEN_MAX_TRIES = int(os.getenv("GRASPGEN_MAX_TRIES", "1"))
GRASPGEN_REMOVE_OUTLIERS = os.getenv("GRASPGEN_REMOVE_OUTLIERS", "1").lower() not in {
    "0",
    "false",
    "no",
}
GRASPGEN_MIN_POINTS = int(os.getenv("GRASPGEN_MIN_POINTS", "32"))

print("Loading SAM2 ...")
_sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device="cuda")
_predictor = SAM2ImagePredictor(_sam2_model)
print("SAM2 loaded.")

print(f"Loading GraspGen from {GRASPGEN_CONFIG} ...")
_grasp_cfg = load_grasp_cfg(GRASPGEN_CONFIG)
_grasp_sampler = GraspGenSampler(_grasp_cfg)
print(
    "GraspGen loaded "
    f"(num_grasps={GRASPGEN_NUM_GRASPS}, topk={GRASPGEN_TOPK}, "
    f"min_grasps={GRASPGEN_MIN_GRASPS}, max_tries={GRASPGEN_MAX_TRIES}, "
    f"remove_outliers={GRASPGEN_REMOVE_OUTLIERS})"
)

# ---------------------------------------------------------------------------
# ZMQ
# ---------------------------------------------------------------------------

context = zmq.Context()
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
        "grasps": np.empty((0, 4, 4), dtype=np.float32),
        "confidences": np.empty((0,), dtype=np.float32),
        "num_grasps": 0,
        "message": reason,
    }


def _draw_overlay(
    rgb: np.ndarray, mask: np.ndarray, u: int, v: int, score: float, n_pts: int
) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr[mask == 1] = (
        bgr[mask == 1] * 0.6 + np.array([0, 200, 0]) * 0.4
    ).astype(np.uint8)
    cv2.circle(bgr, (u, v), 6, (0, 0, 255), -1)
    cv2.circle(bgr, (u, v), 8, (255, 255, 255), 2)
    label = f"score={score:.3f}  pts={n_pts}"
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

def view_image_point_cloud_point(image, point_cloud, point):
    t_start = time.perf_counter()
    rgb = _decode_image_payload(image)
    pc = _decode_point_cloud(point_cloud)  # (H_c, W_c, 3) organized
    t_decode = time.perf_counter()

    point_2d = np.asarray(point).flatten()
    u = max(0, min(int(round(point_2d[0])), rgb.shape[1] - 1))
    v = max(0, min(int(round(point_2d[1])), rgb.shape[0] - 1))

    finite_map = (
        np.isfinite(pc).all(axis=-1)
        if pc.ndim == 3
        else np.isfinite(pc).all(axis=1)
    )
    total_finite = int(finite_map.sum())
    if total_finite > 0:
        valid_flat = pc[finite_map] if pc.ndim == 3 else pc[finite_map]
        print(
            f"Received 2D point: ({u}, {v}), cloud shape: {pc.shape}, "
            f"total finite pts: {total_finite}, "
            f"z range: {valid_flat[:,2].min():.3f}-{valid_flat[:,2].max():.3f}m"
        )
    else:
        print(
            f"Received 2D point: ({u}, {v}), cloud shape: {pc.shape}, "
            "total finite pts: 0"
        )

    with torch.inference_mode():
        _predictor.set_image(rgb)
        masks, scores, _ = _predictor.predict(
            point_coords=np.array([[u, v]]),
            point_labels=np.array([1]),
            multimask_output=False,
        )
    t_sam = time.perf_counter()

    mask = masks[0].astype(np.uint8)  # (H_img, W_img)
    score = float(scores[0])
    print(f"SAM2 mask coverage={mask.mean() * 100:.1f}%, score={score:.3f}")

    # Apply mask to organized cloud — resize mask to match cloud's strided dims
    if pc.ndim == 3:
        H_c, W_c = pc.shape[:2]
        mask_resized = cv2.resize(mask, (W_c, H_c), interpolation=cv2.INTER_NEAREST)
        masked_pixels = int(mask_resized.sum())
        pts = pc[mask_resized == 1]  # (M, 3)
        finite_mask = np.isfinite(pts).all(axis=1)
        z_mask = pts[:, 2] > 0
        print(
            f"Cloud ({H_c}x{W_c}), image ({rgb.shape[0]}x{rgb.shape[1]}), "
            f"masked_pixels={masked_pixels}, "
            f"finite={finite_mask.sum()}, z>0={(finite_mask & z_mask).sum()}"
        )
        pts = pts[finite_mask & z_mask]

        # Textureless surfaces (e.g. cubes) cause ZED depth NaN inside the mask.
        # Fall back to dilated mask to capture valid boundary/edge points.
        if len(pts) == 0 and masked_pixels > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask_dilated = cv2.dilate(mask_resized, kernel)
            pts_d = pc[mask_dilated == 1]
            fm = np.isfinite(pts_d).all(axis=1) & (pts_d[:, 2] > 0)
            pts = pts_d[fm]
            print(f"Dilation fallback: {len(pts)} boundary points")
    else:
        pts = pc  # flat cloud — return as-is

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
        grasps, confidences = GraspGenSampler.run_inference(
            pts.astype(np.float32, copy=False),
            _grasp_sampler,
            grasp_threshold=-1.0,
            num_grasps=GRASPGEN_NUM_GRASPS,
            topk_num_grasps=GRASPGEN_TOPK,
            min_grasps=GRASPGEN_MIN_GRASPS,
            max_tries=GRASPGEN_MAX_TRIES,
            remove_outliers=GRASPGEN_REMOVE_OUTLIERS,
        )

        if len(grasps) == 0:
            grasp_result = _empty_grasp_result("GraspGen returned no grasps")
        else:
            if torch.is_tensor(grasps):
                grasps_np = grasps.detach().cpu().numpy().astype(np.float32)
            else:
                grasps_np = np.asarray(grasps, dtype=np.float32)
            if torch.is_tensor(confidences):
                confidences_np = (
                    confidences.detach().cpu().numpy().astype(np.float32)
                )
            else:
                confidences_np = np.asarray(confidences, dtype=np.float32)
            grasps_np[:, 3, 3] = 1
            grasp_result = {
                "grasps": grasps_np,
                "confidences": confidences_np,
                "num_grasps": len(grasps_np),
            }
    t_graspgen = time.perf_counter()

    if RPC_VISUALIZE:
        vis = _draw_overlay(rgb, mask, u, v, score, len(pts))

        # Drawing every masked pixel dominates latency for large masks.
        if pc.ndim == 3 and len(pts) > 0 and RPC_MAX_MARKERS > 0:
            H_c, W_c = pc.shape[:2]
            scale_r = rgb.shape[0] / H_c
            scale_c = rgb.shape[1] / W_c
            seg_map = (mask_resized == 1) & finite_map & (pc[:, :, 2] > 0)
            coords = np.column_stack(np.where(seg_map))
            if len(coords) > RPC_MAX_MARKERS:
                step = int(np.ceil(len(coords) / RPC_MAX_MARKERS))
                coords = coords[::step][:RPC_MAX_MARKERS]
            for r, c in coords:
                px = (int(c * scale_c), int(r * scale_r))
                cv2.drawMarker(vis, px, (255, 0, 255), cv2.MARKER_STAR, 12, 2)

        cv2.imshow("SAM2 Segmentation", vis)
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
    "view_image_point_cloud_point": view_image_point_cloud_point,
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

print(
    f"RPC Server started on port {RPC_PORT} "
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
