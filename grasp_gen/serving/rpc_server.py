import base64

import cv2
import numpy as np
import torch
import zmq
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ---------------------------------------------------------------------------
# SAM2 — loaded once at startup
# ---------------------------------------------------------------------------

SAM2_CHECKPOINT = "weights/sam2.1_hiera_small.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s"

print("Loading SAM2 ...")
_sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device="cuda")
_predictor = SAM2ImagePredictor(_sam2_model)
print("SAM2 loaded.")

# ---------------------------------------------------------------------------
# ZMQ
# ---------------------------------------------------------------------------

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://0.0.0.0:5555")


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


def _draw_overlay(
    rgb: np.ndarray, mask: np.ndarray, u: int, v: int, score: float
) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr[mask == 1] = (bgr[mask == 1] * 0.6 + np.array([0, 200, 0]) * 0.4).astype(np.uint8)
    cv2.circle(bgr, (u, v), 6, (0, 0, 255), -1)
    cv2.circle(bgr, (u, v), 8, (255, 255, 255), 2)
    label = f"score={score:.3f}"
    cv2.putText(bgr, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bgr, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    return bgr


# ---------------------------------------------------------------------------
# RPC functions
# ---------------------------------------------------------------------------

def view_image_point_cloud_point(image, _point_cloud, point):
    rgb = _decode_image_payload(image)
    point_2d = np.asarray(point).flatten()

    u = max(0, min(int(round(point_2d[1])), rgb.shape[1] - 1))
    v = max(0, min(int(round(point_2d[0])), rgb.shape[0] - 1))

    print(f"Received 2D point: ({u}, {v})")

    with torch.inference_mode():
        _predictor.set_image(rgb)
        masks, scores, _ = _predictor.predict(
            point_coords=np.array([[u, v]]),
            point_labels=np.array([1]),
            multimask_output=False,
        )

    mask = masks[0].astype(np.uint8)
    score = float(scores[0])
    print(f"SAM2 mask coverage={mask.mean() * 100:.1f}%, score={score:.3f}")

    vis = _draw_overlay(rgb, mask, u, v, score)
    cv2.imshow("SAM2 Segmentation", vis)
    cv2.waitKey(1)

    return "ok"


FUNCTIONS = {
    "view_image_point_cloud_point": view_image_point_cloud_point,
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

print("RPC Server started...")

while True:
    message = socket.recv_json()

    func_name = message["function"]
    args = message.get("args", [])

    print("Received:", func_name)

    if func_name in FUNCTIONS:
        try:
            result = FUNCTIONS[func_name](*args)
            socket.send_json({"success": True, "result": result})
        except Exception as exc:
            print(f"Error in {func_name}:", exc)
            socket.send_json({"success": False, "error": str(exc)})
    else:
        socket.send_json({"success": False, "error": "Function not found"})
