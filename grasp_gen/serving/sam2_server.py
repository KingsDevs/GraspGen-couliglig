"""SAM2 segmentation server with ZMQ.

Two sockets:
  SUB  (default port 5560) — subscribes to a camera publisher that sends
       msgpack frames: {rgb: (H,W,3) uint8, depth: (H,W) float32,
                        intrinsics: {fx, fy, cx, cy}}
  REP  (default port 5561) — request/reply for segmentation.
       Request:  {point_3d: [x, y, z]}   (camera-frame 3-D point)
       Response: {mask: (H,W) uint8, score: float, point_2d: [u, v]}

The server stores the latest frame from the SUB socket.  When a REP
request arrives it projects the 3-D point to 2-D (using the intrinsics
from the latest frame), runs SAM2, and returns the binary mask.

Usage:
    python -m grasp_gen.serving.sam2_server \\
        --checkpoint weights/sam2.1_hiera_small.pt \\
        --config    weights/sam2.1_hiera_s.yaml

Camera publisher format (msgpack-numpy):
    {
        "rgb":        np.ndarray (H, W, 3) uint8,
        "depth":      np.ndarray (H, W)    float32  (metres),
        "intrinsics": {"fx": float, "fy": float, "cx": float, "cy": float},
    }
"""

import argparse
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import msgpack
import msgpack_numpy
import numpy as np
import torch
import zmq
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

msgpack_numpy.patch()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def project_3d_to_2d(point_3d: np.ndarray, intrinsics: dict) -> tuple[int, int]:
    """Project a camera-frame 3-D point (X, Y, Z) to pixel (u, v)."""
    x, y, z = point_3d
    if z <= 0:
        raise ValueError(f"Point Z must be positive, got {z}")
    u = int(round(intrinsics["fx"] * x / z + intrinsics["cx"]))
    v = int(round(intrinsics["fy"] * y / z + intrinsics["cy"]))
    return u, v


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class SAM2Server:
    def __init__(
        self,
        checkpoint: str,
        config: str,
        sub_port: int = 5560,
        rep_port: int = 5561,
        device: str = "cuda",
    ) -> None:
        self._sub_port = sub_port
        self._rep_port = rep_port
        self._device = device

        # config is the Hydra config name (e.g. "configs/sam2.1/sam2.1_hiera_s")
        # or a full file path — we strip to the name if it ends in .yaml
        config_name = config.removesuffix(".yaml") if config.endswith(".yaml") else config
        logger.info("Loading SAM2 from %s / %s on %s", checkpoint, config_name, device)
        sam2_model = build_sam2(config_name, checkpoint, device=device)
        self._predictor = SAM2ImagePredictor(sam2_model)
        logger.info("SAM2 loaded")

        # Latest frame shared between the subscriber thread and REP handler
        self._frame_lock = threading.Lock()
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_intrinsics: Optional[dict] = None
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Subscriber thread — keeps latest camera frame up to date
    # ------------------------------------------------------------------

    def _sub_loop(self, ctx: zmq.Context) -> None:
        sock = ctx.socket(zmq.SUB)
        sock.connect(f"tcp://localhost:{self._sub_port}")
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        logger.info("SUB socket connected to port %d", self._sub_port)

        while True:
            try:
                raw = sock.recv()
                frame = msgpack.unpackb(raw, raw=False)

                rgb = np.asarray(frame["rgb"], dtype=np.uint8)
                depth = np.asarray(frame["depth"], dtype=np.float32)
                intrinsics = frame["intrinsics"]

                with self._frame_lock:
                    self._latest_rgb = rgb
                    self._latest_depth = depth
                    self._latest_intrinsics = intrinsics
                    self._frame_count += 1

            except Exception:
                logger.exception("Error in SUB loop")

    # ------------------------------------------------------------------
    # REP handler — segmentation on demand
    # ------------------------------------------------------------------

    def _handle_request(self, request: dict) -> dict:
        with self._frame_lock:
            rgb = self._latest_rgb
            intrinsics = self._latest_intrinsics
            frame_count = self._frame_count

        if rgb is None:
            return {"error": "No camera frame received yet. Is the publisher running?"}

        point_3d_raw = request.get("point_3d")
        if point_3d_raw is None:
            return {"error": "Missing required field 'point_3d'"}

        point_3d = np.asarray(point_3d_raw, dtype=np.float64).flatten()
        if point_3d.shape != (3,):
            return {"error": f"'point_3d' must have 3 elements, got {point_3d.shape}"}

        try:
            u, v = project_3d_to_2d(point_3d, intrinsics)
        except ValueError as e:
            return {"error": str(e)}

        h, w = rgb.shape[:2]
        u = max(0, min(u, w - 1))
        v = max(0, min(v, h - 1))

        logger.info(
            "Segmenting at pixel (%d, %d) from 3D point (%.3f, %.3f, %.3f) [frame %d]",
            u, v, point_3d[0], point_3d[1], point_3d[2], frame_count,
        )

        t0 = time.monotonic()
        with torch.inference_mode():
            self._predictor.set_image(rgb)
            masks, scores, _ = self._predictor.predict(
                point_coords=np.array([[u, v]]),
                point_labels=np.array([1]),  # 1 = foreground
                multimask_output=False,
            )
        elapsed_ms = (time.monotonic() - t0) * 1000

        mask = masks[0].astype(np.uint8)  # (H, W) binary
        score = float(scores[0])

        logger.info(
            "SAM2 done in %.1f ms — mask coverage %.1f%%, score %.3f",
            elapsed_ms,
            mask.mean() * 100,
            score,
        )

        return {
            "mask": mask,
            "score": score,
            "point_2d": [u, v],
            "timing_ms": elapsed_ms,
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def serve_forever(self) -> None:
        ctx = zmq.Context()

        # Start subscriber in background thread
        sub_thread = threading.Thread(target=self._sub_loop, args=(ctx,), daemon=True)
        sub_thread.start()

        # REP socket in main thread
        rep_sock = ctx.socket(zmq.REP)
        rep_sock.bind(f"tcp://0.0.0.0:{self._rep_port}")
        logger.info("REP socket listening on port %d", self._rep_port)
        logger.info("Ready. Waiting for segmentation requests...")

        try:
            while True:
                raw = rep_sock.recv()
                try:
                    request = msgpack.unpackb(raw, raw=False)
                    response = self._handle_request(request)
                except Exception as exc:
                    logger.exception("Error handling request")
                    response = {"error": str(exc)}
                rep_sock.send(msgpack.packb(response, use_bin_type=True))
        except KeyboardInterrupt:
            logger.info("Shutting down")
        finally:
            rep_sock.close()
            ctx.term()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 ZMQ segmentation server")
    parser.add_argument("--checkpoint", required=True, help="Path to SAM2 .pt checkpoint")
    parser.add_argument("--config",     required=True, help="Path to SAM2 .yaml config")
    parser.add_argument("--sub-port",   type=int, default=5560, help="ZMQ SUB port (camera publisher)")
    parser.add_argument("--rep-port",   type=int, default=5561, help="ZMQ REP port (segmentation requests)")
    parser.add_argument("--device",     default="cuda", help="torch device (cuda / cpu)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    server = SAM2Server(
        checkpoint=args.checkpoint,
        config=args.config,
        sub_port=args.sub_port,
        rep_port=args.rep_port,
        device=args.device,
    )
    server.serve_forever()
