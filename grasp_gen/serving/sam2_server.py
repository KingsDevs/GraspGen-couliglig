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

With --visualize:
  An OpenCV window shows the live camera feed.  Click any pixel to segment
  the object at that point — depth is used to back-project the click to 3-D.
  The ZMQ REP socket continues to work in parallel.

Usage:
    python -m grasp_gen.serving.sam2_server \\
        --checkpoint weights/sam2.1_hiera_small.pt \\
        --config     configs/sam2.1/sam2.1_hiera_s \\
        --visualize

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
from typing import Optional

import msgpack
import msgpack_numpy
import numpy as np
import torch
import zmq
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

msgpack_numpy.patch()

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

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


def backproject_2d_to_3d(u: int, v: int, depth: np.ndarray, intrinsics: dict) -> Optional[np.ndarray]:
    """Back-project pixel (u, v) to a 3-D camera-frame point using the depth map."""
    d = float(depth[v, u])
    if d <= 0:
        return None
    x = (u - intrinsics["cx"]) * d / intrinsics["fx"]
    y = (v - intrinsics["cy"]) * d / intrinsics["fy"]
    return np.array([x, y, d], dtype=np.float64)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class SAM2Server:
    def __init__(
        self,
        checkpoint: str,
        config: str,
        sub_host: str = "localhost",
        sub_port: int = 5560,
        rep_port: int = 5561,
        device: str = "cuda",
        visualize: bool = False,
    ) -> None:
        self._sub_host = sub_host
        self._sub_port = sub_port
        self._rep_port = rep_port
        self._device = device
        self._visualize = visualize

        if visualize and not _CV2_AVAILABLE:
            raise RuntimeError("--visualize requires opencv-python: uv pip install opencv-python")

        # config is the Hydra config name (e.g. "configs/sam2.1/sam2.1_hiera_s")
        # or a full file path — we strip to the name if it ends in .yaml
        config_name = config.removesuffix(".yaml") if config.endswith(".yaml") else config
        logger.info("Loading SAM2 from %s / %s on %s", checkpoint, config_name, device)
        sam2_model = build_sam2(config_name, checkpoint, device=device)
        self._predictor = SAM2ImagePredictor(sam2_model)
        logger.info("SAM2 loaded")

        # Latest frame — shared between SUB thread and all handlers
        self._frame_lock = threading.Lock()
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_intrinsics: Optional[dict] = None
        self._frame_count = 0

        # Latest segmentation result — shared between handlers and display loop
        self._result_lock = threading.Lock()
        self._latest_result: Optional[dict] = None  # keys: mask, score, point_2d

        # Only one SAM2 inference at a time
        self._seg_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Subscriber thread — keeps latest camera frame up to date
    # ------------------------------------------------------------------

    def _sub_loop(self, ctx: zmq.Context) -> None:
        sock = ctx.socket(zmq.SUB)
        sock.connect(f"tcp://{self._sub_host}:{self._sub_port}")
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        logger.info("SUB socket connected to %s:%d", self._sub_host, self._sub_port)

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
    # REP loop — segmentation requests from external clients
    # ------------------------------------------------------------------

    def _rep_loop(self, ctx: zmq.Context) -> None:
        rep_sock = ctx.socket(zmq.REP)
        rep_sock.bind(f"tcp://0.0.0.0:{self._rep_port}")
        logger.info("REP socket listening on port %d", self._rep_port)
        logger.info("Ready. Waiting for segmentation requests...")

        try:
            while True:
                raw = rep_sock.recv()
                try:
                    request = msgpack.unpackb(raw, raw=False)
                    response = self._handle_rep_request(request)
                except Exception as exc:
                    logger.exception("Error handling request")
                    response = {"error": str(exc)}
                rep_sock.send(msgpack.packb(response, use_bin_type=True))
        except Exception:
            logger.exception("REP loop exited")
        finally:
            rep_sock.close()

    # ------------------------------------------------------------------
    # Segmentation — triggered by mouse clicks
    # ------------------------------------------------------------------

    def _run_segmentation(self, point_3d: np.ndarray) -> None:
        """Run SAM2 for a 3-D point and store the result. Called from mouse callback."""
        with self._frame_lock:
            rgb = self._latest_rgb
            intrinsics = self._latest_intrinsics
            frame_count = self._frame_count

        if rgb is None:
            logger.warning("No camera frame yet — cannot segment")
            return

        try:
            u, v = project_3d_to_2d(point_3d, intrinsics)
        except ValueError as e:
            logger.warning("Projection failed: %s", e)
            return

        h, w = rgb.shape[:2]
        u = max(0, min(u, w - 1))
        v = max(0, min(v, h - 1))

        logger.info(
            "Segmenting at pixel (%d, %d) from 3D (%.3f, %.3f, %.3f) [frame %d]",
            u, v, point_3d[0], point_3d[1], point_3d[2], frame_count,
        )

        t0 = time.monotonic()
        with self._seg_lock:
            with torch.inference_mode():
                self._predictor.set_image(rgb)
                masks, scores, _ = self._predictor.predict(
                    point_coords=np.array([[u, v]]),
                    point_labels=np.array([1]),
                    multimask_output=False,
                )
        elapsed_ms = (time.monotonic() - t0) * 1000

        mask = masks[0].astype(np.uint8)
        score = float(scores[0])

        logger.info(
            "SAM2 done in %.1f ms — coverage %.1f%%, score %.3f",
            elapsed_ms, mask.mean() * 100, score,
        )

        with self._result_lock:
            self._latest_result = {
                "mask": mask,
                "score": score,
                "point_2d": [u, v],
                "timing_ms": elapsed_ms,
            }

    # ------------------------------------------------------------------
    # REP handler — returns the latest stored result to ZMQ clients
    # ------------------------------------------------------------------

    def _handle_rep_request(self, _request: dict) -> dict:
        with self._result_lock:
            result = self._latest_result
        if result is None:
            return {"error": "No segmentation result yet. Click on the image to segment an object."}
        return result

    # ------------------------------------------------------------------
    # Mouse callback — click to segment (visualize mode only)
    # ------------------------------------------------------------------

    def _on_mouse_click(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        with self._frame_lock:
            depth = self._latest_depth
            intrinsics = self._latest_intrinsics

        if depth is None or intrinsics is None:
            logger.warning("No camera frame yet — cannot segment")
            return

        h, w = depth.shape
        if not (0 <= x < w and 0 <= y < h):
            return

        point_3d = backproject_2d_to_3d(x, y, depth, intrinsics)
        if point_3d is None:
            logger.warning("No depth at clicked pixel (%d, %d) — try a different point", x, y)
            return

        logger.info("Click at (%d, %d) → 3D (%.3f, %.3f, %.3f)", x, y, *point_3d)

        # Run segmentation in a worker thread so the display loop stays responsive
        threading.Thread(target=self._run_segmentation, args=(point_3d,), daemon=True).start()

    # ------------------------------------------------------------------
    # Display loop (main thread when --visualize is set)
    # ------------------------------------------------------------------

    def _display_loop(self) -> None:
        window = "SAM2 Segmentation  [click to segment | ESC to quit]"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, self._on_mouse_click)

        logger.info("Display window open — click on the image to segment an object")

        while True:
            with self._frame_lock:
                rgb = self._latest_rgb.copy() if self._latest_rgb is not None else None

            if rgb is None:
                # Show a placeholder until the first camera frame arrives
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for camera...", (160, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
                cv2.imshow(window, placeholder)
            else:
                with self._result_lock:
                    result = self._latest_result

                vis = self._draw_overlay(rgb, result)
                cv2.imshow(window, vis)

            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                logger.info("ESC pressed — shutting down")
                break

        cv2.destroyAllWindows()

    def _draw_overlay(self, rgb: np.ndarray, result: Optional[dict]) -> np.ndarray:
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if result is not None:
            mask = result["mask"]
            score = result["score"]
            u, v = result["point_2d"]
            elapsed_ms = result.get("timing_ms", 0)

            # Green mask overlay at 40% opacity
            green_layer = vis.copy()
            green_layer[mask == 1] = (
                green_layer[mask == 1] * 0.6 + np.array([0, 200, 0]) * 0.4
            ).astype(np.uint8)
            vis = green_layer

            # Prompt point
            cv2.circle(vis, (u, v), radius=6, color=(0, 0, 255), thickness=-1)
            cv2.circle(vis, (u, v), radius=8, color=(255, 255, 255), thickness=2)

            # Label
            label = f"score={score:.3f}  {elapsed_ms:.0f}ms"
            cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(vis, "Click to segment", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (200, 200, 200), 1, cv2.LINE_AA)

        return vis

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def serve_forever(self) -> None:
        ctx = zmq.Context()

        sub_thread = threading.Thread(target=self._sub_loop, args=(ctx,), daemon=True)
        sub_thread.start()

        if self._visualize:
            # REP runs in background; display loop owns the main thread
            rep_thread = threading.Thread(target=self._rep_loop, args=(ctx,), daemon=True)
            rep_thread.start()
            try:
                self._display_loop()
            finally:
                ctx.term()
        else:
            # REP runs in main thread (original headless behavior)
            try:
                self._rep_loop(ctx)
            except KeyboardInterrupt:
                logger.info("Shutting down")
            finally:
                ctx.term()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 ZMQ segmentation server")
    parser.add_argument("--checkpoint", required=True, help="Path to SAM2 .pt checkpoint")
    parser.add_argument("--config",     required=True, help="SAM2 Hydra config name or .yaml path")
    parser.add_argument("--sub-host",   default="localhost", help="IP/hostname of the camera publisher")
    parser.add_argument("--sub-port",   type=int, default=5560, help="ZMQ SUB port (camera publisher)")
    parser.add_argument("--rep-port",   type=int, default=5561, help="ZMQ REP port (segmentation requests)")
    parser.add_argument("--device",     default="cuda", help="torch device (cuda / cpu)")
    parser.add_argument("--visualize",  action="store_true",
                        help="Open an OpenCV window — click pixels to segment objects")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    server = SAM2Server(
        checkpoint=args.checkpoint,
        config=args.config,
        sub_host=args.sub_host,
        sub_port=args.sub_port,
        rep_port=args.rep_port,
        device=args.device,
        visualize=args.visualize,
    )
    server.serve_forever()
