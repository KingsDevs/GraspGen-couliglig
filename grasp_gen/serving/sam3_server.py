"""SAM3 text-prompt segmentation + grasp server.

Unlike the SAM2 server (sam2_server.py / rpc_server.py), SAM3 is prompted by a
TEXT string describing the object to pick (e.g. "the red mug") instead of a 2D
click point or a bounding box. SAM3 returns every instance matching the concept;
we keep the single highest-scoring instance, project its mask into the organized
point cloud, clean the cloud in 3D, and FORWARD it to the GraspGen ZMQ server
(zmq_server.py) for 6-DOF grasp inference.

Why a separate process for GraspGen: SAM3 needs transformers>=5 (-> a newer
huggingface_hub) which is incompatible with the ancient diffusers GraspGen
depends on. So this server runs in its own environment and talks to GraspGen
over ZMQ via GraspGenClient.

Protocol (msgpack over ZMQ REP socket), matching GraspGenZMQServer:
    Request:  {"action": "infer", "image": ..., "point_cloud": ndarray, "text": str}
              {"action": "metadata"}
              {"action": "health"}
    Response: msgpack-encoded dict with the best grasp / results or {"error": ...}.

Setup note: SAM3 (released Nov 2025) is a gated model — run `huggingface-cli login`
and accept the `facebook/sam3` terms once before first use.

Usage:
    python -m grasp_gen.serving.sam3_server --port 5557 --graspgen-port 5556
"""

import argparse
import base64
import logging
import os
import time
import zlib

import cv2
import msgpack
import msgpack_numpy
import numpy as np
import torch
import zmq
from PIL import Image
from sklearn.cluster import DBSCAN
from transformers import Sam3Model, Sam3Processor

from grasp_gen.serving.zmq_client import GraspGenClient

msgpack_numpy.patch()

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).lower() not in {"0", "false", "no"}


# ---------------------------------------------------------------------------
# Image / cloud decoding
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


# ---------------------------------------------------------------------------
# Grasp selection / formatting
# ---------------------------------------------------------------------------

def _select_grasp_index(
    grasps: np.ndarray, confidences: np.ndarray, selection: str
) -> int:
    """Pick a single grasp index from the candidates.

    "max_confidence" returns the top-scored grasp (jittery across diffusion runs).
    "spatial_median" picks the grasp closest to the median translation of the
    top-K, which is more stable across stochastic samples.
    """
    if selection == "max_confidence" or len(grasps) <= 1:
        return int(np.argmax(confidences))

    positions = grasps[:, :3, 3]
    median_position = np.median(positions, axis=0)
    distances = np.linalg.norm(positions - median_position, axis=1)
    return int(np.argmin(distances))


def _format_best_grasp(
    grasps: np.ndarray, confidences: np.ndarray, selection: str
) -> dict:
    best_idx = _select_grasp_index(grasps, confidences, selection)
    pose = grasps[best_idx].astype(np.float32, copy=False)
    confidence = float(confidences[best_idx])
    return {
        "best_grasp": {
            "pose": pose,
            "position": pose[:3, 3].astype(np.float32, copy=False),
            "rotation_matrix": pose[:3, :3].astype(np.float32, copy=False),
            "confidence": confidence,
            "candidate_index": best_idx,
            "selection": selection,
            "frame": "zed_camera",
            "units": "meters",
        },
        "num_grasps": 1,
        "num_candidates": int(len(grasps)),
        "best_confidence": confidence,
    }


def _empty_grasp_result(reason: str) -> dict:
    return {
        "best_grasp": None,
        "num_grasps": 0,
        "num_candidates": 0,
        "message": reason,
    }


def _draw_overlay(
    rgb: np.ndarray, mask: np.ndarray, u: int, v: int, score: float,
    n_pts: int, text: str,
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
    for color, thickness in (((255, 255, 255), 2), ((0, 0, 0), 1)):
        cv2.putText(
            bgr, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            color, thickness, cv2.LINE_AA,
        )
    return bgr


# ---------------------------------------------------------------------------
# 3D cloud cleanup (the 2D mask drags in table/background + flying pixels)
# ---------------------------------------------------------------------------

def _remove_support_plane(
    pts, seed_xyz, *, enabled, dist_thresh, min_ratio, ransac_iters,
    dbscan_min_samples,
):
    """RANSAC-fit the dominant plane and drop its inliers (the floor/table).

    DBSCAN can't separate an object from the surface it touches because they're
    one connected component. Removing the support plane cuts that link so the
    object survives clustering on its own.

    Skips removal (returns pts unchanged) when no plane explains at least
    `min_ratio` of the points, when the seed point lies on the best plane (a flat
    object, not the floor), or when removal would leave too few points to grasp.
    """
    n = len(pts)
    if not enabled or n < 3 * dbscan_min_samples:
        return pts

    rng = np.random.default_rng()
    best_inliers, best_count, best_plane = None, 0, None
    for _ in range(ransac_iters):
        p0, p1, p2 = pts[rng.choice(n, 3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(normal)
        if nn < 1e-9:
            continue
        normal = normal / nn
        inliers = np.abs((pts - p0) @ normal) < dist_thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers, best_plane = count, inliers, (normal, p0)

    if best_inliers is None or best_count < min_ratio * n:
        return pts  # no dominant plane to remove

    # Abort if the seed sits on the plane — that's the object, not the floor.
    if seed_xyz is not None and np.all(np.isfinite(seed_xyz)):
        normal, p0 = best_plane
        if abs(float((seed_xyz - p0) @ normal)) < dist_thresh:
            return pts

    keep = ~best_inliers
    if int(keep.sum()) < dbscan_min_samples:
        return pts
    return pts[keep]


def _largest_object_cluster(
    pts, seed_xyz, *, enabled, eps, min_samples,
):
    """Keep the dominant 3D cluster of the masked points.

    A 2D SAM3 mask projected into 3D drags in table/background points along the
    silhouette edge and stereo flying pixels at depth discontinuities. DBSCAN
    separates those into distinct clusters; we keep one and drop the rest.

    When `seed_xyz` is given, the cluster whose nearest member is closest to the
    seed is kept (so a small object beside a large background blob still wins);
    otherwise the largest cluster wins. Falls back to the input unchanged when
    clustering is disabled, the cloud is too small, or every point is noise.
    """
    if not enabled or len(pts) < min_samples:
        return pts

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(pts).labels_
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


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class SAM3Server:
    """ZMQ server: SAM3 text-prompt segmentation -> GraspGen forward.

    Loads SAM3 once, segments an RGB image by a text concept, projects the best
    instance mask into the organized point cloud, cleans it in 3D, and forwards
    the result to a GraspGen ZMQ server (via GraspGenClient) for grasp inference.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        host: str = "0.0.0.0",
        port: int = 5557,
        graspgen_host: str = "localhost",
        graspgen_port: int = 5556,
        graspgen_timeout_ms: int = 60_000,
        device: str = "cuda",
        fp16: bool = True,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        visualize: bool = False,
        max_markers: int = 1000,
        # GraspGen forwarding params
        num_grasps: int = 100,
        topk_num_grasps: int = 20,
        min_grasps: int = 10,
        max_tries: int = 3,
        remove_outliers: bool = True,
        min_points: int = 32,
        selection: str = "spatial_median",
        # 3D cleanup params
        dbscan: bool = True,
        dbscan_eps: float = 0.01,
        dbscan_min_samples: int = 10,
        plane_removal: bool = True,
        plane_dist_thresh: float = 0.005,
        plane_min_ratio: float = 0.25,
        plane_ransac_iters: int = 100,
    ) -> None:
        self._host = host
        self._port = port
        self._device = device
        self._fp16 = fp16
        self._dtype = torch.float16 if fp16 else torch.float32
        self._threshold = threshold
        self._mask_threshold = mask_threshold
        self._visualize = visualize
        self._max_markers = max_markers
        self._model_id = model_id

        self._num_grasps = num_grasps
        self._topk_num_grasps = topk_num_grasps
        self._min_grasps = min_grasps
        self._max_tries = max_tries
        self._remove_outliers = remove_outliers
        self._min_points = min_points
        self._selection = selection

        self._dbscan = dbscan
        self._dbscan_eps = dbscan_eps
        self._dbscan_min_samples = dbscan_min_samples
        self._plane_removal = plane_removal
        self._plane_dist_thresh = plane_dist_thresh
        self._plane_min_ratio = plane_min_ratio
        self._plane_ransac_iters = plane_ransac_iters

        logger.info("Loading SAM3 from %s (fp16=%s) ...", model_id, fp16)
        self._model = (
            Sam3Model.from_pretrained(model_id, torch_dtype=self._dtype)
            .to(device)
            .eval()
        )
        self._processor = Sam3Processor.from_pretrained(model_id)
        logger.info("SAM3 loaded.")

        # GraspGen runs in a separate process; forward cleaned clouds to it.
        self._graspgen = GraspGenClient(
            host=graspgen_host,
            port=graspgen_port,
            timeout_ms=graspgen_timeout_ms,
            wait_for_server=False,
        )
        logger.info(
            "GraspGen forwarding -> tcp://%s:%d (num_grasps=%d, topk=%d, "
            "min_grasps=%d, max_tries=%d, remove_outliers=%s, selection=%s)",
            graspgen_host, graspgen_port, num_grasps, topk_num_grasps,
            min_grasps, max_tries, remove_outliers, selection,
        )

        self._metadata = {
            "model_id": model_id,
            "port": port,
            "fp16": fp16,
            "threshold": threshold,
            "mask_threshold": mask_threshold,
            "graspgen_target": f"tcp://{graspgen_host}:{graspgen_port}",
            "selection": selection,
        }

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    def _handle(self, request: dict) -> dict:
        action = request.get("action")
        if action == "health":
            return {"status": "ok"}
        if action == "metadata":
            return self._metadata
        if action == "infer":
            return self._handle_infer(request)
        return {"error": f"Unknown action: {action}"}

    def _segment(self, rgb: np.ndarray, text: str):
        """Run SAM3 and return (mask uint8 HxW, score, n_instances) or None."""
        with torch.inference_mode():
            pil = Image.fromarray(rgb)
            inputs = self._processor(
                images=pil, text=text, return_tensors="pt"
            ).to(self._device)
            # Match float tensors (e.g. pixel_values) to the model dtype; leave
            # integer tensors (input_ids, attention masks) as-is.
            if self._fp16:
                inputs = inputs.to(self._dtype)
            outputs = self._model(**inputs)
            results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=self._threshold,
                mask_threshold=self._mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]

        if len(results["scores"]) == 0:
            return None

        best = int(results["scores"].argmax())
        mask = results["masks"][best].cpu().numpy().astype(np.uint8)
        score = float(results["scores"][best])
        return mask, score, int(len(results["scores"]))

    def _handle_infer(self, request: dict) -> dict:
        image = request.get("image")
        point_cloud = request.get("point_cloud")
        text = request.get("text")
        if image is None or point_cloud is None or text is None:
            return _empty_grasp_result(
                "infer requires 'image', 'point_cloud', and 'text'"
            )

        t_start = time.perf_counter()
        rgb = _decode_image_payload(image)
        client_frame = image.get("frame") if isinstance(image, dict) else None
        client_checksum = image.get("checksum") if isinstance(image, dict) else None
        server_checksum = zlib.crc32(np.ascontiguousarray(rgb).tobytes()) & 0xFFFFFFFF
        logger.info(
            "image client_frame=%s client_crc=%s server_crc=%s text=%r",
            client_frame, client_checksum, server_checksum, text,
        )

        pc = _decode_point_cloud(point_cloud)  # (H_c, W_c, 3) organized
        t_decode = time.perf_counter()

        finite_map = (
            np.isfinite(pc).all(axis=-1)
            if pc.ndim == 3
            else np.isfinite(pc).all(axis=1)
        )

        seg = self._segment(rgb, text)
        t_sam = time.perf_counter()
        if seg is None:
            logger.info("SAM3 found no instance for %r", text)
            return _empty_grasp_result(f"SAM3 found no instance for '{text}'")
        mask, score, n_instances = seg
        logger.info(
            "SAM3 found %d instance(s) for %r; best score=%.3f, coverage=%.1f%%",
            n_instances, text, score, mask.mean() * 100,
        )

        # Mask centroid (image coords) — stands in for the SAM2 click point: drives
        # both the overlay marker and the 3D seed for cluster selection.
        ys, xs = np.where(mask == 1)
        if len(xs) > 0:
            u, v = int(round(xs.mean())), int(round(ys.mean()))
        else:
            u, v = -1, -1

        seed_xyz = None  # 3D anchor under the mask centroid, for cluster selection
        mask_resized = None
        if pc.ndim == 3:
            H_c, W_c = pc.shape[:2]
            mask_resized = cv2.resize(
                mask, (W_c, H_c), interpolation=cv2.INTER_NEAREST
            )
            masked_pixels = int(mask_resized.sum())
            pts = pc[mask_resized == 1]
            finite_mask = np.isfinite(pts).all(axis=1)
            logger.info(
                "Cloud (%dx%d), image (%dx%d), masked_pixels=%d, finite=%d",
                H_c, W_c, rgb.shape[0], rgb.shape[1], masked_pixels,
                int(finite_mask.sum()),
            )
            pts = pts[finite_mask]

            # 3D point under the mask centroid (image coords -> cloud coords).
            if u >= 0 and v >= 0:
                u_c = max(0, min(int(round(u * W_c / rgb.shape[1])), W_c - 1))
                v_c = max(0, min(int(round(v * H_c / rgb.shape[0])), H_c - 1))
                cand = pc[v_c, u_c]
                if np.all(np.isfinite(cand)):
                    seed_xyz = cand

            # Textureless surfaces (e.g. cubes) cause ZED depth NaN inside the
            # mask. Fall back to a dilated mask to capture boundary/edge points.
            if len(pts) == 0 and masked_pixels > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                mask_dilated = cv2.dilate(mask_resized, kernel)
                pts_d = pc[mask_dilated == 1]
                pts = pts_d[np.isfinite(pts_d).all(axis=1)]
                logger.info("Dilation fallback: %d boundary points", len(pts))
        else:
            pts = pc  # flat cloud — return as-is

        # Drop table/background contamination left by the 2D mask: first remove
        # the support plane the object rests on, then keep the dominant cluster.
        if len(pts) >= self._dbscan_min_samples:
            n_before = len(pts)
            pts = _remove_support_plane(
                pts, seed_xyz,
                enabled=self._plane_removal,
                dist_thresh=self._plane_dist_thresh,
                min_ratio=self._plane_min_ratio,
                ransac_iters=self._plane_ransac_iters,
                dbscan_min_samples=self._dbscan_min_samples,
            )
            n_plane = len(pts)
            pts = _largest_object_cluster(
                pts, seed_xyz,
                enabled=self._dbscan,
                eps=self._dbscan_eps,
                min_samples=self._dbscan_min_samples,
            )
            logger.info(
                "3D cleanup: %d -> %d (plane) -> %d (cluster) points (seed=%s)",
                n_before, n_plane, len(pts), "yes" if seed_xyz is not None else "no",
            )

        t_mask = time.perf_counter()

        grasp_result = self._infer_grasp(pts)
        t_graspgen = time.perf_counter()

        if self._visualize:
            self._show(rgb, mask, mask_resized, finite_map, pc, u, v, score,
                       len(pts), text)
        t_vis = time.perf_counter()

        logger.info(
            "Segmented %d 3D points | decode=%.1fms, sam=%.1fms, mask=%.1fms, "
            "graspgen=%.1fms, vis=%.1fms",
            len(pts), (t_decode - t_start) * 1000, (t_sam - t_decode) * 1000,
            (t_mask - t_sam) * 1000, (t_graspgen - t_mask) * 1000,
            (t_vis - t_graspgen) * 1000,
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

    def _infer_grasp(self, pts: np.ndarray) -> dict:
        if pts.ndim != 2 or pts.shape[1] != 3:
            return _empty_grasp_result(
                f"Segmented point cloud must be (N, 3), got {pts.shape}"
            )
        if len(pts) < self._min_points:
            return _empty_grasp_result(
                f"Only {len(pts)} segmented points; "
                f"need at least {self._min_points}"
            )

        # Forward the cleaned cloud to the GraspGen server (separate process).
        try:
            grasps_np, confidences_np = self._graspgen.infer(
                pts.astype(np.float32, copy=False),
                grasp_threshold=-1.0,
                num_grasps=self._num_grasps,
                topk_num_grasps=self._topk_num_grasps,
                min_grasps=self._min_grasps,
                max_tries=self._max_tries,
                remove_outliers=self._remove_outliers,
            )
        except (zmq.error.ZMQError, RuntimeError) as exc:
            return _empty_grasp_result(f"GraspGen forward failed: {exc}")

        if len(grasps_np) == 0:
            return _empty_grasp_result("GraspGen returned no grasps")

        grasps_np = np.asarray(grasps_np, dtype=np.float32)
        confidences_np = np.asarray(confidences_np, dtype=np.float32)
        grasps_np[:, 3, 3] = 1
        return _format_best_grasp(grasps_np, confidences_np, self._selection)

    def _show(self, rgb, mask, mask_resized, finite_map, pc, u, v, score,
              n_pts, text) -> None:
        vis = _draw_overlay(rgb, mask, u, v, score, n_pts, text)
        # Drawing every masked pixel dominates latency for large masks.
        if (pc.ndim == 3 and mask_resized is not None and n_pts > 0
                and self._max_markers > 0):
            H_c, W_c = pc.shape[:2]
            scale_r = rgb.shape[0] / H_c
            scale_c = rgb.shape[1] / W_c
            coords = np.column_stack(np.where((mask_resized == 1) & finite_map))
            if len(coords) > self._max_markers:
                step = int(np.ceil(len(coords) / self._max_markers))
                coords = coords[::step][:self._max_markers]
            for r, c in coords:
                px = (int(c * scale_c), int(r * scale_r))
                cv2.drawMarker(vis, px, (255, 0, 255), cv2.MARKER_STAR, 12, 2)
        cv2.imshow("SAM3 Segmentation", vis)
        cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def serve_forever(self) -> None:
        ctx = zmq.Context.instance()
        socket = ctx.socket(zmq.REP)
        bind_addr = f"tcp://{self._host}:{self._port}"
        socket.bind(bind_addr)
        logger.info("SAM3 server listening on %s", bind_addr)

        try:
            while True:
                raw = socket.recv()
                try:
                    request = msgpack.unpackb(raw, raw=False)
                    response = self._handle(request)
                except Exception as exc:
                    logger.exception("Error handling request")
                    response = {"error": str(exc)}
                socket.send(msgpack.packb(response, use_bin_type=True))
        except KeyboardInterrupt:
            logger.info("Shutting down server")
        finally:
            socket.close()
            ctx.term()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SAM3 text-prompt segmentation + grasp server"
    )
    parser.add_argument("--model-id", default=os.getenv("SAM3_MODEL_ID", "facebook/sam3"),
                        help="Hugging Face model id for SAM3")
    parser.add_argument("--host", default="0.0.0.0", help="ZMQ REP bind address")
    parser.add_argument("--port", type=int, default=int(os.getenv("RPC_PORT", "5557")),
                        help="ZMQ REP port (segmentation + grasp requests)")
    parser.add_argument("--graspgen-host", default=os.getenv("GRASPGEN_HOST", "localhost"),
                        help="Host of the GraspGen ZMQ server")
    parser.add_argument("--graspgen-port", type=int,
                        default=int(os.getenv("GRASPGEN_PORT", "5556")),
                        help="Port of the GraspGen ZMQ server")
    parser.add_argument("--device", default="cuda", help="torch device (cuda / cpu)")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false",
                        help="Load SAM3 in fp32 instead of fp16")
    parser.set_defaults(fp16=_env_flag("SAM3_FP16", "1"))
    parser.add_argument("--threshold", type=float,
                        default=float(os.getenv("SAM3_THRESHOLD", "0.5")),
                        help="Instance score threshold")
    parser.add_argument("--mask-threshold", type=float,
                        default=float(os.getenv("SAM3_MASK_THRESHOLD", "0.5")),
                        help="Per-mask binarization threshold")
    parser.add_argument("--visualize", action="store_true",
                        default=_env_flag("RPC_VISUALIZE", "0"),
                        help="Open an OpenCV window showing the segmentation")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    server = SAM3Server(
        model_id=args.model_id,
        host=args.host,
        port=args.port,
        graspgen_host=args.graspgen_host,
        graspgen_port=args.graspgen_port,
        device=args.device,
        fp16=args.fp16,
        threshold=args.threshold,
        mask_threshold=args.mask_threshold,
        visualize=args.visualize,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
