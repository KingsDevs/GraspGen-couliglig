# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import time
import logging
from typing import Optional

import numpy as np
import zmq
import msgpack
import msgpack_numpy

msgpack_numpy.patch()

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal

logger = logging.getLogger(__name__)


class GraspGenZMQServer:
    """ZMQ server that wraps GraspGenSampler for remote grasp inference.

    Protocol (msgpack over ZMQ REP socket):
        Request:  {"action": "infer", "point_cloud": ndarray(N,3), ...params}
                  {"action": "metadata"}
                  {"action": "health"}
        Response: msgpack-encoded dict with results or error.
    """

    def __init__(
        self,
        gripper_config: str,
        host: str = "0.0.0.0",
        port: int = 5556,
        scale: float = 1.0,
        max_points: int = 2000,
    ) -> None:
        self._host = host
        self._port = port
        self._gripper_config = gripper_config
        # Cap the input cloud size. GraspGen's outlier removal does an O(N^2)
        # torch.cdist (an N x N matrix), so a full-res masked cloud (~100k pts)
        # tries to allocate tens of GB and OOMs the GPU. The model expects a
        # downsampled object cloud anyway (the repo's mesh client uses 2000).
        # 0 disables the cap.
        self._max_points = int(max_points)
        # Scale bridge between real-world sensor units and the magnified space a
        # model was trained in. The Couliglig model was trained entirely in S=7
        # scaled space, so a real-meter cloud is 7x too small for it. We multiply
        # the incoming cloud by `scale` before inference and divide the predicted
        # grasp TRANSLATIONS by `scale` on the way out (rotations are scale-
        # invariant). scale=1.0 is a no-op — correct for the Robotiq model, which
        # was trained at real-world scale. See the Couliglig config notes.
        self._scale = float(scale)

        logger.info("Loading gripper config from %s", gripper_config)
        self._cfg = load_grasp_cfg(gripper_config)
        self._gripper_name = self._cfg.data.gripper_name
        self._model_name = self._cfg.eval.model_name

        logger.info(
            "Initializing GraspGenSampler (model=%s, gripper=%s, scale=%.4g)",
            self._model_name,
            self._gripper_name,
            self._scale,
        )
        self._sampler = GraspGenSampler(self._cfg)
        logger.info("Model loaded and ready for inference")

        self._metadata = {
            "gripper_name": self._gripper_name,
            "model_name": self._model_name,
            "gripper_config": gripper_config,
            "scale": self._scale,
            "max_points": self._max_points,
        }

    def serve_forever(self) -> None:
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REP)
        bind_addr = f"tcp://{self._host}:{self._port}"
        socket.bind(bind_addr)
        logger.info("GraspGen ZMQ server listening on %s", bind_addr)

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

    def _handle(self, request: dict) -> dict:
        action = request.get("action")
        if action == "health":
            return {"status": "ok"}
        if action == "metadata":
            return self._metadata
        if action == "infer":
            return self._handle_infer(request)
        return {"error": f"Unknown action: {action}"}

    def _handle_infer(self, request: dict) -> dict:
        point_cloud = request.get("point_cloud")
        if point_cloud is None:
            return {"error": "Missing required field 'point_cloud'"}

        point_cloud = np.asarray(point_cloud, dtype=np.float32)
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
            return {
                "error": f"point_cloud must be (N, 3), got {point_cloud.shape}"
            }

        # Downsample to bound the O(N^2) outlier-removal cdist (and match the
        # downsampled object cloud the model expects). No-op if already small.
        n_in = len(point_cloud)
        if self._max_points and n_in > self._max_points:
            idx = np.random.choice(n_in, self._max_points, replace=False)
            point_cloud = point_cloud[idx]
            logger.info("Downsampled cloud %d -> %d points", n_in, len(point_cloud))

        # Outlier removal here, in REAL scale, BEFORE up-scaling — its distance
        # threshold (0.014 m) is absolute, so running it after the x7 scale-up
        # would treat every neighbor gap as an outlier and cull the whole cloud.
        # A guard keeps the original cloud if removal would leave too few points.
        # We then pass remove_outliers=False so sample() doesn't redo it post-scale.
        remove_outliers = bool(request.get("remove_outliers", True))
        if remove_outliers:
            filtered, _ = point_cloud_outlier_removal(point_cloud)
            filtered = filtered.cpu().numpy().astype(np.float32)
            if len(filtered) >= 32:
                point_cloud = filtered
            else:
                logger.warning(
                    "Outlier removal left %d pts; keeping unfiltered cloud (%d)",
                    len(filtered), len(point_cloud),
                )

        # Up-scale the real-meter cloud into the model's trained (magnified)
        # space. No-op when scale == 1.0 (Robotiq).
        if self._scale != 1.0:
            point_cloud = point_cloud * np.float32(self._scale)

        params = {
            "grasp_threshold": float(request.get("grasp_threshold", -1.0)),
            "num_grasps": int(request.get("num_grasps", 200)),
            "topk_num_grasps": int(request.get("topk_num_grasps", -1)),
            "min_grasps": int(request.get("min_grasps", 40)),
            "max_tries": int(request.get("max_tries", 6)),
            "remove_outliers": False,  # already done above, in real scale
        }

        t0 = time.monotonic()
        grasps, grasp_conf = GraspGenSampler.run_inference(
            point_cloud, self._sampler, **params
        )
        infer_ms = (time.monotonic() - t0) * 1000

        if len(grasps) == 0:
            return {
                "grasps": np.empty((0, 4, 4), dtype=np.float32),
                "confidences": np.empty((0,), dtype=np.float32),
                "num_grasps": 0,
                "timing": {"infer_ms": infer_ms},
            }

        grasps_np = grasps.cpu().numpy().astype(np.float32)
        conf_np = grasp_conf.cpu().numpy().astype(np.float32)

        # De-scale predicted translations back to real-world meters. Rotations
        # are scale-invariant, so only the translation column changes.
        if self._scale != 1.0:
            grasps_np[:, :3, 3] /= np.float32(self._scale)

        logger.info(
            "Inferred %d grasps in %.1f ms (conf range %.3f - %.3f)",
            len(grasps_np),
            infer_ms,
            conf_np.min(),
            conf_np.max(),
        )

        return {
            "grasps": grasps_np,
            "confidences": conf_np,
            "num_grasps": len(grasps_np),
            "timing": {"infer_ms": infer_ms},
        }
