"""Lightweight ZMQ client for the SAM3 text-prompt grasp server.

Only depends on pyzmq, msgpack, msgpack-numpy, and numpy — no torch / CUDA needed.
Mirrors GraspGenClient (zmq_client.py).

Usage:
    from grasp_gen.serving.sam3_client import SAM3Client

    client = SAM3Client("localhost", 5557)
    result = client.infer(image, point_cloud, "the red mug")
    best = result["best_grasp"]   # None if nothing was found
"""

import logging
import time
from typing import Optional

import numpy as np
import zmq
import msgpack
import msgpack_numpy

msgpack_numpy.patch()

logger = logging.getLogger(__name__)


class SAM3Client:
    """Client that connects to a SAM3Server for text-prompted grasp inference."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5557,
        timeout_ms: int = 60_000,
        wait_for_server: bool = True,
        retry_interval_s: float = 2.0,
    ) -> None:
        self._addr = f"tcp://{host}:{port}"
        self._timeout_ms = timeout_ms
        self._ctx = zmq.Context.instance()
        self._socket: Optional[zmq.Socket] = None
        self._server_metadata: Optional[dict] = None

        if wait_for_server:
            self._wait_for_server(retry_interval_s)

    def _create_socket(self) -> zmq.Socket:
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self._addr)
        return sock

    def _wait_for_server(self, retry_interval_s: float) -> None:
        logger.info("Waiting for SAM3 server at %s ...", self._addr)
        while True:
            try:
                self._socket = self._create_socket()
                self._server_metadata = self._request({"action": "metadata"})
                logger.info("Connected to SAM3 server: %s", self._server_metadata)
                return
            except (zmq.error.Again, zmq.error.ZMQError):
                logger.info("Server not ready, retrying in %.1fs ...", retry_interval_s)
                if self._socket is not None:
                    self._socket.close()
                    self._socket = None
                time.sleep(retry_interval_s)

    def _ensure_connected(self) -> None:
        if self._socket is None:
            self._socket = self._create_socket()

    def _request(self, payload: dict) -> dict:
        self._ensure_connected()
        try:
            self._socket.send(msgpack.packb(payload, use_bin_type=True))
            raw = self._socket.recv()
        except zmq.error.ZMQError:
            # REQ sockets wedge after a timeout — drop and rebuild so the next
            # request can recover.
            self._socket.close()
            self._socket = None
            raise
        response = msgpack.unpackb(raw, raw=False)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    @property
    def server_metadata(self) -> Optional[dict]:
        return self._server_metadata

    def health_check(self) -> bool:
        try:
            resp = self._request({"action": "health"})
            return resp.get("status") == "ok"
        except Exception:
            return False

    def get_metadata(self) -> dict:
        return self._request({"action": "metadata"})

    def infer(self, image, point_cloud: np.ndarray, text: str) -> dict:
        """Segment `text` in `image` and return the best grasp on its 3D cloud.

        Args:
            image: RGB (H, W, 3) uint8 array, or a payload dict accepted by the
                server ({"data", "shape", "dtype"} or {"base64"}).
            point_cloud: organized (H, W, 3) or flat (N, 3) float32 cloud.
            text: concept prompt, e.g. "the red mug".

        Returns:
            Server result dict with `grasps`: a confidence-ranked list of candidate
            grasps (each {pose 4x4, position, rotation_matrix, confidence, rank}) in
            the `zed_camera` frame, meters. Also `best_grasp` (= grasps[0] or None),
            `num_candidates`, `timing`. Run your IK/MoveIt reachability check over
            `grasps` in order and take the first reachable one.
        """
        payload = {
            "action": "infer",
            "image": image,
            "point_cloud": np.asarray(point_cloud, dtype=np.float32),
            "text": text,
        }
        return self._request(payload)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
