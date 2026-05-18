"""Lightweight SAM2 ZMQ client.

Fetches the latest segmentation result from the SAM2 server.
Segmentation is triggered by clicking in the server's visualization window.

Only requires: pyzmq, msgpack, msgpack-numpy, numpy.

Usage:
    from grasp_gen.serving.sam2_client import SAM2Client

    client = SAM2Client("localhost", rep_port=5561)
    mask, score = client.get_result()
"""

import logging
from typing import Optional

import msgpack
import msgpack_numpy
import numpy as np
import zmq

msgpack_numpy.patch()

logger = logging.getLogger(__name__)


class SAM2Client:
    def __init__(
        self,
        host: str = "localhost",
        rep_port: int = 5561,
        timeout_ms: int = 30_000,
    ) -> None:
        self._addr = f"tcp://{host}:{rep_port}"
        self._timeout_ms = timeout_ms
        self._ctx = zmq.Context()
        self._socket: Optional[zmq.Socket] = None

    def _ensure_connected(self) -> None:
        if self._socket is None:
            sock = self._ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
            sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(self._addr)
            self._socket = sock

    def get_result(self) -> tuple[np.ndarray, float]:
        """Fetch the latest segmentation result from the server.

        Segmentation is triggered by clicking in the server's --visualize window.

        Returns:
            mask:  (H, W) uint8 binary mask (1 = object, 0 = background).
            score: SAM2 confidence score for the mask.

        Raises:
            RuntimeError: if the server has no result yet or returns an error.
        """
        self._ensure_connected()
        self._socket.send(msgpack.packb({}, use_bin_type=True))
        raw = self._socket.recv()
        response = msgpack.unpackb(raw, raw=False)
        if "error" in response:
            raise RuntimeError(f"SAM2 server error: {response['error']}")
        mask = np.asarray(response["mask"], dtype=np.uint8)
        score = float(response["score"])
        logger.debug(
            "Received mask %s, score=%.3f, 2D point=%s, timing=%.1f ms",
            mask.shape, score, response.get("point_2d"), response.get("timing_ms", 0),
        )
        return mask, score

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._ctx.term()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
