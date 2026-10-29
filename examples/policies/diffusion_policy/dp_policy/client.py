from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class DPWebsocketClient:
    """Small synchronous client that keeps DP dependencies off the sim process."""

    def __init__(self, host: str, port: int) -> None:
        try:
            import msgpack
            import msgpack_numpy
            from websockets.sync.client import connect
        except ImportError as exc:
            raise ImportError(
                "DP runtime requires msgpack, msgpack-numpy, and websockets."
            ) from exc
        msgpack_numpy.patch()
        self._msgpack = msgpack
        self._socket = connect(
            f"ws://{host}:{port}",
            max_size=None,
            compression=None,
        )

    def infer(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        self._socket.send(
            self._msgpack.packb(dict(observation), use_bin_type=True)
        )
        payload = self._socket.recv()
        response = self._msgpack.unpackb(payload, raw=False)
        if not isinstance(response, Mapping):
            raise TypeError("DP server response must be a mapping.")
        if "error" in response:
            raise RuntimeError(f"DP server error: {response['error']}")
        return response

    def close(self) -> None:
        self._socket.close()
