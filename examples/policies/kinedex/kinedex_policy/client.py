from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class KineDexWebsocketClient:
    def __init__(self, host: str, port: int) -> None:
        import msgpack
        import msgpack_numpy
        from websockets.sync.client import connect

        msgpack_numpy.patch()
        self._msgpack = msgpack
        self._connection = connect(
            f"ws://{host}:{port}", max_size=None, compression=None
        )

    def infer(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        self._connection.send(self._msgpack.packb(observation, use_bin_type=True))
        result = self._msgpack.unpackb(self._connection.recv(), raw=False)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def close(self) -> None:
        self._connection.close()
