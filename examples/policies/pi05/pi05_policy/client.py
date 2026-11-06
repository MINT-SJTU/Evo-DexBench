from __future__ import annotations

from collections.abc import Mapping
import time
from typing import Any


class OpenPIWebsocketClient:
    """Synchronous client for the official OpenPI WebSocket policy server.

    It speaks the same protocol as ``openpi_client.WebsocketClientPolicy``
    and uses its MessagePack codec. Keepalive pings are disabled: the server
    runs inference inside its event loop, so several simulator workers sharing
    one server can delay pong replies beyond the default timeout.
    """

    def __init__(self, host: str, port: int, *, connect_timeout_s: float = 600.0) -> None:
        try:
            from openpi_client import msgpack_numpy
            from websockets.sync.client import connect
        except ImportError as exc:
            raise ImportError(
                "The π0.5 client requires the lightweight official openpi-client "
                "package. Install examples/policies/pi05 in the simulator "
                "environment; do not install the OpenPI training stack there."
            ) from exc
        self._codec = msgpack_numpy
        self._packer = msgpack_numpy.Packer()
        uri = f"ws://{host}:{port}"
        deadline = time.monotonic() + connect_timeout_s
        while True:
            try:
                self._socket = connect(uri, compression=None, max_size=None, ping_interval=None)
                break
            except (ConnectionRefusedError, OSError):
                if time.monotonic() >= deadline:
                    raise ConnectionError(f"No OpenPI policy server is reachable at {uri}.") from None
                time.sleep(2.0)
        metadata = self._codec.unpackb(self._socket.recv())
        self.server_metadata = metadata if isinstance(metadata, Mapping) else {}

    def infer(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        self._socket.send(self._packer.pack(dict(observation)))
        payload = self._socket.recv()
        if isinstance(payload, str):
            # The server sends a traceback string before closing on errors.
            raise RuntimeError(f"OpenPI policy server error:\n{payload}")
        response = self._codec.unpackb(payload)
        if not isinstance(response, Mapping):
            raise TypeError(
                f"OpenPI websocket response must be a mapping, got {type(response).__name__}."
            )
        return response

    def close(self) -> None:
        self._socket.close()
