from __future__ import annotations

import argparse
import time
from pathlib import Path

from dex_benchmark.teleoperation.quest3 import (
    Quest3HandInput,
    Quest3InputFrame,
    Quest3InputStream,
    Quest3VuerConfig,
    Quest3VuerServer,
)


class InputReporter:
    def __init__(self, *, report_hz: float) -> None:
        if report_hz <= 0:
            raise ValueError(f"report_hz must be positive, got {report_hz}.")
        self.report_interval_s = 1.0 / report_hz
        self.last_report_time_s = 0.0
        self.last_report_sequence = 0
        self.last_source_timestamp_s: float | None = None
        self.last_received_timestamp_s: float | None = None
        self.latest_source_dt_ms: float | None = None
        self.latest_receive_dt_ms: float | None = None
        self.max_source_dt_ms = 0.0
        self.max_receive_dt_ms = 0.0

    def __call__(self, frame: Quest3InputFrame) -> None:
        now_s = time.monotonic()

        # Compute deltas for every frame, including frames skipped by the
        # report throttle. This preserves gaps that are later followed by a
        # burst of queued frames.
        self.latest_source_dt_ms = _timestamp_delta_ms(
            frame.source_timestamp_s,
            self.last_source_timestamp_s,
        )
        self.latest_receive_dt_ms = _timestamp_delta_ms(
            frame.received_timestamp_s,
            self.last_received_timestamp_s,
        )
        if self.latest_source_dt_ms is not None:
            self.max_source_dt_ms = max(self.max_source_dt_ms, self.latest_source_dt_ms)
        if self.latest_receive_dt_ms is not None:
            self.max_receive_dt_ms = max(
                self.max_receive_dt_ms,
                self.latest_receive_dt_ms,
            )
        self.last_source_timestamp_s = frame.source_timestamp_s
        self.last_received_timestamp_s = frame.received_timestamp_s

        if now_s - self.last_report_time_s < self.report_interval_s:
            return
        elapsed_s = now_s - self.last_report_time_s
        received_frames = frame.sequence - self.last_report_sequence
        measured_hz = received_frames / elapsed_s if self.last_report_time_s else 0.0
        print(
            " | ".join(
                [
                    f"seq={frame.sequence}",
                    _format_hand("left", frame.left),
                    _format_hand("right", frame.right),
                    f"receive_rate={measured_hz:.1f} Hz",
                    f"source_ts={frame.source_timestamp_s}",
                    f"received_ts={frame.received_timestamp_s:.6f}",
                    f"source_dt={_format_delta_ms(self.latest_source_dt_ms)}",
                    f"receive_dt={_format_delta_ms(self.latest_receive_dt_ms)}",
                    f"max_source_gap={_format_delta_ms(self.max_source_dt_ms or None)}",
                    "max_receive_gap="
                    f"{_format_delta_ms(self.max_receive_dt_ms or None)}",
                ]
            ),
            flush=True,
        )
        self.last_report_time_s = now_s
        self.last_report_sequence = frame.sequence
        self.max_source_dt_ms = 0.0
        self.max_receive_dt_ms = 0.0


def _timestamp_delta_ms(
    current_s: float | None,
    previous_s: float | None,
) -> float | None:
    if current_s is None or previous_s is None:
        return None
    delta_ms = 1000.0 * (float(current_s) - float(previous_s))
    return delta_ms if delta_ms >= 0.0 else None


def _format_delta_ms(value_ms: float | None) -> str:
    return "n/a" if value_ms is None else f"{value_ms:.1f}ms"


def _format_hand(side: str, hand: Quest3HandInput | None) -> str:
    if hand is None:
        return f"{side}=untracked"
    wrist_xyz_m = hand.wrist_matrix[:3, 3]
    gesture = hand.gesture
    gesture_text = "gesture=missing"
    if gesture is not None:
        gesture_text = (
            f"pinch={gesture.pinch}({gesture.pinch_value:.4f} m), "
            f"squeeze={gesture.squeeze}({gesture.squeeze_value:.4f} m)"
        )
    return (
        f"{side}=tracked wrist_webxr_xyz_m="
        f"[{wrist_xyz_m[0]:+.3f}, {wrist_xyz_m[1]:+.3f}, {wrist_xyz_m[2]:+.3f}] "
        f"{gesture_text}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive raw Meta Quest 3 hand tracking through Vuer/WebXR."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--report-hz", type=float, default=2.0)
    parser.add_argument("--cert", type=Path, default=None)
    parser.add_argument("--key", type=Path, default=None)
    parser.add_argument("--disable-left", action="store_true")
    parser.add_argument("--disable-right", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = Quest3VuerConfig(
        host=args.host,
        port=args.port,
        fps=args.fps,
        enable_left=not args.disable_left,
        enable_right=not args.disable_right,
        cert_path=args.cert,
        key_path=args.key,
    )
    stream = Quest3InputStream()
    server = Quest3VuerServer(
        stream,
        config=config,
        on_frame=InputReporter(report_hz=args.report_hz),
    )

    protocol = "wss" if args.cert is not None else "ws"
    print(f"Quest 3 input server listening on {protocol}://{args.host}:{args.port}", flush=True)
    if args.cert is None:
        print(
            "WebXR requires HTTPS/WSS. Put this server behind a TLS tunnel/proxy, "
            "or restart it with --cert and --key.",
            flush=True,
        )
    print(
        "Waiting for HAND_MOVE frames. No coordinate conversion, IK, or retargeting is active.",
        flush=True,
    )
    server.run()


if __name__ == "__main__":
    main()
