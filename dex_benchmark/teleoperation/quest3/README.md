# Quest 3 teleoperation

See the [Quest 3 guide](../../../docs/guides/quest3-teleoperation.md) for TLS,
LAN setup, tracking gates, and troubleshooting. See the
[data collection guide](../../../docs/guides/data-collection.md) for managed
multi-episode collection.

After installing the `vr` profile, start single- or dual-hand control with:

```bash
dexbench teleop quest3 --mode single --task-id m01_v1
dexbench teleop quest3 --mode dual --task-id b01_v1
```

Quest Browser must reach the host through trusted HTTPS/WSS. Certificate and
LAN helpers live under `tools/quest3/`; run each helper with `--help` for its
network options.

Before collecting data, verify:

- left/right hand roles and wrist axes;
- tracking cadence and continuity;
- end-effector, finger-joint, and contact limits;
- viewer and Quest stream cadence.

To inspect only the raw tracking stream:

```bash
dexbench teleop quest3-input --host 0.0.0.0 --port 8012
```

Use `dexbench data collect` for formal collection. It owns manifests, attempts,
generalization assignment, canonical recording, and clean shutdown. Stop and
diagnose tracking loss or abnormal contact instead of raising safety limits.
