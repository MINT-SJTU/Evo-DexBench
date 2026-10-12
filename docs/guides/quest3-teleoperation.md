# Quest 3 teleoperation

The Quest 3 backend receives WebXR hand tracking through Vuer and drives a
single- or dual-arm Panda + Shadow Hand embodiment at a fixed control rate.
Validate connectivity, axes, and safety behavior before collecting data.

## Setup

Create the VR environment with the repository helper:

```bash
tools/setup_environment.sh --profile vr --name evodex-vr
conda activate evodex-vr
source /absolute/path/to/evodex-assets/activate_evodex_assets.sh
```

The VR profile includes simulation and data-collection dependencies; there is
no need to create an additional `evodex-sim` environment. Add
`--with-kinedex-client` to the setup command if this environment will also run
KineDex evaluation.

The host and headset must be mutually reachable. WebXR requires trusted
HTTPS/WSS, and the certificate subject alternative name must contain the LAN
address opened in Quest Browser.

```bash
eval "$(tools/quest3/create_lan_certificate.sh --export)"
```

If VPN routing selects the wrong interface, provide the headset-reachable IP:

```bash
eval "$(tools/quest3/create_lan_certificate.sh --ip 10.42.0.1 --export)"
```

On first use, start the teleoperation command and complete these steps in Quest
Browser:

1. Open `https://$QUEST3_LAN_IP:8013` and accept the certificate.
2. Open the Vuer URL printed by the command.
3. Grant hand-tracking and immersive-mode permissions.
4. Enter immersive mode and keep the required hand or hands visible until the
   terminal reports fresh tracking frames.

The helper exports `QUEST3_CERT`, `QUEST3_KEY`, and `QUEST3_LAN_IP`.

## Single-hand control

```bash
dexbench teleop quest3 \
  --mode single --hand-side right \
  --task-id m01_v1 --layout-id studio_l1 \
  --seed 42 \
  --host 0.0.0.0 --advertise-host "$QUEST3_LAN_IP" \
  --port 8013 --cert "$QUEST3_CERT" --key "$QUEST3_KEY" \
  --control-hz 30 --webxr-fps 30 --sim-freq 120 \
  --render-mode human --quest-view \
  --no-time-limit --no-exit-on-success
```

The hand side must match the task embodiment. Omit `--layout-id` to use the
task's first supported layout; specify it while debugging for reproducibility.

## Dual-hand control

```bash
dexbench teleop quest3 \
  --mode dual \
  --task-id b01_v1 --layout-id kitchen_l1 \
  --advertise-host "$QUEST3_LAN_IP" \
  --cert "$QUEST3_CERT" --key "$QUEST3_KEY" \
  --control-hz 30 --webxr-fps 30 --sim-freq 120 \
  --render-mode human --quest-view --no-time-limit
```

Dual mode requires both hands to remain tracked. Verify that the left and
right hands control the corresponding robots and that wrist and finger motion
are not mirrored.

## Tracking and control behavior

The defaults are 30 Hz control, 120 Hz simulation, 30 Hz WebXR input, and CPU
simulation. Increasing `--webxr-fps` can reduce input age but does not increase
the control rate.

`frame_delta` is the default wrist mode. The first valid frame anchors the
mapping instead of moving the robot to an absolute Quest-space pose. Invalid or
stale tracking holds control; sustained loss resets wrist calibration. Optional
gates include `--require-squeeze` and `--recenter-on-tap`.

Per-frame safety limits are controlled by `--max-position-delta`,
`--max-rotation-delta`, `--max-target-distance`, and
`--max-hand-joint-delta`. Contact filtering reduces inward arm and finger
motion at light and medium contact and stops additional force at the hard
limit. Do not raise these limits to work around tracking or scene problems.

## Host and Quest views

`--render-mode human` opens the desktop viewer. `--quest-view` publishes an
operator view to the headset without changing tracking. For a headless host:

```bash
dexbench teleop quest3 \
  --mode single --task-id m01_v1 --layout-id studio_l1 \
  --advertise-host "$QUEST3_LAN_IP" \
  --cert "$QUEST3_CERT" --key "$QUEST3_KEY" \
  --quest-view --quest-view-only --render-mode none
```

Bandwidth can be adjusted with `--quest-view-resolution`,
`--quest-view-fps`, and `--quest-view-quality`.

The optional `wmctrl` utility only applies requested desktop window geometry.
If it is unavailable, teleoperation continues with the window-manager-selected
size and position.

## Pre-collection checklist

1. Quest Browser accepts the certificate and grants WebXR permission.
2. Tracking frames arrive continuously with correct hand roles.
3. The first frame only anchors; small wrist motions produce small,
   directionally correct TCP motions.
4. Finger retargeting and per-frame joint limits behave correctly.
5. Light contact activates diagnostics and action filtering.
6. Tracking loss holds the robot, and recovery does not create a jump.
7. The measured control and image rates are close to their configured values;
   configured 30 Hz rates do not guarantee 30 Hz wall-clock throughput.

Use `dexbench teleop quest3-input --host 0.0.0.0 --port 8012` to inspect raw
tracking without starting a task. After the checklist passes, use the
[managed data collector](data-collection.md) for multi-episode recording.
