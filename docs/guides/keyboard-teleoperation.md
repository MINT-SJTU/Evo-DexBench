# Keyboard teleoperation

The keyboard entry point is intended for task, layout, camera, stage-predicate,
and tactile debugging. It is not a replacement for managed Quest 3 data
collection.

## Start a task

```bash
dexbench run \
  --task-id m01_v1 \
  --layout-id studio_l1 \
  --seed 42 \
  --control-mode arm_pd_ee_delta_pose_hand_pd_joint_pos \
  --obs-mode rgb+depth \
  --control-hz 30 \
  --sim-freq 120 \
  --idle-policy hold_step \
  --viewer-camera interaction_camera \
  --no-time-limit \
  --no-exit-on-success
```

Studio tasks accept `studio_l1`–`studio_l3`; Kitchen tasks accept
`kitchen_l1`–`kitchen_l3`. Run `dexbench tasks show TASK_ID` to inspect a
task's supported scene and layouts. The standard evaluation timing is 120 Hz
simulation and 30 Hz control.

Common options include `--translation-step`, `--rotation-step`, `--seed`,
`--viewer-camera`, `--idle-policy`, `--print-state`, and `--print-tactile`.
Run `dexbench run --help` for the complete interface.

## Controls

Actions are interpreted in the robot-base frame, not in the freely movable
viewer-camera frame.

| Category | Key | Action |
| --- | --- | --- |
| TCP translation | `I` / `K` | +X / -X |
| TCP translation | `J` / `L` | +Y / -Y |
| TCP translation | `U` / `O` | +Z / -Z |
| TCP rotation | `1` / `2` | roll + / - |
| TCP rotation | `3` / `4` | pitch + / - |
| TCP rotation | `5` / `6` | yaw + / - |
| Hand preset | `Z` | open palm |
| Hand preset | `X` | power grasp |
| Hand preset | `C` | index point |
| Hand preset | `V` | precision pinch |
| Hand preset | `B` | lateral clip |
| Hand preset | `M` | tighten the active closed preset |
| Bimanual | `N` | switch the active left/right manipulator |
| Diagnostics | `G` | toggle actual hand-qpos change logging |
| Camera | `7` / `8` | align to context / interaction camera |
| Camera | `P` | print the current viewer pose |
| Robot base | arrow keys | translate in world XY |
| Robot base | `PageUp` / `PageDown` | translate in world Z |
| Robot base | `0` | print the world pose |
| Session | `R` | reset with the same seed |
| Session | `Q` / `Esc` | close the environment |
| Session | `H` | print key help |

Click the SAPIEN viewer before using keyboard controls so it owns input focus.

## Idle behavior and reset

`freeze_render` refreshes the viewer without stepping physics while no action
key is held. Dynamic objects, articulation drives, and persistence timers do
not progress. `hold_step` submits a holding action and continues simulation;
use it for contact and stability checks.

Do not validate dynamic contact with SAPIEN's direct qpos slider. It bypasses
normal action integration and can create penetration. Use a drive target or a
normal control action instead.

By default the command exits on success and at the task horizon. Use
`--no-exit-on-success` or `--no-time-limit` only for interactive debugging;
these flags do not change the formal evaluation protocol.

## Generalization debugging

Load one axis with one fixed protocol seed:

```bash
dexbench run \
  --task-id m14_v1 \
  --layout-id studio_l1 \
  --generalization-axis camera \
  --generalization-seed 1001 \
  --debug-generalization \
  --idle-policy hold_step
```

`--seed` controls episode reset, while `--generalization-seed` controls the
selected axis. They are independent random sources.

## Tactile diagnostics

```bash
dexbench run \
  --task-id m14_v1 \
  --layout-id studio_l1 \
  --idle-policy hold_step \
  --print-state --print-tactile \
  --print-every-steps 1 \
  --tactile-force-threshold 0.05 \
  --tactile-top-k 8
```

The tactile threshold is measured in newtons and affects diagnostic output
only. It does not change task success predicates or evaluation scores.
