# Keyboard and mouse teleoperation

See the [keyboard guide](../../../docs/guides/keyboard-teleoperation.md) for
controls and troubleshooting.

```bash
dexbench run --task-id m01_v1 --layout-id studio_l1
```

Use this entry point to inspect task loading, cameras, contacts, and stage
predicates. Translation and rotation steps, idle behavior, tactile diagnostics,
and a single generalization axis are configurable; run `dexbench run --help` for
the complete interface.

`freeze_render` does not advance physics while no action key is pressed. Use
`--idle-policy hold_step` when objects or articulation drives must keep moving.
Do not validate dynamic contact with SAPIEN's main articulation-position slider:
it sets joint positions directly instead of integrating a physical action.

To load a reproducible generalization sample:

```bash
dexbench run --task-id m01_v1 \
  --generalization-axis spatial --generalization-seed 1001 \
  --debug-generalization
```
