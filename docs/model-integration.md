# Integrating a model

Evo-DexBench keeps model code out of the simulator. A model adapter implements
the small `dex_benchmark.policy_api` contract and returns physical semantic
actions; the environment converts those actions to the selected ManiSkill
controller. The same adapter works for single-arm and bimanual tasks and for
diagnostic or standard evaluation.

## Choose a deployment pattern

Two patterns are supported:

1. **In-process adapter.** Install a lightweight model package in the simulator
   environment and implement `PolicySession`. This is simplest when the model
   and simulator have compatible dependencies.
2. **Separate model server.** Keep CUDA/model dependencies in a dedicated
   environment and install only a small client adapter with the simulator. The
   adapter still implements `PolicySession`, but its inference method sends a
   request to the model process. The
   [KineDex example](../examples/policies/kinedex/README.md) demonstrates this
   pattern with WebSocket + MessagePack.

WebSocket is not mandated by the benchmark. HTTP, gRPC, ZeroMQ, shared memory,
or another transport can be used behind the adapter. The stable benchmark
boundary is `PolicySession` plus `SemanticAction`; wire payloads are owned and
versioned by each model adapter.

## Policy lifecycle

A factory receives `--policy-kwargs` and returns one session. The evaluator
then calls:

```text
factory(**policy_kwargs)
  -> bind_environment(metadata)   # optional, may be called for each wave
  -> reset()                      # once at the start of each episode wave
  -> act(model_input)             # once per control step
  -> close()                      # always during evaluator cleanup
```

The required session methods are:

```python
class PolicySession:
    def reset(self) -> None: ...
    def act(self, model_input: ModelInput) -> SemanticAction: ...
    def close(self) -> None: ...
```

Implement `bind_environment(metadata)` whenever the adapter needs to validate
the embodiment, action schema, cameras, tactile layout, or control frequency
before rollout. It should be idempotent.

The default protocol runs four simulator workers. Without native batching, the
evaluator creates one session per worker. A model that should be loaded only
once can instead implement:

```python
def act_batch(
    self, model_inputs: Sequence[ModelInput]
) -> Sequence[SemanticAction]: ...
```

One shared session will then receive all active slots in a single inference
call. A remote-server adapter can also keep independent lightweight sessions
while the server batches requests internally.

## Model input

Every call receives:

```python
{
    "instruction": "the current natural-language task instruction",
    "observation": {
        "cameras": {
            "context": {"rgb": ...},
            "interaction": {"rgb": ...},
            "wrist_left": {"rgb": ...},
            "wrist_right": {"rgb": ...},
        },
        "manipulators": {
            "left": {
                "joint_position": ...,
                "joint_velocity": ...,
                "end_effector_pose": ...,
                "palm_pose": ...,
                "fingertip_poses": ...,
                "taxel_normal_force": ...,
            },
            "right": {"...": "same fields"},
        },
    },
}
```

Only roles and modalities available for the selected embodiment and
`--obs-mode` should be consumed. RGB images are runtime `uint8 [H,W,3]` arrays;
`rgb_chw_float()` converts them to contiguous normalized `float32 [3,H,W]`.
Depth is exposed as `observation["cameras"][role]["depth"]` when requested
with `--obs-mode depth` or `rgb+depth`; it is not an RGB channel. Inspect the
actual shape and dtype before preprocessing rather than assuming the model's
depth representation or image resolution. The public handshake declares
camera roles, not a universal depth tensor format.

Each manipulator's `joint_position` contains 31 joints (7 arm + 24 hand).
The `compact_runtime_state()` helper builds a different 31D model state:
7D end-effector pose + 24 hand joint positions, excluding the arm joints.
The observation helpers also reject non-finite state values.

Use these common observation modes:

- state only: `--obs-mode state_dict --env-kwargs '{"render_backend":"none"}'`;
- RGB: `--obs-mode rgb`;
- RGB and depth: `--obs-mode rgb+depth`.

Camera roles are semantic and should not be replaced with simulator sensor
UIDs. Single-hand tasks expose the `right` manipulator; bimanual tasks expose
`left` and `right`.

## Semantic action

Return one mapping for every manipulator role declared by the environment:

```python
{
    "right": {
        "end_effector_delta_pose": np.zeros(6, dtype=np.float32),
        "hand_joint_position": hand_joint_targets.astype(np.float32),  # shape (24,)
    }
}
```

A bimanual action contains both `left` and `right` with the same two fields.
Here `hand_joint_targets` is the model's decoded physical target array, not
an uninitialized placeholder.
The values are physical, not normalized network outputs:

- `end_effector_delta_pose[:3]`: translation in metres in the robot-root frame;
- `end_effector_delta_pose[3:]`: XYZ Euler rotation delta in radians in the
  root-aligned body-rotation frame;
- `hand_joint_position`: absolute hand-joint targets in radians, ordered by the
  action specification.

Read `ActionSpec.from_metadata(metadata)` instead of assuming roles, joint
order, or bounds. The standard evaluator clips finite out-of-bound targets, but
an adapter should decode and validate its outputs correctly before returning
them. Missing roles, unknown fields, non-finite values, or incorrect shapes are
rejected.

## Minimal adapter skeleton

The following skeleton shows the complete benchmark-facing portion. Replace
`backend.predict()` with local inference or a transport client.

```python
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dex_benchmark.policy_api import ActionSpec, PolicyHandshake


class MyPolicySession:
    def __init__(self, *, checkpoint: str, use_tactile: bool = False) -> None:
        self.backend = load_my_backend(checkpoint)  # model-specific
        self.use_tactile = use_tactile
        self.action_spec: ActionSpec | None = None

    def bind_environment(self, metadata: Mapping[str, Any]) -> None:
        handshake = PolicyHandshake.from_environment_metadata(metadata)
        if handshake.protocol_version != "1.0":
            raise ValueError(f"Unsupported protocol {handshake.protocol_version}")
        if handshake.control_frequency_hz != 30:
            raise ValueError("This checkpoint expects 30 Hz control")
        self.action_spec = ActionSpec.from_metadata(metadata)

    def reset(self) -> None:
        self.backend.reset()

    def act(self, model_input: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
        if self.action_spec is None:
            raise RuntimeError("bind_environment() has not been called")
        raw = self.backend.predict(
            observation=model_input["observation"],
            instruction=model_input["instruction"],
        )
        result = {}
        for role, role_spec in self.action_spec.manipulators.items():
            hand_dim = int(role_spec["hand_joint_position"]["shape"][0])
            result[role] = {
                "end_effector_delta_pose": np.asarray(
                    raw[role]["ee_delta"], dtype=np.float32
                ).reshape(6),
                "hand_joint_position": np.asarray(
                    raw[role]["hand_qpos"], dtype=np.float32
                ).reshape(hand_dim),
            }
        return result

    def capture_visual_observation_after_action(self) -> bool:
        return True

    def capture_lowdim_observation_after_action(self) -> bool:
        return True

    def requires_control_period_tactile(self) -> bool:
        return self.use_tactile

    def close(self) -> None:
        self.backend.close()


def create_policy(**kwargs: Any) -> MyPolicySession:
    return MyPolicySession(**kwargs)
```

For an action-chunking policy, store the returned chunk in the session and emit
one semantic action per `act()` call. Return `False` from
`capture_visual_observation_after_action()` while buffered actions do not need a
new image. Keep `capture_lowdim_observation_after_action()` true if state or
tactile feedback is still needed. Return false from
`requires_control_period_tactile()` when tactile is unused; otherwise the
evaluator conservatively enables that work.

`ObservationSpec` and `PolicyCapabilities` can describe static adapter
requirements to external tools. Runtime modality selection is still explicit
through `--obs-mode`, and capture elision uses the three methods shown above.
RGB/depth selection does not independently enable tactile processing. The
evaluator uses `requires_control_period_tactile()` to enable both policy
tactile observations and control-period accumulation. Return `False` for a
non-tactile policy; task contact/success predicates remain active. During
action chunks, disable only observations that subsequent buffered actions
truly do not consume. The next replanning step must receive fresh inputs.

## Register the adapter

For quick local development, make the module importable and pass the factory
directly:

```bash
PYTHONPATH=/path/to/adapter \
dexbench evaluate \
  --policy my_policy:create_policy \
  --policy-kwargs '{"checkpoint":"/path/to/checkpoint"}' \
  --task-id m01_v1 \
  --episodes 1 --num-envs 1 --max-steps 10 \
  --obs-mode state_dict \
  --env-kwargs '{"render_backend":"none"}'
```

For a reusable package, publish a Python entry point:

```toml
[project]
name = "my-evodex-policy"
dependencies = ["dex-benchmark>=0.1.0"]

[project.entry-points."dex_benchmark.policies"]
my_model = "my_policy:create_policy"
```

Install the adapter in the simulator environment:

```bash
python -m pip install -e /path/to/my-evodex-policy
dexbench evaluate --help  # my_model should appear in the registered list
```

Then run the same diagnostic with `--policy my_model`. Keep heavy model
dependencies out of this adapter when using a separate server.

## Remote model example

With a remote adapter, use two terminals:

```bash
# Terminal 1: model environment
python -m my_model.server --checkpoint /path/to/checkpoint --port 8400

# Terminal 2: Evo-DexBench simulator environment
dexbench evaluate \
  --policy my_model --task-id m01_v1 --obs-mode rgb \
  --policy-kwargs '{"host":"127.0.0.1","port":8400}' \
  --episodes 1 --num-envs 1 --max-steps 10 \
  --log results/my_model_smoke.jsonl
```

The client should validate a versioned request/response schema, tensor shapes,
dtypes, checkpoint metadata, and errors before producing `SemanticAction`.
See `examples/policies/kinedex` for a working WebSocket implementation with
native batch inference, RGB cadence control, action chunks, and optional
tactile feedback.

## Standard evaluation

After the diagnostic run succeeds, remove every diagnostic override:

```bash
dexbench evaluate \
  --policy my_model \
  --task-id m01_v1 \
  --obs-mode rgb \
  --policy-kwargs '{"host":"127.0.0.1","port":8400}' \
  --log results/my_model_m01.jsonl
```

The resulting run uses the fixed 120/30 Hz, 100-episode protocol and is formal
only when the log records `standard_protocol=true`. Validate both single and
bimanual embodiments separately if the model claims to support both.

## Integration checklist

- The adapter imports without importing the model training stack into the
  simulator environment.
- `bind_environment()` rejects unsupported embodiment, frequency, cameras,
  tactile layout, action schema, and checkpoint metadata.
- `reset()` clears history and buffered action chunks.
- Outputs contain every declared role, correct joint order, physical units,
  finite `float32` values, and valid shapes.
- State-only execution does not render RGB; unused tactile processing is
  disabled.
- Native batching or the remote server handles four evaluation slots without
  loading four unnecessary model copies.
- A short diagnostic rollout passes before the unmodified standard protocol is
  used.
