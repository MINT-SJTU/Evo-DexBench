# Evaluation

Use `dexbench evaluate` for both short integration checks and formal model
evaluation. The same model-independent Policy API is used in both cases.

## Standard protocol

The fixed execution settings are packaged in
`dex_benchmark/resources/eval/protocol_v1.yaml`; axes and seeds are in
`dex_benchmark/resources/generalization/protocol_v1.yaml`. A standard task run
contains:

- 100 episodes: 20 nominal and 20 for each of spatial, camera, lighting, and
  physical generalization;
- the task's three layouts, assigned 34/33/33 in their configured order;
- generalization seeds 1001–1005, each used four times within every generalized
  condition; episode reset seeds are derived separately;
- four CPU workers;
- 120 Hz simulation and 30 Hz control;
- the task horizon defined by its packaged YAML.

Each episode activates at most one generalization axis. Scheduling is
deterministic and independent of worker completion order.

Start the policy server in its own environment, then run the simulator client:

Install the lightweight KineDex client in the simulation environment using
`--with-kinedex-client` with the `sim` or `vr` setup profile. The model server
setup and checkpoint arguments are in the
[KineDex guide](../../examples/policies/kinedex/README.md#deployment).

```bash
dexbench evaluate \
  --policy kinedex --task-id m01_v1 \
  --control-mode arm_pd_ee_delta_pose_hand_pd_joint_pos \
  --obs-mode rgb \
  --policy-kwargs '{"host":"127.0.0.1","port":8400,"embodiment":"single","execute_steps":8}' \
  --log results/kinedex_m01.jsonl
```

The installed policy plugin supplies the `--policy` name. Do not pass
`--episodes`, `--num-envs`, or `--max-steps` for a standard result; those flags
are diagnostic overrides. A formal log must contain `standard_protocol=true`.
The protocol frequencies cannot be replaced through `--env-kwargs`.
Environment overrides are diagnostic and record `standard_protocol=false`,
including overrides that skip observation validation. The one audited exception
is the equivalent state-only optimization
`--obs-mode state_dict --env-kwargs '{"render_backend":"none"}'`, which avoids
creating an unused renderer without changing the policy observation. The run
header records all effective environment arguments with secrets redacted.

Omitting `--log` creates a unique task/run JSONL file under `log/evaluations/`.
An explicit `--log` must name a new file: existing logs are never overwritten.

## Short integration checks

Before a standard run, invoke the real policy with diagnostic overrides and a
separate log path:

```bash
dexbench evaluate \
  --policy kinedex --task-id m01_v1 \
  --episodes 1 --num-envs 1 --max-steps 3 \
  --policy-kwargs '{"host":"127.0.0.1","port":8400,"embodiment":"single"}' \
  --log results/kinedex_m01_smoke.jsonl
```

This checks the actual model transport, observations, action contract, and
environment. A run with overrides records `standard_protocol=false` and is not
a publishable benchmark result.

## Metrics and result checks

The public evaluation reports only:

- **SR** (`success_rate`): the fraction of episodes that satisfy the final task
  success predicate;
- **StageScore**: the mean highest stage score reached under each task's
  declared ordered or independent stage semantics.

Before publishing a standard result, verify 100 complete episode records, 20
records per condition, a 34/33/33 layout distribution in configured order,
four uses of each fixed generalization seed within every generalized condition,
the task YAML horizon, and
`standard_protocol=true`. See the [task catalog](../task-catalog.md) for public
instructions and stage definitions.
