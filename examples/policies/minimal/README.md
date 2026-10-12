# Minimal policy example

The core package registers a dependency-free policy named `zero`. It validates
the public `PolicySession` lifecycle and emits one bounded semantic action for
the environment's declared embodiment.

```bash
dexbench evaluate \
  --policy zero --task-id m01_v1 \
  --episodes 1 --num-envs 1 --max-steps 5 \
  --obs-mode state_dict --env-kwargs '{"render_backend":"none"}'
```

Third-party packages can expose the same factory through the
`dex_benchmark.policies` Python entry-point group, or pass an explicit
`module:callable` factory string. Model code and model dependencies remain
outside the simulation package.

See the [model integration guide](../../../docs/model-integration.md) for the
observation and semantic-action schemas, lifecycle, batching hooks, plugin
packaging, and remote-server pattern.
