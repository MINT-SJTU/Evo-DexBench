# Exact replay of recorded episodes

`dexbench data replay --exact` re-simulates a canonical HDF5 episode and checks
that **every recorded robot joint position is reproduced bit for bit at every
step**. Recorded `success`/`stage_score` labels are compared and reported
separately because they were produced by the scoring definitions active when
the episode was collected.

```bash
dexbench data replay path/to/episode.h5 --exact --report replay_report.json
```

The report contains `physics_exact`, `first_qpos_difference_step`,
`max_joint_error_rad`, a `labels` section, the replay recipe that was used, the
CPU-numerics setup, and runtime versions. `status` is one of `exact`,
`physics_exact_labels_differ` or `diverged`.

## What exact replay needs

Use the released assets and the simulation environment installed by
`tools/setup_environment.sh --profile sim`: CPU physics with `torch==2.5.1`,
`sapien==3.0.3`, `mani_skill==3.0.1`, and `pin==2.7.0`.

Replay uses the recorded native actions and controller recovery events.
Bundled recipes in `dex_benchmark/resources/replay/` are selected automatically
by the HDF5 file's SHA-256, including the appropriate CPU-numerics mode. The
selected recipe and any inferred controller re-anchors are included in the
report. Recipes never change recorded data or comparison tolerances.

## Intel and AMD CPUs

The x86 approximate-reciprocal instructions `RCPPS`/`RSQRTPS` return different
values on Intel and AMD CPUs. They are executed by PhysX contact code
(`libsapien.so`) and by Intel MKL's portable vector-math kernels inside
`libtorch_cpu.so` (`sqrt`/`acos`/`asin`), which MKL always uses on AMD CPUs.
These small numerical differences can accumulate during contact simulation,
causing replay to diverge across CPU vendors even with identical code,
packages, and recorded actions.

`dex_benchmark.sim.determinism` removes this difference for episodes recorded on
AMD CPUs (`cpu_numerics: amd`): on an Intel host it sets `MKL_CBWR=COMPATIBLE`
and redirects every `RCPPS`/`RSQRTPS` in the affected functions, in memory only,
to a table emulation of the AMD instructions. The emulation is bit-exact (both
vendor tables were verified against hardware for all 2^32 inputs). On AMD hosts
nothing is patched. Libraries and patched byte windows are verified by SHA-256
and original bytes first; unsupported builds raise an error instead of being
patched.

Requirements: x86-64 Linux and a C compiler (`cc`). Bundled patch specs cover
the pinned `libsapien.so` (`sapien 3.0.3`) and `torch 2.5.1` CUDA 12.4 and 12.1
wheels; other builds need GNU binutils (`objdump`, `nm`) to generate a spec at
runtime.

Episodes with `cpu_numerics: native` were recorded on an Intel CPU and replay
exactly on Intel hosts; their Intel numerics cannot be emulated on AMD hosts.

Use `--cpu-numerics native` to disable the emulation (diagnostics only).

## Headless replay and NVIDIA driver shutdown

Some NVIDIA Vulkan drivers can crash in a `[vkrt] Analysis` thread during
library unloading, even after the replay report has been written. Disabling
rendering alone may not avoid driver initialization during device discovery.
Do not treat a report followed by a nonzero process exit as a successful run.

For state-only verification on Linux, select the Mesa software Vulkan ICD and
disable rendering for that process:

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  dexbench data replay path/to/episode.h5 --exact --render-backend none \
  --report replay_report.json
```

The example path is provided by `mesa-vulkan-drivers` on Ubuntu x86-64; verify
the ICD path on your host first. This workaround leaves CPU physics, recorded
actions and the episode's CPU-numerics recipe unchanged. It is for nonvisual
verification, not RGB/video replay. Keep the setting local to the replay
process rather than exporting it for GPU-rendered evaluation or teleoperation.

## Limits

* Exactness checks robot joint positions. It does not independently verify
  every object's full physics state or rendered pixels.
* `physics_exact_labels_differ` means the trajectory is reproduced but the
  recorded labels follow a scoring definition that has since changed.
* `expected: no_exact_replay_known` marks episodes without a validated exact
  replay recipe; it is not proof that the original demonstration was invalid.
