from __future__ import annotations

import argparse
import signal
from typing import Any

from dex_benchmark.sim.tasks import make_task_env
from dex_benchmark.sim.generalization import GeneralizationAxis, GeneralizationProtocol
from dex_benchmark.teleoperation.keyboard_mouse import (
    KeyboardMouseTeleopConfig,
    KeyboardMouseTeleopRunner,
)


"""
Manual teleoperation examples (after installation, from the repository root):

conda activate evodex-sim

The combined simulation/VR environment ``evodex-vr`` can also be used.

dexbench run \
  --task-id m01_v1 \
  --layout-id studio_l1 \
  --control-mode arm_pd_ee_delta_pose_hand_pd_joint_pos \
  --obs-mode rgb+depth \
  --seed 42 \
  --idle-policy hold_step \
  --no-time-limit

To load one benchmark generalization axis and inspect the actual interaction
camera through the SAPIEN viewer:

dexbench run \
  --task-id m14_v1 \
  --layout-id studio_l1 \
  --generalization-axis camera \
  --generalization-seed 1001 \
  --viewer-camera interaction_camera \
  --debug-generalization

Or:
dexbench run \
    --task-id m14_v1 \
    --layout-id studio_l1 \
    --generalization-axis spatial \
    --generalization-seed 1001 \
    --debug-generalization \
    --idle-policy hold_step \
    --no-time-limit

Use ``--idle-policy hold_step`` instead when physics should continue while no
keyboard command is pressed.

Warning: do not use the main joint-position sliders in SAPIEN's Articulation
window to test contacts with movable contents. Those sliders call ``set_qpos``
directly and therefore teleport articulation links without a physically
integrated velocity. For a drawer, expand the joint details and use a finite
``Velocity Target`` or ``Position Target`` drive together with
``--idle-policy hold_step``.

Test tactile sensors with:
dexbench run \
    --task-id m14_v1 \
    --layout-id studio_l1 \
    --robot-uid panda_shadow_distributed_tactile_123 \
    --control-mode arm_pd_ee_delta_pose_hand_pd_joint_pos \
    --obs-mode rgb+depth \
    --seed 42 \
    --idle-policy hold_step \
    --print-tactile \
    --print-every-steps 1 \
    --tactile-force-threshold 0.05 \
    --tactile-top-k 8 \
    --no-time-limit
"""

[]
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    protocol = GeneralizationProtocol.load()
    parser.add_argument("--task-id", default="m01_v1")
    parser.add_argument("--layout-id", default=None)
    parser.add_argument(
        "--robot-uid",
        default=None,
        help=(
            "Optional single-robot debug override. Normally omit this so the "
            "task selects its fixed single- or dual-arm embodiment."
        ),
    )
    parser.add_argument("--control-mode", default="arm_pd_ee_delta_pose_hand_pd_joint_pos")
    parser.add_argument("--obs-mode", default="rgb+depth")
    parser.add_argument("--reward-mode", default="sparse")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Episode reset seed. This is separate from --generalization-seed.",
    )
    parser.add_argument(
        "--generalization-axis",
        choices=tuple(axis.value for axis in GeneralizationAxis),
        default=None,
        help="Load exactly one benchmark generalization axis.",
    )
    parser.add_argument(
        "--generalization-seed",
        type=int,
        choices=protocol.seeds,
        default=None,
        help="Fixed benchmark seed for the selected generalization axis.",
    )
    parser.add_argument(
        "--debug-generalization",
        action="store_true",
        help="Print every sampled generalization value when the environment loads.",
    )
    parser.add_argument(
        "--viewer-camera",
        choices=("free", "context_camera", "interaction_camera"),
        default="free",
        help=(
            "Initial SAPIEN viewer camera. Sensor choices use the exact camera "
            "pose and FOV loaded by the environment."
        ),
    )
    parser.add_argument("--translation-step", type=float, default=0.01)
    parser.add_argument("--rotation-step", type=float, default=0.03)
    parser.add_argument("--control-hz", type=int, default=20)
    parser.add_argument("--sim-freq", type=int, default=100)
    parser.add_argument(
        "--idle-policy",
        choices=("freeze_render", "hold_step"),
        default="freeze_render",
    )
    parser.add_argument("--shader", default="default")
    parser.add_argument(
        "--no-exit-on-success",
        action="store_true",
        help="Keep the viewer open after task success. Press R to reset or Q/Escape to quit.",
    )
    parser.add_argument(
        "--no-time-limit",
        action="store_true",
        help="Ignore time-limit truncation and keep teleoperation active until success or quit.",
    )
    parser.add_argument(
        "--accumulate-ee-target",
        action="store_true",
        help="Use ManiSkill's accumulated EE target instead of resetting target to current pose each step.",
    )
    parser.add_argument("--print-state", action="store_true")
    parser.add_argument(
        "--print-tactile",
        action="store_true",
        help=(
            "Print active-hand tactile force summaries and Top-K Taxel "
            "force/nominal-pressure values."
        ),
    )
    parser.add_argument("--print-every-steps", type=int, default=20)
    parser.add_argument(
        "--tactile-force-threshold",
        type=float,
        default=0.05,
        help="Only list Taxels at or above this normal-force threshold in N.",
    )
    parser.add_argument(
        "--tactile-top-k",
        type=int,
        default=8,
        help="Maximum number of active Taxels listed in each tactile summary.",
    )
    args = parser.parse_args(argv)
    if (args.generalization_axis is None) != (
        args.generalization_seed is None
    ):
        parser.error(
            "--generalization-axis and --generalization-seed must be provided together."
        )
    if args.debug_generalization and args.generalization_axis is None:
        parser.error(
            "--debug-generalization requires --generalization-axis and "
            "--generalization-seed."
        )
    return args


def make_env_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    env_kwargs: dict[str, Any] = {
        "control_mode": args.control_mode,
        "obs_mode": args.obs_mode,
        "reward_mode": args.reward_mode,
        "render_mode": "human",
        "sim_config": {
            "sim_freq": args.sim_freq,
            "control_freq": int(args.control_hz),
        },
        "viewer_camera_configs": {"shader_pack": args.shader},
    }
    if args.robot_uid is not None:
        env_kwargs["robot_uids"] = args.robot_uid
    if args.layout_id is not None:
        env_kwargs["layout_id"] = args.layout_id
    if args.generalization_axis is not None:
        env_kwargs["generalization_axis"] = args.generalization_axis
        env_kwargs["generalization_seed"] = args.generalization_seed
        env_kwargs["debug_generalization"] = args.debug_generalization
    return env_kwargs


def main(argv: list[str] | None = None) -> None:
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    args = parse_args(argv)

    print(
        "[teleop] Warning: SAPIEN's main articulation joint sliders directly "
        "set qpos and can cause tunneling. Use the detailed drive targets with "
        "--idle-policy hold_step when testing movable contents."
    )
    if args.generalization_axis is not None:
        print(
            "[teleop] Loading generalization "
            f"axis={args.generalization_axis!r}, "
            f"seed={args.generalization_seed}."
        )
    env = make_task_env(args.task_id, **make_env_kwargs(args))
    try:
        runner = KeyboardMouseTeleopRunner(
            env,
            config=KeyboardMouseTeleopConfig(
                translation_step_m=args.translation_step,
                rotation_step_rad=args.rotation_step,
                control_hz=args.control_hz,
                idle_policy=args.idle_policy,
                exit_on_success=not args.no_exit_on_success,
                no_time_limit=args.no_time_limit,
                reset_ee_target_each_step=not args.accumulate_ee_target,
                print_state=args.print_state,
                print_tactile=args.print_tactile,
                print_every_steps=args.print_every_steps,
                tactile_force_threshold_n=args.tactile_force_threshold,
                tactile_top_k=args.tactile_top_k,
                viewer_camera_uid=(
                    None if args.viewer_camera == "free" else args.viewer_camera
                ),
            ),
        )
        runner.run(seed=args.seed)
    finally:
        env.close()


if __name__ == "__main__":
    main()
