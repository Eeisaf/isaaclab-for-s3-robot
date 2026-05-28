from __future__ import annotations

"""Collect expert trajectories using the joint solutions stored in reachable_targets.pt.

The reachable target file contains static FK pairs:

    target_position <-> joint_position_that_reaches_it

This script turns those pairs into simulated trajectories. It fixes each environment's target command to one sampled
target_position and drives the controlled joints toward the paired q_goal with the same normalized delta action used by
DeltaJointPositionAction.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect joint-reference trajectories from reachable target FK pairs.")
parser.add_argument("--task", type=str, default="Template-Demo-Learn-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel environments.")
parser.add_argument("--num_samples", type=int, default=200_000, help="Number of obs/action pairs to collect.")
parser.add_argument(
    "--reachable_targets",
    type=str,
    default="source/DEMO_learn/DEMO_learn/tasks/manager_based/demo_learn/reachable_targets.pt",
    help="Path to reachable_targets.pt containing positions and joint_pos.",
)
parser.add_argument(
    "--output",
    type=str,
    default="datasets/demo_learn/reachable_joint_expert.pt",
    help="Output dataset path.",
)
parser.add_argument("--ee_body", type=str, default="trunk_link4", help="End-effector body name.")
parser.add_argument(
    "--joint_names",
    type=str,
    nargs="+",
    default=["trunk_joint1", "trunk_joint2", "trunk_joint3"],
    help="Controlled joints in the same order as the RL action.",
)
parser.add_argument("--action_scale", type=float, default=0.05, help="DeltaJointPositionAction scale in radians.")
parser.add_argument("--traj_delta", type=float, default=0.025, help="Nominal joint-space interpolation step in radians.")
parser.add_argument("--min_traj_steps", type=int, default=30, help="Minimum number of reference trajectory steps.")
parser.add_argument("--hold_steps", type=int, default=60, help="Extra steps holding q_goal after the interpolation ends.")
parser.add_argument("--min_traj_len", type=int, default=2, help="Minimum successful trajectory length to keep.")
parser.add_argument("--print_interval", type=int, default=25, help="Print progress every N collection steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import DEMO_learn.tasks  # noqa: F401

from bc_policy import policy_obs


def _new_traj_buffer() -> dict[str, list[torch.Tensor]]:
    return {
        "observations": [],
        "actions": [],
        "target_positions": [],
        "joint_positions": [],
        "ee_positions": [],
        "q_goals": [],
    }


def _load_reachable_targets(path: str, device: str, controlled_joint_names: list[str]):
    data = torch.load(os.path.abspath(path), map_location="cpu")
    if not isinstance(data, dict) or "positions" not in data or "joint_pos" not in data:
        raise ValueError(f"Expected {path} to contain 'positions' and 'joint_pos'.")

    stored_joint_names = list(data.get("joint_names", []))
    if not stored_joint_names:
        raise ValueError(f"Expected {path} to contain 'joint_names'.")

    joint_cols = []
    for joint_name in controlled_joint_names:
        if joint_name not in stored_joint_names:
            raise ValueError(f"Joint {joint_name!r} not found in reachable target joint_names={stored_joint_names}")
        joint_cols.append(stored_joint_names.index(joint_name))

    positions = data["positions"].float().to(device)
    joint_pos = data["joint_pos"].float().to(device)[:, joint_cols]
    if positions.shape[0] != joint_pos.shape[0]:
        raise ValueError(f"positions and joint_pos have different sample counts: {positions.shape[0]} vs {joint_pos.shape[0]}")
    return positions, joint_pos, stored_joint_names


def _sample_targets(
    env,
    positions: torch.Tensor,
    joint_pos: torch.Tensor,
    q_starts: torch.Tensor,
    q_goals: torch.Tensor,
    traj_steps: torch.Tensor,
    traj_progress: torch.Tensor,
    env_ids: torch.Tensor,
    controlled_joint_ids: list[int],
):
    command_term = env.unwrapped.command_manager.get_term("target_pose")
    target_ids = torch.randint(0, positions.shape[0], (env_ids.shape[0],), device=positions.device)
    command_term.pose_command_b[env_ids, :3] = positions[target_ids]
    command_term.pose_command_b[env_ids, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=positions.device)
    q_starts[env_ids] = env.unwrapped.scene["robot"].data.joint_pos[env_ids][:, controlled_joint_ids]
    q_goals[env_ids] = joint_pos[target_ids]
    max_joint_distance = torch.max(torch.abs(q_goals[env_ids] - q_starts[env_ids]), dim=-1).values
    traj_steps[env_ids] = torch.clamp(
        torch.ceil(max_joint_distance / args_cli.traj_delta).long(),
        min=args_cli.min_traj_steps,
    )
    traj_progress[env_ids] = 0


def _append_step_to_buffers(
    buffers: list[dict[str, list[torch.Tensor]]],
    obs_policy: torch.Tensor,
    expert_actions: torch.Tensor,
    target_pos: torch.Tensor,
    joint_pos: torch.Tensor,
    ee_pos: torch.Tensor,
    q_goals: torch.Tensor,
):
    for env_id, buffer in enumerate(buffers):
        buffer["observations"].append(obs_policy[env_id])
        buffer["actions"].append(expert_actions[env_id])
        buffer["target_positions"].append(target_pos[env_id])
        buffer["joint_positions"].append(joint_pos[env_id])
        buffer["ee_positions"].append(ee_pos[env_id])
        buffer["q_goals"].append(q_goals[env_id])


def _flush_successful_trajectory(
    buffer: dict[str, list[torch.Tensor]],
    observations: list[torch.Tensor],
    actions: list[torch.Tensor],
    target_positions: list[torch.Tensor],
    joint_positions: list[torch.Tensor],
    ee_positions: list[torch.Tensor],
    q_goals: list[torch.Tensor],
    max_new_samples: int,
    min_traj_len: int,
) -> int:
    traj_len = len(buffer["observations"])
    if traj_len < min_traj_len or max_new_samples <= 0:
        return 0

    take = min(traj_len, max_new_samples)
    observations.append(torch.stack(buffer["observations"][:take], dim=0))
    actions.append(torch.stack(buffer["actions"][:take], dim=0))
    target_positions.append(torch.stack(buffer["target_positions"][:take], dim=0))
    joint_positions.append(torch.stack(buffer["joint_positions"][:take], dim=0))
    ee_positions.append(torch.stack(buffer["ee_positions"][:take], dim=0))
    q_goals.append(torch.stack(buffer["q_goals"][:take], dim=0))
    return take


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device

    positions, target_joint_pos, stored_joint_names = _load_reachable_targets(
        args_cli.reachable_targets, device, args_cli.joint_names
    )

    robot = env.unwrapped.scene["robot"]
    body_id = robot.find_bodies(args_cli.ee_body)[0][0]
    joint_ids, joint_names = robot.find_joints(args_cli.joint_names)
    q_starts = torch.zeros((env.unwrapped.num_envs, len(joint_ids)), device=device)
    q_goals = torch.zeros((env.unwrapped.num_envs, len(joint_ids)), device=device)
    traj_steps = torch.ones(env.unwrapped.num_envs, dtype=torch.long, device=device)
    traj_progress = torch.zeros(env.unwrapped.num_envs, dtype=torch.long, device=device)

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Reachable targets: {os.path.abspath(args_cli.reachable_targets)}")
    print(f"[INFO] Stored joint names: {stored_joint_names}")
    print(f"[INFO] Driving joints: {joint_names}")
    print(f"[INFO] Dataset target: {args_cli.output}")

    obs, _ = env.reset()
    all_env_ids = torch.arange(env.unwrapped.num_envs, device=device)
    _sample_targets(env, positions, target_joint_pos, q_starts, q_goals, traj_steps, traj_progress, all_env_ids, joint_ids)
    obs = env.unwrapped.observation_manager.compute(update_history=True)

    observations = []
    actions = []
    target_positions = []
    joint_positions = []
    ee_positions = []
    saved_q_goals = []
    buffers = [_new_traj_buffer() for _ in range(env.unwrapped.num_envs)]

    collected = 0
    successful_trajectories = 0
    discarded_trajectories = 0
    step = 0
    while simulation_app.is_running() and collected < args_cli.num_samples:
        with torch.inference_mode():
            current_joint_pos = robot.data.joint_pos[:, joint_ids]
            next_progress = torch.minimum(traj_progress + 1, traj_steps + args_cli.hold_steps)
            alpha = torch.clamp(next_progress.float() / traj_steps.float(), max=1.0).unsqueeze(-1)
            q_ref_next = q_starts + alpha * (q_goals - q_starts)
            expert_actions = torch.clamp((q_ref_next - current_joint_pos) / args_cli.action_scale, -1.0, 1.0)
            target_pos = env.unwrapped.command_manager.get_command("target_pose")[:, :3]
            ee_pos = robot.data.body_pos_w[:, body_id] - robot.data.root_pos_w

            _append_step_to_buffers(
                buffers,
                policy_obs(obs).detach().cpu(),
                expert_actions.detach().cpu(),
                target_pos.detach().cpu(),
                current_joint_pos.detach().cpu(),
                ee_pos.detach().cpu(),
                q_goals.detach().cpu(),
            )

            obs, _, terminated, truncated, _ = env.step(expert_actions)
            traj_progress = next_progress
            done = terminated | truncated
            if done.any():
                done_ids = done.nonzero(as_tuple=False).squeeze(-1)
                for env_id in done_ids.detach().cpu().tolist():
                    if bool(terminated[env_id]) and collected < args_cli.num_samples:
                        collected += _flush_successful_trajectory(
                            buffers[env_id],
                            observations,
                            actions,
                            target_positions,
                            joint_positions,
                            ee_positions,
                            saved_q_goals,
                            args_cli.num_samples - collected,
                            args_cli.min_traj_len,
                        )
                        successful_trajectories += 1
                    else:
                        discarded_trajectories += 1
                    buffers[env_id] = _new_traj_buffer()

                _sample_targets(
                    env,
                    positions,
                    target_joint_pos,
                    q_starts,
                    q_goals,
                    traj_steps,
                    traj_progress,
                    done_ids,
                    joint_ids,
                )
                obs = env.unwrapped.observation_manager.compute(update_history=True)

            step += 1
            if args_cli.print_interval > 0 and step % args_cli.print_interval == 0:
                target_pos = env.unwrapped.command_manager.get_command("target_pose")[:, :3]
                ee_pos = robot.data.body_pos_w[:, body_id] - robot.data.root_pos_w
                pos_error = torch.linalg.norm(ee_pos - target_pos, dim=-1)
                print(
                    f"[INFO] step={step} collected={collected}/{args_cli.num_samples} "
                    f"success_traj={successful_trajectories} discarded_traj={discarded_trajectories} "
                    f"mean_error={pos_error.mean().item():.4f}m"
                )

    if not observations:
        raise RuntimeError("No successful joint-reference trajectories were collected.")

    dataset = {
        "observations": torch.cat(observations, dim=0),
        "actions": torch.cat(actions, dim=0),
        "target_positions": torch.cat(target_positions, dim=0),
        "joint_positions": torch.cat(joint_positions, dim=0),
        "ee_positions": torch.cat(ee_positions, dim=0),
        "q_goals": torch.cat(saved_q_goals, dim=0),
        "metadata": {
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "reachable_targets": os.path.abspath(args_cli.reachable_targets),
            "stored_joint_names": stored_joint_names,
            "joint_names": joint_names,
            "ee_body": args_cli.ee_body,
            "action_scale": args_cli.action_scale,
            "traj_delta": args_cli.traj_delta,
            "min_traj_steps": args_cli.min_traj_steps,
            "hold_steps": args_cli.hold_steps,
            "min_traj_len": args_cli.min_traj_len,
            "successful_trajectories": successful_trajectories,
            "discarded_trajectories": discarded_trajectories,
            "obs_dim": int(observations[0].shape[-1]),
            "action_dim": int(actions[0].shape[-1]),
        },
    }

    output_path = os.path.abspath(args_cli.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(dataset, output_path)
    print(f"[INFO] Saved {dataset['observations'].shape[0]} samples to: {output_path}")
    print(f"[INFO] Kept successful trajectories: {successful_trajectories}")
    print(f"[INFO] Discarded timed-out trajectories: {discarded_trajectories}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
