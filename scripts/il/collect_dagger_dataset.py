from __future__ import annotations

"""Collect DAgger data by rolling out a BC policy and relabeling visited states with global sampled IK.

Unlike the initial expert collector, this script executes the learned policy in the simulator. For each state visited
by that policy, it stores the global IK expert action. This targets the distribution-shift problem where BC
drifts into states that were not present in the original expert-success trajectories.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect DAgger relabeling data for BC fine-tuning.")
parser.add_argument("--task", type=str, default="Template-Demo-Learn-v0", help="Gym task name.")
parser.add_argument("--checkpoint", type=str, required=True, help="BC checkpoint to roll out.")
parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel environments.")
parser.add_argument("--num_samples", type=int, default=200_000, help="Number of visited-state labels to collect.")
parser.add_argument("--output", type=str, default="datasets/demo_learn/dagger_iter1.pt", help="Output dataset path.")
parser.add_argument(
    "--reachable_targets",
    type=str,
    default="source/DEMO_learn/DEMO_learn/tasks/manager_based/demo_learn/reachable_targets.pt",
    help="Path to reachable_targets.pt used as the global sampled IK search space.",
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
parser.add_argument(
    "--global_ik_joint_weight",
    type=float,
    default=0.001,
    help="Tie-breaker weight favoring IK samples near the current joint state.",
)
parser.add_argument("--global_ik_up_axis_weight", type=float, default=1.0, help="Weight for enforcing end-effector +Z up.")
parser.add_argument("--global_ik_chunk_size", type=int, default=16384, help="Candidate chunk size for global IK search.")
parser.add_argument("--print_interval", type=int, default=25, help="Print progress every N rollout steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_tasks  # noqa: F401

import DEMO_learn.tasks  # noqa: F401

from bc_policy import load_bc_policy, policy_obs
from global_reachable_ik import GlobalReachableIK


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device

    policy, obs_mean, obs_std = load_bc_policy(args_cli.checkpoint, device)

    robot = env.unwrapped.scene["robot"]
    body_id = robot.find_bodies(args_cli.ee_body)[0][0]
    joint_ids, joint_names = robot.find_joints(args_cli.joint_names)
    global_ik = GlobalReachableIK(
        args_cli.reachable_targets,
        joint_names,
        device=device,
        joint_weight=args_cli.global_ik_joint_weight,
        up_axis_weight=args_cli.global_ik_up_axis_weight,
        chunk_size=args_cli.global_ik_chunk_size,
    )
    q_goals = torch.zeros((env.unwrapped.num_envs, len(joint_ids)), device=device)
    ik_pos_dist = torch.zeros(env.unwrapped.num_envs, device=device)
    ik_up_error = torch.zeros(env.unwrapped.num_envs, device=device)

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Rolling out BC policy: {args_cli.checkpoint}")
    print(f"[INFO] Relabeling with global reachable IK: {os.path.abspath(args_cli.reachable_targets)}")
    print(f"[INFO] Relabeling with joints: {joint_names}")
    print(f"[INFO] End-effector body: {args_cli.ee_body} body_id={body_id}")
    print(f"[INFO] Dataset target: {args_cli.output}")

    obs, _ = env.reset()
    target_pos = env.unwrapped.command_manager.get_command("target_pose")[:, :3]
    current_joint_pos = robot.data.joint_pos[:, joint_ids]
    q_goals[:], ik_pos_dist[:], ik_up_error[:] = global_ik.solve(target_pos, current_joint_pos)

    observations = []
    actions = []
    target_positions = []
    joint_positions = []
    ee_positions = []

    collected = 0
    step = 0
    cumulative_successes = 0
    cumulative_timeouts = 0
    cumulative_episodes = 0

    while simulation_app.is_running() and collected < args_cli.num_samples:
        with torch.inference_mode():
            obs_policy = policy_obs(obs).to(device)
            current_joint_pos = robot.data.joint_pos[:, joint_ids]
            expert_actions = torch.clamp((q_goals - current_joint_pos) / args_cli.action_scale, -1.0, 1.0)

            remaining = args_cli.num_samples - collected
            take = min(remaining, obs_policy.shape[0])
            observations.append(obs_policy.detach().cpu()[:take])
            actions.append(expert_actions.detach().cpu()[:take])
            target_positions.append(env.unwrapped.command_manager.get_command("target_pose").detach().cpu()[:take, :3])
            joint_positions.append(robot.data.joint_pos.detach().cpu()[:take, joint_ids])
            ee_positions.append((robot.data.body_pos_w[:, body_id] - robot.data.root_pos_w).detach().cpu()[:take])
            collected += take

            norm_obs = (obs_policy - obs_mean) / obs_std
            rollout_actions = policy(norm_obs)
            obs, _, terminated, truncated, _ = env.step(rollout_actions)
            done = terminated | truncated
            if done.any():
                done_ids = done.nonzero(as_tuple=False).squeeze(-1)
                target_pos = env.unwrapped.command_manager.get_command("target_pose")[done_ids, :3]
                current_joint_pos = robot.data.joint_pos[done_ids][:, joint_ids]
                q_goals[done_ids], ik_pos_dist[done_ids], ik_up_error[done_ids] = global_ik.solve(target_pos, current_joint_pos)

            step_successes = int(terminated.sum().item())
            step_timeouts = int(truncated.sum().item())
            cumulative_successes += step_successes
            cumulative_timeouts += step_timeouts
            cumulative_episodes += step_successes + step_timeouts
            step += 1

            if args_cli.print_interval > 0 and step % args_cli.print_interval == 0:
                target_pos = env.unwrapped.command_manager.get_command("target_pose")[:, :3]
                ee_pos = robot.data.body_pos_w[:, body_id] - robot.data.root_pos_w
                pos_error = torch.linalg.norm(ee_pos - target_pos, dim=-1)
                success_rate = cumulative_successes / max(cumulative_episodes, 1)
                timeout_rate = cumulative_timeouts / max(cumulative_episodes, 1)
                print(
                    f"[DAGGER] step={step} collected={collected}/{args_cli.num_samples} "
                    f"mean_error={pos_error.mean().item():.4f}m "
                    f"ik_nn_error={ik_pos_dist.mean().item():.4f}m "
                    f"ik_up_error={ik_up_error.mean().item():.4f} "
                    f"episodes={cumulative_episodes} success_rate={success_rate:.3f} timeout_rate={timeout_rate:.3f}"
                )

    dataset = {
        "observations": torch.cat(observations, dim=0),
        "actions": torch.cat(actions, dim=0),
        "target_positions": torch.cat(target_positions, dim=0),
        "joint_positions": torch.cat(joint_positions, dim=0),
        "ee_positions": torch.cat(ee_positions, dim=0),
        "metadata": {
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "joint_names": joint_names,
            "ee_body": args_cli.ee_body,
            "action_scale": args_cli.action_scale,
            "reachable_targets": os.path.abspath(args_cli.reachable_targets),
            "global_ik_joint_weight": args_cli.global_ik_joint_weight,
            "global_ik_up_axis_weight": args_cli.global_ik_up_axis_weight,
            "global_ik_chunk_size": args_cli.global_ik_chunk_size,
            "rolled_out_checkpoint": os.path.abspath(args_cli.checkpoint),
            "rollout_successes": cumulative_successes,
            "rollout_timeouts": cumulative_timeouts,
            "rollout_episodes": cumulative_episodes,
            "obs_dim": int(observations[0].shape[-1]),
            "action_dim": int(actions[0].shape[-1]),
        },
    }

    output_path = os.path.abspath(args_cli.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(dataset, output_path)
    print(f"[INFO] Saved {dataset['observations'].shape[0]} DAgger samples to: {output_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
