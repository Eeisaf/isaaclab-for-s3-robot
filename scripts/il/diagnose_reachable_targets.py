from __future__ import annotations

"""Diagnose whether reachable_targets.pt joint solutions match the current Isaac Lab environment."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Check FK consistency and rollout quality for reachable target pairs.")
parser.add_argument("--task", type=str, default="Template-Demo-Learn-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel environments.")
parser.add_argument("--num_checks", type=int, default=4096, help="Number of reachable target pairs to check.")
parser.add_argument(
    "--reachable_targets",
    type=str,
    default="source/DEMO_learn/DEMO_learn/tasks/manager_based/demo_learn/reachable_targets.pt",
    help="Path to reachable_targets.pt containing positions and joint_pos.",
)
parser.add_argument("--ee_body", type=str, default="trunk_link4", help="End-effector body name.")
parser.add_argument("--success_threshold", type=float, default=0.05, help="Position success threshold.")
parser.add_argument("--settle_steps", type=int, default=4, help="Simulation steps after writing q_goal directly.")
parser.add_argument("--action_scale", type=float, default=0.05, help="DeltaJointPositionAction scale in radians.")
parser.add_argument("--rollout_steps", type=int, default=1800, help="Joint-action rollout steps to test.")
parser.add_argument("--print_worst", type=int, default=5, help="Print this many worst direct-FK samples.")
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


CONTROLLED_JOINT_NAMES = ["trunk_joint1", "trunk_joint2", "trunk_joint3", "trunk_joint4"]
ACTION_JOINT_NAMES = ["trunk_joint1", "trunk_joint2", "trunk_joint3"]


def _load_reachable_targets(path: str, device: str):
    data = torch.load(os.path.abspath(path), map_location="cpu")
    if not isinstance(data, dict) or "positions" not in data or "joint_pos" not in data:
        raise ValueError(f"Expected {path} to contain 'positions' and 'joint_pos'.")
    stored_joint_names = list(data.get("joint_names", []))
    if not stored_joint_names:
        raise ValueError(f"Expected {path} to contain 'joint_names'.")

    joint_cols = []
    for joint_name in CONTROLLED_JOINT_NAMES:
        if joint_name not in stored_joint_names:
            raise ValueError(f"Joint {joint_name!r} not found in stored joint names: {stored_joint_names}")
        joint_cols.append(stored_joint_names.index(joint_name))

    return data["positions"].float().to(device), data["joint_pos"].float().to(device)[:, joint_cols], stored_joint_names


def _sample_batch(positions: torch.Tensor, joint_pos: torch.Tensor, num: int):
    ids = torch.randint(0, positions.shape[0], (num,), device=positions.device)
    return ids, positions[ids], joint_pos[ids]


def _print_stats(name: str, values: torch.Tensor):
    quantiles = torch.quantile(values.float(), torch.tensor([0.0, 0.5, 0.9, 0.95, 0.99, 1.0], device=values.device))
    print(
        f"[{name}] mean={values.mean().item():.6f} "
        f"q0={quantiles[0].item():.6f} q50={quantiles[1].item():.6f} q90={quantiles[2].item():.6f} "
        f"q95={quantiles[3].item():.6f} q99={quantiles[4].item():.6f} q100={quantiles[5].item():.6f}"
    )


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device

    positions, stored_joint_pos, stored_joint_names = _load_reachable_targets(args_cli.reachable_targets, device)
    robot = env.unwrapped.scene["robot"]
    body_id = robot.find_bodies(args_cli.ee_body)[0][0]
    full_joint_ids, full_joint_names = robot.find_joints(CONTROLLED_JOINT_NAMES)
    action_joint_ids, action_joint_names = robot.find_joints(ACTION_JOINT_NAMES)

    print(f"[INFO] Stored joint names: {stored_joint_names}")
    print(f"[INFO] Environment controlled joint names: {full_joint_names}")
    print(f"[INFO] Environment action joint names: {action_joint_names}")
    print(f"[INFO] End-effector body: {args_cli.ee_body} id={body_id}")

    env.reset()
    num_total = 0
    direct_errors = []
    direct_ids = []
    while num_total < args_cli.num_checks:
        batch = min(args_cli.num_envs, args_cli.num_checks - num_total)
        target_ids, target_pos, q_goal = _sample_batch(positions, stored_joint_pos, batch)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos[:batch, full_joint_ids] = q_goal
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.reset()
        for _ in range(args_cli.settle_steps):
            env.unwrapped.sim.step(render=False)
            env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)

        ee_pos = robot.data.body_pos_w[:batch, body_id] - robot.data.root_pos_w[:batch]
        error = torch.linalg.norm(ee_pos - target_pos, dim=-1)
        direct_errors.append(error.detach().cpu())
        direct_ids.append(target_ids.detach().cpu())
        num_total += batch

    direct_errors_t = torch.cat(direct_errors, dim=0)
    direct_ids_t = torch.cat(direct_ids, dim=0)
    _print_stats("DIRECT_FK_ERROR", direct_errors_t.to(device))
    print(f"[DIRECT_FK] success_fraction={(direct_errors_t < args_cli.success_threshold).float().mean().item():.4f}")
    if args_cli.print_worst > 0:
        worst_errors, worst_order = torch.topk(direct_errors_t, k=min(args_cli.print_worst, direct_errors_t.numel()))
        print("[DIRECT_FK] worst samples:")
        for rank, (err, order_idx) in enumerate(zip(worst_errors.tolist(), worst_order.tolist(), strict=True), start=1):
            sample_id = int(direct_ids_t[order_idx].item())
            print(
                f"  rank={rank} sample_id={sample_id} error={err:.6f} "
                f"target={positions[sample_id].detach().cpu().tolist()} "
                f"q_goal={stored_joint_pos[sample_id].detach().cpu().tolist()}"
            )

    obs, _ = env.reset()
    command_term = env.unwrapped.command_manager.get_term("target_pose")
    target_ids, target_pos, q_goal = _sample_batch(positions, stored_joint_pos, env.unwrapped.num_envs)
    command_term.pose_command_b[:, :3] = target_pos
    command_term.pose_command_b[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    obs = env.unwrapped.observation_manager.compute(update_history=True)

    del obs
    done_once = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=device)
    success_once = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=device)
    timeout_once = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=device)
    last_joint_error = torch.zeros(env.unwrapped.num_envs, device=device)
    last_pos_error = torch.zeros(env.unwrapped.num_envs, device=device)
    for step in range(args_cli.rollout_steps):
        active_before = ~done_once
        current_q = robot.data.joint_pos[:, action_joint_ids]
        ee_pos = robot.data.body_pos_w[:, body_id] - robot.data.root_pos_w
        current_target = env.unwrapped.command_manager.get_command("target_pose")[:, :3]
        last_joint_error[active_before] = torch.linalg.norm(q_goal[active_before, :3] - current_q[active_before], dim=-1)
        last_pos_error[active_before] = torch.linalg.norm(ee_pos[active_before] - current_target[active_before], dim=-1)
        actions = torch.clamp((q_goal[:, :3] - current_q) / args_cli.action_scale, -1.0, 1.0)
        actions[done_once] = 0.0
        _, _, terminated, truncated, _ = env.step(actions)
        success_once |= terminated & active_before
        timeout_once |= truncated & active_before
        done_once |= (terminated | truncated) & active_before
        if done_once.all():
            break

    never_done = ~done_once
    print(f"[ROLLOUT] success_fraction={success_once.float().mean().item():.4f} over {step + 1} steps")
    print(f"[ROLLOUT] timeout_fraction={timeout_once.float().mean().item():.4f}")
    print(f"[ROLLOUT] never_done_fraction={never_done.float().mean().item():.4f}")
    failed = timeout_once | never_done
    if failed.any():
        _print_stats("ROLLOUT_FAILED_LAST_JOINT_ERROR", last_joint_error[failed])
        _print_stats("ROLLOUT_FAILED_LAST_POS_ERROR", last_pos_error[failed])
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
