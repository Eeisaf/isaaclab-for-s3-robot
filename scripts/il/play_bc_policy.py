from __future__ import annotations

"""Run a trained BC policy in the Isaac Lab reaching environment."""

import argparse
import sys
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play a behavior-cloned policy in Isaac Lab.")
parser.add_argument("--task", type=str, default="Template-Demo-Learn-v0", help="Gym task name.")
parser.add_argument("--checkpoint", type=str, required=True, help="BC checkpoint from train_bc_policy.py.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--ee_body", type=str, default="trunk_link4", help="End-effector body name.")
parser.add_argument("--success_threshold", type=float, default=0.05, help="Success distance threshold in meters.")
parser.add_argument("--max_steps", type=int, default=0, help="Maximum number of policy steps. 0 means run until closed.")
parser.add_argument("--print_interval", type=int, default=30, help="Print metrics every N steps.")
parser.add_argument("--real-time", action="store_true", default=False, help="Throttle playback to real time.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

import DEMO_learn.tasks  # noqa: F401

from bc_policy import load_bc_policy, policy_obs


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device
    policy, obs_mean, obs_std = load_bc_policy(args_cli.checkpoint, device)

    robot = env.unwrapped.scene["robot"]
    body_id = robot.find_bodies(args_cli.ee_body)[0][0]
    try:
        step_dt = env.step_dt
    except AttributeError:
        step_dt = env.unwrapped.step_dt

    obs, _ = env.reset()
    timestep = 0
    cumulative_successes = 0
    cumulative_timeouts = 0
    cumulative_episodes = 0
    print(f"[INFO] Loaded BC policy: {args_cli.checkpoint}")
    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Action space: {env.action_space}")

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            obs_tensor = policy_obs(obs).to(device)
            norm_obs = (obs_tensor - obs_mean) / obs_std
            actions = policy(norm_obs)
            obs, _, terminated, truncated, _ = env.step(actions)
            step_successes = int(terminated.sum().item())
            step_timeouts = int(truncated.sum().item())
            step_episodes = step_successes + step_timeouts
            cumulative_successes += step_successes
            cumulative_timeouts += step_timeouts
            cumulative_episodes += step_episodes

            if args_cli.print_interval > 0 and timestep % args_cli.print_interval == 0:
                target_pos = env.unwrapped.command_manager.get_command("target_pose")[:, :3]
                ee_pos = robot.data.body_pos_w[:, body_id] - robot.data.root_pos_w
                pos_error = torch.linalg.norm(ee_pos - target_pos, dim=-1)
                ee_quat_b = quat_mul(quat_inv(robot.data.root_quat_w), robot.data.body_quat_w[:, body_id, :])
                ee_up_axis_b = quat_apply(
                    ee_quat_b,
                    torch.tensor([0.0, 0.0, 1.0], device=device).expand(env.unwrapped.num_envs, -1),
                )
                success = (pos_error < args_cli.success_threshold) & (ee_up_axis_b[:, 2] > 0.98)
                cumulative_success_rate = cumulative_successes / max(cumulative_episodes, 1)
                cumulative_timeout_rate = cumulative_timeouts / max(cumulative_episodes, 1)
                print(
                    f"[BC] step={timestep:06d} "
                    f"mean_error={pos_error.mean().item():.4f}m "
                    f"min_error={pos_error.min().item():.4f}m "
                    f"up_dot={ee_up_axis_b[:, 2].mean().item():.3f} "
                    f"success_now={success.float().mean().item():.3f} "
                    f"step_success={step_successes} "
                    f"step_timeout={step_timeouts} "
                    f"episodes={cumulative_episodes} "
                    f"success_rate={cumulative_success_rate:.3f} "
                    f"timeout_rate={cumulative_timeout_rate:.3f}"
                )

        timestep += 1
        if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:
            break
        sleep_time = step_dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
