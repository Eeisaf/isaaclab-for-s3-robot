# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to test a trained RL agent with a custom arbitrary end-effector pose.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import math

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test a trained RL agent with a custom arbitrary target pose.")

# Custom pose arguments (用户自定义的目标位姿参数)
parser.add_argument("--pos_x", type=float, default=0.5, help="Target X position (m).")
parser.add_argument("--pos_y", type=float, default=0.0, help="Target Y position (m).")
parser.add_argument("--pos_z", type=float, default=1.0, help="Target Z position (m).")
parser.add_argument("--roll", type=float, default=0.0, help="Target roll angle (degrees).")
parser.add_argument("--pitch", type=float, default=0.0, help="Target pitch angle (degrees).")
parser.add_argument("--yaw", type=float, default=0.0, help="Target yaw angle (degrees).")

parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate (default: 1 for testing).")
parser.add_argument("--task", type=str, default="Template-Demo-Learn-v0", help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    help="Name of the RL algorithm to use.",
)
parser.add_argument("--real-time", action="store_true", default=True, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import time

import gymnasium as gym
import skrl
import torch

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import DEMO_learn.tasks  # noqa: F401

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent and custom pose."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    
    # ==========================================================
    # 核心逻辑：强制覆盖配置中的命令生成范围，将最大值与最小值设为一致
    # 从而使得随机生成的命令“退化”为固定的唯一用户指定位姿
    # ==========================================================
    
    # 角度转弧度
    roll_rad = math.radians(args_cli.roll)
    pitch_rad = math.radians(args_cli.pitch)
    yaw_rad = math.radians(args_cli.yaw)
    
    print(f"\n[INFO] =========================================")
    print(f"[INFO] 正在应用用户自定义目标位姿测试:")
    print(f"  位置 (X, Y, Z): ({args_cli.pos_x}, {args_cli.pos_y}, {args_cli.pos_z}) 米")
    print(f"  旋转 (Roll, Pitch, Yaw): ({args_cli.roll}, {args_cli.pitch}, {args_cli.yaw}) 度")
    print(f"[INFO] =========================================\n")
    
    # 覆盖 commands.target_pose.ranges 参数
    env_cfg.commands.target_pose.ranges.pos_x = (args_cli.pos_x, args_cli.pos_x)
    env_cfg.commands.target_pose.ranges.pos_y = (args_cli.pos_y, args_cli.pos_y)
    env_cfg.commands.target_pose.ranges.pos_z = (args_cli.pos_z, args_cli.pos_z)
    env_cfg.commands.target_pose.ranges.roll = (roll_rad, roll_rad)
    env_cfg.commands.target_pose.ranges.pitch = (pitch_rad, pitch_rad)
    env_cfg.commands.target_pose.ranges.yaw = (yaw_rad, yaw_rad)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    
    # 寻找最新模型文件
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "custom_pose_play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] 正在从以下路径加载模型权重: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.enable_training_mode(False, apply_to_models=True)

    obs, _ = env.reset()
    states = env.state()
    timestep = 0
    
    print("[INFO] =========================================")
    print("[INFO] 虚拟测试环境运行中 (图形界面已开启)... 按 Ctrl+C 停止测试")
    print("[INFO] =========================================\n")
    
    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
                
            obs, _, _, _, _ = env.step(actions)
            states = env.state()
            
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()