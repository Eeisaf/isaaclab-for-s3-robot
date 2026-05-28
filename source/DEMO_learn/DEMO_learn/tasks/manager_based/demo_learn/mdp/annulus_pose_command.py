from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class AnnulusPoseCommand(UniformPoseCommand):
    """Sample target position on X-Z annulus, with Y fixed/ranged by cfg.ranges.pos_y."""

    cfg: AnnulusPoseCommandCfg

    def __init__(self, cfg: AnnulusPoseCommandCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._reachable_targets = None
        if self.cfg.reachable_targets_path:
            targets_path = os.path.abspath(self.cfg.reachable_targets_path)
            if not os.path.isfile(targets_path):
                raise FileNotFoundError(
                    f"Reachable target point cloud not found: {targets_path}. "
                    "Run scripts/generate_reachable_targets.py before training."
                )

            data = torch.load(targets_path, map_location="cpu")
            if isinstance(data, dict):
                data = data.get("positions", data.get("target_positions"))
            if data is None:
                raise ValueError(f"Reachable target file does not contain 'positions': {targets_path}")

            targets = torch.as_tensor(data, dtype=torch.float32, device=self.device)
            if targets.ndim != 2 or targets.shape[1] != 3:
                raise ValueError(
                    f"Reachable targets must have shape (N, 3), got {tuple(targets.shape)} from {targets_path}"
                )
            if targets.shape[0] == 0:
                raise ValueError(f"Reachable target file is empty: {targets_path}")
            self._reachable_targets = targets

    def _resample_command(self, env_ids: Sequence[int]):
        num = len(env_ids)
        if num == 0:
            return

        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        r = torch.empty(num, device=self.device)

        if self._reachable_targets is not None:
            target_ids = torch.randint(0, self._reachable_targets.shape[0], (num,), device=self.device)
            self.pose_command_b[env_ids_t, :3] = self._reachable_targets[target_ids]
        elif self.cfg.sample_reachable_poses:
            # Sample goals from actually observed end-effector poses. Every sampled target is reachable by
            # at least one valid robot state, avoiding unreachable random Cartesian commands.
            source_env_ids = torch.randint(0, self.num_envs, (num,), device=self.device)
            ee_pos_b = self.robot.data.body_pos_w[source_env_ids, self.body_idx] - self.robot.data.root_pos_w[source_env_ids]
            self.pose_command_b[env_ids_t, :3] = ee_pos_b
        else:
            # Y still follows configured range
            self.pose_command_b[env_ids_t, 1] = r.uniform_(*self.cfg.ranges.pos_y)

            # Annulus sampling in X-Z plane
            theta = torch.empty(num, device=self.device).uniform_(self.cfg.theta_min, self.cfg.theta_max)
            u = torch.empty(num, device=self.device).uniform_(0.0, 1.0)
            if self.cfg.uniform_area:
                radius = torch.sqrt(
                    u * (self.cfg.radius_max**2 - self.cfg.radius_min**2) + self.cfg.radius_min**2
                )
            else:
                radius = self.cfg.radius_min + u * (self.cfg.radius_max - self.cfg.radius_min)

            self.pose_command_b[env_ids_t, 0] = self.cfg.center_x + radius * torch.cos(theta)
            self.pose_command_b[env_ids_t, 2] = self.cfg.center_z + radius * torch.sin(theta)

        # Keep target_pose a valid pose command. The 4th joint has its own command term.
        euler_angles = torch.zeros(num, 3, device=self.device)
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        self.pose_command_b[env_ids_t, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat


@configclass
class AnnulusPoseCommandCfg(UniformPoseCommandCfg):
    """Pose command sampled on an annulus in the X-Z plane (base frame)."""

    class_type: type = AnnulusPoseCommand

    center_x: float = 0.0
    center_z: float = 0.0
    radius_min: float = 0.5
    radius_max: float = 0.8
    uniform_area: bool = True
    theta_min: float = -math.pi
    theta_max: float = math.pi
    sample_reachable_poses: bool = False
    reachable_targets_path: str | None = None


class UniformJointCommand(CommandTerm):
    """Sample a scalar joint target independently from pose commands."""

    cfg: "UniformJointCommandCfg"

    def __init__(self, cfg: "UniformJointCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.joint_command = torch.zeros(self.num_envs, 1, device=self.device)

    def __str__(self) -> str:
        msg = "UniformJointCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return self.joint_command

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.joint_command[env_ids_t, 0].uniform_(*self.cfg.ranges.joint_pos)

    def _update_command(self):
        pass


@configclass
class UniformJointCommandCfg(CommandTermCfg):
    """Configuration for a scalar uniform joint command."""

    class_type: type = UniformJointCommand

    @configclass
    class Ranges:
        joint_pos: tuple[float, float] = MISSING

    ranges: Ranges = MISSING

