from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass


class DeltaJointPositionAction(ActionTerm):
    """Apply policy actions as joint-position increments around the current joint state."""

    cfg: "DeltaJointPositionActionCfg"

    def __init__(self, cfg: "DeltaJointPositionActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        self._raw_actions = torch.zeros((self.num_envs, len(self._joint_ids)), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        limits = torch.as_tensor(cfg.joint_limits, dtype=torch.float32, device=self.device)
        if limits.ndim == 1 and limits.numel() == 2:
            self._joint_lower_limits = torch.full((len(self._joint_ids),), limits[0].item(), device=self.device)
            self._joint_upper_limits = torch.full((len(self._joint_ids),), limits[1].item(), device=self.device)
        elif limits.shape == (len(self._joint_ids), 2):
            self._joint_lower_limits = limits[:, 0]
            self._joint_upper_limits = limits[:, 1]
        else:
            raise ValueError(
                f"joint_limits must be (lower, upper) or one (lower, upper) pair per joint, got {cfg.joint_limits}"
            )

    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        delta = torch.clamp(actions, -1.0, 1.0) * self.cfg.scale

        current_joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        target_joint_pos = current_joint_pos + delta
        self._processed_actions[:] = torch.clamp(
            target_joint_pos,
            self._joint_lower_limits.unsqueeze(0),
            self._joint_upper_limits.unsqueeze(0),
        )

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: list[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0


@configclass
class DeltaJointPositionActionCfg(ActionTermCfg):
    class_type: type = DeltaJointPositionAction
    asset_name: str = "robot"
    joint_names: list[str] = MISSING
    scale: float = 0.05
    joint_limits: tuple[float, float] | list[tuple[float, float]] = (-1.57, 1.57)
