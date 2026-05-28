from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_eval_pos_w(asset: Articulation, body_id: int) -> torch.Tensor:
    """Return world-frame body frame origin position for evaluation consistency."""
    return asset.data.body_pos_w[:, body_id, :]


def end_effector_position_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the end-effector distance to the target position."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0] if asset_cfg.body_ids is not None and not isinstance(asset_cfg.body_ids, slice) and len(asset_cfg.body_ids) > 0 else -1
    ee_pos = _get_body_eval_pos_w(asset, body_id)
    ee_pos = ee_pos - asset.data.root_pos_w

    target_command = env.command_manager.get_command("target_pose")
    target_pos = target_command[:, :3]

    return torch.norm(ee_pos - target_pos, dim=-1)


def end_effector_position_error_squared(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return squared end-effector distance to strongly penalize large misses."""
    pos_error = end_effector_position_error(env, asset_cfg)
    return torch.square(pos_error)


def end_effector_position_fine_tracking(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return a narrow positive tracking reward centered on the final precision target."""
    pos_error = end_effector_position_error(env, asset_cfg)
    return torch.exp(-torch.square(pos_error / std))


def end_effector_position_excess_error(
    env: ManagerBasedRLEnv,
    pos_threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return only the distance outside the success radius."""
    pos_error = end_effector_position_error(env, asset_cfg)
    return torch.clamp(pos_error - pos_threshold, min=0.0)


def end_effector_up_axis_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return 1 - dot(local end-effector +Z, base +Z). Zero means vertical-up alignment."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0] if asset_cfg.body_ids is not None and not isinstance(asset_cfg.body_ids, slice) and len(asset_cfg.body_ids) > 0 else -1
    ee_quat_b = quat_mul(quat_inv(asset.data.root_quat_w), asset.data.body_quat_w[:, body_id, :])
    ee_up_axis_b = quat_apply(
        ee_quat_b,
        torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, -1),
    )
    return 1.0 - ee_up_axis_b[:, 2]


def time_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant per-step penalty so finishing earlier is better than waiting near the target."""
    return torch.ones(env.num_envs, device=env.device)


def target_reached_bonus(
    env: ManagerBasedRLEnv,
    pos_threshold: float,
    asset_cfg: SceneEntityCfg,
    up_axis_threshold: float | None = None,
) -> torch.Tensor:
    """
    任务达成奖励：如果机器人末端到达目标位置，返回 1.0。
    结合配置里的超高权重，这会在触发回合终止的最后一帧给予机器人巨大的奖励，从而彻底覆盖掉在目标边缘“蹭分”的收益。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0] if asset_cfg.body_ids is not None and not isinstance(asset_cfg.body_ids, slice) and len(asset_cfg.body_ids) > 0 else -1

    ee_pos = _get_body_eval_pos_w(asset, body_id)
    ee_pos = ee_pos - asset.data.root_pos_w # 转换为相对于基座的坐标

    target_command = env.command_manager.get_command("target_pose")
    target_pos = target_command[:, :3]

    pos_error = torch.norm(ee_pos - target_pos, dim=-1)

    reached = pos_error < pos_threshold
    if up_axis_threshold is not None:
        reached = reached & ((1.0 - end_effector_up_axis_error(env, asset_cfg)) > up_axis_threshold)

    # 到达目标返回 1.0，未到达返回 0.0
    return reached.float()


def target_pos_only_success(
    env: ManagerBasedRLEnv,
    pos_threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """监控指标：仅检查位置是否达标（不考虑姿态）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0] if asset_cfg.body_ids is not None and not isinstance(asset_cfg.body_ids, slice) and len(asset_cfg.body_ids) > 0 else -1

    ee_pos = _get_body_eval_pos_w(asset, body_id)
    ee_pos = ee_pos - asset.data.root_pos_w # 转换为相对于基座的坐标
    target_pose = env.command_manager.get_command("target_pose")
    target_pos = target_pose[:, :3]

    pos_error = torch.norm(ee_pos - target_pos, dim=-1)
    return (pos_error < pos_threshold).float()


def log_all_env_pos_success(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pos_threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """监控指标：忽略 reset batch，统计所有环境当前是否在成功半径内。"""
    del env_ids
    success = target_pos_only_success(env, pos_threshold, asset_cfg)
    return success.mean()


def log_all_env_pos_error(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """监控指标：忽略 reset batch，统计所有环境当前平均位置误差。"""
    del env_ids
    return end_effector_position_error(env, asset_cfg).mean()


def approach_target_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    过程奖励：根据末端执行器靠近或远离目标的速度给予奖励或惩罚。
    如果当前速度方向指向目标（即距离在缩小），给予正奖励；如果背离目标，给予负惩罚。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0] if asset_cfg.body_ids is not None and not isinstance(asset_cfg.body_ids, slice) and len(asset_cfg.body_ids) > 0 else -1

    # 当前实际位置 (世界坐标系)
    ee_pos_w = _get_body_eval_pos_w(asset, body_id)
    # 转换为相对于基座的坐标
    ee_pos = ee_pos_w - asset.data.root_pos_w

    # 目标位置
    target_command = env.command_manager.get_command("target_pose")
    target_pos = target_command[:, :3]

    # 位置误差向量 (从当前指向目标)
    pos_error_vec = target_pos - ee_pos
    distance = torch.norm(pos_error_vec, dim=-1)

    # 避免除以零
    distance = torch.clamp(distance, min=1e-6)

    # 归一化误差向量（方向）
    direction = pos_error_vec / distance.unsqueeze(-1)

    # 末端执行器的线速度 (世界坐标系)
    ee_vel_w = asset.data.body_lin_vel_w[:, body_id, :]
    # 由于基座固定，世界坐标系下的速度即为相对于基座的速度

    # 投影速度到目标方向上 (点乘)
    # 结果为正表示正在靠近目标，结果为负表示正在远离目标
    approach_speed = torch.sum(ee_vel_w * direction, dim=-1)

    return approach_speed
