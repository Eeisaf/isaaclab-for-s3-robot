import math
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul


def _get_body_eval_pos_w(asset: Articulation, body_id: int) -> torch.Tensor:
    """Return world-frame body frame origin position for evaluation consistency."""
    return asset.data.body_pos_w[:, body_id, :]


def reached_target_pose(
    env: ManagerBasedRLEnv, 
    pos_threshold: float, 
    asset_cfg: SceneEntityCfg,
    up_axis_threshold: float | None = None,
) -> torch.Tensor:
    """
    判断机器人末端执行器是否成功到达了目标位置。
    如果是，则该环境回合会被判定为“完成”并触发重置 (Termination)。
    """
    # 从环境中获取指定的机器人资产
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取末端执行器的具体 Body ID (如果没设定默认用最后一个部件)
    body_id = asset_cfg.body_ids[0] if asset_cfg.body_ids is not None and not isinstance(asset_cfg.body_ids, slice) and len(asset_cfg.body_ids) > 0 else -1
    
    # 获取当前实际的 3D位置(使用末端坐标系原点)
    ee_pos = _get_body_eval_pos_w(asset, body_id)
    ee_pos = ee_pos - asset.data.root_pos_w # 转换为相对于基座的坐标
    
    # 获取目标位姿
    target_command = env.command_manager.get_command("target_pose")
    target_pos = target_command[:, :3] # 目标位置
    
    # 计算当前位置和目标位置之间的欧氏距离（位置误差）
    pos_error = torch.norm(ee_pos - target_pos, dim=-1)
    
    # 如果误差小于设定的阈值，就返回 True（到达目标），否则返回 False
    reached = pos_error < pos_threshold
    if up_axis_threshold is not None:
        ee_quat_b = quat_mul(quat_inv(asset.data.root_quat_w), asset.data.body_quat_w[:, body_id, :])
        ee_up_axis_b = quat_apply(
            ee_quat_b,
            torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, -1),
        )
        reached = reached & (ee_up_axis_b[:, 2] > up_axis_threshold)
    return reached
