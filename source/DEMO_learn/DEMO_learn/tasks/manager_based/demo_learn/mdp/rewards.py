from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def end_effector_position_tracking(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    计算末端执行器“位置追踪”的奖励。
    距离目标位置越近，奖励越高，呈指数函数形式递减。
    
    Args:
        env (ManagerBasedRLEnv): 当前的强化学习环境实例。
        std (float): 高斯函数的标准差，用来控制距离误差到奖励值的衰减速度。
        asset_cfg (SceneEntityCfg): 场景实体的配置。
        
    Returns:
        torch.Tensor: 形状为 (num_envs,) 的奖励张量。距离越近越趋近于1。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0] if asset_cfg.body_ids is not None and not isinstance(asset_cfg.body_ids, slice) and len(asset_cfg.body_ids) > 0 else -1
    ee_pos = asset.data.body_pos_w[:, body_id, :] # 当前实际位置
    
    target_pose = env.command_manager.get_command("target_pose")
    target_pos = target_pose[:, :3] # 目标位置
    
    # 计算位置欧氏距离误差 (Distance error)
    pos_error = torch.norm(ee_pos - target_pos, dim=-1)
    # 将距离误差转化为奖励值：使用指数函数 exp(-error / std)，误差为0时奖励为1
    return torch.exp(-pos_error / std)

def end_effector_orientation_tracking(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    计算末端执行器“姿态（方向）追踪”的奖励。
    当前方向与目标方向越一致，奖励越高。
    
    Args:
        env (ManagerBasedRLEnv): 当前的强化学习环境实例。
        std (float): 衰减系数标准差。
        asset_cfg (SceneEntityCfg): 场景实体的配置。
        
    Returns:
        torch.Tensor: 形状为 (num_envs,) 的奖励张量。方向完全一致时越趋近于1。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0] if asset_cfg.body_ids is not None and not isinstance(asset_cfg.body_ids, slice) and len(asset_cfg.body_ids) > 0 else -1
    ee_quat = asset.data.body_quat_w[:, body_id, :] # 当前实际四元数旋转
    
    target_pose = env.command_manager.get_command("target_pose")
    target_quat = target_pose[:, 3:7] # 目标四元数旋转
    
    # 计算四元数误差幅度
    quat_error = quat_error_magnitude(ee_quat, target_quat)
    # 将旋转误差转化为奖励值：使用指数函数
    return torch.exp(-quat_error / std)

def upright_posture_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    计算机器人偏离“直立姿态”的惩罚。
    鼓励末端执行器的局部Z轴在世界坐标系下始终朝上。偏离越多，惩罚越大。
    
    Args:
        env (ManagerBasedRLEnv): 当前的强化学习环境实例。
        asset_cfg (SceneEntityCfg): 场景实体的配置。
        
    Returns:
        torch.Tensor: 形状为 (num_envs,) 的惩罚张量（正数，外面会在配置文件中赋予负的权重）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0] if asset_cfg.body_ids is not None and not isinstance(asset_cfg.body_ids, slice) and len(asset_cfg.body_ids) > 0 else -1
    ee_quat = asset.data.body_quat_w[:, body_id, :] # 获取当前的旋转四元数
    
    from isaaclab.utils.math import quat_rotate_inverse
    # 定义世界坐标系下的全局Z轴 (0, 0, 1)
    global_z = torch.zeros((env.num_envs, 3), device=env.device)
    global_z[:, 2] = 1.0
    
    # 使用四元数旋转，计算末端执行器的局部Z轴在世界坐标系下的方向
    from isaaclab.utils.math import quat_rotate
    local_z_in_w = quat_rotate(ee_quat, global_z)
    
    # 如果保持直立，局部的Z应该与全局Z平行，即 local_z_in_w[:, 2] 应该等于 1。
    # 用 1 减去其Z轴投影，作为偏离直立的惩罚度。
    return 1.0 - local_z_in_w[:, 2]

def joint_limit_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, bounds: list[tuple[float, float]]) -> torch.Tensor:
    """
    关节限位惩罚。
    惩罚任何超出指定物理边界（上限或下限）的关节运动，促使机器人学会在安全范围内动作。
    支持为不同的关节设置不同的限制。
    
    Args:
        env (ManagerBasedRLEnv): 当前的强化学习环境实例。
        asset_cfg (SceneEntityCfg): 场景实体的配置。
        bounds (list[tuple[float, float]]): 每个关节的 (下界, 上界) 阈值列表。
        
    Returns:
        torch.Tensor: 形状为 (num_envs,) 的惩罚张量，返回每个机器人超限关节的个数。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos # 获取所有关节当前的位置/角度
    
    # 将 bounds 列表转换为 tensor
    lower_bounds = torch.tensor([b[0] for b in bounds], device=env.device)
    upper_bounds = torch.tensor([b[1] for b in bounds], device=env.device)
    
    # 生成一个布尔掩码，标记出低于下限或高于上限的关节
    out_of_bounds = torch.logical_or(joint_pos < lower_bounds, joint_pos > upper_bounds)
    # 统计每个机器人有多少个关节越界，作为惩罚信号返回
    return torch.sum(out_of_bounds.float(), dim=-1)

def phase_based_penalty(env: ManagerBasedRLEnv, height_threshold: float, asset_cfg: SceneEntityCfg, frozen_joints: list[int] = [0, 1, 2]) -> torch.Tensor:
    """
    基于阶段的关节速度惩罚（分阶段约束）：
    - 阶段 1：未到达指定高度。此阶段前几个关节运动调整高度，最后几个关节也可以微调（或者不受此项惩罚）。
    - 阶段 2：已经到达指定高度（高度误差小于阈值）。此阶段严格惩罚（冻结）前几个特定关节的速度，只允许最后一个关节调整姿态。
    
    Args:
        env (ManagerBasedRLEnv): 当前的强化学习环境实例。
        height_threshold (float): 高度误差的阈值（米）。当高度差小于此值，认为进入阶段2。
        asset_cfg (SceneEntityCfg): 场景实体的配置。
        frozen_joints (list[int]): 在阶段2需要被冻结的关节索引列表。
        
    Returns:
        torch.Tensor: 形状为 (num_envs,) 的惩罚张量。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0] if asset_cfg.body_ids is not None and not isinstance(asset_cfg.body_ids, slice) and len(asset_cfg.body_ids) > 0 else -1
    ee_pos = asset.data.body_pos_w[:, body_id, :]
    
    target_pose = env.command_manager.get_command("target_pose")
    target_pos = target_pose[:, :3]
    
    # 计算当前高度与目标高度的绝对差值
    height_error = torch.abs(ee_pos[:, 2] - target_pos[:, 2])
    
    # 阶段2的布尔掩码（满足此条件即进入阶段2）
    phase_2 = height_error < height_threshold
    
    # 对指定的关节速度求平方并求和，作为阶段2的速度惩罚基数
    joint_vel_penalty = torch.zeros(env.num_envs, device=env.device)
    for j_idx in frozen_joints:
        if j_idx < asset.data.joint_vel.shape[1]:
            joint_vel_penalty += torch.square(asset.data.joint_vel[:, j_idx])
            
    # 只在进入阶段2的环境中应用此惩罚，阶段1乘0消除该项惩罚
    return phase_2.float() * joint_vel_penalty