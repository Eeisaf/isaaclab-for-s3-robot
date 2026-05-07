import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_error_magnitude

def reached_target_pose(
    env: ManagerBasedRLEnv, 
    pos_threshold: float, 
    quat_threshold: float, 
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    判断机器人末端执行器是否成功到达了目标位姿（包括位置和旋转）。
    如果是，则该环境回合会被判定为“完成”并触发重置 (Termination)。
    
    Args:
        env (ManagerBasedRLEnv): 当前的强化学习环境实例。
        pos_threshold (float): 位置误差的容忍阈值（米）。当误差小于此值被认为到达位置。
        quat_threshold (float): 旋转(四元数)误差的容忍阈值。当误差小于此值被认为达到姿态。
        asset_cfg (SceneEntityCfg): 场景实体的配置，用来定位末端执行器。
        
    Returns:
        torch.Tensor: 形状为 (num_envs,) 的布尔张量 (或 0/1 张量)，True 表示到达目标。
    """
    # 从环境中获取指定的机器人资产
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取末端执行器的具体 Body ID (如果没设定默认用最后一个部件)
    body_id = asset_cfg.body_ids[0] if asset_cfg.body_ids is not None and not isinstance(asset_cfg.body_ids, slice) and len(asset_cfg.body_ids) > 0 else -1
    
    # 获取当前实际的 3D位置 和 4D旋转四元数
    ee_pos = asset.data.body_pos_w[:, body_id, :]
    ee_quat = asset.data.body_quat_w[:, body_id, :]
    
    # 获取目标位姿
    target_pose = env.command_manager.get_command("target_pose")
    target_pos = target_pose[:, :3] # 目标位置
    target_quat = target_pose[:, 3:7] # 目标旋转
    
    # 计算当前位置和目标位置之间的欧氏距离（位置误差）
    pos_error = torch.norm(ee_pos - target_pos, dim=-1)
    # 计算当前旋转和目标旋转之间的幅度差异（旋转误差）
    quat_error = quat_error_magnitude(ee_quat, target_quat)
    
    # 如果两者的误差都小于各自设定的阈值，就返回 True（到达目标），否则返回 False
    return torch.logical_and(pos_error < pos_threshold, quat_error < quat_threshold)
