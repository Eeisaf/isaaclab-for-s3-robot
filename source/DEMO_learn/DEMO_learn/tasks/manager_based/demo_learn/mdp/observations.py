import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation

def end_effector_pose(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    获取机器人末端执行器（End Effector）在世界坐标系下的位姿（位置 + 四元数方向）。
    这是提供给神经网络的观测状态之一，让智能体知道自己当前手抓在哪。
    
    Args:
        env (ManagerBasedRLEnv): 当前的强化学习环境实例。
        asset_cfg (SceneEntityCfg): 场景实体的配置（用来指定是哪个机器人、哪个部件）。
        
    Returns:
        torch.Tensor: 形状为 (num_envs, 7) 的张量，包含每个环境中末端执行器的 3D位置(xyz) 和 4D四元数(wxyz)。
    """
    # 从环境中获取指定的机器人资产（Articulation，即具有关节的刚体集合）
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 判断是否在配置中指定了具体的 body（部件）ID
    if asset_cfg.body_ids is not None and not isinstance(asset_cfg.body_ids, slice) and len(asset_cfg.body_ids) > 0:
        body_id = asset_cfg.body_ids[0] # 如果指定了，就取指定的第一个部件
    else:
        body_id = -1 # Assume the last body is the end effector if specific body is not provided（默认取最后一个部件作为末端执行器）
    
    # 获取末端执行器在世界坐标系下的 3D 位置 (X, Y, Z)
    pos = asset.data.body_pos_w[:, body_id, :]
    # 获取末端执行器在世界坐标系下的 4D 旋转四元数 (W, X, Y, Z)
    quat = asset.data.body_quat_w[:, body_id, :]
    
    # 将位置和四元数拼接成一个 7 维的向量作为状态返回
    return torch.cat((pos, quat), dim=-1)

def target_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    获取当前回合（Episode）分配给机器人的目标位姿。
    这也是提供给神经网络的观测状态，告诉智能体“你要去哪里”。
    
    Args:
        env (ManagerBasedRLEnv): 当前的强化学习环境实例。
        
    Returns:
        torch.Tensor: 形状为 (num_envs, 7) 的张量，包含目标 3D位置(xyz) 和 4D四元数(wxyz)。
    """
    # 从环境的指令管理器(command_manager)中读取名为 "target_pose" 的指令
    return env.command_manager.get_command("target_pose")
