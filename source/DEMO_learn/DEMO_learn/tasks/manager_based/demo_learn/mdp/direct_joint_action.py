import torch
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass

class DirectJointCommandAction(ActionTerm):
    cfg: "DirectJointCommandActionCfg"

    def __init__(self, cfg: "DirectJointCommandActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._joint_idx = self._asset.find_joints(cfg.joint_name)[0][0]
        
    @property
    def action_dim(self) -> int:
        return 0 # This action term does not take input from the neural network

    @property
    def raw_actions(self) -> torch.Tensor:
        return torch.empty((self.num_envs, 0), device=self.device)

    @property
    def processed_actions(self) -> torch.Tensor:
        return torch.empty((self.num_envs, 0), device=self.device)

    def process_actions(self, actions: torch.Tensor):
        pass

    def apply_actions(self):
        target_command = self._env.command_manager.get_command(self.cfg.command_name)
        target_joint_pos = target_command[:, self.cfg.command_index].unsqueeze(-1)
        self._asset.set_joint_position_target(target_joint_pos, joint_ids=[self._joint_idx])

    def reset(self, env_ids: list[int] | None = None) -> None:
        pass

@configclass
class DirectJointCommandActionCfg(ActionTermCfg):
    class_type: type = DirectJointCommandAction
    asset_name: str = "robot"
    joint_name: str = "trunk_joint4"
    command_name: str = "target_joint4"
    command_index: int = 0
