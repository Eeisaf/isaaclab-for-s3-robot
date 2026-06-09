import torch
import sys
import os

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

sys.path.insert(0, os.path.abspath("source/DEMO_learn"))
from DEMO_learn.tasks.manager_based.demo_learn.demo_learn_env_cfg import TRUNK_ROBOT_CFG, END_EFFECTOR_BODY_NAME

@configclass
class ReachableTargetsSceneCfg(InteractiveSceneCfg):
    robot = TRUNK_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
sim = sim_utils.SimulationContext(sim_cfg)
scene_cfg = ReachableTargetsSceneCfg(num_envs=10000, env_spacing=2.0)
scene = InteractiveScene(scene_cfg)
sim.reset()

robot = scene["robot"]
ee_body_id = robot.find_bodies(END_EFFECTOR_BODY_NAME)[0][0]
controlled_joint_ids, _ = robot.find_joints(["trunk_joint1", "trunk_joint2", "trunk_joint3", "trunk_joint4"])

# Random search
torch.manual_seed(0)
joint_pos = robot.data.default_joint_pos.clone()
joint_pos[:, controlled_joint_ids[0]] = torch.rand(10000, device=sim.device) * 3.14 - 1.57
joint_pos[:, controlled_joint_ids[1]] = torch.rand(10000, device=sim.device) * 3.14 - 1.57
joint_pos[:, controlled_joint_ids[2]] = torch.rand(10000, device=sim.device) * 3.14 - 1.57
joint_pos[:, controlled_joint_ids[3]] = 0.0

robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
robot.reset()
sim.step()
scene.update(dt=sim.get_physics_dt())

ee_pos_b = robot.data.body_pos_w[:, ee_body_id, :] - robot.data.root_pos_w
ee_quat_b = quat_mul(quat_inv(robot.data.root_quat_w), robot.data.body_quat_w[:, ee_body_id, :])
ee_up_axis_b = quat_apply(ee_quat_b, torch.tensor([0.0, 0.0, 1.0], device=sim.device).expand(scene.num_envs, -1))

target = torch.tensor([0.0, 0.0, 0.60], device=sim.device)
dist = torch.norm(ee_pos_b - target, dim=-1)

# Filter by upright
upright_mask = ee_up_axis_b[:, 2] >= 0.98
dist[~upright_mask] = 1000.0

min_dist, min_idx = torch.min(dist, dim=0)
print(f"Minimum distance to (0,0,0.60) with upright constraint: {min_dist.item():.4f}m")
if min_dist.item() < 100.0:
    print(f"Best joints: {joint_pos[min_idx, controlled_joint_ids].tolist()}")
    print(f"Best pos: {ee_pos_b[min_idx].tolist()}")

simulation_app.close()
