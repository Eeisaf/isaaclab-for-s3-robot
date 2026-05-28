import argparse
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from demo_learn_env_cfg import WorkspaceSceneCfg

def main():
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(WorkspaceSceneCfg())
    sim.reset()
    
    robot = scene["robot"]
    body_id = -1
    
    robot.write_joint_state_to_sim(torch.zeros_like(robot.data.joint_pos), torch.zeros_like(robot.data.joint_vel))
    robot.reset()
    sim.step()
    
    print("Initial ee_quat_w:", robot.data.body_quat_w[0, body_id, :])
    print("Initial root_quat_w:", robot.data.root_quat_w[0, :])
    
    simulation_app.close()

if __name__ == "__main__":
    main()
