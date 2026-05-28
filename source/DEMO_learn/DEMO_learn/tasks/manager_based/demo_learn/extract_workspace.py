import argparse
import math
import os
import torch

from isaaclab.app import AppLauncher

# 1. Initialize the simulation app
parser = argparse.ArgumentParser(description="Extract the reachable workspace of the robot.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

TRUNK_ROBOT_USD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../trunk_robot/trunk_robot.usd"))

TRUNK_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=TRUNK_ROBOT_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "all": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=300.0,
            velocity_limit=3.0,
            stiffness=400.0,
            damping=40.0,
        ),
    },
)

@configclass
class WorkspaceSceneCfg(InteractiveSceneCfg):
    """Configuration for a minimal scene to compute the workspace."""
    num_envs = 1000
    env_spacing = 2.0
    
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    
    robot = TRUNK_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    # 2. Setup the simulation context
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)

    # 3. Setup the scene
    scene_cfg = WorkspaceSceneCfg()
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulator
    sim.reset()
    
    robot: Articulation = scene["robot"]
    
    body_id = -1
    
    print("[INFO] Starting workspace sampling...")
    
    all_ee_pos = []
    num_sampling_steps = 100
    
    joint_limits = robot.data.soft_joint_pos_limits
    lower_limits = joint_limits[0, :, 0]
    upper_limits = joint_limits[0, :, 1]
    
    print(f"[INFO] Joint lower limits: {lower_limits}")
    print(f"[INFO] Joint upper limits: {upper_limits}")
    
    for step in range(num_sampling_steps):
        rand_noise = torch.rand((scene.num_envs, robot.num_joints), device=sim.device)
        random_joint_pos = lower_limits + rand_noise * (upper_limits - lower_limits)
        
        robot.write_joint_state_to_sim(random_joint_pos, torch.zeros_like(random_joint_pos))
        robot.reset()
        
        sim.step()
        
        ee_pos_w = robot.data.body_pos_w[:, body_id, :]
        base_pos_w = robot.data.root_pos_w
        ee_pos_b = ee_pos_w - base_pos_w
        
        ee_quat_w = robot.data.body_quat_w[:, body_id, :]
        
        all_ee_pos.append(ee_pos_b.clone())
        
        if step == 0:
            print(f"[INFO] First step ee_quat_w: {ee_quat_w[0]}")
            
    all_ee_pos = torch.cat(all_ee_pos, dim=0)
    
    print(f"[INFO] Total sampled points: {all_ee_pos.shape[0]}")
    
    x = all_ee_pos[:, 0]
    y = all_ee_pos[:, 1]
    z = all_ee_pos[:, 2]
    
    radius = torch.sqrt(x**2 + z**2)
    theta = torch.atan2(z, x)
    
    print("\n" + "="*50)
    print("WORKSPACE ANALYSIS RESULTS")
    print("="*50)
    
    print(f"X bounds: [{x.min().item():.4f}, {x.max().item():.4f}]")
    print(f"Y bounds: [{y.min().item():.4f}, {y.max().item():.4f}]")
    print(f"Z bounds: [{z.min().item():.4f}, {z.max().item():.4f}]")
    
    print("\nAnnulus Parameters (X-Z plane):")
    print(f"Radius bounds: [{radius.min().item():.4f}, {radius.max().item():.4f}]")
    print(f"Theta bounds:  [{theta.min().item():.4f}, {theta.max().item():.4f}]")
    
    print("\n" + "="*50)
    print("SUGGESTED CONFIGURATION FOR CommandsCfg")
    print("="*50)
    print(f"""
    target_pose = mdp.AnnulusPoseCommandCfg(
        asset_name="robot",
        body_name=".*", 
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        center_x=0.0,
        center_z=0.0,
        radius_min={radius.min().item():.4f},
        radius_max={radius.max().item():.4f},
        uniform_area=True,
        theta_min={theta.min().item():.4f},
        theta_max={theta.max().item():.4f},
        ranges=mdp.AnnulusPoseCommandCfg.Ranges(
            pos_x=(-0.2, 0.2),  # Ignored by AnnulusPoseCommand
            pos_y=({y.min().item():.4f}, {y.max().item():.4f}), # Y range
            pos_z=(0.5, 0.8),   # Ignored by AnnulusPoseCommand
            roll=(-0.8, 0.8),
            pitch=(-0.8, 0.8),
            yaw=(0.0, 0.0),
        ),
    )
    """)
    
    simulation_app.close()

if __name__ == "__main__":
    main()
