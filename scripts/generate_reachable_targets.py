"""Generate a reachable end-effector target point cloud from real simulated FK."""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Generate reachable end-effector target positions.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel robots used per sampling batch.")
parser.add_argument("--num_samples", type=int, default=2000000, help="Number of reachable target positions to save.")
parser.add_argument(
    "--output",
    type=str,
    default="source/DEMO_learn/DEMO_learn/tasks/manager_based/demo_learn/reachable_targets.pt",
    help="Output .pt file containing reachable target positions.",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--radius_min", type=float, default=0.25, help="Minimum target radius to keep in X-Z plane.")
parser.add_argument("--radius_max", type=float, default=0.65, help="Maximum target radius to keep in X-Z plane.")
parser.add_argument("--theta_min", type=float, default=0.35, help="Minimum target angle to keep in X-Z plane.")
parser.add_argument("--theta_max", type=float, default=math.pi - 0.35, help="Maximum target angle to keep in X-Z plane.")
parser.add_argument("--joint1_min", type=float, default=-1.57, help="Lower sampling bound for trunk_joint1.")
parser.add_argument("--joint1_max", type=float, default=1.57, help="Upper sampling bound for trunk_joint1.")
parser.add_argument("--joint2_min", type=float, default=-1.57, help="Lower sampling bound for trunk_joint2.")
parser.add_argument("--joint2_max", type=float, default=1.57, help="Upper sampling bound for trunk_joint2.")
parser.add_argument("--joint3_min", type=float, default=-1.57, help="Lower sampling bound for trunk_joint3.")
parser.add_argument("--joint3_max", type=float, default=1.57, help="Upper sampling bound for trunk_joint3.")
parser.add_argument("--joint4", type=float, default=0.0, help="Fixed trunk_joint4 position.")
parser.add_argument(
    "--min_up_dot",
    type=float,
    default=0.98,
    help="Keep samples whose end-effector local z axis has this minimum dot with base +Z.",
)
parser.add_argument(
    "--max_attempt_multiplier",
    type=int,
    default=200,
    help="Maximum sampling batches as a multiple of the batches needed for num_samples.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

sys.path.insert(0, os.path.abspath("source/DEMO_learn"))
from DEMO_learn.tasks.manager_based.demo_learn.demo_learn_env_cfg import (  # noqa: E402
    END_EFFECTOR_BODY_NAME,
    TRUNK_ROBOT_CFG,
)


@configclass
class ReachableTargetsSceneCfg(InteractiveSceneCfg):
    """Minimal scene containing only the robot."""

    robot = TRUNK_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    torch.manual_seed(args_cli.seed)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = ReachableTargetsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    ee_body_id = robot.find_bodies(END_EFFECTOR_BODY_NAME)[0][0]
    controlled_joint_ids, controlled_joint_names = robot.find_joints(
        ["trunk_joint1", "trunk_joint2", "trunk_joint3", "trunk_joint4"]
    )
    print(f"[INFO] Sampling joints: {controlled_joint_names}")
    print(f"[INFO] End-effector body: {END_EFFECTOR_BODY_NAME} (id={ee_body_id})")

    joint_bounds = torch.tensor(
        [
            [args_cli.joint1_min, args_cli.joint1_max],
            [args_cli.joint2_min, args_cli.joint2_max],
            [args_cli.joint3_min, args_cli.joint3_max],
            [args_cli.joint4, args_cli.joint4],
        ],
        device=sim.device,
    )
    soft_joint_limits = robot.data.soft_joint_pos_limits[0, controlled_joint_ids]
    requested_joint_bounds = joint_bounds.clone()
    joint_bounds[:, 0] = torch.maximum(joint_bounds[:, 0], soft_joint_limits[:, 0])
    joint_bounds[:, 1] = torch.minimum(joint_bounds[:, 1], soft_joint_limits[:, 1])
    if torch.any(joint_bounds[:, 0] > joint_bounds[:, 1]):
        raise RuntimeError(
            f"Requested joint bounds do not overlap the robot soft limits. "
            f"requested={requested_joint_bounds.detach().cpu().tolist()} "
            f"soft_limits={soft_joint_limits.detach().cpu().tolist()}"
        )
    print(f"[INFO] Requested joint bounds: {requested_joint_bounds.detach().cpu().tolist()}")
    print(f"[INFO] Robot soft joint limits: {soft_joint_limits.detach().cpu().tolist()}")
    print(f"[INFO] Effective joint bounds: {joint_bounds.detach().cpu().tolist()}")

    kept_positions = []
    kept_joint_pos = []
    kept_quat = []
    kept_up_axis = []
    total_kept = 0
    attempts = 0
    max_attempts = max(100, math.ceil(args_cli.num_samples / args_cli.num_envs) * args_cli.max_attempt_multiplier)

    while total_kept < args_cli.num_samples and attempts < max_attempts:
        attempts += 1
        default_pos = robot.data.default_joint_pos.clone()
        joint_pos = default_pos.clone()
        noise = torch.rand((scene.num_envs, len(controlled_joint_ids)), device=sim.device)
        sampled = joint_bounds[:, 0] + noise * (joint_bounds[:, 1] - joint_bounds[:, 0])
        joint_pos[:, controlled_joint_ids] = sampled

        robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
        robot.reset()
        sim.step()
        scene.update(dt=sim.get_physics_dt())

        ee_pos_b = robot.data.body_pos_w[:, ee_body_id, :] - robot.data.root_pos_w
        ee_quat_b = quat_mul(quat_inv(robot.data.root_quat_w), robot.data.body_quat_w[:, ee_body_id, :])
        ee_up_axis_b = quat_apply(ee_quat_b, torch.tensor([0.0, 0.0, 1.0], device=sim.device).expand(scene.num_envs, -1))
        radius = torch.linalg.norm(ee_pos_b[:, [0, 2]], dim=-1)
        theta = torch.atan2(ee_pos_b[:, 2], ee_pos_b[:, 0])
        keep = (
            (radius >= args_cli.radius_min)
            & (radius <= args_cli.radius_max)
            & (theta >= args_cli.theta_min)
            & (theta <= args_cli.theta_max)
            & (ee_up_axis_b[:, 2] >= args_cli.min_up_dot)
        )

        if keep.any():
            remaining = args_cli.num_samples - total_kept
            selected_positions = ee_pos_b[keep][:remaining].detach().cpu()
            selected_joints = joint_pos[keep][:remaining, controlled_joint_ids].detach().cpu()
            selected_quat = ee_quat_b[keep][:remaining].detach().cpu()
            selected_up_axis = ee_up_axis_b[keep][:remaining].detach().cpu()
            kept_positions.append(selected_positions)
            kept_joint_pos.append(selected_joints)
            kept_quat.append(selected_quat)
            kept_up_axis.append(selected_up_axis)
            total_kept += selected_positions.shape[0]

        if attempts % 10 == 0 or total_kept >= args_cli.num_samples:
            print(f"[INFO] attempts={attempts} kept={total_kept}/{args_cli.num_samples}")

    if total_kept == 0:
        raise RuntimeError("No reachable targets were sampled. Relax radius/theta or joint bounds.")
    if total_kept < args_cli.num_samples:
        print(f"[WARN] Only sampled {total_kept} targets after {attempts} attempts.")

    positions = torch.cat(kept_positions, dim=0)[: args_cli.num_samples]
    joint_pos = torch.cat(kept_joint_pos, dim=0)[: args_cli.num_samples]
    ee_quat_b = torch.cat(kept_quat, dim=0)[: args_cli.num_samples]
    ee_up_axis_b = torch.cat(kept_up_axis, dim=0)[: args_cli.num_samples]
    output_path = os.path.abspath(args_cli.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(
        {
            "positions": positions,
            "joint_pos": joint_pos,
            "ee_quat_b": ee_quat_b,
            "ee_up_axis_b": ee_up_axis_b,
            "joint_names": controlled_joint_names,
            "end_effector_body_name": END_EFFECTOR_BODY_NAME,
            "radius_range": (args_cli.radius_min, args_cli.radius_max),
            "theta_range": (args_cli.theta_min, args_cli.theta_max),
            "min_up_dot": args_cli.min_up_dot,
        },
        output_path,
    )

    radius = torch.linalg.norm(positions[:, [0, 2]], dim=-1)
    theta = torch.atan2(positions[:, 2], positions[:, 0])
    print(f"[INFO] Saved {positions.shape[0]} reachable targets to: {output_path}")
    print(f"[INFO] X range: [{positions[:, 0].min().item():.4f}, {positions[:, 0].max().item():.4f}]")
    print(f"[INFO] Y range: [{positions[:, 1].min().item():.4f}, {positions[:, 1].max().item():.4f}]")
    print(f"[INFO] Z range: [{positions[:, 2].min().item():.4f}, {positions[:, 2].max().item():.4f}]")
    print(f"[INFO] Radius range: [{radius.min().item():.4f}, {radius.max().item():.4f}]")
    print(f"[INFO] Theta range: [{theta.min().item():.4f}, {theta.max().item():.4f}]")
    print(f"[INFO] Up-axis dot range: [{ee_up_axis_b[:, 2].min().item():.4f}, {ee_up_axis_b[:, 2].max().item():.4f}]")


if __name__ == "__main__":
    main()
    simulation_app.close()
