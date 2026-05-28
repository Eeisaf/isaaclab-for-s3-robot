from __future__ import annotations

import os

import torch


class GlobalReachableIK:
    """Global sampled IK over a reachable FK point cloud.

    The solver searches all saved FK samples and returns the joint solution minimizing:

        ||fk_position - target_position||^2 + joint_weight * ||q_sample - q_current||^2

    This is not a local Jacobian method. Its behavior is constrained by the joint ranges used to generate
    reachable_targets.pt.
    """

    def __init__(
        self,
        reachable_targets_path: str,
        controlled_joint_names: list[str],
        device: str,
        joint_weight: float = 0.0,
        up_axis_weight: float = 1.0,
        target_up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
        chunk_size: int = 16384,
    ):
        data = torch.load(os.path.abspath(reachable_targets_path), map_location="cpu")
        if not isinstance(data, dict) or "positions" not in data or "joint_pos" not in data:
            raise ValueError(f"Expected {reachable_targets_path} to contain 'positions' and 'joint_pos'.")

        stored_joint_names = list(data.get("joint_names", []))
        if not stored_joint_names:
            raise ValueError(f"Expected {reachable_targets_path} to contain 'joint_names'.")

        joint_cols = []
        for joint_name in controlled_joint_names:
            if joint_name not in stored_joint_names:
                raise ValueError(f"Joint {joint_name!r} not found in reachable target joint_names={stored_joint_names}")
            joint_cols.append(stored_joint_names.index(joint_name))

        self.positions = data["positions"].float().to(device)
        self.joint_pos = data["joint_pos"].float().to(device)[:, joint_cols]
        self.up_axis = None
        if "ee_up_axis_b" in data:
            self.up_axis = data["ee_up_axis_b"].float().to(device)
        self.stored_joint_names = stored_joint_names
        self.controlled_joint_names = controlled_joint_names
        self.joint_weight = joint_weight
        self.up_axis_weight = up_axis_weight
        self.target_up_axis = torch.tensor(target_up_axis, dtype=torch.float32, device=device)
        self.chunk_size = chunk_size

    def solve(
        self,
        target_pos: torch.Tensor,
        current_joint_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return q_goal, Cartesian nearest-neighbor distance, and up-axis error for each target."""
        best_cost = torch.full((target_pos.shape[0],), float("inf"), device=target_pos.device)
        best_ids = torch.zeros((target_pos.shape[0],), dtype=torch.long, device=target_pos.device)
        best_pos_dist = torch.full((target_pos.shape[0],), float("inf"), device=target_pos.device)
        best_up_error = torch.full((target_pos.shape[0],), float("inf"), device=target_pos.device)
        row_ids = torch.arange(target_pos.shape[0], device=target_pos.device)

        for start in range(0, self.positions.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, self.positions.shape[0])
            candidate_pos = self.positions[start:end]
            pos_dist_sq = torch.sum(torch.square(target_pos[:, None, :] - candidate_pos[None, :, :]), dim=-1)
            cost = pos_dist_sq
            up_error = torch.zeros_like(pos_dist_sq)
            if self.up_axis is not None and self.up_axis_weight > 0.0:
                candidate_up = self.up_axis[start:end]
                up_dot = torch.sum(candidate_up[None, :, :] * self.target_up_axis.view(1, 1, 3), dim=-1)
                up_error = torch.square(1.0 - up_dot).expand_as(pos_dist_sq)
                cost = cost + self.up_axis_weight * up_error
            if self.joint_weight > 0.0 and current_joint_pos is not None:
                candidate_q = self.joint_pos[start:end]
                joint_dist_sq = torch.sum(torch.square(current_joint_pos[:, None, :] - candidate_q[None, :, :]), dim=-1)
                cost = cost + self.joint_weight * joint_dist_sq

            chunk_cost, chunk_ids = torch.min(cost, dim=1)
            update = chunk_cost < best_cost
            if update.any():
                best_cost[update] = chunk_cost[update]
                best_ids[update] = chunk_ids[update] + start
                update_rows = row_ids[update]
                best_pos_dist[update] = torch.sqrt(pos_dist_sq[update_rows, chunk_ids[update]])
                best_up_error[update] = torch.sqrt(up_error[update_rows, chunk_ids[update]])

        return self.joint_pos[best_ids], best_pos_dist, best_up_error
