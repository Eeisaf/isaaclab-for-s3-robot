import torch
import sys

data = torch.load("source/DEMO_learn/DEMO_learn/tasks/manager_based/demo_learn/reachable_targets.pt", map_location="cpu")
pos = data["positions"]

target = torch.tensor([0.0, 0.0, 0.60])
dist = torch.norm(pos - target, dim=-1)
min_dist, min_idx = torch.min(dist, dim=0)

closest_pos = pos[min_idx]
print(f"Closest target in dataset: {closest_pos.tolist()} (distance: {min_dist.item():.4f}m)")
