from __future__ import annotations

"""Train a behavior-cloning policy from collected expert demonstrations."""

import argparse
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from bc_policy import BCPolicy, BCPolicySpec


def _load_datasets(paths: list[str]) -> tuple[torch.Tensor, torch.Tensor, dict]:
    observations = []
    actions = []
    metadata = {}
    for path in paths:
        data = torch.load(path, map_location="cpu")
        observations.append(data["observations"].float())
        actions.append(data["actions"].float())
        metadata = data.get("metadata", metadata)
    return torch.cat(observations, dim=0), torch.cat(actions, dim=0), metadata


def _resolve_paths(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(str(path) for path in Path().glob(pattern))
        paths.extend(matches if matches else [pattern])
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Dataset file(s) not found: {missing}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Train a BC MLP policy from Isaac Lab expert datasets.")
    parser.add_argument("datasets", nargs="+", help="Dataset .pt files or glob patterns.")
    parser.add_argument("--output", type=str, default="logs/il/demo_learn/bc_policy.pt", help="Output checkpoint path.")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 64], help="MLP hidden dimensions.")
    parser.add_argument("--activation", type=str, default="elu", choices=["elu", "relu", "tanh", "mish"], help="Activation.")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "diffusion"], help="Policy model type.")
    parser.add_argument("--diffusion_timesteps", type=int, default=100, help="Number of diffusion denoising steps.")
    parser.add_argument("--batch_size", type=int, default=4096, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of BC epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate.")
    parser.add_argument("--val_fraction", type=float, default=0.05, help="Validation split fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    dataset_paths = _resolve_paths(args.datasets)
    obs, actions, metadata = _load_datasets(dataset_paths)
    obs_mean = obs.mean(dim=0)
    obs_std = obs.std(dim=0).clamp_min(1e-6)
    obs = (obs - obs_mean) / obs_std

    dataset = TensorDataset(obs, actions)
    val_size = int(len(dataset) * args.val_fraction)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    spec = BCPolicySpec(
        obs_dim=obs.shape[-1],
        action_dim=actions.shape[-1],
        hidden_dims=tuple(args.hidden_dims),
        activation=args.activation,
        model_type=args.model_type,
        diffusion_timesteps=args.diffusion_timesteps,
    )
    if args.model_type == "diffusion":
        from bc_policy import DiffusionPolicy
        policy = DiffusionPolicy(spec).to(args.device)
    else:
        policy = BCPolicy(spec).to(args.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        policy.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_obs, batch_actions in train_loader:
            batch_obs = batch_obs.to(args.device)
            batch_actions = batch_actions.to(args.device)
            
            if args.model_type == "diffusion":
                loss = policy.compute_loss(batch_obs, batch_actions)
            else:
                pred_actions = policy(batch_obs)
                loss = loss_fn(pred_actions, batch_actions)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * batch_obs.shape[0]
            train_count += batch_obs.shape[0]

        policy.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.inference_mode():
            for batch_obs, batch_actions in val_loader:
                batch_obs = batch_obs.to(args.device)
                batch_actions = batch_actions.to(args.device)
                
                if args.model_type == "diffusion":
                    loss = policy.compute_loss(batch_obs, batch_actions)
                else:
                    pred_actions = policy(batch_obs)
                    loss = loss_fn(pred_actions, batch_actions)
                    
                val_loss_sum += loss.item() * batch_obs.shape[0]
                val_count += batch_obs.shape[0]

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}

        print(f"[BC] epoch={epoch:04d} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

    if best_state is not None:
        policy.load_state_dict(best_state)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": policy.cpu().state_dict(),
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "obs_dim": spec.obs_dim,
            "action_dim": spec.action_dim,
            "hidden_dims": spec.hidden_dims,
            "activation": spec.activation,
            "model_type": spec.model_type,
            "diffusion_timesteps": spec.diffusion_timesteps,
            "metadata": metadata,
            "dataset_paths": dataset_paths,
            "best_val_mse": best_val_loss,
        },
        output_path,
    )
    print(f"[INFO] Saved BC policy to: {output_path}")
    print(f"[INFO] Best validation MSE: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
