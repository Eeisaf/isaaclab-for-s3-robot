from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BCPolicySpec:
    obs_dim: int
    action_dim: int
    hidden_dims: tuple[int, ...] = (64, 64)
    activation: str = "elu"
    model_type: str = "mlp"
    diffusion_timesteps: int = 100


def _make_activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "mish":
        return nn.Mish()
    raise ValueError(f"Unsupported activation: {name}")


class BCPolicy(nn.Module):
    """Small MLP matching the current low-dimensional reaching policy shape."""

    def __init__(self, spec: BCPolicySpec):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = spec.obs_dim
        for hidden_dim in spec.hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(_make_activation(spec.activation))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, spec.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # The environment action term clamps actions to [-1, 1]; keeping the policy bounded
        # makes BC playback match collection-time expert actions.
        return torch.tanh(self.net(obs))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConditionalMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, time_dim=32, hidden_dims=(256, 256), activation="elu"):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            _make_activation(activation),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        input_dim = obs_dim + action_dim + time_dim
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(_make_activation(activation))
            last_dim = h
        layers.append(nn.Linear(last_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, action, obs, time):
        t_emb = self.time_mlp(time)
        x = torch.cat([action, obs, t_emb], dim=-1)
        return self.net(x)


class DiffusionPolicy(nn.Module):
    """Diffusion Policy using DDPM and a conditional MLP noise predictor."""

    def __init__(self, spec: BCPolicySpec):
        super().__init__()
        self.action_dim = spec.action_dim
        self.num_train_timesteps = spec.diffusion_timesteps
        
        self.model = ConditionalMLP(
            obs_dim=spec.obs_dim, 
            action_dim=spec.action_dim, 
            hidden_dims=spec.hidden_dims,
            activation=spec.activation
        )
        
        # DDPM scheduler parameters (linear schedule)
        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, self.num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Inference (Sampling)
        batch_size = obs.shape[0]
        device = obs.device
        
        # Start from pure noise
        action = torch.randn((batch_size, self.action_dim), device=device)
        
        for i in reversed(range(self.num_train_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.model(action, obs, t)
            
            # DDPM step
            alpha = 1.0 - self.betas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            if i > 0:
                noise = torch.randn_like(action)
            else:
                noise = torch.zeros_like(action)
                
            action = (1 / torch.sqrt(alpha)) * (action - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * noise_pred) + torch.sqrt(beta) * noise
            
        return torch.clamp(action, -1.0, 1.0)

    def compute_loss(self, obs, action):
        batch_size = obs.shape[0]
        device = obs.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device).long()
        
        # Add noise to action
        noise = torch.randn_like(action)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        noisy_action = sqrt_alpha_cumprod_t * action + sqrt_one_minus_alpha_cumprod_t * noise
        
        # Predict noise
        noise_pred = self.model(noisy_action, obs, t)
        
        # MSE loss between predicted noise and actual noise
        return nn.functional.mse_loss(noise_pred, noise)


def policy_obs(obs) -> torch.Tensor:
    """Return the flattened policy observation from raw Gym/Isaac Lab observations."""
    if isinstance(obs, dict):
        if "policy" not in obs:
            raise KeyError(f"Expected observation dict to contain 'policy', got keys: {list(obs.keys())}")
        return obs["policy"]
    return obs


def load_bc_policy(checkpoint_path: str, device: str | torch.device) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    spec = BCPolicySpec(
        obs_dim=int(checkpoint["obs_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dims=tuple(checkpoint.get("hidden_dims", (64, 64))),
        activation=checkpoint.get("activation", "elu"),
        model_type=checkpoint.get("model_type", "mlp"),
        diffusion_timesteps=checkpoint.get("diffusion_timesteps", 100),
    )
    
    if spec.model_type == "diffusion":
        policy = DiffusionPolicy(spec).to(device)
    else:
        policy = BCPolicy(spec).to(device)
        
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    obs_mean = checkpoint["obs_mean"].to(device)
    obs_std = checkpoint["obs_std"].to(device)
    return policy, obs_mean, obs_std
