from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class PolicyConfig:
    in_dim: int = 3            # dx, dy, vy
    hidden1: int = 64          # neuron number of layer 1 
    hidden2: int = 32          # neuron number of layer 2
    out_dim: int = 1           # flap probability
    activation: str = "relu"   # Activation function


def _act(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class Policy(nn.Module):
    def __init__(self, cfg: PolicyConfig = PolicyConfig()):
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(cfg.in_dim, cfg.hidden1)
        self.fc2 = nn.Linear(cfg.hidden1, cfg.hidden2)
        self.head = nn.Linear(cfg.hidden2, cfg.out_dim)
        self.act = _act(cfg.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if x.ndim == 1:
            x = x.view(1, -1)
            squeeze_back = True
        else:
            squeeze_back = False

        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        y = torch.sigmoid(self.head(h))  # (B, out_dim)

        if self.cfg.out_dim == 1:
            y = y.squeeze(-1)            # (B,)
        if squeeze_back:
            y = y.view(()) if y.ndim == 0 else y.view(-1)[0]
        return y


def build_policy(**overrides) -> Policy:
    cfg = PolicyConfig(**{**PolicyConfig().__dict__, **overrides})
    return Policy(cfg)


def save_policy(model: Policy, path: str) -> None:
    torch.save({
        "format": "policy.v2.two_hidden",
        "cfg": model.cfg.__dict__,
        "state_dict": model.state_dict(),
    }, path)


def load_policy(path: str, map_location: str | torch.device = "cpu") -> Policy:
    payload = torch.load(path, map_location=map_location)
    cfg_dict = dict(payload.get("cfg", {}))

    if "hidden" in cfg_dict and ("hidden1" not in cfg_dict or "hidden2" not in cfg_dict):
        h = int(cfg_dict["hidden"])
        cfg_dict.setdefault("hidden1", h)
        cfg_dict.setdefault("hidden2", h)
        cfg_dict.pop("hidden", None)

    cfg = PolicyConfig(**cfg_dict)
    model = Policy(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
