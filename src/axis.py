"""Activation-steering hook + assistant-axis loading."""
from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download


def load_axis(
    repo_id: str = "Butanium/llama-3.1-8b-instruct-assistant-axis",
    filename: str = "assistant_axis.pt",
    cache_dir: str | None = None,
    repo_type: str = "dataset",
) -> torch.Tensor:
    """Download and load the assistant-axis tensor.

    Shape expected: [num_layers, hidden_dim] for a per-layer steering vector.
    Butanium publishes this as a HF *dataset* repo, not a model repo.
    """
    local = hf_hub_download(
        repo_id=repo_id, filename=filename, cache_dir=cache_dir, repo_type=repo_type
    )
    tensor = torch.load(local, map_location="cpu", weights_only=True)
    if isinstance(tensor, dict):
        # Some checkpoints store under a key; try common names.
        for key in ("axis", "assistant_axis", "vector", "direction"):
            if key in tensor:
                tensor = tensor[key]
                break
        else:
            raise ValueError(f"axis .pt is a dict with keys {list(tensor.keys())}; cannot resolve")
    if tensor.dim() == 1:
        # If a single hidden_dim vector, broadcast as a "any layer" axis.
        return tensor
    if tensor.dim() != 2:
        raise ValueError(f"unexpected axis shape {tuple(tensor.shape)}")
    return tensor


def validate_axis(axis: torch.Tensor, model) -> None:
    hidden = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    if axis.dim() == 1:
        if axis.shape[0] != hidden:
            raise ValueError(f"axis hidden dim {axis.shape[0]} != model hidden {hidden}")
    else:
        nl, hd = axis.shape
        if hd != hidden:
            raise ValueError(f"axis hidden dim {hd} != model hidden {hidden}")
        if nl != num_layers:
            # Often axis files include the embedding row too; tolerate +/- 1.
            if abs(nl - num_layers) > 1:
                raise ValueError(f"axis num_layers {nl} != model num_layers {num_layers}")


def get_layer_module(model, layer_idx: int):
    """Return the residual-stream-producing module for layer i (LlamaDecoderLayer)."""
    return model.model.layers[layer_idx]


class SteeringHook:
    """Forward-hook context manager that adds `coefficient * vector` to a decoder layer's
    output residual stream every forward pass.
    """

    def __init__(self, model, vector: torch.Tensor, layer_idx: int, coefficient: float):
        self.model = model
        self.layer_idx = layer_idx
        self.coefficient = float(coefficient)
        self.vector = vector.detach()
        self._handle = None

    def __enter__(self):
        if self.coefficient == 0.0:
            # Skip installing — saves a tiny amount of work and keeps the model untouched.
            return self
        module = get_layer_module(self.model, self.layer_idx)
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        v = (self.coefficient * self.vector.to(device=device, dtype=dtype)).detach()

        def hook(_mod, _args, output):
            if isinstance(output, tuple):
                hs = output[0]
                hs = hs + v
                return (hs,) + output[1:]
            return output + v

        self._handle = module.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


def random_matched_norm_vector(reference: torch.Tensor, seed: int) -> torch.Tensor:
    """Return a random vector with the same L2 norm as `reference`, on CPU."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(reference.shape, generator=g, dtype=torch.float32)
    norm = reference.float().norm()
    v = v * (norm / v.norm())
    return v
