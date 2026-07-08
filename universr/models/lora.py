from __future__ import annotations

from collections.abc import Mapping
from fnmatch import fnmatchcase
from typing import Any

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base_linear = base_linear
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base_linear.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base_linear.out_features, bias=False)

        for parameter in self.base_linear.parameters():
            parameter.requires_grad = False
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_linear(x) + self.scale * self.lora_B(self.lora_A(self.dropout(x)))


class LoRAConv2d1x1(nn.Module):
    def __init__(self, base_conv: nn.Conv2d, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        if base_conv.kernel_size != (1, 1) or base_conv.groups != 1:
            raise ValueError("LoRAConv2d1x1 only supports grouped-free 1x1 Conv2d layers.")
        self.base_conv = base_conv
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Conv2d(base_conv.in_channels, self.rank, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(self.rank, base_conv.out_channels, kernel_size=1, bias=False)

        for parameter in self.base_conv.parameters():
            parameter.requires_grad = False
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_conv(x) + self.scale * self.lora_B(self.lora_A(self.dropout(x)))


def _config_value(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _normalise_config(config: Any) -> dict[str, Any]:
    def _patterns(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    return {
        "enabled": bool(_config_value(config, "enabled", False)),
        "rank": int(_config_value(config, "rank", 16)),
        "alpha": float(_config_value(config, "alpha", 16)),
        "dropout": float(_config_value(config, "dropout", 0.0)),
        "target": str(_config_value(config, "target", "linear")),
        "include_patterns": _patterns(_config_value(config, "include_patterns", [])),
        "exclude_patterns": _patterns(_config_value(config, "exclude_patterns", [])),
    }


def _get_parent_module(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent = model
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _is_linear_lora_target(name: str) -> bool:
    return (
        name.endswith(".pwconv1")
        or name.endswith(".pwconv2")
        or ".time_adapter." in name
        or name == "sr_projector"
        or name == "film_generator"
        or name == "conditioning_encoder.film_generator"
        or name.startswith("conditioning_encoder.sr_adapter.")
    )


def _is_conditioning_linear_target(name: str) -> bool:
    return (
        name == "sr_projector"
        or name == "film_generator"
        or name == "conditioning_encoder.film_generator"
        or name.startswith("conditioning_encoder.sr_adapter.")
        or (
            name.startswith("conditioning_encoder.blocks.")
            and (name.endswith(".pwconv1") or name.endswith(".pwconv2"))
        )
    )


def _is_mid_decoder_linear_target(name: str) -> bool:
    return (
        (name.startswith("midcoder.") or name.startswith("decoders."))
        and _is_linear_lora_target(name)
    )


def _is_1x1_conv_lora_target(name: str, module: nn.Module) -> bool:
    return (
        name in {"conditioning_encoder.head", "init_conv.0", "final_conv"}
        and isinstance(module, nn.Conv2d)
        and module.kernel_size == (1, 1)
        and module.groups == 1
    )


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatchcase(name, pattern) for pattern in patterns)


def _is_lora_target(name: str, module: nn.Module, config: Mapping[str, Any]) -> bool:
    target = config["target"]
    include_patterns = config["include_patterns"]
    exclude_patterns = config["exclude_patterns"]
    if exclude_patterns and _matches_any(name, exclude_patterns):
        return False

    included_by_pattern = bool(include_patterns and _matches_any(name, include_patterns))
    if isinstance(module, nn.Linear):
        if included_by_pattern:
            return True
        if target == "conditioning_only":
            return _is_conditioning_linear_target(name)
        if target == "mid_decoder_only":
            return _is_mid_decoder_linear_target(name)
        return target in {"linear", "linear_and_1x1_conv"} and _is_linear_lora_target(name)

    if isinstance(module, nn.Conv2d):
        if not _is_1x1_conv_lora_target(name, module):
            return False
        return target == "linear_and_1x1_conv" or included_by_pattern

    return False


def apply_lora(model: nn.Module, config: Any) -> list[str]:
    lora_config = _normalise_config(config)
    if not lora_config["enabled"]:
        return []
    valid_targets = {"conditioning_only", "linear", "linear_and_1x1_conv", "mid_decoder_only"}
    if lora_config["target"] not in valid_targets:
        raise ValueError(f"LoRA target must be one of {sorted(valid_targets)}.")

    wrapped: list[str] = []
    for name, module in list(model.named_modules()):
        if not name or isinstance(module, (LoRALinear, LoRAConv2d1x1)):
            continue
        replacement: nn.Module | None = None
        if not _is_lora_target(name, module, lora_config):
            continue
        if isinstance(module, nn.Linear):
            replacement = LoRALinear(
                module,
                rank=lora_config["rank"],
                alpha=lora_config["alpha"],
                dropout=lora_config["dropout"],
            )
        elif isinstance(module, nn.Conv2d):
            replacement = LoRAConv2d1x1(
                module,
                rank=lora_config["rank"],
                alpha=lora_config["alpha"],
                dropout=lora_config["dropout"],
            )
        if replacement is None:
            continue
        parent, child_name = _get_parent_module(model, name)
        setattr(parent, child_name, replacement)
        wrapped.append(name)

    model.lora_config = lora_config
    model.lora_target_modules = wrapped
    return wrapped


def mark_only_lora_trainable(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAConv2d1x1)):
            for parameter in module.lora_A.parameters():
                parameter.requires_grad = True
            for parameter in module.lora_B.parameters():
                parameter.requires_grad = True


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if ".lora_A." in key or ".lora_B." in key
    }


def load_lora_state_dict(model: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(dict(state_dict), strict=False)
    unexpected_lora = [key for key in unexpected if ".lora_" in key]
    missing_lora = [key for key in missing if ".lora_" in key]
    if unexpected_lora or missing_lora:
        raise RuntimeError(
            f"LoRA state mismatch. Missing LoRA keys: {missing_lora}; "
            f"unexpected LoRA keys: {unexpected_lora}"
        )


def count_lora_params(model: nn.Module) -> float:
    params = sum(
        parameter.numel()
        for module in model.modules()
        if isinstance(module, (LoRALinear, LoRAConv2d1x1))
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return params / 1_000_000
