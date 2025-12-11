#!/usr/bin/env python3
"""
LLRD-enabled YOLOv8 Trainer

Implements Layer-wise Learning Rate Decay (LLRD) for improved fine-tuning performance.
LLRD applies different learning rates to different network layers:
- Detection head: highest learning rate
- Neck layers: medium learning rate
- Backbone layers: lowest learning rate

Expected improvement: +1-3% mAP on few-shot datasets.

Reference:
- Fine-Tuning Without Forgetting (arXiv:2505.01016)
- Layer-wise learning rate decay concept from BERT fine-tuning
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch import nn, optim
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, colorstr


@dataclass
class LLRDConfig:
    """
    Configuration for Layer-wise Learning Rate Decay.

    Attributes:
        enabled: Whether LLRD is enabled
        decay_rate: LR decay factor per layer depth (0.0-1.0)
                   layer_lr = base_lr * (decay_rate ^ depth)
                   where depth=0 is the detection head (highest LR)
    """

    enabled: bool = False
    decay_rate: float = 0.9

    def __post_init__(self):
        """Validate configuration values."""
        if not 0.0 < self.decay_rate <= 1.0:
            raise ValueError(f"decay_rate must be in (0.0, 1.0], got {self.decay_rate}")


class LLRDDetectionTrainer(DetectionTrainer):
    """
    Detection Trainer with Layer-wise Learning Rate Decay.

    Extends Ultralytics DetectionTrainer to implement LLRD by overriding
    the build_optimizer method to create parameter groups with different
    learning rates based on layer depth.

    YOLOv8 Layer Structure (23 layers, 0-22):
        - Backbone (layers 0-9): Conv, C2f, SPPF - feature extraction
        - Neck (layers 10-21): Upsample, Concat, C2f - feature fusion
        - Head (layer 22): Detect - prediction output

    LLRD Formula:
        layer_lr = base_lr * (decay_rate ^ depth)
        where depth = max_layer - layer_idx
        So layer 22 (head) has depth=0 (highest LR)
        and layer 0 (early backbone) has depth=22 (lowest LR)
    """

    # YOLOv8 layer boundaries
    BACKBONE_END = 9  # Layers 0-9 are backbone
    NECK_END = 21  # Layers 10-21 are neck
    HEAD_LAYER = 22  # Layer 22 is detection head

    def __init__(
        self,
        cfg=DEFAULT_CFG_DICT,
        overrides=None,
        _callbacks=None,
        llrd_config: Optional[LLRDConfig] = None,
    ):
        """
        Initialize LLRD Detection Trainer.

        Args:
            cfg: Trainer configuration (defaults to Ultralytics default config)
            overrides: Configuration overrides dictionary
            _callbacks: Callback functions
            llrd_config: LLRD configuration (if None, uses default disabled config)
        """
        # Store LLRD config before parent init (parent may call methods)
        self.llrd_config = llrd_config or LLRDConfig()

        # Call parent with Ultralytics standard pattern
        super().__init__(cfg, overrides, _callbacks)

        if self.llrd_config.enabled:
            LOGGER.info(
                f"{colorstr('LLRD:')} Enabled with decay_rate={self.llrd_config.decay_rate}"
            )

    def _get_layer_depth(self, layer_idx: int) -> int:
        """
        Calculate depth from detection head.

        Layer 22 (Detect) = depth 0 (highest LR)
        Layer 0 (first Conv) = depth 22 (lowest LR)

        Args:
            layer_idx: Layer index (0-22 for YOLOv8)

        Returns:
            Depth value where 0 is head and increasing towards backbone
        """
        return self.HEAD_LAYER - layer_idx

    def _get_layer_category(self, layer_idx: int) -> str:
        """
        Categorize layer as backbone, neck, or head.

        Args:
            layer_idx: Layer index (0-22 for YOLOv8)

        Returns:
            Category string: 'backbone', 'neck', or 'head'
        """
        if layer_idx <= self.BACKBONE_END:
            return "backbone"
        elif layer_idx <= self.NECK_END:
            return "neck"
        else:
            return "head"

    def build_optimizer(
        self,
        model,
        name="auto",
        lr=0.001,
        momentum=0.9,
        decay=1e-5,
        iterations=1e5,
    ):
        """
        Build optimizer with LLRD parameter groups.

        Overrides BaseTrainer.build_optimizer to implement layer-wise LR decay.
        Falls back to standard optimizer if LLRD is disabled.

        Args:
            model: The model to optimize
            name: Optimizer name ('AdamW', 'SGD', etc. or 'auto')
            lr: Base learning rate (used as highest LR for head)
            momentum: Momentum coefficient
            decay: Weight decay
            iterations: Number of iterations (used for 'auto' optimizer selection)

        Returns:
            Configured optimizer with LLRD parameter groups
        """
        if not self.llrd_config.enabled:
            # Fall back to standard optimizer
            return super().build_optimizer(model, name, lr, momentum, decay, iterations)

        LOGGER.info(
            f"{colorstr('LLRD:')} Building optimizer with Layer-wise LR Decay "
            f"(decay_rate={self.llrd_config.decay_rate})"
        )

        # Auto-select optimizer if needed (same logic as parent class)
        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        if name == "auto":
            nc = self.data.get("nc", 10)
            lr_fit = round(0.002 * 5 / (4 + nc), 6)
            name, lr, momentum = (
                ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            )
            self.args.warmup_bias_lr = 0.0

        name = {x.lower(): x for x in optimizers}.get(name.lower(), name)

        # Build LLRD parameter groups
        param_groups = self._build_llrd_param_groups(model, lr, decay)

        # Create optimizer with parameter groups
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(
                param_groups, betas=(momentum, 0.999)
            )
        elif name == "RMSProp":
            optimizer = optim.RMSprop(param_groups, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(param_groups, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}."
            )

        # Log parameter groups summary
        self._log_param_groups(optimizer, lr)

        return optimizer

    def _build_llrd_param_groups(
        self,
        model: nn.Module,
        base_lr: float,
        weight_decay: float,
    ) -> List[Dict[str, Any]]:
        """
        Build parameter groups with layer-wise learning rate decay.

        Groups parameters by:
        1. Layer index (determines LR via LLRD formula)
        2. Parameter type (bias, BatchNorm, regular weights)

        Bias and BatchNorm parameters get no weight decay as per standard practice.

        Args:
            model: The model to extract parameters from
            base_lr: Base learning rate (highest, for detection head)
            weight_decay: Weight decay for regular weight parameters

        Returns:
            List of parameter group dictionaries for optimizer
        """
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        param_groups = []

        # Track parameters by layer
        # Structure: {layer_idx: {'bias': [], 'bn': [], 'weight': []}}
        layer_params: Dict[int, Dict[str, List]] = {}

        for module_name, module in model.named_modules():
            # Extract layer index from module name
            # Expected format: "model.{layer_idx}.{rest}" or "model.model.{layer_idx}.{rest}"
            parts = module_name.split(".")

            # Handle both "model.X.Y" and "X.Y" formats
            layer_idx = None
            for i, part in enumerate(parts):
                if part == "model" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        break
                    except ValueError:
                        continue
                elif i == 0:
                    try:
                        layer_idx = int(part)
                        break
                    except ValueError:
                        continue

            if layer_idx is None:
                continue

            if layer_idx not in layer_params:
                layer_params[layer_idx] = {"bias": [], "bn": [], "weight": []}

            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue

                if "bias" in param_name:
                    layer_params[layer_idx]["bias"].append(param)
                elif isinstance(module, bn) or "logit_scale" in param_name:
                    layer_params[layer_idx]["bn"].append(param)
                else:
                    layer_params[layer_idx]["weight"].append(param)

        # Build parameter groups with LLRD
        for layer_idx in sorted(layer_params.keys()):
            depth = self._get_layer_depth(layer_idx)
            category = self._get_layer_category(layer_idx)

            # Calculate layer LR with decay
            # depth=0 (head) gets base_lr, deeper layers get reduced LR
            layer_lr = base_lr * (self.llrd_config.decay_rate**depth)

            lp = layer_params[layer_idx]

            # Bias group (no weight decay)
            if lp["bias"]:
                param_groups.append(
                    {
                        "params": lp["bias"],
                        "lr": layer_lr,
                        "weight_decay": 0.0,
                        "layer_idx": layer_idx,
                        "param_type": "bias",
                        "category": category,
                    }
                )

            # BatchNorm group (no weight decay)
            if lp["bn"]:
                param_groups.append(
                    {
                        "params": lp["bn"],
                        "lr": layer_lr,
                        "weight_decay": 0.0,
                        "layer_idx": layer_idx,
                        "param_type": "bn",
                        "category": category,
                    }
                )

            # Weight group (with weight decay)
            if lp["weight"]:
                param_groups.append(
                    {
                        "params": lp["weight"],
                        "lr": layer_lr,
                        "weight_decay": weight_decay,
                        "layer_idx": layer_idx,
                        "param_type": "weight",
                        "category": category,
                    }
                )

        return param_groups

    def _log_param_groups(self, optimizer: optim.Optimizer, base_lr: float) -> None:
        """
        Log LLRD parameter group information.

        Args:
            optimizer: The configured optimizer
            base_lr: Base learning rate for reference
        """
        # Aggregate statistics by layer
        layer_info: Dict[int, Dict[str, Any]] = {}
        for pg in optimizer.param_groups:
            layer_idx = pg.get("layer_idx", -1)
            if layer_idx not in layer_info:
                layer_info[layer_idx] = {
                    "lr": pg["lr"],
                    "params": 0,
                    "category": pg.get("category", "unknown"),
                }
            layer_info[layer_idx]["params"] += sum(p.numel() for p in pg["params"])

        LOGGER.info(f"{colorstr('LLRD:')} Parameter groups by layer (base_lr={base_lr}):")

        # Log summary by category
        categories = {"backbone": [], "neck": [], "head": []}
        for layer_idx in sorted(layer_info.keys()):
            info = layer_info[layer_idx]
            categories[info["category"]].append(
                f"L{layer_idx}:{info['lr']:.6f}"
            )

        for cat in ["head", "neck", "backbone"]:
            if categories[cat]:
                LOGGER.info(f"  {cat.capitalize():8s}: {', '.join(categories[cat])}")

        # Log total parameters
        total_params = sum(info["params"] for info in layer_info.values())
        LOGGER.info(f"  Total parameters: {total_params:,}")

        # Log LR range
        all_lrs = [info["lr"] for info in layer_info.values()]
        if all_lrs:
            LOGGER.info(
                f"  LR range: {min(all_lrs):.6f} (backbone) -> {max(all_lrs):.6f} (head)"
            )
