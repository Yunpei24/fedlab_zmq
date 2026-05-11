"""
diagnostics/layer_mismatch.py
==============================
Layer Mismatch Diagnostic: Measures structural misalignment between feature
extractor and classifier layers after FedAvg-style aggregation.

In FL, after aggregation, the global model's feature extractor and classifier
are no longer aligned because each was trained on different data distributions.
This module provides three metrics to quantify this misalignment:

1. Representation Drift: ||h_l^before - h_l^after||_F / ||h_l^before||_F
   - Measures how much intermediate activations change after aggregation

2. Post-Aggregation Loss Jump: Loss(global) - Loss(local)
   - Measures how much each client's loss increases after receiving global model

3. CKA (Centered Kernel Alignment): Similarity between layer representations
   - CKA = 1 means identical representations, CKA = 0 means completely different

Usage:
    from diagnostics import LayerMismatchDiagnostic

    diagnostic = LayerMismatchDiagnostic(
        model=global_model,
        ref_dataloader=test_loader,
        device="cpu",
        layer_names=["layer1", "layer2", "fc"]  # or None for auto-detect
    )

    # After aggregation:
    results = diagnostic.compute(
        model_before_agg=global_model_before,
        model_after_agg=global_model_after,
        client_models=client_models_list,
        client_dataloaders=client_loaders,
        criterion=nn.CrossEntropyLoss()
    )

    mismatch_score = diagnostic.summary_score(results)
    print(f"Layer mismatch score: {mismatch_score:.4f}")

References:
  - Kornblith et al. (2019). Similarity of Neural Network Representations
    Revisited. ICML 2019.
  - Li et al. (2021). Federated Learning on Non-IID Data Silos: An Experimental
    Study. ICDE 2021.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
from collections import OrderedDict
import numpy as np


class LayerMismatchDiagnostic:
    """
    Diagnostic tool for measuring layer mismatch in federated learning.

    Attributes:
        model: Global model to analyze
        ref_batch: Fixed reference batch for activation capture
        device: Device to run computations on
        layer_names: List of layer names to monitor (or None for auto-detect)
    """

    def __init__(
        self,
        model: nn.Module,
        ref_dataloader: torch.utils.data.DataLoader,
        device: str = "cpu",
        layer_names: Optional[List[str]] = None,
        max_batches_per_client: int = 2,
        max_clients_cka: int = 8,
    ):
        """
        Initialize the layer mismatch diagnostic.

        Args:
            model: Global model (used for layer detection)
            ref_dataloader: Reference dataloader (we'll take one batch)
            device: Device to run on ("cpu", "cuda", "mps")
            layer_names: Specific layers to monitor, or None to auto-detect
                        Conv2d and Linear layers
            max_batches_per_client: Max batches used per client in loss_jump
                                    (2 batches ≈ 64 samples, ~26× faster than full scan)
            max_clients_cka: Max clients sampled for CKA pairwise comparison
                             (8 clients → 28 pairs vs 435 pairs for 30 clients)
        """
        self.device = device
        self.max_batches_per_client = max_batches_per_client
        self.max_clients_cka = max_clients_cka

        # Get reference batch (fixed for all rounds)
        try:
            self.ref_batch = next(iter(ref_dataloader))
        except StopIteration:
            raise ValueError("ref_dataloader is empty")

        # Auto-detect layers if not provided
        if layer_names is None:
            self.layer_names = self._auto_detect_layers(model)
        else:
            self.layer_names = layer_names

        if len(self.layer_names) == 0:
            raise ValueError("No layers found for monitoring. Check model architecture.")

    def _auto_detect_layers(self, model: nn.Module) -> List[str]:
        """
        Auto-detect Conv2d and Linear layers in the model.

        Returns:
            List of layer names (e.g., ["stem.0", "layer1.conv1", "fc"])
        """
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layers.append(name)
        return layers

    def _get_activations(
        self,
        model: nn.Module,
        batch: Tuple[torch.Tensor, torch.Tensor],
        layer_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Capture activations at specified layers using forward hooks.

        Args:
            model: Model to capture from
            batch: (inputs, targets) tuple
            layer_names: List of layer names to capture

        Returns:
            Dict mapping layer_name -> activation tensor (on CPU)
        """
        activations = {}
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                # Move to CPU to avoid OOM on small devices
                activations[name] = output.detach().cpu()
            return hook

        # Register hooks
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)

        # Forward pass
        model.eval()
        x, _ = batch
        x = x.to(self.device)

        with torch.no_grad():
            _ = model(x)

        # Clean up hooks
        for hook in hooks:
            hook.remove()

        return activations

    def representation_drift(
        self,
        model_before: nn.Module,
        model_after: nn.Module,
        ref_batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Metric 1: Representation Drift

        For each layer l, compute:
            drift_l = ||h_l^before - h_l^after||_F / ||h_l^before||_F

        where h_l are the intermediate activations at layer l.

        Args:
            model_before: Model before aggregation
            model_after: Model after aggregation
            ref_batch: Fixed reference batch (x, y)

        Returns:
            Dict mapping layer_name -> drift_value
        """
        model_before.to(self.device)
        model_after.to(self.device)

        # Get activations before and after
        acts_before = self._get_activations(model_before, ref_batch, self.layer_names)
        acts_after = self._get_activations(model_after, ref_batch, self.layer_names)

        drifts = {}
        for layer_name in self.layer_names:
            if layer_name not in acts_before or layer_name not in acts_after:
                continue

            h_before = acts_before[layer_name]
            h_after = acts_after[layer_name]

            # Flatten to 2D (batch_size, features)
            h_before_flat = h_before.flatten(1)
            h_after_flat = h_after.flatten(1)

            # Compute relative Frobenius norm
            diff_norm = torch.norm(h_before_flat - h_after_flat, p='fro').item()
            before_norm = torch.norm(h_before_flat, p='fro').item()

            if before_norm > 1e-8:
                drift = diff_norm / before_norm
            else:
                drift = 0.0

            drifts[layer_name] = drift

        return drifts

    def loss_jump(
        self,
        global_model: nn.Module,
        local_models_before_agg: List[nn.Module],
        client_dataloaders: List[torch.utils.data.DataLoader],
        criterion: nn.Module
    ) -> Dict[str, float]:
        """
        Metric 2: Post-Aggregation Loss Jump

        For each client i:
            delta_loss_i = Loss_i(global_model) - Loss_i(local_model_i)

        Args:
            global_model: Global model after aggregation
            local_models_before_agg: List of local models before aggregation
            client_dataloaders: List of client dataloaders
            criterion: Loss function (e.g., nn.CrossEntropyLoss())

        Returns:
            Dict with keys:
                - "client_0", "client_1", ...: delta_loss per client
                - "mean": mean loss jump across clients
                - "std": std of loss jump
                - "max": maximum loss jump
        """
        if len(local_models_before_agg) != len(client_dataloaders):
            raise ValueError(
                f"Mismatch: {len(local_models_before_agg)} models but "
                f"{len(client_dataloaders)} dataloaders"
            )

        global_model.to(self.device)
        global_model.eval()

        jumps = []
        result = {}
        max_batches = self.max_batches_per_client

        for i, (local_model, loader) in enumerate(zip(local_models_before_agg, client_dataloaders)):
            local_model.to(self.device)
            local_model.eval()

            # Compute loss on local data for both models
            loss_global = 0.0
            loss_local = 0.0
            total_samples = 0

            with torch.no_grad():
                for batch_idx, (x, y) in enumerate(loader):
                    if batch_idx >= max_batches:
                        break
                    x, y = x.to(self.device), y.to(self.device)
                    batch_size = x.size(0)

                    # Global model loss
                    out_global = global_model(x)
                    loss_global += criterion(out_global, y).item() * batch_size

                    # Local model loss
                    out_local = local_model(x)
                    loss_local += criterion(out_local, y).item() * batch_size

                    total_samples += batch_size

            if total_samples > 0:
                loss_global /= total_samples
                loss_local /= total_samples
                delta = loss_global - loss_local
            else:
                delta = 0.0

            jumps.append(delta)
            result[f"client_{i}"] = delta

        # Aggregate statistics
        if len(jumps) > 0:
            result["mean"] = float(np.mean(jumps))
            result["std"] = float(np.std(jumps))
            result["max"] = float(np.max(jumps))
        else:
            result["mean"] = 0.0
            result["std"] = 0.0
            result["max"] = 0.0

        return result

    def cka_between_clients(
        self,
        model: nn.Module,
        client_dataloaders: List[torch.utils.data.DataLoader],
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Metric 3: CKA (Centered Kernel Alignment) between client representations

        For each layer l and each pair of clients (i, j):
            CKA(H_i^l, H_j^l)

        Uses linear CKA:
            CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
        where K = H @ H.T (Gram matrix of activations)

        Args:
            model: Model to use (same weights for all clients)
            client_dataloaders: List of client dataloaders
            layer_names: Layers to analyze (default: self.layer_names)

        Returns:
            Dict mapping layer_name -> mean_CKA_across_all_client_pairs
        """
        if layer_names is None:
            layer_names = self.layer_names

        model.to(self.device)
        model.eval()

        # Subsample clients to keep pairwise cost O(max_clients^2) instead of O(K^2)
        sampled_loaders = client_dataloaders
        if self.max_clients_cka is not None and len(client_dataloaders) > self.max_clients_cka:
            rng = np.random.default_rng(seed=0)
            indices = rng.choice(len(client_dataloaders), size=self.max_clients_cka, replace=False)
            sampled_loaders = [client_dataloaders[i] for i in sorted(indices)]

        # Collect activations from sampled clients
        client_activations = []
        for loader in sampled_loaders:
            # Take first batch from each client
            try:
                batch = next(iter(loader))
            except StopIteration:
                continue

            acts = self._get_activations(model, batch, layer_names)
            client_activations.append(acts)

        if len(client_activations) < 2:
            # Need at least 2 clients for CKA
            return {name: 1.0 for name in layer_names}

        # Compute pairwise CKA for each layer
        cka_results = {}
        for layer_name in layer_names:
            cka_values = []

            # All pairs of clients
            for i in range(len(client_activations)):
                for j in range(i + 1, len(client_activations)):
                    if layer_name not in client_activations[i] or \
                       layer_name not in client_activations[j]:
                        continue

                    H_i = client_activations[i][layer_name]
                    H_j = client_activations[j][layer_name]

                    # Flatten to (batch_size, features)
                    H_i_flat = H_i.flatten(1)
                    H_j_flat = H_j.flatten(1)

                    # Compute linear CKA
                    cka_val = self._linear_cka(H_i_flat, H_j_flat)
                    cka_values.append(cka_val)

            # Average CKA across all pairs
            if len(cka_values) > 0:
                cka_results[layer_name] = float(np.mean(cka_values))
            else:
                cka_results[layer_name] = 1.0

        return cka_results

    def _linear_cka(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Compute Linear CKA between two feature matrices.

        CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
        where K = X @ X.T, L = Y @ Y.T

        Args:
            X: (n, d1) feature matrix for representation 1
            Y: (n, d2) feature matrix for representation 2

        Returns:
            CKA similarity score in [0, 1]
        """
        X = X.float()
        Y = Y.float()

        # Center the features
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

        # Gram matrices (on CPU to avoid OOM)
        X = X.cpu()
        Y = Y.cpu()

        # Linear CKA uses inner products directly (more efficient)
        # HSIC(X, Y) = (1/(n-1)^2) * tr(X @ X.T @ Y @ Y.T)
        # Simplifies to: (1/(n-1)^2) * ||X.T @ Y||_F^2

        n = X.size(0)
        if n <= 1:
            return 1.0

        # Compute HSIC efficiently
        hsic_xy = torch.norm(X.T @ Y, p='fro') ** 2 / ((n - 1) ** 2)
        hsic_xx = torch.norm(X.T @ X, p='fro') ** 2 / ((n - 1) ** 2)
        hsic_yy = torch.norm(Y.T @ Y, p='fro') ** 2 / ((n - 1) ** 2)

        # CKA
        denominator = torch.sqrt(hsic_xx * hsic_yy)
        if denominator > 1e-8:
            cka = hsic_xy / denominator
            return float(cka.clamp(0.0, 1.0))
        else:
            return 1.0

    def compute(
        self,
        model_before_agg: nn.Module,
        model_after_agg: nn.Module,
        client_models: List[nn.Module],
        client_dataloaders: List[torch.utils.data.DataLoader],
        criterion: nn.Module
    ) -> Dict[str, any]:
        """
        Compute all three layer mismatch metrics.

        Args:
            model_before_agg: Global model before aggregation
            model_after_agg: Global model after aggregation
            client_models: List of local client models (before aggregation)
            client_dataloaders: List of client dataloaders
            criterion: Loss function

        Returns:
            Dictionary containing:
                - "representation_drift": Dict[layer_name, drift]
                - "loss_jump": Dict with mean, std, max, per-client values
                - "cka": Dict[layer_name, mean_cka]
        """
        results = {}

        # Metric 1: Representation Drift
        try:
            drift = self.representation_drift(
                model_before_agg,
                model_after_agg,
                self.ref_batch
            )
            results["representation_drift"] = drift
        except Exception as e:
            print(f"Warning: Representation drift computation failed: {e}")
            results["representation_drift"] = {}

        # Metric 2: Loss Jump
        try:
            jump = self.loss_jump(
                model_after_agg,
                client_models,
                client_dataloaders,
                criterion
            )
            results["loss_jump"] = jump
        except Exception as e:
            print(f"Warning: Loss jump computation failed: {e}")
            results["loss_jump"] = {"mean": 0.0, "std": 0.0, "max": 0.0}

        # Metric 3: CKA
        try:
            cka = self.cka_between_clients(
                model_after_agg,
                client_dataloaders
            )
            results["cka"] = cka
        except Exception as e:
            print(f"Warning: CKA computation failed: {e}")
            results["cka"] = {}

        return results

    def summary_score(self, results: Dict[str, any]) -> float:
        """
        Compute a single scalar mismatch score from all metrics.

        Score is a weighted average of normalized metrics:
          - Higher drift → higher mismatch
          - Higher loss jump → higher mismatch
          - Lower CKA → higher mismatch

        Returns:
            Scalar in [0, 1] where 0 = no mismatch, 1 = severe mismatch
        """
        scores = []

        # 1. Representation drift (average across layers)
        drift_values = list(results.get("representation_drift", {}).values())
        if len(drift_values) > 0:
            avg_drift = np.mean(drift_values)
            # Normalize: assume drift > 0.5 is severe
            drift_score = min(avg_drift / 0.5, 1.0)
            scores.append(drift_score)

        # 2. Loss jump (mean across clients)
        loss_jump_mean = results.get("loss_jump", {}).get("mean", 0.0)
        # Normalize: assume jump > 1.0 is severe
        jump_score = min(abs(loss_jump_mean) / 1.0, 1.0)
        scores.append(jump_score)

        # 3. CKA (average across layers, invert because low CKA = high mismatch)
        cka_values = list(results.get("cka", {}).values())
        if len(cka_values) > 0:
            avg_cka = np.mean(cka_values)
            # Invert: CKA=1 → score=0, CKA=0 → score=1
            cka_score = 1.0 - avg_cka
            scores.append(cka_score)

        # Weighted average (equal weights)
        if len(scores) > 0:
            return float(np.mean(scores))
        else:
            return 0.0
