"""Isolated RCIG policy ablation; original, source-locked algorithms unchanged.

All arms share the original gate and two causal views. Only the deployed
reference differs. Counterfactual calculations never select an arm. The seed
and tensor fingerprints below are simulator-only audit information, NOT an
additional private protocol release or a production source of DP randomness.
"""

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import random
from unittest.mock import patch

import numpy as np
import torch

from algorithms.ldp_gradient_far import LDPGradientFAR
from algorithms.rcig_temporal_reference import (
    TemporalRCIGReferenceState,
    _public_coordinates,
)
from algorithms.gaussian_aware_reference_k7_rcig import (
    robust_covariance_innovation_fusion,
)


def stream_seed(seed, client_id, round_num, purpose):
    key = f"rcig-n10-v1/{seed}/{client_id}/{round_num}/{purpose}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % (
        2**63 - 1
    )


def tensor_digest(tensor):
    value = tensor.detach().contiguous().cpu()
    h = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def model_digest(model):
    return hashlib.sha256(
        "".join(k + tensor_digest(v) for k, v in model.state_dict().items()).encode()
    ).hexdigest()


@contextmanager
def isolated_simulator_rng(seed, client_id, round_num, *, device, audit=None):
    """Save/restore ambient streams; independent per-client/round batch/noise.

    No attack, regime or policy appears in the seed derivation. The CPU batch
    stream is independent of the MPS Gaussian stream. Audit records only hashes
    of actual draws, outside the aggregation boundary. Single-process use only.
    """
    cpu = torch.get_rng_state()
    use_mps = torch.device(device).type == "mps"
    mps = torch.mps.get_rng_state() if use_mps else None
    py, np_state = random.getstate(), np.random.get_state()
    batch_seed = stream_seed(seed, client_id, round_num, "batch")
    noise_seed = stream_seed(seed, client_id, round_num, "noise")
    # Seed CPU directly: torch.manual_seed would also mutate accelerator RNGs.
    torch.random.default_generator.manual_seed(batch_seed)
    if use_mps:
        torch.mps.manual_seed(noise_seed)
    random.seed(batch_seed)
    np.random.seed(batch_seed % (2**32))
    original_permutation, original_normal = torch.randperm, torch.randn_like

    def permutation(*args, **kwargs):
        result = original_permutation(*args, **kwargs)
        if audit is not None:
            audit.setdefault("permutations", []).append(tensor_digest(result))
        return result

    def normal(*args, **kwargs):
        result = original_normal(*args, **kwargs)
        if audit is not None:
            audit.setdefault("standard_gaussians", []).append(tensor_digest(result))
        return result

    try:
        with (
            patch.object(torch, "randperm", permutation),
            patch.object(torch, "randn_like", normal),
        ):
            yield
    finally:
        torch.set_rng_state(cpu)
        if use_mps:
            torch.mps.set_rng_state(mps)
        random.setstate(py)
        np.random.set_state(np_state)


class AblationTemporalState(TemporalRCIGReferenceState):
    def __init__(self, config, arm):
        if arm not in {"recent", "midpoint", "rolling", "freeze"}:
            raise ValueError("undeclared temporal arm")
        expected_policy = "freeze_hysteresis" if arm == "freeze" else "rolling"
        if config.persistent_policy != expected_policy:
            raise ValueError("state policy does not match experimental arm")
        super().__init__(config)
        self.arm = arm
        self._accepted_at = None

    def reference_for_round(self, **kwargs):
        result = super().reference_for_round(**kwargs)
        diag = dict(result.diagnostics)
        diag["rcig_n10_arm"] = self.arm
        if not result.ready:
            return replace(result, diagnostics=diag)
        t = kwargs["round_num"]
        candidates = result.candidate_references
        if self.arm in {"recent", "midpoint"}:
            name = "identity_new" if self.arm == "recent" else "midpoint"
            reference = candidates[name].detach().clone()
            result = replace(
                result,
                reference=reference,
                _persistent_transition=(reference, False, 0),
            )
            diag.update(
                rcig_deployed_candidate=name,
                rcig_persistent_action=f"deploy_{name}",
                rcig_persistent_frozen_after_commit=False,
                rcig_recovery_count_after_commit=0,
                rcig_newer_view_trust=1.0 if self.arm == "recent" else 0.5,
            )
        # The original `gate_active` is a rolling-test diagnostic. It is not
        # a decision made by controls nor the frozen-state indicator.
        diag["rcig_n10_innovation_drives_deployment"] = self.arm in {
            "rolling",
            "freeze",
        }
        diag["rcig_n10_reference_age_rounds"] = (
            t - self._accepted_at
            if diag["rcig_persistent_frozen_after_commit"]
            and self._accepted_at is not None
            else 0
        )
        for name, covariance in (
            ("older", result.covariance_older),
            ("newer", result.covariance_newer),
        ):
            eig = torch.linalg.eigvalsh(covariance)
            diag[f"rcig_n10_{name}_cov_trace"] = float(torch.trace(covariance))
            diag[f"rcig_n10_{name}_cov_eigen_min"] = float(eig.min())
            diag[f"rcig_n10_{name}_cov_eigen_max"] = float(eig.max())
        if self._persistent_frozen:
            indices = _public_coordinates(
                kwargs["dimension"],
                self.config.subspace_dimension,
                self.config.subspace_seed,
            )
            _, recovery = robust_covariance_innovation_fusion(
                self._last_accepted_reference.index_select(0, indices),
                result.newer_view.index_select(0, indices),
                result.covariance_older,
                result.covariance_newer,
                process_variance=self.config.process_variance,
                ridge=self.config.ridge,
                innovation_threshold=self.config.innovation_threshold,
                influence_cap=self.config.server_clip_norm,
                covariance_mode="full",
                return_diagnostics=True,
            )
            diag["rcig_n10_recovery_stat"] = float(recovery["standardized_innovation"])
            diag["rcig_n10_recovery_threshold"] = float(self.config.recovery_threshold)
        diag["rcig_reference_norm"] = float(torch.linalg.vector_norm(result.reference))
        return replace(result, diagnostics=diag)

    def commit_snapshot(self, snapshot, *, reference_result=None):
        super().commit_snapshot(snapshot, reference_result=reference_result)
        if reference_result is not None and not self._persistent_frozen:
            self._accepted_at = snapshot.round_num


class N10PolicyAblation(LDPGradientFAR):
    """Registered only by the dedicated entrypoint, never globally on import."""

    # Set by the simulator entrypoint; no fingerprints passed to the server.
    audit_sink = None

    def client_update(self, model, dataloader, state, config):
        device = str(config.get("device"))
        if device not in {"mps", "mps:0"}:
            raise ValueError("N10 private gradients must run on MPS")
        row = {
            "client_id": int(state.client_id),
            "round": int(config["_server_round"]) + 1,
            "model_before": model_digest(model),
        }
        with isolated_simulator_rng(
            config["rcig_n10_pairing_seed"],
            state.client_id,
            config["_server_round"],
            device=device,
            audit=row,
        ):
            update, metadata = super().client_update(model, dataloader, state, config)
        if len(row.get("permutations", [])) != 1 or not row.get("standard_gaussians"):
            raise RuntimeError("missing actual batch/noise fingerprints")
        row["private_upload"] = hashlib.sha256(
            "".join(k + tensor_digest(v) for k, v in update.items()).encode()
        ).hexdigest()
        if self.audit_sink is not None:
            self.audit_sink(row)
        return update, metadata

    def _far_reference_override(self, **kwargs):
        config = kwargs["config"]
        if not self._uses_rcig(config):
            return None
        rcig_config = self._rcig_config(config, float(config["far_server_clip_norm"]))
        if not hasattr(self, "_rcig_state"):
            layout = kwargs["layout"]
            self._rcig_state = AblationTemporalState(
                rcig_config, config["rcig_n10_arm"]
            )
            self._rcig_signature = (
                tuple(layout.keys),
                tuple(tuple(s) for s in layout.shapes),
                len(kwargs["client_updates"]),
                rcig_config,
            )
        return super()._far_reference_override(**kwargs)

    def server_aggregate(self, global_model, client_updates, round_num, config):
        adapted = dict(config)
        # RFA comparator shares the same uniform warmup as the temporal arms.
        if round_num < 12:
            adapted["far_alpha"] = 0.0
        return super().server_aggregate(
            global_model, client_updates, round_num, adapted
        )
