import pytest
import torch
import torch.nn.functional as F

from algorithms.dmd.config import DMDConfig
from algorithms.label_skew import effective_number_class_weights, label_skew_criterion

COUNTS = [14, 6, 0]  # the last class is absent from this client


def test_effective_number_spans_cross_entropy_to_inverse_frequency() -> None:
    uniform = effective_number_class_weights(COUNTS, 3, 0.0, device="cpu")
    assert torch.allclose(uniform, torch.tensor([1.0, 1.0, 0.0]))
    near_inverse = effective_number_class_weights(COUNTS, 3, 0.999999, device="cpu")
    inverse = torch.tensor([1 / 14, 1 / 6])
    assert torch.allclose(near_inverse[:2], inverse / inverse.mean(), atol=1e-4)
    assert near_inverse[2] == 0


def test_logit_adjustments_are_cross_entropy_when_counts_are_uniform() -> None:
    logits = torch.randn(8, 3, generator=torch.Generator().manual_seed(0))
    targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    plain = F.cross_entropy(logits, targets)
    for kind in ("balanced_softmax", "fedlc"):
        adjusted = label_skew_criterion(kind, [5, 5, 5], 3, device="cpu", tau=1.7)
        assert torch.allclose(adjusted(logits, targets), plain, atol=1e-6)


def test_logit_adjustments_favour_rare_and_drop_absent_classes() -> None:
    for kind in ("balanced_softmax", "fedlc"):
        offsets = label_skew_criterion(kind, COUNTS, 3, device="cpu", tau=1.0).offsets
        # Training on shifted logits lowers the rarer class's logit, so the model
        # learns to raise it: rarer observed classes get the smaller offset.
        assert offsets[1] < offsets[0]
        assert offsets[2] < offsets[1] - 15


def test_unknown_loss_and_missing_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown label-skew loss"):
        label_skew_criterion("ldam", COUNTS, 3, device="cpu")
    with pytest.raises(ValueError, match="client_class_counts"):
        label_skew_criterion("fedlc", None, 3, device="cpu")


def test_base_loss_conflicting_with_the_earlier_spelling_is_rejected() -> None:
    assert DMDConfig(ce_class_weighting="inverse_frequency").validate().effective_base_loss == (
        "inverse_frequency"
    )
    with pytest.raises(ValueError, match="conflicts"):
        DMDConfig(ce_class_weighting="inverse_frequency", base_loss="fedlc").validate()
