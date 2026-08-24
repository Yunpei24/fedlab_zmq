"""FedLab ``FLAlgorithm`` adapters for deployable DMD variants."""

from __future__ import annotations

from algorithms.base import FLAlgorithm, register_algorithm

from .client import client_update
from .config import default_algorithm_config
from .server import server_aggregate


class _DMDAlgorithm(FLAlgorithm):
    variant = "mean"
    reference_mode = "robust"
    description = "Decision-Margin Deficit fairness with frozen stale context."

    def client_update(self, model, dataloader, state, config):
        return client_update(
            self, model, dataloader, state, config, variant=self.variant
        )

    def server_aggregate(self, global_model, client_updates, round_num, config):
        return server_aggregate(
            global_model,
            client_updates,
            round_num,
            config,
            variant=self.variant,
        )

    def get_default_config(self):
        config = default_algorithm_config(self.variant)
        config["reference_mode"] = self.reference_mode
        return config


@register_algorithm("dmd_mean")
class DMDMeanAlgorithm(_DMDAlgorithm):
    variant = "mean"
    description = "DMD-Mean: CE plus mean quadratic decision-margin deficit."


@register_algorithm("dmd_usv")
class DMDUSVAlgorithm(_DMDAlgorithm):
    variant = "upper_semivariance"
    description = "DMD-USV: DMD-Mean plus upper-semivariance deficit control."


@register_algorithm("dmd_tail")
class DMDTailAlgorithm(_DMDAlgorithm):
    variant = "cvar"
    description = "DMD-Tail: DMD-Mean plus frozen-threshold upper-tail CVaR."


# Historical experiment identifiers remain registry aliases. These adapters
# do not replace the historical monolithic runner; they make old config names
# resolve in the generic FedLab registry during the migration.
@register_algorithm("dmd_deficit_mean_fixed_zero")
class DMDMeanFixedZeroAlias(DMDMeanAlgorithm):
    reference_mode = "fixed_zero"


@register_algorithm("dmd_deficit_upper_semivariance_fixed_zero")
class DMDUSVFixedZeroAlias(DMDUSVAlgorithm):
    reference_mode = "fixed_zero"


@register_algorithm("dmd_deficit_cvar_fixed_zero")
class DMDTailFixedZeroAlias(DMDTailAlgorithm):
    reference_mode = "fixed_zero"


__all__ = [
    "DMDMeanAlgorithm",
    "DMDUSVAlgorithm",
    "DMDTailAlgorithm",
    "DMDMeanFixedZeroAlias",
    "DMDUSVFixedZeroAlias",
    "DMDTailFixedZeroAlias",
]
