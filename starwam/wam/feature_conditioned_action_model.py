"""Feature-conditioned action-model taxonomy entry."""

from __future__ import annotations

from typing import Any

from starwam.wam.base import WAMModel


class FeatureConditionedActionModel(WAMModel):
    """Feature-conditioned action-model taxonomy entry."""

    taxonomy_model_family = "feature_conditioned_action_model"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "FeatureConditionedActionModel is not implemented for LIBERO training yet. "
            "Use taxonomy.model_family='mot_wam'."
        )
