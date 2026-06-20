"""Modèle "full" : nesting résiduel sur le modèle "limited".

`full(x) = limited(x_limited) + correction(h_limited, x_outdoor)`

- `limited` apporte la tendance de fond (bilan énergétique régional, à
  partir de la météo). Il est gelé par défaut : la "full" model affine la
  prédiction sans dégrader le modèle de planification.
- `correction` apprend l'écart dû au rayonnement direct mesuré sur les 4
  façades et aux échanges d'air (capteurs extérieurs façades).

En production, si les capteurs façades tombent en panne, on peut retomber
sur `limited` seul (dégradation gracieuse) sans réentraînement.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config
from models.layers import MultiResolutionEncoder, RegressionHead
from models.limited import LimitedModel


class FullModel(nn.Module):
    def __init__(
        self,
        limited_model: LimitedModel,
        n_outdoor_features: int,
        n_targets: int,
        correction_hidden_size: int = config.FULL_CORRECTION_HIDDEN_SIZE,
        num_layers: int = config.GRU_NUM_LAYERS,
        dropout: float = config.GRU_DROPOUT,
        history_hours: float = config.HISTORY_HOURS,
        freeze_limited: bool = True,
    ) -> None:
        super().__init__()
        self.limited_model = limited_model
        self.freeze_limited = freeze_limited
        if freeze_limited:
            for p in self.limited_model.parameters():
                p.requires_grad = False

        self.outdoor_encoder = MultiResolutionEncoder(
            input_size=n_outdoor_features,
            hidden_size=correction_hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            history_hours=history_hours,
        )
        self.correction_head = RegressionHead(
            input_size=correction_hidden_size + self.limited_model.encoder.hidden_size,
            output_size=n_targets,
            hidden_size=correction_hidden_size,
        )

    def forward(
        self, x_limited: torch.Tensor, x_outdoor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retourne (prédiction finale, prédiction limited, correction)."""
        h_limited = self.limited_model.encoder(x_limited)
        base_pred = self.limited_model.head(h_limited)
        if self.freeze_limited:
            h_limited = h_limited.detach()
            base_pred = base_pred.detach()

        h_outdoor = self.outdoor_encoder(x_outdoor)
        correction = self.correction_head(torch.cat([h_limited, h_outdoor], dim=-1))

        return base_pred + correction, base_pred, correction
