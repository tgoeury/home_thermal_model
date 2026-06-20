"""Modèle "limited" : prédit la température/humidité intérieure (N pièces) à
partir des données météo (observées ou prévisions), des features solaires et
de l'état volets/fenêtres — sans les capteurs extérieurs façades.

Utilisé en mode "planification" : à partir d'une prévision météo sur la
journée, on peut faire un rollout (auto-régressif) pour estimer l'évolution
de la température intérieure et choisir les créneaux d'ouverture/fermeture
des volets/fenêtres (cf. module `strategy/`, à venir).
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config
from models.layers import MultiResolutionEncoder, RegressionHead


class LimitedModel(nn.Module):
    def __init__(
        self,
        n_limited_features: int,
        n_targets: int,
        hidden_size: int = config.GRU_HIDDEN_SIZE,
        num_layers: int = config.GRU_NUM_LAYERS,
        dropout: float = config.GRU_DROPOUT,
        history_hours: float = config.HISTORY_HOURS,
    ) -> None:
        super().__init__()
        self.encoder = MultiResolutionEncoder(
            input_size=n_limited_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            history_hours=history_hours,
        )
        self.head = RegressionHead(input_size=hidden_size, output_size=n_targets)

    def forward(self, x_limited: torch.Tensor) -> torch.Tensor:
        """x_limited: (batch, n_steps, n_limited_features) -> (batch, n_targets)."""
        h = self.encoder(x_limited)
        return self.head(h)
