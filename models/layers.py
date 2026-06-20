"""Briques partagées par les modèles "limited" et "full"."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import config
from data.pipeline import compute_window_offsets


class MultiResolutionEncoder(nn.Module):
    """Encodeur GRU pour une fenêtre d'historique multi-résolution.

    Le découpage temporel (quels pas sont échantillonnés et à quelle
    résolution) est décidé en amont par `data.pipeline.compute_window_offsets`
    et appliqué pendant la construction du `Dataset`. Cet encodeur ajoute, à
    chaque pas, l'écart de temps (en minutes, normalisé par le plus grand
    écart) avec le pas précédent : le GRU "sait" ainsi que les pas anciens
    représentent des intervalles de temps plus longs que les pas récents.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = config.GRU_HIDDEN_SIZE,
        num_layers: int = config.GRU_NUM_LAYERS,
        dropout: float = config.GRU_DROPOUT,
        history_hours: float = config.HISTORY_HOURS,
    ) -> None:
        super().__init__()

        offsets = compute_window_offsets(config.resolution_segments_for(history_hours))
        dt_minutes = np.empty(len(offsets), dtype=np.float32)
        if len(offsets) > 1:
            dt_minutes[1:] = (offsets[:-1] - offsets[1:]) * config.SAMPLE_INTERVAL_MINUTES
            dt_minutes[0] = dt_minutes[1]
        else:
            dt_minutes[:] = config.SAMPLE_INTERVAL_MINUTES
        dt_minutes /= dt_minutes.max()
        self.register_buffer("dt", torch.from_numpy(dt_minutes).view(1, -1, 1))

        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=input_size + 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_steps, input_size) -> dernier état caché (batch, hidden_size)."""
        dt = self.dt.expand(x.shape[0], -1, -1)
        x = torch.cat([x, dt], dim=-1)
        _, h_n = self.gru(x)
        return h_n[-1]


class RegressionHead(nn.Module):
    """MLP simple : représentation encodée -> N valeurs prédites."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int | None = None) -> None:
        super().__init__()
        hidden_size = hidden_size or input_size
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
