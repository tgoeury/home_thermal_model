"""Tests unitaires pour models/layers.py, models/limited.py, models/full.py."""

from __future__ import annotations

import torch
import pytest

import config
from models.layers import MultiResolutionEncoder, RegressionHead
from models.limited import LimitedModel
from models.full import FullModel


N_STEPS = 60      # 2h d'historique en résolution native
N_FEATURES = 10
N_OUTDOOR = 6
N_TARGETS = 4
BATCH = 8
HISTORY_HOURS = 2.0   # fenêtre courte pour les tests (rapide)


# ---------------------------------------------------------------------------
# MultiResolutionEncoder
# ---------------------------------------------------------------------------

def test_multi_resolution_encoder_output_shape():
    enc = MultiResolutionEncoder(input_size=N_FEATURES, history_hours=HISTORY_HOURS)
    x = torch.randn(BATCH, N_STEPS, N_FEATURES)
    h = enc(x)
    assert h.shape == (BATCH, config.GRU_HIDDEN_SIZE)


def test_multi_resolution_encoder_batch_size_1():
    enc = MultiResolutionEncoder(input_size=N_FEATURES, history_hours=HISTORY_HOURS)
    x = torch.randn(1, N_STEPS, N_FEATURES)
    h = enc(x)
    assert h.shape == (1, config.GRU_HIDDEN_SIZE)


def test_multi_resolution_encoder_dt_buffer_registered():
    enc = MultiResolutionEncoder(input_size=N_FEATURES, history_hours=HISTORY_HOURS)
    assert hasattr(enc, "dt")
    assert enc.dt.shape[0] == 1  # (1, n_steps, 1)
    assert enc.dt.shape[2] == 1


def test_multi_resolution_encoder_custom_hidden_size():
    enc = MultiResolutionEncoder(input_size=N_FEATURES, hidden_size=32, history_hours=HISTORY_HOURS)
    x = torch.randn(BATCH, N_STEPS, N_FEATURES)
    assert enc(x).shape == (BATCH, 32)


# ---------------------------------------------------------------------------
# RegressionHead
# ---------------------------------------------------------------------------

def test_regression_head_output_shape():
    head = RegressionHead(input_size=64, output_size=N_TARGETS)
    x = torch.randn(BATCH, 64)
    out = head(x)
    assert out.shape == (BATCH, N_TARGETS)


def test_regression_head_custom_hidden():
    head = RegressionHead(input_size=64, output_size=N_TARGETS, hidden_size=16)
    x = torch.randn(BATCH, 64)
    assert head(x).shape == (BATCH, N_TARGETS)


# ---------------------------------------------------------------------------
# LimitedModel
# ---------------------------------------------------------------------------

def test_limited_model_forward_shape():
    model = LimitedModel(n_limited_features=N_FEATURES, n_targets=N_TARGETS, history_hours=HISTORY_HOURS)
    x = torch.randn(BATCH, N_STEPS, N_FEATURES)
    pred = model(x)
    assert pred.shape == (BATCH, N_TARGETS)


def test_limited_model_eval_mode_no_grad():
    model = LimitedModel(n_limited_features=N_FEATURES, n_targets=N_TARGETS, history_hours=HISTORY_HOURS)
    model.eval()
    x = torch.randn(BATCH, N_STEPS, N_FEATURES)
    with torch.no_grad():
        pred = model(x)
    assert pred.shape == (BATCH, N_TARGETS)


def test_limited_model_deterministic_in_eval():
    model = LimitedModel(n_limited_features=N_FEATURES, n_targets=N_TARGETS, history_hours=HISTORY_HOURS)
    model.eval()
    x = torch.randn(BATCH, N_STEPS, N_FEATURES)
    with torch.no_grad():
        p1 = model(x)
        p2 = model(x)
    torch.testing.assert_close(p1, p2)


# ---------------------------------------------------------------------------
# FullModel
# ---------------------------------------------------------------------------

def _make_full_model(freeze: bool = True) -> tuple[LimitedModel, FullModel]:
    limited = LimitedModel(n_limited_features=N_FEATURES, n_targets=N_TARGETS, history_hours=HISTORY_HOURS)
    full = FullModel(
        limited_model=limited,
        n_outdoor_features=N_OUTDOOR,
        n_targets=N_TARGETS,
        history_hours=HISTORY_HOURS,
        freeze_limited=freeze,
    )
    return limited, full


def test_full_model_forward_returns_three_tensors():
    _, full = _make_full_model()
    x_lim = torch.randn(BATCH, N_STEPS, N_FEATURES)
    x_out = torch.randn(BATCH, N_STEPS, N_OUTDOOR)
    result = full(x_lim, x_out)
    assert len(result) == 3


def test_full_model_final_pred_shape():
    _, full = _make_full_model()
    x_lim = torch.randn(BATCH, N_STEPS, N_FEATURES)
    x_out = torch.randn(BATCH, N_STEPS, N_OUTDOOR)
    pred, base, corr = full(x_lim, x_out)
    assert pred.shape == (BATCH, N_TARGETS)
    assert base.shape == (BATCH, N_TARGETS)
    assert corr.shape == (BATCH, N_TARGETS)


def test_full_model_pred_equals_base_plus_correction():
    _, full = _make_full_model()
    full.eval()
    x_lim = torch.randn(BATCH, N_STEPS, N_FEATURES)
    x_out = torch.randn(BATCH, N_STEPS, N_OUTDOOR)
    with torch.no_grad():
        pred, base, corr = full(x_lim, x_out)
    torch.testing.assert_close(pred, base + corr)


def test_full_model_limited_frozen_by_default():
    limited, full = _make_full_model(freeze=True)
    for p in limited.parameters():
        assert not p.requires_grad


def test_full_model_limited_not_frozen_when_disabled():
    limited, full = _make_full_model(freeze=False)
    for p in limited.parameters():
        assert p.requires_grad


def test_full_model_only_correction_trainable_when_frozen():
    _, full = _make_full_model(freeze=True)
    trainable = [p for p in full.parameters() if p.requires_grad]
    assert len(trainable) > 0
    # Aucun param du limited_model ne doit être dans la liste entraînable
    limited_params = set(id(p) for p in full.limited_model.parameters())
    for p in trainable:
        assert id(p) not in limited_params
