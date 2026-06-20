"""Tests unitaires pour data/pipeline.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from data.pipeline import (
    FeatureStats,
    HouseDataset,
    _select_columns,
    build_feature_table,
    compute_stats,
    compute_window_offsets,
)


# ---------------------------------------------------------------------------
# compute_window_offsets
# ---------------------------------------------------------------------------

def test_compute_window_offsets_last_is_zero():
    offsets = compute_window_offsets()
    assert offsets[-1] == 0


def test_compute_window_offsets_decreasing():
    offsets = compute_window_offsets()
    assert (np.diff(offsets) < 0).all()


def test_compute_window_offsets_default_count():
    # 60 pts à 2min + 24 pts à 10min + 24 pts à 30min = 108
    offsets = compute_window_offsets()
    assert len(offsets) == 108


def test_compute_window_offsets_short_history():
    segs = config.resolution_segments_for(2.0)  # 2h → 60 pts à 2min
    offsets = compute_window_offsets(segs)
    assert len(offsets) == 60
    assert offsets[-1] == 0


def test_compute_window_offsets_max_offset_matches_history():
    offsets = compute_window_offsets()
    # L'offset maximum (plus ancien) doit être ≈ 18h / 2min = 540 pas
    # (quelques unités près selon l'arrondi des segments)
    total_steps = config.HISTORY_HOURS * 60 / config.SAMPLE_INTERVAL_MINUTES
    assert offsets[0] <= total_steps


# ---------------------------------------------------------------------------
# FeatureStats
# ---------------------------------------------------------------------------

def test_feature_stats_transform_inverse_roundtrip():
    rng = np.random.default_rng(0)
    arr = rng.normal(5.0, 2.0, (100, 4)).astype(np.float32)
    stats = compute_stats(arr)
    transformed = stats.transform(arr)
    recovered = stats.inverse_transform(transformed)
    np.testing.assert_allclose(arr, recovered, atol=1e-5)


def test_compute_stats_ignores_nan():
    arr = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    stats = compute_stats(arr)
    assert np.isfinite(stats.mean).all()
    assert np.isfinite(stats.std).all()


def test_compute_stats_constant_column_std_becomes_one():
    arr = np.ones((10, 2), dtype=np.float32)
    stats = compute_stats(arr)
    np.testing.assert_array_equal(stats.std, np.ones(2))


# ---------------------------------------------------------------------------
# _select_columns
# ---------------------------------------------------------------------------

def test_select_columns_basic():
    df = pd.DataFrame(columns=["indoor__salon__temp", "weather__irr", "solar__elev", "outdoor__N__temp"])
    cols = _select_columns(df, ("indoor__",))
    assert cols == ["indoor__salon__temp"]


def test_select_columns_exclude():
    df = pd.DataFrame(columns=["weather__irr", "weather__kind"])
    cols = _select_columns(df, ("weather__",), exclude=("weather__kind",))
    assert "weather__kind" not in cols
    assert "weather__irr" in cols


# ---------------------------------------------------------------------------
# HouseDataset — construction minimale
# ---------------------------------------------------------------------------

def _make_minimal_table(n: int = 300) -> pd.DataFrame:
    """Table minimale avec les colonnes requises par HouseDataset."""
    idx = pd.date_range("2026-01-01", periods=n, freq="2min", tz="UTC")
    rng = np.random.default_rng(0)
    sep = config.COLUMN_SEP
    return pd.DataFrame(
        {
            f"indoor{sep}salon{sep}temperature": rng.normal(20, 1, n).astype(np.float32),
            f"indoor{sep}salon{sep}humidity": rng.normal(50, 3, n).astype(np.float32),
            f"weather{sep}outdoor_temperature": rng.normal(15, 2, n).astype(np.float32),
            f"solar{sep}elevation": rng.uniform(-20, 60, n).astype(np.float32),
            f"house{sep}salon{sep}shutter": rng.integers(0, 2, n).astype(np.float32),
        },
        index=idx,
    )


def test_house_dataset_len():
    table = _make_minimal_table(300)
    ds = HouseDataset(table, history_hours=2.0, horizon_steps=1)
    # valid_indices doit exclure les premières lignes (fenêtre) et la dernière (horizon)
    assert len(ds) > 0
    assert len(ds) < 300


def test_house_dataset_item_keys():
    table = _make_minimal_table(300)
    ds = HouseDataset(table, history_hours=2.0, horizon_steps=1)
    sample = ds[0]
    assert "x_limited" in sample
    assert "y" in sample
    # Pas de capteurs outdoor dans cette table minimale
    assert "x_outdoor" not in sample


def test_house_dataset_x_limited_shape():
    table = _make_minimal_table(300)
    ds = HouseDataset(table, history_hours=2.0, horizon_steps=1)
    sample = ds[0]
    expected_steps = len(compute_window_offsets(config.resolution_segments_for(2.0)))
    assert sample["x_limited"].shape == (expected_steps, ds.n_limited_features)


def test_house_dataset_y_shape():
    table = _make_minimal_table(300)
    ds = HouseDataset(table, history_hours=2.0, horizon_steps=1)
    sample = ds[0]
    assert sample["y"].shape == (ds.n_targets,)


def test_house_dataset_chronological_split():
    table = _make_minimal_table(300)
    ds = HouseDataset(table, history_hours=2.0, horizon_steps=1)
    train_idx, val_idx = ds.chronological_split(0.8)
    # Aucun chevauchement
    assert len(set(train_idx) & set(val_idx)) == 0
    # Les indices val sont tous après les indices train
    assert val_idx.min() > train_idx.max()


def test_house_dataset_normalization_applies():
    table = _make_minimal_table(300)
    ds = HouseDataset(table, history_hours=2.0, horizon_steps=1)
    train_idx, _ = ds.chronological_split(0.8)
    stats = ds.compute_normalization_stats(up_to_sample=len(train_idx))
    ds.apply_normalization(stats)
    # Après normalisation, les données doivent avoir des valeurs centrées réduites
    assert ds.limited_stats is not None
    assert ds.target_stats is not None


def test_house_dataset_no_indoor_raises():
    sep = config.COLUMN_SEP
    table = pd.DataFrame(
        {f"weather{sep}outdoor_temperature": np.ones(200)},
        index=pd.date_range("2026-01-01", periods=200, freq="2min", tz="UTC"),
    )
    with pytest.raises(ValueError, match="indoor"):
        HouseDataset(table, history_hours=2.0)


def test_house_dataset_no_limited_raises():
    sep = config.COLUMN_SEP
    table = pd.DataFrame(
        {f"indoor{sep}salon{sep}temperature": np.ones(200)},
        index=pd.date_range("2026-01-01", periods=200, freq="2min", tz="UTC"),
    )
    with pytest.raises(ValueError, match="weather"):
        HouseDataset(table, history_hours=2.0)
