"""Tests unitaires pour l'export ONNX (train.export_onnx) et les loaders ORT."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import config
from models.limited import LimitedModel
from models.full import FullModel
from train import export_onnx


N_STEPS = 60
N_FEATURES = 10
N_OUTDOOR = 6
N_TARGETS = 4
HISTORY_HOURS = 2.0


def _limited_checkpoint(model: LimitedModel) -> dict:
    return {
        "model_state": model.state_dict(),
        "n_limited_features": N_FEATURES,
        "n_targets": N_TARGETS,
        "history_hours": HISTORY_HOURS,
        "horizon_steps": 1,
        "limited_columns": [f"feat_{i}" for i in range(N_FEATURES)],
        "target_columns": [f"target_{i}" for i in range(N_TARGETS)],
        "limited_stats": None,
        "target_stats": None,
    }


def _full_checkpoint(model: FullModel) -> dict:
    return {
        "model_state": model.state_dict(),
        "n_limited_features": N_FEATURES,
        "n_outdoor_features": N_OUTDOOR,
        "n_targets": N_TARGETS,
        "history_hours": HISTORY_HOURS,
        "horizon_steps": 1,
        "limited_columns": [f"feat_{i}" for i in range(N_FEATURES)],
        "outdoor_columns": [f"out_{i}" for i in range(N_OUTDOOR)],
        "target_columns": [f"target_{i}" for i in range(N_TARGETS)],
        "limited_stats": None,
        "outdoor_stats": None,
        "target_stats": None,
    }


# ---------------------------------------------------------------------------
# Export limited
# ---------------------------------------------------------------------------

def test_export_onnx_limited_creates_file(tmp_path):
    model = LimitedModel(n_limited_features=N_FEATURES, n_targets=N_TARGETS, history_hours=HISTORY_HOURS)
    checkpoint = _limited_checkpoint(model)
    out = tmp_path / "limited.onnx"
    export_onnx(model, checkpoint, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_export_onnx_limited_embeds_metadata(tmp_path):
    import onnxruntime as ort

    model = LimitedModel(n_limited_features=N_FEATURES, n_targets=N_TARGETS, history_hours=HISTORY_HOURS)
    checkpoint = _limited_checkpoint(model)
    out = tmp_path / "limited.onnx"
    export_onnx(model, checkpoint, out)

    sess = ort.InferenceSession(str(out))
    custom = sess.get_modelmeta().custom_metadata_map
    assert "home_model_meta" in custom
    meta = json.loads(custom["home_model_meta"])
    assert meta["n_limited_features"] == N_FEATURES
    assert meta["n_targets"] == N_TARGETS
    assert meta["history_hours"] == HISTORY_HOURS
    assert "limited_stats" in meta
    assert "target_stats" in meta
    assert "n_outdoor_features" not in meta


def test_export_onnx_limited_inference(tmp_path):
    import onnxruntime as ort

    model = LimitedModel(n_limited_features=N_FEATURES, n_targets=N_TARGETS, history_hours=HISTORY_HOURS)
    model.eval()
    checkpoint = _limited_checkpoint(model)
    out = tmp_path / "limited.onnx"
    export_onnx(model, checkpoint, out)

    sess = ort.InferenceSession(str(out))
    x = np.random.randn(2, N_STEPS, N_FEATURES).astype(np.float32)
    preds = sess.run(["predictions"], {"x_limited": x})[0]
    assert preds.shape == (2, N_TARGETS)


def test_export_onnx_limited_matches_pytorch(tmp_path):
    import onnxruntime as ort

    model = LimitedModel(n_limited_features=N_FEATURES, n_targets=N_TARGETS, history_hours=HISTORY_HOURS)
    model.eval()
    checkpoint = _limited_checkpoint(model)
    out = tmp_path / "limited.onnx"
    export_onnx(model, checkpoint, out)

    x_np = np.random.randn(1, N_STEPS, N_FEATURES).astype(np.float32)
    x_pt = torch.from_numpy(x_np)

    with torch.no_grad():
        pt_pred = model(x_pt).numpy()

    sess = ort.InferenceSession(str(out))
    ort_pred = sess.run(["predictions"], {"x_limited": x_np})[0]

    np.testing.assert_allclose(pt_pred, ort_pred, atol=1e-5)


# ---------------------------------------------------------------------------
# Export full
# ---------------------------------------------------------------------------

def test_export_onnx_full_creates_file(tmp_path):
    limited = LimitedModel(n_limited_features=N_FEATURES, n_targets=N_TARGETS, history_hours=HISTORY_HOURS)
    full = FullModel(limited, n_outdoor_features=N_OUTDOOR, n_targets=N_TARGETS, history_hours=HISTORY_HOURS)
    checkpoint = _full_checkpoint(full)
    out = tmp_path / "full.onnx"
    export_onnx(full, checkpoint, out)
    assert out.exists()


def test_export_onnx_full_embeds_metadata(tmp_path):
    import onnxruntime as ort

    limited = LimitedModel(n_limited_features=N_FEATURES, n_targets=N_TARGETS, history_hours=HISTORY_HOURS)
    full = FullModel(limited, n_outdoor_features=N_OUTDOOR, n_targets=N_TARGETS, history_hours=HISTORY_HOURS)
    checkpoint = _full_checkpoint(full)
    out = tmp_path / "full.onnx"
    export_onnx(full, checkpoint, out)

    sess = ort.InferenceSession(str(out))
    meta = json.loads(sess.get_modelmeta().custom_metadata_map["home_model_meta"])
    assert meta["n_outdoor_features"] == N_OUTDOOR
    assert meta["outdoor_columns"] == [f"out_{i}" for i in range(N_OUTDOOR)]
    assert "outdoor_stats" in meta


def test_load_limited_onnx(tmp_path, monkeypatch):
    """load_limited_onnx reconstruit le checkpoint depuis les métadonnées embarquées."""
    import onnxruntime as ort
    from evaluate import load_limited_onnx

    model = LimitedModel(n_limited_features=N_FEATURES, n_targets=N_TARGETS, history_hours=HISTORY_HOURS)
    checkpoint = _limited_checkpoint(model)
    out = tmp_path / "limited.onnx"
    export_onnx(model, checkpoint, out)

    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path)
    sess, cp = load_limited_onnx(onnx_path=out)
    assert isinstance(sess, ort.InferenceSession)
    assert cp["n_limited_features"] == N_FEATURES
    assert cp["n_targets"] == N_TARGETS
    assert cp["limited_columns"] == [f"feat_{i}" for i in range(N_FEATURES)]


def test_load_full_onnx(tmp_path, monkeypatch):
    """load_full_onnx reconstruit le checkpoint depuis les métadonnées embarquées."""
    import onnxruntime as ort
    from evaluate import load_full_onnx

    limited = LimitedModel(n_limited_features=N_FEATURES, n_targets=N_TARGETS, history_hours=HISTORY_HOURS)
    full = FullModel(limited, n_outdoor_features=N_OUTDOOR, n_targets=N_TARGETS, history_hours=HISTORY_HOURS)
    checkpoint = _full_checkpoint(full)
    out = tmp_path / "full.onnx"
    export_onnx(full, checkpoint, out)

    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path)
    sess, cp = load_full_onnx(onnx_path=out)
    assert isinstance(sess, ort.InferenceSession)
    assert cp["n_outdoor_features"] == N_OUTDOOR


def test_export_onnx_full_inference(tmp_path):
    import onnxruntime as ort

    limited = LimitedModel(n_limited_features=N_FEATURES, n_targets=N_TARGETS, history_hours=HISTORY_HOURS)
    full = FullModel(limited, n_outdoor_features=N_OUTDOOR, n_targets=N_TARGETS, history_hours=HISTORY_HOURS)
    full.eval()
    checkpoint = _full_checkpoint(full)
    out = tmp_path / "full.onnx"
    export_onnx(full, checkpoint, out)

    sess = ort.InferenceSession(str(out))
    x_lim = np.random.randn(2, N_STEPS, N_FEATURES).astype(np.float32)
    x_out = np.random.randn(2, N_STEPS, N_OUTDOOR).astype(np.float32)
    outputs = sess.run(["predictions", "base_pred", "correction"], {"x_limited": x_lim, "x_outdoor": x_out})
    assert outputs[0].shape == (2, N_TARGETS)
    assert outputs[1].shape == (2, N_TARGETS)
    assert outputs[2].shape == (2, N_TARGETS)


def test_export_onnx_full_pred_equals_base_plus_correction(tmp_path):
    import onnxruntime as ort

    limited = LimitedModel(n_limited_features=N_FEATURES, n_targets=N_TARGETS, history_hours=HISTORY_HOURS)
    full = FullModel(limited, n_outdoor_features=N_OUTDOOR, n_targets=N_TARGETS, history_hours=HISTORY_HOURS)
    full.eval()
    checkpoint = _full_checkpoint(full)
    out = tmp_path / "full.onnx"
    export_onnx(full, checkpoint, out)

    sess = ort.InferenceSession(str(out))
    x_lim = np.random.randn(1, N_STEPS, N_FEATURES).astype(np.float32)
    x_out = np.random.randn(1, N_STEPS, N_OUTDOOR).astype(np.float32)
    pred, base, corr = sess.run(["predictions", "base_pred", "correction"],
                                 {"x_limited": x_lim, "x_outdoor": x_out})
    np.testing.assert_allclose(pred, base + corr, atol=1e-5)
