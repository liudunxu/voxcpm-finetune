import importlib
from pathlib import Path

import pytest
import torch


def test_adapter_spectrum_matches_full_product(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    module = importlib.import_module('audit_training_hparams')
    generator = torch.Generator().manual_seed(123)
    a = torch.randn(64, 80, generator=generator, dtype=torch.float64)
    b = torch.randn(90, 64, generator=generator, dtype=torch.float64)
    expected = torch.linalg.svdvals(b @ a).square()
    result = module.spectrum(a, b)
    assert result['update_frobenius'] == pytest.approx(float(expected.sum().sqrt()))
    assert result['top32_energy_share'] == pytest.approx(float(expected[:32].sum()/expected.sum()))
    assert result['rank95'] == int(torch.searchsorted(expected.cumsum(0), expected.sum()*.95)) + 1
    assert module.spectrum(a, torch.zeros_like(b))['rank95'] == 0
