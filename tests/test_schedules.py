"""Tests for pure schedule/utility functions in schedules.py."""

import pytest

from schedules import get_lr_multiplier, get_muon_momentum, get_weight_decay, has_ve

# Default constants matching train.py defaults
WEIGHT_DECAY = 0.2
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0


class TestHasVE:
    def test_single_layer(self) -> None:
        assert has_ve(0, 1) is True

    def test_two_layers(self) -> None:
        assert has_ve(1, 2) is True
        assert has_ve(0, 2) is False

    def test_even_depth(self) -> None:
        results = [has_ve(i, 8) for i in range(8)]
        assert results[7] is True
        for i in range(7):
            assert results[i] != results[i + 1]

    def test_odd_depth(self) -> None:
        results = [has_ve(i, 7) for i in range(7)]
        assert results[6] is True
        for i in range(6):
            assert results[i] != results[i + 1]


class TestGetLRMultiplier:
    def _lr(self, progress: float) -> float:
        return get_lr_multiplier(progress, WARMUP_RATIO, WARMDOWN_RATIO, FINAL_LR_FRAC)

    def test_start(self) -> None:
        assert self._lr(0.0) == 1.0

    def test_flat_region(self) -> None:
        assert self._lr(0.25) == 1.0
        assert self._lr(0.49) == 1.0

    def test_zero_warmdown_ratio(self) -> None:
        assert get_lr_multiplier(0.99, WARMUP_RATIO, 0.0, FINAL_LR_FRAC) == 1.0
        assert get_lr_multiplier(1.0, WARMUP_RATIO, 0.0, FINAL_LR_FRAC) == FINAL_LR_FRAC

    def test_warmdown_start(self) -> None:
        assert self._lr(0.5) == pytest.approx(1.0)

    def test_warmdown_midpoint(self) -> None:
        assert self._lr(0.75) == pytest.approx(0.5)

    def test_end(self) -> None:
        assert self._lr(1.0) == pytest.approx(0.0)

    def test_monotonic_decrease_in_warmdown(self) -> None:
        values = [self._lr(p) for p in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]

    def test_with_warmup(self) -> None:
        lr = get_lr_multiplier(
            0.05, warmup_ratio=0.1, warmdown_ratio=0.5, final_lr_frac=0.0
        )
        assert lr == pytest.approx(0.5)

    def test_with_nonzero_final_lr(self) -> None:
        lr = get_lr_multiplier(
            1.0, warmup_ratio=0.0, warmdown_ratio=0.5, final_lr_frac=0.1
        )
        assert lr == pytest.approx(0.1)


class TestGetMuonMomentum:
    def test_step_zero(self) -> None:
        assert get_muon_momentum(0) == pytest.approx(0.85)

    def test_step_300(self) -> None:
        assert get_muon_momentum(300) == pytest.approx(0.95)

    def test_step_beyond_300(self) -> None:
        assert get_muon_momentum(1000) == pytest.approx(0.95)

    def test_midpoint(self) -> None:
        assert get_muon_momentum(150) == pytest.approx(0.90)

    def test_monotonic_increase(self) -> None:
        values = [get_muon_momentum(s) for s in range(0, 400, 50)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]

    def test_custom_ramp_steps(self) -> None:
        assert get_muon_momentum(50, ramp_steps=100) == pytest.approx(0.90)

    def test_zero_ramp_steps_raises(self) -> None:
        with pytest.raises(ValueError, match="ramp_steps must be positive"):
            get_muon_momentum(10, ramp_steps=0)


class TestGetWeightDecay:
    def test_start(self) -> None:
        assert get_weight_decay(0.0, WEIGHT_DECAY) == pytest.approx(WEIGHT_DECAY)

    def test_end(self) -> None:
        assert get_weight_decay(1.0, WEIGHT_DECAY) == pytest.approx(0.0)

    def test_midpoint(self) -> None:
        assert get_weight_decay(0.5, WEIGHT_DECAY) == pytest.approx(WEIGHT_DECAY / 2)

    def test_monotonic_decrease(self) -> None:
        values = [
            get_weight_decay(p, WEIGHT_DECAY) for p in [0.0, 0.25, 0.5, 0.75, 1.0]
        ]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]
