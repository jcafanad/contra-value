"""
Torch-free tests for compute_initial_weight (pquasqua/weight_extractor.py).

These tests use plain Python lists of mocked logits — no torch dependency,
no network access. The test file is always collected regardless of whether
torch is installed.

Design note
-----------
compute_initial_weight operates on an already-computed 3-element logit
sequence, applying softmax in pure Python. Tests verify:
  - Uniform logits → max(p_E, p_C) = 1/3
  - High entailment logit → score close to 1
  - High contradiction logit → score close to 1 (direction-neutral)
  - Entailment and contradiction are treated symmetrically
  - Numerical stability under large logit spread
  - Input via plain Python list (the torch-free contract)

XNLI label order (Recognai/bert-base-spanish-wwm-cased-xnli):
    index 0 = contradiction, index 1 = neutral, index 2 = entailment
"""

import math
import pytest

from pquasqua.weight_extractor import compute_initial_weight


class TestComputeInitialWeight:

    def test_uniform_logits_give_one_third(self):
        """
        Uniform logits (0, 0, 0) → p_C = p_N = p_E = 1/3.
        max(p_E, p_C) = 1/3.
        """
        score = compute_initial_weight([0.0, 0.0, 0.0])
        assert abs(score - 1 / 3) < 1e-6

    def test_high_entailment_logit_gives_score_near_one(self):
        """Large entailment logit (index 2) → p_E ≈ 1 → score ≈ 1."""
        score = compute_initial_weight([0.0, 0.0, 20.0])
        assert score > 0.99

    def test_high_contradiction_logit_gives_score_near_one(self):
        """Large contradiction logit (index 0) → p_C ≈ 1 → score ≈ 1."""
        score = compute_initial_weight([20.0, 0.0, 0.0])
        assert score > 0.99

    def test_direction_neutrality(self):
        """
        Symmetric logits for entailment vs. contradiction must yield
        identical scores — direction-neutrality is the key invariant.
        """
        entailment_dominant = [0.0, 0.0, 8.0]
        contradiction_dominant = [8.0, 0.0, 0.0]
        score_e = compute_initial_weight(entailment_dominant)
        score_c = compute_initial_weight(contradiction_dominant)
        assert abs(score_e - score_c) < 1e-6

    def test_neutral_dominant_logit_gives_low_score(self):
        """Large neutral logit (index 1) → p_E and p_C are both small."""
        score = compute_initial_weight([0.0, 20.0, 0.0])
        # p_E = p_C ≈ 0 when neutral dominates; score ≈ 0
        assert score < 0.01

    def test_score_is_in_unit_interval(self):
        """Score must lie in [0, 1] for any input."""
        for logits in [
            [1.0, 2.0, 3.0],
            [-5.0, 0.0, 5.0],
            [100.0, 0.0, 0.0],
            [0.0, 0.0, 100.0],
        ]:
            score = compute_initial_weight(logits)
            assert 0.0 <= score <= 1.0, (
                f"compute_initial_weight({logits}) = {score} is outside [0,1]"
            )

    def test_numerical_stability_large_spread(self):
        """
        Large logit spread should not cause overflow or NaN.
        Numerical stability is achieved via the max-subtraction trick in the
        pure-Python softmax implementation.
        """
        score = compute_initial_weight([500.0, -500.0, 0.0])
        assert math.isfinite(score)
        assert abs(score - 1.0) < 1e-6   # contradiction dominates with logit 500

    def test_accepts_plain_list(self):
        """
        The torch-free contract: input must be a plain Python list.
        No torch import required anywhere in this test file.
        """
        logits = [1.0, 0.5, 2.0]   # entailment logit highest
        score = compute_initial_weight(logits)
        assert isinstance(score, float)
        assert score > 0.0

    def test_dialetheia_case_intermediate_score(self):
        """
        High-E and high-C simultaneously (genuine dialetheia).
        score = max(p_E, p_C) is non-trivially large for both.
        Paramuno lifeworld case: NLI cannot decide between entailment and
        contradiction — this is computationally present in initial_weight.
        """
        # E logit = 1.5, C logit = 1.4, N logit = 0.0
        score = compute_initial_weight([1.4, 0.0, 1.5])
        # Both p_E and p_C are significant; score should be clearly above 1/3
        assert score > 0.35

    def test_return_type_is_float(self):
        """Return value must be a native Python float, not numpy or torch."""
        score = compute_initial_weight([0.0, 0.0, 0.0])
        assert type(score) is float
