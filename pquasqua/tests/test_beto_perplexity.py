"""
BETO pseudo-perplexity test — epistemic violence measurement validation.

Validates that λ_⊥ (the perplexity coefficient in pquasqua/transducer.py)
is a meaningful signal rather than uniform noise:

  ppl_random >> ppl_paramuno >= ppl_iberian

If ppl_paramuno ≈ ppl_random: something is broken (BETO cannot model Spanish).
If ppl_paramuno >> ppl_iberian (e.g. 50 vs 8): BETO over-penalises Andean
  Spanish → λ_⊥ threshold needs recalibration.
If ppl_paramuno ≈ ppl_iberian: language mismatch is not inflating λ_⊥ →
  proceed with λ_⊥ as designed.
If ppl_paramuno > ppl_iberian (e.g. 12 vs 8): confirms BETO sees Andean
  features as slightly foreign → λ_⊥ is meaningful and should be preserved.

CRITICAL POSITIONING: if BETO systematically ⊥-flags Paramuno discourse,
this is the framework working as designed. Do NOT recalibrate to reduce
perplexity — the colonial geometry is real. Document and let the metric stand.

Test structure
--------------
TestPseudoPerplexityUnit  — mocked BETO, no network, always runs if torch present
TestBETOPerplexityEmpirical — real BETO download; skipped unless
                               BETO_EMPIRICAL=1 env var is set
"""

import os
import math
import pytest

torch = pytest.importorskip("torch", reason="torch not installed; skipping perplexity tests")
import torch as _torch  # noqa: E402 — importorskip returns the module

from unittest.mock import MagicMock, patch
from transformers import AutoTokenizer, AutoModelForMaskedLM


# ---------------------------------------------------------------------------
# Core function under test
# ---------------------------------------------------------------------------

def compute_pseudo_perplexity(text: str, model, tokenizer) -> float:
    """
    Compute pseudo-perplexity for a text using BETO (masked LM).

    BETO is BERT-style (masked LM), not autoregressive. True perplexity
    requires P(w_i | w_1,...,w_{i-1}), which BERT does not provide.
    We approximate via mean masked-token log-probability:

        pseudo_ppl = exp(-mean_i [ log P(w_i | w_{≠i}) ])

    Each token is masked in turn; the model predicts it from context.
    High pseudo-perplexity = BETO finds the text linguistically distant
    from its training distribution (Iberian/Chilean Spanish).

    This is λ_⊥: the epistemic violence coefficient. High values mark
    atoms where BETO's colonial geometry distorts the discourse signal.

    Args:
        text:      input text (one or a few sentences)
        model:     AutoModelForMaskedLM instance (dccuchile/bert-base-spanish-wwm-cased)
        tokenizer: matching AutoTokenizer

    Returns:
        pseudo-perplexity (float, ≥ 1.0)
    """
    encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encoding["input_ids"]

    log_probs = []
    # Skip position 0 ([CLS]) and last ([SEP])
    for i in range(1, input_ids.shape[1] - 1):
        masked = input_ids.clone()
        masked[0, i] = tokenizer.mask_token_id

        with _torch.no_grad():
            logits = model(masked).logits  # (1, seq_len, vocab_size)

        token_id = input_ids[0, i].item()
        prob = _torch.softmax(logits[0, i], dim=0)[token_id].item()
        # Guard against zero probability (numerical edge case)
        log_probs.append(math.log(max(prob, 1e-10)))

    if not log_probs:
        raise ValueError(f"No maskable tokens found in: {repr(text)}")

    import numpy as np
    return float(np.exp(-np.mean(log_probs)))


# ---------------------------------------------------------------------------
# Corpus samples
# ---------------------------------------------------------------------------

# Actual Paramuno Spanish — Speaker GR transcript, line 12.
# "pobrecita" is a Colombianism (empathetic diminutive), absent from
# Iberian Spanish; used here to introduce slight register distance from BETO.
PARAMUNO_SAMPLE = (
    "valoramos el entorno natural, pobrecita esa gente que vive en las ciudades, "
    "es que, qué bueno la tranquilidad que tenemos aquí, esto no lo tiene cualquiera"
)

# Equivalent Iberian Spanish — same semantic content, standard register.
IBERIAN_SAMPLE = (
    "valoramos el entorno natural, pobre gente la que vive en las ciudades, "
    "qué bueno la tranquilidad que tenemos aquí, esto no lo tiene cualquiera"
)

# Incoherent random token string — sanity-check upper bound.
RANDOM_SAMPLE = "asjdkf qwoeiru zxmcnv asldkfj bnmqwe rtyuio"

# Additional Paramuno samples for robustness
PARAMUNO_EXTENDED = [
    # Conceptual uncertainty — "no sé qué signifique valor"
    "aunque ya no sé qué signifique valor ya a estas alturas",
    # Relational value — "poner a valer a través del trabajo"
    "eso que es propio mío lo tengo que poner a valer a través del trabajo",
    # Lifeworld ethics — buen vivir framing
    "queremos que ese ambiente natural, la tranquilidad, el conocimiento, se traduzca en calidad de vida",
]


# ---------------------------------------------------------------------------
# Unit tests (mocked — no network, no download)
# ---------------------------------------------------------------------------

class TestPseudoPerplexityUnit:
    """
    Tests the compute_pseudo_perplexity logic without network access.
    Uses a minimal mock BERT that returns uniform logits.
    """

    def _make_mock_model_and_tokenizer(self, vocab_size: int = 100,
                                       seq_len: int = 5):
        """
        Minimal mock of AutoModelForMaskedLM + AutoTokenizer.

        Returns uniform logits so that P(token) = 1/vocab_size for every
        position. Expected pseudo-perplexity = vocab_size exactly.
        """
        import numpy as np

        tokenizer = MagicMock()
        tokenizer.mask_token_id = 103

        # Simulate encoding: [CLS] + (seq_len - 2) real tokens + [SEP]
        fake_ids = _torch.zeros(1, seq_len, dtype=_torch.long)
        tokenizer.return_value = {"input_ids": fake_ids}
        tokenizer.side_effect = None

        # Uniform logits: softmax → 1/vocab_size for every token
        uniform_logits = _torch.zeros(1, seq_len, vocab_size)
        model = MagicMock()
        model.return_value = MagicMock(logits=uniform_logits)

        return model, tokenizer

    def test_returns_float(self):
        model, tokenizer = self._make_mock_model_and_tokenizer(seq_len=5)
        result = compute_pseudo_perplexity("texto de prueba", model, tokenizer)
        assert isinstance(result, float)
        assert result >= 1.0

    def test_uniform_logits_give_vocab_size_perplexity(self):
        """
        With uniform logits, P(each token) = 1/vocab_size.
        pseudo_ppl = exp(-log(1/V)) = V.
        """
        vocab_size = 100
        model, tokenizer = self._make_mock_model_and_tokenizer(
            vocab_size=vocab_size, seq_len=5
        )
        result = compute_pseudo_perplexity("texto de prueba", model, tokenizer)
        assert abs(result - vocab_size) < 1.0, (
            f"Expected perplexity ≈ {vocab_size}, got {result:.2f}"
        )

    def test_certain_prediction_gives_low_perplexity(self):
        """
        If model is certain (P(correct token) → 1), perplexity → 1.
        """
        vocab_size = 100
        seq_len = 5

        # Logits: very high for token 0 at every position
        logits = _torch.full((1, seq_len, vocab_size), -1e9)
        logits[0, :, 0] = 1e9  # token_id=0 gets all probability mass

        tokenizer = MagicMock()
        tokenizer.mask_token_id = 103
        # All token ids = 0 so model is always right
        tokenizer.return_value = {"input_ids": _torch.zeros(1, seq_len, dtype=_torch.long)}
        model = MagicMock()
        model.return_value = MagicMock(logits=logits)

        result = compute_pseudo_perplexity("texto", model, tokenizer)
        assert result < 2.0, f"Certain model should give perplexity ≈ 1, got {result:.4f}"

    def test_empty_maskable_tokens_raises(self):
        """
        Sequence with only [CLS] and [SEP] (length 2) has no maskable tokens.
        """
        tokenizer = MagicMock()
        tokenizer.mask_token_id = 103
        tokenizer.return_value = {
            "input_ids": _torch.zeros(1, 2, dtype=_torch.long)  # only CLS + SEP
        }
        model = MagicMock()

        with pytest.raises(ValueError, match="No maskable tokens"):
            compute_pseudo_perplexity("x", model, tokenizer)


# ---------------------------------------------------------------------------
# Empirical tests (real BETO — requires network + BETO_EMPIRICAL=1)
# ---------------------------------------------------------------------------

BETO_EMPIRICAL = os.environ.get("BETO_EMPIRICAL", "0") == "1"
BETO_MODEL_ID  = "dccuchile/bert-base-spanish-wwm-cased"


@pytest.mark.skipif(not BETO_EMPIRICAL,
                    reason="Set BETO_EMPIRICAL=1 to run empirical BETO tests (requires network)")
class TestBETOPerplexityEmpirical:
    """
    Empirical validation of λ_⊥ as epistemic violence measure.

    Requires: network access, ~500MB BETO download, torch.
    Run with: BETO_EMPIRICAL=1 pytest pquasqua/tests/test_beto_perplexity.py

    Results should be documented in internal/CONFIG_ANALYSIS.md.
    """

    @pytest.fixture(scope="class")
    def beto(self):
        tokenizer = AutoTokenizer.from_pretrained(BETO_MODEL_ID)
        model = AutoModelForMaskedLM.from_pretrained(BETO_MODEL_ID)
        model.eval()
        return model, tokenizer

    def test_random_much_higher_than_real_language(self, beto):
        """
        Sanity check: BETO must find random token strings more perplexing
        than real Spanish. If this fails, something is fundamentally broken.
        """
        model, tokenizer = beto
        ppl_iberian = compute_pseudo_perplexity(IBERIAN_SAMPLE, model, tokenizer)
        ppl_random  = compute_pseudo_perplexity(RANDOM_SAMPLE,  model, tokenizer)
        assert ppl_random > ppl_iberian * 5, (
            f"Random perplexity ({ppl_random:.2f}) should be >> Iberian ({ppl_iberian:.2f})"
        )

    def test_paramuno_not_treated_as_random(self, beto):
        """
        Paramuno Spanish must not be as perplexing as random text.
        If this fails, BETO has collapsed — λ_⊥ is meaningless noise.
        """
        model, tokenizer = beto
        ppl_paramuno = compute_pseudo_perplexity(PARAMUNO_SAMPLE, model, tokenizer)
        ppl_random   = compute_pseudo_perplexity(RANDOM_SAMPLE,   model, tokenizer)
        assert ppl_paramuno < ppl_random * 0.5, (
            f"Paramuno ({ppl_paramuno:.2f}) is too close to random ({ppl_random:.2f})"
        )

    def test_report_paramuno_vs_iberian_gap(self, beto, capsys):
        """
        Measures and reports the Paramuno/Iberian perplexity gap.
        Does NOT assert a direction — documents the empirical finding.

        Expected hypothesis: ppl_paramuno >= ppl_iberian (BETO sees Andean
        features as slightly foreign). But the framework is correct either way:
        - Gap exists → λ_⊥ is a valid colonial geometry signal
        - No gap     → language mismatch is not inflating λ_⊥; proceed as-is
        - Large gap  → recalibrate λ_⊥ threshold before drawing conclusions
        """
        model, tokenizer = beto
        ppl_paramuno = compute_pseudo_perplexity(PARAMUNO_SAMPLE,  model, tokenizer)
        ppl_iberian  = compute_pseudo_perplexity(IBERIAN_SAMPLE,   model, tokenizer)
        ppl_random   = compute_pseudo_perplexity(RANDOM_SAMPLE,    model, tokenizer)

        ratio = ppl_paramuno / ppl_iberian if ppl_iberian > 0 else float("inf")

        print(f"\n--- BETO pseudo-perplexity results ---")
        print(f"Paramuno sample:  {ppl_paramuno:.3f}")
        print(f"Iberian sample:   {ppl_iberian:.3f}")
        print(f"Random sample:    {ppl_random:.3f}")
        print(f"Paramuno/Iberian ratio: {ratio:.3f}")

        if ratio > 5:
            print("INTERPRETATION: Large gap — λ_⊥ threshold needs recalibration.")
        elif ratio > 1.1:
            print("INTERPRETATION: Moderate gap — λ_⊥ is a meaningful colonial geometry signal.")
        else:
            print("INTERPRETATION: No significant gap — language mismatch is not inflating λ_⊥.")

        print("Record these results in internal/CONFIG_ANALYSIS.md.")

    def test_extended_paramuno_samples(self, beto):
        """
        Computes perplexity for additional Paramuno samples. No assertion —
        documents the per-sample distribution of λ_⊥ values.
        """
        model, tokenizer = beto
        for sample in PARAMUNO_EXTENDED:
            ppl = compute_pseudo_perplexity(sample, model, tokenizer)
            # All real Spanish should be well below random
            ppl_random = compute_pseudo_perplexity(RANDOM_SAMPLE, model, tokenizer)
            assert ppl < ppl_random, (
                f"Paramuno sample perplexity ({ppl:.2f}) >= random ({ppl_random:.2f}):\n  {sample}"
            )
