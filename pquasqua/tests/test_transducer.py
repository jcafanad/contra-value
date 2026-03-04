"""
Tests for SituatedTransducer squeeze_map logic.
These tests use mock NLI scores — no network access required.
Skipped if torch is not installed (torch is optional at test time).
"""

import math
import pytest
torch = pytest.importorskip("torch", reason="torch not installed; skipping transducer tests")
from unittest.mock import MagicMock
from chuaque.formulas import Atom, Implication
from pquasqua.transducer import SituatedTransducer


def _mock_transducer(
    desired_entailment: float,
    desired_contradiction: float,
) -> SituatedTransducer:
    """
    Build a SituatedTransducer with mocked NLI output.

    Args:
        desired_entailment:    desired post-softmax entailment probability
        desired_contradiction: desired post-softmax contradiction probability
        Neutral probability is inferred as 1 - E - C (must be > 0).

    Uses log-probabilities as logits: softmax(log(p)) = p exactly.
    This avoids the double-softmax bug where passing raw probabilities as
    logits compresses them toward uniform on the second softmax pass.
    """
    desired_neutral = 1.0 - desired_entailment - desired_contradiction
    assert desired_neutral > 0, \
        f"E+C must be < 1.0; got E={desired_entailment}, C={desired_contradiction}"

    # softmax(log(p)) = p exactly — no compression toward uniform
    logits = torch.tensor([[
        math.log(desired_entailment),
        math.log(desired_neutral),
        math.log(desired_contradiction),
    ]])

    t = SituatedTransducer.__new__(SituatedTransducer)
    t.entailment_floor     = 0.5
    t.designated_threshold = 0.4
    t._adu_cache           = {}

    # Mock embed -> zero vector
    t.embed_tokenizer = MagicMock()
    t.embed_model     = MagicMock()
    t.embed_model.return_value = MagicMock(
        last_hidden_state=torch.zeros(1, 1, 768)
    )
    t.embed_tokenizer.return_value = {
        "attention_mask": torch.ones(1, 1),
        "input_ids":      torch.zeros(1, 1, dtype=torch.long),
    }
    t.nli_tokenizer = MagicMock()
    t.nli_tokenizer.return_value = {
        "input_ids":      torch.zeros(1, 1, dtype=torch.long),
        "attention_mask": torch.ones(1, 1),
    }
    t.nli_model = MagicMock()
    t.nli_model.return_value = MagicMock(logits=logits)
    t.nli_labels = {0: "entailment", 1: "neutral", 2: "contradiction"}
    return t


class TestSqueezeMap:
    def test_implication_case(self):
        """E>0.5, C≤0.4: Implication added to support; squeeze_map has Implication key."""
        t = _mock_transducer(0.80, 0.10)  # N=0.10 — valid
        arg, squeeze_map, adus = t.mine_argument(["premise"], "claim", "A")
        assert any(isinstance(k, Implication) for k in squeeze_map)
        assert Implication(Atom("premise"), Atom("claim")) in arg.support

    def test_dialetheia_case(self):
        """E>0.5, C>0.4: ⊗-atom in squeeze_map; NO Implication in support."""
        t = _mock_transducer(0.51, 0.41)  # N=0.08 — valid, E>0.5 AND C>0.4
        arg, squeeze_map, adus = t.mine_argument(["premise"], "claim", "A")
        oplus_keys = [k for k in squeeze_map if isinstance(k, Atom) and "⊗" in k.name]
        assert len(oplus_keys) == 1
        assert oplus_keys[0] == Atom("premise⊗claim")
        assert not any(isinstance(k, Implication) for k in squeeze_map)

    def test_refutation_case(self):
        """
        E≤0.5, C>0.4: ⊘-atom in squeeze_map; no Implication in support.
        Paramuno lifeworld case: a Paramuno speaker asserts territorial sovereignty —
        "este páramo es nuestro territorio" — actively contradicting the State's
        public-utility resource framing. The NLI classifier finds high contradiction,
        low entailment. The Paramuno no must leave a formal trace.
        See pipeline.py module docstring for the full round-by-round account.
        """
        t = _mock_transducer(0.05, 0.85)  # N=0.10 — valid
        arg, squeeze_map, adus = t.mine_argument(
            ["paramuno_territorio"], "recurso_hidrico_estado", "paramuno_refutation"
        )
        ominus_keys = [k for k in squeeze_map if isinstance(k, Atom) and "⊘" in k.name]
        assert len(ominus_keys) == 1
        assert ominus_keys[0] == Atom("paramuno_territorio⊘recurso_hidrico_estado")
        assert not any(isinstance(k, Implication) for k in squeeze_map)

    def test_silent_case(self):
        """E≤0.5, C≤0.4: neither signal; squeeze_map empty."""
        t = _mock_transducer(0.34, 0.34)  # N=0.32 — valid
        arg, squeeze_map, adus = t.mine_argument(["premise"], "claim", "A")
        assert len(squeeze_map) == 0

    def test_adu_caching(self):
        """Same text produces the same ADU object from cache."""
        t = _mock_transducer(0.80, 0.10)
        adu1 = t._get_adu("el páramo")
        adu2 = t._get_adu("el páramo")
        assert adu1 is adu2
        assert t.embed_model.call_count == 1  # embedded once only
