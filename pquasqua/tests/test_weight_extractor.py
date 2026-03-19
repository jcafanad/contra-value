"""
Tests for BETOWeightExtractor, VALUE_NET, and weight computation.

Unit tests use mocked BETO models — no network required.
The empirical baseline (real BETO, 2026-03-18) is documented in
internal/CONFIG_ANALYSIS.md and in _compute_pseudo_perplexity's docstring.

Empirical test (BETO_EMPIRICAL=1): multi-speaker topology with 4 speakers
drawn from actual corpus claims, cross-speaker attacks wired, λ_⊥ and
Atom.weight reported across the full graph.
"""

import math
import os
import pytest

torch = pytest.importorskip("torch", reason="torch not installed; skipping weight extractor tests")

from unittest.mock import MagicMock
from pquasqua.transducer import Atom
from pquasqua.weight_extractor import (
    BETOWeightExtractor,
    ValueNetNode,
    VALUE_NET,
    _compute_pseudo_perplexity,
    _nli_engagement,
    _compute_atom_weight,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _mock_masked_lm(vocab_size: int = 100, seq_len: int = 5):
    """Masked LM returning uniform logits. Expected perplexity = vocab_size."""
    tokenizer = MagicMock()
    tokenizer.mask_token_id = 103
    tokenizer.return_value = {"input_ids": torch.zeros(1, seq_len, dtype=torch.long)}

    uniform_logits = torch.zeros(1, seq_len, vocab_size)
    model = MagicMock()
    model.return_value = MagicMock(logits=uniform_logits)
    return model, tokenizer


def _mock_nli(seq_len: int = 5):
    """
    NLI classifier returning uniform logits over 3 classes.
    Expected: p_contradiction = p_neutral = p_entailment = 1/3.
    Therefore max(p_E, p_C) = 1/3.
    """
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": torch.zeros(1, seq_len, dtype=torch.long)}

    uniform_logits = torch.zeros(1, 3)   # [contradiction, neutral, entailment]
    model = MagicMock()
    model.return_value = MagicMock(logits=uniform_logits)
    return model, tokenizer


def _mock_extractor(vocab_size: int = 100, seq_len: int = 5) -> BETOWeightExtractor:
    """BETOWeightExtractor with both models mocked."""
    e = BETOWeightExtractor.__new__(BETOWeightExtractor)
    e.model, e.tokenizer = _mock_masked_lm(vocab_size=vocab_size, seq_len=seq_len)
    e.nli_model, e.nli_tokenizer = _mock_nli(seq_len=seq_len)
    return e


# ---------------------------------------------------------------------------
# ValueNetNode
# ---------------------------------------------------------------------------

class TestValueNetNode:

    def test_value_net_has_four_labels(self):
        assert set(VALUE_NET.keys()) == {"value", "labour", "gender", "nature"}

    def test_all_nodes_have_theory_hypothesis(self):
        for node in VALUE_NET.values():
            assert node.theory_hypothesis.strip()

    def test_default_alpha_is_one(self):
        for node in VALUE_NET.values():
            assert node.alpha == 1.0

    def test_default_corpus_prototypes_empty(self):
        for node in VALUE_NET.values():
            assert node.corpus_prototypes == []

    def test_value_hypothesis_content(self):
        """Value hypothesis encodes fetishistic social mediation as property."""
        h = VALUE_NET["value"].theory_hypothesis
        assert "valor" in h
        assert "propiedad" in h or "medida" in h

    def test_labour_hypothesis_content(self):
        """Labour hypothesis encodes ontologisation as universal creative essence."""
        h = VALUE_NET["labour"].theory_hypothesis
        assert "trabajo" in h
        assert "esencia" in h or "universal" in h

    def test_gender_hypothesis_content(self):
        """Gender hypothesis encodes dissociation from value valorisation sphere."""
        h = VALUE_NET["gender"].theory_hypothesis
        assert "femenino" in h
        assert "separada" in h or "valor" in h

    def test_nature_hypothesis_content(self):
        """Nature hypothesis covers both passive-resource and intrinsic-value modes."""
        h = VALUE_NET["nature"].theory_hypothesis
        assert "naturaleza" in h
        assert "recurso" in h or "valor" in h


# ---------------------------------------------------------------------------
# _nli_engagement
# ---------------------------------------------------------------------------

class TestNLIEngagement:

    def test_uniform_logits_give_one_third(self):
        """Uniform logits: p_E = p_C = 1/3, so max = 1/3."""
        nli_model, nli_tokenizer = _mock_nli()
        score = _nli_engagement("texto", "hipótesis", nli_model, nli_tokenizer)
        assert abs(score - 1/3) < 1e-5

    def test_high_entailment_logit(self):
        """Logit strongly favouring entailment (index 2) gives score close to 1."""
        nli_model, nli_tokenizer = _mock_nli()
        nli_model.return_value = MagicMock(logits=torch.tensor([[0.0, 0.0, 10.0]]))
        score = _nli_engagement("texto", "hipótesis", nli_model, nli_tokenizer)
        assert score > 0.9

    def test_high_contradiction_logit(self):
        """Logit strongly favouring contradiction (index 0) also gives score close to 1."""
        nli_model, nli_tokenizer = _mock_nli()
        nli_model.return_value = MagicMock(logits=torch.tensor([[10.0, 0.0, 0.0]]))
        score = _nli_engagement("texto", "hipótesis", nli_model, nli_tokenizer)
        assert score > 0.9

    def test_direction_neutrality(self):
        """Entailment and contradiction scores are treated symmetrically."""
        nli_model, nli_tokenizer = _mock_nli()
        logit = torch.tensor([[8.0, 0.0, 0.0]])
        nli_model.return_value = MagicMock(logits=logit)
        score_c = _nli_engagement("texto", "hipótesis", nli_model, nli_tokenizer)

        logit_e = torch.tensor([[0.0, 0.0, 8.0]])
        nli_model.return_value = MagicMock(logits=logit_e)
        score_e = _nli_engagement("texto", "hipótesis", nli_model, nli_tokenizer)

        assert abs(score_c - score_e) < 1e-5


# ---------------------------------------------------------------------------
# _compute_atom_weight
# ---------------------------------------------------------------------------

class TestComputeAtomWeight:

    def setup_method(self):
        self.nli_model, self.nli_tokenizer = _mock_nli()

    def test_no_lval_returns_zero(self):
        atom = Atom(id=1, claim="texto", span_indices=(0, 5))
        assert _compute_atom_weight(atom, self.nli_model, self.nli_tokenizer) == 0.0

    def test_empty_claim_returns_zero(self):
        atom = Atom(id=1, claim="   ", span_indices=(0, 3), L_val={"value"})
        assert _compute_atom_weight(atom, self.nli_model, self.nli_tokenizer) == 0.0

    def test_known_label_returns_nli_score(self):
        atom = Atom(id=1, claim="texto", span_indices=(0, 5), L_val={"value"})
        w = _compute_atom_weight(atom, self.nli_model, self.nli_tokenizer)
        assert abs(w - 1/3) < 1e-5

    def test_unknown_label_returns_zero(self):
        atom = Atom(id=1, claim="texto", span_indices=(0, 5), L_val={"unknown_label"})
        assert _compute_atom_weight(atom, self.nli_model, self.nli_tokenizer) == 0.0

    def test_multiple_lval_takes_max(self):
        """With uniform logits all labels score 1/3; max is still 1/3."""
        atom = Atom(id=1, claim="texto", span_indices=(0, 5),
                    L_val={"value", "labour"})
        w = _compute_atom_weight(atom, self.nli_model, self.nli_tokenizer)
        assert abs(w - 1/3) < 1e-5

    def test_corpus_prototypes_blend(self):
        """alpha=0.5: score = 0.5*theory + 0.5*corpus = 0.5*(1/3) + 0.5*(1/3) = 1/3."""
        custom_net = {
            "value": ValueNetNode(
                label="value",
                theory_hypothesis="hipótesis teoría",
                corpus_prototypes=["prototipo corpus"],
                alpha=0.5,
            )
        }
        atom = Atom(id=1, claim="texto", span_indices=(0, 5), L_val={"value"})
        w = _compute_atom_weight(atom, self.nli_model, self.nli_tokenizer,
                                 value_net=custom_net)
        assert abs(w - 1/3) < 1e-5

    def test_alpha_zero_uses_corpus_only(self):
        """alpha=0.0: score derived entirely from corpus prototype."""
        custom_net = {
            "value": ValueNetNode(
                label="value",
                theory_hypothesis="ignorada",
                corpus_prototypes=["prototipo corpus"],
                alpha=0.0,
            )
        }
        atom = Atom(id=1, claim="texto", span_indices=(0, 5), L_val={"value"})
        w = _compute_atom_weight(atom, self.nli_model, self.nli_tokenizer,
                                 value_net=custom_net)
        assert abs(w - 1/3) < 1e-5


# ---------------------------------------------------------------------------
# BETOWeightExtractor unit tests
# ---------------------------------------------------------------------------

class TestBETOWeightExtractor:

    def test_annotate_populates_perplexity(self):
        e = _mock_extractor(vocab_size=100, seq_len=5)
        atom = Atom(id=1, claim="valor", span_indices=(0, 5))
        e.annotate([atom])
        assert atom.perplexity > 0.0

    def test_annotate_populates_weight_for_lval_atom(self):
        e = _mock_extractor()
        atom = Atom(id=1, claim="valor", span_indices=(0, 5), L_val={"value"})
        e.annotate([atom])
        assert atom.weight > 0.0

    def test_annotate_weight_zero_for_no_lval(self):
        e = _mock_extractor()
        atom = Atom(id=1, claim="texto sin etiqueta", span_indices=(0, 18))
        e.annotate([atom])
        assert atom.weight == 0.0

    def test_uniform_logits_give_vocab_size_perplexity(self):
        vocab_size = 100
        e = _mock_extractor(vocab_size=vocab_size, seq_len=5)
        atom = Atom(id=1, claim="valor", span_indices=(0, 5))
        e.annotate([atom])
        assert abs(atom.perplexity - vocab_size) < 1.0

    def test_empty_claim_skipped(self):
        e = _mock_extractor()
        atom = Atom(id=1, claim="   ", span_indices=(0, 3))
        e.annotate([atom])
        assert atom.perplexity == 0.0
        assert atom.weight == 0.0

    def test_annotate_multiple_atoms(self):
        e = _mock_extractor()
        atoms = [
            Atom(id=i, claim=f"claim{i}", span_indices=(i, i + 6))
            for i in range(3)
        ]
        e.annotate(atoms)
        assert all(a.perplexity > 0.0 for a in atoms)

    def test_perplexity_method(self):
        e = _mock_extractor(vocab_size=50, seq_len=5)
        ppl = e.perplexity("texto de prueba")
        assert abs(ppl - 50) < 1.0


# ---------------------------------------------------------------------------
# Empirical test — real BETO, multi-speaker topology
# ---------------------------------------------------------------------------

BETO_EMPIRICAL = os.environ.get("BETO_EMPIRICAL", "0") == "1"


@pytest.mark.skipif(not BETO_EMPIRICAL,
                    reason="Set BETO_EMPIRICAL=1 to run empirical weight extractor tests")
class TestMultiSpeakerTopology:
    """
    Four-speaker argumentation graph drawn from actual corpus claims.

    Speakers and atoms
    ------------------
    Speaker GR:
      A1  "valoramos el entorno natural"           L_val={value}
      A2  "la naturaleza no vale"                  L_dia={negation, ligature}
      A3  "poner a valer a través del trabajo"     L_val={labour}
          ↑ automatic subject atom — expected highest λ_⊥ among value atoms

    Speaker JG:
      B1  "el valor no es sólo el precio"          L_dia={negation, ligature}, L_val={value}
      B2  "dejar de valorar lo que es invaluable"  L_dia={negation}, L_val={value}

    Speaker AC:
      C1  "generó la opción de un ingreso para la mujer"  L_val={gender}
          (avoids the cooperative's proper name — community institutional
           nouns unknown to BETO score ~1939 due to out-of-vocabulary
           collapse, not due to any meaningful epistemic signal;
           TODO: institutional name generalisation is incomplete in the
           XMI fixtures — flag for anonymisation pass)

    Speaker JA (researcher):
      D1  "lo peligroso que resulta cuando instrumentalicemos la naturaleza"
                                                    L_ont={episteme}
          ↑ PUREF atom — Kantian framing of nature as instrument

    Attack topology
    ---------------
      B1 → A1  (cross-speaker: JG negates GR's value affirmation)
      B2 → A1  (cross-speaker: JG negates GR's value affirmation)
      A2 → B2  (cross-speaker: GR's "no vale" negates JG's valorisation)

    Expected λ_⊥ pattern
    ---------------------
      A3 > A1  ("poner a valer" carries automatic-subject idiom, high perplexity)
      D1 > A1  (Kantian framing, less colloquial, higher perplexity)
      All atoms << random (42584 baseline)

    Expected weight pattern
    -----------------------
      Atoms with L_val (A1, A3, B1, B2, C1) should have weight > 0.
      A2 and D1 have no L_val → weight = 0.0.
    """

    @pytest.fixture(scope="class")
    def topology(self):
        from pquasqua.transducer import Atom, Relation, is_defeater, is_attackable, PUREF
        from pquasqua.weight_extractor import BETOWeightExtractor

        # --- Atoms ---
        A1 = Atom(id=1,  claim="valoramos el entorno natural",
                  span_indices=(0, 30), L_val={"value"},
                  speaker="Speaker GR", turn=1)
        A2 = Atom(id=2,  claim="la naturaleza no vale",
                  span_indices=(31, 53), L_dia={"negation", "ligature"},
                  speaker="Speaker GR", turn=8)
        A3 = Atom(id=3,  claim="poner a valer a través del trabajo",
                  span_indices=(54, 88), L_val={"labour"},
                  speaker="Speaker GR", turn=10)
        B1 = Atom(id=4,  claim="el valor no es sólo el precio",
                  span_indices=(89, 119), L_dia={"negation", "ligature"}, L_val={"value"},
                  speaker="Speaker JG", turn=3)
        B2 = Atom(id=5,  claim="dejar de valorar lo que es invaluable",
                  span_indices=(120, 157), L_dia={"negation"}, L_val={"value"},
                  speaker="Speaker JG", turn=5)
        C1 = Atom(id=6,  claim="generó la opción de un ingreso para la mujer",
                  span_indices=(158, 203), L_val={"gender"},
                  speaker="Speaker AC", turn=6)
        D1 = Atom(id=7,  claim="lo peligroso que resulta cuando instrumentalicemos la naturaleza",
                  span_indices=(201, 263), L_ont={"episteme"},
                  speaker="Speaker JA", turn=2)

        atoms = [A1, A2, A3, B1, B2, C1, D1]

        # --- Relations (cross-speaker attacks) ---
        r_B1_A1 = Relation(source_id="4", target_id="1", connect=["moral"])
        r_B2_A1 = Relation(source_id="5", target_id="1", connect=["moral"])
        r_A2_B2 = Relation(source_id="2", target_id="5", connect=["valorisation"])

        B1.R_out.append(r_B1_A1); A1.R_in.append(r_B1_A1)
        B2.R_out.append(r_B2_A1); A1.R_in.append(r_B2_A1)
        A2.R_out.append(r_A2_B2); B2.R_in.append(r_A2_B2)

        # --- Populate defeaters ---
        defeaters   = [a for a in atoms if is_defeater(a)]
        attackables = [a for a in atoms if is_attackable(a)]
        for d in defeaters:
            for t in attackables:
                if d.speaker != t.speaker and d.turn > t.turn:
                    t.defeaters.add(d)

        # --- Annotate λ_⊥ and weight ---
        extractor = BETOWeightExtractor()
        extractor.annotate(atoms)

        return atoms, {"A1": A1, "A2": A2, "A3": A3,
                       "B1": B1, "B2": B2, "C1": C1, "D1": D1}

    def test_all_atoms_annotated(self, topology):
        atoms, _ = topology
        assert all(a.perplexity > 0.0 for a in atoms)

    def test_all_atoms_below_random_baseline(self, topology):
        """All real Spanish claims must be far below the random baseline (42584)."""
        atoms, _ = topology
        for a in atoms:
            assert a.perplexity < 1000, (
                f"Atom {a.id} ({repr(a.claim[:40])}) perplexity={a.perplexity:.1f} "
                "is suspiciously close to random — BETO may have failed to model Spanish."
            )

    def test_automatic_subject_atom_above_iberian_baseline(self, topology):
        """
        A3 ("poner a valer a través del trabajo") must score above the
        documented Iberian baseline (7.118, 2026-03-18).
        """
        _, named = topology
        IBERIAN_BASELINE = 7.118
        assert named["A3"].perplexity > IBERIAN_BASELINE, (
            f"λ_⊥(A3) = {named['A3'].perplexity:.3f} is below Iberian baseline "
            f"({IBERIAN_BASELINE}) — unexpected for a Colombian idiomatic claim."
        )

    def test_lval_atoms_have_weight(self, topology):
        """Atoms with L_val labels must have weight > 0."""
        _, named = topology
        for key in ("A1", "A3", "B1", "B2", "C1"):
            assert named[key].weight > 0.0, (
                f"Atom {key} has L_val but weight=0.0"
            )

    def test_no_lval_atoms_have_zero_weight(self, topology):
        """Atoms without L_val labels must have weight = 0.0."""
        _, named = topology
        assert named["A2"].weight == 0.0   # L_val empty (negation only)
        assert named["D1"].weight == 0.0   # L_ont only, no L_val

    def test_attack_topology_wired(self, topology):
        _, named = topology
        assert named["B1"] in named["A1"].defeaters
        assert named["B2"] in named["A1"].defeaters

    def test_puref_atom_present(self, topology):
        from pquasqua.transducer import PUREF
        _, named = topology
        assert PUREF(named["D1"])

    def test_print_topology_report(self, topology, capsys):
        """Print full λ_⊥ and weight report for the multi-speaker graph."""
        from pquasqua.transducer import is_defeater, is_attackable, PUREF
        atoms, _ = topology
        print("\n--- Multi-speaker λ_⊥ / weight topology report ---")
        print(f"{'ID':<4} {'Speaker':<12} {'λ_⊥':>7}  {'weight':>6}  "
              f"{'L_val':<15} {'flags'}")
        print("-" * 80)
        for a in sorted(atoms, key=lambda x: x.perplexity, reverse=True):
            flags = []
            if is_defeater(a):   flags.append("DEFEATER")
            if is_attackable(a): flags.append("ATTACKABLE")
            if PUREF(a):         flags.append("PUREF")
            if a.defeaters:      flags.append(f"←attacked×{len(a.defeaters)}")
            print(f"{a.id:<4} {a.speaker:<12} {a.perplexity:>7.3f}  "
                  f"{a.weight:>6.3f}  {str(a.L_val):<15} {', '.join(flags)}")
        print(f"\nAutomatic subject atom (A3): λ_⊥ = "
              f"{next(a for a in atoms if a.id==3).perplexity:.3f}")
        print("Record notable findings in internal/CONFIG_ANALYSIS.md.")
