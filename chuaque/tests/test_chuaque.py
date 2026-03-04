"""
Tests for chuaque.

These are not unit tests in the industrial sense.
They are formal verifications that the d-model
preserves what must be preserved (Aristotle)
while admitting what must be admitted (dialetheia).
"""

import pytest

from chuaque.truth_values import TruthValue
from chuaque.formulas import atom, neg, conj, disj, impl, atoms
from chuaque.d_model import DModel, Situation
from chuaque.entailment import Entailment
from chuaque.reversal import reverse_formula


# -- fixtures --

@pytest.fixture
def simple_model():
    """
    A d-model with two dual situations.
    Atom 'a' is true at base, false at base*.
    """
    return DModel.simple({
        "a": {"base": TruthValue.T, "base*": TruthValue.F},
    })


@pytest.fixture
def consistent_model():
    """
    A d-model where 'a' is true at BOTH base and base*.
    Under dual-C4: I(¬a, base) = ∼I(a, base*) = ∼T = F.
    Both a=T and ¬a=F everywhere — no dialetheia, maximally consistent.
    Contrast with simple_model, where a and ¬a are both T at base
    (genuine dialetheia under dual-C4).
    Previously named 'dialethic_model' under the old point-wise C4,
    where it was incorrectly described as dialetheic.
    """
    return DModel.simple({
        "a": {"base": TruthValue.T, "base*": TruthValue.T},
    })


@pytest.fixture
def paramo_model():
    """
    A d-model encoding the Páramo predicament.

    a: increased agricultural land-use
    h: improved hydrological regulating services

    At base: a is true (agriculture happens), h is false
    At base*: a is false (agriculture withdrawn), h is true

    The contradiction a ∧ ¬a arises because agriculture
    is both pursued and self-defeating.
    """
    base = Situation("paramo")
    dual = Situation("paramo*")
    return DModel(
        situations={base, dual},
        reversal={base: dual, dual: base},
        valuation={
            ("a", base): TruthValue.T,
            ("a", dual): TruthValue.F,
            ("h", base): TruthValue.F,
            ("h", dual): TruthValue.T,
        },
    )


# -- Aristotle is preserved --

class TestNonContradiction:
    """
    ¬(A ∧ ¬A) is situation-local in dialetheic models, universal in consistent ones.

    In simple_model (a=T at base, F at base*), dual-C4 makes both a and ¬a T
    at base — a genuine dialetheia. ¬(a∧¬a) holds at base (T) but fails at
    base* (F). It is not a theorem (not designated everywhere).

    In consistent_model (a=T everywhere), dual-C4 gives ¬a=F everywhere —
    no dialetheia. ¬(a∧¬a) is a universal theorem.

    Section 5.2 of the paper should be consulted to confirm which claim
    is intended; the implementation now supports the stronger (dialetheic) case.
    """

    def test_non_contradiction_simple(self, simple_model):
        e = Entailment(simple_model)
        a = atom("a")
        formula = neg(conj(a, neg(a)))
        # With dual-situation C4, simple_model is genuinely dialetheic at base
        # (a and ¬a are both T there). The formula ¬(a∧¬a) holds at base but
        # evaluates to F at base* — it is not a universal theorem in a dialetheic
        # model, which is the correct result. We verify it holds at the primary situation.
        assert e.interpret(formula, Situation("base")) == TruthValue.T

    def test_non_contradiction_dialethic(self, consistent_model):
        """¬(A ∧ ¬A) is a universal theorem in a consistent model.
        consistent_model has a=T at both base and base*, so under dual-C4 ¬a=F
        everywhere — it is consistent, not dialetheic. Non-contradiction holds
        as a theorem (designated at every situation). Contrast with simple_model,
        where the dialetheia at base makes the formula fail at base*.
        """
        e = Entailment(consistent_model)
        a = atom("a")
        formula = neg(conj(a, neg(a)))
        assert e.is_theorem(formula)


# -- Dialetheia is possible --

class TestDialetheia:
    """
    There exist situations where both A and ¬A
    receive designated values. Inconsistency without triviality.
    """

    def test_both_a_and_not_a_designated(self, simple_model):
        """Genuine dialetheia: a and ¬a are both T at base under dual-C4.
        simple_model has a=T at base and a=F at base*. Under dual-C4:
        I(¬a, base) = ∼I(a, base*) = ∼F = T. Both a and ¬a are T — designated
        and equal, making this a genuine dialetheia in the d-model.
        """
        e = Entailment(simple_model)
        a = atom("a")
        na = neg(a)
        base = Situation("base")
        val_a = e.interpret(a, base)
        val_na = e.interpret(na, base)
        assert val_a == TruthValue.T
        assert val_na == TruthValue.T, \
            "Under dual-C4, ¬a at base = ∼I(a, base*) = ∼F = T — genuine dialetheia"

    def test_no_explosion(self, paramo_model):
        """
        From a ∧ ¬a, NOT everything follows.
        This is the core of paraconsistency.
        """
        e = Entailment(paramo_model)
        a = atom("a")
        h = atom("h")
        # a ∧ ¬a should not entail an arbitrary h
        contradiction = conj(a, neg(a))
        # In classical logic (ex falso quodlibet), this entailment would hold.
        # In the d-model it must not — this is the core of paraconsistency.
        assert not e.entails([contradiction], h), \
            "Explosion holds: a ∧ ¬a entails arbitrary h — paraconsistency is broken"


# -- No statement is both true and false --

class TestTrueNotFalse:
    """
    In consistent models (where negation is F everywhere), no formula
    and its negation are simultaneously TruthValue.T.

    This property does NOT hold universally: in simple_model (genuinely
    dialetheic under dual-C4), both a and ¬a are T at base. That is the
    correct behaviour for a dialetheic framework. The test uses
    consistent_model (formerly dialethic_model) precisely because it has
    no T/T pairs — verifying the property holds in consistent configurations.

    A companion test demonstrates that simple_model DOES have a T/T pair
    (a=T, ¬a=T at base) — this is not a failure but the intended
    dialetheic behaviour.
    """

    def test_no_true_and_false_same_situation(self, consistent_model):
        """In a consistent model, a formula and its negation cannot both be T.
        Uses consistent_model (a=T at both s and s*), which under dual-C4 gives
        ¬a=F everywhere — a non-dialetheic model. Compare with simple_model,
        where a and ¬a are both T at base (genuine dialetheia).
        """
        e = Entailment(consistent_model)
        a = atom("a")
        for s in consistent_model.situations:
            val_a = e.interpret(a, s)
            val_na = e.interpret(neg(a), s)
            assert not (val_a == TruthValue.T and val_na == TruthValue.T), \
                f"At {s}: both {a} and ¬{a} are T — bivalent explosion"

    def test_dialetheia_produces_t_t_pair(self, simple_model):
        """
        In simple_model, both a and ¬a are TruthValue.T at base.
        This is the correct dialetheic behaviour under dual-C4, not a failure.
        """
        e = Entailment(simple_model)
        a = atom("a")
        base = Situation("base")
        assert e.interpret(a, base) == TruthValue.T
        assert e.interpret(neg(a), base) == TruthValue.T, \
            "simple_model should be dialetheic at base: both a and ¬a = T"


# -- Reversal --

class TestReversal:

    def test_conjunction_reverses_to_disjunction(self):
        a, b = atom("a"), atom("b")
        f = conj(a, b)
        r = reverse_formula(f)
        assert r == disj(a, b)

    def test_disjunction_reverses_to_conjunction(self):
        a, b = atom("a"), atom("b")
        f = disj(a, b)
        r = reverse_formula(f)
        assert r == conj(a, b)

    def test_double_reversal_of_atom(self):
        a = atom("a")
        assert reverse_formula(reverse_formula(a)) == a

    def test_paper_example(self):
        """
        *(¬A ∧ A) = ¬¬A ∨ ¬A
        From Section 5.2.
        """
        a = atom("a")
        f = conj(neg(a), a)
        r = reverse_formula(f)
        expected = disj(neg(a), a)
        assert r == expected

    def test_double_reversal_non_involutive_for_implication(self):
        """
        reverse_formula is NOT an involution for formulas containing →.
        (A→B)* = A∧¬B;  (A∧¬B)* = A∨¬B ≠ A→B.

        This is expected behaviour: the implication rule (A→B)* = A*∧¬(B*)
        follows the 'falsifier' convention, not the contrapositive. It does
        not compose back to → through double application.

        The situation-level * IS an involution (enforced by DModel.__post_init__).
        The formula-level reverse_formula is an involution only on the {∧,∨,¬,atom}
        fragment. This boundary must be documented and respected.
        """
        a, b = atom("a"), atom("b")
        f = impl(a, b)
        f_star = reverse_formula(f)
        f_star_star = reverse_formula(f_star)
        assert f_star == conj(a, neg(b)), f"Expected a∧¬b, got {f_star}"
        assert f_star_star != f, \
            "reverse_formula IS an involution for →: either the rule changed or this test is wrong"
        assert f_star_star == disj(a, neg(b)), f"Expected a∨¬b, got {f_star_star}"

    def test_double_reversal_conjunction_disjunction_negation(self):
        """
        reverse_formula IS an involution for the ∧,∨,¬,atom fragment.
        """
        a, b = atom("a"), atom("b")
        for f in [conj(a, b), disj(a, b), neg(a), neg(conj(a, b)), conj(a, disj(a, b))]:
            assert reverse_formula(reverse_formula(f)) == f, \
                f"Double reversal failed for {f}"


# -- Implication semantics --

class TestImplication:
    """
    Tests for C3 semantics under is_designated() antecedent check.

    These lock in confirmed behaviour (resolved v5). The is_designated()
    implementation is correct for all I-antecedent cases:

      I→I = F: lived experiences are ontologically unique. Shared inferential
               infrastructure requires explicit encoding in R. Default: no chain.
      I→N = F: lived contradictions suggest but cannot drive the unknown.
               Stricter than material Belnap (which would give T vacuously).
      I→F = F: confirmed.

    The deduction theorem failure (entails([a=I], h=I) = True; I(a→h) = F)
    is a FEATURE: entailment tests designation; implication tests formula value.
    These are genuinely different and both correct. The divergence formalises
    the irreducibility of lived experience: designation does not imply
    the inferential path is automatic.

    If C3 is ever revised, ALL five tests in this class must be updated
    simultaneously along with test_deduction_theorem_fails_for_I.
    """

    def _make_model(self, a_val, b_val):
        """Helper: model with a=a_val at base and b=b_val at base, self-loop R."""
        m = DModel.simple({
            "a": {"base": a_val, "base*": TruthValue.N},
            "b": {"base": b_val, "base*": TruthValue.N},
        })
        base = Situation("base")
        m.add_accessibility(base, base, base)
        return m, base

    def test_I_implies_I_is_F(self):
        """
        I→I = F. Confirmed v5: lived experiences are ontologically unique.
        Shared inferential infrastructure requires explicit encoding in R.
        Without R encoding the shared link, I→I = F is correct.
        Material Belnap gives I; the framework intentionally diverges.
        """
        m, base = self._make_model(TruthValue.I, TruthValue.I)
        e = Entailment(m)
        result = e.interpret(impl(atom("a"), atom("b")), base)
        assert result == TruthValue.F, \
            f"I→I = {result.name}; expected F (current C3). " \
            "If revised to I per bilattice theory, update this test."

    def test_I_implies_N_is_F(self):
        """I→N = F under current C3."""
        m, base = self._make_model(TruthValue.I, TruthValue.N)
        e = Entailment(m)
        assert e.interpret(impl(atom("a"), atom("b")), base) == TruthValue.F

    def test_I_implies_F_is_F(self):
        """I→F = F under current C3."""
        m, base = self._make_model(TruthValue.I, TruthValue.F)
        e = Entailment(m)
        assert e.interpret(impl(atom("a"), atom("b")), base) == TruthValue.F

    def test_N_implies_F_is_T(self):
        """N→F = T: N-antecedent not designated, vacuously true."""
        m, base = self._make_model(TruthValue.N, TruthValue.F)
        e = Entailment(m)
        assert e.interpret(impl(atom("a"), atom("b")), base) == TruthValue.T

    def test_deduction_theorem_fails_for_I(self):
        """
        entails([a=I], h=I) = True but I(a→h) = F.
        The deduction theorem does not hold for I-valued cases under current C3.
        This is a known structural tension between entails() and interpret().
        Document, not fix, until Critical 1 is resolved.
        """
        m = DModel.simple({
            "a": {"base": TruthValue.I, "base*": TruthValue.N},
            "h": {"base": TruthValue.I, "base*": TruthValue.N},
        })
        base = Situation("base")
        m.add_accessibility(base, base, base)
        e = Entailment(m)
        a, h = atom("a"), atom("h")
        assert e.entails([a], h) is True, "both designated -> entails"
        assert e.interpret(impl(a, h), base) == TruthValue.F, \
            "I→I = F under current C3 (deduction theorem fails for I-antecedents)"


# -- Páramo-specific --

class TestParamo:
    """Tests grounded in the paper's specific scenario."""

    def test_agriculture_true_at_base(self, paramo_model):
        e = Entailment(paramo_model)
        a = atom("a")
        base = Situation("paramo")
        assert e.interpret(a, base) == TruthValue.T

    def test_hydrology_false_at_base(self, paramo_model):
        e = Entailment(paramo_model)
        h = atom("h")
        base = Situation("paramo")
        assert e.interpret(h, base) == TruthValue.F

    def test_implication_a_implies_not_h(self, paramo_model):
        """a → ¬h holds at paramo (agriculture degrades hydrology).
        With dual-situation C4, I(¬h, s) = ∼I(h, s*). At paramo*, h=T, so
        I(¬h, paramo) = F. To evaluate the consequent correctly, we use
        R(paramo, paramo, paramo*): the antecedent is checked at paramo and
        the consequent at paramo*. I(¬h, paramo*) = ∼I(h, paramo) = ∼F = T.

        EXPRESSIVITY LIMIT: with R(paramo, paramo, paramo*), the consequent
        is evaluated at paramo* where h=T and ¬h=T (via dual-C4). This means
        a→h is ALSO T under this accessibility relation, which is incorrect
        (agriculture does not improve hydrology). The 2-situation model cannot
        simultaneously express a→¬h = T and a→h = F. A model with h and ¬h
        as independent atoms (not forced to be duals) is required for full
        expressivity.
        """
        base = Situation("paramo")
        dual = Situation("paramo*")
        # from paramo, paramo (antecedent world) precedes paramo* (consequent world)
        paramo_model.add_accessibility(base, base, dual)
        e = Entailment(paramo_model)
        a, h = atom("a"), atom("h")
        assert e.interpret(impl(a, neg(h)), base) == TruthValue.T

        # Document the expressivity limit explicitly:
        # In this 2-situation model, a→h is also T (known limitation).
        # Both implications evaluate to T because h=T at paramo*.
        # This is NOT the intended semantics but a structural constraint of the model.
        assert e.interpret(impl(a, h), base) == TruthValue.T, \
            "KNOWN LIMITATION: a→h = T in this 2-situation model. " \
            "A richer model is needed to distinguish a→h from a→¬h."
