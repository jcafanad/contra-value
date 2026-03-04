"""
chuaque/truth_values.py

The Belnap-Dunn FOUR Bilattice.

The mathematical foundation of the contravalue repository.
Implements the four-valued logic required for paraconsistent argumentation.

TWO CRITICAL ALGEBRAIC FACTS — distinct in mechanism, convergent in consequence:

    FACT A. N ⊕_t I = T   (The Violence of Ignorance)
        N (Neither / no information) and I (Both / lived contradiction) are
        incomparable under ≤_t. Their least upper bound under ≤_t is T,
        the unique element strictly above both.

        THIS IS NOT POLITICALLY NEUTRAL. Trace the computational consequence
        during the Knaster-Tarski iteration in ParaconsistentAFSolver:

            Round 1: all non-dialetheic arguments initialise at ⊥_k = N.
            A State argument (B) is initialised at N — unknown, not yet
            evaluated. B attacks the Paramuno's dialetheic argument (A).
            The dialetheic partner injects the constant I (ontological,
            not inferential — see pvaf.py on partner injection).

            attack_value(A) = F ⊕_t N ⊕_t I = N ⊕_t I = T.
            new_label(A) = ∼(T) = F.

            The subaltern's lived contradiction is erased to F in Round 1
            — not because the State's argument is certified true (T), but
            because it is UNKNOWN (N).

        In a colonial geometry, ignorance is not neutral. When the State
        lacks legibility (N) regarding the subaltern's lifeworld, the
        instrumental machinery — forced by ≤_t to resolve incomparables —
        synthesises a hallucinated bivalent certainty (T) and wields it
        immediately to erase the contradiction (F).

        HOWEVER: this erasure is PROVISIONAL. It is self-correcting under
        iteration. If B has no genuine classical warrant and converges to F,
        then in subsequent rounds: attack_value(A) = F ⊕_t I = I, and
        new_label(A) = ∼(I) = I. The Paramuno argument recovers to I at
        the fixed point. Ignorance enacts transient erasure; it cannot
        sustain permanent erasure without a genuine T behind it.

        IMPLEMENTATION DEPENDENCY: The Violence of Ignorance is activated
        by the design decision in pvaf.py to inject the CONSTANT TruthValue.I
        from dialetheic partners, rather than their current label. If partners
        injected their current label (N in Round 0), attack_value would be
        F ⊕_t N ⊕_t N = N, ∼(N) = N — no erasure. The political meaning
        of Fact A is inseparable from this implementation choice.

    FACT B. T ⊕_t I = T   (The Violence of Certified Truth)
        The State's certified truth (T) absorbs the subaltern's lived
        contradiction (I) under the truth-lattice join. T is the absorbing
        element of ⊕_t. Negating: ∼(T) = F.

        A dialetheic argument attacked by a T-labelled dominant argument
        is driven to F at the fixed point. This erasure is PERMANENT: no
        subsequent iteration round can recover it, because T ⊕_t x = T
        for all x. Once a genuinely T-labelled attacker is in the graph,
        the subaltern's contradiction is erased without possibility of
        recovery under this evaluation framework.

        This is the computational formalisation of certified colonial
        violence: the State that knows its dominant logic and uses it.

    THE DISTINCTION BETWEEN A AND B:

        Fact A — Violence of Ignorance:    provisional, self-correcting,
            activated by the constant-I partner injection design,
            a property of the iteration's transient states.

        Fact B — Violence of Certified Truth: permanent, non-recoverable,
            a property of the fixed point itself.

        Both are violent. They are not equivalent. Fact A is the
        instrument of a State that does not yet know — but whose
        institutional machinery erases before it learns. Fact B is the
        instrument of a State that knows and acts. The first is the
        violence of bureaucratic indifference; the second is the violence
        of deliberate domination. In symbolic AI, arithmetic is ontology:
        both are encoded in the same four-element lattice.
"""

from __future__ import annotations
from enum import Enum, unique
from typing import Dict, Tuple


@unique
class TruthValue(Enum):
    """
    The four truth values of the Belnap-Dunn bilattice.

        N — Neither: no information. ⊥_k (bottom of information ordering).
        F — False:   classically rejected. ⊥_t (bottom of truth ordering).
        T — True:    classically accepted. ⊤_t (top of truth ordering).
        I — Both:    lived contradiction; simultaneously supported and negated.
                     ⊤_k (top of information ordering). The Paramuno dialetheia.

    Truth ordering (≤_t):
        F <_t N,  F <_t I,  N <_t T,  I <_t T.
        N and I are ≤_t-INCOMPARABLE.
        Hasse: F at bottom, T at top, N and I as two incomparable middle nodes.

    Information ordering (≤_k):
        N <_k F,  N <_k T,  F <_k I,  T <_k I.
        F and T are ≤_k-INCOMPARABLE.
        Hasse: N at bottom, I at top, F and T as two incomparable middle nodes.
    """
    N = "Neither"
    F = "False"
    T = "True"
    I = "Both"

    def __repr__(self) -> str:
        return f"TruthValue.{self.name}"

    def is_designated(self) -> bool:
        """
        Designated values represent acceptance in paraconsistent logics.
        Both T and I are designated: a formula is accepted if it is true
        OR if it is a lived contradiction that the audience tolerates.
        """
        return self in (TruthValue.T, TruthValue.I)

    # -------------------------------------------------------------------------
    # Truth ordering
    # -------------------------------------------------------------------------

    def leq_t(self, other: TruthValue) -> bool:
        """x ≤_t y: x is at most as true as y."""
        return other in _TRUTH_ORDER_ABOVE[self]

    # -------------------------------------------------------------------------
    # Information ordering
    # -------------------------------------------------------------------------

    def leq_k(self, other: TruthValue) -> bool:
        """x ≤_k y: x carries at most as much information as y."""
        return other in _INFO_ORDER_ABOVE[self]

    # -------------------------------------------------------------------------
    # Belnap Negation ∼
    # -------------------------------------------------------------------------

    @property
    def negation(self) -> TruthValue:
        """
        ∼x: Belnap negation.
            ∼F = T,  ∼T = F,  ∼N = N,  ∼I = I.

        Properties (both required for Knaster-Tarski convergence in pvaf.py):
            (a) Antitone on ≤_t:  x ≤_t y  →  ∼y ≤_t ∼x
            (b) Monotone on ≤_k:  x ≤_k y  →  ∼x ≤_k ∼y

        Proof of (b) — exhaustive:
            N ≤_k F: ∼N=N, ∼F=T, N ≤_k T ✓
            N ≤_k T: ∼N=N, ∼T=F, N ≤_k F ✓
            F ≤_k I: ∼F=T, ∼I=I, T ≤_k I ✓
            T ≤_k I: ∼T=F, ∼I=I, F ≤_k I ✓
        """
        return _NEGATION[self]

    # -------------------------------------------------------------------------
    # Truth-lattice join ⊕_t
    # -------------------------------------------------------------------------

    def join_t(self, other: TruthValue) -> TruthValue:
        """
        x ⊕_t y: least upper bound under ≤_t.
        Identity: F.  Absorbing: T.

        Full table (symmetric):
              ⊕_t | N  F  T  I
            ------+------------
              N   | N  N  T  T   ← N ⊕_t I = T: bilattice resolution of
              F   | N  F  T  I     incomparables (Fact A, NOT Epistemic Erasure)
              T   | T  T  T  T   ← T ⊕_t I = T: Epistemic Erasure (Fact B)
              I   | T  I  T  I
        """
        return _JOIN_T[(self, other)]

    # -------------------------------------------------------------------------
    # Information-lattice join ⊕_k
    # -------------------------------------------------------------------------

    def join_k(self, other: TruthValue) -> TruthValue:
        """
        x ⊕_k y: least upper bound under ≤_k.
        Identity: N.  Absorbing: I.

        Key fact: F ⊕_k T = I.
        When the NLI classifier simultaneously supports Entailment (→T) and
        Contradiction (→F) for a premise-claim pair, their ⊕_k-join is I —
        the formal representation of classifier-detected paraconsistency.

        Full table (symmetric):
              ⊕_k | N  F  T  I
            ------+------------
              N   | N  F  T  I
              F   | F  F  I  I
              T   | T  I  T  I
              I   | I  I  I  I
        """
        return _JOIN_K[(self, other)]

    # -------------------------------------------------------------------------
    # Class-level aliases (for calling conventions in pvaf.py)
    # -------------------------------------------------------------------------

    @classmethod
    def join(cls, a: TruthValue, b: TruthValue) -> TruthValue:
        """Alias for join_t — the operation used by ParaconsistentAFSolver."""
        return a.join_t(b)

    @classmethod
    def negate(cls, a: TruthValue) -> TruthValue:
        """Alias for .negation property — for use in procedural contexts."""
        return a.negation

    @classmethod
    def meet(cls, a: TruthValue, b: TruthValue) -> TruthValue:
        """
        Truth-lattice meet (glb under ≤_t): De Morgan dual of join_t.

        meet_t(a, b) = ∼(∼a ⊕_t ∼b)

        Full table (symmetric):
              meet | N  F  T  I
            ------+------------
              N   | N  F  N  F
              F   | F  F  F  F
              T   | N  F  T  I
              I   | F  F  I  I
        """
        return a.negation.join_t(b.negation).negation

    @staticmethod
    def truth_join(val1: TruthValue, val2: TruthValue) -> TruthValue:
        """Static alias for join_t, retained for backward compatibility."""
        return val1.join_t(val2)


# =============================================================================
# Lookup tables — explicit enumeration, auditable entry by entry
# =============================================================================

N, F, T, I = TruthValue.N, TruthValue.F, TruthValue.T, TruthValue.I

_TRUTH_ORDER_ABOVE: Dict[TruthValue, frozenset] = {
    N: frozenset({N, T}),
    F: frozenset({F, N, I, T}),
    T: frozenset({T}),
    I: frozenset({I, T}),
}

_INFO_ORDER_ABOVE: Dict[TruthValue, frozenset] = {
    N: frozenset({N, F, T, I}),
    F: frozenset({F, I}),
    T: frozenset({T, I}),
    I: frozenset({I}),
}

_NEGATION: Dict[TruthValue, TruthValue] = {N: N, F: T, T: F, I: I}

_JOIN_T: Dict[Tuple[TruthValue, TruthValue], TruthValue] = {
    (N, N): N, (N, F): N, (N, T): T, (N, I): T,
    (F, N): N, (F, F): F, (F, T): T, (F, I): I,
    (T, N): T, (T, F): T, (T, T): T, (T, I): T,
    (I, N): T, (I, F): I, (I, T): T, (I, I): I,
}

_JOIN_K: Dict[Tuple[TruthValue, TruthValue], TruthValue] = {
    (N, N): N, (N, F): F, (N, T): T, (N, I): I,
    (F, N): F, (F, F): F, (F, T): I, (F, I): I,
    (T, N): T, (T, F): I, (T, T): T, (T, I): I,
    (I, N): I, (I, F): I, (I, T): I, (I, I): I,
}


# =============================================================================
# Runtime self-verification — runs at import time
# =============================================================================

def _verify_bilattice_axioms() -> None:
    """
    Verifies the axioms that pvaf.py and ddg_patch.py depend on.
    Distinguishes the two algebraic facts that were previously conflated.
    """
    vals = list(TruthValue)

    # Involution
    for x in vals:
        assert x.negation.negation == x, f"Involution failed: {x}"

    # ∼ antitone on ≤_t
    for x in vals:
        for y in vals:
            if x.leq_t(y):
                assert y.negation.leq_t(x.negation), \
                    f"∼ not antitone on ≤_t: {x} ≤_t {y} but ∼{y} ≰_t ∼{x}"

    # ∼ monotone on ≤_k (required for Knaster-Tarski)
    for x in vals:
        for y in vals:
            if x.leq_k(y):
                assert x.negation.leq_k(y.negation), \
                    f"∼ not monotone on ≤_k: {x} ≤_k {y} but ∼{x} ≰_k ∼{y}"

    # join_t identity and absorption
    for x in vals:
        assert F.join_t(x) == x, f"join_t identity failed: F ⊕_t {x} ≠ {x}"
        assert T.join_t(x) == T, f"join_t absorption failed: T ⊕_t {x} ≠ T"

    # join_k identity and absorption
    for x in vals:
        assert N.join_k(x) == x, f"join_k identity failed: N ⊕_k {x} ≠ {x}"
        assert I.join_k(x) == I, f"join_k absorption failed: I ⊕_k {x} ≠ I"

    # FACT A — Violence of Ignorance (provisional erasure under N ⊕_t I = T)
    # N and I are ≤_t-incomparable; their lub is T. Politically non-neutral:
    # the State's ignorance (N), forced to interact with the subaltern's
    # contradiction (I) by the instrumental truth-join, synthesises a
    # hallucinated T that enacts transient erasure. Self-correcting under
    # iteration if B has no genuine T warrant. See module docstring.
    assert N.join_t(I) == T, \
        "Violence of Ignorance lemma failed: N ⊕_t I ≠ T"

    # FACT B — Violence of Certified Truth (permanent erasure under T ⊕_t I = T)
    # T is the absorbing element of ⊕_t. A genuine T-labelled attacker
    # permanently erases the subaltern contradiction at the fixed point.
    assert T.join_t(I) == T, \
        "Violence of Certified Truth lemma failed: T ⊕_t I ≠ T"
    assert T.join_t(I).negation == F, \
        "Certified erasure consequence failed: ∼(T ⊕_t I) ≠ F"

    # Classifier paraconsistency: F ⊕_k T = I
    assert F.join_k(T) == I, \
        "Classifier paraconsistency lemma failed: F ⊕_k T ≠ I"

    # join_t is ≤_k-monotone in each argument (Knaster-Tarski prerequisite)
    for x in vals:
        for y in vals:
            for z in vals:
                if x.leq_k(y):
                    assert x.join_t(z).leq_k(y.join_t(z)), \
                        f"join_t not ≤_k-monotone: {x} ≤_k {y} but " \
                        f"({x}⊕_t{z})={x.join_t(z)} ≰_k ({y}⊕_t{z})={y.join_t(z)}"


_verify_bilattice_axioms()