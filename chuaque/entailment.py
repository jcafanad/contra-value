"""
Relevance entailment over a d-model.

Implements conditions C1-C4 from Section 5.2 of
Contra Value, the Colombian Páramo via Symbolic AI.

These conditions guarantee that:
- Most standard results on truth are forthcoming
- Contradictory statements can be simultaneously true (dialetheia)
- In genuinely dialetheic models (where I(a,s) and I(¬a,s) are both
  designated), ¬(A∧¬A) holds at primary situations but is not a
  universal theorem — it fails at dual situations. In consistent models
  (where ¬a=F everywhere), ¬(A∧¬A) is a universal theorem.
- False statements are unprovable
- No formula evaluates to both TruthValue.T and TruthValue.F simultaneously
  (a structural guarantee of the bilattice enum, not a semantic claim)

Since capitalist social praxis is contradictory at its core,
a paraconsistent and dialetheic analysis is more adequate
if the consequences of these contradictions are to be fully grasped.
"""

from typing import List, Set

from chuaque.truth_values import TruthValue
from chuaque.d_model import DModel, Situation
from chuaque.formulas import (
    Formula, Atom, Negation, Conjunction, Disjunction, Implication
)


class Entailment:
    """
    Interpretation and entailment over a d-model.

    The interpretation function I: Formula × Situation → TruthValue
    is defined recursively by C1-C4.
    """

    def __init__(self, model: DModel):
        self.model = model

    def interpret(self, formula: Formula, situation: Situation) -> TruthValue:
        """
        Recursive interpretation of a formula at a situation.

        C1: I(p, A) = val(p, A)                              for atomic p
        C2: I(a ∧ b, C) = I(a, C) ⊓_t I(b, C)              (meet under ≤_t)
        C3: I(a → b, C) = T iff for all B,C' with R(C,B,C')
            and I(a, B) designated (T or I): I(b, C') = T.
            When I(a,B) is not designated (N or F): T vacuously.
            When I(a,B) is designated and I(b,C') ≠ T: F.
            NOTE: I→I = F under this implementation. See entailment.py.
        C4: I(¬a, A) = ∼I(a, A*)                            (Belnap negation at dual situation)
            NOTE: the condition I(¬a,A) = t iff I(a,A*) ≠ t is a 2-valued
            approximation that diverges from ∼ when I(a,A*) = I.
            The implementation uses ∼ (correct for the full bilattice).
        """
        match formula:
            case Atom(name):
                # C1
                return self.model.val(name, situation)

            case Conjunction(left, right):
                # C2: meet under ≤_t
                l = self.interpret(left, situation)
                r = self.interpret(right, situation)
                return TruthValue.meet(l, r)

            case Disjunction(left, right):
                # dual of C2: join under ≤_t
                l = self.interpret(left, situation)
                r = self.interpret(right, situation)
                return TruthValue.join(l, r)

            case Implication(antecedent, consequent):
                # C3
                return self._interpret_implication(
                    antecedent, consequent, situation
                )

            case Negation(inner):
                # C4: I(¬a, A) = ∼I(a, A*) — Belnap negation at the dual situation.
                # Negation is evaluated at the reversal A* of the current situation,
                # coupling the truth of ¬a at A to the truth of a at the dual world.
                return self.interpret(inner, self.model.dual(situation)).negation

    def _interpret_implication(
        self,
        antecedent: Formula,
        consequent: Formula,
        situation: Situation,
    ) -> TruthValue:
        """
        C3: I(a → b, C) = T if for every B,C' with R(C,B,C') where
        I(a, B) is designated (T or I): I(b, C') = T.
        Returns F when any such pair has I(b, C') ≠ T.
        Returns T vacuously when no accessible pair has a designated antecedent.

        T-ANTECEDENT CASES — T is designated, so the loop fires:
            T → T  =  T   (consequent T ✓)
            T → I  =  F   (consequent I ≠ T — implication fails)
            T → N  =  F   (consequent N ≠ T — implication fails)
            T → F  =  F   (consequent F ≠ T — implication fails)

        T-ANTECEDENT RATIONALE (confirmed by Juan):
            State arguments operate classically within their accessible worlds.
            T→I = F and T→N = F: when the State cannot guarantee a classically
            true conclusion, the implication definitively fails rather than
            inheriting the paraconsistent or unknown character of the conclusion.
            This encodes the State's epistemic position: within its accessible
            domain, it makes classical claims or fails.

        INACCESSIBLE WORLDS NOTE:
            The State's classical domain is defined by R-accessible worlds.
            C3 (implication) never checks the dual situation (s*), which is
            inaccessible from the State's primary situation in the standard
            model. However, C4 (negation) ALWAYS operates via the dual:
                I(¬a, s) = ∼I(a, s*)
            If I(a, state) = T and I(a, state*) = F, then C4 gives
            I(¬a, state) = T — both a and ¬a are T at state: a dialetheia
            generated at the State's own primary situation, through the
            inaccessible dual. The State's classical claims, via their
            negations, already carry paraconsistency into the primary world.
            C3 is blind to this; it is C4 that renders it visible.

            State claims that are themselves I-valued (e.g., simultaneous
            conservation mandate and extractive licensing, as in Colombia's
            DNP 2022) fall under the I-antecedent case: they cannot drive
            non-T conclusions — the same constraint as Paramuno contradictions.

        I-ANTECEDENT CASES — I is designated, so the loop fires:
            I → T  =  T   (antecedent designated, consequent T ✓)
            I → I  =  F   (antecedent designated, consequent I ≠ T)
            I → N  =  F   (antecedent designated, consequent N ≠ T)
            I → F  =  F   (antecedent designated, consequent F ≠ T)

        CONFIRMED SEMANTICS (resolved by Juan, v5):
            I → T  =  T   (antecedent designated, consequent T ✓)
            I → I  =  F   Lived experiences are ontologically unique.
                          Shared inferential infrastructure IS possible but must
                          be encoded explicitly in the accessibility relation R.
                          Without R encoding shared infrastructure: I→I = F.
                          When R DOES encode it: the loop finds T-consequents at
                          accessible worlds. Both cases are already in the model.
            I → N  =  F   Lived contradictions suggest the unknown but cannot
                          drive it. The epistemically absent (N) cannot be inferred
                          from I-valued premises. Stricter than material Belnap
                          (which gives T vacuously via ∼I ⊔_t N = T).
            I → F  =  F   Confirmed. No conclusion from lived contradiction.

        DEDUCTION THEOREM NOTE:
            entails([a=I], h=I) = True (both designated) but I(a→h) = F.
            This is a FEATURE, not a defect. entails() tests whether the
            conclusion is designated at all situations where all premises are
            designated — a model-theoretic property. interpret(a→h) tests the
            formula value at a situation — a syntactic property. These are
            genuinely different claims. The divergence formalises the
            irreducibility of lived experience: I-valued premises DO designate
            I-valued conclusions, but the IMPLICATION BETWEEN THEM is false
            because the inferential path is not automatic. The shared infra-
            structure must be explicit in R.
            See test_deduction_theorem_fails_for_I in TestImplication.

        VACUOUS TRUTH NOTE:
            When no accessible pair (B, C') has a designated antecedent
            (I(a, B) ∈ {N, F} for all such pairs), the implication evaluates
            to T. This is classical vacuous truth; in a genuinely paraconsistent
            framework, the epistemically neutral value would be N.
            The T is the algorithm's default for an unobserved conditional —
            the implication-level analogue of Dung's Bivalent Ghost
            (see cubun/af.py). Documented here rather than silently inherited.

        IMPLICATION-LEVEL BIVALENT GHOST:
            N→N = T and N→F = T are instances of the same colonial default
            that cubun/af.py names the Bivalent Ghost (no attacks → T for
            classical Dung arguments). Here: N-antecedent → any consequent = T.

            N→F = T is particularly consequential: from the epistemically
            absent (N — the zone where Muisca cosmologies currently reside,
            outside the Paramuno/State formal ontologies represented in the
            model), falsity vacuously follows. The framework says: if we do
            not know whether a claim holds, and the conclusion is false, the
            implication still holds.

            Material Belnap gives N→F = N (∼N ⊔_t F = N ⊔_t F = N): from
            the absent, the absent follows. At minimum, the epistemic absence
            of the antecedent would be inherited by the implication.

            The current T is a defensible relevance-logic choice (no designated
            antecedent ⟹ no constraint on the consequent). But the political
            naming is required: N→F = T is not neutral — it is the implication
            layer's Bivalent Ghost, and it acts on the same ontological zone
            as Dung's ghost acts on the argumentative layer.
            No code change. The naming is the critical-theoretical act.
        """
        for b in self.model.situations:
            ant_val = self.interpret(antecedent, b)
            if ant_val.is_designated():
                for c in self.model.situations:
                    if self.model.accessible(situation, b, c):
                        con_val = self.interpret(consequent, c)
                        if con_val != TruthValue.T:
                            return TruthValue.F
        return TruthValue.T

    def entails(self, premises: List[Formula], conclusion: Formula) -> bool:
        """
        premises entail conclusion iff in every situation where
        all premises are designated (t or i), the conclusion
        is also designated.
        """
        for situation in self.model.situations:
            all_premises_hold = all(
                self.interpret(p, situation).is_designated()
                for p in premises
            )
            if all_premises_hold:
                conclusion_val = self.interpret(conclusion, situation)
                if not conclusion_val.is_designated():
                    return False
        return True

    def is_theorem(self, formula: Formula) -> bool:
        """
        A formula is a theorem iff it is designated
        in every situation of the d-model.
        """
        return all(
            self.interpret(formula, s).is_designated()
            for s in self.model.situations
        )

    def is_valid(self, formula: Formula) -> bool:
        """
        A formula is valid iff it is t (not merely designated)
        in every situation.
        """
        return all(
            self.interpret(formula, s) == TruthValue.T
            for s in self.model.situations
        )
