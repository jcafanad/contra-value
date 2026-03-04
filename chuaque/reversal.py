"""
The reversal operator * on formulas (syntactic De Morgan dualisation).

e.g., *(¬A ∧ A) = ¬¬A ∨ ¬A

This operator is constitutive of the d-model's capacity
to sustain contradiction without triviality. The dual
of an inconsistent situation is not its resolution
but its inversion.

NOTE ON INVOLUTION:
    reverse_formula is an involution (f** = f) on the {∧, ∨, ¬, atom} fragment.
    It is NOT an involution for formulas containing →:
        (A→B)* = A∧¬B;  (A∧¬B)* = A∨¬B ≠ A→B.
    The situation-level * in DModel IS enforced as an involution (__post_init__).
    These are different operators: formula reversal (syntactic) vs situation
    reversal (semantic). Do not conflate them.
"""

from chuaque.formulas import (
    Formula, Atom, Negation, Conjunction, Disjunction, Implication
)


def reverse_formula(f: Formula) -> Formula:
    """
    The * operator on formulas.

    (A ∧ B)* = A* ∨ B*
    (A ∨ B)* = A* ∧ B*
    (¬A)*    = ¬(A*)
    (A → B)* = A* ∧ ¬(B*)
    atom*    = atom
    """
    match f:
        case Atom(_):
            return f
        case Negation(inner):
            return Negation(reverse_formula(inner))
        case Conjunction(l, r):
            return Disjunction(reverse_formula(l), reverse_formula(r))
        case Disjunction(l, r):
            return Conjunction(reverse_formula(l), reverse_formula(r))
        case Implication(a, c):
            return Conjunction(
                reverse_formula(a),
                Negation(reverse_formula(c))
            )
