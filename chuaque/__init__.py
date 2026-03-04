"""
chuaque — the pre-differentiated condition containing all potential.

The 4-valued paraconsistent logic where contradiction
is an operative condition of the real, not a failure of reason.
"""

from chuaque.truth_values import TruthValue
from chuaque.formulas import (
    Formula, Atom, Negation, Conjunction, Disjunction, Implication,
    atom, neg, conj, disj, impl, atoms, subformulas,
)
from chuaque.d_model import DModel, Situation
from chuaque.entailment import Entailment
from chuaque.reversal import reverse_formula

__all__ = [
    "TruthValue", "Formula", "Atom", "Negation", "Conjunction",
    "Disjunction", "Implication", "atom", "neg", "conj", "disj",
    "impl", "atoms", "subformulas", "DModel", "Situation",
    "Entailment", "reverse_formula",
]