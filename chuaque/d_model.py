"""
The d-model of entailment.

M = (W, R, *, val) where:

W:   nonempty set of situations (possible worlds)
R:   ternary accessibility relation on W
     R(a, b, c) means: from a, b precedes c
     (a cyclic relation, not a classical partial order)
*:   reversal operator taking a situation to its dual
val: valuation function assigning truth values to atoms at situations

The cyclicity of R is decisive. It prevents the construction
of a choice function satisfying the axioms of economically
rational preference formation (Sen, 1993). The orders generated
through DDG are concrete objective choices within inconsistent
domains, not preferences describing mental states complying
with classically logical rational behaviour.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, Optional

from chuaque.truth_values import TruthValue


@dataclass(frozen=True)
class Situation:
    """A possible world/situation in the d-model."""
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass
class DModel:
    """
    A d-model M = (W, R, *, val).

    The d-model can be seen as a VAF with additional structure:
    VAF's orders define the d-model's semilattice, whose set of
    worlds can be identified with the former's set of motivational
    states. (Afanador, 2020)
    """
    situations: Set[Situation]
    reversal: Dict[Situation, Situation]
    valuation: Dict[Tuple[str, Situation], TruthValue]
    _accessibility: Set[Tuple[Situation, Situation, Situation]] = field(
        default_factory=set
    )

    def __post_init__(self):
        # validate: reversal is an involution
        for s, s_star in self.reversal.items():
            assert s_star in self.situations, f"{s_star} not in situations"
            assert self.reversal.get(s_star) == s, f"* is not an involution at {s}"

    def accessible(self, a: Situation, b: Situation, c: Situation) -> bool:
        """R(a, b, c): from a, b precedes c."""
        return (a, b, c) in self._accessibility

    def add_accessibility(self, a: Situation, b: Situation, c: Situation):
        """Assert that R(a, b, c) holds."""
        self._accessibility.add((a, b, c))

    def val(self, atom_name: str, situation: Situation) -> TruthValue:
        """Valuation of an atom at a situation."""
        key = (atom_name, situation)
        if key not in self.valuation:
            return TruthValue.N  # unspecified atoms are neither true nor false
        return self.valuation[key]

    def dual(self, situation: Situation) -> Situation:
        """The reversal * applied to a situation."""
        return self.reversal[situation]

    @staticmethod
    def simple(atoms: dict[str, dict[str, TruthValue]]) -> DModel:
        """
        Convenience constructor for a d-model with a single pair
        of dual situations (base, base*).

        atoms: {atom_name: {"base": TruthValue, "base*": TruthValue}}
        """
        base = Situation("base")
        dual = Situation("base*")
        valuation = {}
        for atom_name, vals in atoms.items():
            if "base" in vals:
                valuation[(atom_name, base)] = vals["base"]
            if "base*" in vals:
                valuation[(atom_name, dual)] = vals["base*"]
        return DModel(
            situations={base, dual},
            reversal={base: dual, dual: base},
            valuation=valuation,
        )
