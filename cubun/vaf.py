"""
Value-based Argumentation Frameworks (Bench-Capon & Atkinson, 2009).

VAF = ⟨A, R, V, val, P⟩

An attack succeeds for a given audience only if the attacked
argument's value is not strictly preferred. This is how
motivational states enter the formalism.

Definition 4.1 of the paper.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from cubun.af import AF, Argument


@dataclass
class VAF:
    """
    VAF = ⟨A, R, V, val, P⟩

    V:   nonempty set of values (e.g., {y, w})
    val: A → V mapping arguments to values
    P:   set of total orders on V (audiences)
    """
    af: AF
    values: Set[str] = field(default_factory=set)
    val: Dict[str, str] = field(default_factory=dict)  # arg.name → value
    audiences: List[List[str]] = field(default_factory=list)

    def value_of(self, arg: Argument) -> str:
        return self.val[arg.name]

    def prefers(self, audience: List[str], v1: str, v2: str) -> bool:
        """Does audience strictly prefer v1 over v2?"""
        return audience.index(v1) < audience.index(v2)

    def defeats(self, attacker: Argument, target: Argument,
                audience: List[str]) -> bool:
        """
        Attack succeeds iff the target's value is NOT
        strictly preferred to the attacker's value.
        """
        if (attacker, target) not in self.af.attacks:
            return False
        target_val = self.value_of(target)
        attacker_val = self.value_of(attacker)
        # attack fails only if target's value is strictly preferred
        if self.prefers(audience, target_val, attacker_val):
            return False
        return True

    def audience_af(self, audience: List[str]) -> AF:
        """
        Reduce VAF to a plain AF for a given audience,
        keeping only the attacks that succeed.
        """
        reduced = AF(
            arguments=set(self.af.arguments),
            attacks={
                (a, t) for a, t in self.af.attacks
                if self.defeats(a, t, audience)
            },
        )
        return reduced

    def preferred_extension(self, audience: List[str]) -> list[Set[Argument]]:
        """Preferred extensions for a given audience."""
        return self.audience_af(audience).preferred_extensions()

    def grounded_extension(self, audience: List[str]) -> Set[Argument]:
        """Grounded extension for a given audience."""
        return self.audience_af(audience).grounded_extension()
