"""
The sentential language on which the d-model operates.

Well-formed formulae built from sentential variables using
→ (implies), ∧ (and), ∨ (or), ¬ (not).

Every argument in the VAF/DDG is formed by combining
these sentential elements.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class Atom:
    """Sentential variable: a, h, r, s, y, w, etc."""
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Negation:
    inner: Formula

    def __repr__(self) -> str:
        return f"¬{self.inner}"


@dataclass(frozen=True)
class Conjunction:
    left: Formula
    right: Formula

    def __repr__(self) -> str:
        return f"({self.left} ∧ {self.right})"


@dataclass(frozen=True)
class Disjunction:
    left: Formula
    right: Formula

    def __repr__(self) -> str:
        return f"({self.left} ∨ {self.right})"


@dataclass(frozen=True)
class Implication:
    antecedent: Formula
    consequent: Formula

    def __repr__(self) -> str:
        return f"({self.antecedent} → {self.consequent})"


Formula = Union[Atom, Negation, Conjunction, Disjunction, Implication]


# convenience constructors
def atom(name: str) -> Atom:
    return Atom(name)

def neg(f: Formula) -> Negation:
    return Negation(f)

def conj(l: Formula, r: Formula) -> Conjunction:
    return Conjunction(l, r)

def disj(l: Formula, r: Formula) -> Disjunction:
    return Disjunction(l, r)

def impl(a: Formula, c: Formula) -> Implication:
    return Implication(a, c)


def atoms(*names: str) -> tuple[Atom, ...]:
    """Create multiple atoms. e.g., a, h, r, s, y, w = atoms('a','h','r','s','y','w')"""
    return tuple(Atom(n) for n in names)


def subformulas(f: Formula) -> set[Formula]:
    """All subformulas of f, including f itself."""
    result = {f}
    match f:
        case Atom(_):
            pass
        case Negation(inner):
            result |= subformulas(inner)
        case Conjunction(l, r) | Disjunction(l, r):
            result |= subformulas(l) | subformulas(r)
        case Implication(a, c):
            result |= subformulas(a) | subformulas(c)
    return result
