"""
Knowledge bases from Section 4 of the paper.

KB0 := {a, a → ¬h, a → y, h → w, r, r → ¬a, r → ¬y, r → h, y}
KB1 := KB0 ∪ {s → y, h → s}

Variables:
    a: increased agricultural land-use
    h: improved hydrological regulating services
    r: greater number of peatland restoration activities
    s: increased water supply
    y: greater agricultural yield
    w: improved living conditions
"""

from chuaque.formulas import Atom, Implication, Negation, Formula, atoms
from typing import Set


# the sentential variables of the Páramo
a, h, r, s, y, w = atoms("a", "h", "r", "s", "y", "w")


def kb0() -> Set[Formula]:
    """
    The base knowledge base.

    Encodes the predicament: agriculture yields income
    but degrades hydrology; restoration opposes agriculture
    but restores hydrology.
    """
    return {
        a,                                  # agriculture happens
        Implication(a, Negation(h)),        # a → ¬h
        Implication(a, y),                  # a → y
        Implication(h, w),                  # h → w
        r,                                  # restoration happens
        Implication(r, Negation(a)),        # r → ¬a
        Implication(r, Negation(y)),        # r → ¬y
        Implication(r, h),                  # r → h
        y,                                  # yield exists
    }


def kb1() -> Set[Formula]:
    """
    Extended knowledge base.

    Adds the recognition that water supply mediates between
    hydrological services and agricultural yield.
    This is what generates argument A5 and reconciles
    the two audiences.
    """
    return kb0() | {
        Implication(s, y),                  # s → y
        Implication(h, s),                  # h → s
    }
