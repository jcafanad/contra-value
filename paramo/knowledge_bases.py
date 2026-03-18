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

from chuaque.formulas import Atom, Implication, Negation, Conjunction, Formula, atoms
from cubun.ddg import MoveType, MoveContext
from typing import List, Set, Tuple


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


class ParamoKnowledgeBase:
    """
    DDG knowledge base for the Guantiva-La Rusia Páramo.

    Implements the two game-rule methods required by DDGEngine:

        legal_attacks(formula)          → admissible attack moves
        legal_defences(formula, context) → admissible defence moves

    "Legal" in the sense of Hamblin (1970, Fallacies, Methuen) and
    Prakken (2006, "Formal systems for persuasion dialogue",
    Knowledge Engineering Review 21:2, 163–188): a move is legal if
    the game rules admit it at the current position. Not every
    syntactically expressible move is legal — structural attacks on an
    implication are inapplicable to a conjunction, and so on.

    Structural rules (Conjunction, Implication) are formula-generic.
    The single domain rule (Atom "a") encodes the KB0 counter-claim
    r → ¬a: the State can always assert ¬a to challenge agricultural
    land-use. This is the minimal domain knowledge required to run
    examples 5.1 and 5.2.
    """

    def legal_attacks(
        self, formula: Formula
    ) -> List[Tuple[MoveType, Formula, MoveContext]]:
        if isinstance(formula, Conjunction):
            return [
                (MoveType.REQUEST, formula.left,
                 MoveContext(structural_target="left_conjunct")),
                (MoveType.REQUEST, formula.right,
                 MoveContext(structural_target="right_conjunct")),
            ]
        if isinstance(formula, Implication):
            return [
                (MoveType.REQUEST, formula.antecedent,
                 MoveContext(structural_target="antecedent")),
                (MoveType.ASSERT, Negation(formula.consequent),
                 MoveContext(structural_target="consequent")),
            ]
        if isinstance(formula, Atom) and formula == a:
            # KB0 domain counter-claim: r → ¬a entails ¬a given r.
            return [(MoveType.ASSERT, Negation(a), MoveContext())]
        return []

    def legal_defences(
        self, formula: Formula, context: MoveContext
    ) -> List[Tuple[MoveType, Formula]]:
        target = context.structural_target
        if target is None:
            return []
        if isinstance(formula, Conjunction):
            if target == "left_conjunct":
                return [(MoveType.ASSERT, formula.left)]
            if target == "right_conjunct":
                return [(MoveType.ASSERT, formula.right)]
        if isinstance(formula, Implication):
            if target == "antecedent":
                return [(MoveType.ASSERT, formula.antecedent)]
            if target == "consequent":
                return [(MoveType.ASSERT, formula.consequent)]
        return []
