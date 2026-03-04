"""
Executable reproductions of the paper's examples.

Example 4.1: Classical VAF over KB0 and KB1
Example 5.1: DDG on A6 ⟺ {a ∧ ¬a}
Example 5.2: DDG on A7 ⟺ {(a ∧ ¬a) → ¬a}
"""

from chuaque.formulas import (
    Atom, Negation, Conjunction, Implication, conj, neg, impl, atoms
)
from cubun.af import AF, Argument
from cubun.vaf import VAF
from cubun.ddg import DDGEngine, Player

from paramo.knowledge_bases import a, h, r, s, y, w


def example_4_1_kb0() -> VAF:
    """
    Build VAFKB0 from the paper.

    Arguments:
    A1 := ⟨{a, a → y}, y⟩
    A2 := ⟨{r, r → ¬a}, ¬(a ∧ (a → y))⟩
    A3 := ⟨{y, y → ¬r}, ¬(r ∧ (r → ¬y))⟩
    A4 := ⟨{(r → h) → (h → ¬a)}, ¬(a ∧ (a → y))⟩
    """
    diamond_y = neg(conj(a, impl(a, y)))
    diamond_w = neg(conj(r, impl(r, neg(a))))

    A1 = Argument("A1", frozenset({a, impl(a, y)}), y)
    A2 = Argument("A2", frozenset({r, impl(r, neg(a))}), diamond_y)
    A3 = Argument("A3", frozenset({y, impl(y, neg(r))}), diamond_w)
    A4 = Argument("A4", frozenset({impl(impl(r, h), impl(h, neg(a)))}), diamond_y)

    af = AF(
        arguments={A1, A2, A3, A4},
        attacks={(A1, A2), (A1, A4), (A2, A3)},
    )

    vaf = VAF(
        af=af,
        values={"y", "w"},
        val={"A1": "y", "A2": "w", "A3": "y", "A4": "w"},
        audiences=[["y", "w"], ["w", "y"]],  # audience-y and audience-w
    )

    return vaf


def example_5_1() -> dict:
    """
    DDG on A6 ⟺ {a ∧ ¬a}.
    Paramuno wins.
    The contradiction is a viable statement.

    TODO: requires a paramo KnowledgeBase implementation providing
    legal_attacks() and legal_defences() for DDGEngine.solve_thesis().
    """
    raise NotImplementedError(
        "example_5_1 requires a paramo KnowledgeBase implementation. "
        "Use DDGEngine(knowledge_base).solve_thesis(thesis, audience, argument_map)."
    )


def example_5_2() -> dict:
    """
    DDG on A7 ⟺ {(a ∧ ¬a) → ¬a}.
    Paramuno wins.
    Refraining from agriculture is a valid non-trivial
    consequence of the contradiction.

    TODO: requires a paramo KnowledgeBase implementation providing
    legal_attacks() and legal_defences() for DDGEngine.solve_thesis().
    """
    raise NotImplementedError(
        "example_5_2 requires a paramo KnowledgeBase implementation. "
        "Use DDGEngine(knowledge_base).solve_thesis(thesis, audience, argument_map)."
    )


if __name__ == "__main__":
    print("=== Example 4.1: VAF over KB0 ===")
    vaf = example_4_1_kb0()
    for audience_name, audience in zip(["audience-y", "audience-w"], vaf.audiences):
        ext = vaf.preferred_extension(audience)
        print(f"  {audience_name}: {ext}")
        grd = vaf.grounded_extension(audience)
        print(f"  {audience_name} (grounded): {grd}")

    print()
    print("=== Example 5.1: DDG on a ∧ ¬a ===")
    print("  [Not yet implemented — requires paramo KnowledgeBase]")

    print()
    print("=== Example 5.2: DDG on (a ∧ ¬a) → ¬a ===")
    print("  [Not yet implemented — requires paramo KnowledgeBase]")
