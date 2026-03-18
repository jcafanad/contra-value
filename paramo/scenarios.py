"""
Executable reproductions of the paper's examples.

Example 4.1: Classical VAF over KB0 and KB1
Example 5.1: DDG on A6 ⟺ {a ∧ ¬a}
Example 5.2: DDG on A7 ⟺ {(a ∧ ¬a) → ¬a}
"""

from chuaque.formulas import (
    Atom, Negation, Conjunction, Implication, conj, neg, impl, atoms
)
from cubun.af import AF, Argument, SituatedArgument, SituatedAudience
from cubun.vaf import VAF
from cubun.ddg import DDGEngine, GameState, Player

from paramo.knowledge_bases import a, h, r, s, y, w, ParamoKnowledgeBase


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


def _paramo_boards():
    """Shared audience pair for examples 5.1 and 5.2."""
    paramuno = SituatedAudience(
        name="Paramuno",
        preferences={"subsistence": 10, "conservation": 10},
        tolerances={("subsistence", "conservation")},
    )
    state = SituatedAudience(
        name="State_Authority",
        preferences={"subsistence": 3, "conservation": 10},
        intolerances={("subsistence", "conservation")},
    )
    return paramuno, state


def _paramo_argument_map():
    """Shared argument map: a → subsistence, ¬a → conservation."""
    return {
        a:      SituatedArgument("A_subsistence", a,      "subsistence"),
        neg(a): SituatedArgument("A_conservation", neg(a), "conservation"),
    }


def example_5_1() -> tuple:
    """
    DDG on A6 ⟺ {a ∧ ¬a}.

    Returns (paramuno_state, state_state): GameState for each board.

    Paramuno board: Proponent wins — the contradiction is a viable
    statement. The dialetheic chain closes because the Paramuno
    audience tolerates ("subsistence", "conservation").

    State board: Opponent wins — the dialetheic chain cannot close
    because the State audience marks the same pair as intolerable.
    Proponent is blocked before producing evidence, not silenced after.
    """
    thesis = conj(a, neg(a))
    argument_map = _paramo_argument_map()
    paramuno_audience, state_audience = _paramo_boards()
    engine = DDGEngine(ParamoKnowledgeBase())
    return (
        engine.solve_thesis(thesis, paramuno_audience, argument_map),
        engine.solve_thesis(thesis, state_audience,    argument_map),
    )


def example_5_2() -> tuple:
    """
    DDG on A7 ⟺ {(a ∧ ¬a) → ¬a}.

    Returns (paramuno_state, state_state): GameState for each board.

    Paramuno board: Proponent wins — refraining from agriculture is a
    valid non-trivial consequence of the contradiction. The Opponent's
    antecedent challenge recurses into example_5_1's subgame; the
    same dialetheic chain closes it.

    State board: Opponent wins by the same mechanism as example_5_1.
    """
    thesis = impl(conj(a, neg(a)), neg(a))
    argument_map = _paramo_argument_map()
    paramuno_audience, state_audience = _paramo_boards()
    engine = DDGEngine(ParamoKnowledgeBase())
    return (
        engine.solve_thesis(thesis, paramuno_audience, argument_map),
        engine.solve_thesis(thesis, state_audience,    argument_map),
    )


def _print_ddg_result(label: str, state: "GameState") -> None:
    print(f"  {label}: winner = {state.winner}")
    for m in state.moves:
        closed = " [CLOSED]" if m.position in state.defended_positions else ""
        ev = f" evidence={m.tolerance_evidence}" if m.tolerance_evidence else ""
        print(f"    [{m.position}]{closed} {m.player.value} {m.move_type.value}"
              f"  {m.formula}{ev}")


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
    p_state, s_state = example_5_1()
    _print_ddg_result("Paramuno board", p_state)
    _print_ddg_result("State board   ", s_state)

    print()
    print("=== Example 5.2: DDG on (a ∧ ¬a) → ¬a ===")
    p_state, s_state = example_5_2()
    _print_ddg_result("Paramuno board", p_state)
    _print_ddg_result("State board   ", s_state)
