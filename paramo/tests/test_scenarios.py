"""
Tests for paramo/knowledge_bases.ParamoKnowledgeBase and paramo/scenarios.
"""

import pytest
from chuaque.formulas import Atom, Negation, Conjunction, Implication, conj, neg, impl
from cubun.ddg import MoveType, MoveContext, Player
from paramo.knowledge_bases import a, h, ParamoKnowledgeBase
from paramo.scenarios import example_5_1, example_5_2


class TestParamoKnowledgeBase:

    def setup_method(self):
        self.kb = ParamoKnowledgeBase()

    # --- legal_attacks ---

    def test_attacks_conjunction_returns_both_conjuncts(self):
        moves = self.kb.legal_attacks(conj(a, neg(a)))
        formulas = [m[1] for m in moves]
        assert a in formulas
        assert neg(a) in formulas

    def test_attacks_conjunction_move_types_are_request(self):
        moves = self.kb.legal_attacks(conj(a, neg(a)))
        assert all(m[0] == MoveType.REQUEST for m in moves)

    def test_attacks_conjunction_structural_targets(self):
        moves = self.kb.legal_attacks(conj(a, neg(a)))
        targets = {m[2].structural_target for m in moves}
        assert targets == {"left_conjunct", "right_conjunct"}

    def test_attacks_implication_requests_antecedent(self):
        moves = self.kb.legal_attacks(impl(a, neg(a)))
        antecedent_attack = next(m for m in moves if m[2].structural_target == "antecedent")
        assert antecedent_attack[0] == MoveType.REQUEST
        assert antecedent_attack[1] == a

    def test_attacks_implication_asserts_negated_consequent(self):
        moves = self.kb.legal_attacks(impl(a, neg(a)))
        consequent_attack = next(m for m in moves if m[2].structural_target == "consequent")
        assert consequent_attack[0] == MoveType.ASSERT
        assert consequent_attack[1] == Negation(neg(a))  # ¬(¬a)

    def test_attacks_atom_a_returns_negation(self):
        moves = self.kb.legal_attacks(a)
        assert len(moves) == 1
        assert moves[0][0] == MoveType.ASSERT
        assert moves[0][1] == neg(a)

    def test_attacks_other_atom_returns_empty(self):
        assert self.kb.legal_attacks(h) == []

    def test_attacks_negation_returns_empty(self):
        assert self.kb.legal_attacks(neg(a)) == []

    # --- legal_defences ---

    def test_defences_no_structural_target_returns_empty(self):
        assert self.kb.legal_defences(conj(a, neg(a)), MoveContext()) == []

    def test_defences_conjunction_left_conjunct(self):
        moves = self.kb.legal_defences(
            conj(a, neg(a)), MoveContext(structural_target="left_conjunct")
        )
        assert len(moves) == 1
        assert moves[0][1] == a

    def test_defences_conjunction_right_conjunct(self):
        moves = self.kb.legal_defences(
            conj(a, neg(a)), MoveContext(structural_target="right_conjunct")
        )
        assert len(moves) == 1
        assert moves[0][1] == neg(a)

    def test_defences_implication_antecedent(self):
        thesis = impl(conj(a, neg(a)), neg(a))
        moves = self.kb.legal_defences(
            thesis, MoveContext(structural_target="antecedent")
        )
        assert len(moves) == 1
        assert moves[0][1] == conj(a, neg(a))

    def test_defences_implication_consequent(self):
        thesis = impl(conj(a, neg(a)), neg(a))
        moves = self.kb.legal_defences(
            thesis, MoveContext(structural_target="consequent")
        )
        assert len(moves) == 1
        assert moves[0][1] == neg(a)

    def test_defences_move_type_is_assert(self):
        moves = self.kb.legal_defences(
            conj(a, neg(a)), MoveContext(structural_target="left_conjunct")
        )
        assert moves[0][0] == MoveType.ASSERT


class TestExample51:
    """
    DDG on a ∧ ¬a.
    Paramuno board: Proponent wins — contradiction is viable.
    State board: Opponent wins — contradiction inadmissible.
    """

    def setup_method(self):
        self.paramuno, self.state = example_5_1()

    def test_paramuno_proponent_wins(self):
        assert self.paramuno.winner == Player.PROPONENT

    def test_state_opponent_wins(self):
        assert self.state.winner == Player.OPPONENT

    def test_paramuno_thesis_is_conjunction(self):
        assert self.paramuno.thesis == conj(a, neg(a))


class TestExample52:
    """
    DDG on (a ∧ ¬a) → ¬a.
    Paramuno board: Proponent wins — consequence is valid.
    State board: Opponent wins.
    """

    def setup_method(self):
        self.paramuno, self.state = example_5_2()

    def test_paramuno_proponent_wins(self):
        assert self.paramuno.winner == Player.PROPONENT

    def test_state_opponent_wins(self):
        assert self.state.winner == Player.OPPONENT

    def test_paramuno_thesis_is_implication(self):
        assert self.paramuno.thesis == impl(conj(a, neg(a)), neg(a))
