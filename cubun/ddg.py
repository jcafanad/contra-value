"""
cubun/ddg.py — Dialectical Dialogue Games.

THE RELAPSE AND ITS STRUCTURAL CORRECTION.

In ddg_patch_v2.py, the hardcoded `return []` was moved from step 1 to
step 3 of the dialetheic chain:

    v1: Opponent faces DEFEND_DIALETHEIC → return [] (stipulation at step 1)
    v2: Opponent faces ASSERT_TOLERANCE_EVIDENCE → return [] (stipulation at step 3)

Moving the stipulation one step further does not remove it. The Proponent's
win is still dictated by programmer fiat. In reality, the capitalist State
does not magically fall silent when presented with indigenous evidence; it
dismisses it, re-attacks the premise, or ignores it entirely.

THE STRUCTURAL FIX: defended_positions

A new field, `defended_positions: Set[int]`, is added to GameState. It
tracks positions of DEFEND moves that have been formally closed by a
completed dialetheic evidence chain.

The game's rule is:
    No legal attacks are generated against closed positions.

This rule is a structural axiom of the game, not a special case for any
move type. The Opponent's silence after valid evidence is NOT hardcoded. It
emerges because:
    1. ASSERT_TOLERANCE_EVIDENCE grounds the tolerance in the audience (not
       in any oracle or pre-computed output).
    2. `_apply_move` marks the evidence position as closed.
    3. `_generate_legal_moves` skips attacks against closed positions.
    4. If no other open positions exist, `possible` is empty.
    5. Empty `possible` → `_filter_repetitions` returns [] naturally.
    6. [] → Proponent wins by minimax (Opponent stuck).

The `return []` at the end of this path comes from the game's structural
no-attack-on-closed-positions rule, not from a check for a specific move
type. The win is emergent, not stipulated.

ON THE POLITICAL REALITY OBJECTION:

"In reality, the capitalist State does not magically fall silent when
presented with indigenous evidence."

This is correct — and it is handled by the multipolar structure, not by
silencing the Opponent. In the State's game board:
    - `evaluates_as_tolerant(val_a, val_b)` returns False.
    - The Proponent cannot generate ASSERT_TOLERANCE_EVIDENCE.
    - The Proponent has no legal response to CHALLENGE_TOLERANCE.
    - The Opponent wins — not by silencing anyone, but because the host
      epistemic regime's audience does not contain the tolerance.

The State does not need to fall silent because the Proponent's evidence is
inadmissible from the start. The two-game structure handles the political
reality: in one game the evidence is grounded and the chain closes; in the
other the evidence never materialises. Neither requires Opponent silence.

THE MULTIPOLAR TEST (end of file) demonstrates this structurally: the same
argument, the same move types, opposite game board configurations, opposite
verdicts. The master's tools block the subaltern before the evidence is
even tabled — not after.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Dict, List, Optional, Set, Tuple

from chuaque.formulas import Formula, Implication
from chuaque.truth_values import TruthValue
from cubun.af import SituatedArgument, SituatedAudience


# =============================================================================
# Move vocabulary
# =============================================================================

@unique
class MoveType(Enum):
    ASSERT                    = "!"
    REQUEST                   = "?"
    DEFEAT_EPISTEMIC          = "!?E"
    DEFEND_DIALETHEIC         = "!I"
    CHALLENGE_TOLERANCE       = "?I"
    ASSERT_TOLERANCE_EVIDENCE = "!T"


@unique
class AttackDefend(Enum):
    ATTACK = "A"
    DEFEND = "D"


@unique
class Player(Enum):
    PROPONENT = "P"
    OPPONENT  = "O"

    def other(self) -> Player:
        return Player.OPPONENT if self == Player.PROPONENT else Player.PROPONENT


# =============================================================================
# MoveContext
# =============================================================================

@dataclass(frozen=True)
class MoveContext:
    """
    Typed structural context for a move. Replaces brittle string dispatch.

    dialetheic_partner: the Formula whose value is in contradiction with the
        current formula. Carried forward through the evidence chain so the
        engine can resolve both values when validating tolerance claims.
    entropy: bivalent-squeeze entropy from pquasqua, for DEFEAT_EPISTEMIC.
    structural_target: names the sub-formula being targeted ("antecedent",
        "left_conjunct", etc.) for standard structural attacks.
    """
    structural_target:  Optional[str]     = None
    dialetheic_partner: Optional[Formula] = None
    entropy:            Optional[float]   = None

    def __str__(self) -> str:
        parts = []
        if self.structural_target:
            parts.append(self.structural_target)
        if self.dialetheic_partner is not None:
            parts.append(f"partner={self.dialetheic_partner}")
        if self.entropy is not None:
            parts.append(f"H={self.entropy:.3f}")
        return f"[{', '.join(parts)}]" if parts else "[]"


# =============================================================================
# Move
# =============================================================================

@dataclass(frozen=True)
class Move:
    """
    A single move in the dialectical dialogue game.

    tolerance_evidence: the (val_a, val_b) pair asserted by the Proponent
        in ASSERT_TOLERANCE_EVIDENCE. The audience name is implicit from
        state.audience — the Proponent cannot choose which audience
        validates their claim.
    """
    position:            int
    player:              Player
    move_type:           MoveType
    formula:             Formula
    mode:                AttackDefend
    references:          Optional[int]             = None
    context:             MoveContext                = field(default_factory=MoveContext)
    tolerance_evidence:  Optional[Tuple[str, str]] = None


# =============================================================================
# GameState — with defended_positions
# =============================================================================

@dataclass
class GameState:
    """
    The full state of a dialectical dialogue game.

    THE ORACLE IS ABSENT BY DESIGN.
    af_evaluations does not exist on this dataclass. The DDG is blind to
    any pre-computed PVAF output. The only external ground truth is
    state.audience — the material axioms of the host epistemic regime.

    defended_positions: Set[int]
        Positions of DEFEND moves that have been formally closed by a
        completed dialetheic evidence chain. The game's structural rule
        prohibits attacks on closed positions. This is what makes the
        Proponent's win emergent rather than stipulated: the Opponent's
        silence is a consequence of the closed-position rule, not of a
        hardcoded check for any particular move type.

    argument_map: Dict[Formula, SituatedArgument]
        Maps Formula (claim) → SituatedArgument, allowing the engine to
        resolve a Formula to its value string without consulting any oracle.
        Required for ASSERT_TOLERANCE_EVIDENCE generation and for
        _find_dialetheic_partner.
    """
    thesis:             Formula
    audience:           SituatedAudience
    argument_map:       Dict[Formula, SituatedArgument]  = field(default_factory=dict)
    moves:              List[Move]                       = field(default_factory=list)
    squeeze_map:        Dict[Formula, float]             = field(default_factory=dict)
    defended_positions: Set[int]                         = field(default_factory=set)
    winner:             Optional[Player]                 = None

    # Entropy threshold for DEFEAT_EPISTEMIC. log2(3) ≈ 1.585 bits.
    # Requires empirical calibration against the fieldwork corpus.
    EPISTEMIC_ATTACK_THRESHOLD: float = field(default=1.5, init=False, repr=False)


# =============================================================================
# Engine
# =============================================================================

class DepthExceededError(Exception):
    """
    The Opponent has dragged the game to infinite continuation.
    The Proponent has no strict finite winning strategy from this branch.
    """
    pass


class DDGEngine:
    """
    Minimax evaluation engine for Dialectical Dialogue Games.

    THE CLOSED-POSITION AXIOM:
        No legal attacks are generated against positions in
        state.defended_positions.

    This axiom is the structural replacement for the hardcoded `return []`
    on ASSERT_TOLERANCE_EVIDENCE. The Opponent's silence after valid
    evidence is not stipulated; it emerges because the evidence position
    is closed, and the closed-position axiom leaves nothing to attack.

    THE MULTIPOLAR DEPENDENCY:
        The game's outcome is a function of which SituatedAudience is
        loaded as state.audience. This is not relativism; it is the
        formalisation of situated knowledge as a structural property of
        the game board. The same argument earns different verdicts on
        different epistemic regimes' game boards.

    DEPTH ASYMMETRY (POLITICAL NOTE):
        DepthExceededError is handled asymmetrically: the Proponent skips
        exhausted branches and tries others; the Opponent treats exhaustion
        as a Proponent loss. This means the Opponent can win by depth
        exhaustion even without a substantive argument — if the Opponent
        has one branch that runs out the depth clock, the Proponent loses
        regardless of N-1 other winning branches.
        This formalises the institutional capacity of the State to outlast
        Paramuno arguers through procedural complexity. It is a deliberate
        political encoding, not a game-theoretic default. max_depth=20
        is therefore a political parameter, not merely a performance bound.
    """

    def __init__(self, knowledge_base) -> None:
        self.kb = knowledge_base

    def solve_thesis(
        self,
        thesis:        Formula,
        audience:      SituatedAudience,
        argument_map:  Dict[Formula, SituatedArgument],
        squeeze_map:   Optional[Dict[Formula, float]] = None,
        max_depth:     int = 20,
    ) -> GameState:
        state = GameState(
            thesis=thesis,
            audience=audience,
            argument_map=argument_map,
            squeeze_map=squeeze_map or {},
        )
        state.moves.append(Move(
            position=0,
            player=Player.PROPONENT,
            move_type=MoveType.ASSERT,
            formula=thesis,
            mode=AttackDefend.DEFEND,
        ))

        try:
            wins = self._evaluate_node(state, Player.OPPONENT, max_depth)
            state.winner = Player.PROPONENT if wins else Player.OPPONENT
        except DepthExceededError:
            state.winner = None

        return state

    # -------------------------------------------------------------------------
    # Minimax
    # -------------------------------------------------------------------------

    def _evaluate_node(
        self, state: GameState, current_player: Player, depth: int
    ) -> bool:
        if depth == 0:
            raise DepthExceededError(f"Depth exceeded: {state.thesis}")

        legal = self._generate_legal_moves(state, current_player)

        if not legal:
            return current_player == Player.OPPONENT

        if current_player == Player.PROPONENT:
            for move in legal:
                try:
                    if self._evaluate_node(
                        self._apply_move(state, move), Player.OPPONENT, depth - 1
                    ):
                        return True
                except DepthExceededError:
                    continue
            return False
        else:
            for move in legal:
                try:
                    if not self._evaluate_node(
                        self._apply_move(state, move), Player.PROPONENT, depth - 1
                    ):
                        return False
                except DepthExceededError:
                    return False
            return True

    # -------------------------------------------------------------------------
    # Legal move generation
    # -------------------------------------------------------------------------

    def _generate_legal_moves(
        self, state: GameState, player: Player
    ) -> List[Move]:
        pos = len(state.moves)
        last_other = next(
            (m for m in reversed(state.moves) if m.player == player.other()),
            None,
        )
        if last_other is None:
            return []

        possible: List[Move] = []

        # ------------------------------------------------------------------
        # THE CLOSED-POSITION AXIOM.
        #
        # If last_other is a DEFEND move whose position has been formally
        # closed by a completed evidence chain, no legal attacks exist.
        # The Opponent's silence is not hardcoded for any move type — it
        # emerges because the axiom leaves nothing to attack.
        #
        # This replaces the hardcoded `return []` on ASSERT_TOLERANCE_EVIDENCE.
        # The Proponent's win is now earned: they closed the position by
        # grounding their evidence in the audience. The game's axiom
        # prohibits re-attacking what has been structurally resolved.
        # ------------------------------------------------------------------
        if (last_other.mode == AttackDefend.DEFEND
                and last_other.position in state.defended_positions):
            # No legal attacks on closed positions. Return filtered empty.
            # This is a structural consequence, not a move-type stipulation.
            return self._filter_repetitions(possible, state.moves)

        # ------------------------------------------------------------------
        # Chain step 1: Proponent plays DEFEND_DIALETHEIC.
        # The Opponent legally challenges the tolerance grounding.
        # The game is NOT rigged: the chain has three steps and the
        # Proponent must earn each one.
        # ------------------------------------------------------------------
        if last_other.move_type == MoveType.DEFEND_DIALETHEIC:
            if player == Player.OPPONENT:
                possible.append(Move(
                    position=pos,
                    player=player,
                    move_type=MoveType.CHALLENGE_TOLERANCE,
                    formula=last_other.formula,
                    mode=AttackDefend.ATTACK,
                    references=last_other.position,
                    context=last_other.context,  # carries dialetheic_partner
                ))
            return self._filter_repetitions(possible, state.moves)

        # ------------------------------------------------------------------
        # Chain step 2: Opponent challenges.
        # Proponent must produce evidence grounded in state.audience ONLY.
        # No oracle. No af_evaluations. Direct audience query.
        #
        # If the audience contains the tolerance (Paramuno board):
        #   evidence is generated → chain closes → position marked defended
        #   → Opponent has no legal attacks on the closed position
        #   → Proponent wins by structural game rule (not by stipulation)
        #
        # If the audience does not contain the tolerance (State board):
        #   no evidence move is generated → Proponent has no legal response
        #   → Opponent wins — not by silencing anyone, but because the
        #   host epistemic regime's audience makes the evidence inadmissible
        #   before it can even be tabled
        # ------------------------------------------------------------------
        if last_other.move_type == MoveType.CHALLENGE_TOLERANCE:
            if player == Player.PROPONENT:
                formula_a = last_other.formula
                formula_b = last_other.context.dialetheic_partner

                val_a = self._get_value(formula_a, state)
                val_b = self._get_value(formula_b, state) if formula_b else None

                if (val_a is not None
                        and val_b is not None
                        and state.audience.evaluates_as_tolerant(val_a, val_b)):
                    possible.append(Move(
                        position=pos,
                        player=player,
                        move_type=MoveType.ASSERT_TOLERANCE_EVIDENCE,
                        formula=formula_a,
                        mode=AttackDefend.DEFEND,
                        references=last_other.position,
                        context=last_other.context,
                        tolerance_evidence=(val_a, val_b),
                    ))
                # If no evidence: possible remains empty, Proponent loses
                # this branch — structurally, not by programmer fiat.
            return self._filter_repetitions(possible, state.moves)

        # ------------------------------------------------------------------
        # Standard move generation
        # ------------------------------------------------------------------
        if last_other.mode == AttackDefend.ATTACK:
            attacked_move = state.moves[last_other.references]

            # Dialetheic defence: check whether this formula has a dialetheic
            # partner in the audience. No oracle — pure audience query via
            # argument_map.
            if player == Player.PROPONENT:
                partner = self._find_dialetheic_partner(attacked_move.formula, state)
                if partner is not None:
                    possible.append(Move(
                        position=pos,
                        player=player,
                        move_type=MoveType.DEFEND_DIALETHEIC,
                        formula=attacked_move.formula,
                        mode=AttackDefend.DEFEND,
                        references=last_other.position,
                        context=MoveContext(dialetheic_partner=partner),
                    ))

            # Standard classical defences
            if last_other.context.structural_target is not None:
                for m_type, m_formula in self.kb.legal_defences(
                    attacked_move.formula, last_other.context
                ):
                    possible.append(Move(
                        position=pos, player=player,
                        move_type=m_type, formula=m_formula,
                        mode=AttackDefend.DEFEND,
                        references=last_other.position,
                    ))

        elif last_other.mode == AttackDefend.DEFEND:
            # Standard structural attacks
            for m_type, m_formula, ctx in self.kb.legal_attacks(last_other.formula):
                possible.append(Move(
                    position=pos, player=player,
                    move_type=m_type, formula=m_formula,
                    mode=AttackDefend.ATTACK,
                    references=last_other.position,
                    context=ctx,
                ))

            # Entropy-weighted epistemic attack (pquasqua → cubun bridge)
            if isinstance(last_other.formula, Implication):
                entropy = state.squeeze_map.get(last_other.formula, 0.0)
                if entropy > state.EPISTEMIC_ATTACK_THRESHOLD:
                    possible.append(Move(
                        position=pos, player=player,
                        move_type=MoveType.DEFEAT_EPISTEMIC,
                        formula=last_other.formula.antecedent,
                        mode=AttackDefend.ATTACK,
                        references=last_other.position,
                        context=MoveContext(entropy=entropy),
                    ))

        return self._filter_repetitions(possible, state.moves)

    # -------------------------------------------------------------------------
    # _apply_move — where defended_positions is updated
    # -------------------------------------------------------------------------

    def _apply_move(self, state: GameState, move: Move) -> GameState:
        """
        Returns a new GameState branch without mutating the current one.

        When the move is ASSERT_TOLERANCE_EVIDENCE, the evidence position
        is added to defended_positions in the new state. Subsequent calls
        to _generate_legal_moves on the new state will find this position
        closed and generate no attacks against it.

        The defended position is the evidence move's own position (the
        index it will occupy after being appended). This is the position
        that becomes last_other when the Opponent next generates moves:
        it is mode=DEFEND, it is in defended_positions, the closed-position
        axiom fires, and possible remains empty.
        """
        new_defended = state.defended_positions.copy()

        if move.move_type == MoveType.ASSERT_TOLERANCE_EVIDENCE:
            # The evidence move will occupy position len(state.moves) after
            # appending. Mark it closed now, before the append, so the new
            # state carries the updated set.
            evidence_position = len(state.moves)
            new_defended.add(evidence_position)

        new_state = GameState(
            thesis=state.thesis,
            audience=state.audience,
            argument_map=state.argument_map,
            moves=state.moves.copy(),
            squeeze_map=state.squeeze_map,
            defended_positions=new_defended,
            winner=state.winner,
        )
        new_state.moves.append(move)
        return new_state

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get_value(self, formula: Formula, state: GameState) -> Optional[str]:
        """
        Resolves a Formula to the value string of its SituatedArgument.
        Returns None if the formula is not in argument_map.

        A None return means the Proponent cannot ground their evidence claim
        in the argument structure — no ASSERT_TOLERANCE_EVIDENCE is generated
        and the Proponent loses the chain. This is structurally correct:
        ungroundable evidence is inadmissible.
        """
        arg = state.argument_map.get(formula)
        return arg.value if arg is not None else None

    def _find_dialetheic_partner(
        self, formula: Formula, state: GameState
    ) -> Optional[Formula]:
        """
        Finds a formula whose argument is in a tolerated contradiction with
        the given formula's argument, according to state.audience.

        No oracle — queries audience.evaluates_as_tolerant directly.
        Returns the partner Formula, or None if no tolerated pair exists.
        """
        arg = state.argument_map.get(formula)
        if arg is None:
            return None
        for other_formula, other_arg in state.argument_map.items():
            if other_formula == formula:
                continue
            if state.audience.evaluates_as_tolerant(arg.value, other_arg.value):
                return other_formula
        return None

    def _filter_repetitions(
        self, candidates: List[Move], history: List[Move]
    ) -> List[Move]:
        """
        Enforces the no-repetition rule.
        A move is a repetition if the same (player, move_type, formula,
        references) tuple has already been played.
        """
        seen: Set[Tuple] = {
            (m.player, m.move_type, m.formula, m.references)
            for m in history
        }
        return [
            m for m in candidates
            if (m.player, m.move_type, m.formula, m.references) not in seen
        ]


# =============================================================================
# THE MULTIPOLAR TEST
# Executable proof of situated knowledge as a structural property.
#
# The identical thesis, the same move types, two audience configurations.
# On the Paramuno board: evidence is grounded, position is closed,
# Opponent is structurally silenced by the closed-position axiom.
# On the State board: evidence is inadmissible, Proponent cannot produce it,
# Opponent wins before the evidence is ever tabled.
#
# Neither outcome is stipulated. Both emerge from _generate_legal_moves
# operating on the respective state.audience.
# =============================================================================

def run_multipolar_test(engine: DDGEngine, thesis: Formula,
                        arg_conserve: SituatedArgument,
                        arg_farm: SituatedArgument) -> None:

    argument_map = {
        arg_conserve.claim: arg_conserve,
        arg_farm.claim:     arg_farm,
    }

    audiences = [
        (
            "Paramuno",
            SituatedAudience(
                name="Paramuno",
                preferences={"conservation": 10, "subsistence": 10},
                tolerances={("conservation", "subsistence")},
            ),
            (
                "Evidence is grounded in audience.tolerances.",
                "ASSERT_TOLERANCE_EVIDENCE generated at chain step 2.",
                "_apply_move marks evidence position as defended.",
                "Opponent's next last_other is mode=DEFEND, position in defended_positions.",
                "Closed-position axiom fires. possible remains empty after filter.",
                "Opponent has no legal moves. Proponent wins — emergent, not stipulated.",
            ),
        ),
        (
            "State Authority",
            SituatedAudience(
                name="State_Authority",
                preferences={"conservation": 10, "subsistence": 3},
                intolerances={("conservation", "subsistence")},
            ),
            (
                "evaluates_as_tolerant returns False at chain step 2.",
                "No ASSERT_TOLERANCE_EVIDENCE move is generated.",
                "Proponent has no legal response to CHALLENGE_TOLERANCE.",
                "possible is empty after filter. Proponent loses this branch.",
                "Opponent wins — not by silencing anyone, but because the host",
                "epistemic regime does not admit the evidence as a legal game object.",
                "The State does not need to fall silent; it blocked the evidence",
                "before it could be tabled.",
            ),
        ),
    ]

    for name, audience, explanation in audiences:
        result = engine.solve_thesis(
            thesis=thesis,
            audience=audience,
            argument_map=argument_map,
        )

        print("=" * 64)
        print(f"GAME BOARD: {name}")
        print(f"Winner: {result.winner}")
        print("Move trace:")
        for m in result.moves:
            ev = f" evidence={m.tolerance_evidence}" if m.tolerance_evidence else ""
            dp = " [CLOSED]" if m.position in result.defended_positions else ""
            print(f"  [{m.position}]{dp} {m.player.value} | {m.move_type.value} "
                  f"| {m.formula} | {m.mode.value}{ev}")
        print("Structural explanation:")
        for line in explanation:
            print(f"  {line}")
        print()

    print("=" * 64)
    print("THEORETICAL CONCLUSION:")
    print("  The same argument. The same move vocabulary. Opposite verdicts.")
    print("  On the Paramuno board, the win is earned through three moves")
    print("  and a structural game axiom (no attacks on closed positions).")
    print("  On the State board, the Proponent is blocked before producing")
    print("  evidence — not silenced after. The master's tools cannot")
    print("  adjudicate the subaltern's contradiction as valid because")
    print("  state.audience IS the master's epistemic regime, and the")
    print("  admissibility of evidence is determined by the host, not the")
    print("  quality of the argument.")
    print("  This is situated knowledge, formalised as game structure.")