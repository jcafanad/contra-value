"""
cubun/af.py — Paraconsistent Value-Based Argumentation Framework.

CRITICAL DEPENDENCY:
    Requires chuaque/truth_values.py implementing the Belnap-Dunn FOUR bilattice.
    This version calls instance methods (.join_t(), .negation) on TruthValue objects
    directly, making it impervious to the @property vs @staticmethod collision that
    broke earlier versions. The situated object evaluates its own material reality.

THEORETICAL ARCHITECTURE:
    Attacks are generated dynamically when a SituatedAudience imposes intolerance
    upon a logical contradiction:
        INTOLERANCE  →  classical attack (directed by value rank)
        TOLERANCE    →  dialetheic bond (positive recognition of the Both)
        APATHY       →  no structural link (evaluates to T via Dung's Bivalent Ghost)

THE ABANDONMENT OF KNASTER-TARSKI ITERATION (not the theorem):
    Knaster-Tarski as a theorem holds unconditionally. What is abandoned is the
    precondition of starting at ⊥_k = N for all arguments. By introducing Ontological
    Seeding (dialetheic arguments initialised at ⊤_k = I), we start from a mixed
    point. Convergence is therefore TOPOLOGY-DEPENDENT. We have traded guaranteed
    computational convergence for material fidelity to the lifeworld.

    When the topology contains a feedback loop between a dialetheic bond and a
    State intervention, the iteration enters a limit cycle. We detect this
    exactly and raise DialecticalOscillation — a critical-theoretical finding
    encoded as an exception.

    The cycle is not static entrapment but ALTERNATING OPPRESSION:
        State 0: conservation=T (classical certainty), subsistence=I (contradiction
                 alive, designated), mediation=I (the colonial institution is itself
                 contradictory — it admits contradiction even as it intervenes).
        State 1: conservation=I (collapses to contradiction), subsistence=F (erased),
                 mediation=F (rejected).

    The subaltern oscillates between two modes of erasure: partial recognition (I)
    and annihilation (F). The colonial machine does not simply trap contradiction —
    it rhythmically destroys and reconstitutes it, with subsistence caught between
    contradiction and elimination on a period-2 clock.

CYCLE DETECTION — COMPLEXITY AND GUARANTEES:

    The bilattice has exactly 4^n possible states for n arguments.
    By the Pigeonhole Principle, the iteration must either:
        (a) reach a fixed point (within 4^n rounds), or
        (b) revisit a prior state (within 4^n + 1 rounds).

    Both cases are detected exactly. Termination is therefore guaranteed within
    4^n + 1 rounds — not empirically, but as a mathematical consequence of the
    finite state space.

    PERFORMANCE: A naive `new_labels in history` scan costs O(history_length × n)
    per round. Over 4^n rounds, total cost O(n × 16^n) — combinatorially explosive
    for n ≥ 10. The correct fix is O(1) per-round lookup via hashing:

        Each state is converted to a hashable frozenset key: {(arg.name, val), ...}.
        A `seen: Set[frozenset]` provides O(n) per-round hash computation + O(1) lookup.
        The `history: List` is retained only for extracting the exact cycle on detection.
        Total cost: O(n × 4^n) instead of O(n × 16^n).

    For the fieldwork use case (n ≤ 10), the difference is 6 vs 12 orders of magnitude.

TWO-ROUND ERASURE (Violence of Ignorance → Violence of Certified Truth):

    ROUND 1 — Violence of Ignorance (Fact A: N ⊕_t I = T):
        B = N (not yet evaluated). A1's partner A2 = I (seeded).
        attack_value(A1) = F ⊕_t N ⊕_t I = N ⊕_t I = T.
        new_label(A1) = ∼(T) = F.
        A1 is erased by the State's IGNORANCE before B establishes any warrant.
        The incomparable pair (N, I) is forcibly resolved to T by the
        instrumental truth-join: hallucinated certainty synthesised from ignorance.

    ROUND 2+ — Violence of Certified Truth (Fact B: T ⊕_t I = T):
        B = T (unattacked, converged). A1 is kept at F by certified dominance.
        The mechanism has shifted; the output (F) is identical.

    SURVIVORSHIP COLLAPSE:
        A2 looks at A1's material state (F) in Round 2.
        attack_value(A2) = F ⊕_t F = F. new_label(A2) = ∼(F) = T.
        The surviving partner is stripped of its paraconsistent richness (I)
        and instrumentally recruited into bivalent T.

DUNG'S BIVALENT GHOST:
    Arguments with no structural links evaluate to T.
    attack_value = F (⊕_t identity), ∼(F) = T.
    Inherited from P.M. Dung's 1995 foundational default: unattacked arguments
    are classically True. In a genuinely paraconsistent framework, they should
    evaluate to N. The T is the algorithm's compulsive resolution of the
    unobserved vacuum — the computational formalisation of the Eurocentric
    default that the uncontested is the True.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from chuaque.formulas import Formula, Negation, Atom
from chuaque.truth_values import TruthValue


# =============================================================================
# Classical Dung Argumentation Framework
# =============================================================================

@dataclass(frozen=True)
class Argument:
    """
    A classical argument: name, support (set of formulas), claim.

    VAF = ⟨A, R, V, val, P⟩ — the argument carries its value via VAF.val.
    """
    name:    str
    support: FrozenSet[Formula]
    claim:   Formula

    def __repr__(self) -> str:
        return self.name


@dataclass
class AF:
    """
    Plain Dung Argumentation Framework ⟨Arguments, Attacks⟩.

    Provides grounded and preferred extension semantics via standard
    fixed-point / admissibility algorithms.
    """
    arguments: Set[Argument]
    attacks:   Set[Tuple[Argument, Argument]]

    def attackers_of(self, arg: Argument) -> Set[Argument]:
        return {a for a, t in self.attacks if t == arg}

    def attacked_by(self, arg: Argument) -> Set[Argument]:
        return {t for a, t in self.attacks if a == arg}

    def _is_conflict_free(self, S: Set[Argument]) -> bool:
        return all(
            (a, b) not in self.attacks
            for a in S for b in S
        )

    def _defends(self, S: Set[Argument], arg: Argument) -> bool:
        """S defends arg iff every attacker of arg is attacked by some member of S."""
        return all(
            any((c, att) in self.attacks for c in S)
            for att in self.attackers_of(arg)
        )

    def _is_admissible(self, S: Set[Argument]) -> bool:
        if not self._is_conflict_free(S):
            return False
        return all(self._defends(S, a) for a in S)

    def grounded_extension(self) -> Set[Argument]:
        """
        Least fixed point of F(S) = {A : S defends A}.
        Computed by monotone iteration from ∅.
        """
        ext: Set[Argument] = set()
        while True:
            new = ext | {a for a in self.arguments if self._defends(ext, a)}
            if new == ext:
                return ext
            ext = new

    def preferred_extensions(self) -> List[Set[Argument]]:
        """
        Maximal admissible sets (preferred extensions).
        Computed by enumerating all subsets and keeping maximal admissible ones.
        Exponential in |arguments| — suitable for small frameworks (n ≤ 15).
        """
        args = list(self.arguments)
        admissible: List[Set[Argument]] = []
        for r in range(len(args) + 1):
            for subset in combinations(args, r):
                S = set(subset)
                if self._is_admissible(S):
                    admissible.append(S)
        preferred = [
            S for S in admissible
            if not any(S < T for T in admissible)
        ]
        return preferred if preferred else [set()]


# =============================================================================
# Exceptions: Theoretical Findings as Exceptions
# =============================================================================

class DialecticalOscillation(Exception):
    """
    Raised when the fixed-point iteration enters a limit cycle.

    Carries the exact CLOSED cycle of evaluation states as computational proof
    that the interaction between the State's instrumental reason and the
    subaltern's dialetheia is topologically irresolvable under the truth-
    ordering machinery. cycle_states[0] == cycle_states[-1] by construction,
    making the repetition explicit in the data structure itself.

    cycle_period: the number of DISTINCT states in the cycle (= len - 1).
    """
    def __init__(
        self,
        cycle_states: List[Dict["SituatedArgument", TruthValue]],
    ) -> None:
        self.cycle_states = cycle_states
        self.cycle_period = len(cycle_states) - 1
        super().__init__(
            f"Unresolvable Dialectic: the truth-ordering machinery has collapsed "
            f"into a limit cycle of period {self.cycle_period}. "
            f"The colonial topology cannot force a resolution."
        )

    def render_cycle(self) -> str:
        """
        Returns a human-readable proof of the cycle.
        The closing ↺ line is identical to line 0, making the repetition visible.
        """
        lines = ["Cycle proof (closed — first state repeats at end):"]
        for i, state in enumerate(self.cycle_states):
            label  = "↺ " if i == len(self.cycle_states) - 1 else f"{i}. "
            values = ", ".join(
                f"{arg.name}({arg.value})={val.value}"
                for arg, val in state.items()
            )
            lines.append(f"  {label}{values}")
        return "\n".join(lines)


# =============================================================================
# Core Dataclasses & Audience Geometry
# =============================================================================

@dataclass(frozen=True)
class SituatedArgument:
    """
    A formal argument embedded in a lifeworld value.
    frozen=True ensures structural equality for use as Dict/Set keys.
    """
    name:  str
    claim: Formula
    value: str


@dataclass
class SituatedAudience:
    """
    An audience defined by its evaluative geometry.

    NAMING CONVENTION:
        is_tolerant / is_intolerant — symmetric primary pair.
        evaluates_as_tolerant — alias retained for ddg_patch_v3.py compatibility.

    __post_init__ enforces mutual exclusion: a value pair cannot be simultaneously
    tolerated and intolerated. This is a data integrity constraint.
    """
    name:                       str
    preferences:                Dict[str, int]
    intolerances:               Set[Tuple[str, str]] = field(default_factory=set)
    tolerances:                 Set[Tuple[str, str]] = field(default_factory=set)
    semantic_incompatibilities: Set[Tuple[Formula, Formula]] = field(default_factory=set)

    def __post_init__(self) -> None:
        reversed_tolerances = {(b, a) for a, b in self.tolerances}
        overlap = self.intolerances & (self.tolerances | reversed_tolerances)
        if overlap:
            raise ValueError(
                f"Audience '{self.name}' has contradictory evaluations "
                f"(both tolerated and intolerated): {overlap}"
            )

    def is_intolerant(self, val_a: str, val_b: str) -> bool:
        return (val_a, val_b) in self.intolerances or (val_b, val_a) in self.intolerances

    def is_tolerant(self, val_a: str, val_b: str) -> bool:
        return (val_a, val_b) in self.tolerances or (val_b, val_a) in self.tolerances

    def evaluates_as_tolerant(self, val_a: str, val_b: str) -> bool:
        """Alias for is_tolerant — used by cubun/ddg.py."""
        return self.is_tolerant(val_a, val_b)


# =============================================================================
# Dynamic Argumentation Framework
# =============================================================================

class DynamicAF:
    """
    Carries the relational topology generated by a SituatedAudience.
        _attackers: classical attack graph (intolerated contradictions)
        _partners:  dialetheic bond graph  (tolerated contradictions)

    initial_labels: Optional pre-seeding dict populated by the pipeline
        orchestrator (paramo/pipeline.py) based on transducer squeeze_map output.
        Entries override the topology-based seed in ParaconsistentAFSolver:
            ⊗-keyed atoms → TruthValue.I   (dialetheia detected)
            ⊘-keyed atoms → TruthValue.F   (refutation detected)
        Default: empty dict. All existing scenarios unaffected.
    """
    def __init__(
        self,
        arguments:        List[SituatedArgument],
        attack_graph:     Dict[SituatedArgument, Set[SituatedArgument]],
        dialetheia_pairs: Set[Tuple[SituatedArgument, SituatedArgument]],
        initial_labels:   Optional[Dict[SituatedArgument, TruthValue]] = None,
    ) -> None:
        self.arguments      = arguments
        self.initial_labels: Dict[SituatedArgument, TruthValue] = \
            initial_labels if initial_labels is not None else {}
        self._attackers: Dict[SituatedArgument, Set[SituatedArgument]] = {
            arg: set() for arg in arguments
        }
        self._partners: Dict[SituatedArgument, Set[SituatedArgument]] = {
            arg: set() for arg in arguments
        }
        for attacker, targets in attack_graph.items():
            for target in targets:
                self._attackers[target].add(attacker)
        for a, b in dialetheia_pairs:
            self._partners[a].add(b)
            self._partners[b].add(a)

    def attackers_of(self, arg: SituatedArgument) -> Set[SituatedArgument]:
        return self._attackers.get(arg, set())

    def partners_of(self, arg: SituatedArgument) -> Set[SituatedArgument]:
        return self._partners.get(arg, set())


# =============================================================================
# Generator
# =============================================================================

class DynamicVAFGenerator:
    """
    Generates a DynamicAF from arguments and a SituatedAudience.
    The three-case switch (intolerance / tolerance / apathy) bridges
    the lifeworld and the formal argumentation layer.
    """
    def __init__(self, arguments: List[SituatedArgument]) -> None:
        self.arguments = arguments

    def _are_materially_contradictory(
        self,
        arg1: SituatedArgument,
        arg2: SituatedArgument,
        audience: SituatedAudience,
    ) -> bool:
        """
        Syntactic contradiction: formal, audience-independent.
        Semantic incompatibility: cultural-epistemic, audience-relative.
        """
        if arg1.claim == Negation(arg2.claim) or arg2.claim == Negation(arg1.claim):
            return True
        return (
            (arg1.claim, arg2.claim) in audience.semantic_incompatibilities
            or (arg2.claim, arg1.claim) in audience.semantic_incompatibilities
        )

    def generate_framework(self, audience: SituatedAudience) -> DynamicAF:
        """
        Generate a DynamicAF from the argument list and a SituatedAudience.

        NOTE ON MISSING PREFERENCES:
            Values not present in audience.preferences default to rank 0.
            If both values in an intolerance pair are absent, both rank 0 and
            the >= condition fires in both directions, producing a mutual attack.
            This is the same mechanism used deliberately for equal-rank parity
            (e.g., conservation=subsistence=10 in the Paramuno audience).
            Callers should ensure all values referenced in intolerances/tolerances
            are explicitly ranked to avoid silent mutual-attack topology.
        """
        attack_graph: Dict[SituatedArgument, Set[SituatedArgument]] = {
            arg: set() for arg in self.arguments
        }
        dialetheia_pairs: Set[Tuple[SituatedArgument, SituatedArgument]] = set()

        for i, arg_a in enumerate(self.arguments):
            for arg_b in self.arguments[i + 1:]:
                if not self._are_materially_contradictory(arg_a, arg_b, audience):
                    continue
                val_a, val_b = arg_a.value, arg_b.value
                if audience.is_intolerant(val_a, val_b):
                    rank_a = audience.preferences.get(val_a, 0)
                    rank_b = audience.preferences.get(val_b, 0)
                    if rank_a >= rank_b:
                        attack_graph[arg_a].add(arg_b)
                    if rank_b >= rank_a:
                        attack_graph[arg_b].add(arg_a)
                elif audience.is_tolerant(val_a, val_b):
                    dialetheia_pairs.add((arg_a, arg_b))
                # Apathy: no link. Dung's ghost grants T by default.

        return DynamicAF(self.arguments, attack_graph, dialetheia_pairs)


# =============================================================================
# State Hashing
# =============================================================================

def _state_key(
    labels: Dict[SituatedArgument, TruthValue]
) -> FrozenSet[Tuple[str, TruthValue]]:
    """
    Converts a label dict to a hashable frozenset key for O(1) set lookup.

    Uses arg.name (string) and val (Enum, hashable by identity) as the tuple.
    Argument names are assumed unique within a framework — a constraint that
    SituatedArgument.name carries by convention. If names collide, two distinct
    arguments would hash identically and cycle detection would produce false
    positives. The DynamicVAFGenerator does not currently enforce name
    uniqueness; this is a known limitation.

    Cost: O(n) per call to compute the frozenset from n (name, val) pairs.
    This replaces the O(history_length × n) cost of `new_labels in history`.
    """
    return frozenset((arg.name, val) for arg, val in labels.items())


# =============================================================================
# The Non-Monotonic Paraconsistent Solver
# =============================================================================

class ParaconsistentAFSolver:
    """
    Computes 4-valued argumentation semantics over a DynamicAF.

    INJECTION SEMANTICS:
        ONTOLOGICAL SEEDING: Dialetheic arguments start at I in Round 0.
        INFERENTIAL INJECTION: Partners inject their current material label.
        WEIGHT INJECTION (Option C): At each round, arguments with weight ≥
            weight_threshold inject TruthValue.I into their own attack_value
            before the negation step. This encodes per-round self-assertion:
            an argument deeply engaged with a real abstraction (high weight)
            keeps inserting itself into the dialectic.

    WEIGHT INJECTION — UPDATE FUNCTION:
        attack_value(a) = ⊕_t { x.join_t() : x ∈ attackers(a) ∪ partners(a) }
        if weight(a) ≥ θ:
            attack_value(a) = attack_value(a) ⊕_t I
        new_label(a) = attack_value(a).negation

    WEIGHT INJECTION — POLITICAL CONSEQUENCES:
        When attack_value = F (no influencers): F ⊕_t I = I → ∼I = I.
            Resists the Bivalent Ghost: the high-weight arg refuses the
            uncontested default (T) and stays in contradiction.
        When attack_value = N (ignorant attacker): N ⊕_t I = T → ∼T = F.
            The injection paradoxically activates Erasure by Ignorance
            against itself. The self-assertion cannot be disentangled from
            the latent T that the truth-join forces on N ⊕_t I.
        When attack_value = T (certified attacker): T ⊕_t I = T → ∼T = F.
            Certified State truth still erases. Weight-based self-injection
            provides no protection against a genuinely T-labelled attacker.
        When attack_value = I: I ⊕_t I = I → ∼I = I.
            Stable. The bond and the self-assertion reinforce each other.

        The recovery path (see test_weight_injection_bond_sustained_coalitional)
        is indirect: the injection prevents allied high-weight arguments from
        being absorbed into the Bivalent Ghost (F→I instead of F→T). This
        propagates through the network until the bond's attacker transitions
        from N/T to I, at which point the erased bond partner recovers to I.

        This reflects the social dynamic: contra-value gestures do not
        immediately protect against State intervention but create conditions
        under which the bond re-emerges through solidaristic argumentation.

    weights: Dict[SituatedArgument, float]
        Per-argument engagement weights from BETOWeightExtractor.annotate().
        Default: empty dict (no injection — backward compatible with all
        existing scenarios, which pass no weights).
    weight_threshold: float
        Arguments with weight ≥ threshold receive the I-injection.
        Default: 0.5.

    CONVERGENCE GUARANTEE:
        The bilattice has 4^n states. By Pigeonhole, the iteration terminates
        within 4^n + 1 rounds — either at a fixed point or a cycle.
        Both are detected exactly. Termination is mathematically guaranteed.
        Weight injection does not affect the state space or the guarantee.

    COMPLEXITY:
        Per-round: O(n) for computing new_labels + O(n) for hashing the state.
        Total: O(n × 4^n). Without hashing (naive list scan): O(n × 16^n).
        For n=10: 10 × 4^10 ≈ 10^7 (manageable) vs 10 × 16^10 ≈ 10^13 (not).

    THE CEILING ASSERTION:
        A hard ceiling at 4^n + 1 rounds is included as an assertion on the
        correctness of cycle detection. By Pigeonhole, if the solver reaches
        4^n + 1 rounds without hitting either the fixed-point or the cycle
        check, then _state_key has produced a collision — two distinct states
        hashed identically — and the cycle was missed. The ceiling surfaces
        this bug explicitly. It is NOT a safety net against memory exhaustion;
        it is a diagnostic for a broken _state_key. If it fires, the fix is
        in _state_key, not in the solver.
    """

    def __init__(
        self,
        af: DynamicAF,
        weights: Optional[Dict[SituatedArgument, float]] = None,
        weight_threshold: float = 0.5,
    ) -> None:
        self.af = af
        self.weights: Dict[SituatedArgument, float] = \
            weights if weights is not None else {}
        self.weight_threshold = weight_threshold

    def evaluate(self) -> Dict[SituatedArgument, TruthValue]:
        # ONTOLOGICAL SEEDING
        # initial_labels (from pipeline orchestrator) override topology seed.
        # Topology seed: I if dialetheic_bonds present, else N.
        current_labels: Dict[SituatedArgument, TruthValue] = {
            arg: self.af.initial_labels.get(
                arg,
                TruthValue.I if self.af.partners_of(arg) else TruthValue.N
            )
            for arg in self.af.arguments
        }

        # `history` retains full state dicts for cycle extraction.
        # `seen` provides O(1) existence checks via frozenset hashing.
        history: List[Dict[SituatedArgument, TruthValue]] = [current_labels]
        seen:    Set[FrozenSet[Tuple[str, TruthValue]]]   = {_state_key(current_labels)}

        # Ceiling: by Pigeonhole, 4^n + 1 rounds guarantees fixed point or cycle.
        # If exceeded: _state_key has a collision bug. See class docstring.
        ceiling = 4 ** len(self.af.arguments) + 1

        for round_number in range(ceiling):
            new_labels: Dict[SituatedArgument, TruthValue] = {}

            for arg in self.af.arguments:
                attack_value = TruthValue.F  # ⊕_t identity

                # POLYMORPHIC INSTANCE EVALUATION:
                # .join_t() and .negation are called on the TruthValue instances
                # directly, bypassing any staticmethod/property naming collision.
                for att in self.af.attackers_of(arg):
                    attack_value = attack_value.join_t(current_labels[att])

                # INFERENTIAL INJECTION: partner's current material label.
                for partner in self.af.partners_of(arg):
                    attack_value = attack_value.join_t(current_labels[partner])

                # WEIGHT INJECTION (Option C): high-weight arguments inject I
                # into their own attack_value at every round. See class docstring
                # for the full political consequences of each case.
                if self.weights.get(arg, 0.0) >= self.weight_threshold:
                    attack_value = attack_value.join_t(TruthValue.I)

                new_labels[arg] = attack_value.negation

            # Fixed point: convergence reached.
            if new_labels == current_labels:
                return new_labels

            # Cycle detection: O(n) hash + O(1) set lookup.
            new_key = _state_key(new_labels)
            if new_key in seen:
                # Locate the exact cycle start position in history.
                # history[i] matches new_labels if _state_key(history[i]) == new_key.
                # Linear scan of history is acceptable here: fires at most once,
                # and history.index requires a full equality check anyway.
                cycle_start = next(
                    i for i, s in enumerate(history)
                    if _state_key(s) == new_key
                )
                closed_cycle = history[cycle_start:] + [new_labels]
                raise DialecticalOscillation(closed_cycle)

            seen.add(new_key)
            history.append(new_labels)
            current_labels = new_labels

        # Unreachable by Pigeonhole if _state_key is collision-free.
        # If this line executes, _state_key has a collision bug.
        raise AssertionError(
            f"Ceiling of {ceiling} rounds exceeded without detecting a fixed point "
            f"or cycle. This indicates a hash collision in _state_key: two distinct "
            f"states produced the same frozenset key, causing the cycle detector to "
            f"miss a repetition. Fix _state_key, not the solver."
        )


# =============================================================================
# Integration Tests
# =============================================================================

def _render(label: str, result: Dict[SituatedArgument, TruthValue]) -> None:
    print(f"\n--- {label} ---")
    for arg, val in result.items():
        print(f"  [{arg.name}] {arg.value}: {val.value}")


if __name__ == "__main__":

    a1 = SituatedArgument("A1", Atom("land"),           "conservation")
    a2 = SituatedArgument("A2", Negation(Atom("land")),  "subsistence")
    a3 = SituatedArgument("A3", Atom("mining"),          "extraction")

    # -----------------------------------------------------------------------
    # Test 1: Paramuno — Tolerance + Intolerance + Apathy
    # Round 0: A1=I, A2=I, A3=N.
    # Round 1: A1=∼(I)=I, A2=∼(I)=I, A3=∼(I)=I (attacker A1=I).
    # Fixed point. Expected: A1=I, A2=I, A3=I.
    # -----------------------------------------------------------------------
    paramuno = SituatedAudience(
        name="Paramuno",
        preferences={"conservation": 10, "subsistence": 10, "extraction": 1},
        tolerances={("conservation", "subsistence")},
        intolerances={("conservation", "extraction")},
        semantic_incompatibilities={(Atom("land"), Atom("mining"))},
    )
    _render(
        "Paramuno (Tolerance + Intolerance + Apathy)",
        ParaconsistentAFSolver(
            DynamicVAFGenerator([a1, a2, a3]).generate_framework(paramuno)
        ).evaluate()
    )

    # -----------------------------------------------------------------------
    # Test 2: State Authority — Classical Erasure
    # Round 0: (N,N,N). Round 1: A1=T. Round 2: A2=F, A3=F. Stable.
    # Expected: A1=T, A2=F, A3=F.
    # -----------------------------------------------------------------------
    state = SituatedAudience(
        name="State_Authority",
        preferences={"conservation": 10, "subsistence": 3, "extraction": 1},
        intolerances={
            ("conservation", "subsistence"),
            ("conservation", "extraction"),
        },
        semantic_incompatibilities={(Atom("land"), Atom("mining"))},
    )
    _render(
        "State Authority (Classical Erasure)",
        ParaconsistentAFSolver(
            DynamicVAFGenerator([a1, a2, a3]).generate_framework(state)
        ).evaluate()
    )

    # -----------------------------------------------------------------------
    # Test 3: Distant Observer — Dung's Bivalent Ghost
    # No structural links. attack_value=F → ∼(F)=T for all.
    # Apathy grants T by default. The algorithm abhors the unobserved vacuum.
    # Expected: A1=T, A2=T, A3=T.
    # -----------------------------------------------------------------------
    distant = SituatedAudience(
        name="Distant_Observer",
        preferences={"conservation": 5, "subsistence": 5, "extraction": 5},
        semantic_incompatibilities={(Atom("land"), Atom("mining"))},
    )
    _render(
        "Distant Observer (Dung's Bivalent Ghost — apathy defaults to T)",
        ParaconsistentAFSolver(
            DynamicVAFGenerator([a1, a2, a3]).generate_framework(distant)
        ).evaluate()
    )

    # -----------------------------------------------------------------------
    # Test 4: Two-Round Erasure (Survivorship Collapse)
    # This test is the executable proof of the docstring's central claim.
    # Without it, the Violence of Ignorance / Violence of Certified Truth
    # distinction is asserted but not demonstrated.
    #
    # B attacks A1; A1↔A2 dialetheic bond.
    # Round 0: A1=I, A2=I, B=N.
    # Round 1: B=T; A1 erased by Violence of Ignorance (B=N, N⊕I=T → F); A2=I.
    # Round 2: B=T; A1 kept at F by Violence of Certified Truth (T⊕I=T → F);
    #          A2=∼(F⊕F)=T — survivorship collapse.
    # Fixed point: A1=F, A2=T, B=T.
    # -----------------------------------------------------------------------
    b_state = SituatedArgument("B", Atom("state_claim"), "dominance")
    two_round = SituatedAudience(
        name="Two_Round_State",
        preferences={"conservation": 5, "subsistence": 5, "dominance": 10},
        tolerances={("conservation", "subsistence")},
        intolerances={("dominance", "conservation")},
        semantic_incompatibilities={
            (Atom("state_claim"), Atom("land")),  # B and A1 are materially contradictory
        },
    )
    _render(
        "Two-Round Erasure (Survivorship Collapse)",
        ParaconsistentAFSolver(
            DynamicVAFGenerator([a1, a2, b_state]).generate_framework(two_round)
        ).evaluate()
    )
    print("  Round 1: A1 erased by Violence of Ignorance (B was N; N⊕_t I=T).")
    print("  Round 2: A1 kept at F by Violence of Certified Truth (B now T; T⊕_t I=T).")
    print("  A2: I → T. The survivor is instrumentally recruited into bivalence.")

    # -----------------------------------------------------------------------
    # Test 5: Unresolvable Dialectic — DialecticalOscillation
    # Topology: A1↔A2 dialetheic, A1 attacks B_med, B_med attacks A2.
    # The closed cycle is the computational proof; render_cycle() shows it.
    # -----------------------------------------------------------------------
    b_med = SituatedArgument("B", Atom("mediation"), "mediation")
    feedback = SituatedAudience(
        name="Feedback_Loop_State",
        preferences={"conservation": 10, "subsistence": 3, "mediation": 5},
        tolerances={("conservation", "subsistence")},
        intolerances={
            ("conservation", "mediation"),   # A1 attacks B (conservation rank 10 >= mediation rank 5)
            ("mediation", "subsistence"),    # B attacks A2 (mediation rank 5 >= subsistence rank 3)
        },
        semantic_incompatibilities={
            (Atom("land"), Atom("mediation")),           # A1 vs B contradictory
            (Atom("mediation"), Negation(Atom("land"))), # B vs A2 contradictory
        },
    )
    print("\n--- Unresolvable Dialectic (Feedback Loop) ---")
    try:
        ParaconsistentAFSolver(
            DynamicVAFGenerator([a1, a2, b_med]).generate_framework(feedback)
        ).evaluate()
        print("  [Converged — topology did not produce the expected cycle]")
    except DialecticalOscillation as e:
        print(f"  {e}")
        print(f"\n{e.render_cycle()}")
        print(
            f"\n  PROOF: The {e.cycle_period} states form a closed loop "
            f"(line 0 = line ↺). The colonial mediation cannot resolve the "
            f"Paramuno contradiction; it oscillates it indefinitely. "
            f"In symbolic AI, topology is politics."
        )
