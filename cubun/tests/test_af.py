"""
Tests for cubun/af.py — DynamicAF and ParaconsistentAFSolver.
"""

from chuaque.formulas import Atom, Negation
from chuaque.truth_values import TruthValue
from cubun.af import (
    SituatedArgument, SituatedAudience, DynamicVAFGenerator,
    DynamicAF, ParaconsistentAFSolver,
)


class TestInitialLabels:
    """
    Tests for DynamicAF.initial_labels — orchestrator seeding.
    Confirms that initial_labels override topology seed without
    affecting existing scenarios (empty dict = no change).
    """

    def test_initial_labels_empty_is_unchanged(self):
        """Empty initial_labels: seeding is identical to topology seed."""
        a1 = SituatedArgument("A1", Atom("land"), "conservation")
        a2 = SituatedArgument("A2", Negation(Atom("land")), "subsistence")
        aud = SituatedAudience(
            name="test",
            preferences={"conservation": 5, "subsistence": 5},
            tolerances={("conservation", "subsistence")},
            semantic_incompatibilities={(Atom("land"), Negation(Atom("land")))},
        )
        af = DynamicVAFGenerator([a1, a2]).generate_framework(aud)
        assert af.initial_labels == {}  # default
        result = ParaconsistentAFSolver(af).evaluate()
        # Both in dialetheic_bonds -> seeded at I
        assert result[a1] == TruthValue.I
        assert result[a2] == TruthValue.I

    def test_initial_labels_F_seeds_refuted_argument(self):
        """
        ⊘-seeding: argument with initial_label=F is seeded at F at Round 0.
        Simulates the Paramuno lifeworld refutation of the State resource claim.

        NOTE: the F-seed is topologically transient for an unattacked argument.
        The Bivalent Ghost overwrites it at Round 1 (no influencers → T).
        Fixed point: paramuno_arg=T (accepted), state_claim=F (rejected).

        The Paramuno refutation works: state_claim is rejected. The path is
        indirect — the F-seed perturbs Round 0 (state_claim receives negation(F)=T),
        and the topology drives state_claim to F at Round 2. The fixed-point
        labels are identical to the no-seed case because the topology
        (conservation intolerant of extraction) already drives state_claim to F.
        The NLI signal enters as a perturbation; the topology completes it.

        See C2 in assessment v6 for the full round-by-round trace and
        the political reading of the Bivalent Ghost operating on the
        Paramuno argument.
        """
        a1r = SituatedArgument("state_claim", Atom("recurso"), "extraction")
        a2r = SituatedArgument("paramuno",    Atom("ser_vivo"), "conservation")
        aud = SituatedAudience(
            name="test",
            preferences={"extraction": 1, "conservation": 10},
            intolerances={("conservation", "extraction")},
            semantic_incompatibilities={(Atom("recurso"), Atom("ser_vivo"))},
        )
        afr = DynamicVAFGenerator([a1r, a2r]).generate_framework(aud)
        # Orchestrator seeds paramuno argument as F (refutation detected)
        afr.initial_labels[a2r] = TruthValue.F
        result = ParaconsistentAFSolver(afr).evaluate()
        assert result[a2r] == TruthValue.T, \
            "paramuno seed=F but fixed-point=T (Bivalent Ghost transient overwrite)"
        assert result[a1r] == TruthValue.F, \
            "state_claim rejected: Paramuno refutation routes through the topology"

    def test_initial_labels_I_seeds_dialetheia(self):
        """
        ⊗-seeding: argument with initial_label=I is seeded at I even without
        topology bond. The seed is topologically transient for an isolated argument
        (no attackers, no partners): Bivalent Ghost overwrites at Round 1.
        Fixed point: both arguments=T.

        This test confirms the solver does not crash and reaches a fixed point.
        For stable I-seeding, the argument must be in a topology with influencers.
        See C2 in assessment v6 for the full trace.
        """
        a1i = SituatedArgument("A1", Atom("land"), "conservation")
        a2i = SituatedArgument("A2", Atom("water"), "subsistence")  # no bond with A1
        aud = SituatedAudience(
            name="test",
            preferences={"conservation": 5, "subsistence": 5},
            # NO tolerances — no dialetheic_bonds
            semantic_incompatibilities=set(),
        )
        afi = DynamicVAFGenerator([a1i, a2i]).generate_framework(aud)
        # Orchestrator seeds A2 as I (dialetheia detected)
        afi.initial_labels[a2i] = TruthValue.I
        result = ParaconsistentAFSolver(afi).evaluate()
        # Both converge to T: no influencers in either argument
        assert result[a1i] == TruthValue.T
        assert result[a2i] == TruthValue.T, \
            "I-seed transient: Bivalent Ghost overwrites at Round 1 for isolated arg"


class TestWeightInjection:
    """
    Tests for Option C weight integration in ParaconsistentAFSolver.

    Weight injection: if weight(arg) >= threshold, inject I into attack_value
    at each round before the negation step.

    Three core behaviours validated here:

    1. Ghost resistance — unattacked high-weight arg → I (not T).
       Without injection: attack_value=F → ∼F=T (Bivalent Ghost).
       With injection:    attack_value=F⊕I=I → ∼I=I.

    2. Coalitional bond recovery — the 4-argument exemplar showing that
       a peripheral high-weight ally (R) refusing the Ghost propagates
       I through the network, allowing a bond partner (P) to recover
       from transient erasure to I at the fixed point.
       Without injection: P=F, X=T, S=F, R=T (bivalent collapse).
       With injection:    P=I, X=I, S=I, R=I (bond sustained).

    3. Certified Truth still erases — weight injection cannot protect
       against a T-labelled attacker (T⊕I=T → ∼T=F). The Violence of
       Certified Truth is not mitigated by engagement intensity.

    4. Backward compatibility — no weights passed → solver behaves
       identically to the pre-weight implementation.
    """

    # ------------------------------------------------------------------
    # Shared fixtures
    # ------------------------------------------------------------------

    def _isolated_arg(self):
        """One isolated argument, no bonds, no attackers."""
        arg = SituatedArgument("P", Atom("territorio"), "nature")
        aud = SituatedAudience(
            name="test",
            preferences={"nature": 5},
        )
        af = DynamicVAFGenerator([arg]).generate_framework(aud)
        return arg, af

    def _coalitional_af(self):
        """
        4-argument topology from the coalitional exemplar:
            P ↔ bond ↔ X   (both "nature", syntactically contradictory)
            S → attacks → P (extraction rank 5 > nature rank 3)
            R → attacks → S (ally rank 10 > extraction rank 5)

        P.claim = Atom("territorio"),           X.claim = Negation(Atom("territorio"))
        S.claim = Atom("recurso"),              R.claim = Atom("alianza")

        Bond: P and X are syntactically contradictory; audience tolerates
              ("nature", "nature") → dialetheic bond, seeded at I.

        S attacks P: semantic_incompatibility (recurso, territorio);
                     extraction(5) >= nature(3) → S→P only (not P→S).

        R attacks S: semantic_incompatibility (alianza, recurso);
                     ally(10) >= extraction(5) → R→S only (not S→R).

        No other links (R/S not contradictory with X by construction).

        Baseline trace (no weights):
            Rd0: P=I, X=I, S=N, R=N
            Rd1: P=F (N⊕I=T→F), X=I, S=N, R=T (Ghost)
            Rd2: P=F, X=T (P=F→Ghost), S=F (R=T→F), R=T
            Fixed: P=F, X=T, S=F, R=T

        Option C trace (P/X/R weight=0.8, S weight=0.1, θ=0.5):
            Rd1: P=F (N⊕I=T, inject T→F), X=I, S=N, R=I (inject F→I, resist Ghost)
            Rd2: P=F (N⊕I=T→F), X=I (inject F→I), S=I (R=I→∼I=I), R=I
            Rd3: P=I (I⊕I=I, inject→I), X=I, S=I, R=I  ← recovery
            Fixed: P=I, X=I, S=I, R=I
        """
        p = SituatedArgument("P", Atom("territorio"),           "nature")
        x = SituatedArgument("X", Negation(Atom("territorio")), "nature")
        s = SituatedArgument("S", Atom("recurso"),              "extraction")
        r = SituatedArgument("R", Atom("alianza"),              "ally")

        aud = SituatedAudience(
            name="test",
            preferences={"nature": 3, "extraction": 5, "ally": 10},
            tolerances={("nature", "nature")},
            intolerances={
                ("extraction", "nature"),  # S attacks P
                ("ally", "extraction"),    # R attacks S
            },
            semantic_incompatibilities={
                (Atom("recurso"), Atom("territorio")),  # S vs P
                (Atom("alianza"), Atom("recurso")),     # R vs S
            },
        )
        af = DynamicVAFGenerator([p, x, s, r]).generate_framework(aud)
        weights = {p: 0.8, x: 0.8, s: 0.1, r: 0.8}
        return p, x, s, r, af, weights

    # ------------------------------------------------------------------
    # Test 1: Ghost resistance
    # ------------------------------------------------------------------

    def test_ghost_resistance_high_weight(self):
        """
        Unattacked high-weight argument → I (resists Bivalent Ghost).

        Without injection: attack_value=F → ∼F=T.
        With injection:    F⊕I=I → ∼I=I.
        """
        arg, af = self._isolated_arg()
        result = ParaconsistentAFSolver(
            af, weights={arg: 0.8}
        ).evaluate()
        assert result[arg] == TruthValue.I, \
            "High-weight isolated arg should resist Ghost and evaluate to I"

    def test_ghost_accepted_low_weight(self):
        """
        Unattacked low-weight argument still → T (Ghost accepted).
        """
        arg, af = self._isolated_arg()
        result = ParaconsistentAFSolver(
            af, weights={arg: 0.2}
        ).evaluate()
        assert result[arg] == TruthValue.T, \
            "Low-weight isolated arg below threshold: Ghost T accepted"

    def test_ghost_accepted_at_threshold_boundary(self):
        """
        Weight exactly at threshold → injection fires → I (not T).
        """
        arg, af = self._isolated_arg()
        result = ParaconsistentAFSolver(
            af, weights={arg: 0.5}, weight_threshold=0.5
        ).evaluate()
        assert result[arg] == TruthValue.I

    # ------------------------------------------------------------------
    # Test 2: Coalitional bond recovery
    # ------------------------------------------------------------------

    def test_weight_injection_bond_sustained_coalitional(self):
        """
        4-argument coalitional scenario: P↔X bond, S→P, R→S.

        Without weight injection (baseline):
            R defaults to T via Ghost → S erased (F) → X collapses (T) →
            P=F, X=T, S=F, R=T.

        With weight injection (C, θ=0.5, P/X/R weight=0.8, S weight=0.1):
            R injects I → resists Ghost → R=I.
            R=I propagates to S=I (not F).
            Once S=I and X=I, P's attack_value=I → inject I → I → P=I.
            Fixed point: P=I, X=I, S=I, R=I.

        Trace (using Round-0 seeds P=I, X=I, S=N, R=N):
            Round 1: P=F (N⊕I=T→F; Erasure by Ignorance), X=I, S=N, R=I
            Round 2: P=F (N⊕I=T→F), X=I, S=I, R=I
            Round 3: P=I (I⊕I=I→I), X=I, S=I, R=I  ← recovery
            Fixed point.
        """
        p, x, s, r, af, weights = self._coalitional_af()

        baseline = ParaconsistentAFSolver(af).evaluate()
        assert baseline[p] == TruthValue.F, "Baseline: P erased"
        assert baseline[x] == TruthValue.T, "Baseline: X survives as T (survivorship)"
        assert baseline[r] == TruthValue.T, "Baseline: R accepted Ghost T"

        injected = ParaconsistentAFSolver(af, weights=weights).evaluate()
        assert injected[p] == TruthValue.I, "With C: P recovers to I"
        assert injected[x] == TruthValue.I, "With C: X sustained as I"
        assert injected[r] == TruthValue.I, "With C: R resists Ghost, stays I"

    # ------------------------------------------------------------------
    # Test 3: Certified Truth still erases
    # ------------------------------------------------------------------

    def test_certified_truth_still_erases_high_weight(self):
        """
        T-labelled attacker erases high-weight target despite injection.

        T⊕I=T → ∼T=F. The Certified Erasure is not mitigated
        by engagement intensity. Weight injection offers no protection
        against a genuinely T-labelled dominating argument.

        Setup: S (weight=0.0) attacks P (weight=0.8).
        S is unattacked and has no injection → S=T via Ghost.
        S=T attacks P: T⊕I=T → P=F.
        """
        p = SituatedArgument("P", Atom("tierra"), "subsistence")
        s = SituatedArgument("S", Negation(Atom("tierra")), "dominance")
        aud = SituatedAudience(
            name="test",
            preferences={"subsistence": 3, "dominance": 10},
            intolerances={("dominance", "subsistence")},
            semantic_incompatibilities={
                (Atom("tierra"), Negation(Atom("tierra"))),
            },
        )
        af = DynamicVAFGenerator([p, s]).generate_framework(aud)
        result = ParaconsistentAFSolver(
            af, weights={p: 0.8, s: 0.0}
        ).evaluate()
        assert result[s] == TruthValue.T, "S unattacked, low weight → Ghost T"
        assert result[p] == TruthValue.F, \
            "High-weight P still erased by certified T: T⊕I=T → F"

    # ------------------------------------------------------------------
    # Test 4: Backward compatibility
    # ------------------------------------------------------------------

    def test_no_weights_backward_compatible(self):
        """
        Passing no weights leaves solver behaviour identical to baseline.
        The dialetheic bond scenario from TestInitialLabels is unaffected.
        """
        a1 = SituatedArgument("A1", Atom("land"), "conservation")
        a2 = SituatedArgument("A2", Negation(Atom("land")), "subsistence")
        aud = SituatedAudience(
            name="test",
            preferences={"conservation": 5, "subsistence": 5},
            tolerances={("conservation", "subsistence")},
            semantic_incompatibilities={(Atom("land"), Negation(Atom("land")))},
        )
        af = DynamicVAFGenerator([a1, a2]).generate_framework(aud)
        result = ParaconsistentAFSolver(af).evaluate()
        assert result[a1] == TruthValue.I
        assert result[a2] == TruthValue.I
