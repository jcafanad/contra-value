"""
Tests for cubun/af.py — DynamicAF and ParaconsistentAFSolver.
"""

from chuaque.formulas import Atom, Negation
from chuaque.truth_values import TruthValue
from cubun.af import (
    SituatedArgument, SituatedAudience, DynamicVAFGenerator,
    ParaconsistentAFSolver,
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
