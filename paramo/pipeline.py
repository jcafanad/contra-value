"""
paramo/pipeline.py — Pipeline Orchestrator.

Bridges the pquasqua transducer and the cubun PVAF solver.
Reads the squeeze_map returned by SituatedTransducer.mine_argument,
detects ⊗-keyed (dialetheia) and ⊘-keyed (refutation) atoms,
and seeds DynamicAF.initial_labels before solver construction.

This is the module that implements the transducer→cubun handoff.
Neither pquasqua nor cubun is coupled to the other at the
SituatedArgument/Argument boundary — the orchestrator mediates.

PARAMUNO LIFEWORLD EXAMPLE (refutation path):
    Premise: a Paramuno speaker asserts territorial sovereignty —
        "este páramo es nuestro territorio, no un recurso del Estado"
    Claim (State): "el páramo es recurso hídrico de utilidad pública"

    NLI output: E≈0.05, C≈0.85 — the Paramuno territorial assertion
    actively contradicts the State's public-utility framing.

    squeeze_map: {Atom("paramuno_territorio⊘recurso_hidrico"): entropy}

    Orchestrator reads the ⊘-atom and sets:
        af.initial_labels[paramuno_arg] = TruthValue.F

    Round 0: paramuno_arg = F. The Paramuno no is in the bilattice.
    Round 1: Bivalent Ghost (no attackers) overwrites paramuno_arg → T.
             state_claim, attacked by F, receives negation(F) = T.
    Round 2: state_claim, attacked by T, receives negation(T) = F.
    Fixed point: paramuno_arg = T, state_claim = F.

    The Paramuno territorial claim is accepted; the State's resource
    framing is rejected. The NLI-detected refutation entered the
    evaluation as a Round-0 perturbation and the topology completed it.
    The Bivalent Ghost — the colonial default — operates on the Paramuno
    argument, but the result is Paramuno victory through the topology.
    This is not incidental: the topology (conservation intolerant of
    extraction) encodes the Paramuno evaluative geometry directly.

    NOTE on the Bivalent Ghost (see cubun/af.py): the F-seed is transient
    for an unattacked argument. From Round 1, the topology determines
    everything. This is Option C — the perturbation is real and the
    topology completes what the NLI signal initiated. The outcome is
    correct; the path is indirect. For stable I/F-seeding in isolated
    arguments, the argument must be given attackers or partners.
"""

from typing import Dict, List, Tuple
from chuaque.formulas import Atom, Formula
from chuaque.truth_values import TruthValue
from cubun.af import (
    SituatedArgument, DynamicAF, DynamicVAFGenerator,
    SituatedAudience, ParaconsistentAFSolver,
)


def _is_dialetheia_key(formula: Formula) -> bool:
    """Returns True if formula is a ⊗-keyed atom (dialetheia signal)."""
    return isinstance(formula, Atom) and "⊗" in formula.name


def _is_refutation_key(formula: Formula) -> bool:
    """Returns True if formula is a ⊘-keyed atom (refutation signal)."""
    return isinstance(formula, Atom) and "⊘" in formula.name



class PipelineOrchestrator:
    """
    Orchestrates the transducer→cubun pipeline.

    Takes:
        - A list of (SituatedArgument, squeeze_map) pairs produced by
          SituatedTransducer.mine_argument
        - A SituatedAudience
    Produces:
        - A ParaconsistentAFSolver with initial_labels seeded from NLI signals

    The orchestrator is the only place where pquasqua and cubun are coupled.
    """

    def __init__(
        self,
        arg_squeeze_pairs: List[Tuple[SituatedArgument, Dict[Formula, float]]],
        audience: SituatedAudience,
    ):
        """
        INTERFACE NOTE: mine_argument returns (Argument, squeeze_map, ADUs).
        This class expects (SituatedArgument, squeeze_map) pairs.
        The caller must bridge the gap:
            arg, squeeze_map, adus = transducer.mine_argument(premises, claim, name)
            value = transducer.extract_motivational_state(claim, audience_values)[0]
            situated = SituatedArgument(name=name, claim=arg.claim, value=value)
            orchestrator = PipelineOrchestrator([(situated, squeeze_map)], audience)
        This step is where NLI and value-mapping are unified into a single
        SituatedArgument. paramo/scenarios.py (forthcoming) will implement this.
        """
        self.arg_squeeze_pairs = arg_squeeze_pairs
        self.audience          = audience

    def build_solver(self) -> ParaconsistentAFSolver:
        """
        Constructs a ParaconsistentAFSolver with transducer-seeded initial_labels.

        Seeding rules:
            ⊗-keyed atom in squeeze_map  →  initial_label = TruthValue.I
            ⊘-keyed atom in squeeze_map  →  initial_label = TruthValue.F
            no signal                    →  topology seed (I if bonds, else N)

        When BOTH ⊗ and ⊘ signals are present for the same argument
        (classifier detected both dialetheia AND refutation for different
        premises), the ⊗ (dialetheia) takes priority — the argument is
        Both rather than definitively False.
        """
        situated_args = [sa for sa, _ in self.arg_squeeze_pairs]

        # Build the DynamicAF from topology
        af = DynamicVAFGenerator(situated_args).generate_framework(self.audience)

        # Seed initial_labels from squeeze_map signals
        for situated_arg, squeeze_map in self.arg_squeeze_pairs:
            has_dialetheia = any(_is_dialetheia_key(k)  for k in squeeze_map)
            has_refutation = any(_is_refutation_key(k)  for k in squeeze_map)

            if has_dialetheia:
                # ⊗-signal: argument is paraconsistently live — seed at I
                af.initial_labels[situated_arg] = TruthValue.I
            elif has_refutation:
                # ⊘-signal: argument is refuted by its premise — seed at F
                af.initial_labels[situated_arg] = TruthValue.F
            # else: topology seed handles it (unchanged from existing behaviour)

        return ParaconsistentAFSolver(af)
