"""
pquasqua/weight_extractor.py — per-atom BETO weight computation.

Populates Atom.perplexity (λ_⊥) for each atom produced by τ transduction.
High perplexity = BETO finds the atom's claim distributionally distant from
its training corpus (Iberian/Chilean Spanish) = epistemic imposition signal.

Atom.weight is left at 0.0 pending resolution of three open questions:

  TODO 1 — Semantic mapping, not taxonomy
      The value dimension (L_val: value, labour, gender, nature) is not a
      flat taxonomy but a semantic mapping — closer to a network of relations
      between concepts than to a lookup table. The mapping structure needs
      thinking through before any proximity score can be defined. Key question:
      what does "distance to value" mean when value itself is what is being
      contested?

  TODO 2 — Value descriptions from INCEPTION layers
      If weight is to measure proximity to value concepts, the descriptions
      used for comparison should not be hand-crafted external strings but
      derived from the INCEPTION annotation layers themselves — from atoms
      already tagged with specific L_val labels in the corpus. The corpus
      speaks; the weight descriptions should listen to it. This connects
      TODO 1 (what the mapping looks like) to the empirical material.

  TODO 3 — Weight integration into the PVAF solver
      Atom.weight will be passed to the PVAF (ParaconsistentAFSolver).
      This requires design decisions in cubun/af.py: does weight modulate
      attack strength, influence fixed-point initialisation, or bias the
      Knaster-Tarski iteration? The solver currently ignores weights entirely.
      Weight integration is a theoretical commitment, not a parameter tweak.
"""

import math
from typing import List, Optional

from pquasqua.transducer import Atom

# torch and transformers are optional — same guard as transducer.py.
try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _compute_pseudo_perplexity(
    text: str,
    model,
    tokenizer,
) -> float:
    """
    Pseudo-perplexity for a text string using a masked LM (BETO).

    Approximates perplexity via mean masked-token log-probability:
        pseudo_ppl = exp(-mean_i [ log P(w_i | w_{≠i}) ])

    High pseudo-perplexity = BETO finds the text linguistically distant
    from its training distribution (Iberian/Chilean Spanish).

    This is λ_⊥: the epistemic imposition coefficient. Elevated values
    mark atoms where BETO's colonial baggage distorts the discourse signal.
    Do not recalibrate to reduce perplexity — the distortion is the signal.

    Empirical baseline (2026-03-18):
        Paramuno sample:  10.512
        Iberian sample:    7.118
        Random text:   42584.809
        Ratio P/I:         1.477  (moderate gap — λ_⊥ is a valid signal)
    """
    encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encoding["input_ids"]

    log_probs = []
    for i in range(1, input_ids.shape[1] - 1):
        masked = input_ids.clone()
        masked[0, i] = tokenizer.mask_token_id
        with torch.no_grad():
            logits = model(masked).logits
        token_id = input_ids[0, i].item()
        prob = torch.softmax(logits[0, i], dim=0)[token_id].item()
        log_probs.append(math.log(max(prob, 1e-10)))

    if not log_probs:
        raise ValueError(f"No maskable tokens found in: {repr(text)}")

    return float(np.exp(-np.mean(log_probs)))


class BETOWeightExtractor:
    """
    Populates Atom.perplexity (λ_⊥) for a list of atoms.

    Loads BETO once and computes pseudo-perplexity for each atom's claim.
    Atom.weight is left at 0.0 — see module docstring TODOs 1–3.

    Usage:
        extractor = BETOWeightExtractor()
        extractor.annotate(atoms)
        # atoms[i].perplexity is now populated
    """

    def __init__(
        self,
        model_id: str = "dccuchile/bert-base-spanish-wwm-cased",
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "torch and transformers are required for BETOWeightExtractor. "
                "Install with: pip install torch transformers"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id)
        self.model.eval()

    def perplexity(self, text: str) -> float:
        """Compute λ_⊥ for a single text string."""
        return _compute_pseudo_perplexity(text, self.model, self.tokenizer)

    def annotate(self, atoms: List[Atom]) -> None:
        """
        Populate Atom.perplexity for each atom in place.

        Atoms with empty claims are skipped (perplexity left at 0.0).
        Atom.weight is left at 0.0 — see module docstring TODOs 1–3.
        """
        for atom in atoms:
            if not atom.claim.strip():
                continue
            atom.perplexity = _compute_pseudo_perplexity(
                atom.claim, self.model, self.tokenizer
            )
