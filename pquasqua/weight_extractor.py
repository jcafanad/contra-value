"""
pquasqua/weight_extractor.py — per-atom BETO weight computation.

Populates Atom.perplexity (λ_⊥) and Atom.weight for each atom produced
by τ transduction.

Atom.perplexity (λ_⊥)
----------------------
Pseudo-perplexity under the masked LM (dccuchile/bert-base-spanish-wwm-cased).
High perplexity = BETO finds the atom's claim distributionally distant from
its training corpus (Iberian/Chilean Spanish) = measure of epistemic imposition.
Do not recalibrate to reduce perplexity — the distortion is the signal.

Atom.weight — VALUE_NET engagement intensity
---------------------------------------------
Direction-neutral measure of how strongly an atom engages with the totalising
identity gesture of its L_val real abstraction(s). Computed via NLI
(Recognai/bert-base-spanish-wwm-cased-xnli):

    weight = max over L_val(atom) of:
        α · max(p_E, p_C) against theory_hypothesis
      + (1-α) · mean_k max(p_E, p_C) against corpus_prototype_k

max(p_E, p_C) is direction-neutral: high whether the atom performs the
totalising identity gesture (high p_E) or contests it (high p_C). The
audience determines performance vs. contestation; weight measures intensity.
λ_⊥ is stored separately — it measures how reliably BETO can engage with
the claim at all, which is a distinct question from engagement intensity.

VALUE_NET — the value semantic net
-----------------------------------
Four real abstractions, each with a NLI hypothesis encoding its specific
totalising identity gesture under capitalist ontologization:

  value   — presents fetishistic social mediation as a property of things
             or a neutral measure of social wealth
  labour  — ontologises a civilisational canon of submission as the
             universal paradigm of human creative activity, of world-making
  gender  — structurally dissociates the feminine from the sphere of value
             valorisation (the core of commodity-producing patriarchy)
  nature  — positions nature as either passive resource or bearer of
             intrinsic value; both modes deliver it to the value-form

The categories are not parallel instances of a common form but real
abstractions co-constituted within the socio-historical specificity of
capitalist social relations. Their shared kernel is totalising identity:
the drive to render the particular commensurable, the historical eternal.

Symbiotic architecture (ValueNetNode)
--------------------------------------
theory_hypothesis: NLI hypothesis encoding the totalising identity gesture.
    Currently hand-crafted from the critical framework; provisional scaffold.
corpus_prototypes: actual claims from corpus atoms tagged with this label.
    Populated in TODO 2 — corpus speaks; prototypes calibrate the theory.
alpha: blend weight (1.0 = pure theory; decrements per label as corpus grows).
    Symbiosis: theory sets direction; corpus pulls toward actual distribution.

Open questions

  TODO 2 — DONE (populate_corpus_prototypes)
      Parses annotation/*/admin.zip files under a corpus directory. Extracts
      text spans tagged with each L_val label, samples up to
      max_prototypes_per_label per label, and sets alpha using:
          alpha = max(alpha_floor, 1 - n / (n + half_life))
      where n is the full corpus count (not the sample). alpha_floor=0.3
      ensures the theory hypothesis is never fully replaced.

  TODO 3 — DONE (Option C — per-round weight injection)
      ParaconsistentAFSolver now accepts weights: Dict[SituatedArgument, float]
      and weight_threshold: float = 0.5. At each round, arguments with
      weight >= threshold inject I into their own attack_value before the
      negation step. See cubun/af.py class docstring for full political
      consequences and cubun/tests/test_af.py::TestWeightInjection for
      the coalitional bond recovery exemplar.
"""

import math
import os
import random
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pquasqua.transducer import Atom

# XMI namespace constants — mirror transducer.py (duplicated to keep this
# module self-contained; do not import from transducer to avoid coupling).
_CUSTOM_NS = "http:///webanno/custom.ecore"
_CAS_NS    = "http:///uima/cas.ecore"

# torch and transformers are optional — same guard as transducer.py.
try:
    import torch
    import numpy as np
    from transformers import (
        AutoTokenizer,
        AutoModelForMaskedLM,
        AutoModelForSequenceClassification,
    )
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# =============================================================================
# Value semantic net
# =============================================================================

@dataclass
class ValueNetNode:
    """
    A node in the value semantic net.

    theory_hypothesis: NLI hypothesis encoding the totalising identity
        gesture specific to this real abstraction. Provisional scaffold —
        calibrated by corpus_prototypes.
    corpus_prototypes: actual claims from corpus atoms tagged with this
        label. Populated by populate_corpus_prototypes(). Empty list =
        pure theory (alpha=1.0).
    alpha: blend weight between theory and corpus (1.0 = pure theory).
        Set by populate_corpus_prototypes() based on full corpus pool size.
        alpha_floor (default 0.3) prevents full replacement of the theory
        hypothesis — the theory scaffold is never discarded, only calibrated.
    """
    label:             str
    theory_hypothesis: str
    corpus_prototypes: List[str] = field(default_factory=list)
    alpha:             float     = 1.0


VALUE_NET: Dict[str, ValueNetNode] = {
    "value": ValueNetNode(
        label="value",
        theory_hypothesis=(
            "el valor es una propiedad de las cosas"
            " o una medida de la riqueza social"
        ),
    ),
    "labour": ValueNetNode(
        label="labour",
        theory_hypothesis=(
            "el trabajo es la esencia universal"
            " de la actividad creadora humana"
        ),
    ),
    "gender": ValueNetNode(
        label="gender",
        theory_hypothesis=(
            "lo femenino pertenece a una esfera separada"
            " de la producción y acumulación de valor"
        ),
    ),
    "nature": ValueNetNode(
        label="nature",
        theory_hypothesis=(
            "la naturaleza es un recurso disponible"
            " o una portadora de valor en sí misma"
        ),
    ),
}


# =============================================================================
# Corpus prototype population — TODO 2
# =============================================================================

def populate_corpus_prototypes(
    corpus_dir: str,
    value_net: Dict[str, "ValueNetNode"] = None,
    alpha_floor: float = 0.3,
    half_life: int = 50,
    max_prototypes_per_label: int = 20,
    seed: int = 42,
) -> Dict[str, int]:
    """
    Populate ValueNetNode.corpus_prototypes from INCEPTION annotation XMI files
    and update alpha per label.

    Parses annotation/*/admin.zip files under corpus_dir. For each
    custom:Categories span carrying a <value> child element whose text matches
    a VALUE_NET key, the text span (text[begin:end]) is collected as a prototype.

    Alpha decrement schedule
    ------------------------
    alpha reflects the size of the FULL corpus pool (before sampling), so that
    labels with more annotated evidence are calibrated more strongly toward the
    corpus. The sampled prototype pool (max_prototypes_per_label) controls
    computational cost at inference time but does not affect alpha.

        alpha = max(alpha_floor, 1 - n / (n + half_life))

    With alpha_floor=0.3, half_life=50 and observed corpus counts:
        value  (n=207): alpha = 0.30  (floor — large pool, trust corpus)
        nature (n=35):  alpha = 0.59
        labour (n=30):  alpha = 0.63
        gender (n=14):  alpha = 0.78  (small pool, stay close to theory)

    The alpha_floor (default 0.3) ensures the theory hypothesis is never fully
    replaced — it remains as scaffold and anchor even when the corpus is large.

    Parameters
    ----------
    corpus_dir             : path to the INCEPTION project export root
                             (contains annotation/ subdirectory)
    value_net              : VALUE_NET dict to update in place; defaults to the
                             module-level VALUE_NET
    alpha_floor            : minimum alpha (theory floor); default 0.3
    half_life              : full corpus n at which alpha = (1 + alpha_floor)/2;
                             default 50
    max_prototypes_per_label: maximum prototypes stored per label (random sample
                             from full pool); default 20
    seed                   : random seed for reproducible sampling; default 42

    Returns
    -------
    Dict[str, int] : {label: full_corpus_count} for logging and tests.
    """
    if value_net is None:
        value_net = VALUE_NET

    rng = random.Random(seed)
    all_claims: Dict[str, List[str]] = defaultdict(list)

    annotation_dir = os.path.join(corpus_dir, "annotation")
    for speaker_dir in sorted(os.listdir(annotation_dir)):
        zip_path = os.path.join(annotation_dir, speaker_dir, "admin.zip")
        if not os.path.exists(zip_path):
            continue

        with zipfile.ZipFile(zip_path) as zf:
            xmi_bytes = zf.read(zf.namelist()[0])

        root = ET.fromstring(xmi_bytes)

        sofa = root.find(f"{{{_CAS_NS}}}Sofa")
        if sofa is None:
            continue
        text = sofa.get("sofaString", "")

        for cat in root.findall(f"{{{_CUSTOM_NS}}}Categories"):
            value_labels = [el.text for el in cat.findall("value") if el.text]
            if not value_labels:
                continue
            begin = int(cat.get("begin", 0))
            end   = int(cat.get("end",   0))
            claim = text[begin:end].strip()
            if not claim:
                continue
            for label in value_labels:
                if label in value_net:
                    all_claims[label].append(claim)

    full_counts: Dict[str, int] = {}
    for label, claims in all_claims.items():
        node = value_net[label]
        full_n = len(claims)
        full_counts[label] = full_n
        # Sample for computational efficiency at inference time
        if full_n > max_prototypes_per_label:
            claims = rng.sample(claims, max_prototypes_per_label)
        node.corpus_prototypes = claims
        # Alpha based on full corpus evidence, not sample size
        node.alpha = max(alpha_floor, 1.0 - full_n / (full_n + half_life))

    return full_counts


# =============================================================================
# Core computation functions
# =============================================================================

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

    This is λ_⊥: the measure of epistemic imposition. Elevated values
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


def _nli_engagement(
    claim: str,
    hypothesis: str,
    model,
    tokenizer,
) -> float:
    """
    Direction-neutral NLI engagement score: max(p_entailment, p_contradiction).

    Measures how strongly the claim engages with the totalising identity
    gesture encoded in the hypothesis, regardless of whether it performs
    the gesture (high p_E) or contests it (high p_C). The audience
    determines direction; this score measures intensity only.

    XNLI label order for Recognai/bert-base-spanish-wwm-cased-xnli:
        index 0 = contradiction, index 1 = neutral, index 2 = entailment
    """
    encoding = tokenizer(
        claim, hypothesis,
        return_tensors="pt", truncation=True, max_length=512,
    )
    with torch.no_grad():
        logits = model(**encoding).logits
    probs = torch.softmax(logits[0], dim=0)
    p_contradiction = probs[0].item()
    p_entailment    = probs[2].item()
    return max(p_entailment, p_contradiction)


def _compute_atom_weight(
    atom: Atom,
    nli_model,
    nli_tokenizer,
    value_net: Dict[str, ValueNetNode] = VALUE_NET,
) -> float:
    """
    Atom.weight = max over L_val(atom) of the per-label engagement score.

    Per-label score:
        alpha * nli_engagement(claim, theory_hypothesis)
      + (1-alpha) * mean_k nli_engagement(claim, corpus_prototype_k)

    Atoms with no L_val labels or empty claims return 0.0.
    Labels not present in value_net are silently skipped.
    """
    if not atom.claim.strip() or not atom.L_val:
        return 0.0

    scores = []
    for label in atom.L_val:
        node = value_net.get(label)
        if node is None:
            continue

        theory_score = _nli_engagement(
            atom.claim, node.theory_hypothesis, nli_model, nli_tokenizer
        )

        if node.corpus_prototypes:
            corpus_scores = [
                _nli_engagement(atom.claim, p, nli_model, nli_tokenizer)
                for p in node.corpus_prototypes
            ]
            corpus_score = sum(corpus_scores) / len(corpus_scores)
            score = node.alpha * theory_score + (1 - node.alpha) * corpus_score
        else:
            score = theory_score

        scores.append(score)

    return max(scores) if scores else 0.0


# =============================================================================
# Extractor
# =============================================================================

class BETOWeightExtractor:
    """
    Populates Atom.perplexity (λ_⊥) and Atom.weight for a list of atoms.

    Loads two BETO models:
        model_id     — masked LM for pseudo-perplexity (λ_⊥)
        nli_model_id — NLI classifier for VALUE_NET engagement weight

    Usage:
        extractor = BETOWeightExtractor()
        extractor.annotate(atoms)
        # atoms[i].perplexity and atoms[i].weight are now populated
    """

    def __init__(
        self,
        model_id:     str = "dccuchile/bert-base-spanish-wwm-cased",
        nli_model_id: str = "Recognai/bert-base-spanish-wwm-cased-xnli",
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "torch and transformers are required for BETOWeightExtractor. "
                "Install with: pip install torch transformers"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id)
        self.model.eval()

        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_id)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_id)
        self.nli_model.eval()

    def perplexity(self, text: str) -> float:
        """Compute λ_⊥ for a single text string."""
        return _compute_pseudo_perplexity(text, self.model, self.tokenizer)

    def annotate(self, atoms: List[Atom]) -> None:
        """
        Populate Atom.perplexity and Atom.weight for each atom in place.

        Atoms with empty claims are skipped (both values left at 0.0).
        """
        for atom in atoms:
            if not atom.claim.strip():
                continue
            atom.perplexity = _compute_pseudo_perplexity(
                atom.claim, self.model, self.tokenizer
            )
            atom.weight = _compute_atom_weight(
                atom, self.nli_model, self.nli_tokenizer
            )
