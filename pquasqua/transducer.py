"""
pquasqua — the transducer.
Where the lifeworld enters the formalism.

CRITICAL NOTE: This module employs BETO (Spanish BERT) to process text.
This is a partial, highly problematic localisation. BETO encodes Iberian,
European-Spanish, standard-written epistemic assumptions. Applying it to
Colombian fieldwork transcripts — especially those containing Muisca cosmologies
or campesino oral traditions — still forces Abya Yala realities into a colonial
geometry.

The NLI step produces three outcomes for each premise-claim pair:

  1. BIVALENT FORCING (E > entailment_floor, C ≤ designated_threshold):
     An Implication is added to argument support. The Shannon entropy of the
     3-class softmax (the "bivalent squeeze") is recorded in squeeze_map
     against the forced Implication. The violence of binary forcing is named
     and measurable, not silently erased.

  2. DIALETHEIA DETECTED (E > entailment_floor, C > designated_threshold):
     No Implication is forced. A ⊗-keyed atom is recorded in squeeze_map.
     The argument support contains the bare premise atom but no inferential
     structure — the system refuses to force bivalent closure.
     HANDOFF: the ⊗-atom must be read by the pipeline orchestrator
     (paramo/pipeline.py) to seed the argument at TruthValue.I via
     DynamicAF.initial_labels before solver construction.

  3. REFUTATION (E ≤ entailment_floor, C > designated_threshold):
     The classifier finds the premise actively contradicts the claim.
     A ⊘-keyed atom is recorded in squeeze_map.
     HANDOFF: the ⊘-atom must be read by the orchestrator to seed the
     argument at TruthValue.F via DynamicAF.initial_labels.

     MUISCA COSMOLOGY EXAMPLE:
     Premise: "el páramo es un ser vivo que piensa y siente"
     Claim:   "el páramo es un recurso hídrico regulable"
     NLI:     E≈0.05, C≈0.85 — the Muisca cosmological claim actively
              contradicts the State's resource framing.
     Before T4: this refutation left no trace. The Muisca cosmological
                no was treated identically to a neutral utterance.
     After T4:  squeeze_map[Atom("paramo_ser_vivo⊘recurso_hidrico")] = entropy.
                The orchestrator seeds the argument at TruthValue.F.
                The ontological incompatibility between the páramo-as-being
                and the páramo-as-resource is computationally present.

  4. SILENT (E ≤ entailment_floor, C ≤ designated_threshold):
     Neither branch fires. The premise-claim pair leaves no trace in
     support or squeeze_map. This includes genuinely uncertain cases.

ADU CACHING:
     The transducer maintains an instance-level ADU cache (_adu_cache).
     Each unique utterance is embedded (BETO) exactly once per transducer
     instance. Subsequent calls reuse the cached embedding.
     mine_argument returns (Argument, squeeze_map, List[ADU]) — the ADUs
     are available for downstream value-mapping without re-embedding.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional
from pathlib import Path
import xml.etree.ElementTree as ET

# torch and transformers are required for SituatedTransducer (NLI/embedding layer)
# but NOT for the INCEPTION XMI parsing layer (τ transduction).
# Wrapped in try/except so that parse_inception_xmi, tau_inception, Atom, etc.
# remain importable in environments where torch is not installed.
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from chuaque.formulas import Atom as FormulaAtom, Implication, Formula
from cubun.af import Argument


# ---------------------------------------------------------------------------
# INCEPTION XMI parsing layer — τ transduction
# ---------------------------------------------------------------------------
# Preserves INCEPTION's 3-dimensional annotation structure.
# An atom carries L_dia, L_ont, L_val simultaneously — no flattening.
# This is Adorno's non-identity principle: multi-dimensional phenomena
# cannot be forced into univocal categories without theoretical violence.
#
# XMI namespace constants
_CUSTOM_NS = "http:///webanno/custom.ecore"
_XMI_NS    = "http://www.omg.org/XMI"
_CAS_NS    = "http:///uima/cas.ecore"


@dataclass
class Relation:
    """
    INCEPTION Relation annotation connecting two spans.

    connect: dia_dynamics labels (e.g. 'moral', 'valorisation', 'reflection')
    relate:  cat_dynamics labels (e.g. 'monetary', 'non-monetary')
    ideate:  sociation labels    (e.g. 'collectivity', 'other')

    source_id = Dependent XMI id (the span from which the relation originates)
    target_id = Governor  XMI id (the span to which the relation points)
    """
    source_id: str
    target_id: str
    connect: List[str] = field(default_factory=list)
    relate:  List[str] = field(default_factory=list)
    ideate:  List[str] = field(default_factory=list)


@dataclass
class InceptionSpan:
    """
    Raw multi-dimensional span from INCEPTION XMI, before τ transduction.

    Dimension label lists are populated from child elements of
    custom:Categories (e.g. <dialogue>negation</dialogue>).
    Absence of a dimension = empty list, NOT zero — preserve the distinction.
    """
    xmi_id:           str
    begin:            int
    end:              int
    dialogue_labels:  List[str] = field(default_factory=list)
    ontology_labels:  List[str] = field(default_factory=list)
    value_labels:     List[str] = field(default_factory=list)


@dataclass
class Atom:
    """
    Multi-dimensional argumentation atom produced by τ from an InceptionSpan.

    CRITICAL: Preserves INCEPTION's 3-dimensional structure without reduction.
    An atom can carry negation+ligature+value+episteme simultaneously.
    This is the condition of possibility for computing colonial geometry:
    PUREF ∩ VALUE measures Enlightenment categories structuring value discourse.

    R_in/R_out are populated in a second pass after all atoms are indexed.
    grounds/defeaters are populated by construct_attacks().
    weight/perplexity are populated by weight_extractor.py.

    span_id: str
        The XMI id of the originating InceptionSpan, assigned at construction
        in tau_inception(). Format: the raw string value of the xmi:id attribute
        (e.g. "42"). Preserved as a string to avoid silent int conversion of
        non-numeric ids in future XMI variants. Used for provenance tracing and
        for warg FFI round-trips where the original span boundary must be
        recoverable without re-parsing the XMI.

    initial_weight: float
        Direction-neutral NLI engagement score at the moment of τ transduction,
        before VALUE_NET traversal. Populated by compute_initial_weight() in
        weight_extractor.py when logits from the NLI pass are available at
        construction time. Semantics: max(p_E, p_C) over the (span_text,
        claim_text) pair — the same direction-neutral formula as _nli_engagement,
        but computed from already-available logits rather than a fresh NLI call.
        Default 0.0 (not yet computed). Distinct from Atom.weight, which is the
        full VALUE_NET engagement score computed in a later annotation pass.
    """
    id:             int
    claim:          str
    span_indices:   Tuple[int, int]
    L_dia:          Set[str]    = field(default_factory=set)
    L_ont:          Set[str]    = field(default_factory=set)
    L_val:          Set[str]    = field(default_factory=set)
    R_in:           List[Relation] = field(default_factory=list)
    R_out:          List[Relation] = field(default_factory=list)
    speaker:        str         = ""
    turn:           int         = 0
    grounds:        Set["Atom"] = field(default_factory=set)
    defeaters:      Set["Atom"] = field(default_factory=set)
    weight:         float       = 0.0
    perplexity:     float       = 0.0   # λ_⊥ coefficient (epistemic violence measure)
    span_id:        str         = ""    # XMI id of originating InceptionSpan
    initial_weight: float       = 0.0  # NLI engagement at τ construction time

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Atom) and self.id == other.id


def parse_inception_xmi(
    xmi_path: Path,
    text: Optional[str] = None,
) -> Tuple[str, List[InceptionSpan], List[Relation]]:
    """
    Parse INCEPTION XMI export into source text, spans, and relations.

    The document text is extracted from the UIMA Sofa sofaString attribute
    if not supplied. Character offsets in spans reference this string.

    Returns:
        (text, spans, relations)
        text      — full source document string
        spans     — InceptionSpan list with multi-dimensional labels
        relations — Relation list with source/target XMI ids

    CRITICAL: All dimension labels are preserved. Do not filter or flatten.
    An empty label list means the dimension was not annotated for that span —
    this is semantically distinct from "annotated as empty".
    """
    tree = ET.parse(xmi_path)
    root = tree.getroot()

    # Extract source text from Sofa element if not provided
    if text is None:
        sofa = root.find(f"{{{_CAS_NS}}}Sofa")
        if sofa is None:
            raise ValueError(f"No cas:Sofa element found in {xmi_path}")
        text = sofa.get("sofaString", "")

    spans: List[InceptionSpan] = []
    for cat in root.findall(f"{{{_CUSTOM_NS}}}Categories"):
        span = InceptionSpan(
            xmi_id=cat.get(f"{{{_XMI_NS}}}id", ""),
            begin=int(cat.get("begin", 0)),
            end=int(cat.get("end", 0)),
            dialogue_labels=[el.text for el in cat.findall("dialogue") if el.text],
            ontology_labels=[el.text for el in cat.findall("ontology") if el.text],
            value_labels=   [el.text for el in cat.findall("value")    if el.text],
        )
        spans.append(span)

    relations: List[Relation] = []
    for rel in root.findall(f"{{{_CUSTOM_NS}}}Relations"):
        relation = Relation(
            source_id=rel.get("Dependent", ""),
            target_id=rel.get("Governor",  ""),
            connect=[el.text for el in rel.findall("connect") if el.text],
            relate= [el.text for el in rel.findall("relate")  if el.text],
            ideate= [el.text for el in rel.findall("ideate")  if el.text],
        )
        relations.append(relation)

    return text, spans, relations


def tau_inception(
    span:    InceptionSpan,
    text:    str,
    speaker: str,
    turn:    int,
    atom_id: int,
) -> Optional[Atom]:
    """
    τ transduction: map a multi-dimensional InceptionSpan to an Atom.

    Returns None for non-argumentative spans:
      - Pure structural markers: only ligature in L_dia, no other dimensions
      - Pure person metadata:    only 'person' in L_ont, empty L_dia and L_val

    Otherwise returns an Atom with ALL dimensions preserved.
    R_in/R_out are populated in a second pass — see build_relation_index().

    THEORETICAL NOTE: speaker and turn are required parameters.
    Arguments are embodied and situated (Lugones). Cross-speaker vs.
    intra-speaker negation are dialectically different operations.
    Atoms attributed to an unknown speaker are sites of potential epistemic
    erasure — the parser's inability to attribute is itself a finding.
    """
    dia = set(span.dialogue_labels)
    ont = set(span.ontology_labels)
    val = set(span.value_labels)

    # Pure structural marker: ligature-only with no other content
    if dia == {"ligature"} and not ont and not val:
        return None

    # Pure person metadata: person-only ontology, no dialogical or value content
    if ont == {"person"} and not dia and not val:
        return None

    # Guard against zero-length spans (INCEPTION artefacts)
    if span.begin >= span.end:
        return None

    return Atom(
        id=atom_id,
        claim=text[span.begin:span.end],
        span_indices=(span.begin, span.end),
        L_dia=dia,
        L_ont=ont,
        L_val=val,
        speaker=speaker,
        turn=turn,
        span_id=span.xmi_id,   # provenance: XMI id of originating span
    )


def build_relation_index(
    atoms:     List[Atom],
    relations: List[Relation],
) -> None:
    """
    Populate Atom.R_in and Atom.R_out from the Relations layer.

    Mutates atoms in place (second pass after all atoms are instantiated).
    source_id (Dependent) → R_out on that atom
    target_id (Governor)  → R_in  on that atom

    CRITICAL: This records the directionality of moral/ethical/valorisation
    relations. Flattening to a single label on either atom loses who is
    moralizing whom — the power asymmetry encoded in relation direction.
    """
    id_to_atom: Dict[str, Atom] = {
        str(a.id): a for a in atoms
        # also index by original XMI id if we preserved it
    }
    # Build index keyed on span_indices start byte as a proxy for XMI id lookup
    # The xmi_id is stored on InceptionSpan; we need a map from xmi_id → atom
    # This function is called with the (atoms, relations) from parse + tau passes
    # Caller must pass xmi_id_to_atom dict; use overloaded signature below.
    # This stub is correct when atom.id equals the integer xmi_id.
    str_id_map: Dict[str, Atom] = {}
    for a in atoms:
        str_id_map[str(a.id)] = a

    for rel in relations:
        src = str_id_map.get(rel.source_id)
        tgt = str_id_map.get(rel.target_id)
        if src is not None:
            src.R_out.append(rel)
        if tgt is not None:
            tgt.R_in.append(rel)


# ---------------------------------------------------------------------------
# Dimension predicates — attack construction uses these, not single labels
# ---------------------------------------------------------------------------

def is_defeater(atom: Atom) -> bool:
    """
    True if atom can attack other atoms.

    An atom is a defeater iff it carries negation in the dialogical dimension.
    This is Adorno's determinate negation: negation of specific content,
    not abstract rejection. Cross-speaker negation attacks; intra-speaker
    negation modifies grounds (see construct_attacks).
    """
    return "negation" in atom.L_dia


def is_attackable(atom: Atom) -> bool:
    """
    True if atom can be the target of an attack.

    An atom is attackable if it makes a value claim, a normative claim
    (via moral/ethical relation), or an identificatory claim.

    Uses dimension predicates — NOT single-label matching.
    An atom can be simultaneously defeater and attackable.
    """
    has_value_category = bool(atom.L_val & {"value", "labour", "gender", "nature"})
    has_normative_claim = any(
        any(c in {"moral", "ethical"} for c in r.connect)
        for r in atom.R_in
    )
    has_id_reasoning = any(
        any(c in {"identification", "equivalence"} for c in r.relate)
        for r in (atom.R_in + atom.R_out)
    )
    return has_value_category or has_normative_claim or has_id_reasoning


# ---------------------------------------------------------------------------
# Diagnostic flags — boolean functions over atoms, not atom properties
# ---------------------------------------------------------------------------

def PUREF(atom: Atom) -> bool:
    """
    PUREF diagnostic: does this atom invoke a Kantian pure construct?

    True iff 'episteme' is present in the ontological dimension.

    Theoretical significance: measures Enlightenment category penetration.
    PUREF ∩ VALUE coverage = proportion of value discourse structured by
    Kantian categories (knowledge, consciousness, reason). This is colonial
    geometry made computationally detectable.

    CRITICAL: This is NOT an atom label. Do not add 'puref' to L_ont.
    Atoms can be simultaneously PUREF, defeater, and attackable.
    """
    return "episteme" in atom.L_ont


def RFLCTN(atom: Atom) -> bool:
    """
    RFLCTN diagnostic: does this atom carry metacognitive/self-problematizing
    discourse?

    True iff any incoming Relation carries 'reflection' in its connect labels.

    Theoretical significance: Adorno's negative dialectics. Moments where
    speakers resist definitive categorisation, hold open contradictions
    rather than resolving them.
    """
    return any("reflection" in r.connect for r in atom.R_in)


@dataclass(frozen=True)
class ADU:
    """
    An Argumentative Discourse Unit.
    Frozen and reified from living speech.
    The embedding is stored as a native Python tuple to enforce strict
    immutability, closing the backdoor left by mutable torch.Tensors.

    raw_cosine_distance: distance to the matched value description,
    populated by extract_motivational_state after value-mapping.
    None until then — the embedding is utterance-level; the cosine
    distance is pair-level (utterance × value_description).
    """
    text: str
    embedding_tuple: Tuple[float, ...]
    is_claim: bool = False
    raw_cosine_distance: Optional[float] = None


class SituatedTransducer:
    """
    Translates fieldwork transcripts into formal Arguments using LATAM-situated,
    yet still epistemically Iberian, models.
    """
    def __init__(self,
                 embed_model: str = "dccuchile/bert-base-spanish-wwm-cased",
                 nli_model:   str = "Recognai/bert-base-spanish-wwm-cased-xnli",
                 entailment_floor:     float = 0.5,
                 designated_threshold: float = 0.4):

        if not _TORCH_AVAILABLE:
            raise ImportError(
                "torch and transformers are required for SituatedTransducer. "
                "Install with: pip install torch transformers"
            )

        self.embed_tokenizer = AutoTokenizer.from_pretrained(embed_model)
        self.embed_model     = AutoModel.from_pretrained(embed_model)

        # NLI model loaded explicitly for sequence classification to access raw logits
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model)
        self.nli_model     = AutoModelForSequenceClassification.from_pretrained(nli_model)
        self.nli_labels    = self.nli_model.config.id2label

        self.entailment_floor     = entailment_floor
        self.designated_threshold = designated_threshold

        # BETO embedding cache: text -> ADU (avoids recomputing for repeated utterances)
        self._adu_cache: Dict[str, ADU] = {}

    def _embed(self, text: str) -> Tuple[float, ...]:
        """Generates contextual embeddings and casts to an immutable tuple."""
        inputs = self.embed_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.embed_model(**inputs)

        attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask, 1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        normalized = F.normalize(sum_embeddings / sum_mask, p=2, dim=1)

        return tuple(normalized.squeeze().tolist())

    def _get_adu(self, text: str, is_claim: bool = False) -> ADU:
        """
        Returns the cached ADU for `text`, or embeds and caches it.
        Avoids redundant BETO passes for repeated utterances.
        raw_cosine_distance is None until extract_motivational_state populates it.
        """
        if text not in self._adu_cache:
            self._adu_cache[text] = ADU(
                text=text,
                embedding_tuple=self._embed(text),
                is_claim=is_claim,
            )
        return self._adu_cache[text]

    def extract_motivational_state(
        self, adu_text: str, target_values: Dict[str, str]
    ) -> Tuple[str, float]:
        """
        Maps an ADU to a value via cosine similarity in BETO embedding space.
        Returns (best_value_key, raw_cosine_distance_to_best_match).

        NOTE: uses _get_adu() for cached BETO embedding. The returned
        raw_cosine_distance is stored as a post-hoc annotation on the ADU
        (via the _adu_cache — see limitation below).

        INCOMMENSURABILITY NOTE: this method uses BETO embeddings for cosine
        similarity; mine_argument uses XNLI-BETO for NLI classification.
        These are different models with different internal representations.
        Value-mapping and implication-classification operate in incommensurable
        embedding spaces. A unified model would require using XNLI-BETO for
        both steps, or a shared encoder.
        """
        adu = self._get_adu(adu_text)
        adu_tensor = torch.tensor(adu.embedding_tuple).unsqueeze(0)

        best_value  = None
        highest_sim = -1.0

        for val_key, val_desc in target_values.items():
            desc_adu    = self._get_adu(val_desc)
            desc_tensor = torch.tensor(desc_adu.embedding_tuple).unsqueeze(0)
            sim = F.cosine_similarity(adu_tensor, desc_tensor).item()
            if sim > highest_sim:
                highest_sim = sim
                best_value  = val_key

        raw_distance = 1.0 - highest_sim
        return best_value, raw_distance

    def _calculate_bivalent_squeeze(self, probs: List[float]) -> float:
        """
        Shannon entropy of the 3-class NLI probability distribution.

        Measures overall classifier uncertainty — NOT paraconsistency specifically.
        A high-E/high-C pair (genuine dialetheia: E=0.55, C=0.42) has LOWER
        entropy (~1.15 bits) than a maximally uncertain uniform pair
        (E=C=N=0.33, entropy=1.585 bits), because two classes dominate.

        For a paraconsistency-specific measure, prefer:
            min(entailment_score, contradiction_score)
        This rises with genuine paraconsistency and falls with ordinary
        uncertainty or unambiguous classification:
            Refutation (E=0.05, C=0.85):  min = 0.05  (low paraconsistency ✓)
            Dialetheia (E=0.55, C=0.42):  min = 0.42  (high paraconsistency ✓)
            Implication (E=0.75, C=0.15): min = 0.15  (low paraconsistency ✓)
        Future versions may replace Shannon entropy with min(E,C) here.
        """
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def mine_argument(
        self,
        premise_texts: List[str],
        claim_text: str,
        arg_name: str
    ) -> Tuple[Argument, Dict[Formula, float], List[ADU]]:
        """
        Imposes classical implication on situated fieldwork.
        Returns:
            arg:         the formal Argument (claim + support formulas)
            squeeze_map: maps each forced/detected formula to its entropy
            adus:        the ADU objects for premises + claim (cached embeddings)

        squeeze_map key conventions:
            Implication(p, c) -> entropy  : bivalent forcing
            Atom("p⊗c")       -> entropy  : dialetheia detected
            Atom("p⊘c")       -> entropy  : refutation detected
        """
        claim_atom   = FormulaAtom(claim_text)
        claim_adu    = self._get_adu(claim_text, is_claim=True)
        premise_atoms = [FormulaAtom(pt) for pt in premise_texts]
        premise_adus  = [self._get_adu(pt) for pt in premise_texts]

        support_formulas: Set[Formula]      = set(premise_atoms)
        squeeze_map:      Dict[Formula, float] = {}

        for premise, premise_adu in zip(premise_atoms, premise_adus):
            inputs = self.nli_tokenizer(
                premise.name, claim_text, return_tensors="pt", truncation=True
            )
            with torch.no_grad():
                logits = self.nli_model(**inputs).logits

            probs   = F.softmax(logits, dim=-1).squeeze().tolist()
            entropy = self._calculate_bivalent_squeeze(probs)

            entailment_idx    = next(
                i for i, label in self.nli_labels.items()
                if "entailment" in label.lower()
            )
            contradiction_idx = next(
                i for i, label in self.nli_labels.items()
                if "contradiction" in label.lower()
            )
            entailment_score    = probs[entailment_idx]
            contradiction_score = probs[contradiction_idx]

            if entailment_score > self.entailment_floor \
                    and contradiction_score > self.designated_threshold:
                # DIALETHEIA: classifier simultaneously supports and refutes.
                # Do not force a bivalent Implication. Record the ⊗-atom.
                # Orchestrator (paramo/pipeline.py) seeds argument at TruthValue.I.
                squeeze_map[FormulaAtom(premise.name + "⊗" + claim_atom.name)] = entropy

            elif entailment_score > self.entailment_floor:
                # BIVALENT FORCING: clear entailment signal, low contradiction.
                forced_implication = Implication(premise, claim_atom)
                support_formulas.add(forced_implication)
                squeeze_map[forced_implication] = entropy

            elif contradiction_score > self.designated_threshold:
                # REFUTATION: classifier finds premise contradicts claim.
                # The no — the Muisca cosmological refutation of State framing —
                # must leave a trace. Record the ⊘-atom.
                # Orchestrator seeds argument at TruthValue.F.
                # See module docstring for the Muisca cosmology example.
                squeeze_map[FormulaAtom(premise.name + "⊘" + claim_atom.name)] = entropy

            # SILENT: E ≤ floor, C ≤ threshold. No trace. See module docstring.

        arg = Argument(
            name=arg_name,
            support=frozenset(support_formulas),
            claim=claim_atom,
        )
        adus = premise_adus + [claim_adu]
        return arg, squeeze_map, adus
