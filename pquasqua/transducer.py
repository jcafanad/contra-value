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
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional

from chuaque.formulas import Atom, Implication, Formula
from cubun.af import Argument


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
        claim_atom   = Atom(claim_text)
        claim_adu    = self._get_adu(claim_text, is_claim=True)
        premise_atoms = [Atom(pt) for pt in premise_texts]
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
                squeeze_map[Atom(premise.name + "⊗" + claim_atom.name)] = entropy

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
                squeeze_map[Atom(premise.name + "⊘" + claim_atom.name)] = entropy

            # SILENT: E ≤ floor, C ≤ threshold. No trace. See module docstring.

        arg = Argument(
            name=arg_name,
            support=frozenset(support_formulas),
            claim=claim_atom,
        )
        adus = premise_adus + [claim_adu]
        return arg, squeeze_map, adus
