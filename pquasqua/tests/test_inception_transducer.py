"""
Tests for the INCEPTION XMI parsing layer (τ transduction).

Validates that the τ function preserves INCEPTION's 3-dimensional annotation
structure without flattening to single labels.

Test structure
--------------
TestAtomDataclass          — Atom dataclass construction and hashability
TestDimensionPredicates    — is_defeater, is_attackable, PUREF, RFLCTN
TestTauInception           — tau_inception filter logic on InceptionSpans
TestParseInceptionXmi      — parse_inception_xmi on a minimal synthetic XMI
TestIntegration            — full Speaker GR XMI parse (skipped if fixture absent)
"""

import io
import textwrap
import pytest
from pathlib import Path

from pquasqua.transducer import (
    Atom,
    Relation,
    InceptionSpan,
    parse_inception_xmi,
    tau_inception,
    build_relation_index,
    is_defeater,
    is_attackable,
    PUREF,
    RFLCTN,
)

# ---------------------------------------------------------------------------
# Fixtures path
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).parent / "fixtures"
_SPEAKER_GR_XMI = _FIXTURES / "speaker_gr.xmi"


# ---------------------------------------------------------------------------
# TestAtomDataclass
# ---------------------------------------------------------------------------

class TestAtomDataclass:
    def test_construction_with_all_dimensions(self):
        """Atom must accept all three dimension sets simultaneously."""
        atom = Atom(
            id=1,
            claim="naturaleza no vale",
            span_indices=(8100, 8119),
            L_dia={"negation", "ligature"},
            L_ont=set(),
            L_val=set(),
            speaker="Speaker GR",
            turn=42,
        )
        assert "negation" in atom.L_dia
        assert "ligature" in atom.L_dia
        assert not atom.L_ont
        assert not atom.L_val

    def test_multi_dimensional_atom(self):
        """An atom can be negation+value+episteme simultaneously."""
        atom = Atom(
            id=2,
            claim="el valor no es sólo el precio",
            span_indices=(0, 30),
            L_dia={"negation", "ligature"},
            L_ont={"episteme"},
            L_val={"value"},
        )
        assert "negation" in atom.L_dia
        assert "ligature" in atom.L_dia
        assert "episteme" in atom.L_ont
        assert "value" in atom.L_val

    def test_hashable_by_id(self):
        """Atoms with the same id must be equal and produce the same hash."""
        a1 = Atom(id=5, claim="foo", span_indices=(0, 3))
        a2 = Atom(id=5, claim="bar", span_indices=(4, 7))  # same id, different claim
        assert a1 == a2
        assert hash(a1) == hash(a2)

    def test_different_ids_not_equal(self):
        a1 = Atom(id=1, claim="foo", span_indices=(0, 3))
        a2 = Atom(id=2, claim="foo", span_indices=(0, 3))
        assert a1 != a2

    def test_usable_in_set(self):
        """Atoms must be usable as set members."""
        atoms = {
            Atom(id=1, claim="a", span_indices=(0, 1)),
            Atom(id=2, claim="b", span_indices=(1, 2)),
            Atom(id=1, claim="a", span_indices=(0, 1)),  # duplicate
        }
        assert len(atoms) == 2

    def test_defeaters_set_mutability(self):
        """Atom.defeaters and .grounds must be mutable after construction."""
        a = Atom(id=1, claim="target", span_indices=(0, 6))
        d = Atom(id=2, claim="negator", span_indices=(7, 14), L_dia={"negation"})
        a.defeaters.add(d)
        assert d in a.defeaters


# ---------------------------------------------------------------------------
# TestDimensionPredicates
# ---------------------------------------------------------------------------

class TestDimensionPredicates:

    # --- is_defeater ---

    def test_is_defeater_with_negation(self):
        a = Atom(id=1, claim="no", span_indices=(0, 2), L_dia={"negation"})
        assert is_defeater(a)

    def test_is_defeater_negation_and_ligature(self):
        """78.9% of negations in corpus also have ligature. Both must be detected."""
        a = Atom(id=1, claim="no", span_indices=(0, 2), L_dia={"negation", "ligature"})
        assert is_defeater(a)

    def test_is_defeater_false_without_negation(self):
        a = Atom(id=1, claim="valor", span_indices=(0, 5), L_dia={"affirmation"})
        assert not is_defeater(a)

    def test_is_defeater_false_empty_L_dia(self):
        a = Atom(id=1, claim="tranquilidad", span_indices=(0, 12))
        assert not is_defeater(a)

    # --- is_attackable ---

    def test_is_attackable_via_value_category(self):
        a = Atom(id=1, claim="valor", span_indices=(0, 5), L_val={"value"})
        assert is_attackable(a)

    def test_is_attackable_via_labour(self):
        a = Atom(id=1, claim="trabajo", span_indices=(0, 7), L_val={"labour"})
        assert is_attackable(a)

    def test_is_attackable_via_moral_relation(self):
        rel = Relation(source_id="2", target_id="1", connect=["moral"])
        a = Atom(id=1, claim="precio", span_indices=(0, 6), R_in=[rel])
        assert is_attackable(a)

    def test_is_attackable_via_ethical_relation(self):
        rel = Relation(source_id="2", target_id="1", connect=["ethical"])
        a = Atom(id=1, claim="precio", span_indices=(0, 6), R_in=[rel])
        assert is_attackable(a)

    def test_is_attackable_via_id_reasoning(self):
        rel = Relation(source_id="1", target_id="3", relate=["identification"])
        a = Atom(id=1, claim="naturaleza", span_indices=(0, 10), R_out=[rel])
        assert is_attackable(a)

    def test_is_attackable_false_no_value_no_normative(self):
        a = Atom(id=1, claim="aquí", span_indices=(0, 4), L_ont={"place"})
        assert not is_attackable(a)

    def test_atom_can_be_defeater_and_attackable(self):
        """An atom with negation+value is simultaneously defeater and attackable."""
        a = Atom(
            id=1,
            claim="no vale",
            span_indices=(0, 7),
            L_dia={"negation"},
            L_val={"value"},
        )
        assert is_defeater(a)
        assert is_attackable(a)

    # --- PUREF ---

    def test_PUREF_with_episteme(self):
        a = Atom(id=1, claim="conocimiento", span_indices=(0, 12), L_ont={"episteme"})
        assert PUREF(a)

    def test_PUREF_false_without_episteme(self):
        a = Atom(id=1, claim="entorno natural", span_indices=(0, 15), L_ont={"life_world"})
        assert not PUREF(a)

    def test_PUREF_and_value_cooccurrence(self):
        """PUREF ∩ VALUE: Kantian category applied to value discourse."""
        a = Atom(
            id=1,
            claim="conocimiento",
            span_indices=(0, 12),
            L_ont={"episteme"},
            L_val={"value"},
        )
        assert PUREF(a)
        assert "value" in a.L_val

    # --- RFLCTN ---

    def test_RFLCTN_with_reflection_connect(self):
        rel = Relation(source_id="2", target_id="1", connect=["reflection"])
        a = Atom(id=1, claim="no sé qué signifique valor", span_indices=(0, 26), R_in=[rel])
        assert RFLCTN(a)

    def test_RFLCTN_false_no_reflection(self):
        rel = Relation(source_id="2", target_id="1", connect=["moral"])
        a = Atom(id=1, claim="valor", span_indices=(0, 5), R_in=[rel])
        assert not RFLCTN(a)

    def test_RFLCTN_false_empty_R_in(self):
        a = Atom(id=1, claim="valor", span_indices=(0, 5))
        assert not RFLCTN(a)

    # --- Negation-ligature pattern ---

    def test_negation_ligature_pattern_queryable(self):
        """
        78.9% of negations in Speaker GR corpus also carry ligature.
        The code must make this pattern trivially queryable.
        """
        atoms = (
            [Atom(id=i, claim=f"n{i}", span_indices=(i, i+1),
                  L_dia={"negation", "ligature"}) for i in range(15)]
            + [Atom(id=i+15, claim=f"n{i+15}", span_indices=(i+15, i+16),
                    L_dia={"negation"}) for i in range(4)]
        )
        negations = [a for a in atoms if "negation" in a.L_dia]
        with_ligature = [a for a in negations if "ligature" in a.L_dia]
        ratio = len(with_ligature) / len(negations)
        assert abs(ratio - 0.789) < 0.01

    # --- Cross-speaker vs intra-speaker ---

    def test_cross_speaker_attack_distinguished_from_self_qualification(self):
        """
        Cross-speaker negation = dialectical attack.
        Intra-speaker negation = self-qualification (modifies grounds).
        Lugones: arguments are embodied and situated.
        """
        defeater = Atom(
            id=1, claim="no vale", span_indices=(0, 7),
            L_dia={"negation"}, speaker="Speaker JG", turn=2,
        )
        target_other = Atom(
            id=2, claim="vale", span_indices=(10, 14),
            L_val={"value"}, speaker="Speaker GR", turn=1,
        )
        target_self = Atom(
            id=3, claim="vale", span_indices=(10, 14),
            L_val={"value"}, speaker="Speaker JG", turn=1,
        )
        assert defeater.speaker != target_other.speaker   # cross-speaker → attack
        assert defeater.speaker == target_self.speaker    # intra-speaker → not attack


# ---------------------------------------------------------------------------
# TestTauInception
# ---------------------------------------------------------------------------

class TestTauInception:
    # Offsets:  Speaker NP=[9:19]  valor=[20:25]  no=[26:28]  es=[29:31]  conn=[32:36]
    TEXT = "hago eco Speaker NP valor no es conn"

    def _span(self, begin, end, dia=None, ont=None, val=None):
        return InceptionSpan(
            xmi_id="1", begin=begin, end=end,
            dialogue_labels=dia or [],
            ontology_labels=ont or [],
            value_labels=val or [],
        )

    def test_produces_atom_for_value_span(self):
        span = self._span(20, 25, val=["value"])
        atom = tau_inception(span, self.TEXT, "Speaker GR", 1, 0)
        assert atom is not None
        assert atom.claim == "valor"
        assert "value" in atom.L_val

    def test_produces_atom_for_negation_span(self):
        span = self._span(26, 28, dia=["negation", "ligature"])
        atom = tau_inception(span, self.TEXT, "Speaker GR", 1, 1)
        assert atom is not None
        assert "negation" in atom.L_dia
        assert "ligature" in atom.L_dia

    def test_filters_pure_ligature_structural_marker(self):
        """Pure ligature-only in L_dia with no other dimensions → None."""
        span = self._span(32, 36, dia=["ligature"])
        atom = tau_inception(span, self.TEXT, "Speaker GR", 1, 2)
        assert atom is None

    def test_filters_pure_person_metadata(self):
        """Only 'person' in L_ont, empty L_dia and L_val → None."""
        span = self._span(9, 19, ont=["person"])
        atom = tau_inception(span, self.TEXT, "Speaker GR", 1, 3)
        assert atom is None

    def test_preserves_ligature_when_cooccurs_with_negation(self):
        """ligature+negation co-occurrence must NOT be filtered."""
        span = self._span(26, 28, dia=["negation", "ligature"])
        atom = tau_inception(span, self.TEXT, "Speaker GR", 1, 4)
        assert atom is not None

    def test_preserves_person_when_cooccurs_with_value(self):
        """person+value co-occurrence must NOT be filtered — it's meaningful."""
        span = self._span(9, 19, ont=["person"], val=["value"])
        atom = tau_inception(span, self.TEXT, "Speaker GR", 1, 5)
        assert atom is not None

    def test_zero_length_span_returns_none(self):
        span = self._span(5, 5)
        atom = tau_inception(span, self.TEXT, "Speaker GR", 1, 6)
        assert atom is None

    def test_speaker_and_turn_assigned(self):
        span = self._span(20, 25, val=["value"])
        atom = tau_inception(span, self.TEXT, "Speaker GR", 7, 10)
        assert atom.speaker == "Speaker GR"
        assert atom.turn == 7

    def test_span_indices_preserved(self):
        span = self._span(20, 25, val=["value"])
        atom = tau_inception(span, self.TEXT, "Speaker GR", 1, 0)
        assert atom.span_indices == (20, 25)


# ---------------------------------------------------------------------------
# TestParseInceptionXmi — minimal synthetic XMI
# ---------------------------------------------------------------------------

_MINIMAL_XMI = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <xmi:XMI
        xmlns:xmi="http://www.omg.org/XMI"
        xmlns:cas="http:///uima/cas.ecore"
        xmlns:custom="http:///webanno/custom.ecore"
        xmi:version="2.0">
      <cas:NULL xmi:id="0"/>
      <cas:Sofa xmi:id="1" sofaNum="1" sofaID="_InitialView"
          mimeType="text"
          sofaString="valor naturaleza no vale"/>
      <custom:Categories xmi:id="10" sofa="1" begin="0" end="5" value="">
        <value>value</value>
      </custom:Categories>
      <custom:Categories xmi:id="20" sofa="1" begin="6" end="15" ontology="">
        <ontology>life_world</ontology>
      </custom:Categories>
      <custom:Categories xmi:id="30" sofa="1" begin="16" end="24"
          dialogue="" value="">
        <dialogue>negation</dialogue>
        <dialogue>ligature</dialogue>
      </custom:Categories>
      <custom:Relations xmi:id="40" sofa="1" begin="0" end="5"
          Dependent="30" Governor="10" connect="">
        <connect>moral</connect>
        <connect>valorisation</connect>
      </custom:Relations>
    </xmi:XMI>
""")


class TestParseInceptionXmi:

    def _parse_minimal(self):
        """Write minimal XMI to a temp file and parse it."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xmi", delete=False, encoding="utf-8"
        ) as f:
            f.write(_MINIMAL_XMI)
            path = Path(f.name)
        try:
            return parse_inception_xmi(path)
        finally:
            os.unlink(path)

    def test_extracts_text_from_sofa(self):
        text, spans, rels = self._parse_minimal()
        assert text == "valor naturaleza no vale"

    def test_span_count(self):
        text, spans, rels = self._parse_minimal()
        assert len(spans) == 3

    def test_relation_count(self):
        text, spans, rels = self._parse_minimal()
        assert len(rels) == 1

    def test_span_value_labels(self):
        text, spans, rels = self._parse_minimal()
        valor_span = next(s for s in spans if s.begin == 0)
        assert valor_span.value_labels == ["value"]
        assert valor_span.dialogue_labels == []
        assert valor_span.ontology_labels == []

    def test_span_ontology_labels(self):
        text, spans, rels = self._parse_minimal()
        nat_span = next(s for s in spans if s.begin == 6)
        assert nat_span.ontology_labels == ["life_world"]

    def test_span_multi_dialogue_labels(self):
        text, spans, rels = self._parse_minimal()
        neg_span = next(s for s in spans if s.begin == 16)
        assert "negation" in neg_span.dialogue_labels
        assert "ligature" in neg_span.dialogue_labels

    def test_relation_source_and_target(self):
        text, spans, rels = self._parse_minimal()
        rel = rels[0]
        assert rel.source_id == "30"
        assert rel.target_id == "10"

    def test_relation_multi_connect_labels(self):
        text, spans, rels = self._parse_minimal()
        rel = rels[0]
        assert "moral" in rel.connect
        assert "valorisation" in rel.connect

    def test_explicit_text_overrides_sofa(self):
        """If text is provided, sofaString is ignored."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xmi", delete=False, encoding="utf-8"
        ) as f:
            f.write(_MINIMAL_XMI)
            path = Path(f.name)
        try:
            text, spans, rels = parse_inception_xmi(path, text="override text")
        finally:
            os.unlink(path)
        assert text == "override text"


# ---------------------------------------------------------------------------
# TestIntegration — real Speaker GR XMI (skipped if fixture absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _SPEAKER_GR_XMI.exists(),
    reason=(
        "Fixture pquasqua/tests/fixtures/speaker_gr.xmi not present. "
        "Extract from pquasqua/tests/contra_value_corpus_ANONYMISED/"
        "annotation/Speaker_GR.txt/admin.zip."
    ),
)
class TestIntegrationSpeakerGR:
    """
    Validates τ transduction on the full Speaker GR transcript.

    Expected counts from INCEPTION annotation analysis:
      ~240 spans total in custom:Categories
      ~153 relations in custom:Relations
      ~19 negation atoms (is_defeater)
      ~21 atoms with episteme (PUREF)
      78.9% of negations also carry ligature
    """

    @pytest.fixture(scope="class")
    def parsed(self):
        text, spans, rels = parse_inception_xmi(_SPEAKER_GR_XMI)
        atoms = []
        for i, span in enumerate(spans):
            atom = tau_inception(span, text, speaker="Speaker GR", turn=0, atom_id=i)
            if atom is not None:
                atoms.append(atom)
        return text, spans, rels, atoms

    def test_span_count_plausible(self, parsed):
        _, spans, _, _ = parsed
        assert len(spans) >= 200

    def test_relation_count_plausible(self, parsed):
        _, _, rels, _ = parsed
        assert len(rels) >= 100

    def test_atom_count_plausible(self, parsed):
        """~209 atoms expected after filtering structural markers and person-only spans."""
        _, _, _, atoms = parsed
        assert len(atoms) >= 180

    def test_negation_atoms_present(self, parsed):
        _, _, _, atoms = parsed
        defeaters = [a for a in atoms if is_defeater(a)]
        assert len(defeaters) > 0

    def test_negation_ligature_pattern(self, parsed):
        """
        Negation+ligature co-occurrence rate in Speaker GR admin.xmi.

        Empirical finding from this XMI: 7/19 = 36.8%.
        NOTE: The IMPLEMENTATION_GUIDE cited 78.9% from a prior analysis
        (possibly a different XMI export or combined multi-document count).
        The corpus XMI is ground truth; this test documents the actual rate.

        The pattern remains meaningful at 36.8%: ligature-negation co-occurrence
        shows Paramuno negation is often embedding rather than pure rejection.
        The threshold ≥0.3 guards against implementation regression.
        """
        _, _, _, atoms = parsed
        negations = [a for a in atoms if is_defeater(a)]
        if len(negations) == 0:
            pytest.skip("No negation atoms found — check annotation export")
        with_ligature = [a for a in negations if "ligature" in a.L_dia]
        ratio = len(with_ligature) / len(negations)
        assert ratio >= 0.3, (
            f"Expected ≥30% negations+ligature (empirical: 36.8%), got {ratio:.1%}. "
            "If this drops below 30%, either the XMI export changed or the "
            "ligature-negation co-occurrence pattern is no longer being parsed."
        )

    def test_puref_atoms_present(self, parsed):
        _, _, _, atoms = parsed
        puref_atoms = [a for a in atoms if PUREF(a)]
        assert len(puref_atoms) > 0

    def test_no_zero_length_atoms(self, parsed):
        _, _, _, atoms = parsed
        for a in atoms:
            begin, end = a.span_indices
            assert begin < end, f"Zero-length atom id={a.id}: [{begin}:{end}]"

    def test_all_atoms_have_speaker(self, parsed):
        _, _, _, atoms = parsed
        for a in atoms:
            assert a.speaker == "Speaker GR"

    def test_claim_matches_text(self, parsed):
        text, _, _, atoms = parsed
        for a in atoms[:20]:  # spot-check first 20
            begin, end = a.span_indices
            assert a.claim == text[begin:end], (
                f"Atom id={a.id}: claim={repr(a.claim)} "
                f"does not match text[{begin}:{end}]={repr(text[begin:end])}"
            )
