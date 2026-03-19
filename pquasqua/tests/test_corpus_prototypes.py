"""
pquasqua/tests/test_corpus_prototypes.py — tests for populate_corpus_prototypes.

Torch-free. Two groups:

  TestAlphaFormula    — unit tests for the alpha decrement schedule and
                        prototype sampling logic using synthetic XMI in
                        temporary directories. No corpus required.

  TestCorpusIntegration — integration tests against the committed corpus
                          (pquasqua/tests/contra_value_corpus_ANONYMISED/).
                          Skipped automatically if the corpus directory is absent.
"""

import os
import zipfile
import copy

import pytest

from pquasqua.weight_extractor import (
    populate_corpus_prototypes,
    VALUE_NET,
    ValueNetNode,
)

# ---------------------------------------------------------------------------
# Path to the committed anonymised corpus
# ---------------------------------------------------------------------------

_HERE       = os.path.dirname(__file__)
_CORPUS_DIR = os.path.join(_HERE, "contra_value_corpus_ANONYMISED")
_CORPUS_PRESENT = os.path.isdir(os.path.join(_CORPUS_DIR, "annotation"))

_skip_no_corpus = pytest.mark.skipif(
    not _CORPUS_PRESENT,
    reason="contra_value_corpus_ANONYMISED not present",
)

# ---------------------------------------------------------------------------
# XMI helpers for synthetic fixtures
# ---------------------------------------------------------------------------

_XMI_NS = {
    "cas":    "http:///uima/cas.ecore",
    "custom": "http:///webanno/custom.ecore",
}

def _make_xmi(text: str, spans: list[tuple[int, int, list[str]]]) -> bytes:
    """
    Build a minimal INCEPTION-style XMI byte string.

    text  : sofaString value
    spans : list of (begin, end, [label, ...]) tuples
    """
    cat_elements = []
    for begin, end, labels in spans:
        label_elems = "".join(f"<value>{lbl}</value>" for lbl in labels)
        cat_elements.append(
            f'<custom:Categories xmlns:custom="{_XMI_NS["custom"]}"'
            f' begin="{begin}" end="{end}">{label_elems}</custom:Categories>'
        )
    cats = "\n  ".join(cat_elements)
    # NOTE: no leading whitespace — ET.fromstring requires <?xml at column 0.
    xmi = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<xmi:XMI xmlns:xmi="http://www.omg.org/XMI"'
        f' xmlns:cas="{_XMI_NS["cas"]}"'
        f' xmlns:custom="{_XMI_NS["custom"]}">\n'
        f'  <cas:Sofa sofaNum="1" sofaID="_InitialView" mimeType="text"'
        f' sofaString="{text}"/>\n'
        f'  {cats}\n'
        f'</xmi:XMI>\n'
    ).encode("utf-8")
    return xmi


def _make_corpus_dir(tmp_path, xmi_bytes: bytes, speaker: str = "Speaker_A") -> str:
    """
    Write a single admin.zip inside tmp_path/annotation/<speaker>/ and
    return tmp_path as the corpus_dir string.
    """
    ann_dir = tmp_path / "annotation" / speaker
    ann_dir.mkdir(parents=True)
    zip_path = ann_dir / "admin.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("document.xmi", xmi_bytes)
    return str(tmp_path)


# ---------------------------------------------------------------------------
# TestAlphaFormula — unit-level; no real corpus needed
# ---------------------------------------------------------------------------

class TestAlphaFormula:
    """Alpha decrement schedule and prototype sampling with synthetic XMI."""

    def _fresh_net(self):
        """Deep-copy VALUE_NET so tests don't share state."""
        return {k: copy.copy(v) for k, v in VALUE_NET.items()}

    # ------------------------------------------------------------------
    # Alpha schedule
    # ------------------------------------------------------------------

    def test_alpha_formula_small_n(self, tmp_path):
        """Small corpus (n=10, half_life=50) → alpha close to 1."""
        text = "abcdefghij"
        # 10 spans of length 1 each, all tagged value
        spans = [(i, i + 1, ["value"]) for i in range(10)]
        corpus_dir = _make_corpus_dir(tmp_path, _make_xmi(text, spans))
        net = self._fresh_net()
        counts = populate_corpus_prototypes(corpus_dir, value_net=net, half_life=50)
        assert counts["value"] == 10
        expected = max(0.3, 1 - 10 / (10 + 50))
        assert abs(net["value"].alpha - expected) < 1e-9

    def test_alpha_formula_large_n_hits_floor(self, tmp_path):
        """Large corpus (n=207) → alpha = floor (0.3)."""
        text = "x" * 207
        spans = [(i, i + 1, ["value"]) for i in range(207)]
        corpus_dir = _make_corpus_dir(tmp_path, _make_xmi(text, spans))
        net = self._fresh_net()
        counts = populate_corpus_prototypes(corpus_dir, value_net=net, half_life=50)
        assert counts["value"] == 207
        assert net["value"].alpha == pytest.approx(0.3)

    def test_alpha_floor_respected(self, tmp_path):
        """Custom alpha_floor is honoured even for very large n."""
        text = "x" * 500
        spans = [(i, i + 1, ["nature"]) for i in range(500)]
        corpus_dir = _make_corpus_dir(tmp_path, _make_xmi(text, spans))
        net = self._fresh_net()
        populate_corpus_prototypes(
            corpus_dir, value_net=net, alpha_floor=0.2, half_life=50
        )
        assert net["nature"].alpha >= 0.2

    def test_alpha_based_on_full_count_not_sample(self, tmp_path):
        """Alpha reflects full corpus count even when sample is capped."""
        text = "x" * 100
        spans = [(i, i + 1, ["labour"]) for i in range(100)]
        corpus_dir = _make_corpus_dir(tmp_path, _make_xmi(text, spans))
        net = self._fresh_net()
        populate_corpus_prototypes(
            corpus_dir, value_net=net,
            max_prototypes_per_label=10,  # sample only 10
            half_life=50,
        )
        # alpha based on n=100, not n=10
        expected = max(0.3, 1 - 100 / (100 + 50))
        assert abs(net["labour"].alpha - expected) < 1e-9
        # but only 10 prototypes stored
        assert len(net["labour"].corpus_prototypes) == 10

    # ------------------------------------------------------------------
    # Prototype sampling
    # ------------------------------------------------------------------

    def test_prototypes_stored_when_under_cap(self, tmp_path):
        """When n < max_prototypes_per_label, all claims are stored."""
        text = "aaa bbb ccc"
        spans = [(0, 3, ["nature"]), (4, 7, ["nature"]), (8, 11, ["nature"])]
        corpus_dir = _make_corpus_dir(tmp_path, _make_xmi(text, spans))
        net = self._fresh_net()
        populate_corpus_prototypes(corpus_dir, value_net=net, max_prototypes_per_label=20)
        assert len(net["nature"].corpus_prototypes) == 3
        assert set(net["nature"].corpus_prototypes) == {"aaa", "bbb", "ccc"}

    def test_prototypes_capped_at_max(self, tmp_path):
        """When n > max_prototypes_per_label, prototypes are sampled."""
        text = "x" * 50
        spans = [(i, i + 1, ["value"]) for i in range(50)]
        corpus_dir = _make_corpus_dir(tmp_path, _make_xmi(text, spans))
        net = self._fresh_net()
        populate_corpus_prototypes(
            corpus_dir, value_net=net, max_prototypes_per_label=10
        )
        assert len(net["value"].corpus_prototypes) == 10

    def test_sampling_reproducible_same_seed(self, tmp_path):
        """Same seed → identical prototype list."""
        text = " ".join(f"w{i:03d}" for i in range(50))
        # 50 non-overlapping 4-char spans: "w000", "w001", ...
        spans = [(i * 5, i * 5 + 4, ["value"]) for i in range(50)]
        corpus_dir = _make_corpus_dir(tmp_path, _make_xmi(text, spans))
        net_a = self._fresh_net()
        net_b = self._fresh_net()
        populate_corpus_prototypes(
            corpus_dir, value_net=net_a, max_prototypes_per_label=15, seed=99
        )
        populate_corpus_prototypes(
            corpus_dir, value_net=net_b, max_prototypes_per_label=15, seed=99
        )
        assert net_a["value"].corpus_prototypes == net_b["value"].corpus_prototypes

    def test_sampling_differs_different_seed(self, tmp_path):
        """Different seeds likely produce different prototype lists (probabilistic)."""
        text = " ".join(f"w{i:03d}" for i in range(50))
        spans = [(i * 5, i * 5 + 4, ["value"]) for i in range(50)]
        corpus_dir = _make_corpus_dir(tmp_path, _make_xmi(text, spans))
        net_a = self._fresh_net()
        net_b = self._fresh_net()
        populate_corpus_prototypes(
            corpus_dir, value_net=net_a, max_prototypes_per_label=15, seed=1
        )
        populate_corpus_prototypes(
            corpus_dir, value_net=net_b, max_prototypes_per_label=15, seed=2
        )
        # With 50 items sampled to 15, collision probability is negligible
        assert net_a["value"].corpus_prototypes != net_b["value"].corpus_prototypes

    # ------------------------------------------------------------------
    # Label routing
    # ------------------------------------------------------------------

    def test_multi_label_span_routed_to_both(self, tmp_path):
        """A span tagged with two labels contributes to both nodes."""
        text = "hello world"
        spans = [(0, 5, ["value", "nature"])]
        corpus_dir = _make_corpus_dir(tmp_path, _make_xmi(text, spans))
        net = self._fresh_net()
        counts = populate_corpus_prototypes(corpus_dir, value_net=net)
        assert "hello" in net["value"].corpus_prototypes
        assert "hello" in net["nature"].corpus_prototypes

    def test_unknown_label_silently_skipped(self, tmp_path):
        """Labels not in VALUE_NET do not raise and are not stored."""
        text = "hello"
        spans = [(0, 5, ["unknown_category"])]
        corpus_dir = _make_corpus_dir(tmp_path, _make_xmi(text, spans))
        net = self._fresh_net()
        counts = populate_corpus_prototypes(corpus_dir, value_net=net)
        # No prototypes added to any node
        for node in net.values():
            assert node.corpus_prototypes == []

    def test_empty_claim_skipped(self, tmp_path):
        """Spans that resolve to empty/whitespace strings are skipped."""
        text = "   "
        spans = [(0, 3, ["value"])]  # whitespace only
        corpus_dir = _make_corpus_dir(tmp_path, _make_xmi(text, spans))
        net = self._fresh_net()
        populate_corpus_prototypes(corpus_dir, value_net=net)
        assert net["value"].corpus_prototypes == []

    def test_missing_admin_zip_skipped(self, tmp_path):
        """Speaker directories without admin.zip are silently skipped."""
        ann_dir = tmp_path / "annotation" / "Speaker_Empty"
        ann_dir.mkdir(parents=True)
        # no admin.zip
        net = self._fresh_net()
        counts = populate_corpus_prototypes(str(tmp_path), value_net=net)
        assert counts == {}

    def test_returns_full_counts_not_sample_counts(self, tmp_path):
        """Return value reflects full corpus count, not sample size."""
        text = "x" * 30
        spans = [(i, i + 1, ["gender"]) for i in range(30)]
        corpus_dir = _make_corpus_dir(tmp_path, _make_xmi(text, spans))
        net = self._fresh_net()
        counts = populate_corpus_prototypes(
            corpus_dir, value_net=net, max_prototypes_per_label=5
        )
        assert counts["gender"] == 30

    # ------------------------------------------------------------------
    # Multi-speaker accumulation
    # ------------------------------------------------------------------

    def test_multi_speaker_accumulates(self, tmp_path):
        """Claims from multiple speaker admin.zip files are pooled."""
        ann_dir = tmp_path / "annotation"

        def add_speaker(name, text, spans):
            d = ann_dir / name
            d.mkdir(parents=True)
            with zipfile.ZipFile(d / "admin.zip", "w") as zf:
                zf.writestr("doc.xmi", _make_xmi(text, spans))

        add_speaker("Speaker_A", "hello", [(0, 5, ["nature"])])
        add_speaker("Speaker_B", "world", [(0, 5, ["nature"])])

        net = self._fresh_net()
        counts = populate_corpus_prototypes(str(tmp_path), value_net=net)
        assert counts["nature"] == 2
        assert set(net["nature"].corpus_prototypes) == {"hello", "world"}


# ---------------------------------------------------------------------------
# TestCorpusIntegration — real annotated corpus
# ---------------------------------------------------------------------------

@_skip_no_corpus
class TestCorpusIntegration:
    """
    Integration tests against pquasqua/tests/contra_value_corpus_ANONYMISED/.

    Expected counts (from empirical run 2026-03-18):
        value  ~207   gender ~14
        nature  ~35   labour ~30

    These are lower bounds — any future annotations can only increase them.
    """

    def _run(self, **kwargs):
        import copy as _copy
        net = {k: _copy.copy(v) for k, v in VALUE_NET.items()}
        counts = populate_corpus_prototypes(_CORPUS_DIR, value_net=net, **kwargs)
        return net, counts

    def test_all_four_labels_present(self):
        """All four VALUE_NET labels are annotated in the corpus."""
        _, counts = self._run()
        assert set(counts.keys()) >= {"value", "nature", "labour", "gender"}

    def test_value_label_largest_pool(self):
        """'value' is the most annotated label (n ≥ 100)."""
        _, counts = self._run()
        assert counts["value"] >= 100
        assert counts["value"] == max(counts.values())

    def test_gender_pool_smaller_than_value(self):
        """'gender' has fewer annotations than 'value'."""
        _, counts = self._run()
        assert counts["gender"] < counts["value"]

    def test_alpha_ordering(self):
        """Larger corpus pool → lower alpha (more calibrated toward corpus)."""
        net, counts = self._run()
        # value has largest pool → lowest alpha
        # gender has smallest pool → highest alpha
        assert net["value"].alpha <= net["nature"].alpha
        assert net["value"].alpha <= net["labour"].alpha
        assert net["value"].alpha <= net["gender"].alpha

    def test_value_alpha_at_floor(self):
        """'value' pool is large enough to pin alpha at the floor (0.3)."""
        net, _ = self._run(alpha_floor=0.3, half_life=50)
        assert net["value"].alpha == pytest.approx(0.3)

    def test_gender_alpha_above_floor(self):
        """'gender' pool is small enough that alpha is above floor."""
        net, _ = self._run(alpha_floor=0.3, half_life=50)
        assert net["gender"].alpha > 0.3

    def test_prototypes_capped_at_default_max(self):
        """No label stores more than max_prototypes_per_label=20 prototypes."""
        net, _ = self._run(max_prototypes_per_label=20)
        for label, node in net.items():
            assert len(node.corpus_prototypes) <= 20, \
                f"Label '{label}' has {len(node.corpus_prototypes)} prototypes (max 20)"

    def test_prototypes_are_non_empty_strings(self):
        """All stored prototypes are non-empty, stripped strings."""
        net, _ = self._run()
        for label, node in net.items():
            for proto in node.corpus_prototypes:
                assert isinstance(proto, str), f"Non-string prototype in '{label}'"
                assert proto.strip() == proto, \
                    f"Unstripped prototype in '{label}': {repr(proto)}"
                assert proto, f"Empty prototype in '{label}'"

    def test_reproducibility_default_seed(self):
        """Two runs with default seed produce identical prototype lists."""
        import copy as _copy
        net_a = {k: _copy.copy(v) for k, v in VALUE_NET.items()}
        net_b = {k: _copy.copy(v) for k, v in VALUE_NET.items()}
        populate_corpus_prototypes(_CORPUS_DIR, value_net=net_a)
        populate_corpus_prototypes(_CORPUS_DIR, value_net=net_b)
        for label in VALUE_NET:
            assert net_a[label].corpus_prototypes == net_b[label].corpus_prototypes, \
                f"Non-reproducible prototypes for '{label}'"

    def test_alpha_not_mutated_for_absent_labels(self):
        """Labels absent from the corpus retain alpha=1.0 (theory-only)."""
        import copy as _copy
        # Build a net with an extra label not in corpus
        net = {k: _copy.copy(v) for k, v in VALUE_NET.items()}
        net["phantom"] = ValueNetNode(
            label="phantom",
            theory_hypothesis="phantom hypothesis",
        )
        populate_corpus_prototypes(_CORPUS_DIR, value_net=net)
        assert net["phantom"].alpha == 1.0
        assert net["phantom"].corpus_prototypes == []
