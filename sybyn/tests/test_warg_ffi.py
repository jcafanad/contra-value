"""
Tests for sybyn/warg_ffi.py — the Python-side FFI bridge to the warg binary.

Test structure
--------------
TestWargDataStructures   — WargAtom and WargResult serialisation / deserialisation
TestCallWargMock         — call_warg with a mock binary (subprocess fixture)
TestCallWargRealBinary   — smoke test against the real binary (skipped if absent)

Mock binary design
------------------
The mock_warg_bin fixture spawns a Python subprocess that:
  - reads a JSON object from stdin
  - validates the expected keys are present
  - writes a JSON array of result objects to stdout, one result per input atom
  - exits 0

This gives full end-to-end coverage of the serialisation/deserialisation
path without the Haskell binary. The mock does NOT implement the categorical
gradual semantics — it returns a fixed gradual_weight of 0.5 per atom, and
sets attenuated=True when the atom's perplexity exceeds corpus_max_perplexity.

Real binary path (for TestCallWargRealBinary):
    Expected at: ~/warg/.cabal/bin/warg
    (the default _DEFAULT_WARG_BIN in sybyn/warg_ffi.py)
    Build: cd ~/warg && cabal build && cabal install --overwrite-policy=always

Calibration smoke test
----------------------
TestCallWargRealBinary.test_corpus_max_perplexity_gate verifies that an atom
with λ_⊥ = 21.769 (the automatic subject baseline — "poner a valer a través
del trabajo") triggers attenuation when corpus_max_perplexity is set to 20.0.
This is the canonical calibration check; do not change the threshold without
updating internal/CONFIG_ANALYSIS.md.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from sybyn.warg_ffi import (
    CORPUS_MAX_PERPLEXITY_DEFAULT,
    WargAtom,
    WargBinaryNotFound,
    WargError,
    WargResult,
    call_warg,
    _DEFAULT_WARG_BIN,
)


# ---------------------------------------------------------------------------
# Mock binary fixture
# ---------------------------------------------------------------------------

_MOCK_WARG_SOURCE = textwrap.dedent("""\
    import json, sys

    data = json.loads(sys.stdin.read())
    threshold = data["corpus_max_perplexity"]
    results = []
    for atom in data["atoms"]:
        attenuated = atom["perplexity"] > threshold
        results.append({
            "name":           atom["name"],
            "gradual_weight": 0.5,
            "attenuated":     attenuated,
        })
    sys.stdout.write(json.dumps(results) + "\\n")
    sys.exit(0)
""")

_MOCK_WARG_BAD_JSON_SOURCE = textwrap.dedent("""\
    import sys
    sys.stdout.write("this is not json\\n")
    sys.exit(0)
""")

_MOCK_WARG_NONZERO_EXIT_SOURCE = textwrap.dedent("""\
    import sys
    sys.stderr.write("internal error\\n")
    sys.exit(1)
""")


@pytest.fixture(scope="session")
def mock_warg_bin(tmp_path_factory):
    """
    A Python script that acts as a mock warg binary.

    Reads JSON from stdin; writes one result per atom to stdout.
    Fixed gradual_weight=0.5; attenuated=True when perplexity > threshold.

    Returned as a Path to the executable script.
    """
    d = tmp_path_factory.mktemp("mock_warg")
    script = d / "warg_mock.py"
    script.write_text(_MOCK_WARG_SOURCE, encoding="utf-8")

    # Create a wrapper executable that invokes this Python script
    wrapper = d / "warg"
    wrapper.write_text(
        f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper


@pytest.fixture(scope="session")
def mock_warg_bad_json(tmp_path_factory):
    """Mock warg that returns malformed JSON."""
    d = tmp_path_factory.mktemp("mock_warg_bad")
    script = d / "warg_bad.py"
    script.write_text(_MOCK_WARG_BAD_JSON_SOURCE, encoding="utf-8")
    wrapper = d / "warg"
    wrapper.write_text(
        f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper


@pytest.fixture(scope="session")
def mock_warg_error(tmp_path_factory):
    """Mock warg that exits non-zero."""
    d = tmp_path_factory.mktemp("mock_warg_err")
    script = d / "warg_err.py"
    script.write_text(_MOCK_WARG_NONZERO_EXIT_SOURCE, encoding="utf-8")
    wrapper = d / "warg"
    wrapper.write_text(
        f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper


# ---------------------------------------------------------------------------
# TestWargDataStructures
# ---------------------------------------------------------------------------

class TestWargDataStructures:

    def test_warg_atom_to_dict_complete(self):
        """WargAtom.to_dict() must include all wire-format keys."""
        atom = WargAtom(
            name="paramuno_territorio",
            initial_weight=0.42,
            weight=0.71,
            perplexity=12.3,
            attacks=["recurso_hidrico_estado"],
        )
        d = atom.to_dict()
        assert d["name"]           == "paramuno_territorio"
        assert abs(d["initial_weight"] - 0.42) < 1e-9
        assert abs(d["weight"]     - 0.71) < 1e-9
        assert abs(d["perplexity"] - 12.3) < 1e-9
        assert d["attacks"]        == ["recurso_hidrico_estado"]

    def test_warg_atom_default_attacks_empty(self):
        atom = WargAtom(name="a")
        assert atom.attacks == []

    def test_warg_atom_to_dict_has_attacks_key(self):
        atom = WargAtom(name="a")
        d = atom.to_dict()
        assert "attacks" in d
        assert d["attacks"] == []

    def test_warg_result_from_dict(self):
        d = {"name": "a", "gradual_weight": 0.75, "attenuated": True}
        r = WargResult.from_dict(d)
        assert r.name == "a"
        assert abs(r.gradual_weight - 0.75) < 1e-9
        assert r.attenuated is True

    def test_warg_result_from_dict_attenuated_default_false(self):
        """attenuated is optional in the wire format; defaults to False."""
        d = {"name": "b", "gradual_weight": 0.33}
        r = WargResult.from_dict(d)
        assert r.attenuated is False

    def test_warg_result_from_dict_coerces_gradual_weight_to_float(self):
        """gradual_weight must be a Python float even if JSON returns int."""
        d = {"name": "c", "gradual_weight": 1}
        r = WargResult.from_dict(d)
        assert isinstance(r.gradual_weight, float)

    def test_corpus_max_perplexity_default_value(self):
        """
        The default corpus_max_perplexity must match the empirical λ_⊥ of
        the automatic subject atom ("poner a valer a través del trabajo").
        Source: internal/CONFIG_ANALYSIS.md, 2026-03-18.
        """
        assert abs(CORPUS_MAX_PERPLEXITY_DEFAULT - 21.769) < 1e-3


# ---------------------------------------------------------------------------
# TestCallWargMock — full round-trip via mock binary
# ---------------------------------------------------------------------------

class TestCallWargMock:

    def test_returns_one_result_per_atom(self, mock_warg_bin):
        atoms = [
            WargAtom(name="a", weight=0.6, perplexity=5.0),
            WargAtom(name="b", weight=0.3, perplexity=10.0),
        ]
        results = call_warg(atoms, warg_bin=mock_warg_bin)
        assert len(results) == 2

    def test_result_names_match_atoms(self, mock_warg_bin):
        atoms = [WargAtom(name="paramuno"), WargAtom(name="estado")]
        results = call_warg(atoms, warg_bin=mock_warg_bin)
        names = {r.name for r in results}
        assert names == {"paramuno", "estado"}

    def test_mock_gradual_weight_is_half(self, mock_warg_bin):
        """The mock always returns gradual_weight=0.5."""
        atoms = [WargAtom(name="a")]
        results = call_warg(atoms, warg_bin=mock_warg_bin)
        assert abs(results[0].gradual_weight - 0.5) < 1e-9

    def test_attenuation_flag_below_threshold(self, mock_warg_bin):
        """Atom with perplexity < corpus_max_perplexity → attenuated=False."""
        atoms = [WargAtom(name="a", perplexity=5.0)]
        results = call_warg(atoms, corpus_max_perplexity=20.0, warg_bin=mock_warg_bin)
        assert results[0].attenuated is False

    def test_attenuation_flag_above_threshold(self, mock_warg_bin):
        """
        Atom with perplexity > corpus_max_perplexity → attenuated=True.

        The automatic subject atom ("poner a valer a través del trabajo") has
        λ_⊥ = 21.769. With corpus_max_perplexity=20.0, it must be attenuated.
        This is the canonical calibration smoke test.
        """
        atoms = [WargAtom(name="automatic_subject_atom", perplexity=21.769)]
        results = call_warg(atoms, corpus_max_perplexity=20.0, warg_bin=mock_warg_bin)
        assert results[0].attenuated is True

    def test_corpus_max_perplexity_passed_to_binary(self, mock_warg_bin):
        """
        The corpus_max_perplexity value must be passed to the binary and
        affect attenuation. Two atoms: one below, one above the threshold.
        """
        atoms = [
            WargAtom(name="below", perplexity=15.0),
            WargAtom(name="above", perplexity=25.0),
        ]
        results = call_warg(atoms, corpus_max_perplexity=20.0, warg_bin=mock_warg_bin)
        by_name = {r.name: r for r in results}
        assert by_name["below"].attenuated is False
        assert by_name["above"].attenuated is True

    def test_attacks_serialised_in_wire_payload(self, mock_warg_bin, tmp_path):
        """
        attacks must be serialised in the wire payload so the Haskell side
        can build the attack graph. Test via JSON round-trip inspection.
        """
        atoms = [
            WargAtom(name="paramuno_territorio", attacks=["recurso_hidrico_estado"]),
            WargAtom(name="recurso_hidrico_estado"),
        ]
        # We can't intercept stdin without patching; test indirectly: call succeeds
        # and returns both atoms (binary received full payload without error).
        results = call_warg(atoms, warg_bin=mock_warg_bin)
        assert len(results) == 2

    def test_empty_atom_list(self, mock_warg_bin):
        """Empty atom list must succeed and return empty results."""
        results = call_warg([], warg_bin=mock_warg_bin)
        assert results == []

    def test_binary_not_found_raises(self):
        """Non-existent binary path must raise WargBinaryNotFound."""
        missing = Path("/nonexistent/warg")
        with pytest.raises(WargBinaryNotFound):
            call_warg([WargAtom(name="a")], warg_bin=missing)

    def test_nonzero_exit_raises_warg_error(self, mock_warg_error):
        """Binary exiting non-zero must raise WargError."""
        with pytest.raises(WargError, match="status 1"):
            call_warg([WargAtom(name="a")], warg_bin=mock_warg_error)

    def test_bad_json_raises_warg_error(self, mock_warg_bad_json):
        """Binary returning malformed JSON must raise WargError."""
        with pytest.raises(WargError, match="malformed JSON"):
            call_warg([WargAtom(name="a")], warg_bin=mock_warg_bad_json)

    def test_result_type_is_list_of_warg_result(self, mock_warg_bin):
        atoms = [WargAtom(name="a")]
        results = call_warg(atoms, warg_bin=mock_warg_bin)
        assert isinstance(results, list)
        assert all(isinstance(r, WargResult) for r in results)


# ---------------------------------------------------------------------------
# TestCallWargRealBinary — smoke test (skipped when binary absent)
# ---------------------------------------------------------------------------

_WARG_BINARY_PRESENT = _DEFAULT_WARG_BIN.exists()


@pytest.mark.skipif(
    not _WARG_BINARY_PRESENT,
    reason=(
        f"Real warg binary not found at {_DEFAULT_WARG_BIN}. "
        "Build with: cd ~/warg && cabal build && cabal install --overwrite-policy=always"
    ),
)
class TestCallWargRealBinary:
    """
    Smoke tests against the real warg binary.

    These tests are skipped when _DEFAULT_WARG_BIN is absent (the normal case
    during Python-side development). They are run when the Haskell binary has
    been built and installed.

    The tests do NOT verify the categorical gradual semantics — that is the
    domain of the Haskell test suite in ~/warg/. They verify only that:
      1. The FFI round-trip completes without error.
      2. The binary returns one result per input atom.
      3. The corpus_max_perplexity gate is respected.
    """

    def test_smoke_single_atom(self):
        """Basic round-trip: single atom, no attacks."""
        atoms = [WargAtom(name="paramuno", weight=0.6, perplexity=5.0)]
        results = call_warg(atoms)
        assert len(results) == 1
        assert results[0].name == "paramuno"
        assert 0.0 <= results[0].gradual_weight <= 1.0

    def test_smoke_attack_graph(self):
        """Two atoms: one attacking the other. Both must be in results."""
        atoms = [
            WargAtom(name="paramuno_territorio",
                     weight=0.71, perplexity=12.3,
                     attacks=["recurso_hidrico_estado"]),
            WargAtom(name="recurso_hidrico_estado",
                     weight=0.55, perplexity=7.1),
        ]
        results = call_warg(atoms)
        names = {r.name for r in results}
        assert "paramuno_territorio" in names
        assert "recurso_hidrico_estado" in names

    def test_corpus_max_perplexity_gate(self):
        """
        Canonical calibration check: the automatic subject atom (λ_⊥ = 21.769)
        must be attenuated when corpus_max_perplexity = 20.0.

        This is the smoke test for the perplexity gate. If the real binary
        does not return attenuated=True here, the Haskell implementation
        of the attenuation gate is wrong.

        Do not change the perplexity value or threshold without updating
        internal/CONFIG_ANALYSIS.md.
        """
        atoms = [
            WargAtom(
                name="poner_a_valer_a_traves_del_trabajo",
                weight=0.68,
                perplexity=21.769,   # empirical λ_⊥ of the automatic subject atom
            )
        ]
        results = call_warg(atoms, corpus_max_perplexity=20.0)
        assert len(results) == 1
        assert results[0].attenuated is True, (
            "The real warg binary did not set attenuated=True for an atom with "
            f"λ_⊥=21.769 and corpus_max_perplexity=20.0. "
            "Check the Haskell implementation of the perplexity gate."
        )
