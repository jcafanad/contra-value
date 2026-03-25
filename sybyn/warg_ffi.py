"""
sybyn/warg_ffi.py — Python FFI bridge to the warg categorical argumentation engine.

warg is a Haskell implementation of categorical argumentation semantics — the
gradual semantics of abstract argumentation, categorically, per the ACT 2025
presentation. It lives as a sibling project at ~/warg/ (i.e. the same parent
directory as contra-value/).

Expected binary location:
    ~/warg/dist-newstyle/build/.../<ghc-version>/warg-0.1/x/warg/build/warg/warg
    or, after `cabal install --overwrite-policy=always`:
    ~/.cabal/bin/warg

The FFI contract is a simple subprocess pipe:
  - Input:  newline-terminated JSON object written to stdin
  - Output: newline-terminated JSON object read from stdout
  - Non-zero exit codes propagate as WargError

This module contains NO Haskell source. It is the Python side of the FFI only.
The Haskell side is in ~/warg/ — a separate session concern.

corpus_max_perplexity
---------------------
A calibration gate for the perplexity-aware gradual semantics. When an atom's
λ_⊥ (Atom.perplexity) exceeds corpus_max_perplexity, the atom's weight is
attenuated before being passed to warg. The rationale: atoms where BETO's
colonial training corpus creates extreme pseudo-perplexity (e.g. Muisca
cosmological claims rendered unintelligible by Iberian BETO) should not have
their formal weight silently erased — they should be marked as epistemically
imposed and their weight modulated accordingly.

The default value 21.769 is the empirically documented λ_⊥ of "poner a valer
a través del trabajo" (the automatic subject atom, Speaker GR), the highest
perplexity value observed in the Paramuno corpus that is still clearly within
natural Spanish. This is documented as the key theoretical finding in
project_automatic_subject_finding.md.

Do not recalibrate to reduce perplexity — the distortion is the signal.
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Default path to the warg binary. Resolved at call time (not import time)
#: so that the module is importable without the binary present.
_DEFAULT_WARG_BIN: Path = Path.home() / "warg" / ".cabal" / "bin" / "warg"

#: Empirical λ_⊥ of "poner a valer a través del trabajo" — automatic subject atom.
#: Used as the default corpus_max_perplexity calibration gate.
#: Source: internal/CONFIG_ANALYSIS.md, 2026-03-18 run with real BETO.
CORPUS_MAX_PERPLEXITY_DEFAULT: float = 21.769


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class WargError(RuntimeError):
    """Raised when the warg binary exits with a non-zero status or malformed output."""


class WargBinaryNotFound(FileNotFoundError):
    """Raised when the warg binary cannot be located at the expected path."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WargAtom:
    """
    A single atom in the form understood by the warg binary.

    name:            identifier (matches Atom.span_id or Atom.id for provenance)
    initial_weight:  Atom.initial_weight — NLI engagement at τ construction time
    weight:          Atom.weight — full VALUE_NET engagement score
    perplexity:      Atom.perplexity — λ_⊥ epistemic imposition measure
    attacks:         list of atom names that this atom attacks

    The perplexity field is passed to warg so that the Haskell categorical
    semantics can implement the corpus_max_perplexity attenuation gate on its
    side, where gradual weights are computed over the argumentation lattice.
    """
    name:           str
    initial_weight: float = 0.0
    weight:         float = 0.0
    perplexity:     float = 0.0
    attacks:        List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name":           self.name,
            "initial_weight": self.initial_weight,
            "weight":         self.weight,
            "perplexity":     self.perplexity,
            "attacks":        self.attacks,
        }


@dataclass
class WargResult:
    """
    Result returned by the warg binary for a single atom.

    name:            atom name (matches WargAtom.name)
    gradual_weight:  the gradual acceptability degree computed over the
                     argumentation lattice by the categorical semantics
    attenuated:      True if the atom's weight was attenuated by the
                     corpus_max_perplexity gate (λ_⊥ > threshold)
    """
    name:           str
    gradual_weight: float
    attenuated:     bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WargResult":
        return cls(
            name=d["name"],
            gradual_weight=float(d["gradual_weight"]),
            attenuated=bool(d.get("attenuated", False)),
        )


# ---------------------------------------------------------------------------
# FFI call
# ---------------------------------------------------------------------------

def call_warg(
    atoms: Sequence[WargAtom],
    corpus_max_perplexity: float = CORPUS_MAX_PERPLEXITY_DEFAULT,
    warg_bin: Optional[Path] = None,
    timeout: float = 30.0,
) -> List[WargResult]:
    """
    Call the warg binary with a sequence of atoms and return gradual weights.

    Serialises the input as a JSON object to stdin, reads a JSON array of
    result objects from stdout.

    Wire format (stdin):
        {
          "corpus_max_perplexity": <float>,
          "atoms": [
            { "name": ..., "initial_weight": ..., "weight": ...,
              "perplexity": ..., "attacks": [...] },
            ...
          ]
        }

    Wire format (stdout):
        [
          { "name": ..., "gradual_weight": ..., "attenuated": ... },
          ...
        ]

    Parameters
    ----------
    atoms                  : sequence of WargAtom instances
    corpus_max_perplexity  : λ_⊥ attenuation threshold (default CORPUS_MAX_PERPLEXITY_DEFAULT)
    warg_bin               : path to the warg binary; defaults to _DEFAULT_WARG_BIN
    timeout                : subprocess timeout in seconds (default 30)

    Raises
    ------
    WargBinaryNotFound : if the binary path does not exist
    WargError          : if the binary exits non-zero or output is malformed
    subprocess.TimeoutExpired : if the binary exceeds the timeout
    """
    if warg_bin is None:
        warg_bin = _DEFAULT_WARG_BIN

    if not Path(warg_bin).exists():
        raise WargBinaryNotFound(
            f"warg binary not found at {warg_bin}. "
            "Build with: cd ~/warg && cabal build && cabal install --overwrite-policy=always"
        )

    payload = json.dumps({
        "corpus_max_perplexity": corpus_max_perplexity,
        "atoms": [a.to_dict() for a in atoms],
    })

    try:
        result = subprocess.run(
            [str(warg_bin)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise WargBinaryNotFound(str(exc)) from exc

    if result.returncode != 0:
        raise WargError(
            f"warg exited with status {result.returncode}. "
            f"stderr: {result.stderr.strip()!r}"
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise WargError(
            f"warg returned malformed JSON: {exc}. "
            f"stdout: {result.stdout[:200]!r}"
        ) from exc

    try:
        return [WargResult.from_dict(item) for item in data]
    except (KeyError, TypeError, ValueError) as exc:
        raise WargError(f"warg result schema mismatch: {exc}") from exc
