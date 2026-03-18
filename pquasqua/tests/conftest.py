"""
pquasqua/tests/conftest.py — shared fixtures and skip logic.

XMI Fixtures
------------
Integration tests that parse real INCEPTION XMI exports require fixtures
that are NOT committed to the repository. The fixtures contain the original
unanonymised sofaString (participant first names, institutional names) from
the fieldwork corpus. The source .txt transcripts were anonymised in the
corpus commit, but the XMI sofaString references character offsets — any
substitution that changes string length invalidates all 240+ span begin/end
positions. Length-preserving anonymisation of XMI is deferred; in the
interim, fixtures remain local-only.

Fixture directory: pquasqua/tests/fixtures/  (in .git/info/exclude)

To populate fixtures on a fresh checkout:

    python - <<'EOF'
    import zipfile
    from pathlib import Path

    corpus = Path("pquasqua/tests/contra_value_corpus_ANONYMISED")
    fixtures = Path("pquasqua/tests/fixtures")
    fixtures.mkdir(exist_ok=True)

    with zipfile.ZipFile(corpus / "annotation/Speaker_GR.txt/admin.zip") as z:
        with z.open("admin.xmi") as src:
            (fixtures / "speaker_gr.xmi").write_bytes(src.read())

    print(f"Extracted speaker_gr.xmi ({(fixtures / 'speaker_gr.xmi').stat().st_size} bytes)")
    EOF

Tests that require fixtures are skipped automatically when the fixture
directory is absent. To run them:

    pytest pquasqua/tests/ -v -p no:typeguard
    # integration tests will skip with a clear message

    # After populating fixtures:
    pytest pquasqua/tests/test_inception_transducer.py -v -p no:typeguard
    # TestIntegrationSpeakerGR will now run

Empirical BETO tests (downloads ~440MB on first run):

    BETO_EMPIRICAL=1 pytest pquasqua/tests/test_beto_perplexity.py -v -s -p no:typeguard
    BETO_EMPIRICAL=1 pytest pquasqua/tests/test_weight_extractor.py -v -s -p no:typeguard
"""

from pathlib import Path
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SPEAKER_GR_XMI = FIXTURES_DIR / "speaker_gr.xmi"


def pytest_collection_modifyitems(config, items):
    """
    Auto-skip any test marked `requires_xmi_fixtures` if the fixture
    directory is absent or empty.
    """
    if FIXTURES_DIR.exists() and SPEAKER_GR_XMI.exists():
        return
    skip = pytest.mark.skip(
        reason=(
            "XMI fixtures not present. "
            "See pquasqua/tests/conftest.py for extraction instructions."
        )
    )
    for item in items:
        if "requires_xmi_fixtures" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def speaker_gr_xmi_path():
    """
    Path to the Speaker GR XMI fixture.
    Skips the test automatically if the fixture is absent.
    """
    if not SPEAKER_GR_XMI.exists():
        pytest.skip(
            "speaker_gr.xmi not present. "
            "See pquasqua/tests/conftest.py for extraction instructions."
        )
    return SPEAKER_GR_XMI
