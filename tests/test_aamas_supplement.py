"""Anonymous packaging must preserve science while rejecting identifying bytes."""
from importlib.util import module_from_spec, spec_from_file_location
import hashlib
import io
import json
from pathlib import Path
import zipfile
import zlib

import numpy as np
import pytest

SPEC = spec_from_file_location("aamas_supplement", Path(__file__).resolve().parents[1] / "scripts/package_aamas_supplement.py")
module = module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def fixture_repository(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    names = ["docs/cage_adaptation_protocol.json", "results/cage_adaptation/selection.json",
             "results/cage_adaptation/geometry/primary_sources.json", "data/bank.npz"]
    for name in names:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / names[0]).write_bytes(b'{"article":{"path":"C:/Math/aamas_revision/main.tex"}}\r\n')
    (root / names[1]).write_bytes(b'{"selected":{"scalar":{"window":16,"rho":0.25}}}\n')
    (root / names[2]).write_bytes(b'{"source_checkpoint_sha256":{"seed_1.npz":"abc"}}\r\n')
    np.savez_compressed(root / names[3], values=np.arange(8), labels=np.array(["host_compromise"]))
    analysis = {"primary": {"mean": 1.104, "contrast": -.0618}, "protocol_sha256": "frozen",
                "publication_scope": {"archive_url": "https://github.com/Jew-Yeah/aamas-lowdim-experiments"}}
    analysis_path = root / "results/cage_adaptation/analysis.json"
    analysis_path.write_bytes(module.canonical_json(analysis))
    monkeypatch.setattr(module, "allowlist", lambda: names)
    return root, names, analysis


def test_analysis_projection_removes_only_presentation_fields():
    original = {"primary": {"methods": [1., 2.], "seeds": [4, 5]}, "frozen_at": "original",
                "publication_scope": {"archive_url": "https://github.com/Jew-Yeah/aamas-lowdim-experiments"}}
    raw = module.canonical_json(original)
    projected, provenance = module.anonymous_analysis(raw)
    assert json.loads(projected) == {key: value for key, value in original.items() if key != "publication_scope"}
    assert provenance["original_sha256"] == hashlib.sha256(raw).hexdigest()
    assert provenance["derivative_sha256"] == hashlib.sha256(projected).hexdigest()
    assert provenance["scientific_fields_unchanged"] is True


def test_package_preserves_frozen_crlf_arrays_and_selection_and_is_deterministic(tmp_path, monkeypatch):
    root, names, analysis = fixture_repository(tmp_path, monkeypatch)
    before = {name: (root / name).read_bytes() for name in names}
    result1 = module.build_supplement(root, tmp_path / "one.zip")
    result2 = module.build_supplement(root, tmp_path / "two.zip")
    assert result1 == result2
    assert (tmp_path / "one.zip").read_bytes() == (tmp_path / "two.zip").read_bytes()
    with zipfile.ZipFile(tmp_path / "one.zip") as archive:
        manifest = json.loads(archive.read("MANIFEST.json"))
        for name, content in before.items():
            assert archive.read(name) == content
            assert (root / name).read_bytes() == content
        assert json.loads(archive.read("results/cage_adaptation/analysis.json"))["primary"] == analysis["primary"]
        for name, value in manifest["files_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == value
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
        assert "WORK_LOG.md" not in archive.namelist()


@pytest.mark.parametrize("text", ["https://github.com/Jew-Yeah/aamas-lowdim-experiments",
                                  "C:/Users/NamedAuthor/project", "/home/named_author/project",
                                  "named_author@example.org", "codex://threads/private-id"])
def test_identifying_text_is_rejected(text):
    with pytest.raises(ValueError, match="Identifying"):
        module.scan_member("README.md", text.encode())


def test_embedded_unicode_array_and_compressed_pdf_metadata_are_scanned():
    buffer = io.BytesIO()
    np.savez_compressed(buffer, metadata=np.array(["C:/Users/NamedAuthor/private"]))
    with pytest.raises(ValueError, match="Identifying"):
        module.scan_member("data.npz", buffer.getvalue())
    pdf = b"%PDF-1.4\nstream\n" + zlib.compress(b"/Author (Jew-Yeah)") + b"\nendstream"
    with pytest.raises(ValueError, match="Identifying"):
        module.scan_member("figure.pdf", pdf)


def test_numeric_binary_is_not_misread_as_personal_text_and_object_arrays_fail():
    buffer = io.BytesIO()
    np.savez_compressed(buffer, numeric=np.frombuffer(b"email@example.org", dtype=np.uint8))
    module.scan_member("scientific.npz", buffer.getvalue())
    objects = io.BytesIO()
    np.savez_compressed(objects, metadata=np.array([{"identity": "named"}], dtype=object))
    with pytest.raises(ValueError, match="Object"):
        module.scan_member("unsafe.npz", objects.getvalue())


@pytest.mark.parametrize("name", ["../outside.txt", "/absolute.txt", "C:/outside.txt", ".git/config", "a\\b"])
def test_archive_paths_cannot_escape_or_include_git_metadata(name):
    with pytest.raises(ValueError, match="Unsafe"):
        module.safe_member(name)


def test_missing_allowlisted_file_fails_and_oversize_preserves_existing_output(tmp_path, monkeypatch):
    root, names, _ = fixture_repository(tmp_path, monkeypatch)
    output = tmp_path / "supplement.zip"
    output.write_bytes(b"previous valid archive")
    with pytest.raises(ValueError, match="exceeds"):
        module.build_supplement(root, output, max_bytes=100)
    assert output.read_bytes() == b"previous valid archive"
    (root / names[0]).unlink()
    with pytest.raises(ValueError, match="Missing"):
        module.build_supplement(root, output)


def test_optional_anonymous_paper_inputs_are_included_and_identifying_readme_is_rejected(tmp_path, monkeypatch):
    root, _, _ = fixture_repository(tmp_path, monkeypatch)
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "experiments.tex").write_text("\\section{Experiments}\n", encoding="utf-8")
    ai = tmp_path / "ai.md"
    ai.write_text("AI assistance was used for code and language editing.\n", encoding="utf-8")
    module.build_supplement(root, tmp_path / "anonymous.zip", paper_dir=paper, ai_statement=ai)
    with zipfile.ZipFile(tmp_path / "anonymous.zip") as archive:
        assert archive.read("paper/experiments.tex") == (paper / "experiments.tex").read_bytes()
        assert archive.read("AI_DISCLOSURE.md") == ai.read_bytes()
    readme = tmp_path / "reviewer.md"
    readme.write_text("Contact author@example.org", encoding="utf-8")
    with pytest.raises(ValueError, match="Identifying"):
        module.build_supplement(root, tmp_path / "unsafe.zip", reviewer_readme=readme)


def test_default_allowlist_contains_manuscript_builder_test_and_english_exports():
    names = module.allowlist()
    assert "scripts/build_aamas_paper_figures.py" in names
    assert "tests/test_aamas_paper_figures.py" in names
    assert "results/cage_adaptation/aamas/provenance.json" in names
    assert "results/cage_adaptation/aamas/captions.tex" in names
    for figure in module.AAMAS_FIGURES:
        for extension in ("pdf", "png"):
            assert f"results/cage_adaptation/aamas/{figure}.{extension}" in names
    assert not any(".ru." in name or name == "WORK_LOG.md" for name in names)


def test_manuscript_exports_are_mirrored_exactly_and_conflicting_figures_fail(tmp_path, monkeypatch):
    root, names, _ = fixture_repository(tmp_path, monkeypatch)
    new_names = []
    for figure in module.AAMAS_FIGURES:
        name = f"results/cage_adaptation/aamas/{figure}.pdf"
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"%PDF-1.4\n% Anonymous vector export\n")
        new_names.append(name)
    monkeypatch.setattr(module, "allowlist", lambda: names + new_names)
    output = tmp_path / "supplement.zip"
    module.build_supplement(root, output)
    with zipfile.ZipFile(output) as archive:
        for name in new_names:
            assert archive.read(name) == archive.read("paper/figures/" + Path(name).name)
        manifest = json.loads(archive.read("MANIFEST.json"))
        assert manifest["presentation_provenance_analysis_mapping"]["anonymous_analysis_sha256"] == manifest["analysis_projection"]["derivative_sha256"]
    paper = tmp_path / "paper"
    (paper / "figures").mkdir(parents=True)
    (paper / "figures/primary_comparison.pdf").write_bytes(b"%PDF-1.4\n% altered result\n")
    with pytest.raises(ValueError, match="differs from the canonical"):
        module.build_supplement(root, output, paper_dir=paper)
