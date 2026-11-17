"""Integrity tests for the resource library: content/readings.yaml, library/papers.yaml,
library/RESOURCES.md and the offline PDFs in library/pdfs/."""
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
TRACKS = ROOT / "content" / "tracks"
READINGS = ROOT / "content" / "readings.yaml"
PAPERS = ROOT / "library" / "papers.yaml"
RESOURCES = ROOT / "library" / "RESOURCES.md"
PDF_DIR = ROOT / "library" / "pdfs"

KINDS = {"lecture", "notes", "book", "video", "paper", "repo", "interactive"}


def _skill_ids() -> set[str]:
    ids: set[str] = set()
    for f in sorted(TRACKS.glob("*.yaml")):
        data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        for s in data.get("skills", []) or []:
            ids.add(s["id"])
    return ids


@pytest.fixture(scope="module")
def skill_ids() -> set[str]:
    ids = _skill_ids()
    assert ids, "no skills found in content/tracks"
    return ids


@pytest.fixture(scope="module")
def readings() -> dict:
    data = yaml.safe_load(READINGS.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


@pytest.fixture(scope="module")
def papers() -> list:
    data = yaml.safe_load(PAPERS.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    return data


def _is_pdf(p: Path) -> bool:
    with p.open("rb") as fh:
        return fh.read(4) == b"%PDF"


# ------------------------------------------------------------------ readings.yaml
def test_readings_parse(readings):
    assert len(readings) > 0


def test_every_skill_has_a_reading(skill_ids, readings):
    missing = sorted(s for s in skill_ids if not readings.get(s))
    assert not missing, f"skills without readings: {missing}"


def test_no_readings_for_unknown_skills(skill_ids, readings):
    unknown = sorted(set(readings) - skill_ids)
    assert not unknown, f"readings for unknown skill ids: {unknown}"


def test_reading_fields(readings):
    for sid, items in readings.items():
        assert isinstance(items, list) and 1 <= len(items) <= 4, sid
        for r in items:
            for key in ("title", "url", "kind"):
                assert r.get(key), f"{sid}: reading missing {key}: {r}"
            assert r["kind"] in KINDS, f"{sid}: bad kind {r['kind']}"
            assert re.match(r"^https?://", r["url"]), f"{sid}: bad url {r['url']}"
            if "minutes" in r:
                assert isinstance(r["minutes"], int) and r["minutes"] > 0


def test_reading_local_pdfs_exist(readings):
    for sid, items in readings.items():
        for r in items:
            if r.get("local"):
                p = ROOT / r["local"]
                assert p.is_file(), f"{sid}: missing local file {r['local']}"
                assert _is_pdf(p), f"{sid}: not a PDF {r['local']}"


# ------------------------------------------------------------------ papers.yaml
def test_papers(skill_ids, papers):
    assert 12 <= len(papers) <= 16
    ids = [p["id"] for p in papers]
    assert len(ids) == len(set(ids)), "duplicate paper ids"
    for p in papers:
        for key in ("id", "title", "authors", "year", "url", "skills", "why", "reading_guide", "key_equations"):
            assert p.get(key), f"{p.get('id')}: missing {key}"
        assert re.match(r"^https?://", p["url"])
        bad = [s for s in p["skills"] if s not in skill_ids]
        assert not bad, f"{p['id']}: unknown skills {bad}"
        assert len(p["reading_guide"]) >= 5, f"{p['id']}: needs >=5 reading_guide questions"
        if p.get("local"):
            lp = ROOT / p["local"]
            assert lp.is_file() and _is_pdf(lp), f"{p['id']}: bad local file {p['local']}"


# ------------------------------------------------------------------ RESOURCES.md local downloads
def test_resources_local_pdfs_exist():
    text = RESOURCES.read_text(encoding="utf-8")
    section = text.split("## Local downloads", 1)
    assert len(section) == 2, "RESOURCES.md lacks a 'Local downloads' section"
    body = section[1].split("\n## ", 1)[0]
    listed = re.findall(r"^\|\s*`([\w.-]+\.pdf)`", body, re.M)
    assert listed, "no local PDFs listed"
    for name in listed:
        p = PDF_DIR / name
        assert p.is_file(), f"listed but missing: {name}"
        assert _is_pdf(p), f"not a PDF: {name}"
    # every PDF on disk should be documented
    on_disk = {p.name for p in PDF_DIR.glob("*.pdf")}
    assert on_disk <= set(listed), f"undocumented PDFs: {sorted(on_disk - set(listed))}"
