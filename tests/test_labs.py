"""Every registered lab must: have notebook + checks + reference solution; the reference solution must
pass all checks; an empty namespace (unsolved notebook) must fail at least one check.

Set LC_SKIP_GPU_LABS=1 to skip labs marked gpu: true.
"""
import json
import os
import runpy
import sys

import pytest

from engine import content, labs

sys.path.insert(0, str(labs.LABS))
from lc_client import run_checks  # noqa: E402

ALL = list(labs.load_labs())


def test_lab_registry_consistent():
    ids = [l.id for l in ALL]
    assert len(ids) == len(set(ids)), "duplicate lab ids"
    sk = content.skills_by_id()
    for l in ALL:
        for s in l.skills + l.unlock:
            assert s in sk, f"{l.id}: unknown skill {s}"
        assert l.notebook.exists(), f"{l.id}: missing notebook"
        assert l.checks.exists(), f"{l.id}: missing checks"
        assert l.solution.exists(), f"{l.id}: missing reference solution"
        nb = json.loads(l.notebook.read_text(encoding="utf-8"))
        src = "".join("".join(c["source"]) for c in nb["cells"])
        assert f'grade("{l.id}"' in src or f"grade('{l.id}'" in src, f"{l.id}: notebook lacks grade() cell"


@pytest.mark.parametrize("lab", ALL, ids=[l.id for l in ALL])
def test_solution_passes_and_stub_fails(lab, monkeypatch):
    if lab.gpu and os.environ.get("LC_SKIP_GPU_LABS"):
        pytest.skip("gpu lab")
    monkeypatch.chdir(labs.LABS)
    ns = runpy.run_path(str(lab.solution), run_name="__lab_solution__")
    rep = run_checks(lab.id, ns)
    failed = [r for r in rep["results"] if not r["ok"]]
    assert rep["total"] >= 2, f"{lab.id}: needs at least 2 checks"
    assert not failed, f"{lab.id}: solution fails {failed}"
    empty = run_checks(lab.id, {})
    assert empty["passed"] < empty["total"], f"{lab.id}: checks pass on an empty namespace"
