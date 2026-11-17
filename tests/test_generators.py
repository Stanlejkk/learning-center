"""Every generator, every level, many seeds: must produce a problem whose own answer passes the checker."""
import os

import pytest

from engine import checker, content

SEEDS = int(os.environ.get("LC_SEEDS", "50"))
ALL_GENS = [g for gens in content.load_generators().values() for g in gens]


def test_skill_graph_is_valid():
    errs = content.validate_graph()
    assert not errs, "\n".join(errs)


@pytest.mark.parametrize("g", ALL_GENS, ids=[g.gen_id for g in ALL_GENS])
def test_generator_self_consistent(g):
    for level in g.levels:
        for seed in range(SEEDS):
            p = g.make(seed, level)
            assert p.prompt.strip(), f"{g.gen_id} L{level} s{seed}: empty prompt"
            assert p.rubric, f"{g.gen_id} L{level} s{seed}: empty rubric"
            assert p.prompt.count("$") % 2 == 0, f"{g.gen_id} L{level} s{seed}: unbalanced $ in prompt"
            assert p.solution.count("$") % 2 == 0, f"{g.gen_id} L{level} s{seed}: unbalanced $ in solution"
            ans = p.answer
            # the reference answer, typed as a user would type it, must be accepted
            typed = ", ".join(map(str, ans)) if isinstance(ans, (list, tuple)) and p.kind != "matrix" else ans
            if p.kind == "matrix" and isinstance(ans, (list, tuple)):
                typed = checker.fmt(ans) if not isinstance(ans, str) else ans
            r = checker.check(str(typed), ans, p.kind, p.tol)
            assert r.correct, f"{g.gen_id} L{level} seed {seed}: reference answer {ans!r} rejected ({r.message})\n{p.prompt}"
            # misconception answers must differ from the correct answer
            for wrong, tag in p.misconceptions:
                r2 = checker.check(str(", ".join(map(str, wrong)) if isinstance(wrong, (list, tuple)) else wrong),
                                   ans, p.kind, p.tol)
                assert not r2.correct, f"{g.gen_id} L{level} seed {seed}: misconception {tag} equals answer"


def test_deterministic():
    for g in ALL_GENS[:50]:
        a, b = g.make(7, g.levels[0]), g.make(7, g.levels[0])
        assert a.prompt == b.prompt and a.answer == b.answer, g.gen_id
