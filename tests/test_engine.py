import random
from pathlib import Path

import pytest

from app import db
from engine import checker, content, mastery, placement, scheduler


# ------------------------------------------------------------------ checker
@pytest.mark.parametrize("user,ref,kind", [
    ("2x+2", "2*(x+1)", "expr"),
    ("2(x+1)", "2*x+2", "expr"),
    ("x^2", "x**2", "expr"),
    ("sqrt(2)/2", "1/sqrt(2)", "expr"),
    ("0.3333333", "1/3", "num"),
    ("pi/3", "1.0471975512", "num"),
    ("[[1,2],[3,4]]", [[1, 2], [3, 4]], "matrix"),
    ("[1, -2, 3]", "[1,-2,3]", "vector"),
    ("-3, 2", "2, -3", "set"),
    ("{2, -3}", ["-3", "2"], "set"),
    ("b", "B", "choice"),
    ("3 + 4i", "3+4*I", "expr"),
    ("ln(2)/0.05", "log(2)/0.05", "expr"),
    ("e^(2x)", "exp(2*x)", "expr"),
    ("\\frac{1}{2}", "0.5", "num"),
])
def test_checker_accepts(user, ref, kind):
    assert checker.check(user, ref, kind).correct


@pytest.mark.parametrize("user,ref,kind", [
    ("2x+1", "2*(x+1)", "expr"),
    ("0.33", "1/3", "num"),
    ("[[1,2],[4,3]]", [[1, 2], [3, 4]], "matrix"),
    ("2, 3", "2, -3", "set"),
    ("1, 2", "2, 1", "list"),
    ("A", "B", "choice"),
])
def test_checker_rejects(user, ref, kind):
    assert not checker.check(user, ref, kind).correct


def test_checker_rejects_code():
    r = checker.check("__import__('os').system('x')", "1", "expr")
    assert not r.correct


def test_misconception_tag():
    r = checker.check("5/7", "29/28", "num", misconceptions=[("5/7", "added-denominators")])
    assert not r.correct and r.misconception == "added-denominators"


# ------------------------------------------------------------------ BKT
def test_bkt_moves_in_right_direction():
    p = 0.5
    assert mastery.bkt_update(p, 1.0) > p
    assert mastery.bkt_update(p, 0.0, learn=False) < p
    up = p
    for _ in range(12):
        up = mastery.bkt_update(up, 1.0)
    assert up > mastery.MASTERED


def test_partial_credit_between():
    p = 0.5
    full, none, half = (mastery.bkt_update(p, s, learn=False) for s in (1.0, 0.0, 0.4))
    assert none < half < full


# ------------------------------------------------------------------ DB-backed fixtures
@pytest.fixture()
def tmpdb(tmp_path: Path):
    db.set_db_path(tmp_path / "t.db")
    db.init_db()
    yield
    db.set_db_path(None)


def simulate_placement(true_known: set[str], slip=0.0, seed=0):
    rng = random.Random(seed)
    st = placement.new_state(seed=seed)
    n = 0
    while (item := placement.next_item(st)) is not None:
        ok = item["skill"] in true_known and rng.random() >= slip
        placement.record(st, item, ok)
        n += 1
        assert n < 200
    return st, n


def test_placement_terminates_and_converges():
    tracks = content.load_tracks()
    # a learner who knows the first half of each track
    known = {s.id for t in tracks for s in t.skills[: len(t.skills) // 2]}
    st, n = simulate_placement(known)
    assert n <= placement.MAX_PER_TRACK * len(tracks)
    pri = placement.priors(st)
    agree = sum((pri[s] >= 0.6) == (s in known) for s in pri)
    assert agree / len(pri) >= 0.8, f"placement agreement {agree}/{len(pri)}"


def test_placement_all_unknown_and_all_known():
    st, _ = simulate_placement(set())
    assert all(v < 0.5 for v in placement.priors(st).values())
    allk = set(content.skills_by_id())
    st, _ = simulate_placement(allk)
    assert all(v >= 0.6 for v in placement.priors(st).values())


def test_gap_detection_triggers_remediation(tmpdb):
    with db.connect() as con:
        states = mastery.load_states(con)
        # pick a skill with a prereq
        sk = next(s for s in content.all_skills_ordered() if s.prereqs)
        pre = sk.prereqs[0]
        mastery.set_prior(con, states, pre, 0.65, "test")
        for other in sk.prereqs[1:]:
            mastery.set_prior(con, states, other, 0.9, "test")
        weakened = []
        for _ in range(4):
            res = mastery.record_attempt(con, states, sk.id, correct=False, rubric_fraction=0.0)
            weakened += res["weakened"]
        assert pre in weakened
        assert pre in scheduler.weak_prereqs(states, [sk.id])


def test_homework_plan_mix(tmpdb):
    with db.connect() as con:
        states = mastery.load_states(con)
        first = content.all_skills_ordered()
        for s in first[:4]:
            mastery.set_prior(con, states, s.id, 0.97, "test")
        items = scheduler.plan_homework(states, n=8, seed=1)
        assert len(items) == 8
        roles = {i["role"] for i in items}
        assert "frontier" in roles and "review" in roles
        for i in items:
            assert content.make_problem(i["skill"], i["level"], i["seed"]) is not None


def test_empty_state_starts_at_beginning(tmpdb):
    with db.connect() as con:
        states = mastery.load_states(con)
        items = scheduler.plan_homework(states, n=5, seed=1)
        assert items and items[0]["skill"] == content.all_skills_ordered()[0].id


def test_review_scheduling(tmpdb):
    with db.connect() as con:
        states = mastery.load_states(con)
        sid = content.all_skills_ordered()[0].id
        for _ in range(15):
            mastery.record_attempt(con, states, sid, True, 1.0)
        s = states[sid]
        assert s.mastered and s.review_due is not None and s.review_interval >= 1
