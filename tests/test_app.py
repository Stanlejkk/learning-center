"""End-to-end flow through the web app with a temporary database."""
import re

import pytest
from fastapi.testclient import TestClient

from app import db, services
from app.main import app
from engine import content, labs, mastery


@pytest.fixture()
def client(tmp_path):
    db.set_db_path(tmp_path / "e2e.db")
    with TestClient(app) as c:
        yield c
    db.set_db_path(None)


def test_pages_render(client):
    for path in ["/", "/placement", "/skills", "/labs", "/library", "/history", "/settings", "/formulas"]:
        r = client.get(path)
        assert r.status_code == 200, path
    sid = content.all_skills_ordered()[5].id
    assert client.get(f"/skill/{sid}").status_code == 200


def test_placement_flow(client):
    client.post("/placement/start", data={"tracks": ["found", "calc"]})
    for _ in range(40):
        r = client.get("/placement", follow_redirects=False)
        if r.status_code == 303:
            break
        assert "Submit answer" in r.text
        # regression: the problem text and level must actually render (a template variable once shadowed them)
        assert re.search(r'<div class="prompt">\s*\S', r.text), "placement question rendered empty"
        assert re.search(r'level \d', r.text)
        client.post("/placement/answer", data={"answer": "1", "action": "submit"}, follow_redirects=False)
    r = client.get("/placement", follow_redirects=True)
    assert "Placement" in r.text
    with db.connect() as con:
        st = mastery.load_states(con)
        assert any(st[s.id].source == "placement" for s in content.load_tracks()[0].skills)


def test_homework_cycle_updates_mastery_and_finds_gaps(client):
    r = client.post("/homework/new", data={"n": "6"}, follow_redirects=False)
    sid = int(r.headers["location"].rsplit("/", 1)[1])
    page = client.get(f"/session/{sid}").text
    assert "Problems stay hidden" in page and "a_" not in page          # hidden until timer starts
    assert "Study first" in page and "Worked example" in page and "openstax.org" in page
    client.post(f"/session/{sid}/start")
    page = client.get(f"/session/{sid}").text
    ids = [int(x) for x in re.findall(r'name="a_(\d+)"', page)]
    assert '<div class="prompt"></div>' not in page
    assert ids
    with db.connect() as con:
        items = services.get_items(con, sid)
    answers = {}
    for k, it in enumerate(items):
        a = it["problem"]["answer"]
        right = ", ".join(map(str, a)) if isinstance(a, list) and it["problem"]["kind"] != "matrix" else str(a)
        answers[f"a_{it['id']}"] = right if k % 2 == 0 else "12345"
    client.post(f"/session/{sid}/submit", data=answers)
    page = client.get(f"/session/{sid}").text
    assert "Record results" in page
    rub = {f"r_{it['id']}_0": "on" for it in items}
    client.post(f"/session/{sid}/grade", data=rub)
    page = client.get(f"/session/{sid}").text
    assert "Result:" in page
    with db.connect() as con:
        sess = services.get_session(con, sid)
        assert sess["status"] == "graded"
        st = mastery.load_states(con)
        assert any(s.n_attempts > 0 for s in st.values())


def test_print_view_starts_timer(client):
    r = client.post("/homework/new", data={"n": "4"}, follow_redirects=False)
    sid = int(r.headers["location"].rsplit("/", 1)[1])
    page = client.get(f"/session/{sid}/print").text
    assert "Final answer" in page and "Formula sheet" in page and 'class="ftex"' in page
    assert 'class="ftex"' not in client.get(f"/session/{sid}/print?formulas=0").text
    with db.connect() as con:
        assert services.get_session(con, sid)["started_at"]


def test_lab_submit_api(client):
    all_labs = labs.load_labs()
    if not all_labs:
        pytest.skip("no labs registered yet")
    lab = all_labs[0]
    r = client.post("/api/labs/submit", json={"lab_id": lab.id, "passed": 3, "total": 3, "results": []})
    assert r.json()["status"] == "passed"
    with db.connect() as con:
        st = mastery.load_states(con)
        assert all(st[s].n_attempts >= 1 for s in lab.skills)
    assert client.post("/api/labs/submit", json={"lab_id": "nope"}).status_code == 400


def test_checkpoint_and_remediation_routes(client):
    r = client.post("/checkpoint/found", follow_redirects=False)
    assert r.status_code == 303
    r = client.post("/remediation/new", data={"skills": ["found.fractions"]}, follow_redirects=False)
    assert r.status_code == 303
    r = client.post("/practice/found.trig", follow_redirects=False)
    assert r.status_code == 303


def test_all_correct_set_never_reports_gaps_from_mistakes(client):
    with db.connect() as con:
        st = mastery.load_states(con)
        mastery.set_prior(con, st, "found.linear_eq", 0.12, "placement")   # an open gap from earlier
        mastery.set_prior(con, st, "found.fractions", 0.9, "placement")
        sid = services.new_practice(con, "found.functions", n=3)
        services.start_session(con, sid)
        items = services.get_items(con, sid)
        answers = {it["id"]: (", ".join(map(str, it["problem"]["answer"])) if isinstance(it["problem"]["answer"], list)
                              else str(it["problem"]["answer"])) for it in items}
        services.submit_answers(con, sid, answers)
        sm = services.grade_session(con, sid, {it["id"]: list(range(len(it["problem"]["rubric"]))) for it in items})
    assert sm["score"] == 1.0
    assert sm["weakened"] == []                  # nothing blamed on this set
    page = client.get(f"/session/{sid}").text
    assert "your mistakes point to" not in page
