"""Session lifecycle: create -> start -> submit answers -> self-grade rubric -> mastery update."""
from __future__ import annotations

import json
import random
from datetime import datetime

from app import db
from engine import checker, content, mastery, placement, scheduler
from engine import labs as labreg

KIND_TITLES = {
    "homework": "Homework",
    "remediation": "Gap remediation",
    "review": "Spaced review",
    "checkpoint": "Checkpoint test",
    "practice": "Skill practice",
    "placement": "Placement test",
}


# ------------------------------------------------------------------ helpers
def states(con) -> dict[str, mastery.SkillState]:
    return mastery.load_states(con)


def get_session(con, sid: int) -> dict | None:
    row = con.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    if not row:
        return None
    s = dict(row)
    s["data"] = json.loads(s["data"] or "{}")
    return s


def get_items(con, sid: int) -> list[dict]:
    out = []
    for r in con.execute("SELECT * FROM items WHERE session_id=? ORDER BY idx", (sid,)).fetchall():
        d = dict(r)
        d["problem"] = json.loads(d["problem"])
        d["rubric_checked"] = json.loads(d["rubric_checked"]) if d["rubric_checked"] else []
        d["skill_name"] = content.skills_by_id().get(d["skill_id"]).name if d["skill_id"] in content.skills_by_id() else d["skill_id"]
        out.append(d)
    return out


def _insert_items(con, sid: int, plan: list[dict]) -> int:
    n = 0
    for it in plan:
        p = content.make_problem(it["skill"], it["level"], it["seed"])
        if p is None:
            continue
        con.execute(
            "INSERT INTO items(session_id, idx, skill_id, level, gen_id, seed, role, problem) VALUES (?,?,?,?,?,?,?,?)",
            (sid, n, p.skill, p.level, p.gen_id, p.seed, it.get("role"), json.dumps(p.to_dict(), default=str)))
        n += 1
    return n


def create_session(con, kind: str, plan: list[dict], title: str | None = None, track: str | None = None,
                   time_limit_min: int | None = None, data: dict | None = None) -> int:
    cur = con.execute(
        "INSERT INTO sessions(kind, track, title, created_at, status, time_limit_min, data) VALUES (?,?,?,?,?,?,?)",
        (kind, track, title or KIND_TITLES.get(kind, kind.title()), db.now(), "created", time_limit_min,
         json.dumps(data or {})))
    sid = cur.lastrowid
    if _insert_items(con, sid, plan) == 0:
        con.execute("DELETE FROM sessions WHERE id=?", (sid,))
        raise ValueError("No problems available for this selection yet.")
    db.log_event(con, "session_created", detail={"id": sid, "kind": kind})
    return sid


def new_homework(con, n: int = 8, track: str | None = None) -> int:
    st = states(con)
    plan = scheduler.plan_homework(st, n=n, track=track, seed=random.randrange(10**9))
    roles = {p["role"] for p in plan}
    title = "Homework" + (" + gap remediation" if "remediation" in roles else "")
    if track:
        title += f" · {next(t.name for t in content.load_tracks() if t.id == track)}"
    return create_session(con, "homework", plan, title=title, track=track)


def new_practice(con, skill_id: str, n: int = 5) -> int:
    st = states(con)
    lvl = scheduler.level_for(st[skill_id].p)
    rng = random.Random()
    levels = [max(1, lvl - 1)] + [lvl] * (n - 2) + [min(3, lvl + 1)]
    plan = [{"skill": skill_id, "level": l, "seed": rng.randrange(10**9), "role": "practice"} for l in levels]
    return create_session(con, "practice", plan, title=f"Practice · {content.skills_by_id()[skill_id].name}")


def new_remediation(con, skills: list[str]) -> int:
    plan = scheduler.plan_remediation(states(con), skills, seed=random.randrange(10**9))
    names = ", ".join(content.skills_by_id()[s].name for s in skills)
    return create_session(con, "remediation", plan, title=f"Gap remediation · {names}")


def new_review(con) -> int:
    st = states(con)
    due = scheduler.due_reviews(st)[:8]
    if not due:
        raise ValueError("Nothing is due for review today.")
    rng = random.Random()
    plan = [{"skill": s, "level": max(2, scheduler.level_for(st[s].p)), "seed": rng.randrange(10**9),
             "role": "review"} for s in due]
    return create_session(con, "review", plan, title="Spaced review")


def new_checkpoint(con, track: str) -> int:
    plan = scheduler.plan_checkpoint(states(con), track, seed=random.randrange(10**9))
    tname = next(t.name for t in content.load_tracks() if t.id == track)
    return create_session(con, "checkpoint", plan, title=f"Checkpoint · {tname}", track=track,
                          )


def start_session(con, sid: int) -> None:
    con.execute("UPDATE sessions SET started_at=COALESCE(started_at, ?), status=CASE WHEN status='created' "
                "THEN 'started' ELSE status END WHERE id=?", (db.now(), sid))


def submit_answers(con, sid: int, answers: dict[int, str]) -> None:
    """Check typed final answers. Mastery is NOT updated yet (that happens after the rubric self-grade)."""
    for it in get_items(con, sid):
        ua = answers.get(it["id"], "")
        p = it["problem"]
        res = checker.check(ua, p["answer"], p["kind"], p.get("tol"), p.get("misconceptions"))
        con.execute("UPDATE items SET user_answer=?, correct=?, misconception=?, answered_at=? WHERE id=?",
                    (ua, int(res.correct), res.misconception, db.now(), it["id"]))
    con.execute("UPDATE sessions SET submitted_at=?, status='submitted' WHERE id=?", (db.now(), sid))


def grade_session(con, sid: int, rubric: dict[int, list[int]]) -> dict:
    """Apply rubric self-grades, update mastery, detect gaps. Returns a summary."""
    sess = get_session(con, sid)
    if sess["status"] == "graded":
        return sess["data"].get("summary", {})
    st = states(con)
    changes, weakened, by_skill = [], [], {}
    used_rubric = any(rubric.values())  # rubric skipped entirely -> only answers count
    for it in get_items(con, sid):
        steps = it["problem"].get("rubric") or []
        checked = sorted(set(rubric.get(it["id"], [])))
        frac = (len(checked) / len(steps)) if (steps and used_rubric) else None
        score = mastery.combined_score(bool(it["correct"]), frac)
        con.execute("UPDATE items SET rubric_checked=?, score=? WHERE id=?", (json.dumps(checked), score, it["id"]))
        ch = mastery.record_attempt(con, st, it["skill_id"], bool(it["correct"]), frac)
        changes.append(ch)
        weakened += [w for w in ch["weakened"] if w not in weakened]
        b = by_skill.setdefault(it["skill_id"], {"n": 0, "score": 0.0})
        b["n"] += 1
        b["score"] += score
        db.log_event(con, "attempt", it["skill_id"], {"correct": bool(it["correct"]), "score": score,
                                                        "misconception": it["misconception"]})
    total = sum(c["score"] for c in changes) / max(1, len(changes))
    summary = {"score": round(total, 3), "weakened": weakened,
               "skills": {k: {"n": v["n"], "avg": round(v["score"] / v["n"], 3),
                              "p": round(st[k].p, 3)} for k, v in by_skill.items()}}
    if sess["kind"] == "checkpoint":
        passed = total >= 0.8
        summary["checkpoint_passed"] = passed
        for k, v in by_skill.items():
            if v["score"] / v["n"] >= 0.8:
                st[k].certified = True
                mastery.save_state(con, st[k])
    # `weakened`: gaps implied by mistakes made IN THIS SET (only possible if something was wrong).
    # `open_gaps`: weak spots that were already open (e.g. from the placement test) - reported separately,
    # never blamed on this set.
    summary["weakened"] = weakened
    frontier = scheduler.frontier(st, limit=4)
    summary["open_gaps"] = [w for w in scheduler.weak_prereqs(st, frontier) if w not in weakened][:3]
    sess["data"]["summary"] = summary
    con.execute("UPDATE sessions SET graded_at=?, status='graded', data=? WHERE id=?",
                (db.now(), json.dumps(sess["data"]), sid))
    return summary


# ------------------------------------------------------------------ placement
def placement_session(con) -> dict | None:
    row = con.execute("SELECT id FROM sessions WHERE kind='placement' AND status='active' ORDER BY id DESC LIMIT 1").fetchone()
    return get_session(con, row["id"]) if row else None


def start_placement(con, tracks: list[str] | None = None) -> int:
    state = placement.new_state(tracks)
    cur = con.execute("INSERT INTO sessions(kind, title, created_at, started_at, status, data) VALUES (?,?,?,?,?,?)",
                      ("placement", "Placement test", db.now(), db.now(), "active", json.dumps({"state": state})))
    return cur.lastrowid


def placement_current(con, sess: dict) -> tuple[dict | None, dict | None]:
    """Return (item, problem) for the current placement question, generating it if needed."""
    data = sess["data"]
    state = data["state"]
    item = data.get("pending")
    if item is None:
        # skip skills that have no generators yet
        while True:
            item = placement.next_item(state)
            if item is None or content.generators_for(item["skill"]):
                break
            placement.record(state, item, False)
        data["pending"] = item
        con.execute("UPDATE sessions SET data=? WHERE id=?", (json.dumps(data), sess["id"]))
    if item is None:
        return None, None
    p = content.make_problem(item["skill"], item["level"], item["seed"])
    return item, p.to_dict()


def placement_answer(con, sess: dict, answer: str | None, skip_track: bool = False) -> dict:
    data = sess["data"]
    state, item = data["state"], data.get("pending")
    result = {}
    if item:
        if skip_track:
            placement.skip_track(state, item["track"])
            result = {"skipped_track": item["track"]}
        else:
            p = content.make_problem(item["skill"], item["level"], item["seed"])
            res = checker.check(answer or "", p.answer, p.kind, p.tol, p.misconceptions)
            placement.record(state, item, res.correct)
            con.execute("INSERT INTO items(session_id, idx, skill_id, level, gen_id, seed, role, problem, user_answer, correct, misconception, answered_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (sess["id"], len(data.get("log", [])), p.skill, p.level, p.gen_id, p.seed, "placement",
                         json.dumps(p.to_dict(), default=str), answer, int(res.correct), res.misconception, db.now()))
            data.setdefault("log", []).append({"skill": item["skill"], "correct": res.correct})
    data["pending"] = None
    con.execute("UPDATE sessions SET data=? WHERE id=?", (json.dumps(data), sess["id"]))
    return result


def placement_progress(sess: dict) -> dict:
    state = sess["data"]["state"]
    done = sum(1 for t in state["track_order"] if state["tracks"][t]["done"])
    asked = sum(len(state["tracks"][t]["asked"]) for t in state["track_order"])
    cur = state["track_order"][min(state["current"], len(state["track_order"]) - 1)]
    return {"tracks_done": done, "tracks_total": len(state["track_order"]), "asked": asked,
            "current_track": next(t.name for t in content.load_tracks() if t.id == cur)}


def finish_placement(con, sess: dict) -> None:
    state = sess["data"]["state"]
    placement.finish(state)
    st = states(con)
    for sid, p in placement.priors(state).items():
        mastery.set_prior(con, st, sid, p, "placement")
    sess["data"]["state"] = state
    con.execute("UPDATE sessions SET data=?, status='done', graded_at=? WHERE id=?",
                (json.dumps(sess["data"]), db.now(), sess["id"]))
    db.log_event(con, "placement_done", detail={"id": sess["id"]})


# ------------------------------------------------------------------ labs
def lab_status(con, st: dict[str, mastery.SkillState]) -> list[dict]:
    rows = {r["lab_id"]: dict(r) for r in con.execute("SELECT * FROM labs").fetchall()}
    out = []
    for lab in labreg.load_labs():
        locked_by = [u for u in lab.unlock if st[u].p < labreg.UNLOCK_P]
        r = rows.get(lab.id, {})
        out.append({"lab": lab, "locked_by": locked_by, "status": r.get("status", "not started"),
                    "passed": r.get("passed"), "total": r.get("total"), "submitted_at": r.get("submitted_at"),
                    "exists": lab.notebook.exists()})
    return out


def record_lab(con, report: dict) -> str:
    lab = labreg.get_lab(report.get("lab_id", ""))
    if not lab:
        raise ValueError("unknown lab")
    passed, total = int(report.get("passed", 0)), int(report.get("total", 0))
    status = "passed" if total and passed == total else "in progress"
    prev = con.execute("SELECT status FROM labs WHERE lab_id=?", (lab.id,)).fetchone()
    con.execute("""INSERT INTO labs(lab_id,status,score,passed,total,details,submitted_at) VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(lab_id) DO UPDATE SET status=CASE WHEN labs.status='passed' THEN 'passed' ELSE excluded.status END,
                   score=excluded.score, passed=excluded.passed, total=excluded.total, details=excluded.details,
                   submitted_at=excluded.submitted_at""",
                (lab.id, status, passed / total if total else 0, passed, total, json.dumps(report), db.now()))
    if status == "passed" and (not prev or prev["status"] != "passed"):
        st = states(con)
        for s in lab.skills:
            mastery.record_attempt(con, st, s, True, 1.0, propagate=False)
    db.log_event(con, "lab_submit", detail={"lab": lab.id, "passed": passed, "total": total})
    return status


# ------------------------------------------------------------------ stats
def activity(con) -> dict:
    rows = con.execute("SELECT substr(at,1,10) d, COUNT(*) n FROM events WHERE kind IN ('attempt','lab_submit') "
                       "GROUP BY d ORDER BY d DESC").fetchall()
    days = {r["d"]: r["n"] for r in rows}
    streak, d = 0, datetime.now().date()
    from datetime import timedelta  # noqa: PLC0415
    if d.isoformat() not in days:
        d -= timedelta(days=1)
    while d.isoformat() in days:
        streak += 1
        d -= timedelta(days=1)
    mins = con.execute("SELECT started_at, submitted_at FROM sessions WHERE started_at IS NOT NULL AND submitted_at IS NOT NULL").fetchall()
    total_min = 0.0
    for r in mins:
        try:
            dt = (datetime.fromisoformat(r["submitted_at"]) - datetime.fromisoformat(r["started_at"])).total_seconds() / 60
            total_min += min(dt, 240)
        except ValueError:
            pass
    return {"streak": streak, "days": days, "total_hours": round(total_min / 60, 1),
            "attempts": sum(days.values())}


# ------------------------------------------------------------------ maintenance
def rebuild_mastery(con) -> dict:
    """Recompute every skill's mastery from scratch: placement priors, then every graded answer and passed
    lab replayed in time order with the CURRENT scoring rules. Safe to run any time (used after scoring changes)."""
    con.execute("DELETE FROM mastery")
    st = states(con)
    # 1. placement priors (latest finished placement wins per skill)
    for r in con.execute("SELECT data FROM sessions WHERE kind='placement' AND status='done' ORDER BY id").fetchall():
        state = json.loads(r["data"]).get("state")
        if state:
            for sid, p in placement.priors(state).items():
                if sid in st:
                    mastery.set_prior(con, st, sid, p, "placement")
    # 2. graded sessions, in order
    n = 0
    for s in con.execute("SELECT id, graded_at FROM sessions WHERE status='graded' ORDER BY graded_at, id").fetchall():
        rows = con.execute("SELECT * FROM items WHERE session_id=? ORDER BY idx", (s["id"],)).fetchall()
        used_rubric = any(json.loads(r["rubric_checked"] or "[]") for r in rows)
        when = datetime.fromisoformat(s["graded_at"]) if s["graded_at"] else datetime.now()
        for r in rows:
            if r["correct"] is None or r["skill_id"] not in st:
                continue
            steps = json.loads(r["problem"]).get("rubric") or []
            checked = json.loads(r["rubric_checked"] or "[]")
            frac = (len(checked) / len(steps)) if (steps and used_rubric) else None
            mastery.record_attempt(con, st, r["skill_id"], bool(r["correct"]), frac, today=when)
            con.execute("UPDATE items SET score=? WHERE id=?", (mastery.combined_score(bool(r["correct"]), frac), r["id"]))
            n += 1
    # 3. passed labs
    for r in con.execute("SELECT lab_id FROM labs WHERE status='passed'").fetchall():
        lab = labreg.get_lab(r["lab_id"])
        for s in (lab.skills if lab else []):
            mastery.record_attempt(con, st, s, True, 1.0, propagate=False)
    db.log_event(con, "rebuild_mastery", detail={"answers": n})
    return {"answers_replayed": n}
