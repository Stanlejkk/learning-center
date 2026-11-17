"""Learning Center — local web app. Run via start.bat / `uv run python start.py`."""
from __future__ import annotations

import html
import json
import os
import re
import shutil
from pathlib import Path

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import db, services
from engine import content, mastery, scheduler
from engine import labs as labreg

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="Learning Center", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "templates")

JUPYTER_PORT = int(os.environ.get("LC_JUPYTER_PORT", "8888"))


# ------------------------------------------------------------------ template helpers
_MATH = re.compile(r"(\$\$.+?\$\$|\$.+?\$)", re.S)


def md(text: str | None) -> str:
    """Tiny Markdown: **bold**, *italic*, `code`, paragraphs/line breaks, '- ' lists. Math ($...$) untouched."""
    if not text:
        return ""
    parts = _MATH.split(str(text))
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append(html.escape(part, quote=False))
            continue
        s = html.escape(part, quote=False)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"(?m)^- (.*)$", r"<li>\1</li>", s)
        s = s.replace("\n\n", "<br><br>").replace("\n", "<br>")
        out.append(s)
    return "".join(out)


def pct(p: float) -> int:
    return int(round(100 * (p or 0)))


def heat(p: float) -> str:
    """CSS class for a mastery value."""
    if p is None:
        return "h0"
    if p >= mastery.MASTERED:
        return "h5"
    if p >= mastery.FRONTIER_READY:
        return "h4"
    if p >= mastery.WEAK:
        return "h3"
    if p >= 0.3:
        return "h2"
    return "h1"


templates.env.filters["md"] = md
templates.env.filters["pct"] = pct
templates.env.filters["heat"] = heat
templates.env.globals["skills"] = content.skills_by_id


def render(request: Request, name: str, **ctx):
    return templates.TemplateResponse(request, name, ctx)


@app.on_event("startup")  # noqa: simple and sufficient here
def _startup():
    db.init_db()
    (db.DATA / "scans").mkdir(parents=True, exist_ok=True)


def jupyter_url(path: str = "") -> str:
    tok_file = db.DATA / "jupyter_token.txt"
    tok = tok_file.read_text().strip() if tok_file.exists() else ""
    base = f"http://127.0.0.1:{JUPYTER_PORT}/lab/tree/{path}"
    return f"{base}?token={tok}" if tok else base


def load_readings() -> dict:
    p = ROOT / "content" / "readings.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {} if p.exists() else {}


def reading_links(skill_id: str) -> list[dict]:
    """Readings for a skill, each with an optional link to the offline PDF copy."""
    out = []
    sk = content.skills_by_id().get(skill_id)
    for r in load_readings().get(skill_id, []) + (sk.readings if sk else []):
        r = dict(r)
        if r.get("local"):
            name = Path(r["local"]).name
            if (ROOT / "library" / "pdfs" / name).exists():
                r["local_url"] = f"/library/pdf/{name}"
        out.append(r)
    return out


def study_plan(skill_ids: list[str], st) -> list[dict]:
    """What to study before a homework set: per distinct skill, summary + readings + a worked example."""
    seen, out = set(), []
    for sid in skill_ids:
        if sid in seen or sid not in content.skills_by_id():
            continue
        seen.add(sid)
        sk = content.skills_by_id()[sid]
        ex = content.make_problem(sid, 1, 424242)  # fixed seed: a stable worked example, not one of your problems
        out.append({"skill": sk, "p": st[sid].p, "assessed": st[sid].assessed,
                    "new": st[sid].p < mastery.WEAK, "readings": reading_links(sid),
                    "example": ex, "minutes": sum(int(r.get("minutes") or 0) for r in reading_links(sid)[:2])})
    return out


def load_formulas() -> dict:
    p = ROOT / "content" / "formulas.yaml"
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.exists() else {}


def formula_sheet(skill_ids: list[str] | None = None, with_prereqs: bool = True) -> list[dict]:
    """Formula sheet grouped by track. skill_ids=None -> everything; otherwise those skills (+ prerequisites)."""
    f = load_formulas()
    if skill_ids is None:
        wanted = set(content.skills_by_id())
    else:
        wanted = {s for s in skill_ids if s in content.skills_by_id()}
        if with_prereqs:
            for s in list(wanted):
                wanted |= content.ancestors(s)
    groups = []
    for t in content.load_tracks():
        rows = [{"skill": s, "formulas": f.get(s.id, []), "in_set": skill_ids is not None and s.id in skill_ids}
                for s in t.skills if s.id in wanted and f.get(s.id)]
        if rows:
            groups.append({"track": t, "rows": rows})
    return groups


def load_papers() -> list:
    p = ROOT / "library" / "papers.yaml"
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or []
    return data.get("papers", data) if isinstance(data, dict) else data


# ------------------------------------------------------------------ dashboard
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    with db.connect() as con:
        st = services.states(con)
        tracks = []
        for t in content.load_tracks():
            ps = [st[s.id].p for s in t.skills]
            tracks.append({"t": t, "mastered": sum(p >= mastery.MASTERED for p in ps), "n": len(ps),
                           "avg": sum(ps) / len(ps),
                           "certified": sum(st[s.id].certified for s in t.skills)})
        recent = [services.get_session(con, r["id"]) for r in
                  con.execute("SELECT id FROM sessions ORDER BY id DESC LIMIT 8").fetchall()]
        open_sessions = [s for s in recent if s["status"] in ("created", "started", "submitted")]
        acts = scheduler.next_actions(st)
        act = services.activity(con)
        placement_active = services.placement_session(con)
        total = len(st)
        mastered = sum(s.p >= mastery.MASTERED for s in st.values())
        # progress on the path to quantum: only real evidence counts (unassessed skills sit at the
        # P_INIT baseline and contribute 0); a skill counts fully once mastered
        span = mastery.MASTERED - mastery.P_INIT
        path_pct = sum(min(max(s.p - mastery.P_INIT, 0) / span, 1) for s in st.values() if s.assessed) / len(st)
        due = scheduler.due_reviews(st)
    return render(request, "dashboard.html", tracks=tracks, st=st, recent=recent, open_sessions=open_sessions,
                  acts=acts, act=act, placement_active=placement_active, total=total, mastered=mastered,
                  path_pct=path_pct, due=due, kind_titles=services.KIND_TITLES)


# ------------------------------------------------------------------ placement
@app.get("/placement", response_class=HTMLResponse)
def placement_page(request: Request):
    with db.connect() as con:
        sess = services.placement_session(con)
        if not sess:
            done = con.execute("SELECT id FROM sessions WHERE kind='placement' AND status='done' ORDER BY id DESC LIMIT 1").fetchone()
            return render(request, "placement_intro.html", tracks=content.load_tracks(), last_done=done)
        item, prob = services.placement_current(con, sess)
        if item is None:
            services.finish_placement(con, sess)
            return RedirectResponse(f"/placement/result/{sess['id']}", status_code=303)
        prog = services.placement_progress(sess)
    return render(request, "placement_q.html", item=item, p=prob, prog=prog, sess=sess)


@app.post("/placement/start")
def placement_start(tracks: list[str] = Form(default=[])):
    with db.connect() as con:
        if not services.placement_session(con):
            services.start_placement(con, tracks or None)
    return RedirectResponse("/placement", status_code=303)


@app.post("/placement/answer")
def placement_answer(answer: str = Form(default=""), action: str = Form(default="submit")):
    with db.connect() as con:
        sess = services.placement_session(con)
        if not sess:
            return RedirectResponse("/placement", status_code=303)
        if action == "finish":
            services.finish_placement(con, sess)
            return RedirectResponse(f"/placement/result/{sess['id']}", status_code=303)
        services.placement_answer(con, sess, "skip" if action == "dontknow" else answer,
                                  skip_track=(action == "skiptrack"))
    return RedirectResponse("/placement", status_code=303)


@app.get("/placement/result/{sid}", response_class=HTMLResponse)
def placement_result(request: Request, sid: int):
    with db.connect() as con:
        sess = services.get_session(con, sid)
        st = services.states(con)
        items = services.get_items(con, sid)
        front = scheduler.frontier(st, limit=6)
    return render(request, "placement_result.html", sess=sess, st=st, tracks=content.load_tracks(),
                  items=items, front=front)


# ------------------------------------------------------------------ sessions
@app.post("/homework/new")
def homework_new(n: int = Form(default=8), track: str = Form(default="")):
    with db.connect() as con:
        sid = services.new_homework(con, n=max(3, min(n, 20)), track=track or None)
    return RedirectResponse(f"/session/{sid}", status_code=303)


@app.post("/practice/{skill_id}")
def practice_new(skill_id: str):
    if skill_id not in content.skills_by_id():
        raise HTTPException(404)
    with db.connect() as con:
        try:
            sid = services.new_practice(con, skill_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    return RedirectResponse(f"/session/{sid}", status_code=303)


@app.post("/remediation/new")
def remediation_new(skills: list[str] = Form(...)):
    with db.connect() as con:
        sid = services.new_remediation(con, [s for s in skills if s in content.skills_by_id()])
    return RedirectResponse(f"/session/{sid}", status_code=303)


@app.post("/review/new")
def review_new():
    with db.connect() as con:
        try:
            sid = services.new_review(con)
        except ValueError:
            return RedirectResponse("/", status_code=303)
    return RedirectResponse(f"/session/{sid}", status_code=303)


@app.post("/checkpoint/{track}")
def checkpoint_new(track: str):
    with db.connect() as con:
        sid = services.new_checkpoint(con, track)
    return RedirectResponse(f"/session/{sid}", status_code=303)


@app.get("/session/{sid}", response_class=HTMLResponse)
def session_page(request: Request, sid: int):
    with db.connect() as con:
        sess = services.get_session(con, sid)
        if not sess or sess["kind"] == "placement":
            raise HTTPException(404)
        items = services.get_items(con, sid)
        scans = con.execute("SELECT * FROM scans WHERE session_id=?", (sid,)).fetchall()
        st = services.states(con)
    study = study_plan([it["skill_id"] for it in items], st)
    # after grading: what to re-read for the problems you got wrong
    wrong_skills = [it["skill_id"] for it in items if it["correct"] == 0]
    reread = study_plan(wrong_skills, st) if sess["status"] in ("submitted", "graded") else []
    sheet = formula_sheet([it["skill_id"] for it in items])
    return render(request, "session.html", sess=sess, items=items, scans=scans, st=st, study=study, reread=reread,
                  sheet=sheet)


@app.get("/session/{sid}/print", response_class=HTMLResponse)
def session_print(request: Request, sid: int):
    with db.connect() as con:
        services.start_session(con, sid)
        sess = services.get_session(con, sid)
        items = services.get_items(con, sid)
    sheet = formula_sheet([it["skill_id"] for it in items]) if request.query_params.get("formulas", "1") != "0" else []
    return render(request, "print.html", sess=sess, items=items, sheet=sheet)


@app.get("/formulas", response_class=HTMLResponse)
def formulas_page(request: Request):
    track = request.query_params.get("track") or None
    groups = formula_sheet(None)
    if track:
        groups = [g for g in groups if g["track"].id == track]
    return render(request, "formulas.html", groups=groups, tracks=content.load_tracks(), track=track,
                  total=sum(len(r["formulas"]) for g in groups for r in g["rows"]))


@app.post("/session/{sid}/start")
def session_start(sid: int):
    with db.connect() as con:
        services.start_session(con, sid)
    return RedirectResponse(f"/session/{sid}", status_code=303)


@app.post("/session/{sid}/submit")
async def session_submit(request: Request, sid: int):
    form = await request.form()
    answers = {int(k[2:]): str(v) for k, v in form.items() if k.startswith("a_")}
    with db.connect() as con:
        sess = services.get_session(con, sid)
        if sess["status"] not in ("started", "created"):
            return RedirectResponse(f"/session/{sid}", status_code=303)
        services.submit_answers(con, sid, answers)
    return RedirectResponse(f"/session/{sid}#review", status_code=303)


@app.post("/session/{sid}/grade")
async def session_grade(request: Request, sid: int):
    form = await request.form()
    rubric: dict[int, list[int]] = {}
    for k in form.keys():
        if k.startswith("r_"):
            _, item_id, step = k.split("_")
            rubric.setdefault(int(item_id), []).append(int(step))
    with db.connect() as con:
        services.grade_session(con, sid, rubric)
    return RedirectResponse(f"/session/{sid}#summary", status_code=303)


@app.post("/session/{sid}/scan")
async def session_scan(sid: int, files: list[UploadFile] = File(...)):
    folder = db.DATA / "scans" / f"session_{sid}"
    folder.mkdir(parents=True, exist_ok=True)
    with db.connect() as con:
        for f in files:
            if not f.filename:
                continue
            ext = Path(f.filename).suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".pdf"}:
                continue
            n = len(list(folder.iterdir())) + 1
            dest = folder / f"page_{n:02d}{ext}"
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            con.execute("INSERT INTO scans(session_id, path, uploaded_at) VALUES (?,?,?)",
                        (sid, str(dest.relative_to(db.DATA)), db.now()))
    return RedirectResponse(f"/session/{sid}#scans", status_code=303)


@app.get("/scan/{scan_id}")
def scan_file(scan_id: int):
    from fastapi.responses import FileResponse  # noqa: PLC0415
    with db.connect() as con:
        r = con.execute("SELECT path FROM scans WHERE id=?", (scan_id,)).fetchone()
    if not r:
        raise HTTPException(404)
    path = (db.DATA / r["path"]).resolve()
    if db.DATA.resolve() not in path.parents:
        raise HTTPException(403)
    return FileResponse(path)


@app.post("/session/{sid}/delete")
def session_delete(sid: int):
    with db.connect() as con:
        sess = services.get_session(con, sid)
        if sess and sess["status"] != "graded":
            con.execute("DELETE FROM items WHERE session_id=?", (sid,))
            con.execute("DELETE FROM sessions WHERE id=?", (sid,))
    return RedirectResponse("/", status_code=303)


# ------------------------------------------------------------------ skills
@app.get("/skills", response_class=HTMLResponse)
def skills_page(request: Request):
    with db.connect() as con:
        st = services.states(con)
    front = set(scheduler.frontier(st, limit=200))
    return render(request, "skills.html", tracks=content.load_tracks(), st=st, front=front,
                  has_gen=lambda s: bool(content.generators_for(s)))


@app.get("/skill/{skill_id}", response_class=HTMLResponse)
def skill_page(request: Request, skill_id: str):
    sk = content.skills_by_id().get(skill_id)
    if not sk:
        raise HTTPException(404)
    with db.connect() as con:
        st = services.states(con)
        hist = con.execute("SELECT i.*, s.kind FROM items i JOIN sessions s ON s.id=i.session_id "
                           "WHERE i.skill_id=? AND i.correct IS NOT NULL ORDER BY i.id DESC LIMIT 20",
                           (skill_id,)).fetchall()
        labs = [l for l in services.lab_status(con, st) if skill_id in l["lab"].skills]
    readings = reading_links(skill_id)
    children = sorted(content.descendants(skill_id) & {s.id for s in content.skills_by_id().values()
                                                         if skill_id in s.prereqs},
                      key=lambda x: content.skills_by_id()[x].order)
    sample = content.make_problem(skill_id, scheduler.level_for(st[skill_id].p), 1)
    papers = [p for p in load_papers() if skill_id in (p.get("skills") or [])]
    return render(request, "skill.html", sk=sk, st=st, s=st[skill_id], hist=hist, readings=readings,
                  track=content.track_of(skill_id), children=children, labs=labs, sample=sample,
                  blocked=scheduler.blocked_by(st, skill_id), papers=papers)


# ------------------------------------------------------------------ labs
@app.get("/labs", response_class=HTMLResponse)
def labs_page(request: Request):
    with db.connect() as con:
        st = services.states(con)
        rows = services.lab_status(con, st)
    return render(request, "labs.html", rows=rows, tracks=content.load_tracks(), jupyter_url=jupyter_url)


@app.post("/api/labs/submit")
async def api_lab_submit(request: Request):
    report = await request.json()
    with db.connect() as con:
        try:
            status = services.record_lab(con, report)
        except ValueError as exc:
            return JSONResponse({"status": "error", "error": str(exc)}, status_code=400)
    return {"status": status}


# ------------------------------------------------------------------ library & history
@app.get("/library", response_class=HTMLResponse)
def library_page(request: Request):
    papers = load_papers()
    pdf_dir = ROOT / "library" / "pdfs"
    pdfs = sorted(p.name for p in pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    readings = load_readings()
    return render(request, "library.html", papers=papers, pdfs=pdfs, tracks=content.load_tracks(),
                  readings=readings)


@app.get("/library/pdf/{name}")
def library_pdf(name: str):
    from fastapi.responses import FileResponse  # noqa: PLC0415
    path = (ROOT / "library" / "pdfs" / name).resolve()
    if path.parent != (ROOT / "library" / "pdfs").resolve() or not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="application/pdf")


@app.get("/history", response_class=HTMLResponse)
def history_page(request: Request):
    with db.connect() as con:
        sessions = [dict(r) for r in con.execute("SELECT * FROM sessions ORDER BY id DESC").fetchall()]
        for s in sessions:
            s["data"] = json.loads(s["data"] or "{}")
        act = services.activity(con)
        mis = con.execute("SELECT misconception, skill_id, COUNT(*) n FROM items WHERE misconception IS NOT NULL "
                          "AND misconception != 'unknown' GROUP BY misconception, skill_id ORDER BY n DESC LIMIT 15").fetchall()
    return render(request, "history.html", sessions=sessions, act=act, mis=mis, kind_titles=services.KIND_TITLES)


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    return render(request, "settings.html", db_path=db.db_path(), data_dir=db.DATA)


@app.post("/settings/rebuild")
def settings_rebuild():
    with db.connect() as con:
        services.rebuild_mastery(con)
    return RedirectResponse("/", status_code=303)


@app.post("/settings/reset")
def settings_reset(confirm: str = Form(default="")):
    if confirm.strip().upper() != "RESET":
        return RedirectResponse("/settings?err=1", status_code=303)
    with db.connect() as con:
        backup = db.DATA / f"progress_backup_{db.now().replace(':', '-')}.db"
        con.execute(f"VACUUM INTO '{backup.as_posix()}'")
        for t in ("items", "sessions", "mastery", "labs", "scans", "events"):
            con.execute(f"DELETE FROM {t}")
    return RedirectResponse("/", status_code=303)
