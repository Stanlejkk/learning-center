"""Content framework: skill graph (YAML) + parametric problem generators (Python).

Authoring a generator
---------------------
In content/problems/<track>.py:

    from engine.content import gen, Problem, R

    @gen("calc.derivative_rules", levels=(1, 2, 3))
    def derivative_rules(rng: random.Random, level: int) -> Problem:
        a = rng.randint(2, 9)
        return Problem(
            prompt=f"Differentiate $f(x) = {a}x^3$.",
            answer=f"{3*a}*x^2", kind="expr",
            rubric=["Apply the power rule", "Multiply the coefficient"],
            solution=f"$f'(x) = {3*a}x^2$",
            misconceptions=[(f"{a}*x^2", "power-rule-forgot-multiply")],
        )

* `prompt` and `solution` are Markdown with $inline$ / $$display$$ KaTeX math.
* `answer` is a string (or list for list/vector/set kinds) the checker understands.
* `rubric` lists the steps the user should see in their handwritten working;
  after submitting they tick the steps they actually did correctly.
* `answer_hint` tells the user the expected format (e.g. "a matrix like [[1,2],[3,4]]").
* Generators MUST be deterministic given rng, and must work for every listed level.
"""
from __future__ import annotations

import importlib
import pkgutil
import random
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content"


@dataclass
class Problem:
    prompt: str
    answer: object
    kind: str = "expr"
    rubric: list[str] = field(default_factory=list)
    solution: str = ""
    answer_hint: str = ""
    tol: float | None = None
    misconceptions: list = field(default_factory=list)
    hint: str = ""
    # filled in by the registry
    skill: str = ""
    level: int = 1
    gen_id: str = ""
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Generator:
    gen_id: str
    skill: str
    levels: tuple[int, ...]
    fn: Callable[[random.Random, int], Problem]

    def make(self, seed: int, level: int) -> Problem:
        lvl = level if level in self.levels else min(self.levels, key=lambda l: abs(l - level))
        rng = random.Random(f"{self.gen_id}:{seed}:{lvl}")
        p = self.fn(rng, lvl)
        p.skill, p.level, p.gen_id, p.seed = self.skill, lvl, self.gen_id, seed
        if not p.answer_hint:
            p.answer_hint = DEFAULT_HINTS.get(p.kind, "")
        if p.misconceptions:  # drop distractors that coincide with the right answer for this seed
            from engine.checker import check  # noqa: PLC0415
            keep = []
            for wrong, tag in p.misconceptions:
                typed = ", ".join(map(str, wrong)) if isinstance(wrong, (list, tuple)) and p.kind != "matrix" else wrong
                try:
                    if not check(str(typed), p.answer, p.kind, p.tol).correct:
                        keep.append((wrong, tag))
                except Exception:  # noqa: BLE001
                    pass
            p.misconceptions = keep
        return p


DEFAULT_HINTS = {
    "expr": "an expression, e.g. 3*x^2 + 2*x  or  sqrt(2)/2",
    "num": "a number or exact value, e.g. 0.25, 1/4, pi/3",
    "matrix": "a matrix as rows, e.g. [[1,2],[3,4]]",
    "vector": "a vector, e.g. [1, -2, 3]",
    "list": "comma-separated values in order, e.g. 1, 2, 3",
    "set": "comma-separated values in any order, e.g. 2, -3",
    "choice": "the letter of your choice, e.g. B",
    "text": "a short word or phrase",
}

_REGISTRY: dict[str, list[Generator]] = {}


def gen(skill: str, levels=(1, 2, 3), name: str | None = None):
    """Decorator registering a problem generator for a skill."""
    def deco(fn):
        g = Generator(gen_id=f"{skill}/{name or fn.__name__}", skill=skill,
                      levels=tuple(levels), fn=fn)
        _REGISTRY.setdefault(skill, []).append(g)
        return fn
    return deco


_loaded = False


def load_generators() -> dict[str, list[Generator]]:
    global _loaded
    if not _loaded:
        import content.problems as pkg  # noqa: PLC0415
        for mod in pkgutil.iter_modules(pkg.__path__):
            importlib.import_module(f"content.problems.{mod.name}")
        _loaded = True
    return _REGISTRY


def generators_for(skill: str) -> list[Generator]:
    return load_generators().get(skill, [])


def get_generator(gen_id: str) -> Generator | None:
    for gens in load_generators().values():
        for g in gens:
            if g.gen_id == gen_id:
                return g
    return None


def make_problem(skill: str, level: int, seed: int) -> Problem | None:
    gens = generators_for(skill)
    if not gens:
        return None
    usable = [g for g in gens if level in g.levels] or gens
    g = usable[seed % len(usable)]
    return g.make(seed, level)


# ---------------------------------------------------------------- skill graph
@dataclass
class Skill:
    id: str
    name: str
    track: str
    order: int
    prereqs: list[str]
    summary: str = ""
    readings: list[dict] = field(default_factory=list)
    labs: list[str] = field(default_factory=list)


@dataclass
class Track:
    id: str
    name: str
    order: int
    description: str
    sources: list[dict]
    skills: list[Skill]


@lru_cache(maxsize=1)
def load_tracks() -> tuple[Track, ...]:
    tracks = []
    for path in sorted((CONTENT / "tracks").glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        skills = []
        for i, s in enumerate(data.get("skills", [])):
            skills.append(Skill(
                id=s["id"], name=s["name"], track=data["id"], order=i,
                prereqs=list(s.get("prereqs", []) or []), summary=s.get("summary", ""),
                readings=list(s.get("readings", []) or []), labs=list(s.get("labs", []) or []),
            ))
        tracks.append(Track(id=data["id"], name=data["name"], order=int(data["order"]),
                            description=data.get("description", ""),
                            sources=list(data.get("sources", []) or []), skills=skills))
    tracks.sort(key=lambda t: t.order)
    return tuple(tracks)


@lru_cache(maxsize=1)
def skills_by_id() -> dict[str, Skill]:
    return {s.id: s for t in load_tracks() for s in t.skills}


def track_of(skill_id: str) -> Track:
    sk = skills_by_id()[skill_id]
    return next(t for t in load_tracks() if t.id == sk.track)


def all_skills_ordered() -> list[Skill]:
    """Global curriculum order: by track, then within-track order."""
    return [s for t in load_tracks() for s in t.skills]


def ancestors(skill_id: str) -> set[str]:
    sk = skills_by_id()
    out, stack = set(), list(sk[skill_id].prereqs)
    while stack:
        p = stack.pop()
        if p in out or p not in sk:
            continue
        out.add(p)
        stack.extend(sk[p].prereqs)
    return out


def descendants(skill_id: str) -> set[str]:
    sk = skills_by_id()
    children: dict[str, list[str]] = {}
    for s in sk.values():
        for p in s.prereqs:
            children.setdefault(p, []).append(s.id)
    out, stack = set(), list(children.get(skill_id, []))
    while stack:
        c = stack.pop()
        if c in out:
            continue
        out.add(c)
        stack.extend(children.get(c, []))
    return out


def validate_graph() -> list[str]:
    """Return a list of problems with the skill graph (missing prereqs, cycles, no generators)."""
    errs = []
    sk = skills_by_id()
    gens = load_generators()
    for s in sk.values():
        for p in s.prereqs:
            if p not in sk:
                errs.append(f"{s.id}: unknown prereq {p}")
        if s.id not in gens:
            errs.append(f"{s.id}: no generators")
        if s.id in ancestors(s.id):
            errs.append(f"{s.id}: cycle")
    for g in gens:
        if g not in sk:
            errs.append(f"generator for unknown skill {g}")
    return errs
