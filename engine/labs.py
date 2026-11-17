"""Coding-lab registry. Each track owns labs/registry/<track>.yaml:

labs:
  - id: dl_01_micrograd            # notebook = labs/<id>.ipynb, checks = labs/tests/check_<id>.py,
    title: "Build micrograd"       # reference solution = labs/solutions/<id>.py
    track: dl
    skills: [dl.backprop]          # credited when all checks pass
    unlock: [dl.backprop]          # paper skills that must be p >= UNLOCK_P first (the paper homework comes first)
    gpu: false
    minutes: 90
    summary: "..."
    sources: [{title: "...", url: "..."}]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LABS = ROOT / "labs"
UNLOCK_P = 0.50


@dataclass
class Lab:
    id: str
    title: str
    track: str
    skills: list[str]
    unlock: list[str]
    gpu: bool = False
    minutes: int = 60
    summary: str = ""
    sources: list[dict] = field(default_factory=list)

    @property
    def notebook(self) -> Path:
        return LABS / f"{self.id}.ipynb"

    @property
    def checks(self) -> Path:
        return LABS / "tests" / f"check_{self.id}.py"

    @property
    def solution(self) -> Path:
        return LABS / "solutions" / f"{self.id}.py"


@lru_cache(maxsize=1)
def load_labs() -> tuple[Lab, ...]:
    out = []
    for path in sorted((LABS / "registry").glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for d in data.get("labs", []) or []:
            out.append(Lab(id=d["id"], title=d["title"], track=d["track"], skills=list(d.get("skills", [])),
                           unlock=list(d.get("unlock", [])), gpu=bool(d.get("gpu", False)),
                           minutes=int(d.get("minutes", 60)), summary=d.get("summary", ""),
                           sources=list(d.get("sources", []) or [])))
    return tuple(out)


def get_lab(lab_id: str) -> Lab | None:
    return next((l for l in load_labs() if l.id == lab_id), None)
