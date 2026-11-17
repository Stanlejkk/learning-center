# Learning Center

A private study app that runs entirely on your own PC. It takes you from an algebra refresh through calculus, linear algebra, probability, optimization, information theory, ML and LLMs, all the way to **quantum computing**.

## Requirements
- Windows, Linux or macOS with **[uv](https://docs.astral.sh/uv/)** installed (uv fetches Python 3.12 itself).
- Optional: an NVIDIA GPU for the deep-learning labs. `pyproject.toml` pulls PyTorch built for CUDA 12.8 (`cu128`), which covers RTX 20xx through RTX 50xx. Without a GPU everything else still works; the two GPU labs (`dl_05_gpt`, `dl_08_lora`) just run slowly on CPU.

## Install
```bash
git clone https://github.com/Stanlejkk/learning-center.git
cd learning-center
uv sync                                   # creates .venv with all dependencies (torch is ~3 GB)
uv run python scripts/fetch_library.py    # optional: downloads the ~65 MB offline PDF library
```
The PDFs are not in the repo because they belong to their authors and publishers. The script downloads them from the original public sources listed in `library/RESOURCES.md`. If one fails, download it by hand from the URL it prints and save it in `library/pdfs/`.

Check the GPU (optional):
```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Start
On Windows, double-click **`start.bat`**. Anywhere else, run `uv run python start.py`.
- App: http://127.0.0.1:8765, which opens in your browser.
- JupyterLab for the coding labs: http://127.0.0.1:8888, started automatically with a token saved in `data/jupyter_token.txt`.

Both servers listen on localhost only. Your progress, scans and the Jupyter token live in `data/`, which git ignores, so nothing personal is ever committed.

**First run:** take the placement test from the dashboard. It takes 20–40 minutes and decides where every track starts.

## How it works
1. **Placement test.** An adaptive binary search over each track's skills, about 3–6 questions per track. It sets a starting mastery estimate for all 110 skills.
2. **Homework on paper.**
   - Press *New homework set*. The problems stay hidden until you press start, either by printing the sheet or by starting on screen. There is no clock and time is never scored.
   - Solve everything by hand, then type only the **final answers**. They're checked symbolically with sympy, so `2(x+1)` and `2x+2` both count.
   - Optionally tick the **rubric steps** your handwritten work shows. A correct answer always counts strongly; ticks fine-tune it and give partial credit on wrong answers. Skip them and only answers count.
   - Each set mixes roughly 70% frontier skills, 20% spaced review and 10% stretch.
   - Optional: upload photos of your pages. They're stored in `data/scans/`.
3. **Gap detection.**
   - Mastery is tracked per skill with Bayesian Knowledge Tracing.
   - Wrong answers lower the estimate for that skill's prerequisites. Known wrong answers are tagged as named misconceptions (for example *added-denominators*).
   - Once a prerequisite drops below 60%, a short **remediation** set is offered.
4. **Spaced review.** Mastered skills come back at growing intervals.
5. **Checkpoint tests.** A test per track (untimed). Scoring 80% or more on a skill certifies it (★).
6. **Coding labs.**
   - Notebooks with `# TODO` stubs. A lab unlocks once its concept's paper homework is at least 50%.
   - The last cell, `grade("<lab>", globals())`, runs hidden checks and reports the result to the app.
   - The LLM labs build micrograd up to a GPT trained on your GPU, followed by BPE, sampling and LoRA.
   - The quantum labs use Qiskit and Aer.
7. **Library.** Landmark papers with pen-and-paper reading guides, plus per-skill readings from MIT, Harvard, Columbia, Cornell, Princeton, Stanford and Caltech, with PDFs saved under `library/pdfs/` (fetched by `scripts/fetch_library.py`).

## Layout
```
app/       FastAPI app (main.py routes, services.py session logic, db.py SQLite)
engine/    checker.py (answer equivalence), content.py (skill graph + generator registry),
           mastery.py (BKT), placement.py (adaptive test), scheduler.py (what next), labs.py
content/   tracks/*.yaml skill graph · problems/t0X_*.py parametric problem generators · readings.yaml
labs/      *.ipynb notebooks · tests/check_*.py hidden checks · solutions/ reference solutions · registry/
library/   papers.yaml, RESOURCES.md, pdfs/ (downloaded, not committed)
scripts/   fetch_library.py
data/      progress.db, scans/  (your data, back this folder up)
```

## Adding problems
Write a generator in `content/problems/` (see the docstring in `engine/content.py` and `t00_foundations.py`):
```python
@gen("la.eigen", levels=(1, 2, 3))
def my_problem(rng, level) -> Problem: ...
```
Then run `uv run pytest`. Every generator is checked with 50 seeds at every level, and its own answer must pass the checker.

## Tests
`uv run pytest`. Set `LC_SKIP_GPU_LABS=1` to skip the GPU lab checks, or `LC_SEEDS=10` for a faster generator sweep.

The solutions in `labs/solutions/` are reference answers. Don't open them until you've tried the lab.
