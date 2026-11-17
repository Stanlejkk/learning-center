# Resource library

The curated sources behind `content/readings.yaml` (per-skill deep links) and
`library/papers.yaml` (landmark papers with reading guides).

* **Verification.** Every URL in `readings.yaml` and `papers.yaml`, and every URL in this file
  that is not marked otherwise, was fetched on **2026-11-17**. Each returned HTTP 200, and its
  title or content matched the topic. A few sites block automated clients (HTTP 403). Those were
  checked with a web search only, are marked *(search-verified)* and are **not** used in `readings.yaml`.
* **Licenses.** Licenses are as stated on each site on that date. "All rights reserved / free to
  read" means the author or publisher hosts the material free online but grants no reuse license.
  Read it online, and keep a local copy only where the terms below allow it.
* **Local copies** live in `library/pdfs/` (26 files, 67.1 MB in total). They are for personal
  offline study only. Do not redistribute them.

---

## 0 · Foundations (algebra, functions, trig, sums, induction, complex numbers)

| Course / book | Institution | URL | License | Use it for | Local |
|---|---|---|---|---|---|
| OpenStax *Prealgebra 2e*, *College Algebra 2e*, *Precalculus 2e* | Rice University (OpenStax) | https://openstax.org/books/college-algebra-2e/pages/1-4-polynomials (section deep links in readings.yaml) | CC BY-NC-SA 4.0 (from the book metadata on the page) | Short, worked-example sections for fractions, exponents, polynomials, quadratics, functions, logs, trig and complex numbers | – |
| MIT 6.042J *Mathematics for Computer Science*, Fall 2010 video lectures (Leighton) | MIT | https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-fall-2010/ | CC BY-NC-SA 4.0 | Induction (L2–L3), sums (L12–13), counting (L16–17) | – |
| *Mathematics for Computer Science* textbook (Lehman, Leighton, Meyer), 6.042J Spring 2015 | MIT | https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/resources/mit6_042js15_textbook_pdf/ | CC BY-NC-SA 4.0 | Ch. 5 Induction, Ch. 13 Sums & Asymptotics, Ch. 14 Cardinality Rules (counting), Ch. 16–19 probability | `mit_6042_mcs.pdf` |
| MIT 18.03SC: Complex Arithmetic and Exponentials | MIT | https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/pages/unit-i-first-order-differential-equations/complex-arithmetic-and-exponentials/ | CC BY-NC-SA 4.0 | Euler's formula and complex exponentials before the quantum track | – |

## 1 · Calculus

| Course / book | Institution | URL | License | Use it for | Local |
|---|---|---|---|---|---|
| MIT 18.01SC Single Variable Calculus (Jerison) | MIT | https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/ | CC BY-NC-SA 4.0 | One "session" page per topic (video, notes, problems and solutions); deep-linked per skill | – |
| MIT 18.02SC Multivariable Calculus (Auroux) | MIT | https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/ | CC BY-NC-SA 4.0 | Partial derivatives, gradient, chain rule, Lagrange multipliers, double integrals | – |
| MIT 18.S096 Matrix Calculus for ML and Beyond (Edelman & Johnson, IAP 2023) | MIT | https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/ ; notes: https://arxiv.org/abs/2501.14787 | CC BY-NC-SA 4.0 (course and arXiv notes) | Derivatives as linear maps, Jacobians, gradients of matrix functions | – |
| 3Blue1Brown *Essence of Calculus* | 3Blue1Brown | https://www.3blue1brown.com/lessons/essence-of-calculus | Standard YouTube/website terms (free to watch) | Visual intuition before the formal sessions | – |

## 2 · Linear algebra

| Course / book | Institution | URL | License | Use it for | Local |
|---|---|---|---|---|---|
| MIT 18.06SC Linear Algebra (Strang) | MIT | https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/ | CC BY-NC-SA 4.0 | The core lectures: elimination, the four subspaces, projections, determinants, eigenvalues, SVD | – |
| MIT 18.065 Matrix Methods in Data Analysis, Signal Processing, and ML (Strang) | MIT | https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/ | CC BY-NC-SA 4.0 | Bridge to ML: SVD, Eckart–Young, least squares, gradient descent, SGD, momentum, backprop, CNNs (videos deep-linked) | – |
| Daniel Hsu, linear-algebra notes (SVD, eigenvectors) | Columbia | https://www.cs.columbia.edu/~djhsu/CLA/notes/svd.pdf | All rights reserved / free to read | Concise proofs of the spectral theorem and SVD | – |
| 3Blue1Brown *Essence of Linear Algebra* | 3Blue1Brown | https://www.3blue1brown.com/lessons/vectors | Free to watch | Geometric intuition: span, determinant, change of basis, eigenvectors | – |

## 3 · Probability & statistics

| Course / book | Institution | URL | License | Use it for | Local |
|---|---|---|---|---|---|
| Harvard Stat 110 (Blitzstein): lecture videos | Harvard | https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo | Free to watch (YouTube) | The best probability lectures available; individual lectures deep-linked per skill | – |
| Harvard Stat 110 course site (handouts, strategic practice, review) | Harvard | https://stat110.hsites.harvard.edu/ *(search-verified; returns 403 to automated clients)* | All rights reserved / free to read | Strategic-practice problems with solutions | – |
| Blitzstein & Hwang, *Introduction to Probability* 2e, free online edition | Harvard | https://probabilitybook.net (opens the authors' view-only Google Drive copy) | All rights reserved (CRC Press). The online copy is view-only and not downloadable | Ch. 1–4 counting through expectation; Ch. 7 joint distributions; Ch. 10 limit theorems; Ch. 11 Markov chains | **Not downloaded**: the authors share it as view-only, not as a download |
| MIT 18.05 Introduction to Probability and Statistics (Spring 2022) | MIT | https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/ | CC BY-NC-SA 4.0 | Short class readings (PDF pages deep-linked): MLE, Bayesian updating, NHST, confidence intervals, bootstrap | – |
| MIT 6.041SC Probabilistic Systems Analysis (Tsitsiklis) | MIT | https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/ | CC BY-NC-SA 4.0 | Rigorous lectures on axioms, conditioning, random variables, CLT, Markov chains and inference | – |
| Seeing Theory | Brown University | https://seeing-theory.brown.edu/ | Free to use online (no license stated on the site) | Interactive visualizations of probability, distributions and inference | – |

## 4 · Optimization

| Course / book | Institution | URL | License | Use it for | Local |
|---|---|---|---|---|---|
| Boyd & Vandenberghe, *Convex Optimization* | Stanford | https://web.stanford.edu/~boyd/cvxbook/ | © Cambridge University Press, which allows the book to stay free on the web | Ch. 2–3 convex sets and functions; Ch. 5 duality and KKT; Ch. 6 least squares and regularization; Ch. 9 gradient descent, condition number, Newton | `boyd_convex_optimization.pdf` (from https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) |
| *Why Momentum Really Works* (Goh) | Distill | https://distill.pub/2017/momentum/ | CC BY 2.0 | Interactive conditioning and momentum analysis | – |
| Cornell CS4780 (Weinberger) lecture notes | Cornell | https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/ | All rights reserved / free to read | GD and Newton (L7), ERM and regularizers (L10) | – |
| Adam / AdamW papers, Ruder's optimizer overview | arXiv | see papers.yaml / readings.yaml | arXiv non-exclusive license | Adaptive methods | `adam.pdf` |

## 5 · Information theory (and data analysis)

| Course / book | Institution | URL | License | Use it for | Local |
|---|---|---|---|---|---|
| MacKay, *Information Theory, Inference, and Learning Algorithms* | University of Cambridge | https://www.inference.org.uk/itprnn/book.pdf (page: https://www.inference.org.uk/mackay/itila/) | © D. MacKay / Cambridge University Press. The author's page offers the whole-book PDF as a download for on-screen viewing | Ch. 2 entropy and relative entropy; Ch. 4 source coding theorem; Ch. 5 Kraft and Huffman; Ch. 8 joint entropy and MI; Ch. 9–10 channel capacity; Ch. 20 and 22 clustering | `mackay_itila.pdf` |
| MacKay lecture videos | University of Cambridge | https://www.inference.org.uk/itprnn_lectures/ | Free to watch | Lectures that accompany the book | – |
| Shannon (1948), *A Mathematical Theory of Communication* | Bell Labs (copy hosted by Harvard Math) | https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf | © Bell System Technical Journal. Hosted copy | The original paper (see papers.yaml) | **Not downloaded**: not on arXiv and the reuse terms are unclear |
| MIT 6.050J Information and Entropy (Penfield), course textbook | MIT | https://ocw.mit.edu/courses/6-050j-information-and-entropy-spring-2008/ | CC BY-NC-SA 4.0 | A gentle, unified treatment of bits, codes, compression and entropy | `mit_6050j_info_entropy.pdf` |
| MIT 6.441 Information Theory (graduate) | MIT | https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/ | CC BY-NC-SA 4.0 | Next step after MacKay (rigorous) | – |
| Harvard CS109A Introduction to Data Science (2019) | Harvard | https://harvard-iacs.github.io/2019-CS109A/ | MIT (repo Harvard-IACS/2019-CS109A) | EDA (L2), model selection and regularization (L7), PCA (L14), trees (L15), experimental design and A/B testing (L23–24) | – |

## 6 · Machine learning

| Course / book | Institution | URL | License | Use it for | Local |
|---|---|---|---|---|---|
| Cornell CS4780 Machine Learning (Weinberger) | Cornell | https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/ | All rights reserved / free to read | Clear one-page notes: kNN, MLE/MAP, naive Bayes, logistic regression, SVM, kernels, bias-variance, trees | – |
| Columbia COMS 4771 Machine Learning (Daniel Hsu) | Columbia | https://www.cs.columbia.edu/~djhsu/ML/ | All rights reserved / free to read | Rigorous handouts: prediction theory, linear and logistic regression, kernels, regularization, multiclass, calibration, language models. URLs point to the current offering and may change each semester | – |
| Stanford CS229 Machine Learning main notes | Stanford | https://cs229.stanford.edu/main_notes.pdf | All rights reserved / free to read | Chapters on linear regression, logistic regression, GLMs/softmax, generative models, kernels, SVMs, generalization, regularization, k-means, EM, PCA, KL facts | **Not downloaded**: no explicit license |
| MIT 6.036 Introduction to Machine Learning | MIT | https://ocw.mit.edu/courses/6-036-introduction-to-machine-learning-fall-2020/ | CC BY-NC-SA 4.0 | Alternative full course | – |
| Google ML Crash Course: ROC and AUC | Google | https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc | CC BY 4.0 (Google Developers site default) | Evaluation metrics | – |

## 7 · Deep learning & LLMs

| Course / book | Institution | URL | License | Use it for | Local |
|---|---|---|---|---|---|
| Karpathy, *Neural Networks: Zero to Hero* (videos) | – | https://github.com/karpathy/nn-zero-to-hero | MIT (code) | micrograd/backprop, makemore (bigram, MLP, BatchNorm), GPT from scratch, tokenizer, GPT-2 reproduction | – |
| Stanford CS231n notes | Stanford | https://cs231n.github.io/ | MIT (repo cs231n/cs231n.github.io) | Backprop, initialization, BatchNorm, optimizers, CNNs, softmax classifier | – |
| Stanford CS224N NLP with Deep Learning | Stanford | https://web.stanford.edu/class/cs224n/ | All rights reserved / free to read | Word vectors; self-attention and transformer notes | – |
| Stanford CS336 Language Modeling from Scratch (2025) | Stanford | https://stanford-cs336.github.io/spring2025/ | Course site free. Assignments under MIT | Build an LM end to end; scaling laws; alignment | – |
| Jurafsky & Martin, *Speech and Language Processing* 3e draft | Stanford | https://web.stanford.edu/~jurafsky/slp3/ | All rights reserved / free draft | Ch. 2 tokens/BPE, Ch. 3 n-grams and perplexity, Ch. 4 logistic regression, Ch. 5 embeddings, Ch. 7 transformers, Ch. 8 post-training, App. B naive Bayes (chapter numbering of the Aug 2025 draft) | – |
| Harvard NLP, *The Annotated Transformer* | Harvard | https://nlp.seas.harvard.edu/annotated-transformer/ | MIT (repo harvardnlp/annotated-transformer) | Line-by-line PyTorch implementation of the transformer paper | – |
| MIT 6.S191 Introduction to Deep Learning | MIT | https://introtodeeplearning.com/ | Free; labs repo MITDeepLearning/introtodeeplearning is MIT-licensed | Survey lectures and labs | – |
| 3Blue1Brown *Neural Networks* series | 3Blue1Brown | https://www.3blue1brown.com/lessons/neural-networks | Free to watch | Visual intuition for MLPs, backprop, attention and GPT | – |
| Hugging Face, *How to generate text* | Hugging Face | https://huggingface.co/blog/how-to-generate | Free to read | Greedy, beam, top-k and top-p decoding | – |
| kipply, *Transformer Inference Arithmetic* | – | https://kipp.ly/transformer-inference-arithmetic/ | Free to read | KV-cache memory and FLOPs vs. bandwidth math | – |
| Landmark papers (transformer, GPT-2/3, scaling laws, Chinchilla, InstructGPT, LoRA, FlashAttention, DPO, LayerNorm, Bengio 2003, VAE) | various | see `library/papers.yaml` | arXiv non-exclusive license. DPO is CC BY 4.0. JMLR is open access. GPT-2 is an OpenAI report | Primary sources with reading guides | see table below |

## 8 · Quantum computing

| Course / book | Institution | URL | License | Use it for | Local |
|---|---|---|---|---|---|
| IBM Quantum Learning (Watrous): *Basics of Quantum Information*, *Fundamentals of Quantum Algorithms*, *General Formulation of Quantum Information*, *Foundations of QEC*; plus *Variational Algorithm Design* | IBM Quantum | https://quantum.cloud.ibm.com/learning/en/courses/basics-of-quantum-information | Free online (IBM terms; no open license stated) | The main quantum text: lessons deep-linked per skill, with Qiskit code | – |
| Preskill, Ph219/CS219 lecture notes | Caltech | https://preskill.caltech.edu/ph219/ | All rights reserved / free to read (hosted by the author) | Ch. 2–3 foundations, Ch. 4 entanglement, Ch. 5 circuits, Ch. 6 algorithms, Ch. 7 error correction, Ch. 10 quantum Shannon theory | `preskill_ch2.pdf` … `preskill_ch10.pdf` |
| Ronald de Wolf, *Quantum Computing: Lecture Notes* | QuSoft / Univ. of Amsterdam | https://arxiv.org/abs/1907.09415 | arXiv non-exclusive license | Ch. 1 qubits and teleportation, Ch. 2 circuits and Deutsch–Jozsa, Ch. 4 QFT and phase estimation, Ch. 5 Shor, Ch. 7 Grover, App. A linear algebra | `dewolf_qc_notes.pdf` |
| MIT 18.435J Quantum Computation (Shor, 2003) | MIT | https://ocw.mit.edu/courses/18-435j-quantum-computation-fall-2003/pages/lecture-notes/ | CC BY-NC-SA 4.0 | Scribe notes from Shor's own course (optional depth) | – |
| Scott Aaronson, *Introduction to Quantum Information Science* lecture notes | UT Austin | https://www.scottaaronson.com/qclec.pdf | Free to read | Conceptual alternative to de Wolf | – |
| Nielsen & Chuang, *Quantum Computation and Quantum Information* | Cambridge University Press | https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE | Commercial book, **not free** | Reference only | – |

---

## GitHub repos

All of these were checked through the GitHub API on 2026-11-17. Each exists and has the license shown.

| Repo | What it is | License | Notes |
|---|---|---|---|
| [karpathy/micrograd](https://github.com/karpathy/micrograd) | Tiny scalar autograd engine | MIT | Pairs with `dl.backprop` |
| [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) | Notebooks for the Zero to Hero video series | MIT | |
| [karpathy/makemore](https://github.com/karpathy/makemore) | Character-level LMs (bigram → transformer) | MIT | `dl.ngram_lm`, `dl.embeddings` |
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | Minimal GPT training/finetuning | MIT | `dl.training`, `dl.gpt` |
| [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) | Commit-by-commit GPT-2 (124M) reproduction | none stated | |
| [karpathy/minbpe](https://github.com/karpathy/minbpe) | Minimal byte-level BPE tokenizer | MIT | `dl.tokenization` |
| [karpathy/llm.c](https://github.com/karpathy/llm.c) | GPT-2 training in raw C/CUDA | MIT | Systems deep dive |
| [karpathy/nanochat](https://github.com/karpathy/nanochat) | Full-stack small ChatGPT clone (pretrain → SFT → RL) | MIT | End-to-end capstone |
| [stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics) | CS336 A1: tokenizer, transformer, AdamW from scratch | MIT | |
| [stanford-cs336/assignment2-systems](https://github.com/stanford-cs336/assignment2-systems) | CS336 A2: profiling, FlashAttention-2 in Triton, distributed | MIT | `dl.efficient_inference` |
| [stanford-cs336/assignment3-scaling](https://github.com/stanford-cs336/assignment3-scaling) | CS336 A3: scaling laws | MIT | `dl.scaling_laws` |
| [stanford-cs336/assignment4-data](https://github.com/stanford-cs336/assignment4-data) | CS336 A4: data filtering | MIT | |
| [stanford-cs336/assignment5-alignment](https://github.com/stanford-cs336/assignment5-alignment) | CS336 A5: SFT and RL-based alignment of an LM | none stated | `dl.alignment` |
| [mitmath/1806](https://github.com/mitmath/1806) | MIT 18.06 (current offering) course repo | none stated | Julia notebooks |
| [mitmath/18065](https://github.com/mitmath/18065) | MIT 18.065 course repo | none stated | |
| [mitmath/matrixcalc](https://github.com/mitmath/matrixcalc) | MIT Matrix Calculus for ML short course | none stated | `la.matrix_calculus` |
| [mitmath/18S096](https://github.com/mitmath/18S096) | MIT 18.S096 three-week course | MIT | |
| [harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer) | The Annotated Transformer notebook | MIT | `dl.attention`, `dl.transformer` |
| [Harvard-IACS/2019-CS109A](https://github.com/Harvard-IACS/2019-CS109A) | Harvard CS109A Data Science materials | MIT | `info.data_eda`, `ml.*` |
| [cs109/2015](https://github.com/cs109/2015) | Harvard CS109 (2015) public material | none stated | |
| [cs231n/cs231n.github.io](https://github.com/cs231n/cs231n.github.io) | Stanford CS231n notes source | MIT | |
| [openai/tiktoken](https://github.com/openai/tiktoken) | Fast BPE tokenizer for OpenAI models | MIT | Already a project dependency |
| [openai/gpt-2](https://github.com/openai/gpt-2) | Original GPT-2 code | Modified MIT (NOASSERTION) | Archived |
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | FlashAttention kernels | BSD-3-Clause | |
| [microsoft/LoRA](https://github.com/microsoft/LoRA) | loralib reference implementation | MIT | |
| [eric-mitchell/direct-preference-optimization](https://github.com/eric-mitchell/direct-preference-optimization) | DPO reference implementation | Apache-2.0 | |
| [Qiskit/qiskit](https://github.com/Qiskit/qiskit) | Qiskit SDK | Apache-2.0 | Already a project dependency |
| [Qiskit/textbook](https://github.com/Qiskit/textbook) | Source of the (legacy) Qiskit Textbook | Apache-2.0 | **Archived**. The content moved to IBM Quantum Learning |
| [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | Build a GPT-style LLM step by step (book code) | custom (NOASSERTION) | Optional alternative to nanoGPT |

No public course repos were found for Columbia COMS 4771 (Hsu) or Cornell CS4780. Their notes are linked above. `google-deepmind/chinchilla` does not exist.

---

## Local downloads (`library/pdfs/`)

Only material whose terms clearly allow a personal offline copy was downloaded: arXiv preprints,
the open-access JMLR paper, MIT OCW (CC BY-NC-SA), and author-hosted books and notes whose authors
offer the PDF as a free download. Each file was checked for a `%PDF` header.

| File | Source URL | Size | Terms |
|---|---|---|---|
| `adam.pdf` | https://arxiv.org/pdf/1412.6980 | 0.6 MB | arXiv non-exclusive distrib. |
| `transformer.pdf` | https://arxiv.org/pdf/1706.03762 | 2.2 MB | arXiv non-exclusive distrib. |
| `gpt3.pdf` | https://arxiv.org/pdf/2005.14165 | 6.8 MB | arXiv non-exclusive distrib. |
| `kaplan_scaling.pdf` | https://arxiv.org/pdf/2001.08361 | 2.5 MB | arXiv non-exclusive distrib. |
| `chinchilla.pdf` | https://arxiv.org/pdf/2203.15556 | 6.0 MB | arXiv non-exclusive distrib. |
| `instructgpt.pdf` | https://arxiv.org/pdf/2203.02155 | 1.8 MB | arXiv non-exclusive distrib. |
| `lora.pdf` | https://arxiv.org/pdf/2106.09685 | 1.6 MB | arXiv non-exclusive distrib. |
| `flashattention.pdf` | https://arxiv.org/pdf/2205.14135 | 2.6 MB | arXiv non-exclusive distrib. |
| `dpo.pdf` | https://arxiv.org/pdf/2305.18290 | 1.3 MB | CC BY 4.0 |
| `layernorm.pdf` | https://arxiv.org/pdf/1607.06450 | 0.6 MB | arXiv non-exclusive distrib. |
| `vae.pdf` | https://arxiv.org/pdf/1312.6114 | 3.9 MB | arXiv non-exclusive distrib. |
| `shor.pdf` | https://arxiv.org/pdf/quant-ph/9508027 | 0.3 MB | arXiv (pre-2004 assumed license) |
| `grover.pdf` | https://arxiv.org/pdf/quant-ph/9605043 | 0.07 MB | arXiv (pre-2004 assumed license) |
| `bengio2003.pdf` | https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf | 0.14 MB | JMLR open access |
| `dewolf_qc_notes.pdf` | https://arxiv.org/pdf/1907.09415 | 1.4 MB | arXiv non-exclusive distrib. |
| `mackay_itila.pdf` | https://www.inference.org.uk/itprnn/book.pdf | 11.7 MB | Author-offered download (on-screen viewing) |
| `boyd_convex_optimization.pdf` | https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf | 6.9 MB | Free on the web with CUP permission |
| `mit_6042_mcs.pdf` | https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/mit6_042js15_textbook2.pdf | 7.4 MB | CC BY-NC-SA 4.0 |
| `mit_6050j_info_entropy.pdf` | https://ocw.mit.edu/courses/6-050j-information-and-entropy-spring-2008/mit6_050j_s08-_coursetextbook.pdf | 5.8 MB | CC BY-NC-SA 4.0 |
| `preskill_ch2.pdf` | http://www.preskill.caltech.edu/ph219/chap2_15.pdf | 0.4 MB | Author-hosted free notes |
| `preskill_ch3.pdf` | http://www.preskill.caltech.edu/ph219/chap3_15.pdf | 0.4 MB | Author-hosted free notes |
| `preskill_ch4.pdf` | http://www.preskill.caltech.edu/ph229/notes/chap4_01.pdf | 0.4 MB | Author-hosted free notes |
| `preskill_ch5.pdf` | http://www.preskill.caltech.edu/ph219/chap5_15.pdf | 0.3 MB | Author-hosted free notes |
| `preskill_ch6.pdf` | http://www.preskill.caltech.edu/ph219/chap6_20_6A_2022.pdf | 0.5 MB | Author-hosted free notes |
| `preskill_ch7.pdf` | http://www.preskill.caltech.edu/ph219/chap7_26_6A.pdf | 0.7 MB | Author-hosted free notes |
| `preskill_ch10.pdf` | http://www.preskill.caltech.edu/ph219/chap10_6A_2025.pdf | 0.7 MB | Author-hosted free notes |

**Total: 26 files, 67.1 MB.**

These were not downloaded on purpose:

* **Blitzstein–Hwang** is shared by the authors only as a view-only online copy.
* **Shannon 1948** and the **GPT-2** report are not on arXiv, and their reuse terms are unclear.
* **CS229 notes**, the **SLP3 chapters** and the **Hsu, Cornell and CS224N notes** carry no explicit license. Read them online.

## Sources that could not be verified or were dropped

* **Princeton COS 324 / COS 484.** The COS 324 archive returns HTTP 403 to automated clients, so it was dropped.
  The COS 484 homepage (https://princeton-nlp.github.io/cos484/) loads fine, but no specific lecture was deep-linked.
* **Harvard Stat 110 site** (stat110.hsites.harvard.edu, projects.iq.harvard.edu/stat110) returns 403 to
  automated clients and was confirmed by web search only. The Stat 110 lecture videos and probabilitybook.net are used instead.
* **Yale, Brown and Penn.** The only Brown resource found and verified as useful is Seeing Theory. No Yale or Penn
  material was added.
* **Preskill's old `theory.caltech.edu/~preskill/ph229` URLs** now return 404. The notes moved to `preskill.caltech.edu`.
