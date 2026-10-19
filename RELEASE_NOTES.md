# v1.0.5-submission — Submission to IEEE T-ITS (Oct 2026)

**When Is Periodic Detection Enough for UAV Traffic Monitoring? Theory and a Pre-Registered Test of Cue-Triggered Gating**
Dat Lam Quoc (FPT School of Business and Technology (FSB) – FPT University, Ho Chi Minh City, Vietnam; ORCID 0009-0004-5432-9343). Submitted to IEEE Transactions on Intelligent Transportation Systems, October 2026.

A closed theory of periodic versus cue-triggered detection for newly appearing vehicles (gain–cost identity, win-set interval,
iso-KPI condition, density and ego-motion impossibility) and a pre-registered test on UAVDT: 72 confirmatory predictions frozen before
the test split was opened. Result: 62 correct, 5 wrong, 5 inconclusive (frozen criteria not violated), but the predicted gate corner did
not appear and no gate beat the periodic schedule at matched cost; one refutation criterion could not be evaluated.

| item | value |
|---|---|
| `PREREG_28.md` | sha256 `81c19b43473e54d9be13be4e7ecaebbaab9e35d479f6027b247afc6b0a3640ab` |
| `results/p2/prereg_cells.json` | sha256 `fb30790c2057755dfbc510e9001ef662a4ad3277b7934da2099a8fe41c8a087e` |
| `paper_submitted/main.pdf` | sha256 `b6bcbfde2028e309f54a4b03fa391506b673f3910ef296b1a3410f1cf6092b06` |
| `paper_submitted/supplementary.pdf` | sha256 `e2003d817f544817a3686da76c2a12752c251b10e14a95379b4d9500ebbf7c28` |
| `paper_submitted/main_portal.pdf` (IEEE Author Portal bundle) | sha256 `78641b44de581cbc8892b6f4aea22a31f12e27341d701be721bacfc97accf12e` |
| `paper_submitted/supplementary_portal.pdf` | sha256 `ed3e8f231c082ad1b3123691e75ae3ddba7ad34003ade7478f0ea49bd36e12c4` |
| `source_latex_portal.zip` (IEEE Author Portal, not included) | sha256 `0f3600cbb89b43e85ba2b9cc5ad2b744afe2cc2c9d664501530a80885a3d1e75` |
| `source_latex.zip` (ScholarOne, not included) | sha256 `10a12253e3737521f538430f5e00c22c3d9751e0273f5f83526553e6519dc5bf` |
| pre-registration freeze | tag `prereg-28-v1`, commit `5cfe9014d61faa9e14bc42b7bb1bf3094b752f7c` of the development repository (2026-10-01) |

Code and results are identical to `v1.0-submission`. `v1.0.1-submission` added paper metadata and the submitted PDFs; `v1.0.2-submission`
re-issues the PDFs with the 4 arXiv-only references printed as DOIs `10.48550/arXiv.<id>` (no text, number or result changed);
`v1.0.3-submission` re-issues `main.pdf` with the author biography without a degree title (supplementary unchanged);
`v1.0.4-submission` adds the IEEE Author Portal bundle: `main_portal.pdf` and `supplementary_portal.pdf`, compiled from single flattened
LaTeX files (one-line title, abstract numbers written out, Fig. 1 as a vector PDF); text and numbers unchanged, earlier PDFs kept;
`v1.0.5-submission` re-issues `supplementary_portal.pdf`: in Table S1 the 16 CI$_{hi}$ cells that printed the bootstrap sentinel
-1000000000.000 (encoding of $-\infty$ when $a_P(\varepsilon)=\infty$) now print $-\infty$; finite values, verdicts and all other numbers unchanged.
