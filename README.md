# DynaCoalition: Handoff Docs

Context for the DynaCoalition course project (Advanced Topics in RL, UW–Madison). Akhil Arya and Abhishek Sharma.

## Submitted documents

- **Project proposal:** the original plan (SMAC + VMAS, learned affinity). Graded 4.5/5.
- **Literature survey:** the current, approved design. It includes the refined architecture, Figure 1, and the pseudocode. **When the proposal and the survey disagree, the survey is authoritative.**

## Docs in this folder

| File | What it covers | Read it when |
|---|---|---|
| [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) | The research question, what's new vs. prior work, the project history and professor feedback, and why the survey design replaced the proposal design | First, to understand why we're doing this |
| [CONCEPTS_PRIMER.md](CONCEPTS_PRIMER.md) | The nine background topics (QMIX, monotonicity, MinCutPool, saturation, Double-Q, statistics, VAST, VMAS, exploration) plus common clarifications | Before implementing, or whenever a concept is unclear |
| [Architecture.md](Architecture.md) | The exact equations, tensor shapes, interfaces, environment contract, defaults, pseudocode, unit tests, and a bug checklist | While implementing |
| [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) | Environment design, baselines and controls, protocol, metrics, statistics, success criteria, diagnostics, the work split, and the decisions log | Before running experiments, and when planning work |
| [RELATED_WORK_NOTES.md](RELATED_WORK_NOTES.md) | Verified facts about the closest papers, and citation corrections | When writing the final report |
| [work_division.md](work_division.md) | The proposed work split, the repo layout that keeps us asynchronous, and the day-1 handshake | Before starting implementation |

The approved literature survey PDF is in the repo as `CS839_RL_Literature_Survey.pdf`; its Figure 1 is the architecture diagram (also at `figures/figure1_architecture.png`, embedded in `Architecture.md`).
