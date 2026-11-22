# Experiment Plan, Work Split, and Decisions

## 1. Environment: VMAS multi-package transport

A modification of VMAS's `transport` scenario.

| Item | Spec |
|---|---|
| Team sizes | N ∈ {12, 24} for the core study; N = 48 optional |
| Packages | M = N/3, each needing about 3 agents pushing together |
| Respawning | A delivered package reappears at a random valid location (forces coalitions to re-form within an episode) |
| Actions | Discrete {stay, up, down, left, right} (`continuous_actions=False`) |
| Episode length / γ | 200 steps, γ = 0.99 |
| Reward (shared) | Σ over packages of progress toward the goal, + a bonus per delivery, − a small collision penalty |
| Ground-truth labels | `gt_coalitions[i] = k` if agent i is in contact with package k (distance to the package edge below a threshold), else 0. Tie-break: the nearer package |

**Physics tuning (critical):** set package mass and agent force so that **1–2 agents move a package slowly and 3 move it clearly faster.** If a package doesn't move at all with fewer than 3 agents, progress shaping gives no signal early and nothing learns. Options if learning stalls:
- a small approach reward (risk: hovering),
- a curriculum of light packages first,
- a longer ε decay,
- fallback config parameters `num_packages` and `sensing_radius` to simplify the task.

**The day-3 go/no-go check:** QMIX at N = 12 should show packages delivered per episode rising above 0. If no method learns, it's an **environment** problem, so fix it before spending compute on the grid.

## 2. Methods: the comparison ladder

| Method | Groups | Mixer | Question it answers |
|---|---|---|---|
| **QMIX** | none | flat QMIX | Main flat baseline (tune it well: Adam, TD(λ), long ε decay, large buffer) |
| **VDN** | none | sum | Simplest baseline and sanity check |
| **VAST-spatial** | hard, by map region (k-means if labeled "VAST-style") | QMIX over the K summed sub-team values | Prior-work grouping baseline |
| **Position clustering** | hard or soft by position | our coalition mixer | Isolates our mixer vs. VAST's operator |
| **Detached ablation** | soft, structure losses only | ours | Learned groups without the reward signal |
| **Full DynaCoalition** | soft, structure + TD | ours | **The main method** |
| **Oracle (contact)** | true contact groups | ours | Does correct grouping help at all? (a reference ceiling) |
| **Random S** | random | ours | Does any grouping structure matter? |
| **Nearest-package** | nearest package | ours | Task-aware heuristic (drop first if compute is tight) |

**The key comparison:** Full vs. Detached, the same model differing **only** in whether the TD gradient trains the grouping.

## 3. Priority order (2-week budget)

1. **Must run:** QMIX, Full, and Detached at N ∈ {12, 24}, 5 seeds each.
2. **Then:** Oracle → VAST-spatial → Position clustering → VDN → Random S.
3. **If time remains:** N = 48, Nearest-package, no-structure-loss ablation, extensions.

Measure training throughput early (day 2) and size the grid from it.

## 4. Protocol

- **Seeds:** 5 per method and setting. One seed = one full training run from scratch.
- **Pairing:** Full and Detached run with the **same seeds on the same machine and device**.
- **Training budget:** decided after the throughput benchmark. Initial guesses from the original plan: 500k steps at N = 12, 750k at N = 24.
- **Evaluation:** greedy policy (ε = 0) every 10k steps. The number of episodes per evaluation is open (Section 9).
- **Score per run:** the mean over the last few evaluations, not just the final one.
- **Fairness:** equal hyperparameter-tuning budget per method; the same agent network, replay settings, and target-update rule (soft Polyak) for all.

## 5. Metrics

| Metric | Definition |
|---|---|
| **Return** | Episodic team return |
| **Packages delivered** | Per episode |
| **ARI (main)** | `adjusted_rand_score(gt, argmax_k S)` over **agents in contact only**, per timestep (skip steps with fewer than 2 in contact), averaged over steps and episodes |
| **ARI (secondary)** | The same, over all agents, with background as its own cluster |
| **Switch rate** | Fraction of agents whose argmax slot changes per step. Caveat: slot relabeling can count false switches |
| **Overhead** | Wall-clock time per environment step relative to QMIX, on the same machine |
| **Diagnostics** | ‖∇_GNN L_TD‖, mean max membership, K_eff; ARI between Full's and Detached's groupings |

## 6. Statistics

- **IQM** with 95% **stratified bootstrap CIs** (rliable, about 2,000 resamples), stratified by team size.
- **Paired differences** Full − Detached per seed, with a bootstrap CI on the differences.
- With 5 seeds, also show per-seed curves, and don't overclaim.

## 7. Success criteria

**Primary:**
1. **Full beats Detached:** the paired-difference CI excludes 0 on at least one team size (return or packages delivered).
2. **Full is competitive with or better than QMIX and VAST-spatial** in IQM return.

**Secondary** (carried over from the proposal):
- **Grouping quality:** ARI (contact) ≥ 0.5 for the full method, vs. about 0 for random.
- **Stable but not frozen:** while the task is static, ≤ 1 coalition switch per agent per 50 steps; after a delivery, the involved agents regroup within 10 steps in ≥ 80% of events.
- **Scaling trend:** the return gap over QMIX grows from N = 12 to N = 24 (and to 48 if run).
- **Cost ceiling:** overhead ≤ the agreed threshold (open; Section 9).

**If Full doesn't beat Detached,** that's still a reportable finding, **provided the diagnostics rule out training failure** (next section).

## 8. Reading a null result

| What you see | Conclusion | Action |
|---|---|---|
| TD gradient healthy, memberships soft, groupings differ, performance equal | **Real finding:** reward reshapes groups but doesn't improve performance | Report it |
| TD gradient healthy, but Full and Detached groupings nearly identical (ARI between them ≈ 1) | Structure losses dominate, so the TD push is too weak | Lower β_cut, retry |
| TD gradient collapses early, and mean max membership → 1 early | **Saturation** (training failure) | Raise T, lower β_sharp, sharpen later, rerun |
| No method learns, QMIX included | **Environment** problem | Tune physics, shaping, or curriculum |
| QMIX learns, Full doesn't | Method or implementation bug | Run the unit tests and check the gradients |

## 9. Work split

**[work_division.md](work_division.md) is the authoritative version of this section.** It holds the agreed split, the repo layout, the task sets and the sync points. The summary below is kept here so this plan reads on its own.

| Abhishek (Claude Pro + GPU) | Akhil (M1 Air) |
|---|---|
| **Shared training loop:** replay of whole episodes, masking, Double-Q, Polyak updates, schedules | **The multi-package VMAS environment:** respawning, contact labels, reward shaping, physics tuning |
| **DynaCoalition modules:** graph network, CoalitionMix, structure losses, unit tests | **The controls as membership providers** (oracle, position clustering, random S, nearest-package) **plus VAST-spatial** |
| **Diagnostics logging:** ‖∇_GNN L_TD‖, mean max membership, K_eff | **The evaluator:** return, packages, ARI (both versions), switch rate; then rliable, CIs, and all figures |
| **All paired runs** (QMIX, Full, Detached), Oracle, VAST-spatial, and everything at N = 24 | **The unpaired runs at N = 12** on the Mac: VDN, random S, nearest-package, position clustering |
| Report sections: method and experimental setup | Report sections: results, analysis, related work, final assembly |

**Balance:** each person owns one hard, uncertain piece (the training loop vs. the environment), one piece of the key science (the method vs. the controls that bracket it), and one half of the report.

**What must stay single-machine:** the paired Full/Detached/QMIX comparison, and the whole wall-clock overhead row in Section 5. Return and packages delivered are machine-independent, so the Mac runs drop into the same figures.

**The early dependency:** agree on the environment contract (Architecture.md, Section 8) first, so the training loop can be built against fake data while the environment is finished. This is done: `src/dc/contracts.py` and `src/dc/envs/fake.py` are in the repo.

**No `--algo` dispatch.** Configs name a dotted path (`dc.membership.oracle:OracleProvider`), resolved by `dc.contracts.resolve`, so adding a control is one new file plus one new yaml and touches nothing the other owner owns.

**Git:** `main` plus one feature branch per person. Don't commit checkpoints or replay buffers; commit only lightweight `metrics.csv` / `summary.json` files under `experiments/`.

## 10. Decisions log

**Decided:**
- VMAS multi-package transport is primary; SMAC is optional.
- Fixed distance affinity; sg(h) into the graph network; the detach covers everything computed from S.
- One shared monotone g over slots, plus the background head and the −g(0) offset.
- Size cap plus sharpness instead of orthogonality; the cap is not set from task knowledge.
- K is **not** set equal to the number of packages (that would leak task knowledge).
- Soft Polyak target updates, the same for all methods.
- Temperature and sharpness annealed late; TD gradient size and membership sharpness logged.
- Full and Detached are paired on the same machine and device.
- The baseline and control set, and the priority order, are as above.
- **K = N/2** (6 at N = 12, 12 at N = 24). It never equals M = N/3 at either size, so no task knowledge leaks. Enforced in `dc.contracts.k_slots_for`.
- **The training loop is written from scratch** (thin) rather than forked from pymarl2 or built on BenchMARL: the detach semantics, the shared g and the background head cut across the mixer.
- **The controls belong to Akhil**, and his M1 owns the unpaired N = 12 runs. No SSH between machines; the fallback for GPU-only work is a config committed to `runs/queue.yaml`.

**Open:**
- **The overhead threshold:** 1.5× (proposal) vs. 2.0× (Akhil's plan).
- **Evaluation episodes:** 16 vs. 32 per evaluation.
- **VAST-spatial assignment:** grid regions vs. k-means.
- **Training budget per run:** after the day-2 throughput benchmark (CPU vs. GPU, `num_envs` ∈ {32, 128, 512}).
- **Physics and reward constants:** after the day-3 check.

**Issues found in the original work-split plan, and how they were resolved:**
- **4-week schedule, 2 weeks available:** the timeline was compressed and the experiment list trimmed.
- **The critical path sat on one person:** the training loop moved to Abhishek.
- **K = M = N/3 leaked task knowledge:** K is now an independent hyperparameter.
- **Full and Detached on different laptops:** they're now on the same machine.
- **Inconsistent extension names:** use the proposal's numbering if extensions are run: E1 = agent failure, E2 = Owen-value credit, E3 = larger teams.
- **Missing tasks added:** reward shaping, ARI handling of background agents, the contact tie-break, TD gradient logging, target copies of the graph network and mixer, equal tuning budgets.
