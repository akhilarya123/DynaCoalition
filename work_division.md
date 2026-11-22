# Work Division, Repo Structure, and Task Plan

How the implementation splits between Abhishek and Akhil, a repo layout that lets us work asynchronously with almost no shared files, and a day-by-day task list for the 2-week budget.

This expands [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) Section 9 and, where the two differ, **this file is now the agreed version** — Section 9 should be updated to match.

**Constraints this is built around:** about 2 weeks; Abhishek has a GPU Windows laptop (WSL2) and a Claude Pro seat; Akhil has an M1 Air that has already handled MuJoCo RL training; no code-review bandwidth on either side; no shared SSH access between machines.

## 1. What carries over from the current plan

Two things in `EXPERIMENT_PLAN.md` Section 9 are right and stay:

- **Each person owns exactly one hard, uncertain piece.** Abhishek: the training loop. Akhil: the environment. These are the two items that can sink the schedule, and they are genuinely parallel once the environment contract (`Architecture.md`, Section 7) is agreed.
- **Full and Detached run paired on the same machine and device.** This is a correctness requirement for the key experiment, not a convenience, so the GPU box runs both regardless of how the rest splits.

## 2. What "equal" means here

Equal cannot mean equal compute. The GPU and the Claude Pro seat sit on one side, so that side will mechanically produce more code and all the headline runs. Measured in lines of code, Akhil ends up as the person who ran the cheap baselines, which is bad for him and bad for the report.

So we define equal as: **each person owns one gating deliverable, one piece of the key science, and one half of the report.** By that measure the split below is even even though the hardware is not.

## 3. The split

| | **Abhishek** (GPU + Claude Pro) | **Akhil** (M1 Air) |
|---|---|---|
| **Gating piece** | Shared training loop: episode replay, masking, Double-Q, Polyak updates, schedules | The VMAS multi-package environment: respawn, contact labels, reward shaping, **physics tuning** |
| **Key science** | `CoalitionMix`: graph net, pooling, shared `g` + `g_bg`, structure losses, the unit tests in `Architecture.md` Section 10 | **The controls**: oracle (contact), position clustering, random S, nearest-package, each as a membership provider; plus VAST-spatial |
| **Measurement** | Diagnostics logging: ‖∇_GNN L_TD‖, mean max membership, K_eff | The evaluator: return, packages delivered, ARI (both versions), switch rate; then rliable, CIs, and all figures |
| **Runs** | All paired runs (QMIX, Full, Detached), plus Oracle and VAST-spatial, and everything at N = 24 | The unpaired baselines and controls at N = 12: VDN, random S, nearest-package, position clustering |
| **Report** | Method and experimental setup | Results, analysis, related work, assembly |

**The controls sit with Akhil** because they *are* the experiment — Full vs. Detached only means something because oracle, position-clustering and random-S bracket it. If one person owns the method and the other owns the evidence for its claim, both are on the critical science rather than one doing plumbing. They are also a clean seam: every control returns `S` of shape `(B, T, N, K+1)` and can be built and tested against fake data without touching the mixer.

**The one shared dependency to schedule early:** Abhishek needs a QMIX mixer module for the QMIX baseline anyway, so it ships by Day 3 and Akhil composes VAST-spatial (hard spatial assignment plus QMIX over the K summed sub-team values) on top of it instead of writing a mixer from scratch.

## 4. Compute, and what runs where

**Akhil's M1 is a real run machine.** It has handled MuJoCo RL training, and VMAS at N = 12 with small networks is CPU-friendly, so it owns the unpaired baselines and controls. Return and packages-delivered are machine-independent quantities, so runs from the two machines are directly comparable in the same figure.

**Two things must stay single-machine:**
- **Full, Detached and QMIX** — the key paired comparison, all on the GPU box, same seeds, same device.
- **The wall-clock overhead table** (`EXPERIMENT_PLAN.md` Section 5) — relative cost only means something measured on one machine, so the whole overhead row is produced on the GPU box.

**No SSH between machines.** If a run genuinely needs the GPU, the fallback is a spec in the repo: Akhil commits a config to `runs/queue.yaml`, Abhishek pulls and launches, and results land in `experiments/`. This should be the exception, not the workflow.

**Physics tuning is the biggest single schedule risk and it sits on one person.** The Day-3 go/no-go in `EXPERIMENT_PLAN.md` Section 1 exists because this can eat a week, so make that check explicitly joint: if packages do not move sensibly by Day 3, both of us stop and fix the environment, because nothing downstream is worth running until it does.

**Extra capacity:** UW–Madison students have CHTC (HTCondor) access. If the grid gets tight at N = 24 × 5 seeds × 6 methods, that is real additional throughput for the **unpaired** runs.

## 5. Repo structure

The structure exists to make sure **neither person has to open a file the other owns.** The only shared files are small, frozen on Day 1, and changed by explicit agreement.

### The four seams

| Seam | Kills the dependency | Who it unblocks |
|---|---|---|
| `contracts.py` plus a **fake env** | Training loop needs the real environment | Abhishek builds and tests the loop with no VMAS at all |
| **Membership provider** signature | Controls need the mixer | Akhil writes the controls against fake `S` shapes |
| **On-disk eval records** | Evaluator needs the training loop | Metrics are tested on dumped rollouts, not live objects |
| **`metrics.csv` schema** | Analysis needs finished runs | rliable plots are built against synthetic CSVs before any run exists |

### Tree

```
DynaCoalition/
├── README.md  Architecture.md  PROJECT_CONTEXT.md        # docs stay at root
│   CONCEPTS_PRIMER.md  EXPERIMENT_PLAN.md  RELATED_WORK_NOTES.md
│   work_division.md
├── CS839_RL_Literature_Survey.pdf
├── figures/figure1_architecture.png
├── pyproject.toml                      # SHARED, day 1. cpu/cuda extras
├── configs/
│   ├── base.yaml                       # SHARED, day 1
│   ├── env/transport_n12.yaml ...      # Akhil
│   └── algo/qmix.yaml  vdn.yaml  full.yaml  detached.yaml
│            oracle.yaml  poscluster.yaml  random_s.yaml
│            vast_spatial.yaml  nearest.yaml
├── src/dc/
│   ├── contracts.py                    # SHARED, FROZEN — the whole handshake
│   ├── device.py  seeding.py           # SHARED, day 1, ~20 lines each
│   ├── envs/
│   │   ├── fake.py                     # SHARED, day 1 — emits contract-shaped batches
│   │   ├── transport.py                # Akhil
│   │   └── labels.py                   # Akhil (contact labels, tie-break)
│   ├── agents/rnn_agent.py             # Abhishek
│   ├── mixers/
│   │   ├── qmix.py  vdn.py             # Abhishek — ship qmix.py FIRST
│   │   ├── coalition_mix.py            # Abhishek
│   │   └── vast.py                     # Akhil (imports qmix.py, does not edit it)
│   ├── membership/
│   │   ├── graph_net.py                # Abhishek
│   │   ├── oracle.py  position.py      # Akhil
│   │   └── random_s.py  nearest.py     # Akhil
│   ├── losses/structure.py             # Abhishek
│   ├── train/
│   │   ├── loop.py  replay.py          # Abhishek
│   │   └── targets.py  schedules.py  logging.py
│   ├── eval/
│   │   ├── evaluator.py  ari.py        # Akhil
│   │   └── switch.py  metrics.py       # Akhil
│   └── analysis/aggregate.py  plots.py # Akhil
├── tests/
│   ├── test_contracts.py               # SHARED — both run it before every push
│   ├── test_coalition_mix.py           # Abhishek — the Architecture.md §10 tests
│   ├── test_env_contract.py            # Akhil — real env satisfies contracts.py
│   ├── test_providers.py  test_metrics.py   # Akhil
├── scripts/train.py  evaluate.py  make_figures.py  smoke.sh
├── experiments/<run_id>/{config.yaml, metrics.csv, summary.json}   # committed, lightweight
└── runs/queue.yaml                     # GPU job requests (fallback path only)
```

### The three shared files

**`contracts.py`** — the only file both of us depend on. It holds the batch spec from `Architecture.md` Section 7 as a dataclass or TypedDict with a `validate(batch)` that asserts every key, shape and dtype; the `MembershipProvider` protocol; and the `EvalRecord` schema. Frozen after Day 1. Changing it is a conversation, not a commit.

**`envs/fake.py`** — written before either of us writes anything real. It generates random batches that pass `validate()`, with plausible positions and `gt_coalitions`. The whole training loop, mixer, losses and unit tests run against it, and the real environment is a one-line config swap later. Akhil's controls and metrics test against it too, so neither of us is blocked on the environment being finished.

**`base.yaml`** — seeds, device, buffer, evaluation cadence, logging paths, and **K = N/2**.

### The one structural trick that matters

**No central `--algo` dispatch file.** An `if algo == "oracle": ...` block is a merge-conflict magnet and it forces Akhil to edit the training loop every time he adds a control. Instead configs name a dotted path:

```yaml
# configs/algo/oracle.yaml
membership: { target: dc.membership.oracle:OracleProvider, args: {...} }
mixer:      { target: dc.mixers.coalition_mix:CoalitionMix, args: {detach: false} }
```

The loop does one `import_from_string(cfg.membership.target)`. Adding a control is then **one new file plus one new yaml, with zero edits to anything the other person owns.** The same applies to mixers, so `vast.py` plugs in without touching `loop.py`.

### Ownership and git hygiene

- **One owner per file.** If you need something in a file the other owns, ask — do not edit. The only exception is `contracts.py`.
- `feat/train-*` and `feat/env-*` branches, PRs into `main`. No deep review, but no direct pushes to `main` either.
- A roughly 15-line GitHub Action running `pytest` on CPU against the fake env catches contract drift before it costs either of us a day. Free, and the cheapest insurance in this plan.
- `.gitignore`: `*.pt`, checkpoints, replay buffers, `wandb/`, raw videos. Commit only `metrics.csv` and `summary.json` under `experiments/`, as `EXPERIMENT_PLAN.md` Section 9 says.

## 6. The Day-1 handshake

Before either of us writes a line of real code, agree on exactly three things. About two hours together, after which we can both go dark for a week and the pieces will still fit.

**1. `contracts.py`.** The batch keys and shapes from `Architecture.md` Section 7, plus:

```python
class MembershipProvider(Protocol):
    """Returns S of shape (B, T, N, K+1); rows sum to 1; column 0 is background."""
    def forward(self, *, p, x, h, s, gt, K, mask) -> Tensor: ...
```

Stateless controls ignore the arguments they do not need; the learned provider is the same signature with parameters. `gt` is passed to everyone but may only be read by the oracle.

**2. `envs/fake.py`.** Random but plausible batches that pass `validate()`.

**3. The `metrics.csv` columns** (one row per evaluation point):

```
run_id, algo, n_agents, seed, env_step, wall_clock_s,
return_mean, return_std, packages_delivered,
ari_contact, ari_all, switch_rate,
grad_gnn_td, mean_max_membership, k_eff
```

Plus the `EvalRecord` dump that the evaluator consumes: per evaluation episode, the per-step `argmax_k S` and `gt_coalitions` arrays with the in-contact mask, so ARI and switch rate are computed offline from files rather than from live objects.

## 7. Task plan

Days are working days from the start of implementation, not calendar dates. **Day 1 is joint; everything after runs in parallel.**

### Abhishek

| Day | Task | Done when |
|---|---|---|
| 1 | Joint handshake. Repo skeleton, `pyproject.toml`, `device.py`, `seeding.py`, `contracts.py`, `envs/fake.py`, pytest CI action | `pytest` green on CI against the fake env |
| 2 | `agents/rnn_agent.py` (GRU agent), `train/replay.py` (whole-episode replay, masking, padding) | Episodes round-trip through the buffer with correct shapes and masks |
| 2 | **Throughput benchmark** on fake + VMAS defaults, `num_envs` ∈ {32, 128, 512} | A steps/second number that sizes the training budget (`EXPERIMENT_PLAN.md` §4) |
| 3 | `mixers/qmix.py` and `mixers/vdn.py` — **ship these first, Akhil's VAST-spatial depends on `qmix.py`** | QMIX trains on the fake env without error; pushed to `main` |
| 3 | `train/targets.py` (Double-Q per `Architecture.md` §7), `schedules.py` (ε, T, β_sharp), `logging.py` (writes `metrics.csv`) | The checklist items in `Architecture.md` §11 for targets and masking all pass |
| 3 | **Joint Day-3 go/no-go** on the environment | QMIX at N = 12 shows packages delivered rising above 0, or we stop and fix the env |
| 4 | `membership/graph_net.py`, `mixers/coalition_mix.py`: pooling, shared `g` hypernetwork, `g_bg`, the −g(0) offset | Monotonicity, permutation-invariance and empty-slot tests pass (`Architecture.md` §10) |
| 5 | `losses/structure.py`: cut, cap, sharpness, temporal + diagnostics logging (‖∇_GNN L_TD‖, mean max membership, K_eff) | Detach test passes exactly (gradient 0 with `detach=True`, > 0 without); cut loss recovers planted partitions |
| 6 | Swap the fake env for Akhil's real env; end-to-end short run at N = 12; walk the whole bug checklist | Full and Detached both train end to end for 20k steps without NaNs |
| 7 | Tune QMIX properly (Adam, TD(λ), long ε decay, large buffer); fix the training budget from the Day-2 benchmark | A QMIX N = 12 curve that clearly learns; budget written into `base.yaml` |
| 8–11 | **Main grid**, paired, same device: QMIX, Full, Detached × 5 seeds × N ∈ {12, 24}. Then Oracle, then VAST-spatial as GPU time allows | `experiments/` filling with `metrics.csv` per run; diagnostics checked daily against `EXPERIMENT_PLAN.md` §8 |
| 12 | Overhead table (all methods, same machine); rerun any failed or diverged seeds | Overhead row complete; no seed missing from the must-run set |
| 13–14 | Report: method and experimental setup sections; support final assembly | Sections drafted in the report repo/Overleaf |

### Akhil

| Day | Task | Done when |
|---|---|---|
| 1 | Joint handshake. `envs/transport.py` skeleton: VMAS transport fork, M = N/3 packages, discrete actions | Env constructs and steps at N = 12 |
| 2 | Respawn on delivery; `envs/labels.py` contact labels with the nearer-package tie-break; `tests/test_env_contract.py` | Real env batches pass `contracts.validate()` |
| 2–3 | **Physics and reward tuning**: mass and force so 1–2 agents move a package slowly and 3 move it clearly faster; progress shaping, delivery bonus, collision penalty | A scripted 3-agent policy delivers a package; a 1-agent one barely moves it |
| 3 | **Joint Day-3 go/no-go** with Abhishek | Decision recorded: proceed, or stop and tune |
| 4 | `eval/evaluator.py`, `ari.py`, `switch.py`, `metrics.py` against dumped `EvalRecord` files; `tests/test_metrics.py` on synthetic records | ARI (contact and all-agents) and switch rate correct on hand-built cases |
| 5 | Controls as membership providers: `oracle.py`, `random_s.py`, `position.py` (k-means or soft RBF), `nearest.py`; `tests/test_providers.py` | Every provider returns `(B, T, N, K+1)`, rows sum to 1, background column used correctly |
| 6 | `mixers/vast.py`: hard spatial assignment plus QMIX over the K summed sub-team values, composed from Abhishek's `qmix.py` | VAST-spatial trains on the fake env |
| 7 | `analysis/aggregate.py` and `plots.py`: read `experiments/*/metrics.csv`, rliable IQM with stratified bootstrap CIs, paired Full − Detached differences | Figures render end to end from **synthetic** CSVs, before any real run finishes |
| 8–11 | **Mac runs** at N = 12, 5 seeds: VDN, random S, nearest-package, position clustering. Regenerate figures as results land | Those four methods complete with committed `metrics.csv` |
| 12 | Final figures from the real grid; per-seed curves; the null-result diagnostics table from `EXPERIMENT_PLAN.md` §8 | All report figures final |
| 13–14 | Report: results, analysis, related work, and assembly | Full draft assembled |

### Dependency edges (the only places we can block each other)

| Edge | By | Fallback if late |
|---|---|---|
| `contracts.py` + `fake.py` (joint) | Day 1 | Nothing proceeds — this is the hard gate |
| `mixers/qmix.py` (Abhishek → Akhil) | Day 3 | Akhil reorders: controls first, VAST-spatial after |
| Real env (Akhil → Abhishek) | Day 6 | Abhishek keeps running on `fake.py`; main grid slips |
| Evaluator (Akhil → Abhishek) | Day 6 | Abhishek logs raw `EvalRecord` dumps; metrics computed later |
| First real `metrics.csv` (Abhishek → Akhil) | Day 8 | Akhil's plots already work on synthetic CSVs |

### Milestones and the cut list

- **Day 3:** environment go/no-go. If no method learns, it is an environment problem — stop and fix it.
- **Day 6:** end-to-end integration. Full and Detached training on the real env.
- **Day 8:** the main grid starts. **If it has not started by Day 8, cut scope**, in the priority order in `EXPERIMENT_PLAN.md` §3: drop N = 48, then nearest-package, then random S, then VDN, then position clustering. Never cut QMIX, Full, Detached, or the 5 seeds.
- **Day 12:** results frozen. No new runs after this, only writing.

## 8. Decisions settled here

- **K = N/2** (6 at N = 12, 12 at N = 24). It never equals M = N/3 at either size, so no task knowledge leaks, and the −g(0) offset already makes spare slots free. Closes the open item in `EXPERIMENT_PLAN.md` §10.
- **The training loop is written from scratch** (thin, roughly 600 lines) rather than forked from pymarl2 or built on BenchMARL. The detach semantics, the shared `g` and the background head cut across the mixer, which is exactly what a framework would fight.
- **The controls belong to Akhil**, as in Section 3.
- **No SSH.** Akhil's Mac owns the unpaired N = 12 baselines and controls; the fallback for GPU-only work is a config committed to `runs/queue.yaml`.

**Still open, and deliberately so:** the training budget per run (set after the Day-2 throughput benchmark), the overhead threshold (1.5× vs. 2.0×), evaluation episodes (16 vs. 32), VAST-spatial assignment (grid regions vs. k-means), and the physics and reward constants (set during Days 2–3). All are tracked in `EXPERIMENT_PLAN.md` §10.

**One thing to pin down:** Day 1 above is relative. Map it to a calendar date against the actual report deadline, and check that Day 14 lands before submission with slack.
