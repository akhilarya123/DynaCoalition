# Work Division and Repo Structure

A working proposal for splitting the implementation between Abhishek and Akhil, and a repo layout designed so the two of us can work asynchronously with almost no shared files.

This expands [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) Section 9. Where it differs from that section, the difference is flagged below and is **still pending Akhil's agreement** — Section 9 stays authoritative until we both sign off.

**Constraints this is built around:** about 2 weeks; Abhishek has a GPU Windows laptop (WSL2) and a Claude Pro seat; Akhil has an M1 Air; no code-review bandwidth on either side.

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
| **Runs** | Every paired GPU run (QMIX, Full, Detached; then Oracle, VAST-spatial) | Environment validation and smoke tests; no runs that feed a figure (see Section 4) |
| **Report** | Method and experimental setup | Results, analysis, related work, assembly |

**The change from Section 9: the controls move to Akhil.** The controls *are* the experiment — Full vs. Detached only means something because oracle, position-clustering and random-S bracket it. If one person owns the method and the other owns the evidence for its claim, both are on the critical science rather than one person doing plumbing.

They are also a clean seam: every control is a function returning `S` of shape `(B, T, N, K+1)`, so they can be built and tested against fake data without touching the mixer. Merge conflicts stay near zero.

**The one shared dependency to schedule early:** Abhishek needs a QMIX mixer module for the QMIX baseline anyway, so it ships first and Akhil composes VAST-spatial (hard spatial assignment plus QMIX over the K summed sub-team values) on top of it instead of writing a mixer from scratch.

## 4. Three risks to settle before task-level planning

**1. The M1 Air is a dev machine, not a run machine.** Fanless, throttles under sustained load, and torch MPS coverage is patchy for the ops we need. It should not produce any run that feeds a figure — including the "cheap CPU runs" (VDN, random S, extra seeds) that Section 9 assigns to it. Those move to the GPU box.

**2. Akhil needs a way to launch runs on the GPU box,** or he is blocked on Abhishek for every result and the split quietly becomes unequal again. Cheapest version: he commits a config to `runs/queue.yaml`, Abhishek pulls and launches, results land in `experiments/`. Better version: Tailscale plus SSH so he launches himself. Worth 30 minutes on day 1 either way.

**3. Physics tuning is the biggest single schedule risk and it sits on one person.** The day-3 go/no-go in `EXPERIMENT_PLAN.md` Section 1 exists because this can eat a week. Make that check explicitly joint: if packages do not move sensibly by day 3, both of us stop and fix the environment, because nothing downstream is worth running until it does.

**Extra capacity:** UW–Madison students have CHTC (HTCondor) access. If the grid gets tight at N = 24 × 5 seeds × 6 methods, that is real additional throughput for the **unpaired** runs. The paired Full/Detached runs still have to stay on one box.

## 5. Repo structure

The structure exists to make sure **neither person has to open a file the other owns.** The only shared files are small, frozen on day 1, and changed by explicit agreement.

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
└── runs/queue.yaml                     # Akhil requests GPU jobs here
```

### The three shared files

**`contracts.py`** — the only file both of us depend on. It holds the batch spec from `Architecture.md` Section 7 as a dataclass or TypedDict with a `validate(batch)` that asserts every key, shape and dtype; the `MembershipProvider` protocol; and the `EvalRecord` schema. Frozen after day 1. Changing it is a conversation, not a commit.

**`envs/fake.py`** — written before either of us writes anything real. It generates random batches that pass `validate()`, with plausible positions and `gt_coalitions`. The whole training loop, mixer, losses and unit tests run against it, and the real environment is a one-line config swap later. Akhil's controls and metrics test against it too, so neither of us is blocked on the environment being finished.

**`base.yaml`** — seeds, device, buffer, evaluation cadence, logging paths. Per-algo yamls override only what differs.

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

## 6. The day-1 handshake

Before either of us writes a line of real code, agree on exactly three things: **`contracts.py`**, **`envs/fake.py`**, and the **`metrics.csv` column schema**. That is about two hours together. After that we can both go dark for a week and the pieces will still fit.

**K belongs in that handshake.** `EXPERIMENT_PLAN.md` Section 10 still lists it as open (a fixed value such as 8 vs. a fraction of N), it lives in `base.yaml`, and it shapes every `S` tensor — so settle it then rather than after.

## 7. Open questions

1. **Do the controls move to Akhil** as proposed in Section 3, or does he keep the environment-plus-evaluator scope from `EXPERIMENT_PLAN.md` Section 9 while Abhishek takes all mixer-side code?
2. **How does Akhil reach the GPU box** — Tailscale/SSH, or the `runs/queue.yaml` request file?
3. **What is K?** (Section 6.)
