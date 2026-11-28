# Akhil's side: design decisions, assumptions, and open questions

For Abhishek. This explains what I built, why I made each choice, and which parts need
your input. Everything here is about the files I own (environment, physics, evaluator, controls, VAST).
I have **not** edited any shared file (`contracts.py`, `fake.py`, `base.yaml`,
`pyproject.toml`, `device.py`, `seeding.py`).

**Status:** Sets 1 and 3 are done; Set 2 (physics) is done as far as a probe can take it;
Set 4 is done **except VAST-spatial**, which is written but blocked on `mixers/qmix.py`
(section 6). Set 5 (analysis) has not started. Last full run on my M1 before Set 4: 77 passed,
0 skipped. The Set 4 tests add 52 (expect 129 passed, 0 skipped); the numpy-side Set 4
math was verified in a sandbox, the torch-gated parts have not run yet (section 7).

*This file is updated after every set. See the update log at the end.*

---

## 1. Read this first: what I need from you

These are the interfaces and decisions that touch your code. Everything else is internal to my side.

| # | Question or action | My proposal | Why it matters |
|---|---|---|---|
| 1 | **Env step API.** `contracts.py` defines the sampled batch but not how the env is stepped. | `TransportEnv(spec, num_envs, device, seed, **physics)`, `reset()`, `step(actions[E,N])` (details in section 3.1) | Your collector and replay buffer consume this. Please confirm or change it before you build against it. |
| 2 | **How does your loader pass `env.physics`?** | `TransportEnv(spec, **cfg.env.physics)` | The physics constants live in `configs/env/transport_n*.yaml` under `env.physics`, not as flat keys. |
| 3 | **Time limit: terminal or bootstrap?** | `done` is the 200-step limit only (packages respawn, so nothing else ends an episode). `info["truncated"]` carries the same flag. | `Architecture.md` §11 says `(1 - done)` zeroes the future value. That treats a timeout as a true terminal state. Your call, but please decide it deliberately. |
| 4 | **Dumping eval records.** | Call `dc.eval.evaluator.write_record(record, run_dir)` from your eval loop (section 5.1). | It is the only thing the trainer imports from my side. |
| 5 | **`assignments` for methods without memberships (QMIX, VDN).** | Fill the array with `-1`. | Then their ARI and switch cells stay empty instead of showing a misleading 0. |
| 6 | **Who writes `metrics.csv`?** | Your `logging.py` writes the trainer columns. My tool fills `ari_contact`, `ari_all`, `switch_rate` afterwards and never overwrites your values. | Two writers on one file would conflict. |
| 7 | **Delivery events in records (optional).** | Put `meta["deliveries"] = [[step, package_idx], ...]` in each `EvalRecord`. | Needed only for the "regroup within 10 steps after a delivery" criterion. The env already reports `info["delivered_now"]` each step. |
| 8 | **`gt_coalitions` has T entries, but the oracle needs T+1.** The target-side oracle call (states 1..T) has no label for the last step. | Make `gt_coalitions` observation-like, `(B, T+1, N)`. Touches `contracts.py` (`_FIELDS`), `fake.py`, `test_contracts.py` and the `Architecture.md` §8 table. My `rollout_to_batch` already produces T+1 labels. If you would rather not change the contract, the oracle can reuse `gt[T-1]` for the last step (wrong at that one step only). | Without this the Oracle control is slightly inexact on the target network. |
| 9 | **`qmix.py` interface for VAST-spatial.** I do not have your file. | I assumed `QMixer(n_agents=K, state_dim=..., **kwargs)` with `forward(values (B,T,K), state (B,T,state_dim)) -> (B,T,1)`. | VAST is blocked until I see the real signature (section 6). |
| 10 | **How does `CoalitionMix` take an external `S`?** And the loader convention. | Providers are called keyword-only: `forward(p=, x=, h=, s=, gt=, k_slots=, mask=)`, aligned with the other inputs (`[:, :-1]` online, `[:, 1:]` target). My algo yamls assume `membership.target(**args)` and `mixer.target(spec, **args)`. | Controls only work if `CoalitionMix` accepts an injected provider instead of always building its graph net. |
| 11 | **Sign of the nearest-package offset differs between `fake.py` and the env.** | `fake.py` stores `agent - package`; `transport.py` stores `package - agent`. `NearestPackageProvider` takes `offset_sign` (+1 default for the real env, -1 for fake), so no shared file needs to change. | Only a trap if someone runs the nearest control on the fake env with the default. |

---

## 2. Files I own, and where they live

```
src/dc/envs/labels.py             contact + delivery geometry (pure functions, numpy or torch)
src/dc/envs/transport.py          the VMAS environment and its wrapper
src/dc/tools/physics_probe.py     physics tuning tool (python -m dc.tools.physics_probe)
src/dc/eval/ari.py                adjusted Rand index (numpy, no sklearn needed)
src/dc/eval/switch.py             switch rate and pair-flip rate
src/dc/eval/metrics.py            per-episode and per-eval-point aggregation
src/dc/eval/evaluator.py          record IO, metrics.csv merge, CLI
src/dc/membership/_xp.py          numpy/torch shim shared by the providers (private)
src/dc/membership/oracle.py       OracleProvider (ground-truth contact)
src/dc/membership/position.py     PositionClusterProvider (k-means / soft), GridProvider, grid helpers
src/dc/membership/random_s.py     RandomProvider (Dirichlet)
src/dc/membership/nearest.py      NearestPackageProvider
src/dc/mixers/vast.py             VASTSpatial + pure pooling helper (QMIX injected)
configs/algo/oracle.yaml  poscluster.yaml  random_s.yaml  nearest.yaml  vast_spatial.yaml
configs/env/transport_n12.yaml    N=12, M=4, K=6
configs/env/transport_n24.yaml    N=24, M=8, K=12
tests/test_labels.py  test_env_contract.py  test_physics_probe.py  test_metrics.py  test_providers.py
```

`labels.py`, the whole `eval/` package, the whole `membership/` package and `vast.py`'s helpers
stay importable without torch.
`transport.py` imports torch and VMAS lazily, so CI without torch can still import it.

---

## 3. The environment

### 3.1 API (proposed, needs your sign-off)

```python
env = TransportEnv(spec, num_envs=128, device="cpu", seed=0, **physics)
data = env.reset()
data, reward, done, info = env.step(actions)      # actions: (E, N) ints in [0, 4]
```

- `data` holds the observation-like batch keys for the *current* state (`obs`, `state`,
  `positions`, `agent_features`, `avail_actions`) plus `gt_coalitions`. Shapes follow
  `Architecture.md` §8 minus the batch/time dims, so they are `(E, N, ...)`.
- `reward` and `done` are `(E, 1)`.
- `info` has `delivered_now` (E,), `delivered_total` (E,) and `truncated` (E, 1).
- **All envs run synchronous 200-step episodes and reset together.** `done` is true for every
  env on the same step. `step` raises if you call it after the limit without `reset()`.
- `rollout_to_batch(env, horizon)` in `transport.py` shows the T / T+1 packing and is what the tests use.
  Observation-like keys get T+1 entries. `actions`, `rewards`, `dones`, `mask` get T.
  `gt_coalitions` gets T entries: the label of the state each action was taken in.
- The wrapper checks `spec` against the env and raises `ContractError` on any mismatch
  (`obs_dim`, `state_dim`, `x_dim=7`, `n_actions=5`).
- Device: I develop and test on CPU. `device.py` keeps MPS opt-in, and I never use it for the env.

### 3.2 Design decisions

| Decision | What I chose | Why / trade-off |
|---|---|---|
| **Base** | Fork of VMAS `transport`, VMAS 1.5.2, torch 2.15.0 on my machine | The plan says modify `transport`. I changed the scenario class, not VMAS itself. |
| **Actions** | 5 actions: stay, up, down, left, right | VMAS discrete mode is 3 per axis (9 total). I map our 5 to VMAS flat indices through a lookup table, which is how we keep `continuous_actions=False`. A test checks each direction moves agents the right way. |
| **Observation** | Own position and velocity (4 numbers) plus the **2 nearest packages**, 6 numbers each (offset to agent, offset to goal, package velocity). `obs_dim = 4 + 6k = 16` | Stock VMAS shows every package, which cannot fit `obs_dim=16` at M=4 and would grow with M. Partial observability is a deliberate choice, and it keeps `obs_dim` constant from N=12 to N=24. |
| **State** | Agent pos+vel, package pos+vel, goal pos. `state_dim = 4N + 4M + 2` (66 at N=12, 130 at N=24) | The `state_dim=24` in `base.yaml` was a placeholder, so I set it per env in my yaml. |
| **Agent features `x`** | position (2), velocity (2), offset to the nearest package (2), distance to it (1) = 7 | Matches `x_dim=7` and `Architecture.md` (never actions or utilities). The offset is `package - agent`. Note: `fake.py` uses the opposite sign, which is harmless since nothing compares them. |
| **`avail_actions`** | All ones | As the contract says for VMAS transport. |
| **Episode end** | Time limit only; `max_steps=None` in VMAS and the wrapper owns the counter | Packages respawn, so there is no natural terminal. See question 3 in section 1. |
| **Where game logic runs** | VMAS `post_step()`: progress, delivery, respawn | `post_step` runs after physics and before observations are built, so observations never show a stale package after a delivery. |
| **Delivery** | Package *centre* inside the goal circle (radius 0.15) | Deterministic and easy to test. Slightly stricter than VMAS's box-vs-circle overlap. |
| **Respawn** | Delivered package jumps to a random valid spot and its velocity, rotation and spin are zeroed. Rejection sampling over 16 candidates: not near the goal, not overlapping other packages. Agents are ignored. | Forces coalitions to re-form inside an episode. If no candidate is valid (rare) it falls back to the first one, which could overlap something. I did not measure how rare. |
| **Reward** | `shaping_factor * progress toward goal` + `delivery_bonus` per delivery − `collision_penalty` per overlapping agent pair. Defaults: 10, 5, 0.01. **All three are placeholders, not tuned.** | Progress is re-baselined on respawn, so the teleport does not create a fake reward. Caveat: agents pushing side by side touch each other, so the collision penalty slightly discourages the behaviour we want. It is small, but flagged. |
| **Teams** | One goal per episode, random spawn; M = N/3 packages; each package a 0.15 × 0.15 box | Matches the plan. N=24 uses the same constants as N=12 and **has not been separately probed**. |

### 3.3 Ground-truth labels (what ARI is scored against)

`gt_coalitions[i] = k` (1-based) if agent *i*'s circle is within **0.02** of package *k*'s rotated box, else `0`.

- The distance is computed as agent-centre-to-box-boundary (a signed distance) minus the agent radius. It is negative when an agent sinks into the box, which soft collisions allow.
- Tie-break: the nearer package; an exact tie goes to the lowest index.
- The label at step *t* describes the state at *t* (before the action taken at *t*).
- The 0.02 threshold is my choice, not from the plan. Raising it makes "contact" more generous and ARI-contact scores more agents.

---

## 4. Physics tuning (Set 2)

### 4.1 What the probe showed

The placeholder VMAS defaults cannot work. Package speed fits one formula, and I checked it
against all 15 entries of the first probe table:

```
speed (units/step) = 0.04 * u * k / (mass + k)        no friction, u = agent_u_multiplier
```

Interpretation (my inference from the fit): each agent applies force `u`, each pusher adds its own mass of 1 to the load,
and linear drag caps the speed. Consequences:
- Agent force is the real speed cap, not package mass. At the default `u = 0.6` no package can ever move faster than 0.024 units/step (42 steps per unit).
- This physics is smooth and concave in the number of pushers. Three pushers beat two by at most 1.5×, so a coalition of 3 is not actually *required*.

### 4.2 The three candidates and my choice

Estimated steps to carry a package 1 unit (from the probe runs on my machine):

| Config | k=1 | k=2 | k=3 | Notes |
|---|---|---|---|---|
| **A**: `u=1.2`, mass 8, no friction | 188 | 104 | 76 | Smooth gradient, but weak 2→3 gain (1.4×). |
| **B**: `u=2.0`, mass 3, friction 0.93 | 200 | 103 | **47** | **Chosen.** 3 is 2.2× faster than 2 and 4× faster than 1. |
| **C**: `u=2.0`, mass 3, friction 1.27 | 200 | 106 | 68 | Weaker threshold effect than B. |

**Chosen constants (in both yaml files):** `agent_u_multiplier=2.0`, `package_mass=3.0`, `package_friction=0.93`.

`package_friction` is a knob I added to `transport.py`. It sets VMAS's Coulomb-style `linear_friction`
on packages, so a package only moves once `k * u > friction * mass`. **This is a deviation from "stock physics"**:
the plan asks for "1–2 agents move it slowly, 3 clearly faster", and linear physics cannot deliver that. Config A is
the fallback if the hard threshold hurts early learning.

### 4.3 Things the probe cannot tell us

- **A single pusher is not stuck under friction.** I predicted it would be, and it was not: k=1 creeps at about 0.005 units/step (200 steps per unit), identical for both friction values. My explanation is that VMAS applies friction only once something is moving, so the package gets a small kick every step. The explanation is consistent with the numbers but I did not isolate it. In practice it is what we want: some progress signal at k=1, but nothing that delivers.
- **k=4 is not meaningful.** A 0.15-wide face is touched by at most 3 agents. The probe queues extra agents behind the front rank, and they transmit force poorly (k=4 came out slower than k=3). Effective coalition size on one face is therefore about 3.
- **Tunneling:** at `u=2.0` a free agent moves 0.08 units/step. A test (`test_a_full_speed_agent_does_not_tunnel_through_a_package`) passes, but it covers one head-on case only.
- **The probe is an empty-arena straight push.** It says nothing about off-centre pushes, agents arriving at different times, or whether any learner can find the behaviour. The real go/no-go is the QMIX run at Sync A.

---

## 5. The evaluator (Set 3)

### 5.1 What you need to know to produce records

Per evaluation episode, build an `EvalRecord` and call `write_record(record, run_dir)`:
- `assignments` (T, N) int: the argmax over the K+1 slots of `S` for each agent and step. Column 0 is background.
  For QMIX and VDN, fill with `-1`.
- `gt_coalitions` (T, N) int: from the env's `data["gt_coalitions"]` at each of the T steps (the T states the actions were taken in, not the final T+1 state).
- `in_contact` (T, N) bool: `gt_coalitions > 0`.
- `episode_return`, `packages_delivered` (the env's `info["delivered_total"]` at the last step), plus run identifiers.

File layout: `<run_dir>/eval/step_<9-digit env_step>/ep_<3-digit idx>.npz`, no pickles, numpy only.

Afterwards: `python -m dc.eval.evaluator experiments/<run_id>`. It takes a fraction of a second per evaluation point (16 episodes × 200 steps × 12 agents).

### 5.2 Decisions and judgment calls

| Decision | What I chose | Why / trade-off |
|---|---|---|
| **ARI implementation** | Plain numpy, not sklearn | No dependency at train time. It matches `sklearn.adjusted_rand_score` to 1e-16 on 5,000 random cases, including degenerate partitions. A test compares them whenever sklearn is installed. |
| **ARI (contact)** | Only agents in contact; per step; steps with fewer than 2 such agents are skipped | As the plan specifies. Background is ignored because "touching nothing" has no true group to recover. |
| **ARI (all agents)** | Everyone; "no contact" is its own true cluster, background slot its own predicted cluster | As the plan specifies. |
| **Averaging** | Per-episode mean over scored steps, then mean over episodes (every episode weighs the same); episodes with no scored step are dropped; none at all gives an empty cell | **The plan does not say**, so this is my call. Pooling over all steps would weight longer or busier episodes more. Easy to change in `metrics.py`. |
| **Std and means** | `return_std` is the sample std (ddof=1) across episodes | One episode gives 0.0. |
| **ARI is coarse** | At a step where every contact agent touches the *same* package the true partition is one cluster, so ARI is 1 or 0 | Correct but noisy early in episodes. Prefer per-step curves over only the mean. |
| **Switch rate** | Mean fraction of agents whose argmax slot changes between consecutive steps (all agents, including background) | The spec metric. Slot labels are arbitrary, so a pure relabelling counts as a false switch, which is the caveat from the plan. |
| **`pair_flip_rate`** | Fraction of agent *pairs* whose "same coalition?" answer flips per step | Relabelling-invariant, so a high switch rate can be diagnosed as real regrouping vs label churn. **Not a `metrics.csv` column**, since the schema is frozen. It is available for `summary.json`. |
| **No-membership methods** | `-1` in `assignments` leaves the ARI and switch columns empty | A 0 would be read as "bad grouping". |
| **`metrics.csv` merge** | Keep every value the trainer wrote; always recompute `ari_contact`, `ari_all`, `switch_rate`; fill other columns only if empty; refuse files with columns outside the schema; idempotent | Avoids two writers fighting over one file. |
| **Success criterion mapping** | "At most 1 switch per agent per 50 steps while static" equals `switch_rate <= 0.02` over static stretches | Not computed automatically, since "static stretch" needs delivery events. |

### 5.3 Not done

- **"Regroup within 10 steps after a delivery"** (in ≥ 80% of events) is not implemented. It needs delivery events and a precise definition of "regroup". See question 7 in section 1.
- No `summary.json` writer yet; its schema is not defined anywhere.

---

## 6. Controls and VAST-spatial (Set 4)

### 6.1 Shared conventions

Every control is a `MembershipProvider`: keyword-only `forward(p, x, h, s, gt, k_slots, mask)` returning
`S` of shape `(B, T, N, K+1)`, float32, rows summing to 1, column 0 = background. `h`, `s` and `mask` are
ignored by all of them; `S` is valid even on padded steps, so downstream masking works as usual. Only the
oracle reads `gt`, and a test checks that no other provider changes its output when `gt` is removed.
All providers are **parameter-free**, so target networks need no copy for them.

I wrote the providers duck-typed over numpy and torch (same approach as `labels.py`), through the small
`membership/_xp.py` shim. The reason: CI has no torch, and I could verify the math in numpy. The torch
paths are covered by gated tests that I could not run in my sandbox (section 7).

### 6.2 Decisions per control

| Control | What I chose | Why / trade-off |
|---|---|---|
| **Oracle** | One-hot of `gt`: label `k` goes to slot `k`, label 0 to background. Slots `M+1..K` stay empty. `check=True` validates the label range (costs a device sync; set `False` in the hot loop). | K = N/2 > M = N/3 guarantees every package a slot. This is the ceiling control. See question 8 about T vs T+1 labels. |
| **Position clustering** | Deterministic k-means per step: farthest-point init starting at agent 0, then 10 Lloyd steps, empty clusters keep their old centre. `mode="soft"` softens with an RBF over the centre distances (`tau=0.1`, **untuned**). Column 0 stays empty. | No RNG, so paired runs and tests are reproducible. Per-step clustering means slot *labels* can flicker between steps. That is harmless to the mixer (shared `g`) but inflates `switch_rate`; read `pair_flip_rate` alongside it. |
| **Grid** | `GridProvider`: hard assignment to K fixed map regions, `grid_shape(K)` = most-square factorisation (6 -> 3x2, 12 -> 4x3). Arena `[-1, 1]^2`, outliers clamped into border cells. | Not in the original control list; added because VAST needs slots with a stable meaning (below). |
| **Random S** | Dirichlet(alpha=1.0) over all K+1 columns, so background gets about 1/(K+1) of the mass like every slot. `mode="iid"` (default) draws fresh noise every step; `mode="episode"` draws once per episode so groups are random but static. | The plan says "random memberships (e.g., Dirichlet)". `iid` is pure noise; `episode` is closer to VAST's random sub-teams, so both are available. `alpha` is kept at 1.0 because float32 Dirichlet with tiny alpha can underflow. Torch path uses torch's *global* RNG (seed with `set_seed`); the numpy path has its own seeded generator. |
| **Nearest package** | Providers get `x` but not package identity, so identity is recovered from geometry: `target = p + offset_sign * offset` is the nearest package's position, and agents share a group when their targets agree within `tol=1e-4`. Groups are numbered by their lowest-index member and capped at K. `radius` (default `None`) optionally sends far agents to background. | Needs no contract change. It reads `x` columns 4:6 (offset) and 6 (distance), which is a layout assumption. See question 11 for the sign trap. Nearest is by *centre* distance, whereas the ground-truth label uses *edge* distance, so the two can differ slightly in the real env. |

### 6.3 VAST-spatial (partly done)

- **What it is:** hard spatial sub-teams, sub-team value = plain sum of members' utilities, then QMIX over the K values.
  K = N/2 sub-teams, the same K as ours; that is my assumption, VAST itself does not fix K.
- **Assignment default is `grid`, not k-means.** QMIX's hypernetwork weights are per input position, so slot `c`
  must mean the same thing at every step. A fixed grid cell does; a per-step k-means index is an arbitrary
  permutation. `assignment="kmeans"` is available. EXPERIMENT_PLAN §10 lists grid-vs-k-means as open; this is my
  recommendation, not a decision. Using k-means would likely handicap VAST for a reason unrelated to its idea.
- **QMIX is injected**, so I neither edit nor assume the internals of your `qmix.py`. `VASTSpatial(spec, mixer)` takes
  any module mapping `(values (B,T,K), state)` to `(B,T,1)`. `VASTSpatial.build(...)` (or the config-friendly
  `build_vast_spatial`) resolves a dotted path and calls it with `n_agents=K, state_dim=...`, which is **my assumption
  about your constructor** (question 9).
- `forward` mirrors `coalition_mix`: `(q_sel, h, x, s, p, *, detach, temperature)` returning `(Q_tot, S, aux)`, with
  `h, x, detach, temperature` ignored. It returns a one-hot `S` so the evaluator can read assignments like for any method.
- IGM still holds (non-negative sums into a monotone QMIX).
- The mixer *does* have parameters, so unlike the providers it needs a target copy and a Polyak update.
- **"Done when: VAST-spatial trains on the fake env" is NOT met.** It needs your `qmix.py` and your loop. Until then the
  pooling math is tested (numpy) and the module is tested with a stand-in mixer (torch, gated).

### 6.4 Configs

`configs/algo/{oracle,poscluster,random_s,nearest}.yaml` follow the dotted-path convention in `work_division.md`
(membership target + `CoalitionMix` mixer). A test resolves every membership target in them and builds the provider.
`vast_spatial.yaml` names `dc.mixers.vast:build_vast_spatial`. Whether your loader calls `target(**args)` for
membership and `target(spec, **args)` for the mixer is my assumption (question 10). `detach` is irrelevant for the
controls, since nothing trains `S`; I left it `false`.

---

## 7. What is verified and what is not

**Verified (ran on real VMAS on my machine):**
- Real rollouts pass `validate_batch` at N=12 and N=24; no NaNs over a 30-step random rollout.
- Action directions, contact labels, delivery counting and package respawn behave as designed; the time limit is the only `done`.
- Physics tables in section 4 are measured, not estimated.
- ARI is numerically identical to sklearn; hand-computed cases pass for ARI, switch rate and the metrics aggregation.
- **Set 4, numpy path (run in a sandbox without torch):** every provider on the fake env at N=12 and N=24 returns the
  right shape, float32, non-negative, rows summing to 1; the oracle equals the one-hot of `gt` and uses background exactly
  where nothing is touched; k-means (hard and soft) recovers planted blobs exactly (ARI = 1), is deterministic and
  translation invariant, and survives duplicate points; the grid maps cell centres to distinct fixed slots; random S uses
  background like any column and honours `iid`/`episode`/seed; nearest recovers planted groups, honours the sign and the
  radius, and on the fake env matches ground truth on contact agents exactly (ARI = 1.0); sub-team pooling conserves the total.

**Not verified:**
- **Set 4 torch paths.** 9 gated tests (numpy/torch parity for the deterministic providers, random S on torch, `VASTSpatial`
  with a stand-in mixer) could not run in the sandbox. They are written against torch semantics I checked by reading, not by
  running, so a first-run failure there is plausible.
- **VAST-spatial with real QMIX, and in training.** Blocked (section 6.3).
- Choice of `tau=0.1` for soft clustering and of the grid default; neither has been tuned or compared.
- That QMIX (or anything) can learn on this physics. That is the Sync A go/no-go.
- Reward constants. They are placeholders.
- N=24 physics. It was constructed and stepped, but never probed.
- Determinism of respawns under a fixed seed. The test checks reset positions only, and respawn uses torch's global RNG (relevant for paired Full-vs-Detached runs).
- How often respawn falls back to an invalid position.
- Behaviour on GPU or MPS. I only use CPU for the env.

---

## 8. Known limitations and risks

- **Hard threshold risk:** under friction a lone agent makes little progress, so early exploration depends on three agents meeting at a package. The plan's fallbacks apply: a small approach reward (risk: hovering), a light-package curriculum, a longer ε decay, or switching to config A.
- **Spatial bias in ground truth:** contact is purely geometric, so the oracle control and the contact ARI favour spatially compact groups, which is also what the detached ablation learns. This is inherent to the plan, not a bug.
- **Coalition size ceiling:** about 3 agents fit on one face of a package, so larger coalitions only help if they push from several faces.
- **Per-step control labels flicker.** Position clustering, nearest and iid random S relabel slots between steps, which
  inflates `switch_rate` without any real regrouping. Compare them using ARI and `pair_flip_rate`.
- **Controls inherit the spatial bias** of the ground truth and of the detached ablation. The oracle, nearest and
  position controls all favour spatially compact groups, so small gaps between them are expected.
- **Observation sees only the 2 nearest packages.** An agent far from any package gets little information about the others.

---

## 9. What comes next (my side)

1. **Finish VAST-spatial** once I have `mixers/qmix.py` (questions 9 and 10), including a smoke run on the fake env.
2. **Set 5: analysis**: rliable aggregation and the figures, built on synthetic CSVs first.
3. **Unpaired runs at N=12** on the M1 (VDN, random S, nearest, position clustering) once your training loop exists.
4. Revisit the regroup-after-delivery metric if you add `meta["deliveries"]` (question 7).

## 10. How to run and test my side

```bash
python -m pytest -q -rs                                   # whole suite; 0 skipped once sklearn is installed
python -m dc.tools.physics_probe --u-mult 2.0 --masses 3 --frictions 0.93 1.27 --pushers 0 1 2 3
python -m dc.eval.evaluator experiments/<run_id>          # after eval records are dumped
python -m pytest tests/test_providers.py -q -rs           # Set 4; torch-gated tests run only with torch
```

Use `python -m pytest`, not bare `pytest`. On my machine bare `pytest` resolved to a pyenv shim outside the venv and silently skipped every torch and VMAS test.

---

## Update log

*(Updated after every set, with code and tests.)*

- **Set 1 (env):** `labels.py`, `transport.py`, env contract tests, N=12/N=24 configs.
- **Set 2 (physics):** `physics_probe.py`; chose `u=2.0`, mass 3, friction 0.93 (config B); added the `package_friction` knob.
- **Set 3 (evaluator):** `ari.py`, `switch.py`, `metrics.py`, `evaluator.py`, `test_metrics.py`.
- **Set 4 (controls):** the four providers, `GridProvider`, `vast.py` (QMIX injected, unfinished), five algo yamls,
  `test_providers.py`. New open questions 8 to 11. Added section 6; sections after it renumbered.
