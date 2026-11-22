# Concepts Primer

A summary of the background needed for DynaCoalition. Each topic ends with why it matters for this project.

---

## 1. How QMIX trains

**Q-values.** Q(situation, action) is the expected total future reward from taking an action now and acting well afterward. If the Q-values are right, acting is easy: pick the highest one. They're learned by moving each estimate toward a **target**, reward + γ·Q(next situation). The gap between the estimate and the target is the **TD error**.

**The multi-agent problem.** With a shared team reward:
- **Credit assignment:** who earned the reward?
- **Joint actions:** there are too many to handle together (5¹² ≈ 244M for 12 agents).

**Value factorization.** Each agent learns its own Qᵢ, and these are combined into a team value Q_tot:
- **VDN:** a simple sum.
- **QMIX:** a learned **mixer**, constrained so that raising any Qᵢ never lowers Q_tot.

That constraint is what allows **centralized training, decentralized execution (CTDE)**: each agent maximizes its own Qᵢ at test time, which also maximizes Q_tot.

**The parts:**
- **Agent network (shared by all agents).**
  - **Input:** the agent's observation, plus its last action and a one-hot agent ID.
  - **Body:** a GRU, whose hidden state hᵢ is the agent's memory.
  - **Output:** one Q-value per action.
- **Mixer (training only).** Takes the Qᵢ of the actions actually taken, plus the **global state s**. Small **hypernetworks** read s and generate the mixer's weights, which are forced non-negative (absolute value) to keep monotonicity.

**The training loop:**
1. Agents play **whole episodes** with ε-greedy exploration; store the episodes in a replay buffer.
2. Sample a batch of episodes, and run the GRU from t = 0 through each one (memory has to be rebuilt step by step).
3. Mix the chosen-action Qᵢ values into Q_tot(t).
4. Target: y(t) = r(t) + γ·(1 − done)·Q_tot^target(t+1).
5. Loss = the **masked** mean of (Q_tot − y)². Masks remove the padded steps of shorter episodes. Backpropagate through the mixer into the agents.

**Tensor shapes:** observations (B, T, N, obs_dim) → Q-values (B, T, N, n_actions) → Q_tot (B, T, 1).

**Credit assignment lives in the mixer.** Its weights decide how much of the team's TD error each agent's Qᵢ receives.

**Implementation details matter.** The pymarl2 paper ("Rethinking Implementation Tricks") showed QMIX improves a lot with Adam, TD(λ) targets, longer ε decay, and larger replay buffers. BenchMARL found QMIX weak on VMAS.

**For us:** DynaCoalition changes only the mixer. Our QMIX baseline must be well tuned, or our results won't be credible.

---

## 2. Monotonicity, IGM, and their limits

**IGM (Individual-Global-Max):** the joint action that maximizes Q_tot equals each agent independently maximizing its own Qᵢ. Monotonicity (∂Q_tot/∂Qᵢ ≥ 0) is **sufficient** for IGM, but **not necessary**.

**The limitation, shown with a coordination game:**

| | Partner: A | Partner: B |
|---|---|---|
| **Me: A** | **+8** | −12 |
| **Me: B** | −12 | 0 |

- Action A is only good if the partner also picks A.
- While the partner explores randomly, my Qᵢ(A) learns the **average** outcome of A, which includes many −12s.
- So A looks bad, and QMIX tends to settle on the safe (B, B) for 0.

This is **relative overgeneralization**. QPLEX and Weighted QMIX relax monotonicity while keeping IGM.

**Allowing non-monotonic mixing inside coalitions?** We discussed it. It breaks independent action: coalition members would need to coordinate their choices at test time, which needs communication and hard coalitions. Kept as future work.

**For us:** our mixer is monotonic, so it inherits this limitation. Within a coalition it's even more limited, since g only sees the sum. Transport has exactly this structure: a package only moves if several agents push.

---

## 3. The MinCutPool loss

**Goal:** given affinities W, find soft memberships S that keep strongly connected agents together.

**L_cut = −Tr(SᵀWS) / Tr(SᵀDS)**, where D = diag(dᵢ) and dᵢ = Σⱼ Wᵢⱼ. It ranges from −1 (best) to 0.

**An intuitive reading:**
- **overlap(i, j) = Σₖ Sᵢₖ Sⱼₖ:** the chance that i and j land in the same group if each is placed randomly according to its membership row.
- **self-overlap(i) = Σₖ Sᵢₖ²:** the chance that two random placements of agent i land in the same group. It's 1 for a fully decided agent and lower for fuzzy ones. **It isn't about an edge to itself (Wᵢᵢ = 0).** It measures how decisive the agent is.
- **The ratio:** the numerator sums overlaps between connected pairs; the denominator sums each agent's self-overlap, weighted by its connections. So L_cut asks: "how much do connected agents agree, relative to the most they could agree?" An agent can't agree with a neighbor more than it agrees with itself.

**L_cut = −1 whenever connected agents have identical membership rows.** That includes two useless groupings:
1. **Everyone in one group** (collapse).
2. **Everyone spread evenly across groups**, e.g., all agents at (0.5, 0.5): overlap = self-overlap = 0.5.

MinCutPool prevents both with an **orthogonality** term, but that also forces equal-sized groups. We replace it with:
- **L_cap:** penalizes coalitions larger than a cap. Prevents collapse.
- **L_sharp:** the average row entropy. Prevents fuzzy rows.

The background slot is excluded from the cut.

**For us:** with a fixed distance-based W, the cut loss pulls nearby agents together. The detached ablation becomes roughly smoothed spatial clustering, while the full method can let the reward override it.

---

## 4. Softmax temperature and gradient saturation

**Temperature.** S = softmax(scores / T). For scores (2, 1, 0):

| T | Memberships |
|---|---|
| 5 | (0.40, 0.33, 0.27): fuzzy |
| 1 | (0.67, 0.24, 0.09) |
| 0.25 | (0.98, 0.02, 0.00): nearly hard |

**Saturation.** A membership's sensitivity to its score is proportional to p(1 − p): 0.25 at p = 0.5, but about 0.01 at p = 0.99. Near-hard memberships pass almost no gradient.

**Why it matters:** if memberships sharpen too early (T drops fast, or β_sharp is too large), the TD gradient into S vanishes. The full method then silently becomes the detached ablation, and a null result would be misread as "the reward doesn't help."

**Mitigations:**
- Keep T high and β_sharp small for most of training, and sharpen only late.
- Log the TD gradient's size at the graph network, ‖∇_GNN L_TD‖, and the mean max membership.
- Note that evaluation uses argmax memberships, so sharpness isn't needed to evaluate.
- **Fallback:** straight-through Gumbel-softmax (hard values going forward, soft gradients going backward).

---

## 5. Double-Q targets and Polyak updates

**Target network** (fixes the moving target). The target uses Q(next step) from the network being trained, so every update moves the target too, which makes training unstable. The fix is a slow copy θ⁻ used only for targets.
- **Hard update:** copy θ → θ⁻ every C steps (the target jumps).
- **Soft (Polyak) update:** θ⁻ ← τθ + (1−τ)θ⁻ every step, τ ≈ 0.005 (a smooth running average).
- **We use soft updates,** because the target includes the graph network and its grouping shouldn't jump. **Use the same rule for every method.**

**Double Q** (fixes overestimation). The max over noisy estimates is biased upward: 5 actions with true value 0 and noise ±1 give a max of about +1.2, and the bias compounds. The fix:
- the **online** network **chooses** the next action,
- the **target** network **evaluates** that choice.

A noise spike in the online network probably isn't present in the target network, so the inflation cancels.

**The target network fixes instability; Double Q fixes the max bias.** We use both.

**Multi-agent recurrent version:** online agents pick a*ᵢ at t+1; target agents evaluate Qᵢ^target(t+1, a*ᵢ); the **target CoalitionMix** (target graph network, w-head, g, g_bg, b) mixes them into Q_tot^target(t+1). Both GRUs are unrolled from t = 0. Code and a bug checklist are in Architecture.md.

---

## 6. Evaluation statistics

**Seeds.** A seed fixes **all the randomness of one complete training run**: weight initialization, exploration, environment randomness, replay sampling. **One seed = one full training run from scratch**, producing a final **score** (return, packages delivered, or ARI, averaged over the last few evaluations). We use 5 seeds per method and setting, because RL results vary a lot between runs. Total runs = methods × team sizes × seeds. Seeds are unrelated to the bootstrap.

**ARI (adjusted Rand index)** measures grouping quality:
- It compares groupings through **pairs** of agents ("together or not?"), so arbitrary slot labels don't matter.
- It's corrected for chance: 1 = perfect, about 0 = random.
- Example: true groups [A,A,A,B,B,B] vs. learned [3,3,3,5,5,5] gives ARI = 1.
- Computed with `sklearn.metrics.adjusted_rand_score`, using each agent's argmax slot.
- Our main version uses agents in contact only (see EXPERIMENT_PLAN.md).

**Switch rate:** the fraction of agents whose argmax slot changes per step. **Caveat:** if slots get relabeled between steps, switches can be counted that didn't really happen.

**IQM (interquartile mean):** sort the runs, drop the top and bottom 25%, and average the rest. Example: [12, 48, 50, 53, 90] has a mean of 50.6 but an IQM of about 50.3. It's robust to one lucky or failed seed.

**Bootstrap confidence interval:**
1. Resample the seeds with replacement.
2. Recompute the IQM.
3. Repeat about 2,000 times.
4. The middle 95% of the results is the 95% CI.

"**Stratified**" means resampling within each task (e.g., N = 12 and N = 24) separately. With 5 seeds, the intervals are wide.

**Paired differences.** Run two methods on the **same seeds**, then compare seed by seed:

| Seed | Full | Detached | Difference |
|---|---|---|---|
| 1–5 | 52, 48, 60, 45, 55 | 50, 45, 57, 44, 51 | +2, +3, +3, +1, +4 |

The raw ranges overlap, but the full method wins on every seed. Bootstrap the differences; a 95% CI that excludes 0 is the evidence. Pairing removes seed-to-seed difficulty, which is why both methods should run **on the same machine**.

**rliable sketch:**
```python
from rliable import library as rly, metrics
scores = {"full": full, "detached": det, "qmix": qmix}   # each (n_seeds, n_tasks)
iqm = lambda x: np.array([metrics.aggregate_iqm(x)])
point, ci = rly.get_interval_estimates(scores, iqm, reps=2000)
```

---

## 7. VAST's actual design

1. **Assign each agent to exactly one of K sub-teams** every step (hard).
2. **Sub-team value:** Q_k = Σ (over i in sub-team k) Qᵢ.
3. **Combine** with a standard operator (VDN, QMIX, or QTRAN): Q_tot = Ψ(Q₁ … Q_K; s). The operator sees K values instead of N, which is VAST's scaling idea.
4. **Assignment strategies:** random, fixed, **spatial** (by map region), and **MetaGrad** (a learned meta-gradient assignment policy, VAST's main learned variant).

**Our baseline, "VAST-spatial":**
- hard groups by map region (k-means is acceptable if labeled "VAST-style"),
- members' utilities summed,
- a QMIX mixer over the K values,
- the same K as DynaCoalition.

With fixed regions, slot labels are meaningful, so an ordinary QMIX mixer works. In DynaCoalition, slot labels are arbitrary, which is why it uses a shared g.

**What "shared g" means:** g is **one** network applied separately to each coalition's (Q_k, s, z_k, m_k), with the results summed. A mixer with per-slot weights cares about slot order; a shared g plus a sum doesn't. It's like one judge scoring every team by the same rubric. Coalitions still get different contributions through their inputs (size, member features, value).

**The comparison ladder:**
- **VAST-spatial → position clustering in our mixer:** only the mixer changes.
- **Position clustering → detached:** learned soft groups replace fixed spatial ones, still without the reward.
- **Detached → full:** adds the reward signal to the grouping.

---

## 8. The VMAS transport scenario

**VMAS** is a 2D PyTorch physics simulator. Agents are circles moved by forces. It's **vectorized**: many environment copies run as one batch, so **every tensor has a leading batch dimension**. It runs on CPU or GPU.

**Stock transport:**
- **The task:** agents push a heavy package to a goal.
- **Reward:** shared and progress-based (how much the package got closer to the goal).
- **Actions:** 5 discrete (stay, up, down, left, right) with `continuous_actions=False`.
- **Observations:** the agent's own position and velocity, plus package and goal relative positions.

Check the actual `transport.py` source for exact defaults.

**One step:**
```python
obs, rewards, dones, infos = env.step(actions)
# obs: list of N tensors (num_envs, obs_dim); rewards: list of N (num_envs,); dones: (num_envs,)
```

Our modifications are described in EXPERIMENT_PLAN.md (environment design).

**Respawning matters:** it forces coalitions to break up and re-form within an episode. Without it, agents form one coalition and keep it, and there's nothing dynamic to learn.

**GPU:**
- **Environment:** helps when running **many** parallel environments (hundreds). With few, the CPU can match it.
- **Training:** a moderate speedup.
- **Practical rules:** keep tensors on one device, benchmark early, and run paired methods on the same device.

---

## 9. Exploration with joint tasks

**The problem:**
- **Sparse reward:** a package needs about 3 agents pushing together. Random ε-greedy behavior almost never produces that, so the team rarely sees reward early on.
- **The coordination trap (Topic 2):** an agent pushing alone gets nothing and learns "pushing is useless."

**The hidden catch:** progress-based shaping doesn't help if a package doesn't move at all with fewer than 3 agents. One or two pushers produce zero progress and zero reward.

**Fixes, in the environment:**
1. **Physics where a package moves slowly with 1–2 agents and faster with 3,** so joining earns a small reward.
2. **A small approach reward,** kept small because agents might learn to hover near packages.
3. **A curriculum:** light packages first.
4. **A longer ε decay.**

**Diagnosis:**

| What you see | Cause |
|---|---|
| No method learns, QMIX included | Environment problem |
| QMIX learns, DynaCoalition doesn't | Method or implementation problem |
| Both learn, with a difference | A real result |

**The day-3 check:** QMIX at N = 12 should start delivering packages.

**For us:** if agents never discover cooperative pushing, there's no reward signal to shape groups, and every grouping method looks the same.

---

## Quick clarifications

**If h has a stop-gradient, how do agents learn?** sg blocks only the path h → graph network → memberships. The main path, h → Qᵢ → mixer → Q_tot → TD loss, is open, as is the path through the w-head. So agents learn exactly as in QMIX. The grouping just can't reshape their memories.

**What is W?** It's not a neural network weight. It's an **affinity matrix** (connection strengths between pairs of agents), computed from distances by a formula, with no trainable parameters. "Fixed" means not learned. It's recomputed every step as agents move.

**Does a fixed W mean fixed groups?** No. The groups S come from the **learned** graph network, which also sees sg(h) and features x (velocity, nearest-package direction). In the full method, the TD gradient can teach it non-spatial groupings, such as an agent joining a coalition before arriving. **Two things work against this:**
- the cut loss pulls toward spatial clusters (keep β_cut small),
- the kNN mask limits message passing (use a large enough neighborhood).

**What is x?** Each agent's raw features from the environment: position, velocity, nearest-package information. Not learned, and never current actions or utilities.
