# Project Context: Why We're Doing What We're Doing

## 1. What DynaCoalition is, in one paragraph

DynaCoalition is a cooperative MARL value-factorization method built on QMIX. Instead of mixing all agents' utilities in one flat mixer, it softly groups agents into **coalitions** at every timestep, pools utilities within each coalition, and combines the coalition values with a shared monotone mixer. Each agent still acts greedily on its own utility at test time, exactly as in QMIX. The coalitions exist only during training: they shape **credit assignment** (how the team's TD error is distributed among agents), not how agents act.

## 2. The research question

**Not:** "Are dynamic groups better than fixed groups?" VAST and GoMARL already showed that grouping helps.

**Our question:** *Given that grouping helps, what should decide the groups?*

- **Full DynaCoalition:** memberships are shaped by the team's reward signal (the TD gradient reaches the membership network), plus light structure losses.
- **Detached ablation:** the same model, but memberships are shaped **only** by the structure losses, which mainly pull nearby agents together.

Both variants have soft, per-step groups. The only difference is whether the reward trains the grouping.

**Secondary question:** do reward-shaped groups match the task's true structure (ARI against ground-truth package labels) better than structure-only groups?

**What makes the result substantial is the analysis, not the headline:**
- **When does reward shaping help?** For example, larger teams, or more frequent package respawns.
- **Does the TD gradient actually keep reaching the grouping,** or does it vanish as memberships sharpen?
- **Do reward-shaped groups differ from spatial clusters?** For example, does an agent heading toward a package join that coalition before touching it?
- **Does better grouping (ARI) translate into better return?**

A clear answer either way is a valid result, as long as we can rule out training failures (see the diagnostics in EXPERIMENT_PLAN.md).

## 3. What's new compared with prior work

| Method | How groups are formed | How DynaCoalition differs |
|---|---|---|
| **VAST** (NeurIPS 2021) | Hard sub-teams from a separate assignment strategy: random, fixed, spatial, or a learned meta-gradient policy (MetaGrad). Each sub-team's value is the sum of its members' utilities | Our memberships are soft, recomputed every step, and trained directly by the TD loss inside the mixer |
| **GoMARL** (NeurIPS 2023) | Hard groups adjusted during training by a select-and-kick-out heuristic until they stabilize | Ours are soft and change within an episode |
| **HYGMA** (ICML 2025) | Spectral clustering of state histories, run periodically outside the learning objective | Ours sit inside the value mixer and can receive the TD gradient |
| **GACG** (2024) | Groups from observation histories, used in a coordination graph and a group behavior loss | Our graph is training-only and only produces coalition memberships for the mixer |

**The claim we make:** to our knowledge, no prior method learns soft, state-dependent coalition memberships trained directly by the value (TD) loss, and none tests that choice with a controlled detached ablation against ground-truth groups.

## 4. Project history and professor feedback

1. **Proposal (Word doc).**
   - **Plan:** SMAC/SMACv2 plus VMAS, a learned affinity, and the goal of beating GoMARL, VAST, and HYGMA.
   - **Grade: 4.5/5.** The only applied rubric item was *"Goal is not completely clear here"* (−0.5).
   - **Professor's comment:** *"I don't have a clear sense of what method you plan to use from the description given. Suggest you come sketch it out for me during office hours."*
   - The other rubric items shown on Gradescope (unclear evaluation, buried RL methodology, toy tasks) were **not** applied to us. They're the full rubric used for all teams.
2. **Office hours.** The professor was short on time and asked us to put the **architecture and pseudocode** into the literature-survey submission instead.
3. **Literature survey.**
   - **Contents:** a refined architecture, the new Figure 1, Algorithms 1–2, and 20 surveyed papers (10 per person), in a 4-page two-column format.
   - **Submission note:** Akhil submitted an earlier draft; Abhishek submitted the final version.
4. **The professor approved the draft.** We now have about 2 weeks to implement and produce final results.

**The lesson from the feedback:** always state the method as "QMIX with X changed," give one end-to-end walkthrough, and include a figure and pseudocode. Do this in the final report too.

## 5. What changed from the proposal to the survey, and why

| Change | Proposal | Survey (current) | Why |
|---|---|---|---|
| **Primary environment** | SMAC/SMACv2 plus VMAS | **VMAS multi-package transport**; SMAC optional | VMAS is faster to iterate on and provides **ground-truth coalitions** for measuring grouping quality |
| **Affinity W** | Learned: σ(eᵢᵀeⱼ/√d) | **Fixed, distance-based:** Mᵢⱼ·exp(−dᵢⱼ²/σ²) | With a learned W, nothing trains W in the detached ablation (TD doesn't reach it, and the cut loss uses W detached), so the ablation would have random affinities and the comparison would be unfair |
| **Graph network input** | Hidden states h | **sg(h)** (stop-gradient) | Agents learn only through their own Qᵢ, as in QMIX. Without it, structure losses (and TD through S) would reshape agent representations, adding a hidden difference between the variants |
| **What the detach covers** | "Block the gradient to S" | Covers **everything computed from S**: pooled Q_k, the background slot, m_k, and z_k | Otherwise the TD gradient leaks into S through m_k and z_k |
| **Softmax temperature** | Not specified | **Temperature T, sharpened only late**, with logging of the TD gradient's size at the graph network | Sharp memberships saturate the softmax and kill the TD gradient, silently turning the full method into the ablation |
| **Q_tot formula** | Background head and −g(0) only in text | **Both in the equation** | Precision |
| **Mixer over coalitions** | Per-coalition mixers (in a draft) | **One shared g applied to every slot, then summed** | Slot labels are arbitrary and can swap between steps. A shared g plus a sum doesn't depend on slot order |
| **"Linear g" claim** | "Grouping would not matter" | Grouping still rescales credit, but coalitions can't model interactions among members | The original claim was wrong |
| **Target networks** | "Updated periodically" (hard copy) | **Soft Polyak**, with the same rule for every method | The target's grouping drifts smoothly instead of jumping, and the comparison stays fair |
| **Orthogonality loss** | MinCutPool's orthogonality term | **A size cap plus a sharpness (entropy) term** | Orthogonality forces equal-sized coalitions, but transport has uneven and empty ones |
| **Baselines** | Beat GoMARL, VAST, HYGMA on SMAC | QMIX, VDN, **VAST-spatial**, the detached ablation, plus controls | GoMARL and HYGMA move with SMAC to the optional extension; VAST-spatial is a real prior-work baseline on VMAS |
| **Success criteria** | Relative to GoMARL and others | **Relative to QMIX, VAST, and the detached ablation**; the grouping and switching criteria are unchanged | Follows from the baseline change |

**Overall:** the survey design is better because the **key experiment is now valid** (the variants differ in exactly one thing), the claims are correct, the method is complete (pseudocode, defined terms), and failure modes are anticipated (saturation logging, a shared target rule).

## 6. Trade-offs and known limitations

- **A fixed W biases toward spatial groups.** The cut loss always pulls nearby agents together. The full method can still learn non-spatial groupings (the graph network sees sg(h) and features x such as velocity and nearest-package direction), but it has to outweigh that pull. **Keep β_cut small.**
- **The detached ablation is roughly "smoothed spatial clustering."** If it performs about the same as the position-clustering control, that's expected.
- **sg(h) removes one possible benefit.** The grouping can't shape agent representations. That's the price of a clean comparison.
- **The monotonic mixer can't escape coordination traps.** Example: action A is good only if a partner also picks A. This is relative overgeneralization. Transport has this structure (packages need several agents pushing), which is why **reward shaping and physics tuning** matter.
- **g sees only the pooled sum Q_k,** so it can't express "agent A matters only if agent B is present." That's less expressive than a full QMIX mixer within a coalition.

## 7. Future-work ideas (not in the current scope)

- **A learned affinity trained by the same auxiliary loss in both variants,** for example predicting which agents will touch the same package soon. It restores flexibility without breaking the fair comparison.
- **Coordination within coalitions at test time** (Deep Coordination Graphs style): pairwise payoffs among coalition members, with a joint action chosen within each small coalition. This addresses the monotonicity limitation, but needs communication and hard coalitions.
- **A QPLEX-style, IGM-preserving combiner** inside coalitions.
- **Owen-value credit within learned coalitions:** a two-level Shapley value whose cost depends on coalition size, not team size. Compare against SHAQ and Shapley Counterfactual Credits.
- **Robustness to agent failure, and larger teams** (N ≥ 96 with kNN affinities).
