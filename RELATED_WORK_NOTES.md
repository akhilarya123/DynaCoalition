# Related Work Notes (Verified)

Facts checked against the papers, abstracts, or code during the literature survey. Use these when writing the final report, so the details don't need to be re-checked.

## Closest grouping methods

**VAST** (Phan et al., NeurIPS 2021)
- Factorizes the team value over sub-teams that can change over time; each sub-team's value is the **sum** of its members' utilities.
- A standard operator (VDN, QMIX, or QTRAN) combines the sub-team values.
- **Assignment strategies:** random, fixed, spatial, and **MetaGrad** (a learned meta-gradient assignment policy, VAST's main learned variant).
- **Don't** describe VAST's assignment as merely "given": MetaGrad is learned.
- Official code: github.com/thomyphan/scalable-marl

**GoMARL** (Zang et al., NeurIPS 2023)
- Learns automatic grouping without domain knowledge, with two-level (group-wise) value factorization and hierarchical control so agents in the same group specialize in similar policies.
- Groups change through a **select-and-kick-out** scheme (agents with low contribution are moved out) and **settle during training**. They aren't recomputed every step.
- Evaluated on SMAC and Google Research Football; built on PyMARL/PyMARL2; SMAC experiments used SC2.4.6.10.
- Official code: github.com/zyfsjycc/GoMARL. A fork (github.com/CalvinYang0/gomarl) adds fixed-grouping modes: one group, singletons, same type, manual.
- An independent paper reported GoMARL performing poorly on SMACv2, attributing it to slow convergence from dynamic grouping. Cite that paper directly if this is used.

**HYGMA** (Chiqiang Liu and Dazi Li, ICML 2025, PMLR vol. 267)
- Spectral clustering on agents' state histories builds time-varying hypergraphs, processed with attention-enhanced hypergraph networks under a unified task-plus-structural-regularization objective.
- Clustering runs **periodically**, outside the learning objective.
- Official code: github.com/mysteryelder/HYGMA

**GACG** (Duan, Lu, and Xuan, 2024, arXiv:2404.10976)
- A group-aware coordination graph capturing pairwise cooperation needs and group-level dependencies.
- Groups are hard, formed from observation histories, and used for communication in the graph plus a loss making group members behave similarly. **Not** value factorization over TD-trained coalitions.

**SCoUT** (Vora et al., Reinforcement Learning Journal / RLC 2026)
- Resamples soft groups every K steps with Gumbel-Softmax; the induced affinity is a differentiable prior over message recipients; a group-aware critic maps group values to per-agent values through the soft assignments.
- An actor-critic communication method with groups fixed over macro-steps.

**DCMGA** (B. Wang, L. Zhang, S. Zhou, H. Chen, J. Huang, J. Gong; *Expert Systems with Applications* 320:132180, 2026)
- Switches agents between global and local cooperation based on how many cooperative partners they have, relative to a learnable threshold.
- A group information-aggregation module gives each group a shared objective.
- Evaluated on SMACv2 and Battles.

## Credit assignment and game theory

- **SHAQ** (Wang, Zhang, Gu, Kim; NeurIPS 2022): a Shapley-based decomposition of Q_tot with a Shapley-Bellman optimality equation.
- **SQDDPG** (Wang et al., AAAI 2020): Shapley Q-values used with an actor-critic method.
- **Shapley Counterfactual Credits** (Li et al., KDD 2021): Monte Carlo-sampled Shapley values over coalitions of agents, evaluated on StarCraft II. The abstract doesn't say "policy gradient," so describe it as "credit assignment in deep MARL."
- **Owen value** (Owen, 1977; Lecture Notes in Economics and Mathematical Systems, vol. 141): the Shapley value for players in a priori unions. It averages over joining orders in which union members join together, so it captures contributions both between and within unions.
- **Nucleolus Credit Assignment** (arXiv 2503.00372, 2025): game-theoretic credit targeting multiple small coalitions. Relevant if Owen-value credit is pursued.

## Factorization, benchmarks, and tooling

- **QMIX** (Rashid et al., ICML 2018), **VDN** (Sunehag et al., AAMAS 2018), **QPLEX** (Wang et al., ICLR 2021: an IGM-preserving duplex dueling factorization, richer than monotonic).
- **pymarl2 / "Rethinking Implementation Tricks":** QMIX improves substantially with Adam, TD(λ), longer ε decay, and larger buffers. Use these settings for our QMIX baseline.
- **BenchMARL** (Bettini, Prorok, Moens; JMLR 2024):
  - **Environments:** VMAS, **SMACv2** (not SMAC), PettingZoo, MeltingPot, MAgent2.
  - **Algorithms:** includes QMIX, VDN, and MAPPO.
  - **On VMAS,** its Q-learning algorithms (IQL, VDN, QMIX) performed suboptimally, which motivates careful QMIX tuning.
- **VMAS** (Bettini, Kortvelesy, Blumenkamp, Prorok; DARS 2022): a vectorized PyTorch simulator that includes a `transport` scenario.
- **SMACv2** (Ellis et al., NeurIPS 2023): procedurally generated, stochastic scenarios, motivated by many SMAC tasks being solvable by open-loop policies.
- **MinCutPool** (Bianchi, Grattarola, Alippi; ICML 2020): a normalized-cut relaxation plus an orthogonality loss.
- **rliable** (Agarwal et al., NeurIPS 2021): IQM, performance profiles, stratified bootstrap confidence intervals.
- **Deep Coordination Graphs** (Böhmer et al., 2020): joint action selection via pairwise payoffs and message passing. Relevant to the future-work idea of coordinating within coalitions.

## Citation corrections made in the survey

- **[15] DCMGA:** "Wang et al." became the full author list, journal, volume, and article number.
- **[10] BenchMARL:** SMAC → SMACv2.
- **[3] VAST:** the description now mentions the learned MetaGrad strategy.
- **[17] Shapley Counterfactual Credits:** the unverified "policy-gradient" wording was removed.
- **[7] Owen:** the full series details (LNEMS vol. 141) were added.
- **[12] HYGMA:** PMLR vol. 267 was added.
