# DynaCoalition in a Nutshell

## The idea

In cooperative MARL, agents share one team reward, but usually only a few agents need to work closely together at any moment (e.g. 3 agents pushing one package). QMIX mixes all agents' values in one big monotonic function. **DynaCoalition adds a middle step: agents are softly grouped into coalitions, values are pooled inside each coalition, and then the coalitions are combined.**

**Question we test:** does letting the team TD loss shape the grouping give better groups and better return than learning the grouping from structure alone?

---

## Architecture

```
obs_i ─► GRU ─► h_i ─► Q_i(a_i)            ← the only part used at execution
                 │
                 ▼  (training only)
          embedding e_i ─► affinity W_ij
                              │
                        graph net ─► S (soft membership, rows sum to 1)
                              │
          Q_k = Σ_i S_ik · w_i · Q_i      (pool utilities per coalition)
                              │
          Q_tot = b(s) + Σ_k g(Q_k)        (same monotone g for every coalition)
                              │
          TD loss ─► agents, mixer, S, W
          + structure losses ─► S
```

**Step by step**

1. **Agent network:** each agent's GRU gives hidden state `h_i` and utilities `Q_i`. This is all that runs at test time.
2. **Affinity:** how strongly two nearby agents should be linked.
3. **Membership `S`:** a small graph network turns the affinity graph into soft coalition assignments. One extra "background" column holds agents that belong to no coalition.
4. **Pooling:** each coalition's value is a weighted sum of its members' utilities.
5. **Mixing:** one shared monotone network `g` turns each coalition value into a contribution to `Q_tot`.

---

## The math

**1. Affinity** (only between nearby agents, via a kNN/radius mask `M`)

$$W_{ij} = M_{ij}\sigma\left(\frac{e_i^\top e_j}{\sqrt d}\right), \qquad W_{ii}=0$$

**2. Soft membership** (`K` coalition slots + 1 background slot)

$$S = \mathrm{softmax}\big(\mathrm{GNN}(W, h, x)\big) \in \mathbb{R}^{N\times(K+1)}$$

**3. Coalition value and team value**

$$Q_k = \sum_i S_{ik} w_i Q_i \qquad\qquad Q_{tot} = b(s) + \sum_k g\big(Q_k; s, z_k, m_k\big)$$

- `w_i ≥ 0`: per-agent weight.
- `m_k = Σ_i S_ik`: coalition size, so `g` can tell a 1-agent group from a 3-agent group.
- `z_k`: average state features of the coalition's members.
- `g` is increasing in `Q_k` and **nonlinear**. If `g` were linear, this would collapse to plain VDN and the grouping would not matter.

**4. Loss**

$$\mathcal L = \mathcal L_{TD} + \beta_{cut}\mathcal L_{cut} + \beta_{cap}\mathcal L_{cap} + \beta_{sharp}\mathcal L_{sharp} + \beta_{temp}\mathcal L_{temp}$$

| Term | What it does |
|---|---|
| `L_TD` | standard Double-Q TD loss; trains agents, mixer, and (in the full method) `S` |
| `L_cut` | pulls strongly linked agents into the same coalition (computed with `W` detached) |
| `L_cap` | stops one coalition from absorbing everyone |
| `L_sharp` | makes memberships closer to hard assignments |
| `L_temp` | discourages assignments from flickering between timesteps |

---

## Why decentralized execution still works

`S`, `w_i`, `m_k`, `z_k` and `b(s)` never depend on the current actions or on any `Q_i`, and all weights multiplying `Q_k` are non-negative. So

$$\frac{\partial Q_{tot}}{\partial Q_i} \ge 0$$

and picking each agent's best action separately also maximizes `Q_tot` (the same guarantee QMIX gives).

---

## The key experiment

| Variant | TD gradient reaches `S`? |
|---|---|
| **DynaCoalition (full)** | Yes |
| **Detached (ablation)** | No. `S` is trained only by the structure losses; everything else is identical |

Other comparisons: QMIX, VDN, an oracle using true package groups, a simple position-clustering heuristic, and a random-`S` control.

---

## Training loop

1. Collect episodes with the GRU agents (ε-greedy) in vectorized VMAS transport.
2. Store them in the replay buffer.
3. Sample sequences and compute `h`, `Q_i`, then `W`, `S`, `Q_k`, `Q_tot`.
4. Build the target with the target networks (Double-Q, target detached).
5. Compute the total loss, backpropagate, clip gradients, update.
6. Periodically update target networks, anneal ε and the softmax temperature, and evaluate (return, packages delivered, grouping ARI, switch rate).

---

## Two details that are easy to miss

- **Empty coalitions:** each slot contributes `g(Q_k) − g(0)`, so unused slots add nothing to `Q_tot`.
- **Background slot:** agents not in any coalition still go through a separate monotone head, so they still get TD gradient and keep learning.
