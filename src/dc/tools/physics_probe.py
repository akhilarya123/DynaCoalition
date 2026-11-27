"""Physics probe: how fast does one package move with k agents pushing it?

Run (from the repo root, venv active)::

    python -m dc.tools.physics_probe                       # default sweep
    python -m dc.tools.physics_probe --masses 10 20 35 --steps 80
    python -m dc.tools.physics_probe --out experiments/physics_probe.csv

Why this exists (EXPERIMENT_PLAN.md Section 1, "Physics tuning (critical)"):
if a package barely moves with fewer than 3 agents, progress shaping gives no
early signal and nothing learns; if one agent can deliver alone, coalitions
are pointless. VMAS has only linear drag, so steady speed should grow roughly
in proportion to the number of pushers, which means the knob that matters is
the *absolute* speed against the 200-step episode, not a ratio. So for every
(mass, k) we measure steady-state speed and convert it to the estimated steps
needed to carry a package ``--travel`` units (default 1.0, about the typical
package-to-goal distance in the [-1, 1]^2 arena).

Target window (a proposal, tune it with Abhishek at Sync A):
    3 pushers deliver in at most ``--fast`` steps (default 100), and
    1 pusher needs at least ``--slow`` steps (default 150).
Mass and ``agent_u_multiplier`` only enter through force/mass, so sweeping
mass is enough; pick whichever knob you prefer for the final config.

Setup per case: package 0 at (-0.5, 0) facing +x; ``k`` agents lined up along
its left face, all pressing "right"; every other agent parks and stays; the
goal and other packages are parked far away, so nothing is delivered.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

from dc.contracts import EnvSpec
from dc.envs.transport import TransportEnv, state_dim_for

RIGHT = 4  # action index for +x (see dc.envs.transport.ACTION_NAMES)


def _set(entity: Any, torch: Any, xy: tuple[float, float], e: int) -> None:
    """Teleport an entity, with zero velocity, rotation and spin, in every env."""
    entity.set_pos(torch.tensor(xy, dtype=torch.float32).expand(e, 2).clone(), batch_index=None)
    zeros2 = torch.zeros(e, 2)
    zeros1 = torch.zeros(e, 1)
    entity.set_vel(zeros2, batch_index=None)
    entity.set_rot(zeros1, batch_index=None)
    entity.set_ang_vel(zeros1, batch_index=None)


def run_case(
    env: TransportEnv,
    n_push: int,
    steps: int,
) -> dict[str, float]:
    """Push package 0 with ``n_push`` agents for ``steps`` steps; return its motion."""
    import torch

    e, n = env.num_envs, env.spec.n_agents
    if n_push > n:
        raise ValueError(f"n_push={n_push} exceeds n_agents={n}")
    env.reset()
    sc = env.scenario
    r, hx = sc.agent_radius, sc.half_extents[0]

    pkg_xy = (-0.5, 0.0)
    _set(sc.packages[0], torch, pkg_xy, e)
    for j, p in enumerate(sc.packages[1:]):
        _set(p, torch, (-0.9 + 0.25 * j, -0.9), e)
    _set(sc.goal, torch, (0.9, 0.5), e)

    spacing = 2.0 * r * 1.02  # just not overlapping
    per_rank = 3  # a 0.15-wide face is touched by at most 3 agents; extras queue behind
    for i, agent in enumerate(sc.world.agents):
        if i < n_push:
            rank, col = divmod(i, per_rank)
            in_rank = min(per_rank, n_push - rank * per_rank)
            dy = (col - (in_rank - 1) / 2.0) * spacing
            dx = rank * spacing
            _set(agent, torch, (pkg_xy[0] - hx - r - 0.005 - dx, pkg_xy[1] + dy), e)
        else:
            j = i - n_push
            _set(agent, torch, (-0.55 + 0.1 * (j % 12), 0.9 - 0.1 * (j // 12)), e)

    act = torch.zeros(e, n, dtype=torch.long)
    act[:, :n_push] = RIGHT
    xs = [float(sc.pkg_pos()[:, 0, 0].mean())]
    for _ in range(steps):
        env.step(act)
        xs.append(float(sc.pkg_pos()[:, 0, 0].mean()))

    half = steps // 2
    speed = (xs[-1] - xs[half]) / max(steps - half, 1)  # steady-ish speed, units/step
    return {
        "disp": xs[-1] - xs[0],
        "speed": speed,
        "dy": float(sc.pkg_pos()[:, 0, 1].mean()) - pkg_xy[1],
        "rot": float(sc.pkg_rot()[:, 0].mean()),
    }


def steps_to_travel(speed: float, travel: float) -> float:
    return travel / speed if speed > 1e-9 else math.inf


def sweep(
    masses: list[float],
    pushers: list[int],
    *,
    steps: int,
    travel: float,
    n_agents: int,
    n_packages: int,
    u_mult: float,
    frictions: list[float | None] | None = None,
    num_envs: int = 2,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    spec = EnvSpec(
        n_agents=n_agents,
        n_packages=n_packages,
        obs_dim=16,
        state_dim=state_dim_for(n_agents, n_packages),
        x_dim=7,
        episode_limit=steps,
    )
    for mass, fric in [(m, f) for m in masses for f in (frictions or [None])]:
        extra = {} if fric is None else {"package_friction": fric}
        env = TransportEnv(
            spec, num_envs=num_envs, device="cpu", seed=0,
            package_mass=mass, agent_u_multiplier=u_mult, **extra,
        )
        for k in pushers:
            out = run_case(env, k, steps)
            rows.append(
                {
                    "mass": mass,
                    "friction": -1.0 if fric is None else fric,
                    "pushers": k,
                    "speed": out["speed"],
                    "disp": out["disp"],
                    "drift_y": out["dy"],
                    "rot": out["rot"],
                    "steps_to_travel": steps_to_travel(out["speed"], travel),
                }
            )
    return rows


def _fmt(x: float) -> str:
    return "never" if math.isinf(x) else f"{x:7.0f}"


def report(rows: list[dict[str, float]], *, fast: float, slow: float, travel: float) -> None:
    pushers = sorted({int(r["pushers"]) for r in rows})
    print(f"\nEstimated steps to carry a package {travel:g} units (speed measured over the last half):")
    head = "  mass | fric | " + " | ".join(f"k={k}".rjust(8) for k in pushers) + " | verdict"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for mass, fric in sorted({(r["mass"], r["friction"]) for r in rows}):
        by_k = {int(r["pushers"]): r for r in rows if r["mass"] == mass and r["friction"] == fric}
        cells = " | ".join(_fmt(by_k[k]["steps_to_travel"]).rjust(8) for k in pushers)
        verdict = ""
        if 1 in by_k and 3 in by_k:
            t1, t3 = by_k[1]["steps_to_travel"], by_k[3]["steps_to_travel"]
            if t3 <= fast and t1 >= slow:
                verdict = "IN WINDOW"
            elif t3 > fast:
                verdict = "too heavy: even 3 are slow"
            else:
                verdict = "too light: 1 agent can deliver"
        fric_s = "off" if fric < 0 else f"{fric:g}"
        print(f"  {mass:5g} | {fric_s:>4} | {cells} | {verdict}")
    print(f"\nWindow: 3 pushers <= {fast:g} steps and 1 pusher >= {slow:g} steps.")
    worst = max(abs(r["drift_y"]) for r in rows)
    print(f"Largest sideways drift of a pushed package: {worst:.3f} (large => uneven pushes, check layout).")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masses", type=float, nargs="+", default=[5, 10, 20, 35, 50])
    ap.add_argument("--pushers", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="k>3 queue behind the front rank and push through the agents in front")
    ap.add_argument("--frictions", type=float, nargs="+", default=None,
                    help="package_friction values to sweep (default: friction off)")
    ap.add_argument("--steps", type=int, default=60, help="steps per case; speed uses the last half")
    ap.add_argument("--travel", type=float, default=1.0)
    ap.add_argument("--fast", type=float, default=100.0)
    ap.add_argument("--slow", type=float, default=150.0)
    ap.add_argument("--n-agents", type=int, default=12)
    ap.add_argument("--n-packages", type=int, default=4)
    ap.add_argument("--u-mult", type=float, default=0.6)
    ap.add_argument("--out", type=Path, default=None, help="optional CSV path")
    args = ap.parse_args(argv)

    rows = sweep(
        args.masses, args.pushers, steps=args.steps, travel=args.travel,
        n_agents=args.n_agents, n_packages=args.n_packages, u_mult=args.u_mult,
        frictions=args.frictions,
    )
    report(rows, fast=args.fast, slow=args.slow, travel=args.travel)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
