"""Stochastic constrained convex toy, median over 30 independent runs.

Keep this file beside reddi_convex_deterministic.py: both entry points use
the same objective, optimizer updates, baseline tuning, and plot style.

Minimize f(x)=x on [-1,1], using g=n^2 with probability 1/n and -n
otherwise. The default n=50 gives E[g]=1 and E[g^2]=127450. All primal
updates are projected onto [-1,1]. The SHANG variants retain the
theory-derived coupled parameters, with L=1, R=2 and P0=sqrt(127450),
for a constrained stability comparison. Adam/AMSGrad tune only their base
rate on independent pilot trajectories. See the shared module for details.

Examples:
  python reddi_convex_stochastic.py
  python reddi_convex_stochastic.py --steps 100000 --pilot-steps 30000
  python reddi_convex_stochastic.py --noise-level 50
  python reddi_convex_stochastic.py --steps 1000 --pilot-steps 300 --runs 3 --pilot-runs 3
  python reddi_convex_stochastic.py --replot-from RUN_DIRECTORY --output NEW_DIRECTORY

Plots: objective gaps and iterates, both pointwise MEDIANS, never the
objective evaluated at the median iterate. Per-run checkpoint data,
final-run values, independent grid searches, and all settings are saved.
"""

from reddi_convex_deterministic import main


if __name__ == "__main__":
    main()
