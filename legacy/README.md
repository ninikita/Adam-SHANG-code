# Retired exploratory scripts

These files are retained for provenance, not as current paper experiments:

- `reddi_convex_deterministic.py`
- `reddi_convex_stochastic.py`
- `reddi_stochastic_convex.py`

They explored fixed-objective reformulations of the online example, which were
removed from the manuscript. Names reflect historical files; in particular,
`reddi_convex_deterministic.py` also contains stochastic experimentation.

Algorithms and defaults are preserved. Some scripts execute long computations
at import time; do not import them from maintained experiments or tests. To use
a historical script deliberately, run it directly by filename from a separate
working/output directory.

For the maintained online benchmark use `../reddi_synthetic_stress_test.py`.
