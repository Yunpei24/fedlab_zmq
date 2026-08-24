# Decision-Margin Deficit package

This package separates DMD's mathematics from FL transport and orchestration.

The three objectives share the local quadratic decision deficit

\[
D_i(w;r)=\frac12\sum_c\rho_{i,c}[r_c-m_{i,c}(w)]_+^2.
\]

- **DMD-Mean:** penalizes the mean deficit.
- **DMD-USV:** adds a frozen-target upper-semivariance term
  \([D_i-\bar D]_+^2\).
- **DMD-Tail:** adds the Rockafellar--Uryasev upper-tail surrogate
  \(\eta+[D_i-\eta]_+/q\), with `eta` computed before the local step and
  detached during SGD.

Neither squaring a deficit nor optimizing a CVaR surrogate makes it a
statistical variance. The objective is fairness-aware because it explicitly
penalizes decision-space shortfalls and their inter-client upper tail.

See [`research/dmd/README.md`](../../research/dmd/README.md) for migration,
reproducibility, and current ZMQ limitations.

