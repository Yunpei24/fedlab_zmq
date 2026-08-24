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

## Native FedLab execution

The public registry names are `dmd_mean`, `dmd_usv`, and `dmd_tail`.
`run_experiment.py` supplies a deterministic client-local train/anchor split.
Each client evaluates the received global model on its anchor before local SGD,
uploads that report with its model update, and consumes the context constructed
by the server one round earlier.

The context is propagated through the framework-wide state hook:

```text
AggregateResult.metrics["_server_state_updates"]
    -> server_algo_state
    -> next round's algo_config
```

`source_round` is checked by the client, so an accidental current-round or
two-round-old reference fails loudly. The initial warm-up uses CE only.

For Toubkal instructions and the non-private confirmation matrix, see
[`docs/dmd_toubkal.md`](../../docs/dmd_toubkal.md).
