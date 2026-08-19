# Stochastic schedule simulation

This package is the isolated stochastic stage between fix-and-optimize and
GNN training-data export.

For every fixed candidate graph it simulates machine-specific Weibull
operating lives and exponential repair durations.  Operations are interrupted
and resumed on the same machine, repairs restore a machine to age zero, the
failure clock advances only during productive processing, and delays propagate
through the unchanged job and immediate-machine-predecessor graph.

All candidates receive short pilot labels for pool selection.  Selected graphs
are simulated again with an independent label seed and the configured full
replication count before one CSV row per graph is written.
