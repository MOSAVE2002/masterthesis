# Stochastic schedule simulation

This package is the isolated stochastic stage between fix-and-optimize and
GNN training-data export.

For every operation it evaluates the same nonlinear machine-down probability
at the nominal operation midpoint as the reference MINLP. One Bernoulli draw
selects the disruption state. Conditional on a disruption, a remaining
exponential repair duration is added to the operation. The delay propagates
through the unchanged job and immediate-machine-predecessor graph by
right-shifting downstream operations.

This midpoint-snapshot process is the only simulation model. It contains no
idle-time failures, continuous virtual machine age, renewal process, or
multiple failures per operation. Therefore the simulated mean direct delay is
consistent with the unscaled nonlinear quantity `Pd(t_midpoint) / repair_rate`.

All candidates receive short pilot labels for pool selection.  Selected graphs
are simulated again with an independent label seed and the configured full
replication count before one CSV row per graph is written.
