# Stochastic schedule simulation

This package provides the independent stochastic post-solve evaluation of
fixed schedules. GNN training-data generation uses analytical expected repair
buffers and does not call this simulator.

There is no probabilistic service-level constraint in the optimization models.
The simulator does not modify or re-optimize a schedule and does not feed a
feasibility cut back to the solver. Its purpose is to measure how schedules
perform under different due-date, failure and repair parameters.

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

The evaluator uses fresh replications and a deterministic evaluation seed to
compare stored nonlinear and GNN schedules. Simulation results are diagnostic
and are not fed back into optimization or GNN training. In one replication, a
job is counted as on time when its simulated completion time is no greater than
its due date. The reported job on-time probability is the corresponding share
of successful replications. Schedule-level tables summarize these job values,
for example by their minimum. If a configured service level is used to classify
a result, that classification is only an evaluation metric, not an optimization
requirement.
