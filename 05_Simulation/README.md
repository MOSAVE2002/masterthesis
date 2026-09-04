# Stochastic schedule simulation

This package provides the stochastic evaluation of fixed schedules. The same
simulator has two deliberately separated uses: training-data generation uses
it to label the propagated expected completion delay of each job, while the
post-solve evaluation uses fresh replications and a separate evaluation seed.

The nonlinear and GNN optimization models contain a soft conservative service
constraint based on Markov's inequality. The simulator itself does not modify
or re-optimize a schedule and does not feed a feasibility cut back to the
solver.

In every replication the simulator draws one Weibull first-failure time and
one exponential repair duration for each used machine. This defines one common
downtime interval per machine. An operation waits when its effective start is
inside that interval. If the failure occurs during processing, the operation
is interrupted and resumes after the repair without losing completed work.
The resulting delay propagates through the unchanged job and
immediate-machine-predecessor graph by right-shifting downstream operations.

This is a single-failure machine-history model: failures can occur during idle
time, but a machine has no second failure after its first repair. It is
deliberately richer than the operation-wise midpoint approximation in the
reference MINLP. Consequently, simulated delays are an independent stochastic
training and evaluation target and are not expected to equal the analytic
quantity `Pd(t_midpoint) / repair_rate` operation by operation.

For a training schedule, the label of job j is the sample mean of
`simulated_completion_j - nominal_completion_j`, including right-shifted delay
from upstream job and machine operations. The label generator also stores the
standard error, replication count and empirical on-time probability. The
configured pilot uses 256 replications; this must be increased for final
training runs. Candidate schedules of the same instance use common random
numbers. Random streams are tied to stable machine identifiers rather than
topological operation order, and a stable instance-specific seed separates
the streams of different instances.

The evaluator uses fresh replications and a deterministic evaluation seed to
compare stored nonlinear and GNN schedules. In one replication, a job is
counted as on time when its simulated completion time is no greater than its
due date. The reported job on-time probability is the corresponding share of
successful replications.
