"""
Compare Gurobi MIP and Hybrid Algorithm (GA+TS) on a text-format FJSP instance.

Usage:
    uv run helper/compare_solvers.py [instance_file] [gurobi_timelimit] [ha_pop] [ha_gen]

Defaults: i20_j10_001.fjsp, 180s, pop=40, gen=30
"""
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import gurobipy as gp
from gurobipy import GRB
from Algorithm.HA import ProcessingTimes, hybrid_ga_ts, PARAMS


# ---------------------------------------------------------------------------
# Text-format FJSP parser
# ---------------------------------------------------------------------------

def parse_text_fjsp(path: Path) -> tuple:
    """
    Parse a text FJSP file (Brandimarte / instance_generator format).

    Format:
        n_jobs  n_machines  avg_options
        n_ops  n_opts  m t  m t ...  n_opts  m t ...   (one line per job)

    Machines are 0-indexed.
    Returns (n_jobs, n_machines, ProcessingTimes).
    """
    with open(path) as f:
        tokens = f.read().split()

    idx = 0
    n_jobs     = int(tokens[idx]);   idx += 1
    n_machines = int(tokens[idx]);   idx += 1
    idx += 1  # skip avg_options (float)

    pt: ProcessingTimes = []
    for _ in range(n_jobs):
        n_ops = int(tokens[idx]); idx += 1
        job_ops = []
        for _ in range(n_ops):
            n_opts = int(tokens[idx]); idx += 1
            alts = []
            for _ in range(n_opts):
                m = int(tokens[idx]); idx += 1
                t = int(tokens[idx]); idx += 1
                alts.append((m, t))
            job_ops.append(alts)
        pt.append(job_ops)

    return n_jobs, n_machines, pt


# ---------------------------------------------------------------------------
# Gurobi model builder from ProcessingTimes
# ---------------------------------------------------------------------------

def build_gurobi_from_pt(model: gp.Model, pt: ProcessingTimes):
    """Build the FJSP MIP model directly from a ProcessingTimes structure."""
    real_ops   = []
    eligible   = {}
    proc_times = {}
    preds      = {}
    job_last   = {}

    op_id = 0
    for j, job_ops in enumerate(pt):
        for o, alts in enumerate(job_ops):
            real_ops.append(op_id)
            eligible[op_id]  = [m for m, _ in alts]
            for m, t in alts:
                proc_times[op_id, m] = t
            preds[op_id] = [] if o == 0 else [op_id - 1]
            if o == len(job_ops) - 1:
                job_last[j] = op_id
            op_id += 1

    H = sum(max(proc_times[i, k] for k in eligible[i]) for i in real_ops)

    C   = model.addVars(real_ops, lb=0.0, vtype=GRB.CONTINUOUS, name="C")
    Cm  = model.addVar(lb=0.0,  vtype=GRB.CONTINUOUS, name="Cmax")
    Y   = model.addVars([(i, k) for i in real_ops for k in eligible[i]],
                        vtype=GRB.BINARY, name="Y")

    X_idx = []
    for a, i in enumerate(real_ops):
        for j in real_ops[a + 1:]:
            for k in set(eligible[i]) & set(eligible[j]):
                X_idx.append((i, j, k))
    X = model.addVars(X_idx, vtype=GRB.BINARY, name="X")

    model.setObjective(Cm, GRB.MINIMIZE)

    for i in real_ops:
        model.addConstr(gp.quicksum(Y[i, k] for k in eligible[i]) == 1)

    for i in real_ops:
        for p in preds[i]:
            pt_i = gp.quicksum(Y[i, k] * proc_times[i, k] for k in eligible[i])
            model.addConstr(C[i] >= C[p] + pt_i)

    for i, j, k in X_idx:
        model.addConstr(C[i] >= C[j] + proc_times[i, k]
                        - H * (2 + X[i, j, k] - Y[i, k] - Y[j, k]))
        model.addConstr(C[j] >= C[i] + proc_times[j, k]
                        - H * (3 - X[i, j, k] - Y[i, k] - Y[j, k]))

    for i in real_ops:
        pt_i = gp.quicksum(Y[i, k] * proc_times[i, k] for k in eligible[i])
        model.addConstr(C[i] >= pt_i)

    for last in job_last.values():
        model.addConstr(Cm >= C[last])

    model.update()
    return model, Cm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]
    instance_file   = args[0] if len(args) > 0 else "i20_j10_001.fjsp"
    gurobi_timelimit= int(args[1]) if len(args) > 1 else 180
    ha_pop          = int(args[2]) if len(args) > 2 else 40
    ha_gen          = int(args[3]) if len(args) > 3 else 30

    path = ROOT_DIR / "data" / "instances_text" / instance_file
    n_jobs, n_machines, pt = parse_text_fjsp(path)
    n_ops = sum(len(ops) for ops in pt)

    print(f"Instance : {instance_file}")
    print(f"Jobs     : {n_jobs}")
    print(f"Machines : {n_machines}")
    print(f"Ops total: {n_ops}")
    print("=" * 50)

    # ---- Hybrid Algorithm ----
    print(f"\n[1] Hybrid Algorithm  (pop={ha_pop}, gen={ha_gen})")
    t0   = time.time()
    best = hybrid_ga_ts(
        pt,
        pop_size     = ha_pop,
        max_gen      = ha_gen,
        max_stagnant = 10,
        ts_iter_base = 30,
    )
    ha_time = time.time() - t0
    print(f"\n    HA Makespan : {best.fitness}")
    print(f"    HA Time     : {ha_time:.1f} s")

    # ---- Gurobi ----
    print(f"\n[2] Gurobi MIP  (TimeLimit={gurobi_timelimit}s)")
    model = gp.Model("FJSP")
    model.Params.TimeLimit   = gurobi_timelimit
    model.Params.OutputFlag  = 1

    t0 = time.time()
    model, Cm = build_gurobi_from_pt(model, pt)
    model.optimize()
    gurobi_time = time.time() - t0

    print(f"\n    Gurobi Status : {model.Status}")
    if model.SolCount > 0:
        gurobi_makespan = int(round(Cm.X))
        gurobi_gap      = model.MIPGap * 100
        print(f"    Gurobi Makespan : {gurobi_makespan}")
        print(f"    Gurobi Gap      : {gurobi_gap:.1f}%")
    else:
        gurobi_makespan = None
        print(f"    Gurobi: no feasible solution found within {gurobi_timelimit}s")
    print(f"    Gurobi Time : {gurobi_time:.1f} s")

    # ---- Summary ----
    print("\n" + "=" * 50)
    print("SUMMARY")
    print(f"  HA     : makespan = {best.fitness}  ({ha_time:.1f}s)")
    if gurobi_makespan is not None:
        print(f"  Gurobi : makespan = {gurobi_makespan}  (gap {model.MIPGap*100:.1f}%, {gurobi_time:.1f}s)")
    else:
        print(f"  Gurobi : no solution  ({gurobi_time:.1f}s)")
