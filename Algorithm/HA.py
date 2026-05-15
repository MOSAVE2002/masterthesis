"""
Hybrid Algorithm (HA) for the Flexible Job Shop Scheduling Problem (FJSP).

Based on:
    Li, X., & Gao, L. (2016). An effective hybrid genetic algorithm and tabu
    search for flexible job shop scheduling problem.
    Int. J. Production Economics, 174, 93-110.

Components:
    - GA (global search / exploration) with:
        * Two OS crossover operators: POX and JBX (selected 50/50)
        * Two-point crossover for MS string
        * Two OS mutation operators: swapping and neighborhood (selected 50/50)
        * MS mutation: reassign half the positions to a different machine
        * Elitist + tournament selection
    - Tabu Search (local search / exploitation) with:
        * Critical-path-based neighborhood (Mastrolilli & Gambardella, 2000)
        * N1: swap adjacent critical ops on the same machine (OS moves)
        * N2: try alternative machines for critical ops (MS moves)
        * Aspiration criterion: accept tabu move if it beats the global best
    - Active schedule decoder
    - Adaptive TS iteration count: maxTSIter = ts_iter_base * (gen / max_gen)
"""

import bisect
from itertools import permutations
import random
import time
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import pickle
import sys

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

# pt[job][op] = [(machine_id, duration), ...]
ProcessingTimes = List[List[List[Tuple[int, int]]]]

# ---------------------------------------------------------------------------
# Default parameters  (Table 2 of the paper)
# ---------------------------------------------------------------------------

PARAMS: Dict = {
    # Paper values (Li & Gao 2016, Table 2) are pop=400, gen=200, ts_iter_base=800,
    # designed for a compiled C/Java implementation — 100-1000x faster than Python.
    # These Python defaults balance quality and runtime (~1-2 min on 10-job instances).
    "pop_size":     400,
    "max_gen":      200,
    "max_stagnant": 100,
    "pr":           0.005,   # elitist reproduction probability
    "pc":           0.8,    # crossover probability
    "pm":           0.1,    # mutation probability
    "ts_iter_base": 800,     # maxTSIterSize = ts_iter_base * (gen / max_gen)
    "tabu_len":     9,
}

# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------

class Individual:
    """Chromosome for FJSP: one OS string and one MS string."""

    def __init__(
        self,
        os: List[int],
        ms: List[int],
        fitness: Optional[int] = None,
    ) -> None:
        self.os = os        # Operation Sequencing string
        self.ms = ms        # Machine Selection string
        self.fitness = fitness

    def copy(self) -> "Individual":
        return Individual(self.os[:], self.ms[:], self.fitness)

    def __repr__(self) -> str:
        return f"Individual(fitness={self.fitness})"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def ms_index(job: int, op: int, pt: ProcessingTimes) -> int:
    """Position of operation (job, op) in the flat MS string."""
    return sum(len(pt[j]) for j in range(job)) + op


def build_ms_offsets(pt: ProcessingTimes) -> List[int]:
    """Prefix-sum array so ms_index(j, o) = ms_offsets[j] + o in O(1)."""
    offsets = [0] * (len(pt) + 1)
    for j, job_ops in enumerate(pt):
        offsets[j + 1] = offsets[j] + len(job_ops)
    return offsets


def flatten_operations(pt: ProcessingTimes) -> List[Tuple[int, int]]:
    """All (job, op) pairs in the same order as the MS string."""
    return [
        (j, o)
        for j, job_ops in enumerate(pt)
        for o in range(len(job_ops))
    ]


def random_individual(pt: ProcessingTimes) -> Individual:
    """Create a random feasible individual."""
    os_str: List[int] = []
    for j, job_ops in enumerate(pt):
        os_str.extend([j] * len(job_ops))
    random.shuffle(os_str)

    ms_str: List[int] = [
        random.randrange(len(pt[j][o]))
        for j, job_ops in enumerate(pt)
        for o in range(len(job_ops))
    ]
    return Individual(os=os_str, ms=ms_str)


# ---------------------------------------------------------------------------
# Active Schedule Decoder
# ---------------------------------------------------------------------------

def _earliest_slot(asij: int, duration: int, intervals: List[Tuple[int, int]]) -> int:
    """
    Return the earliest start time >= asij where an operation of `duration`
    fits into an idle gap in the machine's sorted interval list.
    This produces an active schedule (operations are never delayed unnecessarily).
    """
    if not intervals:
        return asij

    # Gap before first scheduled interval
    if asij + duration <= intervals[0][0]:
        return asij

    # Gaps between consecutive intervals
    for i in range(len(intervals) - 1):
        gap_start = intervals[i][1]
        gap_end   = intervals[i + 1][0]
        candidate = max(asij, gap_start)
        if candidate + duration <= gap_end:
            return candidate

    # After the last interval
    return max(asij, intervals[-1][1])


def decode(ind: Individual, pt: ProcessingTimes) -> Tuple[int, List[Dict]]:
    """
    Decode an Individual into an active schedule.

    Returns:
        (makespan, schedule)  where each schedule entry is a dict with keys:
        job, op, machine, start, end, duration.
    """
    ms_off    = build_ms_offsets(pt)
    job_count = [0] * len(pt)
    job_ready = [0] * len(pt)
    machine_intervals: Dict[int, List[Tuple[int, int]]] = {}
    schedule: List[Dict] = []

    for job in ind.os:
        op = job_count[job]
        if op >= len(pt[job]):
            continue

        idx     = ms_off[job] + op
        m_alt   = ind.ms[idx]
        machine, duration = pt[job][op][m_alt]

        asij      = job_ready[job]
        intervals = machine_intervals.setdefault(machine, [])
        start     = _earliest_slot(asij, duration, intervals)
        end       = start + duration

        bisect.insort(intervals, (start, end))

        schedule.append({
            "job": job, "op": op, "machine": machine,
            "start": start, "end": end, "duration": duration,
        })

        job_ready[job] = end
        job_count[job] += 1

    makespan = max(e["end"] for e in schedule) if schedule else 0
    return makespan, schedule


def _decode_makespan(ind: Individual, pt: ProcessingTimes, ms_off: List[int]) -> int:
    """Fast makespan-only decode; skips building the schedule list."""
    n_jobs    = len(pt)
    nops      = [len(job_ops) for job_ops in pt]
    job_count = [0] * n_jobs
    job_ready = [0] * n_jobs
    mach_ivals: Dict[int, List[Tuple[int, int]]] = {}
    makespan  = 0
    ms        = ind.ms

    for job in ind.os:
        op = job_count[job]
        if op >= nops[job]:
            continue
        machine, duration = pt[job][op][ms[ms_off[job] + op]]
        asij = job_ready[job]

        # Inline _earliest_slot to eliminate per-op function-call overhead
        if machine in mach_ivals:
            intervals = mach_ivals[machine]
            if asij + duration <= intervals[0][0]:
                start = asij
            else:
                start = None
                n = len(intervals)
                for i in range(n - 1):
                    gs = intervals[i][1]
                    ge = intervals[i + 1][0]
                    c  = gs if gs > asij else asij
                    if c + duration <= ge:
                        start = c
                        break
                if start is None:
                    last = intervals[-1][1]
                    start = last if last > asij else asij
        else:
            mach_ivals[machine] = []
            intervals = mach_ivals[machine]
            start = asij

        end = start + duration
        bisect.insort(intervals, (start, end))
        job_ready[job] = end
        job_count[job] += 1
        if end > makespan:
            makespan = end

    return makespan


def evaluate(ind: Individual, pt: ProcessingTimes, ms_off: Optional[List[int]] = None) -> int:
    """Evaluate an individual and store its makespan as fitness."""
    if ms_off is None:
        ms_off = build_ms_offsets(pt)
    ind.fitness = _decode_makespan(ind, pt, ms_off)
    return ind.fitness


# ---------------------------------------------------------------------------
# Genetic Operators — Selection
# ---------------------------------------------------------------------------

def elitist_selection(population: List[Individual], pr: float) -> List[Individual]:
    """Copy the top pr * |pop| individuals into the next generation."""
    n = max(1, int(pr * len(population)))
    return [ind.copy() for ind in sorted(population, key=lambda x: x.fitness)[:n]]


def tournament_selection(population: List[Individual], k: int = 2) -> Individual:
    """Return the best individual from k randomly sampled candidates."""
    return min(random.sample(population, k), key=lambda x: x.fitness)


# ---------------------------------------------------------------------------
# Genetic Operators — Crossover
# ---------------------------------------------------------------------------

def _os_from_sets(
    p1: List[int], p2: List[int],
    js1: Set[int], js2: Set[int],
) -> Tuple[List[int], List[int]]:
    """
    Shared logic for POX and JBX:
      O1 keeps js1-elements from P1, fills gaps from P2.
      O2 keeps js2-elements from P2, fills gaps from P1.
    """
    def make_child(a: List[int], b: List[int], js: Set[int]) -> List[int]:
        child: List[Optional[int]] = [None] * len(a)
        for i, gene in enumerate(a):
            if gene in js:
                child[i] = gene
        remaining = [gene for gene in b if gene not in js]
        r = 0
        for i in range(len(child)):
            if child[i] is None:
                child[i] = remaining[r]
                r += 1
        return child  # type: ignore[return-value]

    return make_child(p1, p2, js1), make_child(p2, p1, js2)


def pox_crossover(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    """Precedence Operation Crossover (POX): both offspring use the same Jobset."""
    jobs = list(set(p1))
    random.shuffle(jobs)
    split = random.randint(1, max(1, len(jobs) - 1))
    js1 = set(jobs[:split])
    return _os_from_sets(p1, p2, js1, js1)


def jbx_crossover(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    """Job-Based Crossover (JBX): each offspring uses a different Jobset."""
    jobs = list(set(p1))
    random.shuffle(jobs)
    split = random.randint(1, max(1, len(jobs) - 1))
    js1 = set(jobs[:split])
    js2 = set(jobs[split:])
    return _os_from_sets(p1, p2, js1, js2)


def crossover_os(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    """Select POX or JBX randomly with equal probability (paper: 50/50)."""
    return pox_crossover(p1, p2) if random.random() < 0.5 else jbx_crossover(p1, p2)


def two_point_crossover_ms(
    ms1: List[int], ms2: List[int]
) -> Tuple[List[int], List[int]]:
    """Two-point crossover for the MS string."""
    if len(ms1) < 2:
        return ms1[:], ms2[:]
    a, b = sorted(random.sample(range(len(ms1)), 2))
    return (ms1[:a] + ms2[a:b] + ms1[b:],
            ms2[:a] + ms1[a:b] + ms2[b:])


# ---------------------------------------------------------------------------
# Genetic Operators — Mutation
# ---------------------------------------------------------------------------

def mutate_os_swap(os: List[int]) -> List[int]:
    """Swapping mutation: swap two randomly selected positions."""
    child = os[:]
    i, j = random.sample(range(len(child)), 2)
    child[i], child[j] = child[j], child[i]
    return child


def mutate_os_neighborhood(os: List[int]) -> List[int]:
    """
    Neighborhood mutation: pick 3 positions with distinct job-values,
    generate all permutations of those values, return one at random.
    Falls back to swapping mutation when fewer than 3 distinct jobs exist.
    """
    child = os[:]
    job_positions: Dict[int, List[int]] = {}
    for i, job in enumerate(child):
        job_positions.setdefault(job, []).append(i)

    distinct_jobs = list(job_positions.keys())
    if len(distinct_jobs) < 3:
        return mutate_os_swap(os)

    sel_jobs = random.sample(distinct_jobs, 3)
    positions = [random.choice(job_positions[j]) for j in sel_jobs]
    values    = [child[p] for p in positions]

    alts = [p for p in permutations(values) if list(p) != values]
    if not alts:
        return child

    chosen = random.choice(alts)
    for pos, val in zip(positions, chosen):
        child[pos] = val
    return child


def mutate_os(os: List[int]) -> List[int]:
    """Select swapping or neighborhood mutation with equal probability."""
    return mutate_os_swap(os) if random.random() < 0.5 else mutate_os_neighborhood(os)


def mutate_ms(ms: List[int], pt: ProcessingTimes, flat_ops: Optional[List[Tuple[int, int]]] = None) -> List[int]:
    """
    MS mutation: select r = |ms|/2 positions, reassign each to a
    different available machine for that operation.
    """
    child = ms[:]
    ops   = flat_ops if flat_ops is not None else flatten_operations(pt)
    r     = max(1, len(child) // 2)
    for pos in random.sample(range(len(child)), r):
        job, op = ops[pos]
        alts = pt[job][op]
        if len(alts) > 1:
            current  = child[pos]
            child[pos] = random.choice([i for i in range(len(alts)) if i != current])
    return child


# ---------------------------------------------------------------------------
# Tabu Search — utilities
# ---------------------------------------------------------------------------

def _find_critical_ops(
    schedule: List[Dict], makespan: int
) -> Set[Tuple[int, int]]:
    """
    Backward traversal to identify all (job, op) pairs on the critical path.
    Uses O(n) lookup via a (machine, end_time) dict.
    """
    op_info: Dict[Tuple[int, int], Dict] = {
        (e["job"], e["op"]): e for e in schedule
    }
    # Machine predecessor lookup: (machine, end_time) -> (job, op)
    mach_end: Dict[Tuple[int, int], Tuple[int, int]] = {
        (e["machine"], e["end"]): (e["job"], e["op"]) for e in schedule
    }

    critical: Set[Tuple[int, int]] = set()
    queue: List[Tuple[int, int]] = [
        (e["job"], e["op"]) for e in schedule if e["end"] == makespan
    ]
    critical.update(queue)

    while queue:
        new_q: List[Tuple[int, int]] = []
        for j, o in queue:
            entry = op_info[(j, o)]
            start   = entry["start"]
            machine = entry["machine"]

            # Job predecessor: same job, previous operation
            if o > 0:
                pred = op_info.get((j, o - 1))
                if pred and pred["end"] == start and (j, o - 1) not in critical:
                    critical.add((j, o - 1))
                    new_q.append((j, o - 1))

            # Machine predecessor: another op ending exactly when this one starts
            key = (machine, start)
            if key in mach_end and mach_end[key] not in critical:
                critical.add(mach_end[key])
                new_q.append(mach_end[key])

        queue = new_q

    return critical


def _op_pos_in_os(os: List[int], job: int, op_idx: int) -> Optional[int]:
    """Return the position of the op_idx-th (0-based) occurrence of job in os."""
    count = 0
    for i, x in enumerate(os):
        if x == job:
            if count == op_idx:
                return i
            count += 1
    return None


def _machine_alt_index(pt: ProcessingTimes, job: int, op: int, machine: int) -> Optional[int]:
    for i, (m, _dur) in enumerate(pt[job][op]):
        if m == machine:
            return i
    return None


def _build_sched_info(schedule: List[Dict], pt: ProcessingTimes) -> Dict:
    ms_off = build_ms_offsets(pt)
    n_ops = sum(len(job_ops) for job_ops in pt)
    n_machines = max(e["machine"] for e in schedule) + 1 if schedule else 0

    info = {
        "start": [0] * n_ops,
        "dur": [0] * n_ops,
        "tail": [0] * n_ops,
        "machine_of": [-1] * n_ops,
        "sched_pos": [-1] * n_ops,
        "job_pred": [-1] * n_ops,
        "job_succ": [-1] * n_ops,
        "mach_pred": [-1] * n_ops,
        "mach_succ": [-1] * n_ops,
        "order": [],
        "machine_seq": [[] for _ in range(n_machines)],
    }

    for pos, e in enumerate(schedule):
        idx = ms_off[e["job"]] + e["op"]
        info["start"][idx] = e["start"]
        info["dur"][idx] = e["duration"]
        info["machine_of"][idx] = e["machine"]
        info["sched_pos"][idx] = pos
        info["order"].append(idx)
        info["machine_seq"][e["machine"]].append(idx)
        if e["op"] > 0:
            pred = ms_off[e["job"]] + e["op"] - 1
            info["job_pred"][idx] = pred
            info["job_succ"][pred] = idx

    for seq in info["machine_seq"]:
        seq.sort(key=lambda idx: (info["start"][idx], idx))
        for i, idx in enumerate(seq):
            if i > 0:
                info["mach_pred"][idx] = seq[i - 1]
            if i + 1 < len(seq):
                info["mach_succ"][idx] = seq[i + 1]

    info["order"].sort(key=lambda idx: (info["start"][idx], idx))
    for idx in reversed(info["order"]):
        best = 0
        js = info["job_succ"][idx]
        ms = info["mach_succ"][idx]
        if js >= 0:
            best = max(best, info["dur"][js] + info["tail"][js])
        if ms >= 0:
            best = max(best, info["dur"][ms] + info["tail"][ms])
        info["tail"][idx] = best

    return info


def _preferred_critical_path(
    schedule: List[Dict],
    pt: ProcessingTimes,
    makespan: int,
    critical: Set[Tuple[int, int]],
    info: Dict,
) -> List[Tuple[int, int]]:
    ms_off = build_ms_offsets(pt)
    flat_ops = flatten_operations(pt)

    end_idx = None
    for idx in info["order"]:
        if info["start"][idx] + info["dur"][idx] + info["tail"][idx] == makespan:
            end_idx = idx
    if end_idx is None:
        return []

    cur = end_idx
    while True:
        jp = info["job_pred"][cur]
        mp = info["mach_pred"][cur]
        jp_crit = jp >= 0 and flat_ops[jp] in critical and info["start"][jp] + info["dur"][jp] == info["start"][cur]
        mp_crit = mp >= 0 and flat_ops[mp] in critical and info["start"][mp] + info["dur"][mp] == info["start"][cur]
        if jp_crit:
            cur = jp
        elif mp_crit:
            cur = mp
        else:
            break

    path: List[Tuple[int, int]] = []
    while True:
        path.append(flat_ops[cur])
        js = info["job_succ"][cur]
        ms = info["mach_succ"][cur]
        js_crit = js >= 0 and flat_ops[js] in critical and info["start"][cur] + info["dur"][cur] == info["start"][js]
        ms_crit = ms >= 0 and flat_ops[ms] in critical and info["start"][cur] + info["dur"][cur] == info["start"][ms]
        if js_crit:
            cur = js
        elif ms_crit:
            cur = ms
        else:
            break
    return path


def _compute_removed_times(pt: ProcessingTimes, info: Dict, v_idx: int) -> Tuple[List[int], List[int]]:
    n_ops = sum(len(job_ops) for job_ops in pt)
    s_minus = [0] * n_ops
    t_minus = [0] * n_ops
    mp = info["mach_pred"][v_idx]
    ms = info["mach_succ"][v_idx]

    for idx in info["order"]:
        if idx == v_idx:
            continue
        finish = s_minus[idx] + info["dur"][idx]
        js = info["job_succ"][idx]
        if js >= 0:
            s_minus[js] = max(s_minus[js], finish)
        msucc = info["mach_succ"][idx]
        if idx == mp:
            msucc = ms
        if msucc == v_idx:
            msucc = -1
        if msucc >= 0:
            s_minus[msucc] = max(s_minus[msucc], finish)

    for idx in reversed(info["order"]):
        if idx == v_idx:
            continue
        best = 0
        js = info["job_succ"][idx]
        if js >= 0:
            best = max(best, info["dur"][js] + t_minus[js])
        msucc = info["mach_succ"][idx]
        if idx == mp:
            msucc = ms
        if msucc == v_idx:
            msucc = -1
        if msucc >= 0:
            best = max(best, info["dur"][msucc] + t_minus[msucc])
        t_minus[idx] = best
    return s_minus, t_minus


def _lpath_score_same_machine(info: Dict, v_idx: int, insert_pos: int) -> float:
    base_seq = info["machine_seq"][info["machine_of"][v_idx]]
    current_pos = base_seq.index(v_idx)
    seq = [idx for idx in base_seq if idx != v_idx]
    new_seq = seq[:]
    new_seq.insert(insert_pos, v_idx)

    start = min(current_pos, insert_pos)
    finish = max(current_pos, insert_pos)
    Q = new_seq[start:finish + 1]
    if not Q:
        return float("inf")

    def job_ready(idx: int) -> int:
        jp = info["job_pred"][idx]
        return info["start"][jp] + info["dur"][jp] if jp >= 0 else 0

    def job_tail(idx: int) -> int:
        js = info["job_succ"][idx]
        return info["dur"][js] + info["tail"][js] if js >= 0 else 0

    rprime = [0] * len(Q)
    pm_out = new_seq[start - 1] if start > 0 else -1
    rprime[0] = max(job_ready(Q[0]), info["start"][pm_out] + info["dur"][pm_out] if pm_out >= 0 else 0)
    for i in range(1, len(Q)):
        rprime[i] = max(job_ready(Q[i]), rprime[i - 1] + info["dur"][Q[i - 1]])

    tprime = [0] * len(Q)
    sm_out = new_seq[finish + 1] if finish + 1 < len(new_seq) else -1
    tprime[-1] = max(job_tail(Q[-1]), info["dur"][sm_out] + info["tail"][sm_out] if sm_out >= 0 else 0)
    for i in range(len(Q) - 2, -1, -1):
        tprime[i] = max(job_tail(Q[i]), tprime[i + 1] + info["dur"][Q[i + 1]])

    return max(rprime[i] + info["dur"][Q[i]] + tprime[i] for i in range(len(Q)))


def _build_paper_move_for_machine(
    pt: ProcessingTimes,
    job: int,
    op: int,
    machine: int,
    info: Dict,
) -> Optional[Dict]:
    ms_off = build_ms_offsets(pt)
    idx = ms_off[job] + op
    old_machine = info["machine_of"][idx]
    new_alt = _machine_alt_index(pt, job, op, machine)
    if new_alt is None:
        return None

    s_minus, t_minus = _compute_removed_times(pt, info, idx)
    sv_minus = s_minus[ms_off[job] + op - 1] + info["dur"][ms_off[job] + op - 1] if op > 0 else 0
    tv_minus = info["dur"][ms_off[job] + op + 1] + t_minus[ms_off[job] + op + 1] if op + 1 < len(pt[job]) else 0

    base_seq = info["machine_seq"][machine]
    seq = [x for x in base_seq if x != idx]
    left, right = 0, len(seq)
    common: List[int] = []
    for i, x in enumerate(seq):
        in_r = info["start"][x] + info["dur"][x] > sv_minus
        in_l = info["dur"][x] + info["tail"][x] > tv_minus
        if in_l and not in_r:
            left = max(left, i + 1)
        if in_r and not in_l:
            right = min(right, i)
        if in_l and in_r:
            common.append(x)
    if left > right:
        return None

    current_pos = info["machine_seq"][old_machine].index(idx) if machine == old_machine else -1
    best_pos = None
    best_score = float("inf")

    def consider(pos: int, score: float) -> None:
        nonlocal best_pos, best_score
        if pos < left or pos > right:
            return
        if machine == old_machine and pos == current_pos:
            return
        if score < best_score:
            best_score = score
            best_pos = pos

    if not common:
        dur = pt[job][op][new_alt][1]
        for pos in range(left, right + 1):
            consider(pos, sv_minus + dur + tv_minus)
    elif machine != old_machine:
        dur = pt[job][op][new_alt][1]
        for i in range(len(common) + 1):
            if i == 0:
                pos = seq.index(common[0])
                consider(pos, dur + sv_minus + info["dur"][common[0]] + info["tail"][common[0]])
            elif i < len(common):
                pos = seq.index(common[i - 1]) + 1
                consider(pos, dur + info["start"][common[i - 1]] + info["dur"][common[i - 1]] + info["dur"][common[i]] + info["tail"][common[i]])
            else:
                pos = seq.index(common[-1]) + 1
                consider(pos, dur + info["start"][common[-1]] + info["dur"][common[-1]] + tv_minus)
    else:
        for pos in range(left, right + 1):
            consider(pos, _lpath_score_same_machine(info, idx, pos))

    if best_pos is None:
        return None
    return {
        "job": job,
        "op": op,
        "flat_idx": idx,
        "old_machine": old_machine,
        "new_machine": machine,
        "new_alt": new_alt,
        "insert_pos": best_pos,
        "score": best_score,
    }


def _reencode_from_machine_sequences(pt: ProcessingTimes, info: Dict, machine_seq: List[List[int]]) -> List[int]:
    ms_off = build_ms_offsets(pt)
    flat_ops = flatten_operations(pt)
    n_ops = sum(len(job_ops) for job_ops in pt)
    mach_pred = [-1] * n_ops
    mach_succ = [-1] * n_ops
    for seq in machine_seq:
        for i, idx in enumerate(seq):
            if i > 0:
                mach_pred[idx] = seq[i - 1]
            if i + 1 < len(seq):
                mach_succ[idx] = seq[i + 1]

    indeg = [0] * n_ops
    for idx in range(n_ops):
        job, op = flat_ops[idx]
        if op > 0:
            indeg[idx] += 1
        if mach_pred[idx] >= 0:
            indeg[idx] += 1

    avail = [idx for idx in range(n_ops) if indeg[idx] == 0]
    os_str: List[int] = []
    while avail:
        idx = min(avail, key=lambda x: (info["start"][x], info["sched_pos"][x]))
        avail.remove(idx)
        job, op = flat_ops[idx]
        os_str.append(job)
        if op + 1 < len(pt[job]):
            js = ms_off[job] + op + 1
            indeg[js] -= 1
            if indeg[js] == 0:
                avail.append(js)
        if mach_succ[idx] >= 0:
            ms = mach_succ[idx]
            indeg[ms] -= 1
            if indeg[ms] == 0:
                avail.append(ms)
    return os_str


def _apply_paper_move(current: Individual, pt: ProcessingTimes, info: Dict, move: Dict) -> Individual:
    child = current.copy()
    ms_off = build_ms_offsets(pt)
    child.ms[ms_off[move["job"]] + move["op"]] = move["new_alt"]
    machine_seq = [seq[:] for seq in info["machine_seq"]]
    machine_seq[move["old_machine"]].remove(move["flat_idx"])
    machine_seq[move["new_machine"]].insert(move["insert_pos"], move["flat_idx"])
    child.os = _reencode_from_machine_sequences(pt, info, machine_seq)
    return child


def _paper_tabu_search(
    ind: Individual,
    pt: ProcessingTimes,
    max_iter: int,
    ms_off: List[int],
) -> Individual:
    current = ind.copy()
    evaluate(current, pt, ms_off)
    best = current.copy()
    tabu_until: Dict[Tuple[int, int], int] = {}

    for it in range(1, max_iter + 1):
        _, schedule = decode(current, pt)
        critical = _find_critical_ops(schedule, current.fitness)
        info = _build_sched_info(schedule, pt)
        path = _preferred_critical_path(schedule, pt, current.fitness, critical, info)
        if not path:
            break

        candidates: List[Dict] = []
        for job, op in path:
            for machine, _dur in pt[job][op]:
                move = _build_paper_move_for_machine(pt, job, op, machine, info)
                if move is not None:
                    candidates.append(move)
        if not candidates:
            break

        aspiration = [mv for mv in candidates if mv["score"] < best.fitness]
        non_tabu = [mv for mv in candidates if tabu_until.get((mv["flat_idx"], mv["old_machine"]), 0) <= it]

        if aspiration:
            chosen = min(aspiration, key=lambda mv: mv["score"])
        elif non_tabu:
            top = sorted(non_tabu, key=lambda mv: mv["score"])[:2]
            chosen = random.choice(top)
        else:
            chosen = min(
                candidates,
                key=lambda mv: (tabu_until.get((mv["flat_idx"], mv["old_machine"]), 0), mv["score"]),
            )

        current = _apply_paper_move(current, pt, info, chosen)
        evaluate(current, pt, ms_off)
        if current.fitness < best.fitness:
            best = current.copy()

        tabu_len = len(path) + len(pt[chosen["job"]][chosen["op"]])
        tabu_until[(chosen["flat_idx"], chosen["old_machine"])] = it + tabu_len

    return best


# ---------------------------------------------------------------------------
# Tabu Search
# ---------------------------------------------------------------------------

def tabu_search(
    ind: Individual,
    pt: ProcessingTimes,
    max_iter: int,
    tabu_len: int = PARAMS["tabu_len"],
    ms_off: Optional[List[int]] = None,
) -> Individual:
    """
    Tabu Search with critical-path-based neighborhood.

    N1 (OS moves): swap pairs of adjacent critical operations on the same machine.
    N2 (MS moves): try alternative machines for each critical operation.
    Aspiration criterion: accept a tabu move if it improves the best known solution.
    Termination: max_iter iterations.
    """
    if ms_off is None:
        ms_off = build_ms_offsets(pt)
    return _paper_tabu_search(ind, pt, max_iter=max_iter, ms_off=ms_off)


# ---------------------------------------------------------------------------
# Hybrid GA + TS  (main algorithm)
# ---------------------------------------------------------------------------

def hybrid_ga_ts(
    pt: ProcessingTimes,
    pop_size: int    = PARAMS["pop_size"],
    max_gen: int     = PARAMS["max_gen"],
    max_stagnant: int= PARAMS["max_stagnant"],
    pc: float        = PARAMS["pc"],
    pm: float        = PARAMS["pm"],
    pr: float        = PARAMS["pr"],
    tabu_len: int    = PARAMS["tabu_len"],
    ts_iter_base: int= PARAMS["ts_iter_base"],
    verbose: bool    = True,
) -> Individual:
    """
    Hybrid Genetic Algorithm + Tabu Search for FJSP makespan minimisation.

    Default parameters match Table 2 of Li & Gao (2016):
        pop_size=400, max_gen=200, max_stagnant=20, pr=0.005,
        pc=0.8, pm=0.1, ts_iter_base=800, tabu_len=9.

    The TS iteration count grows adaptively:
        maxTSIter = ts_iter_base * (gen / max_gen)

    Returns the best Individual found.
    """
    # ---------- Step 2: Initialisation ----------
    flat_ops   = flatten_operations(pt)
    ms_off     = build_ms_offsets(pt)
    population = [random_individual(pt) for _ in range(pop_size)]
    for ind in population:
        evaluate(ind, pt, ms_off)

    best_ever     = min(population, key=lambda x: x.fitness).copy()
    stagnant_cnt  = 0

    for gen in range(1, max_gen + 1):

        # ---------- Step 4: Termination check ----------
        if stagnant_cnt >= max_stagnant:
            if verbose:
                print(f"[HA] Early stop at gen {gen}: "
                      f"no improvement for {max_stagnant} generations.")
            break

        # ---------- Step 5.1: Generate new population ----------
        new_pop = elitist_selection(population, pr)

        while len(new_pop) < pop_size:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            c1, c2 = p1.copy(), p2.copy()

            # Crossover
            if random.random() < pc:
                c1.os, c2.os = crossover_os(p1.os, p2.os)
                c1.ms, c2.ms = two_point_crossover_ms(p1.ms, p2.ms)
                c1.fitness = c2.fitness = None

            # Mutation
            if random.random() < pm:
                c1.os = mutate_os(c1.os)
                c1.ms = mutate_ms(c1.ms, pt, flat_ops)
                c1.fitness = None
            if random.random() < pm:
                c2.os = mutate_os(c2.os)
                c2.ms = mutate_ms(c2.ms, pt, flat_ops)
                c2.fitness = None

            evaluate(c1, pt, ms_off)
            evaluate(c2, pt, ms_off)
            new_pop.extend([c1, c2])

        population = new_pop[:pop_size]

        ts_iters = max(1, int(ts_iter_base * gen / max_gen))
        population = [
            tabu_search(ind, pt, max_iter=ts_iters, tabu_len=tabu_len, ms_off=ms_off)
            for ind in population
        ]

        # ---------- Step 3: Evaluate ----------
        gen_best = min(population, key=lambda x: x.fitness)
        if gen_best.fitness < best_ever.fitness:
            best_ever    = gen_best.copy()
            stagnant_cnt = 0
        else:
            stagnant_cnt += 1

        if verbose:
            ts_iters = max(1, int(ts_iter_base * gen / max_gen))
            print(
                f"[HA] Gen {gen:3d}/{max_gen} | "
                f"gen best: {gen_best.fitness:5d} | "
                f"global best: {best_ever.fitness:5d} | "
                f"TS iters: {ts_iters:4d} | "
                f"stagnant: {stagnant_cnt}"
            )

    return best_ever


# ---------------------------------------------------------------------------
# Instance-loading utilities
# ---------------------------------------------------------------------------

def fjsp_to_processing_times(fjsp_instance) -> ProcessingTimes:
    """
    Convert a FJSPData object (from instance_generator.py) to ProcessingTimes.

    ProcessingTimes: pt[job][op] = [(machine_id, duration), ...]
    """
    pt: ProcessingTimes = []
    for job_idx in range(fjsp_instance.num_jobs):
        job_ops: List[List[Tuple[int, int]]] = []
        num_ops = fjsp_instance.nums_operation[job_idx]
        for op_idx in range(num_ops):
            flat_op = fjsp_instance.num_ope_bias[job_idx] + op_idx
            n_opts  = fjsp_instance.nums_option[flat_op]
            m_bias  = fjsp_instance.num_machine_bias[flat_op]
            alts: List[Tuple[int, int]] = [
                (fjsp_instance.ope_machine[m_bias + k],
                 fjsp_instance.processing_time[m_bias + k])
                for k in range(n_opts)
            ]
            job_ops.append(alts)
        pt.append(job_ops)
    return pt


def export_pt_to_text(pt: ProcessingTimes, path: str) -> None:
    """
    Write ProcessingTimes to the text format read by ha_solver (C++).

    Format:
        n_jobs  n_machines  avg_options
        n_ops  n_opts  m t  m t ...    (one line per job; machines 0-indexed)
    """
    n_machines = max(m for job_ops in pt for alts in job_ops for m, _ in alts) + 1
    total_opts = sum(len(alts) for job_ops in pt for alts in job_ops)
    total_ops  = sum(len(job_ops) for job_ops in pt)
    avg_opts   = total_opts / total_ops if total_ops else 1.0

    with open(path, "w") as f:
        f.write(f"{len(pt)} {n_machines} {avg_opts:.2f}\n")
        for job_ops in pt:
            parts = [str(len(job_ops))]
            for alts in job_ops:
                parts.append(str(len(alts)))
                for m, t in alts:
                    parts += [str(m), str(t)]
            f.write(" ".join(parts) + "\n")


def load_instance(instance_name: str) -> Tuple[ProcessingTimes, object]:
    """
    Load a FJSP instance by name from data/fjsp_instances/ and return
    (processing_times, fjsp_data_object).
    """
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "fjsp_instances" / f"{instance_name}.fjsp"

    if not path.exists():
        available = [f.stem for f in (root / "data" / "fjsp_instances").glob("*.fjsp")]
        raise FileNotFoundError(
            f"Instance '{instance_name}' not found at {path}.\n"
            f"Available: {available}"
        )

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module == "instance_generator" and name == "FJSPData":
                from instance_generator import FJSPData
                return FJSPData
            return super().find_class(module, name)

    with open(path, "rb") as f:
        fjsp = _CompatUnpickler(f).load()

    print(f"Loaded: {fjsp.instance_name} "
          f"({fjsp.num_jobs} jobs, {fjsp.num_machines} machines, "
          f"{fjsp.num_operations} operations)")

    return fjsp_to_processing_times(fjsp), fjsp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HA (GA+TS) solver for FJSP")
    parser.add_argument("instance_name", help="Instance name, e.g. i3_k3_1")
    parser.add_argument("--pop-size",    type=int,   default=PARAMS["pop_size"])
    parser.add_argument("--max-gen",     type=int,   default=PARAMS["max_gen"])
    parser.add_argument("--stagnant",    type=int,   default=PARAMS["max_stagnant"])
    parser.add_argument("--ts-base",     type=int,   default=PARAMS["ts_iter_base"])
    parser.add_argument("--seed",        type=int,   default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    pt, _ = load_instance(args.instance_name)

    t0   = time.time()
    best = hybrid_ga_ts(
        pt,
        pop_size     = args.pop_size,
        max_gen      = args.max_gen,
        max_stagnant = args.stagnant,
        ts_iter_base = args.ts_base,
    )
    elapsed = time.time() - t0

    print(f"\n=== Result ===")
    print(f"Makespan : {best.fitness}")
    print(f"Time     : {elapsed:.2f} s")
    print(f"OS string: {best.os}")
    print(f"MS string: {best.ms}")
