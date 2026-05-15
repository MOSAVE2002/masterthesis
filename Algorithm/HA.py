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

from itertools import permutations
import random
import time
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import pickle

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

# pt[job][op] = [(machine_id, duration), ...]
ProcessingTimes = List[List[List[Tuple[int, int]]]]

# ---------------------------------------------------------------------------
# Default parameters  (Table 2 of the paper)
# ---------------------------------------------------------------------------

PARAMS: Dict = {
    "pop_size":     400,
    "max_gen":      200,
    "max_stagnant": 20,
    "pr":           0.005,  # elitist reproduction probability
    "pc":           0.8,    # crossover probability
    "pm":           0.1,    # mutation probability
    "ts_iter_base": 800,    # maxTSIterSize = ts_iter_base * (gen / max_gen)
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
    job_count = [0] * len(pt)
    job_ready = [0] * len(pt)
    machine_intervals: Dict[int, List[Tuple[int, int]]] = {}
    schedule: List[Dict] = []

    for job in ind.os:
        op = job_count[job]
        if op >= len(pt[job]):
            continue

        idx     = ms_index(job, op, pt)
        m_alt   = ind.ms[idx]
        machine, duration = pt[job][op][m_alt]

        asij      = job_ready[job]
        intervals = machine_intervals.setdefault(machine, [])
        start     = _earliest_slot(asij, duration, intervals)
        end       = start + duration

        intervals.append((start, end))
        intervals.sort()

        schedule.append({
            "job": job, "op": op, "machine": machine,
            "start": start, "end": end, "duration": duration,
        })

        job_ready[job] = end
        job_count[job] += 1

    makespan = max(e["end"] for e in schedule) if schedule else 0
    return makespan, schedule


def evaluate(ind: Individual, pt: ProcessingTimes) -> int:
    """Evaluate an individual and store its makespan as fitness."""
    ind.fitness, _ = decode(ind, pt)
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


def mutate_ms(ms: List[int], pt: ProcessingTimes) -> List[int]:
    """
    MS mutation: select r = |ms|/2 positions, reassign each to a
    different available machine for that operation.
    """
    child = ms[:]
    ops   = flatten_operations(pt)
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


# ---------------------------------------------------------------------------
# Tabu Search
# ---------------------------------------------------------------------------

def tabu_search(
    ind: Individual,
    pt: ProcessingTimes,
    max_iter: int,
    tabu_len: int = PARAMS["tabu_len"],
) -> Individual:
    """
    Tabu Search with critical-path-based neighborhood.

    N1 (OS moves): swap pairs of adjacent critical operations on the same machine.
    N2 (MS moves): try alternative machines for each critical operation.
    Aspiration criterion: accept a tabu move if it improves the best known solution.
    Termination: max_iter iterations.
    """
    current = ind.copy()
    evaluate(current, pt)
    best      = current.copy()
    tabu_list: List[tuple] = []

    for _ in range(max_iter):
        _, schedule = decode(current, pt)
        critical    = _find_critical_ops(schedule, current.fitness)

        neighbors: List[Tuple[Individual, tuple]] = []

        # ---- N1: swap adjacent critical ops on the same machine ----
        # Group critical ops by machine, sorted by start time
        mach_crit: Dict[int, List[Tuple[int, int, int]]] = {}
        for e in schedule:
            if (e["job"], e["op"]) in critical:
                mach_crit.setdefault(e["machine"], []).append(
                    (e["start"], e["job"], e["op"])
                )
        for ops in mach_crit.values():
            ops.sort()

        for ops in mach_crit.values():
            for i in range(len(ops) - 1):
                _, j1, o1 = ops[i]
                _, j2, o2 = ops[i + 1]
                pos1 = _op_pos_in_os(current.os, j1, o1)
                pos2 = _op_pos_in_os(current.os, j2, o2)
                if pos1 is None or pos2 is None:
                    continue
                new_os       = current.os[:]
                new_os[pos1], new_os[pos2] = new_os[pos2], new_os[pos1]
                cand         = Individual(new_os, current.ms[:])
                evaluate(cand, pt)
                neighbors.append((cand, ("os", j1, o1, j2, o2)))

        # ---- N2: try alternative machines for each critical op ----
        for j, o in critical:
            idx     = ms_index(j, o, pt)
            cur_alt = current.ms[idx]
            for alt in range(len(pt[j][o])):
                if alt == cur_alt:
                    continue
                new_ms      = current.ms[:]
                new_ms[idx] = alt
                cand        = Individual(current.os[:], new_ms)
                evaluate(cand, pt)
                neighbors.append((cand, ("ms", j, o, cur_alt, alt)))

        if not neighbors:
            break

        neighbors.sort(key=lambda x: x[0].fitness)

        # Pick best non-tabu neighbor (or tabu if it meets aspiration)
        chosen, chosen_move = neighbors[0]  # fallback if all tabu
        for cand, move in neighbors:
            if move not in tabu_list or cand.fitness < best.fitness:
                chosen, chosen_move = cand, move
                break

        current = chosen
        if current.fitness < best.fitness:
            best = current.copy()

        tabu_list.append(chosen_move)
        if len(tabu_list) > tabu_len:
            tabu_list.pop(0)

    return best


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
    population = [random_individual(pt) for _ in range(pop_size)]
    for ind in population:
        evaluate(ind, pt)

    best_ever     = min(population, key=lambda x: x.fitness).copy()
    stagnant_cnt  = 0

    for gen in range(1, max_gen + 1):

        # ---------- Step 4: Termination check ----------
        if stagnant_cnt >= max_stagnant:
            if verbose:
                print(f"[HA] Early stop at gen {gen}: "
                      f"no improvement for {max_stagnant} generations.")
            break

        population.sort(key=lambda x: x.fitness)

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
                c1.ms = mutate_ms(c1.ms, pt)
                c1.fitness = None
            if random.random() < pm:
                c2.os = mutate_os(c2.os)
                c2.ms = mutate_ms(c2.ms, pt)
                c2.fitness = None

            evaluate(c1, pt)
            evaluate(c2, pt)

            # ---------- Step 5.2: Local improvement via TS ----------
            ts_iters = max(1, int(ts_iter_base * gen / max_gen))
            c1 = tabu_search(c1, pt, max_iter=ts_iters, tabu_len=tabu_len)
            c2 = tabu_search(c2, pt, max_iter=ts_iters, tabu_len=tabu_len)

            new_pop.extend([c1, c2])

        population = new_pop[:pop_size]

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

    with open(path, "rb") as f:
        fjsp = pickle.load(f)

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
