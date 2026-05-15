/**
 * Hybrid GA + Tabu Search for the Flexible Job Shop Scheduling Problem (FJSP).
 *
 * This implementation follows the Li/Gao hybrid framework together with the
 * schedule-based tabu-search neighborhood derived from Mastrolilli/Gambardella.
 *
 * Build:  cd Algorithm && make
 * Usage:  ./ha_solver <instance.txt> [--pop-size N] [--max-gen N]
 *                     [--stagnant N] [--ts-base N] [--seed N] [--quiet]
 *
 * Instance text format (Brandimarte / OR-Library style):
 *   n_jobs  n_machines  [avg_options]
 *   n_ops  n_opts  m t  m t ...    (one line per job; machines 0-indexed)
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <climits>
#include <deque>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

// ─── Types ───────────────────────────────────────────────────────────────────

using Alt  = std::pair<int,int>;           // (machine_id, duration)
using Alts = std::vector<Alt>;
using PT   = std::vector<std::vector<Alts>>; // pt[job][op]
static constexpr unsigned DEFAULT_SEED = 42;

struct Individual {
    std::vector<int> os;
    std::vector<int> ms;
    int fitness = INT_MAX;
};

struct Entry { int job, op, machine, start, end, dur; };

using JO = std::pair<int,int>;

// ─── Problem Context (computed once, passed everywhere) ───────────────────────

struct ProbCtx {
    int n_jobs, n_machines, n_ops;
    std::vector<int>              ms_off;   // ms_off[j]+o = flat index
    std::vector<int>              job_op_counts;
    std::vector<int>              job_ids;
    std::vector<int>              base_os;
    std::vector<std::pair<int,int>> flat_ops; // flat_ops[idx] = (job, op)
};

static ProbCtx make_ctx(const PT& pt) {
    ProbCtx c;
    c.n_jobs = (int)pt.size();
    c.ms_off.resize(c.n_jobs + 1, 0);
    c.job_op_counts.resize(c.n_jobs, 0);
    c.job_ids.resize(c.n_jobs);
    for (int j = 0; j < c.n_jobs; ++j) {
        c.job_op_counts[j] = (int)pt[j].size();
        c.ms_off[j+1] = c.ms_off[j] + c.job_op_counts[j];
        for (int o = 0; o < c.job_op_counts[j]; ++o) {
            c.flat_ops.push_back({j, o});
            c.base_os.push_back(j);
        }
    }
    std::iota(c.job_ids.begin(), c.job_ids.end(), 0);
    c.n_ops = (int)c.flat_ops.size();
    c.n_machines = 0;
    for (auto& job_ops : pt)
        for (auto& alts : job_ops)
            for (auto [m, t] : alts)
                c.n_machines = std::max(c.n_machines, m + 1);
    return c;
}

// ─── Decode State (pre-allocated, reused across millions of decode calls) ─────

struct DecodeState {
    std::vector<std::vector<std::pair<int,int>>> mach_ivals; // [machine] sorted
    std::vector<int> mach_last_end;
    std::vector<int> used_machines;
    std::vector<int> job_count;
    std::vector<int> job_ready;
    std::vector<Entry> schedule;

    // Scratch for find_critical_ops (pre-allocated, avoids per-call heap allocation)
    std::vector<const Entry*>               op_info_flat;
    std::unordered_map<int64_t, JO>         mach_end;
    std::vector<bool>                       critical_flags;  // indexed by flat op idx
    std::vector<int>                        critical_dirty;  // indices to reset
    std::vector<JO>                         critical_list;
    std::vector<JO>                         bfs_q, bfs_nxt;

    // Scratch for N1 neighbor generation in tabu_search
    std::vector<std::vector<std::tuple<int,int,int>>> mach_crit_vec;
    std::vector<int>                        mach_crit_dirty;
    std::vector<std::vector<int>>           pos_in_os;
    std::vector<int>                        mut_positions;

    explicit DecodeState(const ProbCtx& c)
        : mach_ivals(c.n_machines),
          mach_last_end(c.n_machines, 0),
          job_count(c.n_jobs),
          job_ready(c.n_jobs),
          op_info_flat(c.n_ops, nullptr),
          critical_flags(c.n_ops, false),
          mach_crit_vec(c.n_machines),
          pos_in_os(c.n_jobs),
          mut_positions(c.n_ops) {
        int avg_machine_load = std::max(1, c.n_ops / std::max(1, c.n_machines));
        for (auto& iv : mach_ivals) iv.reserve(avg_machine_load);
        for (int j = 0; j < c.n_jobs; ++j) pos_in_os[j].reserve(c.job_op_counts[j]);
        schedule.reserve(c.n_ops);
        used_machines.reserve(c.n_machines);
        mach_end.reserve(c.n_ops * 2);
        critical_dirty.reserve(c.n_ops);
        critical_list.reserve(c.n_ops);
        bfs_q.reserve(c.n_ops);
        bfs_nxt.reserve(c.n_ops);
        mach_crit_dirty.reserve(c.n_machines);
        std::iota(mut_positions.begin(), mut_positions.end(), 0);
    }

    void reset(int /*n_jobs*/) {
        for (int m : used_machines) {
            mach_ivals[m].clear();
            mach_last_end[m] = 0;
        }
        used_machines.clear();
        std::fill(job_count.begin(), job_count.end(), 0);
        std::fill(job_ready.begin(), job_ready.end(), 0);
        schedule.clear();
    }

    std::vector<std::pair<int,int>>& machine(int m) {
        if (mach_ivals[m].empty()) used_machines.push_back(m);
        return mach_ivals[m];
    }
};

// ─── Active Schedule Decoder ─────────────────────────────────────────────────

struct SlotResult {
    int start;
    int insert_pos;
};

static inline SlotResult earliest_slot_and_pos(
    int asij, int dur, const std::vector<std::pair<int,int>>& iv
) {
    if (iv.empty()) return {asij, 0};
    if (asij + dur <= iv[0].first) return {asij, 0};
    for (int i = 0; i + 1 < (int)iv.size(); ++i) {
        int gs = iv[i].second, ge = iv[i + 1].first;
        int c  = gs > asij ? gs : asij;
        if (c + dur <= ge) return {c, i + 1};
    }
    int last = iv.back().second;
    return {last > asij ? last : asij, (int)iv.size()};
}

static inline void place_operation(
    DecodeState& s, int machine, int asij, int dur, int& start, int& end
) {
    auto& iv  = s.machine(machine);
    if (iv.empty() || asij >= s.mach_last_end[machine]) {
        start = asij > s.mach_last_end[machine] ? asij : s.mach_last_end[machine];
        end = start + dur;
        iv.emplace_back(start, end);
    } else {
        SlotResult slot = earliest_slot_and_pos(asij, dur, iv);
        start = slot.start;
        end   = start + dur;
        iv.emplace(iv.begin() + slot.insert_pos, start, end);
    }
    s.mach_last_end[machine] = iv.back().second;
}

// Fast path: makespan only, no schedule list
static int decode_makespan(const Individual& ind, const PT& pt,
                            const ProbCtx& c, DecodeState& s) {
    s.reset(c.n_jobs);
    int makespan = 0;
    for (int job : ind.os) {
        int op = s.job_count[job]++;
        auto [machine, dur] = pt[job][op][ind.ms[c.ms_off[job] + op]];
        int asij  = s.job_ready[job];
        int start, end;
        place_operation(s, machine, asij, dur, start, end);
        s.job_ready[job] = end;
        if (end > makespan) makespan = end;
    }
    return makespan;
}

// Full decode: fills s.schedule and returns makespan
static int decode_full(const Individual& ind, const PT& pt,
                        const ProbCtx& c, DecodeState& s) {
    s.reset(c.n_jobs);
    int makespan = 0;
    for (int pos = 0; pos < (int)ind.os.size(); ++pos) {
        int job = ind.os[pos];
        int op = s.job_count[job]++;
        auto [machine, dur] = pt[job][op][ind.ms[c.ms_off[job] + op]];
        int asij  = s.job_ready[job];
        int start, end;
        place_operation(s, machine, asij, dur, start, end);
        s.schedule.push_back({job, op, machine, start, end, dur});
        s.job_ready[job] = end;
        if (end > makespan) makespan = end;
    }
    return makespan;
}

// ─── Critical Path ───────────────────────────────────────────────────────────

// Fills s.critical_flags / s.critical_list / s.op_info_flat using pre-allocated buffers.
// No heap allocation on repeated calls.
static void find_critical_ops(const std::vector<Entry>& sched, int makespan,
                               const ProbCtx& c, DecodeState& s) {
    // Reset flags from last call
    for (int i : s.critical_dirty) s.critical_flags[i] = false;
    s.critical_dirty.clear();
    s.critical_list.clear();
    s.mach_end.clear();

    // Build op_info_flat and mach_end
    std::fill(s.op_info_flat.begin(), s.op_info_flat.end(), nullptr);
    for (auto& e : sched) {
        s.op_info_flat[c.ms_off[e.job] + e.op] = &e;
        s.mach_end[(int64_t)e.machine * 1000000LL + e.end] = {e.job, e.op};
    }

    auto mark = [&](int j, int o) -> bool {
        int idx = c.ms_off[j] + o;
        if (s.critical_flags[idx]) return false;
        s.critical_flags[idx] = true;
        s.critical_dirty.push_back(idx);
        s.critical_list.push_back({j, o});
        return true;
    };

    s.bfs_q.clear();
    for (auto& e : sched)
        if (e.end == makespan && mark(e.job, e.op))
            s.bfs_q.push_back({e.job, e.op});

    while (!s.bfs_q.empty()) {
        s.bfs_nxt.clear();
        for (auto [j, o] : s.bfs_q) {
            const Entry& e = *s.op_info_flat[c.ms_off[j] + o];
            if (o > 0) {
                const Entry* pred = s.op_info_flat[c.ms_off[j] + o - 1];
                if (pred && pred->end == e.start && mark(j, o - 1))
                    s.bfs_nxt.push_back({j, o - 1});
            }
            auto mit = s.mach_end.find((int64_t)e.machine * 1000000LL + e.start);
            if (mit != s.mach_end.end() && mark(mit->second.first, mit->second.second))
                s.bfs_nxt.push_back(mit->second);
        }
        std::swap(s.bfs_q, s.bfs_nxt);
    }
}

// ─── GA Operators ────────────────────────────────────────────────────────────

static void evaluate(Individual& ind, const PT& pt, const ProbCtx& c, DecodeState& s) {
    ind.fitness = decode_makespan(ind, pt, c, s);
}

static std::vector<Individual> elitist_selection(const std::vector<Individual>& pop,
                                                   double pr) {
    int n = std::max(1, (int)(pr * pop.size()));
    std::vector<int> idx(pop.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin()+n, idx.end(),
                      [&](int a, int b){ return pop[a].fitness < pop[b].fitness; });
    std::vector<Individual> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i) out.push_back(pop[idx[i]]);
    return out;
}

static const Individual& tournament_selection(const std::vector<Individual>& pop,
                                               int k, std::mt19937& rng) {
    std::uniform_int_distribution<> dist(0, (int)pop.size()-1);
    int best = dist(rng);
    for (int i = 1; i < k; ++i) { int x = dist(rng); if (pop[x].fitness < pop[best].fitness) best = x; }
    return pop[best];
}

static std::pair<std::vector<int>, std::vector<int>>
os_from_masks(const std::vector<int>& p1, const std::vector<int>& p2,
              const std::vector<unsigned char>& keep1,
              const std::vector<unsigned char>& keep2) {
    auto make = [](const std::vector<int>& a, const std::vector<int>& b,
                   const std::vector<unsigned char>& keep) {
        std::vector<int> child(a.size(), -1);
        for (int i = 0; i < (int)a.size(); ++i) if (keep[a[i]]) child[i] = a[i];
        std::vector<int> rem;
        rem.reserve(b.size());
        for (int g : b) if (!keep[g]) rem.push_back(g);
        int r = 0; for (auto& x : child) if (x == -1) x = rem[r++];
        return child;
    };
    return {make(p1, p2, keep1), make(p2, p1, keep2)};
}

static std::pair<std::vector<int>, std::vector<int>>
crossover_os_pox(const std::vector<int>& p1, const std::vector<int>& p2,
                 const ProbCtx& c, std::mt19937& rng) {
    std::vector<int> jobs = c.job_ids;
    std::shuffle(jobs.begin(), jobs.end(), rng);
    int split = std::uniform_int_distribution<>(1, std::max(1,(int)jobs.size()-1))(rng);
    std::vector<unsigned char> keep1(c.n_jobs, 0);
    for (int i = 0; i < split; ++i) keep1[jobs[i]] = 1;
    return os_from_masks(p1, p2, keep1, keep1);
}

static std::pair<std::vector<int>, std::vector<int>>
crossover_os_jbx(const std::vector<int>& p1, const std::vector<int>& p2,
                 const ProbCtx& c, std::mt19937& rng) {
    std::vector<int> jobs = c.job_ids;
    std::shuffle(jobs.begin(), jobs.end(), rng);
    int split = std::uniform_int_distribution<>(1, std::max(1,(int)jobs.size()-1))(rng);
    std::vector<unsigned char> keep1(c.n_jobs, 0), keep2(c.n_jobs, 0);
    for (int i = 0; i < split; ++i) keep1[jobs[i]] = 1;
    for (int i = split; i < (int)jobs.size(); ++i) keep2[jobs[i]] = 1;
    return os_from_masks(p1, p2, keep1, keep2);
}

static std::pair<std::vector<int>, std::vector<int>>
crossover_os(const std::vector<int>& p1, const std::vector<int>& p2,
             const ProbCtx& c, std::mt19937& rng) {
    if (std::bernoulli_distribution(0.5)(rng)) {
        return crossover_os_pox(p1, p2, c, rng);
    }
    return crossover_os_jbx(p1, p2, c, rng);
}

static std::pair<std::vector<int>, std::vector<int>>
two_point_crossover_ms(const std::vector<int>& ms1, const std::vector<int>& ms2,
                        std::mt19937& rng) {
    if ((int)ms1.size() < 2) return {ms1, ms2};
    int n = (int)ms1.size();
    int a = std::uniform_int_distribution<>(0,n-1)(rng);
    int b = std::uniform_int_distribution<>(0,n-1)(rng);
    if (a > b) std::swap(a, b);
    auto c1 = ms1, c2 = ms2;
    for (int i = a; i < b; ++i) { c1[i] = ms2[i]; c2[i] = ms1[i]; }
    return {c1, c2};
}

static std::vector<int> mutate_os(const std::vector<int>& os, const ProbCtx&,
                                  std::mt19937& rng) {
    auto child = os;
    int n = (int)child.size();
    if (std::bernoulli_distribution(0.5)(rng)) {
        int i = std::uniform_int_distribution<>(0,n-1)(rng);
        int j = std::uniform_int_distribution<>(0,n-1)(rng);
        std::swap(child[i], child[j]);
    } else {
        if (n < 3) {
            std::swap(child[std::uniform_int_distribution<>(0,n-1)(rng)],
                      child[std::uniform_int_distribution<>(0,n-1)(rng)]);
            return child;
        }
        std::array<int, 3> pos = {-1, -1, -1};
        bool found = false;
        for (int tries = 0; tries < 64 && !found; ++tries) {
            pos[0] = std::uniform_int_distribution<>(0, n - 1)(rng);
            pos[1] = std::uniform_int_distribution<>(0, n - 1)(rng);
            pos[2] = std::uniform_int_distribution<>(0, n - 1)(rng);
            if (pos[0] == pos[1] || pos[0] == pos[2] || pos[1] == pos[2]) continue;
            int a = child[pos[0]], b = child[pos[1]], d = child[pos[2]];
            if (a != b && a != d && b != d) found = true;
        }
        if (!found) return child;

        std::array<int, 3> vals = {child[pos[0]], child[pos[1]], child[pos[2]]};
        std::array<std::array<int, 3>, 5> perms = {{
            {vals[0], vals[2], vals[1]},
            {vals[1], vals[0], vals[2]},
            {vals[1], vals[2], vals[0]},
            {vals[2], vals[0], vals[1]},
            {vals[2], vals[1], vals[0]},
        }};
        const auto& picked = perms[std::uniform_int_distribution<>(0, 4)(rng)];
        child[pos[0]] = picked[0];
        child[pos[1]] = picked[1];
        child[pos[2]] = picked[2];
    }
    return child;
}

static std::vector<int> mutate_ms(const std::vector<int>& ms, const PT& pt,
                                   const ProbCtx& c, DecodeState& s, std::mt19937& rng) {
    auto child = ms;
    int n = (int)child.size();
    int r = std::max(1, n/2);
    auto& positions = s.mut_positions;
    for (int i = 0; i < r; ++i) {
        int swap_idx = std::uniform_int_distribution<>(i, n - 1)(rng);
        std::swap(positions[i], positions[swap_idx]);
    }
    for (int i = 0; i < r; ++i) {
        int pos = positions[i];
        auto [job, op] = c.flat_ops[pos];
        int nalts = (int)pt[job][op].size();
        if (nalts > 1) {
            int cur = child[pos], alt;
            do { alt = std::uniform_int_distribution<>(0,nalts-1)(rng); } while (alt == cur);
            child[pos] = alt;
        }
    }
    for (int i = r - 1; i >= 0; --i) {
        std::swap(positions[i], positions[std::uniform_int_distribution<>(i, n - 1)(rng)]);
    }
    return child;
}

// ─── Tabu Search ─────────────────────────────────────────────────────────────

struct PaperMove {
    int flat_idx = -1;
    int job = -1;
    int op = -1;
    int old_machine = -1;
    int new_machine = -1;
    int new_alt = -1;
    int insert_pos = -1;   // insertion index in machine sequence without v
    double score = 0.0;    // estimated longest path containing the moved op
};

struct PaperSchedInfo {
    std::vector<int> start, dur, tail, machine_of, sched_pos;
    std::vector<int> job_pred, job_succ, mach_pred, mach_succ;
    std::vector<int> order;
    std::vector<std::vector<int>> machine_seq;
};

static int machine_alt_index(const PT& pt, int job, int op, int machine) {
    const auto& alts = pt[job][op];
    for (int i = 0; i < (int)alts.size(); ++i) {
        if (alts[i].first == machine) return i;
    }
    return -1;
}

static void build_paper_sched_info(const std::vector<Entry>& sched, const ProbCtx& c,
                                   PaperSchedInfo& info) {
    info.start.assign(c.n_ops, 0);
    info.dur.assign(c.n_ops, 0);
    info.tail.assign(c.n_ops, 0);
    info.machine_of.assign(c.n_ops, -1);
    info.sched_pos.assign(c.n_ops, -1);
    info.job_pred.assign(c.n_ops, -1);
    info.job_succ.assign(c.n_ops, -1);
    info.mach_pred.assign(c.n_ops, -1);
    info.mach_succ.assign(c.n_ops, -1);
    info.order.clear();
    info.order.reserve(c.n_ops);
    info.machine_seq.assign(c.n_machines, {});

    for (int i = 0; i < (int)sched.size(); ++i) {
        const auto& e = sched[i];
        int idx = c.ms_off[e.job] + e.op;
        info.start[idx] = e.start;
        info.dur[idx] = e.dur;
        info.machine_of[idx] = e.machine;
        info.sched_pos[idx] = i;
        info.order.push_back(idx);
        info.machine_seq[e.machine].push_back(idx);
        if (e.op > 0) {
            int pred = c.ms_off[e.job] + (e.op - 1);
            info.job_pred[idx] = pred;
            info.job_succ[pred] = idx;
        }
    }

    for (auto& seq : info.machine_seq) {
        std::sort(seq.begin(), seq.end(), [&](int a, int b) {
            if (info.start[a] != info.start[b]) return info.start[a] < info.start[b];
            return a < b;
        });
        for (int i = 0; i < (int)seq.size(); ++i) {
            if (i > 0) info.mach_pred[seq[i]] = seq[i - 1];
            if (i + 1 < (int)seq.size()) info.mach_succ[seq[i]] = seq[i + 1];
        }
    }

    std::sort(info.order.begin(), info.order.end(), [&](int a, int b) {
        if (info.start[a] != info.start[b]) return info.start[a] < info.start[b];
        return a < b;
    });
    for (int p = (int)info.order.size() - 1; p >= 0; --p) {
        int idx = info.order[p];
        int best = 0;
        if (info.job_succ[idx] >= 0) {
            int s = info.job_succ[idx];
            best = std::max(best, info.dur[s] + info.tail[s]);
        }
        if (info.mach_succ[idx] >= 0) {
            int s = info.mach_succ[idx];
            best = std::max(best, info.dur[s] + info.tail[s]);
        }
        info.tail[idx] = best;
    }
}

static std::vector<JO> build_preferred_critical_path(int makespan, const ProbCtx& c,
                                                     const DecodeState& s,
                                                     const PaperSchedInfo& info) {
    int end_idx = -1;
    for (int idx : info.order) {
        if (info.start[idx] + info.dur[idx] + info.tail[idx] == makespan) {
            end_idx = idx;
        }
    }
    if (end_idx < 0) return {};

    int cur = end_idx;
    while (true) {
        int jp = info.job_pred[cur];
        int mp = info.mach_pred[cur];
        bool jp_critical = jp >= 0 &&
            s.critical_flags[jp] &&
            info.start[jp] + info.dur[jp] == info.start[cur];
        bool mp_critical = mp >= 0 &&
            s.critical_flags[mp] &&
            info.start[mp] + info.dur[mp] == info.start[cur];
        if (jp_critical) cur = jp;
        else if (mp_critical) cur = mp;
        else break;
    }

    std::vector<JO> path;
    path.reserve(c.n_ops);
    while (cur >= 0) {
        auto [job, op] = c.flat_ops[cur];
        path.push_back({job, op});
        int js = info.job_succ[cur];
        int ms = info.mach_succ[cur];
        bool js_critical = js >= 0 &&
            s.critical_flags[js] &&
            info.start[cur] + info.dur[cur] == info.start[js];
        bool ms_critical = ms >= 0 &&
            s.critical_flags[ms] &&
            info.start[cur] + info.dur[cur] == info.start[ms];
        if (js_critical) cur = js;
        else if (ms_critical) cur = ms;
        else break;
    }
    return path;
}

static void compute_removed_times(const ProbCtx& c, const PaperSchedInfo& info, int v_idx,
                                  std::vector<int>& s_minus, std::vector<int>& t_minus) {
    s_minus.assign(c.n_ops, 0);
    t_minus.assign(c.n_ops, 0);

    int mp = info.mach_pred[v_idx];
    int ms = info.mach_succ[v_idx];

    for (int idx : info.order) {
        if (idx == v_idx) continue;
        int finish = s_minus[idx] + info.dur[idx];

        int js = info.job_succ[idx];
        if (js >= 0) s_minus[js] = std::max(s_minus[js], finish);

        int msucc = info.mach_succ[idx];
        if (idx == mp) msucc = ms;
        if (msucc == v_idx) msucc = -1;
        if (msucc >= 0) s_minus[msucc] = std::max(s_minus[msucc], finish);
    }

    for (int p = (int)info.order.size() - 1; p >= 0; --p) {
        int idx = info.order[p];
        if (idx == v_idx) continue;
        int best = 0;

        int js = info.job_succ[idx];
        if (js >= 0) best = std::max(best, info.dur[js] + t_minus[js]);

        int msucc = info.mach_succ[idx];
        if (idx == mp) msucc = ms;
        if (msucc == v_idx) msucc = -1;
        if (msucc >= 0) best = std::max(best, info.dur[msucc] + t_minus[msucc]);

        t_minus[idx] = best;
    }
}

static int current_insert_pos_without_v(const std::vector<int>& seq, int v_idx) {
    int pos = 0;
    for (int idx : seq) {
        if (idx == v_idx) return pos;
        ++pos;
    }
    return pos;
}

static double lpath_score_for_same_machine_move(const PaperSchedInfo& info,
                                                int v_idx,
                                                int insert_pos) {
    const auto& base_seq = info.machine_seq[info.machine_of[v_idx]];
    int current_pos = -1;
    for (int i = 0; i < (int)base_seq.size(); ++i) {
        if (base_seq[i] == v_idx) {
            current_pos = i;
            break;
        }
    }
    if (current_pos < 0) return 1e100;

    std::vector<int> seq;
    seq.reserve(base_seq.size() - 1);
    for (int idx : base_seq) {
        if (idx != v_idx) seq.push_back(idx);
    }

    std::vector<int> new_seq = seq;
    new_seq.insert(new_seq.begin() + insert_pos, v_idx);

    int start = std::min(current_pos, insert_pos);
    int finish = std::max(current_pos, insert_pos);
    std::vector<int> Q;
    Q.reserve(finish - start + 1);
    for (int p = start; p <= finish; ++p) Q.push_back(new_seq[p]);
    if (Q.empty()) return 1e100;

    std::vector<int> rprime(Q.size(), 0), tprime(Q.size(), 0);
    auto job_ready = [&](int idx) {
        int jp = info.job_pred[idx];
        return jp >= 0 ? info.start[jp] + info.dur[jp] : 0;
    };
    auto job_tail = [&](int idx) {
        int js = info.job_succ[idx];
        return js >= 0 ? info.dur[js] + info.tail[js] : 0;
    };

    int pm_out = (start > 0) ? new_seq[start - 1] : -1;
    rprime[0] = std::max(job_ready(Q[0]),
                         pm_out >= 0 ? info.start[pm_out] + info.dur[pm_out] : 0);
    for (int i = 1; i < (int)Q.size(); ++i) {
        rprime[i] = std::max(job_ready(Q[i]), rprime[i - 1] + info.dur[Q[i - 1]]);
    }

    int sm_out = (finish + 1 < (int)new_seq.size()) ? new_seq[finish + 1] : -1;
    tprime.back() = std::max(job_tail(Q.back()),
                             sm_out >= 0 ? info.dur[sm_out] + info.tail[sm_out] : 0);
    for (int i = (int)Q.size() - 2; i >= 0; --i) {
        tprime[i] = std::max(job_tail(Q[i]), tprime[i + 1] + info.dur[Q[i + 1]]);
    }

    double best = 0.0;
    for (int i = 0; i < (int)Q.size(); ++i) {
        best = std::max(best, (double)(rprime[i] + info.dur[Q[i]] + tprime[i]));
    }
    return best;
}

static std::vector<std::vector<int>> machine_sequences_after_move(const PaperSchedInfo& info,
                                                                  const PaperMove& mv) {
    auto machine_seq = info.machine_seq;
    auto& old_seq = machine_seq[mv.old_machine];
    auto it = std::find(old_seq.begin(), old_seq.end(), mv.flat_idx);
    if (it != old_seq.end()) old_seq.erase(it);
    auto& new_seq = machine_seq[mv.new_machine];
    int pos = std::max(0, std::min(mv.insert_pos, (int)new_seq.size()));
    new_seq.insert(new_seq.begin() + pos, mv.flat_idx);
    return machine_seq;
}

static void reencode_from_machine_sequences(Individual& child, const ProbCtx& c,
                                            const PaperSchedInfo& info,
                                            const std::vector<std::vector<int>>& machine_seq) {
    std::vector<int> mach_pred(c.n_ops, -1), mach_succ(c.n_ops, -1);
    for (int m = 0; m < c.n_machines; ++m) {
        const auto& seq = machine_seq[m];
        for (int i = 0; i < (int)seq.size(); ++i) {
            if (i > 0) mach_pred[seq[i]] = seq[i - 1];
            if (i + 1 < (int)seq.size()) mach_succ[seq[i]] = seq[i + 1];
        }
    }

    std::vector<int> indeg(c.n_ops, 0);
    for (int idx = 0; idx < c.n_ops; ++idx) {
        auto [job, op] = c.flat_ops[idx];
        if (op > 0) ++indeg[idx];
        if (mach_pred[idx] >= 0) ++indeg[idx];
    }

    std::vector<int> avail;
    avail.reserve(c.n_ops);
    for (int idx = 0; idx < c.n_ops; ++idx) {
        if (indeg[idx] == 0) avail.push_back(idx);
    }

    child.os.clear();
    child.os.reserve(c.n_ops);
    auto pop_best = [&]() {
        auto best_it = avail.begin();
        for (auto it2 = avail.begin() + 1; it2 != avail.end(); ++it2) {
            int a = *best_it, b = *it2;
            if (info.start[b] < info.start[a] ||
                (info.start[b] == info.start[a] && info.sched_pos[b] < info.sched_pos[a])) {
                best_it = it2;
            }
        }
        int idx = *best_it;
        avail.erase(best_it);
        return idx;
    };

    while (!avail.empty()) {
        int idx = pop_best();
        child.os.push_back(c.flat_ops[idx].first);

        auto [job, op] = c.flat_ops[idx];
        if (op + 1 < c.job_op_counts[job]) {
            int js = c.ms_off[job] + op + 1;
            if (--indeg[js] == 0) avail.push_back(js);
        }
        if (mach_succ[idx] >= 0) {
            int ms = mach_succ[idx];
            if (--indeg[ms] == 0) avail.push_back(ms);
        }
    }

    if ((int)child.os.size() != c.n_ops) {
        child.os = c.base_os;
    }
}

static void apply_paper_move(Individual& child, const PaperMove& mv,
                             const ProbCtx& c, const PaperSchedInfo& info) {
    child.ms[c.ms_off[mv.job] + mv.op] = mv.new_alt;
    auto machine_seq = machine_sequences_after_move(info, mv);
    reencode_from_machine_sequences(child, c, info, machine_seq);
}

static bool build_paper_move_for_machine(const PT& pt, const ProbCtx& c,
                                         const PaperSchedInfo& info,
                                         int v_idx, int machine, PaperMove& out) {
    auto [job, op] = c.flat_ops[v_idx];
    int old_machine = info.machine_of[v_idx];
    int new_alt = machine_alt_index(pt, job, op, machine);
    if (new_alt < 0) return false;

    std::vector<int> s_minus, t_minus;
    compute_removed_times(c, info, v_idx, s_minus, t_minus);
    int sv_minus = (op > 0) ? (s_minus[c.ms_off[job] + op - 1] + info.dur[c.ms_off[job] + op - 1]) : 0;
    int tv_minus = (op + 1 < c.job_op_counts[job]) ? (info.dur[c.ms_off[job] + op + 1] + t_minus[c.ms_off[job] + op + 1]) : 0;

    const auto& base_seq = info.machine_seq[machine];
    std::vector<int> seq;
    seq.reserve(base_seq.size() + (machine != old_machine ? 1 : 0));
    for (int idx : base_seq) {
        if (idx != v_idx) seq.push_back(idx);
    }

    std::vector<int> in_r, in_l, common;
    in_r.reserve(seq.size());
    in_l.reserve(seq.size());
    common.reserve(seq.size());
    int left = 0, right = (int)seq.size();

    for (int i = 0; i < (int)seq.size(); ++i) {
        int x = seq[i];
        bool r = info.start[x] + info.dur[x] > sv_minus;
        bool l = info.dur[x] + info.tail[x] > tv_minus;
        if (l && !r) left = std::max(left, i + 1);
        if (r && !l) right = std::min(right, i);
        if (l && r) common.push_back(x);
    }
    if (left > right) return false;

    int current_pos = (machine == old_machine) ? current_insert_pos_without_v(base_seq, v_idx) : -1;
    int best_pos = -1;
    double best_score = 1e100;

    auto consider = [&](int pos, double score) {
        if (pos < left || pos > right) return;
        if (machine == old_machine && pos == current_pos) return;
        if (score < best_score) {
            best_score = score;
            best_pos = pos;
        }
    };

    if (common.empty()) {
        for (int pos = left; pos <= right; ++pos) {
            consider(pos, (double)(sv_minus + pt[job][op][new_alt].second + tv_minus));
        }
    } else if (machine != old_machine) {
        int pvk = pt[job][op][new_alt].second;
        int l = (int)common.size();
        for (int i = 0; i <= l; ++i) {
            int pos = (i == 0) ? info.machine_seq[machine].size() : 0;
            if (i == 0) {
                int first_pos = 0;
                while (first_pos < (int)seq.size() && seq[first_pos] != common[0]) ++first_pos;
                pos = first_pos;
                consider(pos, (double)(pvk + sv_minus + info.dur[common[0]] + info.tail[common[0]]));
            } else if (i < l) {
                int left_pos = 0;
                while (left_pos < (int)seq.size() && seq[left_pos] != common[i - 1]) ++left_pos;
                consider(left_pos + 1,
                         (double)(pvk + info.start[common[i - 1]] + info.dur[common[i - 1]]
                                  + info.dur[common[i]] + info.tail[common[i]]));
            } else {
                int last_pos = 0;
                while (last_pos < (int)seq.size() && seq[last_pos] != common[l - 1]) ++last_pos;
                consider(last_pos + 1,
                         (double)(pvk + info.start[common[l - 1]] + info.dur[common[l - 1]] + tv_minus));
            }
        }
    } else {
        for (int pos = left; pos <= right; ++pos) {
            consider(pos, lpath_score_for_same_machine_move(info, v_idx, pos));
        }
    }

    if (best_pos < 0) return false;
    out.flat_idx = v_idx;
    out.job = job;
    out.op = op;
    out.old_machine = old_machine;
    out.new_machine = machine;
    out.new_alt = new_alt;
    out.insert_pos = best_pos;
    out.score = best_score;
    return true;
}

static Individual paper_tabu_search(Individual ind, const PT& pt, int max_iter,
                                    const ProbCtx& c, DecodeState& s, std::mt19937& rng) {
    evaluate(ind, pt, c, s);
    Individual best = ind;
    PaperSchedInfo info;
    std::vector<int> tabu_until(c.n_ops * c.n_machines, 0);

    auto tabu_ref = [&](int flat_idx, int machine) -> int& {
        return tabu_until[flat_idx * c.n_machines + machine];
    };

    for (int iter = 1; iter <= max_iter; ++iter) {
        int mk = decode_full(ind, pt, c, s);
        ind.fitness = mk;
        if (mk < best.fitness) best = ind;

        find_critical_ops(s.schedule, mk, c, s);
        build_paper_sched_info(s.schedule, c, info);
        auto path = build_preferred_critical_path(mk, c, s, info);
        if (path.empty()) break;

        for (int j = 0; j < c.n_jobs; ++j) s.pos_in_os[j].clear();
        for (int i = 0; i < (int)ind.os.size(); ++i) s.pos_in_os[ind.os[i]].push_back(i);

        std::vector<PaperMove> candidates;
        candidates.reserve(path.size() * 2);
        for (auto [job, op] : path) {
            int v_idx = c.ms_off[job] + op;
            for (const auto& [machine, _dur] : pt[job][op]) {
                PaperMove mv;
                if (build_paper_move_for_machine(pt, c, info, v_idx, machine, mv)) {
                    candidates.push_back(mv);
                }
            }
        }
        if (candidates.empty()) break;

        std::vector<int> aspiration, non_tabu, tabu;
        for (int i = 0; i < (int)candidates.size(); ++i) {
            const auto& mv = candidates[i];
            if (mv.score < best.fitness) aspiration.push_back(i);
            else if (tabu_ref(mv.flat_idx, mv.old_machine) <= iter) non_tabu.push_back(i);
            else tabu.push_back(i);
        }

        auto by_score = [&](int a, int b) {
            return candidates[a].score < candidates[b].score;
        };
        int chosen = -1;
        if (!aspiration.empty()) {
            chosen = *std::min_element(aspiration.begin(), aspiration.end(), by_score);
        } else if (!non_tabu.empty()) {
            std::sort(non_tabu.begin(), non_tabu.end(), by_score);
            if ((int)non_tabu.size() >= 2) {
                chosen = non_tabu[std::uniform_int_distribution<>(0, 1)(rng)];
            } else {
                chosen = non_tabu[0];
            }
        } else {
            chosen = *std::min_element(tabu.begin(), tabu.end(), [&](int a, int b) {
                int ta = tabu_ref(candidates[a].flat_idx, candidates[a].old_machine);
                int tb = tabu_ref(candidates[b].flat_idx, candidates[b].old_machine);
                if (ta != tb) return ta < tb;
                return candidates[a].score < candidates[b].score;
            });
        }

        const auto& mv = candidates[chosen];
        Individual next = ind;
        apply_paper_move(next, mv, c, info);
        evaluate(next, pt, c, s);

        int tabu_len = (int)path.size() + (int)pt[mv.job][mv.op].size();
        tabu_ref(mv.flat_idx, mv.old_machine) = iter + tabu_len;
        ind = std::move(next);
    }
    return best;
}

static Individual tabu_search(Individual ind, const PT& pt, int max_iter,
                               const ProbCtx& c, DecodeState& s, std::mt19937& rng) {
    return paper_tabu_search(std::move(ind), pt, max_iter, c, s, rng);
}

// ─── Hybrid GA + TS ──────────────────────────────────────────────────────────
struct Params {
    int    pop_size     = 400;
    int    max_gen      = 200;
    int    max_stagnant = 20;
    double pr           = 0.005;
    double pc           = 0.8;
    double pm           = 0.1;
    int    ts_iter_base = 800;
    bool   verbose      = true;
};

static Individual hybrid_ga_ts(const PT& pt, const Params& p, std::mt19937& rng) {
    const ProbCtx c = make_ctx(pt);
    DecodeState   s(c);
    auto run_start = std::chrono::steady_clock::now();

    int n_threads = 1;
    std::vector<DecodeState> ts_states(n_threads, DecodeState(c));

    std::vector<Individual> pop(p.pop_size);
    for (auto& ind : pop) {
        ind.os = c.base_os;
        ind.ms.reserve(c.n_ops);
        for (int j = 0; j < c.n_jobs; ++j)
            for (int o = 0; o < c.job_op_counts[j]; ++o) {
                ind.ms.push_back(std::uniform_int_distribution<>(0,(int)pt[j][o].size()-1)(rng));
            }
        std::shuffle(ind.os.begin(), ind.os.end(), rng);
        evaluate(ind, pt, c, s);
    }

    Individual best_ever = *std::min_element(pop.begin(), pop.end(),
        [](auto& a, auto& b){ return a.fitness < b.fitness; });
    int stagnant = 0;

    for (int gen = 1; gen <= p.max_gen; ++gen) {
        // HA terminates when maxGen is reached or when no improvement was found
        // for maxStagnant consecutive generations.
        if (stagnant >= p.max_stagnant) {
            if (p.verbose)
                std::cout << "[HA] Early stop at gen " << gen
                          << ": no improvement for " << p.max_stagnant << " gens.\n";
            break;
        }

        // Phase 1: generate all children sequentially (shared DecodeState s for evaluate)
        std::vector<Individual> new_pop = elitist_selection(pop, p.pr);
        while ((int)new_pop.size() < p.pop_size) {
            const Individual& par1 = tournament_selection(pop, 2, rng);
            const Individual& par2 = tournament_selection(pop, 2, rng);
            Individual ch1 = par1, ch2 = par2;

            if (std::bernoulli_distribution(p.pc)(rng)) {
                auto [os1,os2] = crossover_os(par1.os, par2.os, c, rng);
                auto [ms1,ms2] = two_point_crossover_ms(par1.ms, par2.ms, rng);
                ch1.os=os1; ch1.ms=ms1; ch1.fitness=INT_MAX;
                ch2.os=os2; ch2.ms=ms2; ch2.fitness=INT_MAX;
            }
            if (std::bernoulli_distribution(p.pm)(rng)) {
                ch1.os = mutate_os(ch1.os, c, rng);
                ch1.ms = mutate_ms(ch1.ms, pt, c, s, rng);
                ch1.fitness = INT_MAX;
            }
            if (std::bernoulli_distribution(p.pm)(rng)) {
                ch2.os = mutate_os(ch2.os, c, rng);
                ch2.ms = mutate_ms(ch2.ms, pt, c, s, rng);
                ch2.fitness = INT_MAX;
            }
            evaluate(ch1, pt, c, s);
            new_pop.push_back(std::move(ch1));
            if ((int)new_pop.size() < p.pop_size) {
                evaluate(ch2, pt, c, s);
                new_pop.push_back(std::move(ch2));
            }
        }

        // Phase 2: apply TS to all individuals of the new population.
        // Paper: maxTSIterSize = ts_iter_base * (Gen / maxGen).
        // This keeps TS short in early generations and stronger in later ones.
        int ts_iters = std::max(1, p.ts_iter_base * gen / p.max_gen);
        int ts_jobs = (int)new_pop.size();
        if (ts_jobs > 0) {
            int worker_count = std::min(n_threads, ts_jobs);
            std::atomic<int> next_idx{0};
            std::vector<std::thread> threads;
            threads.reserve(worker_count);
            for (int tid = 0; tid < worker_count; ++tid) {
                threads.emplace_back([&new_pop, &pt, ts_iters, &rng,
                                      &c, &ts_states, &next_idx, end = (int)new_pop.size(), tid]() {
                    while (true) {
                        int k = next_idx.fetch_add(1, std::memory_order_relaxed);
                        if (k >= end) break;
                        new_pop[k] = tabu_search(new_pop[k], pt, ts_iters, c, ts_states[tid], rng);
                    }
                });
            }
            for (auto& th : threads) th.join();
        }

        pop = std::move(new_pop);

        auto& gen_best = *std::min_element(pop.begin(), pop.end(),
            [](auto& a, auto& b){ return a.fitness < b.fitness; });

        if (gen_best.fitness < best_ever.fitness) { best_ever = gen_best; stagnant = 0; }
        else ++stagnant;

        if (p.verbose) {
            double elapsed = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - run_start).count();
            std::cout << "[HA] Gen " << std::setw(3) << gen << "/" << p.max_gen
                      << " | gen best: "    << std::setw(5) << gen_best.fitness
                      << " | global best: " << std::setw(5) << best_ever.fitness
                      << " | TS iters: "    << std::setw(4) << ts_iters
                      << " | stagnant: "    << stagnant
                      << " | elapsed: "     << std::fixed << std::setprecision(2) << elapsed << " s\n"
                      << std::flush;
        }
    }
    return best_ever;
}

// ─── Instance Loader ─────────────────────────────────────────────────────────

static PT parse_fjsp(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::cerr << "Cannot open: " << path << "\n"; std::exit(1); }
    std::string header_line;
    std::getline(f, header_line);
    while (header_line.empty() && std::getline(f, header_line)) {}
    if (header_line.empty()) {
        std::cerr << "Empty instance file: " << path << "\n";
        std::exit(1);
    }
    std::istringstream hs(header_line);
    int n_jobs, n_machines;
    double avg = 0.0;
    if (!(hs >> n_jobs >> n_machines)) {
        std::cerr << "Invalid header in: " << path << "\n";
        std::exit(1);
    }
    hs >> avg; // optional third value
    PT pt(n_jobs);
    for (int j = 0; j < n_jobs; ++j) {
        int n_ops; f >> n_ops; pt[j].resize(n_ops);
        for (int o = 0; o < n_ops; ++o) {
            int n_opts; f >> n_opts; pt[j][o].resize(n_opts);
            for (int k = 0; k < n_opts; ++k) f >> pt[j][o][k].first >> pt[j][o][k].second;
        }
    }
    return pt;
}

// ─── Entry Point ─────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ha_solver <instance.txt> [--pop-size N] [--max-gen N]\n"
                  << "                               [--stagnant N] [--ts-base N]\n"
                  << "                               [--seed N] [--quiet]\n";
        return 1;
    }
    std::string path = argv[1];
    Params p; unsigned seed = DEFAULT_SEED;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--pop-size"  && i+1 < argc) p.pop_size     = std::stoi(argv[++i]);
        else if (a == "--max-gen"   && i+1 < argc) p.max_gen      = std::stoi(argv[++i]);
        else if (a == "--stagnant"  && i+1 < argc) p.max_stagnant = std::stoi(argv[++i]);
        else if (a == "--ts-base"   && i+1 < argc) p.ts_iter_base = std::stoi(argv[++i]);
        else if (a == "--seed"      && i+1 < argc) seed           = std::stoul(argv[++i]);
        else if (a == "--quiet")                    p.verbose      = false;
    }

    PT pt = parse_fjsp(path);
    int n_jobs = (int)pt.size(), n_ops = 0;
    for (auto& j : pt) n_ops += (int)j.size();

    std::cout << "Instance : " << path << "\n"
              << "Jobs     : " << n_jobs << "\n"
              << "Ops      : " << n_ops  << "\n"
              << "Seed     : " << seed   << "\n\n";

    std::mt19937 rng(seed);
    auto t0 = std::chrono::steady_clock::now();
    Individual best = hybrid_ga_ts(pt, p, rng);
    double elapsed = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0).count();

    std::cout << "\n=== Result ===\n"
              << "Makespan : " << best.fitness << "\n"
              << "Time     : " << std::fixed << std::setprecision(2) << elapsed << " s\n";
    return 0;
}
