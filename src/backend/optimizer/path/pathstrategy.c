//
// Created by Xuan Chen on 2025/10/31.
// Modified by Xuan Chen on 2025/11/9.
// Modified by Xuan Chen on 2025/11/12.
// Modified by Xuan Chen on 2025/12/22.
//

#define PENALTY_TOLERANCE_FACTOR 1.2

#include "optimizer/pathstrategy.h"

#include <float.h>

typedef double (*basic_select_path_strategy_impl)(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_envelope,
    int effective
);

static double calc_worst_penalty_impl(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_envelope,
    int effective
);

static double calc_expected_penalty_impl(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_envelope,
    int effective
);

static double calc_worst_total_cost_impl(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_envelope,
    int effective
);

static double calc_expected_total_cost_impl(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_envelope,
    int effective
);

static double calc_worst_startup_cost_impl(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_envelope,
    int effective
);

static double calc_expected_startup_cost_impl(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_envelope,
    int effective
);

static double calc_worst_penalty_with_std_impl(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_envelope,
    int effective
);

static double calc_expected_penalty_with_std_impl(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_envelope,
    int effective
);

static double calc_4th_worst_penalty_impl(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_envelope,
    int effective
);

static void basic_select_path_strategy_helper(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count,
    basic_select_path_strategy_impl basic_select_path_strategy_impl_func
);

extern void calc_worst_penalty(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    basic_select_path_strategy_helper(
        cand_list, rank_arr, min_envelope, sample_count, calc_worst_penalty_impl
    );
}

extern void calc_expected_penalty(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    basic_select_path_strategy_helper(
        cand_list, rank_arr, min_envelope, sample_count, calc_expected_penalty_impl
    );
}

extern void calc_worst_total_cost(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    basic_select_path_strategy_helper(
        cand_list, rank_arr, min_envelope, sample_count, calc_worst_total_cost_impl
    );
}

extern void calc_expected_total_cost(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    basic_select_path_strategy_helper(
        cand_list, rank_arr, min_envelope, sample_count, calc_expected_total_cost_impl
    );
}

extern void calc_worst_startup_cost(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    basic_select_path_strategy_helper(
        cand_list, rank_arr, min_envelope, sample_count, calc_worst_startup_cost_impl
    );
}

extern void calc_expected_startup_cost(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    basic_select_path_strategy_helper(
        cand_list, rank_arr, min_envelope, sample_count, calc_expected_startup_cost_impl
    );
}

static void basic_select_path_strategy_helper(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count,
    const basic_select_path_strategy_impl basic_select_path_strategy_impl_func
) {
    int idx = 0;
    ListCell *lc;
    foreach(lc, cand_list) {
        Path *path = lfirst(lc);
        const Sample *startup_cost_sample = path->startup_cost_sample;
        const Sample *total_cost_sample = path->total_cost_sample;

        Assert(startup_cost_sample != NULL);
        Assert(startup_cost_sample->sample_count >= 0 &&
            startup_cost_sample->sample_count <= DIST_MAX_SAMPLE);

        Assert(total_cost_sample != NULL);
        Assert(total_cost_sample->sample_count >= 0 &&
            total_cost_sample->sample_count <= DIST_MAX_SAMPLE);

        int effective = Min(startup_cost_sample->sample_count, total_cost_sample->sample_count);
        effective = Min(effective, sample_count);
        Assert(effective >= 0);

        double score_val;
        if (effective == 0) {
            /* No samples => automatically worst */
            score_val = DBL_MAX;
        } else {
            score_val = basic_select_path_strategy_impl_func(
                startup_cost_sample, total_cost_sample, min_envelope, effective
            );
        }

        rank_arr[idx].path = path;
        rank_arr[idx].score = score_val;
        idx++;
    }
    Assert(idx == cand_count);
}

/*
 * calc_worst_penalty_impl
 *
 * Compute the worst (maximum) penalty among samples.
 *
 * Penalty is measured relative to the global minimum vector:
 *
 *   penalty(j) = (v(j) > min_envelope(j) * PENALTY_TOLERANCE_FACTOR)
 *                  ? (v(j) - min_envelope(j))
 *                  : 0.0
 *
 * This metric reflects the worst-case deviation from the baseline.
 * Zero means the path is comparable to the best observed path.
 */
double calc_worst_penalty_impl(
    const Sample *startup_cost_sample, /* unused */
    const Sample *total_cost_sample,
    const double *min_envelope,
    const int effective
) {
    /* Score >= 0 always */
    double worst_penalty = 0.0;

    Assert(total_cost_sample != NULL);
    Assert(min_envelope != NULL);
    Assert(effective > 0);

    for (int i = 0; i < error_sample_count; ++i) {
        const double cur_thresh = min_envelope[i] * PENALTY_TOLERANCE_FACTOR;
        const double cur_sample
                = (effective == 1)
                      ? total_cost_sample->sample[0]
                      : total_cost_sample->sample[i];
        const double cur_penalty
                = (cur_sample > cur_thresh)
                      ? (cur_sample - min_envelope[i])
                      : 0.0;

        if (cur_penalty > worst_penalty)
            worst_penalty = cur_penalty;
    }
    return worst_penalty;
}

/*
 * calc_expected_penalty_impl
 *
 * Compute the average penalty across all samples.
 *
 * This metric reflects expected (mean) deviation from the baseline,
 * instead of worst-case deviation.
 */
double calc_expected_penalty_impl(
    const Sample *startup_cost_sample, /* unused */
    const Sample *total_cost_sample,
    const double *min_envelope,
    const int effective
) {
    double expected_penalty = 0.0;

    Assert(total_cost_sample != NULL);
    Assert(min_envelope != NULL);
    Assert(effective > 0);

    for (int i = 0; i < error_sample_count; ++i) {
        const double cur_thresh = min_envelope[i] * PENALTY_TOLERANCE_FACTOR;
        const double cur_sample
                = (effective == 1)
                      ? total_cost_sample->sample[0]
                      : total_cost_sample->sample[i];
        const double cur_penalty
                = (cur_sample > cur_thresh)
                      ? (cur_sample - min_envelope[i])
                      : 0.0;
        expected_penalty += cur_penalty;
    }
    return expected_penalty / (double) error_sample_count;
}

/*
 * Compute standard deviation of penalty over the first `effective` samples.
 *
 * Penalty is measured relative to the global minimum vector:
 *
 *   penalty(j) = (v(j) > min_envelope(j) * PENALTY_TOLERANCE_FACTOR)
 *                  ? (v(j) - min_envelope(j))
 *                  : 0.0
 */
static double compute_std_penalty_sample(
    const Sample *total_cost_sample,
    const double *min_envelope,
    const int effective
) {
    if (effective <= 1)
        return 0.0;

    Assert(total_cost_sample != NULL);
    Assert(min_envelope != NULL);

    /* Welford's online algorithm over penalty(j) */
    double mean = 0.0;
    double M2 = 0.0;
    int n = 0;

    for (int i = 0; i < effective; ++i) {
        const double cur_thresh = min_envelope[i] * PENALTY_TOLERANCE_FACTOR;
        const double cur_sample = total_cost_sample->sample[i];
        const double penalty =
                (cur_sample > cur_thresh) ? (cur_sample - min_envelope[i]) : 0.0;

        const double x = penalty;
        n += 1;
        const double delta = x - mean;
        mean += delta / n;
        const double delta2 = x - mean;
        M2 += delta * delta2;
    }

    /* Population standard deviation over `effective` elements */
    const double variance = M2 / (double) effective;
    return sqrt(variance);
}

/*
 * calc_worst_penalty_with_std_impl
 *
 * Compute the worst (maximum) penalty among samples, then add the STD of the
 * penalty across the same `effective` samples.
 *
 * Penalty is measured relative to the global minimum vector:
 *
 *   penalty(j) = (v(j) > min_envelope(j) * PENALTY_TOLERANCE_FACTOR)
 *                  ? (v(j) - min_envelope(j))
 *                  : 0.0
 *
 * Final score:
 *   return max_j penalty(j)  +  std(penalty[0..effective-1])
 *
 * Zero penalty means the path is comparable to the best observed path; the
 * additional STD term rewards stability (lower variance).
 */
double calc_worst_penalty_with_std_impl(
    const Sample *startup_cost_sample, /* unused */
    const Sample *total_cost_sample,
    const double *min_envelope,
    const int effective
) {
    /* Score >= 0 always (penalty >= 0, std >= 0) */
    double worst_penalty = 0.0;

    Assert(total_cost_sample != NULL);
    Assert(min_envelope != NULL);
    Assert(effective > 0);

    for (int i = 0; i < error_sample_count; ++i) {
        const double cur_thresh = min_envelope[i] * PENALTY_TOLERANCE_FACTOR;
        const double cur_sample
                = (effective == 1)
                      ? total_cost_sample->sample[0]
                      : total_cost_sample->sample[i];
        const double cur_penalty
                = (cur_sample > cur_thresh)
                      ? (cur_sample - min_envelope[i])
                      : 0.0;

        if (cur_penalty > worst_penalty)
            worst_penalty = cur_penalty;
    }

    /* Add STD of penalty */
    const double std_penalty =
            compute_std_penalty_sample(total_cost_sample, min_envelope, effective);
    return worst_penalty + std_penalty;
}

/*
 * calc_expected_penalty_with_std_impl
 *
 * Compute the mean penalty across samples, then add the STD of the
 * penalty across the same `effective` samples.
 *
 * Final score:
 *   return mean_j penalty(j) + std(penalty[0..effective-1])
 */
double calc_expected_penalty_with_std_impl(
    const Sample *startup_cost_sample, /* unused */
    const Sample *total_cost_sample,
    const double *min_envelope,
    const int effective
) {
    double sum_penalty = 0.0;

    Assert(total_cost_sample != NULL);
    Assert(min_envelope != NULL);
    Assert(effective > 0);

    for (int i = 0; i < error_sample_count; ++i) {
        const double cur_thresh = min_envelope[i] * PENALTY_TOLERANCE_FACTOR;
        const double cur_sample
                = (effective == 1)
                      ? total_cost_sample->sample[0]
                      : total_cost_sample->sample[i];
        const double cur_penalty
                = (cur_sample > cur_thresh)
                      ? (cur_sample - min_envelope[i])
                      : 0.0;
        sum_penalty += cur_penalty;
    }

    const double mean_penalty = sum_penalty / (double) error_sample_count;

    /* Add STD of penalty */
    const double std_penalty =
            compute_std_penalty_sample(total_cost_sample, min_envelope, effective);
    return mean_penalty + std_penalty;
}

/*
 * calc_4th_worst_penalty_impl
 *
 * Compute the 4th worst (4th largest) penalty among samples.
 *
 * Penalty is measured relative to the global minimum vector:
 *
 *   penalty(j) = (v(j) > min_envelope(j) * PENALTY_TOLERANCE_FACTOR)
 *                  ? (v(j) - min_envelope(j))
 *                  : 0.0
 *
 * We track the top 4 penalties in descending order while scanning.
 * If fewer than 4 samples exist, we return the smallest among the
 * collected penalties (i.e., the "Nth worst" where N = effective).
 */
double calc_4th_worst_penalty_impl(
    const Sample *startup_cost_sample, /* unused */
    const Sample *total_cost_sample,
    const double *min_envelope,
    const int effective
) {
    /* Keep the top 4 penalties in descending order:
     * top4[0] = worst (largest)
     * top4[3] = 4th worst
     */
    double top4[4] = {0.0, 0.0, 0.0, 0.0};
    int seen = 0; /* number of inserted samples, capped at 4 */

    Assert(total_cost_sample != NULL);
    Assert(min_envelope != NULL);
    Assert(effective > 0);

    for (int i = 0; i < error_sample_count; ++i) {
        const double thresh = min_envelope[i] * PENALTY_TOLERANCE_FACTOR;
        const double val
                = (effective == 1)
                      ? total_cost_sample->sample[0]
                      : total_cost_sample->sample[i];
        const double pen = (val > thresh) ? (val - min_envelope[i]) : 0.0;

        /* If we already have 4 penalties and this one is not better,
         * it cannot enter the top 4 list.
         */
        if (seen >= 4 && pen <= top4[3]) {
            continue;
        }

        /* Insert penalty into the descending list */
        if (pen > top4[0]) {
            top4[3] = top4[2];
            top4[2] = top4[1];
            top4[1] = top4[0];
            top4[0] = pen;
        } else if (pen > top4[1]) {
            top4[3] = top4[2];
            top4[2] = top4[1];
            top4[1] = pen;
        } else if (pen > top4[2]) {
            top4[3] = top4[2];
            top4[2] = pen;
        } else {
            /* pen goes into 4th position */
            top4[3] = pen;
        }

        if (seen < 4) {
            seen++;
        }
    }

    /* If we processed >= 4 samples, return the 4th worst.
     * Otherwise return the last valid one (i.e., "seen-th worst").
     */
    if (seen >= 4) {
        return top4[3];
    }
    return top4[seen - 1];
    /* seen >= 1 because effective > 0 */
}

/*
 * calc_worst_total_cost_impl
 *
 * Return the maximum value among total_cost samples.
 *
 * This corresponds to the worst-case total execution cost.
 */
double calc_worst_total_cost_impl(
    const Sample *startup_cost_sample, /* unused */
    const Sample *total_cost_sample,
    const double *min_envelope, /* unused */
    const int effective
) {
    double worst_total_cost = 0.0;

    Assert(total_cost_sample != NULL);
    Assert(effective > 0);

    for (int i = 0; i < error_sample_count; ++i) {
        const double cur_sample
                = (effective == 1)
                      ? total_cost_sample->sample[0]
                      : total_cost_sample->sample[i];
        if (cur_sample > worst_total_cost)
            worst_total_cost = cur_sample;
    }
    return worst_total_cost;
}

/*
 * calc_expected_total_cost_impl
 *
 * Return the average total cost across samples.
 *
 * This corresponds to expected execution cost.
 */
double calc_expected_total_cost_impl(
    const Sample *startup_cost_sample, /* unused */
    const Sample *total_cost_sample,
    const double *min_envelope, /* unused */
    const int effective
) {
    double expected_total_cost = 0.0;

    Assert(total_cost_sample != NULL);
    Assert(effective > 0);

    for (int i = 0; i < error_sample_count; ++i) {
        const double cur_sample
                = (effective == 1)
                      ? total_cost_sample->sample[0]
                      : total_cost_sample->sample[i];
        expected_total_cost += cur_sample;
    }

    return expected_total_cost / (double) error_sample_count;
}

/*
 * calc_worst_startup_cost_impl
 *
 * Return the maximum startup cost among samples.
 *
 * This is the worst-case startup cost, useful for pipelining latency mode.
 */
double calc_worst_startup_cost_impl(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample, /* unused */
    const double *min_envelope, /* unused */
    const int effective
) {
    double worst_startup_cost = 0.0;

    Assert(startup_cost_sample != NULL);
    Assert(effective > 0);

    for (int i = 0; i < error_sample_count; ++i) {
        const double cur_sample
                = (effective == 1)
                      ? startup_cost_sample->sample[0]
                      : startup_cost_sample->sample[i];
        if (cur_sample > worst_startup_cost)
            worst_startup_cost = cur_sample;
    }
    return worst_startup_cost;
}

/*
 * calc_expected_startup_cost_impl
 *
 * Return the average startup cost across samples.
 *
 * This reflects expected first-tuple latency for pipelined plans.
 */
double calc_expected_startup_cost_impl(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample, /* unused */
    const double *min_envelope, /* unused */
    const int effective
) {
    double expected_startup_cost = 0.0;

    Assert(startup_cost_sample != NULL);
    Assert(effective > 0);

    for (int i = 0; i < error_sample_count; ++i) {
        const double cur_sample
                = (effective == 1)
                      ? startup_cost_sample->sample[0]
                      : startup_cost_sample->sample[i];
        expected_startup_cost += cur_sample;
    }

    return expected_startup_cost / (double) error_sample_count;
}

static bool covers_under_eps(
    const double path_cost,
    const double opt_cost,
    const double eps
) {
    if (opt_cost == 0.0) {
        return path_cost == 0.0; /* exact match when optimum is 0 */
    }
    return path_cost <= opt_cost * (1.0 + eps);
}

/*
 * Helper to fetch the "cost" at sample index s for a given Path.
 *
 * Assumptions per your constraints:
 *   - sample_count > 1
 *   - path->rows_sample != NULL
 *
 * Rules implemented:
 *   - If samp->sample_count == sample_count: return samp->sample[s].
 *   - If samp->sample_count == 1:           treat as constant across all s; return samp->sample[0].
 *   - Otherwise: return false (caller may choose to skip this (path,s) pair).
 *
 * Returns true on success and writes *out_cost; otherwise false.
 */
static bool fetch_cost_at(
    const Path *path,
    const int s,
    const int sample_count,
    double *out_cost
) {
    const Sample *samp = path->total_cost_sample;

    /* Expected by the caller */
    if (samp == NULL)
        return false;

    if (samp->sample_count == sample_count) {
        /* Per-sample vector */
        *out_cost = samp->sample[s];
        return true;
    }
    if (samp->sample_count == 1) {
        /* Constant across all s */
        *out_cost = samp->sample[0];
        return true;
    }
    /* Incompatible shape (should not happen) */
    return false;
}

/*
 * calc_robust_coverage
 *
 * Greedy robust coverage with scoring:
 *   - Compute per-sample optimum cost opt[s] across all candidates.
 *   - While uncovered samples remain:
 *       use a greedy loop to pick candidates that bring the largest *new*
 *       coverage under eps, and log each pick (gain, cumulative coverage).
 *   - After the greedy loop, compute for every candidate its *total*
 *     robust coverage (over all samples, independent of greedy order),
 *     and store that as PathRank.score.
 *
 * Inputs:
 *   - cand_list: List* of Path*; each Path has rows_sample != NULL
 *   - rank_arr : preallocated array of length nplans; we fill .path and .score
 *   - min_envelope: unused
 *   - sample_count: > 1
 *
 * Outputs (via rank_arr):
 *   - rank_arr[i].path  = i-th candidate Path
 *   - rank_arr[i].score = robust coverage count
 *                         (# of samples whose cost is within eps of per-sample optimum)
 *
 * Logging (LOG level):
 *   - After computing opt[s], we log a brief summary.
 *   - Each greedy pick logs: pick index, plan index, gain, cumulative coverage.
 *   - Final summary logs total coverage and #picks.
 */
extern void calc_robust_coverage(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    const int nplans = list_length((List *) cand_list);
    const double eps = ROBUST_EPS_DEFAULT;

    if (nplans <= 0) {
        elog(LOG, "[robust_cover] No candidates.");
        return;
    }

    /* --- Materialize candidates into an indexable array --- */
    Path **paths = (Path **) palloc(sizeof(Path *) * nplans);
    {
        int i = 0;
        ListCell *lc;
        foreach(lc, (List *) cand_list)
            paths[i++] = (Path *) lfirst(lc);
    }

    /* --- Working arrays --- */
    double *opt = (double *) palloc(sizeof(double) * sample_count);
    bool *uncovered = (bool *) palloc(sizeof(bool) * sample_count);
    bool *selected = (bool *) palloc0(sizeof(bool) * nplans);

    /* record greedy pick order (indices into paths[]) */
    int *pick_order = (int *) palloc(sizeof(int) * nplans);
    int pick_cnt = 0;

    /* store per-plan total coverage (score) */
    double *plan_cov = (double *) palloc(sizeof(double) * nplans);

    /* --- 1) Fill in opt[] and uncovered[] --- */
    for (int s = 0; s < sample_count; s++) {
        opt[s] = min_envelope[s];
        uncovered[s] = true;
    }

    /* --- 2) Greedy selection loop (defines OUTPUT ORDER) --- */
    int covered_total = 0;

    while (covered_total < sample_count) {
        int best_plan = -1;
        int best_gain = 0;

        for (int i = 0; i < nplans; i++) {
            if (selected[i])
                continue;

            const Path *path = paths[i];
            int gain = 0;

            for (int s = 0; s < sample_count; s++) {
                if (!uncovered[s])
                    continue;

                double v;
                if (!fetch_cost_at(path, s, sample_count, &v))
                    continue;

                if (covers_under_eps(v, opt[s], eps))
                    gain++;
            }

            /* tie-break: keep first encountered (because '>' not '>=') */
            if (gain > best_gain) {
                best_gain = gain;
                best_plan = i;
            }
        }

        if (best_plan < 0 || best_gain == 0) {
            elog(LOG, "[robust_cover] No more gains (best_gain=%d). Stop greedy.", best_gain);
            break;
        }

        /* commit */
        {
            const Path *path = paths[best_plan];
            selected[best_plan] = true;
            pick_order[pick_cnt++] = best_plan;

            int newly = 0;
            for (int s = 0; s < sample_count; s++) {
                if (!uncovered[s])
                    continue;

                double v;
                if (!fetch_cost_at(path, s, sample_count, &v))
                    continue;

                if (covers_under_eps(v, opt[s], eps)) {
                    uncovered[s] = false;
                    newly++;
                }
            }
            covered_total += newly;

            elog(LOG, "[robust_cover] pick #%d plan=%d newly=%d covered=%d/%d",
                 pick_cnt, best_plan, newly, covered_total, sample_count);
        }
    }

    /* --- 3) Compute total coverage score for EVERY candidate (independent of greedy order) --- */
    for (int i = 0; i < nplans; i++) {
        const Path *path = paths[i];
        int coverage = 0;

        for (int s = 0; s < sample_count; s++) {
            double v;
            if (!fetch_cost_at(path, s, sample_count, &v))
                continue;

            if (covers_under_eps(v, opt[s], eps))
                coverage++;
        }
        plan_cov[i] = (double) coverage;
    }

    /* --- 4) Fill rank_arr in GREEDY ORDER, but score = total coverage --- */
    int out = 0;

    /* first: greedy-picked plans, in pick order */
    for (int t = 0; t < pick_cnt; t++) {
        const int i = pick_order[t];
        rank_arr[out].path = paths[i];
        rank_arr[out].score = plan_cov[i];
        out++;
    }

    /* then: the rest (not selected), in original cand_list order for determinism */
    for (int i = 0; i < nplans; i++) {
        if (selected[i])
            continue;
        rank_arr[out].path = paths[i];
        rank_arr[out].score = plan_cov[i];
        out++;
    }

    /* --- cleanup --- */
    pfree(paths);
    pfree(opt);
    pfree(uncovered);
    pfree(selected);
    pfree(pick_order);
    pfree(plan_cov);
}

static unsigned int rng_state = 123456789u;

static unsigned int xorshift32(void) {
    unsigned int x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

static double rng_uniform01(void) {
    return xorshift32() / (double) UINT32_MAX;
}

extern void calc_random_score(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    int idx = 0;
    ListCell *lc;
    foreach(lc, cand_list) {
        Path *path = lfirst(lc);
        rank_arr[idx].path = path;
        rank_arr[idx].score = rng_uniform01();
        idx++;
    }
    Assert(idx == cand_count);
}

extern void calc_postgres_original_score(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    elog(ERROR, "`calc_postgres_original_score` is not implemented.");
}

extern void calc_jointype_based_score(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    elog(ERROR, "`calc_jointype_based_score` is not implemented.");
}

extern void calc_worst_penalty_with_std(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    basic_select_path_strategy_helper(
        cand_list, rank_arr, min_envelope, sample_count, calc_worst_penalty_with_std_impl
    );
}

extern void calc_expected_penalty_with_std(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    basic_select_path_strategy_helper(
        cand_list, rank_arr, min_envelope, sample_count, calc_expected_penalty_with_std_impl
    );
}

extern void calc_4th_worst_penalty(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    basic_select_path_strategy_helper(
        cand_list, rank_arr, min_envelope, sample_count, calc_4th_worst_penalty_impl
    );
}

extern void calc_retention_set(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    elog(ERROR, "`calc_retention_set` cannot be called in this way.");
}

extern void calc_jointype_based_mep(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    elog(ERROR, "`calc_jointype_based_mep` cannot be called in this way.");
}

extern void calc_jointype_based_mec(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    elog(ERROR, "`calc_jointype_based_mec` cannot be called in this way.");
}

/* Global strategy array */
path_strategy path_strategy_funcs[17] = {
    [0] = calc_worst_penalty,
    [1] = calc_expected_penalty,
    [2] = calc_worst_total_cost,
    [3] = calc_expected_total_cost,
    [4] = calc_worst_startup_cost,
    [5] = calc_expected_startup_cost,
    [6] = calc_robust_coverage,
    [7] = calc_random_score,
    [8] = calc_postgres_original_score,
    [9] = calc_jointype_based_score,
    [10] = calc_worst_penalty_with_std,
    [11] = calc_expected_penalty_with_std,
    [12] = calc_4th_worst_penalty,
    [13] = calc_plan_similarity,
    [14] = calc_retention_set,
    [15] = calc_jointype_based_mep,
    [16] = calc_jointype_based_mec,
};
