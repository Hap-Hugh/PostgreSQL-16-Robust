//
// Created by Xuan Chen on 2025/11/21.
// Modified by Xuan Chen on 2025/12/05.
//

/*
 * calc_plan_similarity.c
 *
 * Jensen–Shannon based plan-space reduction via greedy K-center,
 * following the Python implementation of:
 *
 *   k_center_greedy(self.costCollection, R, JS_distance, first_plan=None)
 *
 * For each Path in cand_list we assume there is a per-sample total
 * cost vector stored in a Sample struct:
 *
 *   struct Sample {
 *       int    sample_count;
 *       double sample[DIST_MAX_SAMPLE];   // log-based total cost per sample
 *   };
 *
 * and that each Path has a pointer to such a Sample. Adjust the field
 * access (path->samples) to your actual custom Path type.
 */

#include "postgres.h"

#include <math.h>
#include <float.h>

#include "nodes/pg_list.h"
#include "optimizer/paths.h"
#include "utils/elog.h"
#include "utils/memutils.h"
#include "optimizer/pathstrategy.h"

#define PRUNE_TOLERANCE_FACTOR 2.0

/* ----------------------------------------------------------------
 * Internal helpers: KL, JS, etc.
 * ----------------------------------------------------------------
 */

/*
 * Compute KL(p || q) where p and q are raw non-negative vectors
 * of length sample_count. This matches the Python logic:
 *
 *   - Normalize p and q separately to sum to 1.
 *   - Add a small epsilon to avoid log(0).
 *
 * We do not require p and q to be pre-normalized; we treat them
 * as arbitrary non-negative cost vectors.
 */
static double
kl_divergence(const double *p, const double *q, const int sample_count) {
    double sum_p = 0.0;
    double sum_q = 0.0;
    double kl = 0.0;

    /* 1) Compute sums */
    for (int i = 0; i < sample_count; ++i) {
        sum_p += p[i];
        sum_q += q[i];
    }

    /* Guard against degenerate all-zero vectors */
    if (sum_p <= 0.0 || sum_q <= 0.0) {
        return 0.0;
    }

    /* 2) Normalize, add epsilon, and accumulate KL */
    for (int i = 0; i < sample_count; ++i) {
        const double EPS = 1e-10;
        double pi = p[i] / sum_p;
        double qi = q[i] / sum_q;

        pi += EPS;
        qi += EPS;

        kl += pi * log(pi / qi);
    }

    return kl;
}

/*
 * Jensen–Shannon distance between two cost vectors p and q.
 *
 * This mirrors the Python implementation:
 *
 *   m = 0.5 * (p + q)
 *   js_div = 0.5 * KL(p || m) + 0.5 * KL(q || m)
 *   return sqrt(js_div)
 *
 * Note: p and q are raw non-negative cost vectors; each KL()
 * call normalizes its inputs internally, just like the Python code.
 */
static double
js_distance(const double *p, const double *q, const int sample_count) {
    double *m = palloc(sizeof(double) * sample_count);

    /* Compute mixture distribution m = 0.5 * (p + q) */
    for (int i = 0; i < sample_count; ++i) {
        m[i] = 0.5 * (p[i] + q[i]);
    }

    double js_div = 0.5 * kl_divergence(p, m, sample_count)
                    + 0.5 * kl_divergence(q, m, sample_count);

    if (js_div < 0.0) {
        js_div = 0.0;
    }

    const double result = sqrt(js_div);
    pfree(m);

    return result;
}

/* ----------------------------------------------------------------
 * Main function: JS-based K-center over Path cost samples
 * ----------------------------------------------------------------
 */

/*
 * calc_plan_similarity
 *
 * Use Jensen–Shannon distance over per-sample total cost vectors
 * to perform greedy K-center selection on the candidate Paths,
 * mirroring the Python code:
 *
 *   centers, assignments = k_center_greedy(costCollection, R, JS_distance, first_plan=None)
 *
 * The result is encoded in rank_arr[].score:
 *
 *   - score = 1.0 for center plans (selected representatives)
 *   - score = 0.0 for non-center plans (to be pruned if caller wishes)
 *
 * Caller is expected to use rank_arr to filter/sort candidates.
 *
 * min_envelope unused here for pruning paths.
 */
extern void
calc_plan_similarity(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    ListCell *lc;
    /* indexable view of cand_list */
    /* [cand_count][sample_count] */
    /* indices of selected center plans */
    /* per-plan distance to nearest center */
    /* R: number of centers we want */

    /* ----------------------------------------------------------
     * 0. Basic setup and sanity checks
     * ----------------------------------------------------------
     */

    int cand_count = list_length((List *) cand_list);
    if (cand_count <= 0) {
        elog(ERROR, "calc_plan_similarity: empty candidate list");
    }

    /* Determine number of centers R; clamp to [1, cand_count]. */
    int num_centers = retain_path_limit;
    if (num_centers <= 0) {
        num_centers = 1;
    }

    /* ----------------------------------------------------------
     * 1. Prune and materialize cand_list into an indexable array
     *    and fill rank_arr.path
     * ----------------------------------------------------------
     */

    Cost max_total_cost_thresh = 0.0;
    Assert(min_envelope != NULL);
    for (int i = 0; i < sample_count; ++i) {
        max_total_cost_thresh = Max(max_total_cost_thresh, min_envelope[i]);
    }
    max_total_cost_thresh = max_total_cost_thresh * PRUNE_TOLERANCE_FACTOR;

    Path **path_array = palloc(sizeof(Path *) * cand_count);

    int writer = 0;
    foreach(lc, cand_list) {
        Path *path = lfirst(lc);

        Cost cur_min_total_cost = DBL_MAX;
        const Sample *cur_total_cost_sample = path->total_cost_sample;
        const int cur_sample_count = cur_total_cost_sample->sample_count;
        for (int cur_sample_idx = 0; cur_sample_idx < cur_sample_count; ++cur_sample_idx) {
            cur_min_total_cost = Min(cur_min_total_cost, cur_total_cost_sample->sample[cur_sample_idx]);
        }
        if (cur_min_total_cost > max_total_cost_thresh) {
            continue; /* We skip bad paths, but we don't remove it now. */
        }

        path_array[writer] = path;
        rank_arr[writer].path = path;
        rank_arr[writer].score = 0.0; /* Will be filled after K-center */
        ++writer;
    }
    Assert(writer <= cand_count);

    /* Now we have no paths, but we need at least one path. */
    if (writer == 0) {
        // elog(LOG, "Cannot find any proper paths using similarity score, but we still need one.");

        Path *best_path = NULL;
        Cost min_total_cost = DBL_MAX;
        foreach(lc, cand_list) {
            Path *path = lfirst(lc);
            const Cost cur_total_cost = path->total_cost;
            if (cur_total_cost < min_total_cost) {
                min_total_cost = cur_total_cost;
                best_path = path;
            }
        }

        path_array[writer] = best_path;
        rank_arr[writer].path = best_path;
        rank_arr[writer].score = 0.0; /* Will be filled after K-center */
        writer = 1;
    }
    cand_count = writer; /* Actual cand count */

    if (num_centers > cand_count) {
        num_centers = cand_count;
    }

    /* ----------------------------------------------------------
     * 2. Build cost_matrix: one cost vector per Path
     *    (this corresponds to self.costCollection in Python)
     * ----------------------------------------------------------
     *
     * cost_matrix[i][s] = total cost of plan i at sample s.
     * We assume each Path has a Sample* with at least sample_count
     * entries. Adjust Path->samples access as needed.
     */

    double **cost_matrix = palloc(sizeof(double *) * cand_count);

    for (int i = 0; i < cand_count; ++i) {
        const Sample *sm = path_array[i]->total_cost_sample;

        if (sm == NULL) {
            elog(ERROR, "calc_plan_similarity: path %d has no Sample data", i);
        }

        cost_matrix[i] = (double *) palloc(sizeof(double) * sample_count);

        /*
         * Case 1: exact match → copy directly
         */
        if (sm->sample_count == sample_count) {
            for (int sc = 0; sc < sample_count; ++sc) {
                cost_matrix[i][sc] = sm->sample[sc];
            }
        }
        /*
         * Case 2: only 1 sample → broadcast that single value
         */
        else if (sm->sample_count == 1) {
            const double v = sm->sample[0];
            for (int sc = 0; sc < sample_count; ++sc) {
                cost_matrix[i][sc] = v;
            }
        }
        /*
         * Case 3: illegal → insufficient samples but not 1
         */
        else {
            elog(ERROR,
                 "calc_plan_similarity: path %d has %d samples, required %d "
                 "(only sample_count==1 is allowed for broadcasting)",
                 i, sm->sample_count, sample_count);
        }
    }

    /* ----------------------------------------------------------
     * 3. Run greedy K-center on cost_matrix using JS distance
     *    (this mirrors k_center_greedy(costCollection, R, JS_distance))
     * ----------------------------------------------------------
     */
    int *centers = palloc(sizeof(int) * num_centers);
    double *min_dist = palloc(sizeof(double) * cand_count);

    /* 3.1 Initialize: pick the first center based on the minimum expected total cost. */
    int first_center = -1;
    Cost min_total_cost = DBL_MAX;
    for (int idx = 0; idx < cand_count; ++idx) {
        const Path *cur_path = path_array[idx];
        const Cost cur_total_cost = cur_path->total_cost;
        if (cur_total_cost < min_total_cost) {
            min_total_cost = cur_total_cost;
            first_center = idx;
        }
    }
    centers[0] = first_center;

    for (int idx = 0; idx < cand_count; ++idx) {
        min_dist[idx] = DBL_MAX;
    }

    /* 3.2 Iteratively select the remaining centers */
    for (int k = 1; k < num_centers; ++k) {
        const int last_center = centers[k - 1];
        int farthest_idx = -1;
        double farthest_dist = -1.0;

        /* Update each candidate's distance to the nearest selected center */
        for (int idx = 0; idx < cand_count; ++idx) {
            const double dist = js_distance(
                cost_matrix[idx], cost_matrix[last_center], sample_count
            );

            if (dist < min_dist[idx])
                min_dist[idx] = dist;

            /* Track the point farthest from its closest center */
            if (min_dist[idx] > farthest_dist) {
                farthest_dist = min_dist[idx];
                farthest_idx = idx;
            }
        }

        centers[k] = farthest_idx;
    }

    /* ----------------------------------------------------------
     * 4. Assign scores based on whether the path is selected
     *    as a center.
     *
     * Python code *drops* non-center plans by slicing arrays:
     *   self.costCollection   = [...]
     *   self.penaltyCollection = [...]
     *   self.plan_list        = [...]
     *
     * Here we encode the same semantics via scores:
     *   - center plans:   score = 1.0
     *   - non-center:     score = 0.0
     *
     * The caller can then keep only center plans (score > 0).
     * ----------------------------------------------------------
     */

    /* Default: all non-centers */
    for (int idx = 0; idx < cand_count; ++idx) {
        rank_arr[idx].score = 0.0;
    }

    /* Mark selected centers */
    for (int k = 0; k < num_centers; ++k) {
        const int center_idx = centers[k];

        Assert(center_idx >= 0 && center_idx < cand_count);
        rank_arr[center_idx].score = 1.0;
    }

    /* ----------------------------------------------------------
     * 5. Cleanup temporary structures (optional in short-lived
     *    contexts, but good practice in long-lived sessions)
     * ----------------------------------------------------------
     */

    for (int i = 0; i < cand_count; ++i) {
        pfree(cost_matrix[i]);
    }

    pfree(cost_matrix);
    pfree(centers);
    pfree(min_dist);
    pfree(path_array);
}
