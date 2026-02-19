//
// Created by Xuan Chen on 2025/10/18.
// Modified by Xuan Chen on 2025/10/20.
// Modified by Xuan Chen on 2025/10/29.
// Modified by Xuan Chen on 2025/10/31.
// Modified by Xuan Chen on 2025/11/1.
// Modified by Xuan Chen on 2025/11/23.
// Modified by Xuan Chen on 2025/12/06.
// Modified by Xuan Chen on 2025/12/31.
//

#include "optimizer/addpath.h"
#include "optimizer/pathstrategy.h"
#include <float.h>

/* Forward Declarations (opaque to this file) */
typedef struct Sample Sample;
typedef struct ErrorProfileRaw ErrorProfileRaw;
typedef struct ErrorSampleParams ErrorSampleParams;
typedef struct ErrorProfile ErrorProfile;

/* ----------------------------------------------------------------------------
 * Helpers
 * --------------------------------------------------------------------------*/

/* --------- Fixed-size max-heap (size<=k) for PathRank indices by score ASC --
 * We keep the k *smallest* scores, so top of heap is the *largest* among kept.
 * Comparator: greater(a,b) if a should be closer to root in a MAX-heap.
 * Ties broken by Path* pointer for determinism.
 */

static bool
rank_idx_greater_by_score(const int ia, const int ib, const PathRank *arr) {
    const PathRank *a = &arr[ia];
    const PathRank *b = &arr[ib];
    if (a->score > b->score) return true;
    if (a->score < b->score) return false;
    /* deterministic pointer tiebreaker: greater means closer to root */
    return a->path > b->path;
}

static void
rank_idx_maxheap_sift_up(int *heap, int idx, const PathRank *arr) {
    while (idx > 0) {
        const int parent = (idx - 1) >> 1;
        if (!rank_idx_greater_by_score(heap[idx], heap[parent], arr))
            break;
        const int tmp = heap[parent];
        heap[parent] = heap[idx];
        heap[idx] = tmp;
        idx = parent;
    }
}

static void
rank_idx_maxheap_sift_down(int *heap, const int n, int idx, const PathRank *arr) {
    for (;;) {
        const int l = (idx << 1) + 1;
        const int r = l + 1;
        int largest = idx;

        if (l < n && rank_idx_greater_by_score(heap[l], heap[largest], arr))
            largest = l;
        if (r < n && rank_idx_greater_by_score(heap[r], heap[largest], arr))
            largest = r;
        if (largest == idx)
            break;

        const int tmp = heap[largest];
        heap[largest] = heap[idx];
        heap[idx] = tmp;
        idx = largest;
    }
}

/* Push new index; if heap is full (size==k) and new item is better (smaller score),
 * replace root then sift-down. Returns current heap size (<=k).
 */
static int
rank_idx_maxheap_push_topk(int *heap, const int size, const int k, const int idx, const PathRank *arr) {
    if (size < k) {
        heap[size] = idx;
        rank_idx_maxheap_sift_up(heap, size, arr);
        return size + 1;
    }
    /* if new candidate is better (smaller score) than current worst (root), replace */
    if (rank_idx_greater_by_score(heap[0], idx, arr)) {
        /* root worse than idx */
        heap[0] = idx;
        rank_idx_maxheap_sift_down(heap, k, 0, arr);
    }
    return size; /* unchanged */
}

/*
 * CoverPathRank
 *
 * Composite ranking for robust coverage selection:
 *   - cover_rank: larger is better
 *   - mep:        smaller is better
 *
 * Sorting rule:
 *   - Primary key:  cover_rank (descending).
 *   - Secondary:    mep (ascending).
 */
typedef struct CoverPathRank {
    Path *path;
    int cover_rank; /* primary: larger is better */
    double mep; /* secondary: smaller is better */
} CoverPathRank;

/* A fast "is a better candidate than b" helper (matches compare_cover_path_rank). */
static bool
cover_path_rank_better(const CoverPathRank *a, const CoverPathRank *b) {
    if (a->cover_rank != b->cover_rank)
        return a->cover_rank > b->cover_rank; /* DESC: larger is better */
    if (a->mep != b->mep)
        return a->mep < b->mep; /* ASC */
    return false;
}

/*
 * Comparator for CoverPathRank used by qsort().
 *
 * Order:
 *   1) cover_rank descending (larger = better)
 *   2) mep ascending (smaller = better)
 */
static int
compare_cover_path_rank(const void *a, const void *b) {
    const CoverPathRank *ra = a;
    const CoverPathRank *rb = b;

    if (ra->cover_rank < rb->cover_rank)
        return 1;
    if (ra->cover_rank > rb->cover_rank)
        return -1;

    /* Secondary: mep ascending (smaller = better) */
    if (ra->mep < rb->mep)
        return -1;
    if (ra->mep > rb->mep)
        return 1;

    return 0;
}

/*
 * select_path_by_retention_set
 *
 * From the given candidate list, keep at most `select_path_limit` Paths
 * using multiple scoring strategies and a round-robin selection policy.
 * Winners are appended to `*kept_list_ptr` (which may already contain entries),
 * and the function returns a List* of all pruned (not kept) Paths.
 *
 * Contract (per-strategy):
 *   - Lower score = better (except robust coverage).
 *   - Each strategy function must:
 *       * Fill rank_arr[0..cand_count-1] with {path, score} for each node in
 *         `cand_list`, in the same iteration order.
 *       * Assign DBL_MAX (or equivalent) to paths with zero effective samples.
 *   - `min_envelope` and `sample_count` should already be prepared (e.g., clamped
 *     to available dimensions) by the caller.
 *
 * Multi-strategy policy:
 *   - We build four PathRank arrays:
 *       1) calc_worst_penalty()
 *       2) calc_expected_penalty()
 *       3) calc_expected_total_cost()
 *       4) calc_expected_penalty_with_std()
 *       5) calc_robust_coverage()
 *       6) calc_plan_similarity()
 *   - Selection is done in round-robin order over these four strategies:
 *       strategy 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0 -> 1 -> ...
 *   - For each strategy in turn, we pick the *best* (lowest score) path
 *     not yet selected according to that strategy. Selected paths are
 *     never picked again by later strategies.
 *   - The process stops when either:
 *       * We have selected `select_path_limit` paths, or
 *       * A full round over all strategies fails to select any new path
 *         (e.g., all remaining have DBL_MAX scores).
 *
 * Score exposure:
 *   - If `should_save_score` is true, we store the score from
 *     calc_expected_total_cost() into Path->score for both kept and dropped
 *     paths. This gives a single consistent scalar per Path, even though
 *     the selection itself is multi-strategy and round-robin.
 *
 * Return:
 *   - List* of pruned Paths (those NOT kept). If nothing is pruned, returns NIL.
 *
 * Notes:
 *   - Winners are NOT sorted here; order is deterministic but not guaranteed to
 *     be strictly increasing by score. If a sorted order is required, the caller
 *     should sort `*kept_list_ptr` afterward.
 *   - Existing entries in `*kept_list_ptr` are preserved; winners are appended.
 *   - This function does NOT free list cells of `cand_list` (parameter is const).
 *     If the caller needs to release list cells of `cand_list`, do it outside
 *     after the call.
 */
static List *
select_path_by_retention_set(
    const List *cand_list,
    List **kept_list_ptr,
    const double *min_envelope,
    const int select_path_limit,
    const int sample_count,
    const bool should_save_score
) {
    const int cand_count = list_length(cand_list);

    /* --------------------------------------------------------------------
     * Phase 1: build four PathRank arrays and compute scores via strategies.
     * Each strategy sees the same cand_list in the same iteration order.
     * -------------------------------------------------------------------- */
    PathRank *rank_worst = palloc(sizeof(PathRank) * cand_count);
    PathRank *rank_exp_penalty = palloc(sizeof(PathRank) * cand_count);
    PathRank *rank_total_cost = palloc(sizeof(PathRank) * cand_count);
    PathRank *rank_exp_penalty_std = palloc(sizeof(PathRank) * cand_count);

    /* New: similarity (initialized to 0s to match your existing assumption) */
    PathRank *rank_similarity = palloc0(sizeof(PathRank) * cand_count);

    /* New: robust coverage + MEP composite */
    PathRank *cover_arr = palloc(sizeof(PathRank) * cand_count);
    PathRank *mep_arr = palloc(sizeof(PathRank) * cand_count);
    CoverPathRank *rank_cover_mep = palloc(sizeof(CoverPathRank) * cand_count);

    /* Strategy 0: worst penalty */
    calc_worst_penalty(cand_list, rank_worst, min_envelope, sample_count);

    /* Strategy 1: expected penalty */
    calc_expected_penalty(cand_list, rank_exp_penalty, min_envelope, sample_count);

    /* Strategy 2: expected total cost (also used for Path->score exposure) */
    calc_expected_total_cost(cand_list, rank_total_cost, min_envelope, sample_count);

    /* Strategy 3: expected penalty with standard deviation */
    calc_expected_penalty_with_std(cand_list, rank_exp_penalty_std, min_envelope, sample_count);

    /* Strategy 4 (composite): robust coverage + MEP */
    calc_robust_coverage(cand_list, cover_arr, min_envelope, sample_count);
    calc_expected_penalty(cand_list, mep_arr, min_envelope, sample_count);

    for (int i = 0; i < cand_count; ++i) {
        Path *p_cover = cover_arr[i].path;
        Path *p_mep = mep_arr[i].path;

        if (p_cover != p_mep) {
            elog(ERROR, "inconsistent paths when selecting paths by robust coverages");
        }
        Assert(p_cover == p_mep);

        rank_cover_mep[i].path = p_cover;

        /* Treat cover_arr[i].score as "rank/score where larger is better". */
        rank_cover_mep[i].cover_rank = (int) cover_arr[i].score;
        rank_cover_mep[i].mep = mep_arr[i].score;
    }

    /* Strategy 5: similarity */
    calc_plan_similarity(cand_list, rank_similarity, min_envelope, sample_count);

    /*
     * Post-process similarity into a clean scalar ranking:
     *   - if (score > 0) => Center => score = 0 (best)
     *   - else           => Non-center => score = 1
     *   - path == NULL   => invalid => DBL_MAX (never pick)
     *
     * This is a safer version of your snippet: avoids selecting NULL paths.
     */
    for (int i = 0; i < cand_count; ++i) {
        if (rank_similarity[i].path == NULL) {
            rank_similarity[i].score = DBL_MAX;
            continue;
        }

        if (rank_similarity[i].score > 0.0) {
            rank_similarity[i].score = 0.0; /* Center */
        } else {
            rank_similarity[i].score = 1.0; /* Non-center */
        }
    }

    /* Convenience array for scalar strategies (lower score = better). */
    PathRank *scalar_strategies[5] = {
        rank_worst, /* s=0 */
        rank_exp_penalty, /* s=1 */
        rank_total_cost, /* s=2 */
        rank_exp_penalty_std, /* s=3 */
        rank_similarity /* s=5 */
    };

    /* --------------------------------------------------------------------
     * Phase 2: round-robin top-k selection across the four strategies.
     *
     * For each strategy in turn, pick the best (lowest-score) path that:
     *   - has not been selected yet, and
     *   - has a finite score (< DBL_MAX) in that strategy.
     *
     * Once a path is selected by any strategy, it is marked and excluded
     * from further consideration by all strategies.
     * -------------------------------------------------------------------- */
#define STRAT_COUNT 6
    const int k_target = Min(select_path_limit, cand_count);
    /* Mark which candidate indices are already selected as winners. */
    bool *selected = palloc0(sizeof(bool) * cand_count);

    /* Store indices of winning candidates (into [0 .. winners_cnt-1]). */
    int *winners = palloc(sizeof(int) * k_target);
    int winners_cnt = 0;

    while (winners_cnt < k_target) {
        bool progress_this_round = false;

        for (int s = 0; s < STRAT_COUNT && winners_cnt < k_target; s++) {
            /* Strategy 4: composite (cover_rank DESC, mep ASC) */
            if (s == 4) {
                int best_idx = -1;

                for (int i = 0; i < cand_count; i++) {
                    if (selected[i])
                        continue;
                    if (rank_cover_mep[i].path == NULL)
                        continue;

                    if (best_idx < 0 ||
                        cover_path_rank_better(&rank_cover_mep[i], &rank_cover_mep[best_idx])) {
                        best_idx = i;
                    }
                }

                if (best_idx >= 0) {
                    selected[best_idx] = true;
                    winners[winners_cnt++] = best_idx;
                    progress_this_round = true;
                }

                continue;
            }

            /* Other strategies: scalar (lower score = better) */
            PathRank *cur = NULL;
            if (s < 4) {
                cur = scalar_strategies[s];
            } else {
                /* s == 5 */
                cur = scalar_strategies[4]; /* rank_similarity */
            }

            int best_idx = -1;
            double best_score = DBL_MAX;

            for (int i = 0; i < cand_count; i++) {
                if (selected[i])
                    continue;

                const double score = cur[i].score;
                if (score < best_score) {
                    best_score = score;
                    best_idx = i;
                }
            }

            /* If we found a candidate with a valid score, select it. */
            if (best_idx >= 0 && best_score < DBL_MAX) {
                selected[best_idx] = true;
                winners[winners_cnt++] = best_idx;
                progress_this_round = true;
            }
        }

        /*
         * If a full round over all strategies failed to pick any new path,
         * we stop. Remaining candidates are considered losers.
         */
        if (!progress_this_round)
            break;
    }

    /* Number of losers: all candidates that were not selected. */
    const int losers_cnt = cand_count - winners_cnt;
    int *losers = palloc(sizeof(int) * Max(0, losers_cnt));

    if (losers_cnt > 0) {
        int writer = 0;
        for (int i = 0; i < cand_count; i++) {
            if (!selected[i]) {
                losers[writer++] = i;
            }
        }
        Assert(writer == losers_cnt);
    }

    /* --------------------------------------------------------------------
     * Phase 3: append winners to kept list and build dropped list (return).
     *
     * NOTE:
     *   - For Path->score we expose calc_expected_total_cost()'s score, so
     *     that each Path has a single, comparable scalar, even though the
     *     selection was driven by multiple strategies.
     * -------------------------------------------------------------------- */
    List *kept_list = (*kept_list_ptr != NULL) ? *kept_list_ptr : NIL;

    /* Append winners (in round-robin order) */
    for (int i = 0; i < winners_cnt; i++) {
        const int idx = winners[i];
        const PathRank rank = rank_total_cost[idx]; /* canonical score source */

        Path *keep = rank.path;
        if (should_save_score) {
            /* Expose expected total cost on Path for later stages. */
            keep->score = rank.score;
        }
        kept_list = lappend(kept_list, keep);
    }

    List *dropped_list = NIL;
    if (losers_cnt > 0) {
        for (int i = 0; i < losers_cnt; i++) {
            const int idx = losers[i];
            const PathRank rank = rank_total_cost[idx]; /* same canonical source */

            Path *drop = rank.path;
            if (should_save_score) {
                /* Expose expected total cost on Path for later stages. */
                drop->score = rank.score;
            }
            dropped_list = lappend(dropped_list, drop);
        }
    }

    /* Write back survivors (append result) */
    *kept_list_ptr = kept_list;

    /* --------------------------------------------------------------------
     * Phase 4: cleanup and return pruned paths.
     * -------------------------------------------------------------------- */
    pfree(rank_worst);
    pfree(rank_exp_penalty);
    pfree(rank_total_cost);
    pfree(rank_exp_penalty_std);

    pfree(rank_similarity);

    pfree(cover_arr);
    pfree(mep_arr);
    pfree(rank_cover_mep);

    pfree(selected);
    pfree(winners);
    pfree(losers);

    return dropped_list;
}

/*
 * select_path_by_jointype_based_score
 *
 * From the given candidate list, keep at most `select_path_limit` Paths
 * using a single jointype-based scoring strategy (via `path_strategy_func`).
 *
 * The scoring callback (path_strategy_func) is expected to behave like
 * calc_expected_penalty():
 *   - Lower score = better.
 *   - It fills rank_arr[0..cand_count-1] in the same iteration order
 *     as `cand_list`.
 *   - Paths with zero effective samples must get score = DBL_MAX (or equivalent).
 *
 * Multi-group round-robin selection policy:
 *   - First we compute a single PathRank array using `path_strategy_func()`.
 *   - Then we group all candidates by their Path->pathtype (or join type,
 *     if the caller chooses to change the grouping key).
 *   - We perform round-robin selection over these groups:
 *       * In each outer iteration, we scan all groups in fixed order
 *         (0 .. group_count-1).
 *       * For each group, we pick the *best* (lowest-score) path in that
 *         group that has not yet been selected and has a finite score.
 *       * Selected paths are never considered again in later rounds.
 *   - The process stops when either:
 *       * We have selected `select_path_limit` paths, or
 *       * A full round over all groups fails to select any new path
 *         (e.g., all remaining candidates have DBL_MAX scores).
 *
 * Score exposure:
 *   - If `should_save_score` is true, we store the score computed by
 *     `path_strategy_func()` into Path->score for both kept and dropped
 *     paths. This gives each Path a single, comparable scalar.
 *
 * Return:
 *   - List* of pruned Paths (those NOT kept). If nothing is pruned,
 *     returns NIL.
 *
 * Notes:
 *   - Winners are NOT sorted here; the order is deterministic but reflects
 *     the round-robin selection order across groups. If the caller requires
 *     a strict ordering (e.g., by score), it should sort `*kept_list_ptr`
 *     afterward.
 *   - Existing entries in `*kept_list_ptr` are preserved; winners are
 *     appended at the tail.
 *   - This function does NOT free list cells of `cand_list` (parameter is
 *     const). If the caller needs to release list cells of `cand_list`,
 *     it must do so outside after this call.
 */
static List *
select_path_by_jointype_based_score(
    const List *cand_list,
    List **kept_list_ptr,
    const double *min_envelope,
    const path_strategy path_strategy_func,
    const int select_path_limit,
    const int sample_count,
    const bool should_save_score
) {
    const int cand_count = list_length(cand_list);

    /* Trivial case: no candidates at all. Nothing to keep or drop. */
    if (cand_count == 0)
        return NIL;

    /* --------------------------------------------------------------------
     * Phase 1: compute scores for all candidates via the given strategy.
     *
     * `path_strategy_func()` is expected to:
     *   - Fill rank_jointype[0..cand_count-1] with {path, score}.
     *   - Assign DBL_MAX to paths with no effective samples.
     * -------------------------------------------------------------------- */
    PathRank *rank_jointype = palloc(sizeof(PathRank) * cand_count);

    path_strategy_func(cand_list, rank_jointype, min_envelope, sample_count);

    /* --------------------------------------------------------------------
     * Phase 2: round-robin top-k selection across different path types.
     *
     * We group paths by Path->pathtype (or join type if desired) and then
     * select winners in a round-robin fashion across these groups.
     * -------------------------------------------------------------------- */

    /*
     * k_target is the maximum number of winners we will select.
     * If select_path_limit <= 0, we treat it as "select nothing",
     * meaning all candidates become losers.
     */
    const int k_target = (select_path_limit > 0)
                             ? Min(select_path_limit, cand_count)
                             : 0;

    /* Group assignment arrays (only needed if k_target > 0). */
    int *group_id = NULL;
    int *group_keys = NULL;
    int group_count = 0;

    /*
     * selected[i] marks whether candidate i has been chosen as a winner.
     * This is always allocated, even if k_target == 0, so that the loser
     * construction logic can be uniform.
     */
    bool *selected = palloc0(sizeof(bool) * cand_count);

    /* Store indices of winning candidates. */
    int *winners = NULL;
    int winners_cnt = 0;

    if (k_target > 0) {
        /*
         * Step 2.1: assign each candidate to a "path-type group".
         *
         * We use Path->pathtype as the grouping key here.
         * If you prefer to group by join type, replace the key extraction with:
         *
         *   JoinPath *jpath = (JoinPath *) rank_jointype[i].path;
         *   int key = (int) jpath->jointype;
         *
         * and ensure that all candidates are actually JoinPath nodes.
         *
         * group_keys[g]  = the grouping key value for group g
         * group_id[i]    = index of the group that candidate i belongs to
         * group_count    = total number of distinct groups
         */
        group_id = palloc(sizeof(int) * cand_count);
        group_keys = palloc(sizeof(int) * cand_count); /* upper bound */

        for (int i = 0; i < cand_count; i++) {
            Path *path = rank_jointype[i].path;

            /* Grouping key: use pathtype by default. */
            int key = (int) path->pathtype;

            /* Look for an existing group with the same key. */
            int g;
            for (g = 0; g < group_count; g++) {
                if (group_keys[g] == key)
                    break;
            }

            /* If no existing group matches this key, create a new group. */
            if (g == group_count) {
                group_keys[group_count] = key;
                group_count++;
            }

            group_id[i] = g;
        }

        /* Allocate the winners array once we know k_target. */
        winners = palloc(sizeof(int) * k_target);

        /*
         * Step 2.2: round-robin selection over groups.
         *
         * Outer loop:
         *   - We repeatedly walk through all groups.
         *   - For each group in turn, we pick the best unselected candidate
         *     in that group (if any).
         *   - If a full pass over all groups fails to select any new path,
         *     we stop early.
         */
        while (winners_cnt < k_target) {
            bool progress_this_round = false;

            for (int g = 0; g < group_count && winners_cnt < k_target; g++) {
                int best_idx = -1;
                double best_score = DBL_MAX;

                /*
                 * Scan all candidates, but only consider those:
                 *   - belonging to group g, and
                 *   - not yet selected.
                 */
                for (int i = 0; i < cand_count; i++) {
                    if (selected[i])
                        continue;
                    if (group_id[i] != g)
                        continue;

                    const double score = rank_jointype[i].score;

                    if (score < best_score) {
                        best_score = score;
                        best_idx = i;
                    }
                }

                /*
                 * If we found a candidate in this group with a finite score,
                 * select it and move on to the next group.
                 */
                if (best_idx >= 0 && best_score < DBL_MAX) {
                    selected[best_idx] = true;
                    winners[winners_cnt++] = best_idx;
                    progress_this_round = true;
                }
            }

            /*
             * If we traversed all groups and failed to pick any new candidate,
             * then all remaining candidates are either already selected or have
             * invalid scores (DBL_MAX). We stop here.
             */
            if (!progress_this_round)
                break;
        }
    }

    /* --------------------------------------------------------------------
     * Phase 2.3: compute losers = all candidates that were never selected.
     * -------------------------------------------------------------------- */
    const int losers_cnt = cand_count - winners_cnt;
    int *losers = palloc(sizeof(int) * Max(0, losers_cnt));

    if (losers_cnt > 0) {
        int writer = 0;

        for (int i = 0; i < cand_count; i++) {
            if (!selected[i]) {
                losers[writer++] = i;
            }
        }
        Assert(writer == losers_cnt);
    }

    /* --------------------------------------------------------------------
     * Phase 3: append winners to kept list and build the dropped list.
     *
     * We use the scores from `path_strategy_func()` (rank_jointype[].score)
     * as the canonical scalar that we expose on Path->score (if requested).
     * -------------------------------------------------------------------- */
    List *kept_list = (*kept_list_ptr != NULL) ? *kept_list_ptr : NIL;

    /* Append winners in the round-robin order in which we selected them. */
    if (winners_cnt > 0) {
        for (int i = 0; i < winners_cnt; i++) {
            const int idx = winners[i];
            const PathRank rank = rank_jointype[idx];

            Path *keep = rank.path;

            if (should_save_score) {
                /* Expose the jointype-based strategy score on Path. */
                keep->score = rank.score;
            }
            kept_list = lappend(kept_list, keep);
        }
    }

    List *dropped_list = NIL;

    if (losers_cnt > 0) {
        for (int i = 0; i < losers_cnt; i++) {
            const int idx = losers[i];
            const PathRank rank = rank_jointype[idx];

            Path *drop = rank.path;

            if (should_save_score) {
                /* Expose the jointype-based strategy score on Path. */
                drop->score = rank.score;
            }
            dropped_list = lappend(dropped_list, drop);
        }
    }

    /* Write back survivors (append result). */
    *kept_list_ptr = kept_list;

    /* --------------------------------------------------------------------
     * Phase 4: cleanup and return pruned paths.
     * -------------------------------------------------------------------- */
    pfree(rank_jointype);
    pfree(selected);
    pfree(losers);

    if (winners)
        pfree(winners);
    if (group_id)
        pfree(group_id);
    if (group_keys)
        pfree(group_keys);

    return dropped_list;
}

/*
 * select_path_by_robust_coverage
 *
 * From the given candidate list, keep at most `select_path_limit` Paths
 * using a combined ranking:
 *
 *   - Primary:  cover_rank descending (larger = better, more coverage).
 *   - Secondary: mep   ascending (smaller = better, lower expected penalty).
 *
 * Winners (top-k after sorting by the above rule) are appended to
 * `*kept_list_ptr` (which may already contain entries), and the function
 * returns a List* of all pruned (not kept) Paths.
 *
 * Contract:
 *   - `calc_robust_coverage` must:
 *       * Fill a PathRank array with .path and .score for each candidate.
 *       * .score is an integer-like *coverage amount* (e.g., #covered samples).
 *   - `calc_expected_penalty` must:
 *       * Fill a PathRank array with .path and .score (MEP) for each candidate.
 *       * Lower MEP = better.
 *
 * Score exposure:
 *   - If `should_save_score` is true, we store the cover rank for both kept
 *     and dropped paths.
 *
 * Return:
 *   - List* of pruned Paths (those NOT kept). If nothing is pruned, returns NIL.
 *
 * Notes:
 *   - Winners are NOT sorted by Path->score; they follow the sorted combined
 *     order described above.
 *   - Existing entries in `*kept_list_ptr` are preserved; winners are appended.
 *   - This function does NOT free list cells of `cand_list` (parameter is const).
 *     If the caller needs to release list cells of `cand_list`, do it outside
 *     after the call.
 */
static List *
select_path_by_robust_coverage(
    const List *cand_list,
    List **kept_list_ptr,
    const double *min_envelope,
    const int select_path_limit,
    const int sample_count,
    const bool should_save_score
) {
    const int cand_count = list_length(cand_list);

    /* Clamp limit to number of candidates */
    const int k = Min(select_path_limit, cand_count);

    /* --------------------------------------------------------------------
    * Phase 1: compute robust coverage rank and minimum expected penalty.
    *
    * We use two temporary PathRank arrays:
    *   - cover_arr: scores from calc_robust_coverage (0, 1, 2, ...).
    *   - mep_arr: scores from calc_expected_penalty (double).
    *
    * Then we merge them into a single CoverPathRank array.
    * -------------------------------------------------------------------- */
    PathRank *cover_arr = palloc(sizeof(PathRank) * cand_count);
    PathRank *mep_arr = palloc(sizeof(PathRank) * cand_count);

    /* Robust coverage ranking */
    calc_robust_coverage(cand_list, cover_arr, min_envelope, sample_count);

    /* Minimum expected penalty (MEP) */
    calc_expected_penalty(cand_list, mep_arr, min_envelope, sample_count);

    /* Build composite ranking array */
    CoverPathRank *rank_arr = palloc(sizeof(CoverPathRank) * cand_count);

    for (int i = 0; i < cand_count; ++i) {
        /*
         * Both calc_robust_coverage and calc_expected_penalty must iterate
         * cand_list in the same order, so their i-th entries refer to the
         * same Path*.
         */
        Path *path_cover = cover_arr[i].path;
        const Path *path_mep = mep_arr[i].path;

        /* Sanity: they should match; if not, something is inconsistent. */
        if (path_cover != path_mep) {
            elog(ERROR, "inconsistent paths when selecting paths by robust coverages");
        }
        Assert(path_cover == path_mep);

        rank_arr[i].path = path_cover;
        rank_arr[i].cover_rank = (int) cover_arr[i].score; /* score is integer-like */
        rank_arr[i].mep = mep_arr[i].score; /* minimum expected penalty */
    }

    /* Temporary arrays no longer needed */
    pfree(cover_arr);
    pfree(mep_arr);

    /* --------------------------------------------------------------------
     * Phase 2: sort by (cover_rank DESC, mep ASC) and split into winners/losers.
     * -------------------------------------------------------------------- */
    qsort(rank_arr, cand_count, sizeof(CoverPathRank), compare_cover_path_rank);

    /* Winners: first k entries; losers: remaining entries */
    List *kept_list = (*kept_list_ptr != NULL) ? *kept_list_ptr : NIL;
    List *dropped_list = NIL;

    /* Append winners */
    for (int i = 0; i < k; i++) {
        const CoverPathRank *cover_rank = &rank_arr[i];
        Path *keep = cover_rank->path;

        if (should_save_score) {
            /*
             * Expose cover rank as Path->score, so later stages can see a scalar
             * expectation-based penalty for this Path.
             */
            keep->score = cover_rank->cover_rank;
        }

        kept_list = lappend(kept_list, keep);
    }
    /* Build dropped list (if any) */
    for (int i = k; i < cand_count; i++) {
        const CoverPathRank *cover_rank = &rank_arr[i];
        Path *drop = cover_rank->path;

        if (should_save_score) {
            /* Same convention for losers: expose cover rank into Path->score. */
            drop->score = cover_rank->cover_rank;
        }

        dropped_list = lappend(dropped_list, drop);
    }

    /* Write back survivors (append result) */
    *kept_list_ptr = kept_list;

    /* --------------------------------------------------------------------
     * Phase 3: cleanup and return pruned paths.
     * -------------------------------------------------------------------- */
    pfree(rank_arr);

    return dropped_list;
}

/*
 * select_path_by_strategy_basic
 *
 * From the given candidate list, keep at most `select_path_limit` Paths
 * according to the provided scoring strategy. Winners are appended to
 * `*kept_list_ptr` (which may already contain entries), and the function
 * returns a List* of all pruned (not kept) Paths.
 *
 * Contract:
 *   - Lower score = better.
 *   - `path_strategy_func` must:
 *       * Fill `rank_arr[0..cand_count-1]` with {path, score} for each node in
 *         `cand_list`, in the same iteration order.
 *       * Assign DBL_MAX (or equivalent) to paths with zero effective samples.
 *   - `min_envelope` and `sample_count` should already be prepared (e.g., clamped
 *     to available dimensions) by the caller.
 *
 * Return:
 *   - List* of pruned Paths (those NOT kept). If nothing is pruned, returns NIL.
 *
 * Notes:
 *   - Winners are NOT sorted here; order is deterministic but not guaranteed to
 *     be strictly increasing by score. If a sorted order is required, the caller
 *     should sort `*kept_list_ptr` afterward.
 *   - If `should_save_score` is true, each kept Path gets its computed score
 *     stored into `Path->score`. Losers’ scores are not modified.
 *   - Existing entries in `*kept_list_ptr` are preserved; winners are appended.
 *   - This function does NOT free `cand_list` (parameter is const). If the caller
 *     needs to release list cells of `cand_list`, do it outside after the call.
 */
static List *
select_path_by_strategy_basic(
    const List *cand_list,
    List **kept_list_ptr,
    const double *min_envelope,
    const path_strategy path_strategy_func,
    const int select_path_limit,
    const int sample_count,
    const bool should_save_score
) {
    const int cand_count = list_length(cand_list);

    /* --------------------------------------------------------------------
     * Phase 1: build PathRank array and compute scores via strategy.
     * The strategy function fills rank_arr with (path, score).
     * -------------------------------------------------------------------- */
    PathRank *rank_arr = palloc0(sizeof(PathRank) * cand_count);
    path_strategy_func(cand_list, rank_arr, min_envelope, sample_count);

    /* --------------------------------------------------------------------
     * Phase 2: select global top-k (smallest score) using a fixed MAX-heap.
     * Root of heap = worst among currently kept.
     * -------------------------------------------------------------------- */
    int k = Min(select_path_limit, cand_count);

    /* Reconsider k */
    if (path_strategy_func == calc_plan_similarity) {
        int adjusted_k = 0;
        // FIXME: Not all element in `rank_arr` is valid. But we use `palloc0` to initialize
        // FIXME: `rank_arr`. Thus, `rank_arr[i].score > 0.0` should indicate a valid item now.
        // FIXME: Otherwise: either invalid or non-center about which we do not care.
        for (int i = 0; i < cand_count; i++) {
            if (rank_arr[i].score > 0.0) {
                rank_arr[i].score = 0.0; // Center
                ++adjusted_k;
            } else {
                rank_arr[i].score = 1.0; // Non-center or invalid
            }
        }
        Assert(adjusted_k > 0);
        k = Min(k, adjusted_k);
    }

    int *heap_idx = palloc(sizeof(int) * Max(1, k));
    int hsize = 0;

    for (int i = 0; i < cand_count; i++) {
        hsize = rank_idx_maxheap_push_topk(heap_idx, hsize, k, i, rank_arr);
    }

    /* hsize should equal k because k = min(limit, cand_count) */
    Assert(hsize == k);

    /* --------------------------------------------------------------------
     * Phase 3: materialize winners/losers (by index). Order not finalized here.
     * -------------------------------------------------------------------- */
    int *winners = palloc(sizeof(int) * k);
    for (int i = 0; i < k; i++) {
        winners[i] = heap_idx[i];
    }

    const int losers_cnt = cand_count - k;
    int *losers = palloc(sizeof(int) * losers_cnt);
    if (losers_cnt > 0) {
        bool *selected = palloc0(sizeof(bool) * cand_count);
        for (int i = 0; i < k; i++) {
            selected[winners[i]] = true;
        }

        int writer = 0;
        for (int i = 0; i < cand_count; i++) {
            if (!selected[i]) {
                losers[writer++] = i;
            }
        }
        Assert(writer == losers_cnt);
        pfree(selected);
    }

    /* --------------------------------------------------------------------
     * Phase 4: append winners to kept list and build dropped list (return).
     * Optionally expose per-path score for kept paths.
     * -------------------------------------------------------------------- */
    List *kept_list = (*kept_list_ptr != NULL) ? *kept_list_ptr : NIL;

    /* k = cand_count - losers_cnt. */
    for (int i = 0; i < k; i++) {
        const PathRank rank = rank_arr[winners[i]];
        Path *keep = rank.path;
        if (should_save_score) {
            /* Expose computed score on Path for later stages,
             * and for the final output after the DP process. */
            keep->score = rank.score;
        }
        kept_list = lappend(kept_list, keep);
    }

    List *dropped_list = NIL;
    if (losers_cnt > 0) {
        for (int i = 0; i < losers_cnt; i++) {
            const PathRank rank = rank_arr[losers[i]];
            Path *drop = rank.path;
            if (should_save_score) {
                /* Expose computed score on Path for later stages,
                 * and for the final output after the DP process. */
                drop->score = rank.score;
            }
            dropped_list = lappend(dropped_list, drop);
        }
    }

    /* Write back survivors (append result) */
    *kept_list_ptr = kept_list;

    /* --------------------------------------------------------------------
     * Phase 5: cleanup and return pruned paths.
     * -------------------------------------------------------------------- */
    pfree(rank_arr);
    pfree(heap_idx);
    pfree(winners);
    pfree(losers);

    return dropped_list;
}

/*
 * select_path_by_strategy_dispatch
 *
 * From the given candidate list, keep at most `select_path_limit` Paths
 * according to the provided scoring strategy. Winners are appended to
 * `*kept_list_ptr` (which may already contain entries), and the function
 * returns a List* of all pruned (not kept) Paths.

 * Return:
 *   - List* of pruned Paths (those NOT kept). If nothing is pruned, returns NIL.
 */
List *
select_path_by_strategy_dispatch(
    const List *cand_list,
    List **kept_list_ptr,
    const double *min_envelope,
    const path_strategy path_strategy_func,
    const int select_path_limit,
    const int sample_count,
    const bool should_save_score
) {
    /*
     * Policy note:
     *   - If this stage is the main objective, set should_save_score = true to
     *     expose the computed score on kept Paths for later phases.
     *   - Otherwise (retain-only policy), leave scores as-is.
     */

    /* Basic sanity */
    Assert(kept_list_ptr != NULL);
    Assert(sample_count >= 1);
    Assert(select_path_limit >= 1);
    Assert(sample_count <= DIST_MAX_SAMPLE);

    /* Early exit: no candidates; keep existing kept list untouched */
    const int cand_count = list_length(cand_list);
    if (cand_count <= 0) {
        return NIL;
    }

    if (path_strategy_func == calc_jointype_based_mep) {
        return select_path_by_jointype_based_score(
            cand_list,
            kept_list_ptr,
            min_envelope,
            calc_expected_penalty,
            select_path_limit,
            sample_count,
            should_save_score
        );
    }
    if (path_strategy_func == calc_jointype_based_mec) {
        return select_path_by_jointype_based_score(
            cand_list,
            kept_list_ptr,
            min_envelope,
            calc_expected_total_cost,
            select_path_limit,
            sample_count,
            should_save_score
        );
    }
    if (path_strategy_func == calc_robust_coverage) {
        return select_path_by_robust_coverage(
            cand_list,
            kept_list_ptr,
            min_envelope,
            select_path_limit,
            sample_count,
            should_save_score
        );
    }
    if (path_strategy_func == calc_retention_set) {
        return select_path_by_retention_set(
            cand_list,
            kept_list_ptr,
            min_envelope,
            select_path_limit,
            sample_count,
            should_save_score
        );
    }
    return select_path_by_strategy_basic(
        cand_list,
        kept_list_ptr,
        min_envelope,
        path_strategy_func,
        select_path_limit,
        sample_count,
        should_save_score
    );
}

/*
 * set_path_score
 *
 * Purpose:
 *   Compute a score for every Path in `cand_list` using the provided strategy
 *   function, and store the result into `path->score`.
 *
 * What this function DOES:
 *   - Allocates a temporary PathRank array sized to `cand_count`.
 *   - Invokes `path_strategy_func(cand_list, rank_arr, min_envelope, sample_count)`.
 *     The strategy fills rank_arr[0..cand_count-1] with (path, score) pairs
 *     that correspond to the iteration order of `cand_list`.
 *   - Writes `rank_arr[i].score` back into the associated `Path` via `path->score`.
 *   - Frees the temporary array and returns.
 *
 * What this function DOES NOT do:
 *   - It does NOT select or prune candidates.
 *   - It does NOT sort winners or losers.
 *   - It does NOT modify list topology (no cells are freed or appended).
 *
 * Contract & Assumptions:
 *   - Lower score means better.
 *   - The strategy must assign a finite score for valid candidates and should
 *     assign DBL_MAX (or equivalent “worst”) to paths with zero effective samples.
 *   - `min_envelope` and `sample_count` have already been prepared/clamped by caller.
 *   - `sample_count >= 1` and `sample_count <= DIST_MAX_SAMPLE`.
 *
 * External invariants (referenced but not changed here):
 *   - `kept_list_ptr` and `select_path_limit` are asserted for sanity only.
 *     They are not read for control-flow in this routine and are presumed
 *     to be defined in the surrounding module.
 *
 * Inputs:
 *   - cand_list:     list of Path* to be scored.
 *   - min_envelope:  baseline (per-sample) array needed by the scoring strategy.
 *   - path_strategy_func: callback that computes path scores.
 *   - sample_count:  number of samples the strategy should consider.
 *
 * Side effects:
 *   - Each Path in cand_list receives its computed score in `Path->score`.
 *
 * Return:
 *   - void
 *
 * Error handling:
 *   - Early return if cand_list is empty.
 *   - Asserts enforce basic preconditions in assertions-enabled builds.
 */
void
set_path_score(
    const List *cand_list,
    const double *min_envelope,
    const path_strategy path_strategy_func,
    const int sample_count
) {
    /* --- Sanity checks on global/contextual invariants (not used for control flow) --- */
    Assert(kept_list_ptr != NULL);
    Assert(sample_count >= 1);
    Assert(select_path_limit >= 1);
    Assert(sample_count <= DIST_MAX_SAMPLE);

    /* --- Early exit: nothing to score --- */
    const int cand_count = list_length(cand_list);
    if (cand_count <= 0) {
        return;
    }

    /* --------------------------------------------------------------------
     * Phase 1: allocate rank array & run strategy
     *
     * The strategy fills `rank_arr` with (path, score) for each element of
     * `cand_list` in the same traversal order.
     * -------------------------------------------------------------------- */
    PathRank *rank_arr = palloc(sizeof(PathRank) * cand_count);
    if (path_strategy_func == calc_jointype_based_mep) {
        calc_expected_penalty(cand_list, rank_arr, min_envelope, sample_count);
    } else if (path_strategy_func == calc_jointype_based_mec) {
        calc_expected_total_cost(cand_list, rank_arr, min_envelope, sample_count);
    } else {
        path_strategy_func(cand_list, rank_arr, min_envelope, sample_count);
    }

    /* --------------------------------------------------------------------
     * Phase 2: persist scores onto Path objects
     *
     * We only write scores back; selection/pruning is not handled here.
     * -------------------------------------------------------------------- */
    for (int i = 0; i < cand_count; i++) {
        const PathRank rank = rank_arr[i];
        Path *path = rank.path;
        path->score = rank.score;
    }

    /* --- Phase 3: cleanup temporary workspace --- */
    pfree(rank_arr);
}

/*
 * calc_score_from_pathlist
 *
 * Compute the minimum total_cost for this joinrel over its candidate paths.
 * Two values are produced:
 *
 *   joinrel->min_score          = min(path->total_cost) over pathlist
 *   joinrel->partial_min_score  = min(path->total_cost) over partial_pathlist
 *
 * This captures the best (cheapest) deterministic plan choice seen so far
 * for both regular and partial paths.  No per-sample cost vectors are used
 * here; per-round sampling is handled elsewhere.
 *
 * Notes:
 * - Either list may be empty, in which case the corresponding minimum
 *   remains DBL_MAX.
 * - Caller ensures that 'round' and 'sample_count' are valid for the
 *   surrounding sampling framework; this routine uses total_cost only.
 */
void
calc_score_from_pathlist(
    RelOptInfo *joinrel
) {
    Assert(sample_count >= 0 && sample_count <= DIST_MAX_SAMPLE);
    ListCell *lc;

    double min_score = DBL_MAX;
    foreach(lc, joinrel->pathlist) {
        const Path *path = (Path *) lfirst(lc);
        min_score = Min(path->total_cost, min_score);
    }
    joinrel->min_score = min_score;

    double partial_min_score = DBL_MAX;
    foreach(lc, joinrel->partial_pathlist) {
        const Path *partial_path = (Path *) lfirst(lc);
        partial_min_score = Min(partial_path->total_cost, partial_min_score);
    }
    joinrel->partial_min_score = partial_min_score;
}

/*
 * calc_minimum_envelope
 *
 * Accumulate per-round scalar scores for each joinrel into the first snapshot.
 * Each element of saved_join_rel_levels is a List** holding per-level joinrels
 * for one sampling round. We walk levels [2..levels_needed] in lockstep and, for
 * each joinrel, store this round's scalar score into min_rel->min_score_sample.
 *
 * Notes/Assumptions:
 * - The first snapshot acts as the accumulator (updated in-place and returned).
 * - Sample storage uses an embedded fixed-size array inside `Sample`
 *   (e.g., double sample[DIST_MAX_SAMPLE];), so no extra allocation is needed
 *   for the array itself; we only palloc the `Sample` header once at round 0.
 * - List orders match across rounds so `forboth` walks 1:1; if lengths differ,
 *   `forboth` stops at the shorter list.
 * - This routine records the per-round scalar value:
 *        Min(cur_rel->min_score, cur_rel->partial_min_score)
 *   into the sample slot for `round`. Any later elementwise-min over rounds
 *   (i.e., a true "minimum envelope") is expected to be performed elsewhere.
 */
List **
calc_minimum_envelope(
    List *saved_join_rel_levels,
    const int sample_count,
    const int levels_needed
) {
    ListCell *lc, *lc1, *lc2;

    /* Use the first snapshot as the initial (and final) envelope; updated in-place. */
    List **min_envelope = linitial(saved_join_rel_levels);

    /* Iterate over every saved snapshot of per-level join_rel lists. */
    foreach(lc, saved_join_rel_levels) {
        const int round = foreach_current_index(lc);
        List **cur_saved_join_rel_level = lfirst(lc);

        /* By convention, levels 0/1 are base rels; start from level 2. */
        for (int lev = 2; lev <= levels_needed; ++lev) {
            /*
             * Pairwise walk the lists at the same level in the current snapshot
             * and in the running minimum envelope. Assumes identical ordering.
             * If lengths differ, forboth stops at the shorter list.
             */
            forboth(lc1, cur_saved_join_rel_level[lev], lc2, min_envelope[lev]) {
                const RelOptInfo *cur_rel = lfirst(lc1);
                RelOptInfo *min_rel = lfirst(lc2);
                if (round == 0) {
                    /*
                     * Allocate the Sample header once; the underlying
                     * `sample[]` is an embedded fixed-size array in `Sample`,
                     * so no separate allocation is required here.
                     * Caller ensures sample_count <= array capacity.
                     */
                    min_rel->min_score_sample = palloc(sizeof(Sample));
                    min_rel->min_score_sample->sample_count = sample_count;
                }
                /* Current round's scalar score for this joinrel */
                const double cur_rel_score = Min(
                    cur_rel->min_score, cur_rel->partial_min_score
                );
                /* Store into the embedded sample array at index = round */
                min_rel->min_score_sample->sample[round] = cur_rel_score;
            }
        }
    }
    /* Return the first snapshot pointer, now populated with per-round scores. */
    return min_envelope;
}

/*
 * sort_pathlist_by_total_cost
 *
 * Sort a Path list by total_cost in ascending order.
 *
 * Since pathlists are typically small (often < 50 elements),
 * an insertion sort is used instead of list_qsort().
 * Insertion sort has lower overhead for tiny lists and preserves
 * stability (keeps original order on ties).
 *
 * Returns a new List with the same Path* pointers in sorted order.
 */
List *
sort_pathlist_by_total_cost(List *pathlist) {
    if (pathlist == NIL) {
        return NIL;
    }

    const int n = list_length(pathlist);
    if (n <= 1) {
        return pathlist;
    }

    /* Copy to array for O(n^2) insertion sort */
    Path **arr = palloc(sizeof(Path *) * n);

    int i = 0;
    ListCell *lc;
    foreach(lc, pathlist) {
        arr[i++] = (Path *) lfirst(lc);
    }

    /* Insertion sort by total_cost ASC */
    for (int j = 1; j < n; j++) {
        Path *key = arr[j];
        const double key_cost = key->total_cost;
        int k = j - 1;

        while (k >= 0 && arr[k]->total_cost > key_cost) {
            arr[k + 1] = arr[k];
            k--;
        }
        arr[k + 1] = key;
    }

    /* Rebuild list in sorted order */
    List *new_list = NIL;
    for (int j = 0; j < n; j++) {
        new_list = lappend(new_list, arr[j]);
    }

    pfree(arr);
    list_free(pathlist); /* free only list cells, not Path objects */

    return new_list;
}
