//
// Created by Xuan Chen on 2025/10/31.
// Modified by Xuan Chen on 2025/11/9.
//

#ifndef PATHSTRATEGY_H
#define PATHSTRATEGY_H

#include "postgres.h"
#include "nodes/pathnodes.h"
#include "optimizer/sample.h"

#define ROBUST_EPS_DEFAULT 0.01

/* Per-candidate ranking info (only score is relevant here). */
typedef struct PathRank {
    Path *path;
    double score; /* vs. per-type per-sample minima */
} PathRank;

typedef void (*path_strategy)(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_worst_penalty(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_expected_penalty(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_worst_total_cost(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_expected_total_cost(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_worst_startup_cost(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_expected_startup_cost(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_robust_coverage(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_random_score(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_postgres_original_score(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_jointype_based_score(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_worst_penalty_with_std(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_expected_penalty_with_std(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_4th_worst_penalty(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_plan_similarity(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_retention_set(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_jointype_based_mep(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_jointype_based_mec(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

/*
 * Global array of scoring strategies.
 * Indexed [0..14].
 *
 * Caller can use: path_strategy_funcs[i](...)
 */
extern path_strategy path_strategy_funcs[17];

#endif // PATHSTRATEGY_H
