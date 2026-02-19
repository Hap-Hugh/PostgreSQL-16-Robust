//
// Created by Xuan Chen on 2025/10/18.
// Modified by Xuan Chen on 2025/10/29.
//

#ifndef ADDPATH_H
#define ADDPATH_H

#include "pathstrategy.h"
#include "postgres.h"
#include "nodes/pathnodes.h"
#include "optimizer/sample.h"
#include "nodes/pg_list.h"

extern List *
select_path_by_strategy_dispatch(
    const List *cand_list,
    List **kept_list_ptr,
    const double *min_envelope,
    path_strategy path_strategy_func,
    int select_path_limit,
    int sample_count,
    bool should_save_score
);

void
set_path_score(
    const List *cand_list,
    const double *min_envelope,
    path_strategy path_strategy_func,
    int sample_count
);

void
calc_score_from_pathlist(
    RelOptInfo *joinrel
);

List **
calc_minimum_envelope(
    List *saved_join_rel_levels,
    int sample_count,
    int levels_needed
);

List *
sort_pathlist_by_total_cost(
    List *pathlist
);

#endif // ADDPATH_H
