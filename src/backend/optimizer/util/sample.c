//
// Created by Xuan Chen on 2025/9/22.
// Modified by Xuan Chen on 2025/9/24.
// Modified by Xuan Chen on 2025/10/2.
//

#include "optimizer/sample.h"
#include "optimizer/kde.h"
#include "optimizer/ep.h"
#include "utils/smem.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* GUC Parameters */
bool enable_rows_dist;
int error_sample_count;
int error_sample_seed;

/* ------------------------------- Utilities ------------------------------- */
double clamp01(const double sel) {
    if (sel < 0.0) {
        return 0.0;
    }
    if (sel > 1.0) {
        return 1.0;
    }
    return sel;
}

void calc_mean_std(
    const double *array,
    const int n_samples,
    double *res_mean,
    double *res_std
) {
    if (n_samples <= 0) {
        *res_mean = 0.0;
        *res_std = 0.0;
        return;
    }
    double sum = 0.0;
    for (int i = 0; i < n_samples; ++i) {
        sum += array[i];
    }
    double sum_of_squares = 0.0;
    const double mean = sum / n_samples;
    for (int i = 0; i < n_samples; ++i) {
        const double diff = mean - array[i];
        sum_of_squares += diff * diff;
    }
    if (n_samples == 1) {
        *res_std = 0.0;
    } else {
        *res_std = sqrt(sum_of_squares / (n_samples - 1));
    }
}

/* ------------------------------- Samples ------------------------------- */
Sample *initialize_sample(
    const int sample_count
) {
    Sample *dst_sample = palloc0(sizeof(Sample));
    dst_sample->sample_count = sample_count;
    return dst_sample;
}

Sample *duplicate_sample(
    const Sample *src_sample
) {
    Sample *dst_sample = palloc0(sizeof(Sample));
    memcpy(dst_sample, src_sample, sizeof(Sample));
    return dst_sample;
}

Sample *make_sample_by_single_value(
    const double val
) {
    Sample *dst_sample = palloc0(sizeof(Sample));
    dst_sample->sample_count = 1;
    dst_sample->sample[0] = val;
    return dst_sample;
}

Sample *make_sample_by_scale_factor(
    const Sample *src_sample,
    const double factor
) {
    Assert(src_sample != NULL);
    Sample *dst_sample = palloc0(sizeof(Sample));
    Assert(dst_sample != NULL);
    dst_sample->sample_count = src_sample->sample_count;
    for (int i = 0; i < dst_sample->sample_count; ++i) {
        dst_sample->sample[i] = src_sample->sample[i] * factor;
    }
    return dst_sample;
}

/*
 * Build conditional samples of sel_true given sel_est.
 *
 * Source bin stores *samples of log-error*:
 *   log_error = log(sel_true / sel_est)
 *
 * We transform each log-error sample to a sel_true sample via:
 *   y = sel_est * exp(log_error)
 *
 * Output:
 *   - out->sample[k] : the k-th transformed sample y
 *   - out->sample_count : number of samples written (<= DIST_MAX_SAMPLE)
 *
 * Notes:
 *   - This returns *raw samples only* (no probabilities).
 *   - 'ep->error_sample[b]' is assumed to contain log-error samples.
 *   - Static buffer 'out' is overwritten on each call; use palloc0 if needed.
 */
Sample *make_sample_by_bin(const ErrorProfile *ep, const double sel_est) {
    Sample *dst_sample = palloc0(sizeof(Sample));

    /* reset */
    dst_sample->sample_count = 0;
    memset(dst_sample->sample, 0, sizeof(dst_sample->sample));

    if (!ep) {
        return NULL;
    }
    const int b = find_bin_by_sel_est(ep, sel_est);
    if (b < 0) {
        return NULL;
    }

    /* Source bin holds log-error samples */
    const Sample *src = &ep->error_sample[b];
    if (src->sample_count <= 0) {
        return NULL;
    }

    int m = src->sample_count;
    if (m > DIST_MAX_SAMPLE) {
        m = DIST_MAX_SAMPLE;
    }

    /* Transform: y = sel_est * exp(log_error) */
    for (int i = 0; i < m; ++i) {
        /* src->sample[i] stores log(sel_true / sel_est) */
        const double log_err = src->sample[i];
        dst_sample->sample[i] = sel_est * exp(log_err);
    }
    dst_sample->sample_count = m;

    return dst_sample;
}

Sample *make_sample_by_join_sample(
    const Sample *outer_rows_sample,
    const Sample *inner_rows_sample,
    const Sample *sel_sample,
    const int target_samples
) {
    /* Each input must be either a scalar (sample_count==1) or have target_samples */
    Assert(outer_rows_sample->sample_count == 1 || outer_rows_sample->sample_count == target_samples);
    Assert(inner_rows_sample->sample_count == 1 || inner_rows_sample->sample_count == target_samples);
    Assert(sel_sample->sample_count == 1 || sel_sample->sample_count == target_samples);

    /* Whether to broadcast each sample as a constant */
    const bool outer_to_constant = outer_rows_sample->sample_count == 1;
    const bool inner_to_constant = inner_rows_sample->sample_count == 1;
    const bool sel_to_constant = sel_sample->sample_count == 1;

    /* Constant values (used when sample_count==1) */
    const double outer_const_sample = outer_rows_sample->sample[0];
    const double inner_const_sample = inner_rows_sample->sample[0];
    const double sel_const_sample = sel_sample->sample[0];

    /* Fast path: if all are constants, multiply once and return a single-point sample */
    if (outer_to_constant && inner_to_constant && sel_to_constant) {
        Sample *join_sample = make_sample_by_single_value(
            outer_const_sample * inner_const_sample * sel_const_sample
        );
        return join_sample;
    }

    /* Allocate result with target_samples */
    Sample *join_sample = palloc0(sizeof(Sample));
    join_sample->sample_count = target_samples;

    /* Element-wise multiply across i; broadcast any scalar inputs */
    for (int i = 0; i < target_samples; ++i) {
        double current_sample = 1.0;

        /* Outer: use constant when broadcasting, otherwise use i-th sample */
        if (outer_to_constant) {
            current_sample *= outer_const_sample;
        } else {
            current_sample *= outer_rows_sample->sample[i];
        }

        /* Inner: use constant when broadcasting, otherwise use i-th sample */
        if (inner_to_constant) {
            current_sample *= inner_const_sample;
        } else {
            current_sample *= inner_rows_sample->sample[i];
        }

        /* Selectivity: use constant when broadcasting, otherwise use i-th sample */
        if (sel_to_constant) {
            current_sample *= sel_const_sample;
        } else {
            current_sample *= sel_sample->sample[i];
        }

        join_sample->sample[i] = current_sample;
    }

    return join_sample;
}

/* ------------------------------- Relations ------------------------------- */
void set_baserel_rows_sample(
    const PlannerInfo *root,
    RelOptInfo *baserel,
    const double sel_est
) {
    /* 0. Prepare fallback rows estimation result.
     * Note: we don't use `baserel->rows`, which has been clamped already. */
    const double rows_fallback = sel_est * baserel->tuples;

    /* 1. Resolve relation aliases (original alias and a standard fallback). */
    const char *alias = get_alias(root, baserel->relid);
    const char *alias_fallback = get_std_alias(root, baserel->relid);
    // elog(LOG, "[baserel %s] considering relation rows sample.", alias);

    /* 2. Allocate an error profile holder and try to populate it from cache. */
    ErrorProfile *ep;
    const bool found = get_error_profile(alias, alias_fallback, &ep);

    /* 2.1 If no profile is available, fall back to a degenerate sample. */
    if (!found) {
        // elog(LOG, "[baserel %s] no error profile is available, using a single sample.", alias);
        baserel->rows = rows_fallback;
        baserel->rows_sample = make_sample_by_single_value(rows_fallback);
        return;
    }

    /* 3. Get conditional sample p(true_sel | sel_est=e0). */
    const Sample *sel_true_sample = make_sample_by_bin(ep, sel_est);

    /* 3.1 Fallback to a single sample if we fail to build `sel_true_sample`. */
    if (sel_true_sample == NULL) {
        // elog(LOG, "[baserel %s] failed to build conditional sample, using a single sample.", alias);
        baserel->rows = rows_fallback;
        baserel->rows_sample = make_sample_by_single_value(rows_fallback);
        return;
    }

    /* 4. Scale the `sel_true_sample` -- from selectivity sample to rows sample. */
    Sample *rows_sample = make_sample_by_scale_factor(sel_true_sample, baserel->tuples);

    /* 5. Calculate the expectation of the rows sample and update the relation's rows estimation. */
    double rows_sample_mean = 0.0;
    for (int i = 0; i < rows_sample->sample_count; ++i) {
        rows_sample_mean += rows_sample->sample[i];
    }
    rows_sample_mean /= (double) rows_sample->sample_count;

    /* 5.1 Save the rows and rows sample. */
    // elog(LOG, "[baserel %s] original: %g rows -> adjusted: %g rows.", alias, baserel->rows, rows_sample_mean);
    baserel->rows = rows_sample_mean;
    baserel->rows_sample = rows_sample;
}

void set_joinrel_rows_sample(
    const PlannerInfo *root,
    RelOptInfo *joinrel,
    const RelOptInfo *outer_rel,
    const RelOptInfo *inner_rel,
    List *restrictlist,
    const double sel_est
) {
    /* 0. Prepare fallback rows estimation result.
     * Note: we don't use `joinrel->rows`, which has been clamped already. */
    double rows_fallback;
    if (root->pass == 1) {
        /* Determine the outer relation's single-point row estimate. */
        double outer_rows_single_point;
        if (outer_rel->relid > 0 && outer_rel->rows_sample->sample_count > 1) {
            outer_rows_single_point = outer_rel->rows_sample->sample[root->round];
        } else {
            outer_rows_single_point = outer_rel->rows;
        }
        /* Determine the inner relation's single-point row estimate. */
        double inner_rows_single_point;
        if (inner_rel->relid > 0 && inner_rel->rows_sample->sample_count > 1) {
            inner_rows_single_point = inner_rel->rows_sample->sample[root->round];
        } else {
            inner_rows_single_point = inner_rel->rows;
        }
        rows_fallback = sel_est * outer_rows_single_point * inner_rows_single_point;
    } else {
        rows_fallback = sel_est * outer_rel->rows * inner_rel->rows;
    }

    /* 1. Resolve relation aliases (original alias and a standard fallback). */
    // FIXME: We assume that the first `rinfo` we found `can_join`.
    // FIXME: We should also check `rinfo->hashjoinoperator`.
    char alias[SM_KEY_LEN];
    char alias_fallback[SM_KEY_LEN];
    memset(alias, 0, sizeof(alias));
    memset(alias_fallback, 0, sizeof(alias_fallback));

    ListCell *lc;
    foreach(lc, restrictlist) {
        const RestrictInfo *rinfo = lfirst(lc);
        if (!IsA(rinfo->clause, OpExpr)) {
            continue;
        }
        const OpExpr *opexpr = (OpExpr *) rinfo->clause;
        if (list_length(opexpr->args) != 2) {
            continue;
        }

        /* Notes: Both sides must be Vars (skip RelabelType etc.). */
        Node *l = linitial(opexpr->args);
        Node *r = lsecond(opexpr->args);
        if (!IsA(l, Var) || !IsA(r, Var)) {
            continue;
        }
        const Var *leftvar = (Var *) l;
        const Var *rightvar = (Var *) r;

        /* Accept either orientation: (outer,left)-(inner,right) or swapped. */
        const bool l_in_outer = bms_is_member(leftvar->varno, outer_rel->relids);
        const bool r_in_inner = bms_is_member(rightvar->varno, inner_rel->relids);
        const bool r_in_outer = bms_is_member(rightvar->varno, outer_rel->relids);
        const bool l_in_inner = bms_is_member(leftvar->varno, inner_rel->relids);

        if ((l_in_outer && r_in_inner) || (r_in_outer && l_in_inner)) {
            const char *left_rel_alias = get_alias(root, leftvar->varno);
            const char *right_rel_alias = get_alias(root, rightvar->varno);
            const char *left_rel_std_alias = get_std_alias(root, leftvar->varno);
            const char *right_rel_std_alias = get_std_alias(root, rightvar->varno);

            /* Canonicalize order to avoid duplicate “A=B” vs “B=A”. */
            if (strcmp(left_rel_alias, right_rel_alias) < 0)
                snprintf(alias, sizeof(alias), "%s=%s", left_rel_alias, right_rel_alias);
            else
                snprintf(alias, sizeof(alias), "%s=%s", right_rel_alias, left_rel_alias);

            // elog(LOG, "alias: %s; join key: %s.%d = %s.%d.",
            //      alias, left_rel_alias, leftvar->varattno,
            //      right_rel_alias, rightvar->varattno);

            /* Canonicalize order to avoid duplicate “A=B” vs “B=A”. */
            if (strcmp(left_rel_std_alias, right_rel_std_alias) < 0)
                snprintf(alias_fallback, sizeof(alias_fallback), "%s=%s", left_rel_std_alias, right_rel_std_alias);
            else
                snprintf(alias_fallback, sizeof(alias_fallback), "%s=%s", right_rel_std_alias, left_rel_std_alias);

            // elog(LOG, "alias fallback: %s; join key: %s.%d = %s.%d.",
            //      alias_fallback, left_rel_std_alias, leftvar->varattno,
            //      right_rel_std_alias, rightvar->varattno);

            break;
        }
    }
    // TODO: Check whether we have a sane alias with its fallback version.
    // elog(LOG, "[joinrel %s] considering relation rows sample.", alias);

    /* 2. Allocate an error profile holder and try to populate it from cache. */
    ErrorProfile *ep;
    const bool found = get_error_profile(alias, alias_fallback, &ep);

    /* 2.1 If no profile is available, fall back to a degenerate sample. */
    if (!found) {
        // elog(LOG, "[joinrel %s] no profile is available, using a single sample.", alias);
        joinrel->rows = rows_fallback;
        joinrel->rows_sample = make_sample_by_single_value(rows_fallback);
        return;
    }

    /* 3. Get conditional sample p(true_sel | sel_est=e0). */
    const Sample *sel_true_sample = make_sample_by_bin(ep, sel_est);

    /* 3.1 Fallback to single point sample if we fail to build `sel_true_sample`. */
    if (sel_true_sample == NULL) {
        // elog(LOG, "[joinrel %s] failed to build conditional sample, using a single sample.", alias);
        joinrel->rows = rows_fallback;
        joinrel->rows_sample = make_sample_by_single_value(rows_fallback);
        return;
    }

    /* 4. Push selectivity uncertainty through the join-size model to get rows sample.
     * Notes: we assume that both outer relation's and inner relation's rows sample exist. */
    Sample *rows_sample = make_sample_by_join_sample(
        outer_rel->rows_sample,
        inner_rel->rows_sample,
        sel_true_sample,
        error_sample_count
    );

    /* 5. Calculate the expectation of the rows sample and update the relation's rows estimation. */
    double rows_sample_mean = 0.0;
    for (int i = 0; i < rows_sample->sample_count; ++i) {
        rows_sample_mean += rows_sample->sample[i];
    }
    rows_sample_mean /= (double) rows_sample->sample_count;

    /* 5.1 Save the rows and rows sample. */
    // elog(LOG, "[joinrel %s] original: %g rows -> adjusted: %g rows.", alias, joinrel->rows, rows_sample_mean);
    joinrel->rows = rows_sample_mean;
    joinrel->rows_sample = rows_sample;
}
