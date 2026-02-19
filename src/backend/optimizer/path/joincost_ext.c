//
// Created by Xuan Chen on 2025/11/1.
//

#include "postgres.h"

#include <limits.h>
#include <math.h>

#include "access/amapi.h"
#include "access/htup_details.h"
#include "executor/executor.h"
#include "executor/nodeAgg.h"
#include "executor/nodeHash.h"
#include "executor/nodeMemoize.h"
#include "miscadmin.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/cost.h"
#include "optimizer/cost_ext.h"
#include "optimizer/sample.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/restrictinfo.h"
#include "utils/lsyscache.h"
#include "utils/selfuncs.h"
#include "utils/tuplesort.h"


/* ==== ==== ==== ==== ==== ==== JOIN COST HELPERS ==== ==== ==== ==== ==== ==== */
double get_path_rows_1p(
    const Path *path,
    const int round
) {
    /*
     * 1-pass DP row retrieval.
     *
     * For joinrels (`relid == 0`), every path row estimate is already a
     * single scalar value. No per-round sampling applies here, so we just
     * return `path->rows`.
     */
    if (path->parent->relid == 0)
        return path->rows;
    /*
     * Baserel case: rows or rows_sample may carry multi-point uncertainty.
     *
     * If `sample_count > 1`, the baserel has an error profile, and the
     * correct row estimate for this DP round is taken from the sample vector.
     *
     * If `sample_count == 1`, the rows_sample is a single-point structure
     * equal to the mean; in this case returning the scalar `rows` is correct.
     */
    if (path->rows_sample->sample_count > 1)
        return path->rows_sample->sample[round];

    return path->rows;
}

/*
 * Estimate the fraction of the work that each worker will do given the
 * number of workers budgeted for the path.
 */
double get_parallel_divisor(
    const Path *path
) {
    double parallel_divisor = path->parallel_workers;

    /*
     * Early experience with parallel query suggests that when there is only
     * one worker, the leader often makes a very substantial contribution to
     * executing the parallel portion of the plan, but as more workers are
     * added, it does less and less, because it's busy reading tuples from the
     * workers and doing whatever non-parallel post-processing is needed.  By
     * the time we reach 4 workers, the leader no longer makes a meaningful
     * contribution.  Thus, for now, estimate that the leader spends 30% of
     * its time servicing each worker, and the remainder executing the
     * parallel plan.
     */
    if (parallel_leader_participation) {
        const double leader_contribution = 1.0 - (0.3 * path->parallel_workers);
        if (leader_contribution > 0)
            parallel_divisor += leader_contribution;
    }

    return parallel_divisor;
}

/*
 * approx_tuple_count
 *		Quick-and-dirty estimation of the number of join rows passing
 *		a set of qual conditions.
 *
 * The quals can be either an implicitly-ANDed list of boolean expressions,
 * or a list of RestrictInfo nodes (typically the latter).
 *
 * We intentionally compute the selectivity under JOIN_INNER rules, even
 * if it's some type of outer join.  This is appropriate because we are
 * trying to figure out how many tuples pass the initial merge or hash
 * join step.
 *
 * This is quick-and-dirty because we bypass clauselist_selectivity, and
 * simply multiply the independent clause selectivities together.  Now
 * clauselist_selectivity often can't do any better than that anyhow, but
 * for some situations (such as range constraints) it is smarter.  However,
 * we can't effectively cache the results of clauselist_selectivity, whereas
 * the individual clause selectivities can be and are cached.
 *
 * Since we are only using the results to estimate how many potential
 * output tuples are generated and passed through qpqual checking, it
 * seems OK to live with the approximation.
 */
double approx_tuple_count(
    PlannerInfo *root,
    const JoinPath *path,
    List *quals
) {
    double tuples;
    double outer_tuples = path->outerjoinpath->rows;
    double inner_tuples = path->innerjoinpath->rows;
    SpecialJoinInfo sjinfo;
    Selectivity selec = 1.0;
    ListCell *l;

    /*
     * Make up a SpecialJoinInfo for JOIN_INNER semantics.
     */
    sjinfo.type = T_SpecialJoinInfo;
    sjinfo.min_lefthand = path->outerjoinpath->parent->relids;
    sjinfo.min_righthand = path->innerjoinpath->parent->relids;
    sjinfo.syn_lefthand = path->outerjoinpath->parent->relids;
    sjinfo.syn_righthand = path->innerjoinpath->parent->relids;
    sjinfo.jointype = JOIN_INNER;
    sjinfo.ojrelid = 0;
    sjinfo.commute_above_l = NULL;
    sjinfo.commute_above_r = NULL;
    sjinfo.commute_below_l = NULL;
    sjinfo.commute_below_r = NULL;
    /* we don't bother trying to make the remaining fields valid */
    sjinfo.lhs_strict = false;
    sjinfo.semi_can_btree = false;
    sjinfo.semi_can_hash = false;
    sjinfo.semi_operators = NIL;
    sjinfo.semi_rhs_exprs = NIL;

    /* Get the approximate selectivity */
    foreach(l, quals) {
        Node *qual = (Node *) lfirst(l);

        /* Note that clause_selectivity will be able to cache its result */
        selec *= clause_selectivity(root, qual, 0, JOIN_INNER, &sjinfo);
    }

    /* Apply it to the input relation sizes */
    tuples = selec * outer_tuples * inner_tuples;

    return clamp_row_est(tuples);
}

/*
 * get_expr_width
 *		Estimate the width of the given expr attempting to use the width
 *		cached in a Var's owning RelOptInfo, else fallback on the type's
 *		average width when unable to or when the given Node is not a Var.
 */
int32 get_expr_width(
    const PlannerInfo *root,
    const Node *expr
) {
    int32 width;

    if (IsA(expr, Var)) {
        const Var *var = (const Var *) expr;

        /* We should not see any upper-level Vars here */
        Assert(var->varlevelsup == 0);

        /* Try to get data from RelOptInfo cache */
        if (!IS_SPECIAL_VARNO(var->varno) &&
            var->varno < root->simple_rel_array_size) {
            RelOptInfo *rel = root->simple_rel_array[var->varno];

            if (rel != NULL &&
                var->varattno >= rel->min_attr &&
                var->varattno <= rel->max_attr) {
                int ndx = var->varattno - rel->min_attr;

                if (rel->attr_widths[ndx] > 0)
                    return rel->attr_widths[ndx];
            }
        }

        /*
         * No cached data available, so estimate using just the type info.
         */
        width = get_typavgwidth(var->vartype, var->vartypmod);
        Assert(width > 0);

        return width;
    }

    width = get_typavgwidth(exprType(expr), exprTypmod(expr));
    Assert(width > 0);
    return width;
}

/*
 * relation_byte_size
 *	  Estimate the storage space in bytes for a given number of tuples
 *	  of a given width (size in bytes).
 */
double relation_byte_size(
    const double tuples,
    const int width
) {
    return tuples * (MAXALIGN(width) + MAXALIGN(SizeofHeapTupleHeader));
}

/*
 * page_size
 *	  Returns an estimate of the number of pages covered by a given
 *	  number of tuples of a given width (size in bytes).
 */
double page_size(
    const double tuples,
    const int width
) {
    return ceil(relation_byte_size(tuples, width) / BLCKSZ);
}

/*
 * has_indexed_join_quals
 *	  Check whether all the joinquals of a nestloop join are used as
 *	  inner index quals.
 *
 * If the inner path of a SEMI/ANTI join is an indexscan (including bitmap
 * indexscan) that uses all the joinquals as indexquals, we can assume that an
 * unmatched outer tuple is cheap to process, whereas otherwise it's probably
 * expensive.
 */
bool has_indexed_join_quals(
    NestPath *path
) {
    JoinPath *joinpath = &path->jpath;
    Relids joinrelids = joinpath->path.parent->relids;
    Path *innerpath = joinpath->innerjoinpath;
    List *indexclauses;
    bool found_one;
    ListCell *lc;

    /* If join still has quals to evaluate, it's not fast */
    if (joinpath->joinrestrictinfo != NIL)
        return false;
    /* Nor if the inner path isn't parameterized at all */
    if (innerpath->param_info == NULL)
        return false;

    /* Find the indexclauses list for the inner scan */
    switch (innerpath->pathtype) {
        case T_IndexScan:
        case T_IndexOnlyScan:
            indexclauses = ((IndexPath *) innerpath)->indexclauses;
            break;
        case T_BitmapHeapScan: {
            /* Accept only a simple bitmap scan, not AND/OR cases */
            Path *bmqual = ((BitmapHeapPath *) innerpath)->bitmapqual;

            if (IsA(bmqual, IndexPath))
                indexclauses = ((IndexPath *) bmqual)->indexclauses;
            else
                return false;
            break;
        }
        default:

            /*
             * If it's not a simple indexscan, it probably doesn't run quickly
             * for zero rows out, even if it's a parameterized path using all
             * the joinquals.
             */
            return false;
    }

    /*
     * Examine the inner path's param clauses.  Any that are from the outer
     * path must be found in the indexclauses list, either exactly or in an
     * equivalent form generated by equivclass.c.  Also, we must find at least
     * one such clause, else it's a clauseless join which isn't fast.
     */
    found_one = false;
    foreach(lc, innerpath->param_info->ppi_clauses) {
        RestrictInfo *rinfo = (RestrictInfo *) lfirst(lc);

        if (join_clause_is_movable_into(rinfo,
                                        innerpath->parent->relids,
                                        joinrelids)) {
            if (!is_redundant_with_indexclauses(rinfo, indexclauses))
                return false;
            found_one = true;
        }
    }
    return found_one;
}

/*
 * run mergejoinscansel() with caching
 */
static MergeScanSelCache *cached_scansel(
    PlannerInfo *root,
    RestrictInfo *rinfo,
    const PathKey *pathkey
) {
    MergeScanSelCache *cache;
    ListCell *lc;
    Selectivity leftstartsel,
            leftendsel,
            rightstartsel,
            rightendsel;
    MemoryContext oldcontext;

    /* Do we have this result already? */
    foreach(lc, rinfo->scansel_cache) {
        cache = (MergeScanSelCache *) lfirst(lc);
        if (cache->opfamily == pathkey->pk_opfamily &&
            cache->collation == pathkey->pk_eclass->ec_collation &&
            cache->strategy == pathkey->pk_strategy &&
            cache->nulls_first == pathkey->pk_nulls_first)
            return cache;
    }

    /* Nope, do the computation */
    mergejoinscansel(
        root,
        (Node *) rinfo->clause,
        pathkey->pk_opfamily,
        pathkey->pk_strategy,
        pathkey->pk_nulls_first,
        &leftstartsel,
        &leftendsel,
        &rightstartsel,
        &rightendsel
    );

    /* Cache the result in suitably long-lived workspace */
    oldcontext = MemoryContextSwitchTo(root->planner_cxt);

    cache = (MergeScanSelCache *) palloc(sizeof(MergeScanSelCache));
    cache->opfamily = pathkey->pk_opfamily;
    cache->collation = pathkey->pk_eclass->ec_collation;
    cache->strategy = pathkey->pk_strategy;
    cache->nulls_first = pathkey->pk_nulls_first;
    cache->leftstartsel = leftstartsel;
    cache->leftendsel = leftendsel;
    cache->rightstartsel = rightstartsel;
    cache->rightendsel = rightendsel;

    rinfo->scansel_cache = lappend(rinfo->scansel_cache, cache);

    MemoryContextSwitchTo(oldcontext);

    return cache;
}

/*
 * cost_memoize_rescan
 *	  Determines the estimated cost of rescanning a Memoize node.
 *
 * In order to estimate this, we must gain knowledge of how often we expect to
 * be called and how many distinct sets of parameters we are likely to be
 * called with. If we expect a good cache hit ratio, then we can set our
 * costs to account for that hit ratio, plus a little bit of cost for the
 * caching itself.  Caching will not work out well if we expect to be called
 * with too many distinct parameter values.  The worst-case here is that we
 * never see any parameter value twice, in which case we'd never get a cache
 * hit and caching would be a complete waste of effort.
 */
static void cost_memoize_rescan(
    PlannerInfo *root,
    MemoizePath *mpath,
    Cost *rescan_startup_cost,
    Cost *rescan_total_cost
) {
    EstimationInfo estinfo;
    ListCell *lc;
    Cost input_startup_cost = mpath->subpath->startup_cost;
    Cost input_total_cost = mpath->subpath->total_cost;
    double tuples = mpath->subpath->rows;
    double calls = mpath->calls;
    int width = mpath->subpath->pathtarget->width;

    double hash_mem_bytes;
    double est_entry_bytes;
    double est_cache_entries;
    double ndistinct;
    double evict_ratio;
    double hit_ratio;
    Cost startup_cost;
    Cost total_cost;

    /* available cache space */
    hash_mem_bytes = get_hash_memory_limit();

    /*
     * Set the number of bytes each cache entry should consume in the cache.
     * To provide us with better estimations on how many cache entries we can
     * store at once, we make a call to the executor here to ask it what
     * memory overheads there are for a single cache entry.
     */
    est_entry_bytes = relation_byte_size(tuples, width) +
                      ExecEstimateCacheEntryOverheadBytes(tuples);

    /* include the estimated width for the cache keys */
    foreach(lc, mpath->param_exprs)
        est_entry_bytes += get_expr_width(root, (Node *) lfirst(lc));

    /* estimate on the upper limit of cache entries we can hold at once */
    est_cache_entries = floor(hash_mem_bytes / est_entry_bytes);

    /* estimate on the distinct number of parameter values */
    ndistinct = estimate_num_groups(root, mpath->param_exprs, calls, NULL,
                                    &estinfo);

    /*
     * When the estimation fell back on using a default value, it's a bit too
     * risky to assume that it's ok to use a Memoize node.  The use of a
     * default could cause us to use a Memoize node when it's really
     * inappropriate to do so.  If we see that this has been done, then we'll
     * assume that every call will have unique parameters, which will almost
     * certainly mean a MemoizePath will never survive add_path().
     */
    if ((estinfo.flags & SELFLAG_USED_DEFAULT) != 0)
        ndistinct = calls;

    /*
     * Since we've already estimated the maximum number of entries we can
     * store at once and know the estimated number of distinct values we'll be
     * called with, we'll take this opportunity to set the path's est_entries.
     * This will ultimately determine the hash table size that the executor
     * will use.  If we leave this at zero, the executor will just choose the
     * size itself.  Really this is not the right place to do this, but it's
     * convenient since everything is already calculated.
     */
    mpath->est_entries = Min(Min(ndistinct, est_cache_entries),
                             PG_UINT32_MAX);

    /*
     * When the number of distinct parameter values is above the amount we can
     * store in the cache, then we'll have to evict some entries from the
     * cache.  This is not free. Here we estimate how often we'll incur the
     * cost of that eviction.
     */
    evict_ratio = 1.0 - Min(est_cache_entries, ndistinct) / ndistinct;

    /*
     * In order to estimate how costly a single scan will be, we need to
     * attempt to estimate what the cache hit ratio will be.  To do that we
     * must look at how many scans are estimated in total for this node and
     * how many of those scans we expect to get a cache hit.
     */
    hit_ratio = ((calls - ndistinct) / calls) *
                (est_cache_entries / Max(ndistinct, est_cache_entries));

    Assert(hit_ratio >= 0 && hit_ratio <= 1.0);

    /*
     * Set the total_cost accounting for the expected cache hit ratio.  We
     * also add on a cpu_operator_cost to account for a cache lookup. This
     * will happen regardless of whether it's a cache hit or not.
     */
    total_cost = input_total_cost * (1.0 - hit_ratio) + cpu_operator_cost;

    /* Now adjust the total cost to account for cache evictions */

    /* Charge a cpu_tuple_cost for evicting the actual cache entry */
    total_cost += cpu_tuple_cost * evict_ratio;

    /*
     * Charge a 10th of cpu_operator_cost to evict every tuple in that entry.
     * The per-tuple eviction is really just a pfree, so charging a whole
     * cpu_operator_cost seems a little excessive.
     */
    total_cost += cpu_operator_cost / 10.0 * evict_ratio * tuples;

    /*
     * Now adjust for storing things in the cache, since that's not free
     * either.  Everything must go in the cache.  We don't proportion this
     * over any ratio, just apply it once for the scan.  We charge a
     * cpu_tuple_cost for the creation of the cache entry and also a
     * cpu_operator_cost for each tuple we expect to cache.
     */
    total_cost += cpu_tuple_cost + cpu_operator_cost * tuples;

    /*
     * Getting the first row must be also be proportioned according to the
     * expected cache hit ratio.
     */
    startup_cost = input_startup_cost * (1.0 - hit_ratio);

    /*
     * Additionally we charge a cpu_tuple_cost to account for cache lookups,
     * which we'll do regardless of whether it was a cache hit or not.
     */
    startup_cost += cpu_tuple_cost;

    *rescan_startup_cost = startup_cost;
    *rescan_total_cost = total_cost;
}

/*
 * cost_rescan
 *		Given a finished Path, estimate the costs of rescanning it after
 *		having done so the first time.  For some Path types a rescan is
 *		cheaper than an original scan (if no parameters change), and this
 *		function embodies knowledge about that.  The default is to return
 *		the same costs stored in the Path.  (Note that the cost estimates
 *		actually stored in Paths are always for first scans.)
 *
 * This function is not currently intended to model effects such as rescans
 * being cheaper due to disk block caching; what we are concerned with is
 * plan types wherein the executor caches results explicitly, or doesn't
 * redo startup calculations, etc.
 */
static void cost_rescan(
    PlannerInfo *root,
    Path *path,
    Cost *rescan_startup_cost, /* output parameters */
    Cost *rescan_total_cost
) {
    switch (path->pathtype) {
        case T_FunctionScan:

            /*
             * Currently, nodeFunctionscan.c always executes the function to
             * completion before returning any rows, and caches the results in
             * a tuplestore.  So the function eval cost is all startup cost
             * and isn't paid over again on rescans. However, all run costs
             * will be paid over again.
             */
            *rescan_startup_cost = 0;
            *rescan_total_cost = path->total_cost - path->startup_cost;
            break;
        case T_HashJoin:

            /*
             * If it's a single-batch join, we don't need to rebuild the hash
             * table during a rescan.
             */
            if (((HashPath *) path)->num_batches == 1) {
                /* Startup cost is exactly the cost of hash table building */
                *rescan_startup_cost = 0;
                *rescan_total_cost = path->total_cost - path->startup_cost;
            } else {
                /* Otherwise, no special treatment */
                *rescan_startup_cost = path->startup_cost;
                *rescan_total_cost = path->total_cost;
            }
            break;
        case T_CteScan:
        case T_WorkTableScan: {
            /*
             * These plan types materialize their final result in a
             * tuplestore or tuplesort object.  So the rescan cost is only
             * cpu_tuple_cost per tuple, unless the result is large enough
             * to spill to disk.
             */
            Cost run_cost = cpu_tuple_cost * path->rows;
            double nbytes = relation_byte_size(path->rows, path->pathtarget->width);
            long work_mem_bytes = work_mem * 1024L;

            if (nbytes > work_mem_bytes) {
                /* It will spill, so account for re-read cost */
                double npages = ceil(nbytes / BLCKSZ);

                run_cost += seq_page_cost * npages;
            }
            *rescan_startup_cost = 0;
            *rescan_total_cost = run_cost;
        }
        break;
        case T_Material:
        case T_Sort: {
            /*
             * These plan types not only materialize their results, but do
             * not implement qual filtering or projection.  So they are
             * even cheaper to rescan than the ones above.  We charge only
             * cpu_operator_cost per tuple.  (Note: keep that in sync with
             * the run_cost charge in cost_sort, and also see comments in
             * cost_material before you change it.)
             */
            Cost run_cost = cpu_operator_cost * path->rows;
            double nbytes = relation_byte_size(path->rows, path->pathtarget->width);
            long work_mem_bytes = work_mem * 1024L;

            if (nbytes > work_mem_bytes) {
                /* It will spill, so account for re-read cost */
                double npages = ceil(nbytes / BLCKSZ);

                run_cost += seq_page_cost * npages;
            }
            *rescan_startup_cost = 0;
            *rescan_total_cost = run_cost;
        }
        break;
        case T_Memoize:
            /* All the hard work is done by cost_memoize_rescan */
            cost_memoize_rescan(root, (MemoizePath *) path,
                                rescan_startup_cost, rescan_total_cost);
            break;
        default:
            *rescan_startup_cost = path->startup_cost;
            *rescan_total_cost = path->total_cost;
            break;
    }
}

/* ==== ==== ==== ==== ==== ==== 1-PASS NEST LOOP COST MODEL ==== ==== ==== ==== ==== ==== */

void initial_cost_nestloop_1p(
    PlannerInfo *root,
    JoinCostWorkspace *workspace,
    JoinType jointype,
    const Path *outer_path,
    Path *inner_path,
    const JoinPathExtraData *extra
) {
    Cost startup_cost = 0;
    Cost run_cost = 0;
    /*
     * Use per-round sampled row estimates for base scan paths (Seq/Index).
     * In single-sample mode (sample_count == 1) or non-scan paths,
     * fall back to the planner's deterministic row estimate.
     */
    const double outer_path_rows = outer_path->rows;

    /* Use per-round sampled costs for base scans; otherwise deterministic costs */
    const Cost outer_startup_cost = outer_path->startup_cost;
    const Cost outer_total_cost = outer_path->total_cost;
    const Cost inner_startup_cost = inner_path->startup_cost;
    const Cost inner_total_cost = inner_path->total_cost;

    Cost inner_rescan_start_cost;
    Cost inner_rescan_total_cost;
    Cost inner_run_cost;
    Cost inner_rescan_run_cost;

    /* estimate costs to rescan the inner relation */
    cost_rescan(root, inner_path,
                &inner_rescan_start_cost,
                &inner_rescan_total_cost);

    /* cost of source data */

    /*
     * NOTE: clearly, we must pay both outer and inner paths' startup_cost
     * before we can start returning tuples, so the join's startup cost is
     * their sum.  We'll also pay the inner path's rescan startup cost
     * multiple times.
     */
    startup_cost += outer_startup_cost + inner_startup_cost;
    run_cost += outer_total_cost - outer_startup_cost;
    if (outer_path_rows > 1)
        run_cost += (outer_path_rows - 1) * inner_rescan_start_cost;

    inner_run_cost = inner_total_cost - inner_startup_cost;
    inner_rescan_run_cost = inner_rescan_total_cost - inner_rescan_start_cost;

    if (jointype == JOIN_SEMI || jointype == JOIN_ANTI ||
        extra->inner_unique) {
        /*
         * With a SEMI or ANTI join, or if the innerrel is known unique, the
         * executor will stop after the first match.
         *
         * Getting decent estimates requires inspection of the join quals,
         * which we choose to postpone to final_cost_nestloop.
         */

        /* Save private data for final_cost_nestloop */
        workspace->inner_run_cost = inner_run_cost;
        workspace->inner_rescan_run_cost = inner_rescan_run_cost;
    } else {
        /* Normal case; we'll scan whole input rel for each outer row */
        run_cost += inner_run_cost;
        if (outer_path_rows > 1)
            run_cost += (outer_path_rows - 1) * inner_rescan_run_cost;
    }

    /* CPU costs left for later */

    /* Public result fields */
    workspace->startup_cost = startup_cost;
    workspace->total_cost = startup_cost + run_cost;
    /* Save private data for final_cost_nestloop */
    workspace->run_cost = run_cost;
}

void final_cost_nestloop_1p(
    PlannerInfo *root,
    NestPath *path,
    JoinCostWorkspace *workspace,
    const JoinPathExtraData *extra
) {
    Path *outer_path = path->jpath.outerjoinpath;
    Path *inner_path = path->jpath.innerjoinpath;
    /*
     * Use per-round sampled row estimates for base scan paths (Seq/Index).
     * In single-sample mode (sample_count == 1) or non-scan paths,
     * fall back to the planner's deterministic row estimate.
     */
    double outer_path_rows = outer_path->rows;
    /*
     * Same logic for inner path. Use sampled rows only for scan paths
     * when running in multi-sample mode.
     */
    double inner_path_rows = inner_path->rows;

    Cost startup_cost = workspace->startup_cost;
    Cost run_cost = workspace->run_cost;
    Cost cpu_per_tuple;
    QualCost restrict_qual_cost;
    double ntuples;

    /* Protect some assumptions below that rowcounts aren't zero */
    if (outer_path_rows <= 0)
        outer_path_rows = 1;
    if (inner_path_rows <= 0)
        inner_path_rows = 1;
    /* Mark the path with the correct row estimate */
    if (path->jpath.path.param_info) {
        path->jpath.path.rows = path->jpath.path.param_info->ppi_rows;
    } else {
        path->jpath.path.rows = path->jpath.path.parent->rows;
    }
    path->jpath.path.rows_sample = NULL;

    /* For partial paths, scale row estimate. */
    if (path->jpath.path.parallel_workers > 0) {
        double parallel_divisor = get_parallel_divisor(&path->jpath.path);

        path->jpath.path.rows =
                clamp_row_est(path->jpath.path.rows / parallel_divisor);
    }

    /*
     * We could include disable_cost in the preliminary estimate, but that
     * would amount to optimizing for the case where the join method is
     * disabled, which doesn't seem like the way to bet.
     */
    if (!enable_nestloop)
        startup_cost += disable_cost;

    /* cost of inner-relation source data (we already dealt with outer rel) */

    if (path->jpath.jointype == JOIN_SEMI || path->jpath.jointype == JOIN_ANTI ||
        extra->inner_unique) {
        /*
         * With a SEMI or ANTI join, or if the innerrel is known unique, the
         * executor will stop after the first match.
         */
        Cost inner_run_cost = workspace->inner_run_cost;
        Cost inner_rescan_run_cost = workspace->inner_rescan_run_cost;
        double outer_matched_rows;
        double outer_unmatched_rows;
        Selectivity inner_scan_frac;

        /*
         * For an outer-rel row that has at least one match, we can expect the
         * inner scan to stop after a fraction 1/(match_count+1) of the inner
         * rows, if the matches are evenly distributed.  Since they probably
         * aren't quite evenly distributed, we apply a fuzz factor of 2.0 to
         * that fraction.  (If we used a larger fuzz factor, we'd have to
         * clamp inner_scan_frac to at most 1.0; but since match_count is at
         * least 1, no such clamp is needed now.)
         */
        outer_matched_rows = rint(outer_path_rows * extra->semifactors.outer_match_frac);
        outer_unmatched_rows = outer_path_rows - outer_matched_rows;
        inner_scan_frac = 2.0 / (extra->semifactors.match_count + 1.0);

        /*
         * Compute number of tuples processed (not number emitted!).  First,
         * account for successfully-matched outer rows.
         */
        ntuples = outer_matched_rows * inner_path_rows * inner_scan_frac;

        /*
         * Now we need to estimate the actual costs of scanning the inner
         * relation, which may be quite a bit less than N times inner_run_cost
         * due to early scan stops.  We consider two cases.  If the inner path
         * is an indexscan using all the joinquals as indexquals, then an
         * unmatched outer row results in an indexscan returning no rows,
         * which is probably quite cheap.  Otherwise, the executor will have
         * to scan the whole inner rel for an unmatched row; not so cheap.
         */
        if (has_indexed_join_quals(path)) {
            /*
             * Successfully-matched outer rows will only require scanning
             * inner_scan_frac of the inner relation.  In this case, we don't
             * need to charge the full inner_run_cost even when that's more
             * than inner_rescan_run_cost, because we can assume that none of
             * the inner scans ever scan the whole inner relation.  So it's
             * okay to assume that all the inner scan executions can be
             * fractions of the full cost, even if materialization is reducing
             * the rescan cost.  At this writing, it's impossible to get here
             * for a materialized inner scan, so inner_run_cost and
             * inner_rescan_run_cost will be the same anyway; but just in
             * case, use inner_run_cost for the first matched tuple and
             * inner_rescan_run_cost for additional ones.
             */
            run_cost += inner_run_cost * inner_scan_frac;
            if (outer_matched_rows > 1)
                run_cost += (outer_matched_rows - 1) * inner_rescan_run_cost * inner_scan_frac;

            /*
             * Add the cost of inner-scan executions for unmatched outer rows.
             * We estimate this as the same cost as returning the first tuple
             * of a nonempty scan.  We consider that these are all rescans,
             * since we used inner_run_cost once already.
             */
            run_cost += outer_unmatched_rows *
                    inner_rescan_run_cost / inner_path_rows;

            /*
             * We won't be evaluating any quals at all for unmatched rows, so
             * don't add them to ntuples.
             */
        } else {
            /*
             * Here, a complicating factor is that rescans may be cheaper than
             * first scans.  If we never scan all the way to the end of the
             * inner rel, it might be (depending on the plan type) that we'd
             * never pay the whole inner first-scan run cost.  However it is
             * difficult to estimate whether that will happen (and it could
             * not happen if there are any unmatched outer rows!), so be
             * conservative and always charge the whole first-scan cost once.
             * We consider this charge to correspond to the first unmatched
             * outer row, unless there isn't one in our estimate, in which
             * case blame it on the first matched row.
             */

            /* First, count all unmatched join tuples as being processed */
            ntuples += outer_unmatched_rows * inner_path_rows;

            /* Now add the forced full scan, and decrement appropriate count */
            run_cost += inner_run_cost;
            if (outer_unmatched_rows >= 1)
                outer_unmatched_rows -= 1;
            else
                outer_matched_rows -= 1;

            /* Add inner run cost for additional outer tuples having matches */
            if (outer_matched_rows > 0)
                run_cost += outer_matched_rows * inner_rescan_run_cost * inner_scan_frac;

            /* Add inner run cost for additional unmatched outer tuples */
            if (outer_unmatched_rows > 0)
                run_cost += outer_unmatched_rows * inner_rescan_run_cost;
        }
    } else {
        /* Normal-case source costs were included in preliminary estimate */

        /* Compute number of tuples processed (not number emitted!) */
        ntuples = outer_path_rows * inner_path_rows;
    }

    /* CPU costs */
    cost_qual_eval(&restrict_qual_cost, path->jpath.joinrestrictinfo, root);
    startup_cost += restrict_qual_cost.startup;
    cpu_per_tuple = cpu_tuple_cost + restrict_qual_cost.per_tuple;
    run_cost += cpu_per_tuple * ntuples;

    /* tlist eval costs are paid per output row, not per tuple scanned */
    startup_cost += path->jpath.path.pathtarget->cost.startup;
    run_cost += path->jpath.path.pathtarget->cost.per_tuple * path->jpath.path.rows;

    path->jpath.path.startup_cost = startup_cost;
    path->jpath.path.total_cost = startup_cost + run_cost;
}

/* ==== ==== ==== ==== ==== ==== 1-PASS MERGE JOIN COST MODEL ==== ==== ==== ==== ==== ==== */

void initial_cost_mergejoin_1p(
    PlannerInfo *root,
    JoinCostWorkspace *workspace,
    JoinType jointype,
    List *mergeclauses,
    Path *outer_path,
    Path *inner_path,
    List *outersortkeys,
    List *innersortkeys,
    JoinPathExtraData *extra
) {
    Cost startup_cost = 0;
    Cost run_cost = 0;
    /*
     * Use per-round sampled row estimates for base scan paths (Seq/Index).
     * In single-sample mode (sample_count == 1) or non-scan paths,
     * fall back to the planner's deterministic row estimate.
     */
    double outer_path_rows = outer_path->rows;
    /*
     * Same logic for inner path. Use sampled rows only for scan paths
     * when running in multi-sample mode.
     */
    double inner_path_rows = inner_path->rows;

    /* Use per-round sampled costs for base scans; otherwise deterministic costs */
    const Cost outer_startup_cost = outer_path->startup_cost;
    const Cost outer_total_cost = outer_path->total_cost;
    const Cost inner_startup_cost = inner_path->startup_cost;
    const Cost inner_total_cost = inner_path->total_cost;

    Cost inner_run_cost;
    double outer_rows,
            inner_rows,
            outer_skip_rows,
            inner_skip_rows;
    Selectivity outerstartsel,
            outerendsel,
            innerstartsel,
            innerendsel;
    Path sort_path; /* dummy for result of cost_sort */

    /* Protect some assumptions below that rowcounts aren't zero */
    if (outer_path_rows <= 0)
        outer_path_rows = 1;
    if (inner_path_rows <= 0)
        inner_path_rows = 1;

    /*
     * A merge join will stop as soon as it exhausts either input stream
     * (unless it's an outer join, in which case the outer side has to be
     * scanned all the way anyway).  Estimate fraction of the left and right
     * inputs that will actually need to be scanned.  Likewise, we can
     * estimate the number of rows that will be skipped before the first join
     * pair is found, which should be factored into startup cost. We use only
     * the first (most significant) merge clause for this purpose. Since
     * mergejoinscansel() is a fairly expensive computation, we cache the
     * results in the merge clause RestrictInfo.
     */
    if (mergeclauses && jointype != JOIN_FULL) {
        RestrictInfo *firstclause = (RestrictInfo *) linitial(mergeclauses);
        List *opathkeys;
        List *ipathkeys;
        PathKey *opathkey;
        PathKey *ipathkey;
        MergeScanSelCache *cache;

        /* Get the input pathkeys to determine the sort-order details */
        opathkeys = outersortkeys ? outersortkeys : outer_path->pathkeys;
        ipathkeys = innersortkeys ? innersortkeys : inner_path->pathkeys;
        Assert(opathkeys);
        Assert(ipathkeys);
        opathkey = (PathKey *) linitial(opathkeys);
        ipathkey = (PathKey *) linitial(ipathkeys);
        /* debugging check */
        if (opathkey->pk_opfamily != ipathkey->pk_opfamily ||
            opathkey->pk_eclass->ec_collation != ipathkey->pk_eclass->ec_collation ||
            opathkey->pk_strategy != ipathkey->pk_strategy ||
            opathkey->pk_nulls_first != ipathkey->pk_nulls_first)
            elog(ERROR, "left and right pathkeys do not match in mergejoin");

        /* Get the selectivity with caching */
        cache = cached_scansel(root, firstclause, opathkey);

        if (bms_is_subset(firstclause->left_relids,
                          outer_path->parent->relids)) {
            /* left side of clause is outer */
            outerstartsel = cache->leftstartsel;
            outerendsel = cache->leftendsel;
            innerstartsel = cache->rightstartsel;
            innerendsel = cache->rightendsel;
        } else {
            /* left side of clause is inner */
            outerstartsel = cache->rightstartsel;
            outerendsel = cache->rightendsel;
            innerstartsel = cache->leftstartsel;
            innerendsel = cache->leftendsel;
        }
        if (jointype == JOIN_LEFT ||
            jointype == JOIN_ANTI) {
            outerstartsel = 0.0;
            outerendsel = 1.0;
        } else if (jointype == JOIN_RIGHT ||
                   jointype == JOIN_RIGHT_ANTI) {
            innerstartsel = 0.0;
            innerendsel = 1.0;
        }
    } else {
        /* cope with clauseless or full mergejoin */
        outerstartsel = innerstartsel = 0.0;
        outerendsel = innerendsel = 1.0;
    }

    /*
     * Convert selectivities to row counts.  We force outer_rows and
     * inner_rows to be at least 1, but the skip_rows estimates can be zero.
     */
    outer_skip_rows = rint(outer_path_rows * outerstartsel);
    inner_skip_rows = rint(inner_path_rows * innerstartsel);
    outer_rows = clamp_row_est(outer_path_rows * outerendsel);
    inner_rows = clamp_row_est(inner_path_rows * innerendsel);

    Assert(outer_skip_rows <= outer_rows);
    Assert(inner_skip_rows <= inner_rows);

    /*
     * Readjust scan selectivities to account for above rounding.  This is
     * normally an insignificant effect, but when there are only a few rows in
     * the inputs, failing to do this makes for a large percentage error.
     */
    outerstartsel = outer_skip_rows / outer_path_rows;
    innerstartsel = inner_skip_rows / inner_path_rows;
    outerendsel = outer_rows / outer_path_rows;
    innerendsel = inner_rows / inner_path_rows;

    Assert(outerstartsel <= outerendsel);
    Assert(innerstartsel <= innerendsel);

    /* cost of source data */

    if (outersortkeys) {
        /* do we need to sort outer? */
        cost_sort(&sort_path,
                  root,
                  outersortkeys,
                  outer_total_cost,
                  outer_path_rows,
                  outer_path->pathtarget->width,
                  0.0,
                  work_mem,
                  -1.0);
        startup_cost += sort_path.startup_cost;
        startup_cost += (sort_path.total_cost - sort_path.startup_cost)
                * outerstartsel;
        run_cost += (sort_path.total_cost - sort_path.startup_cost)
                * (outerendsel - outerstartsel);
    } else {
        startup_cost += outer_startup_cost;
        startup_cost += (outer_total_cost - outer_startup_cost) * outerstartsel;
        run_cost += (outer_total_cost - outer_startup_cost) * (outerendsel - outerstartsel);
    }

    if (innersortkeys) {
        /* do we need to sort inner? */
        cost_sort(&sort_path,
                  root,
                  innersortkeys,
                  inner_total_cost,
                  inner_path_rows,
                  inner_path->pathtarget->width,
                  0.0,
                  work_mem,
                  -1.0);
        startup_cost += sort_path.startup_cost;
        startup_cost += (sort_path.total_cost - sort_path.startup_cost)
                * innerstartsel;
        inner_run_cost = (sort_path.total_cost - sort_path.startup_cost)
                         * (innerendsel - innerstartsel);
    } else {
        startup_cost += inner_startup_cost;
        startup_cost += (inner_total_cost - inner_startup_cost) * innerstartsel;
        inner_run_cost = (inner_total_cost - inner_startup_cost) * (innerendsel - innerstartsel);
    }

    /*
     * We can't yet determine whether rescanning occurs, or whether
     * materialization of the inner input should be done.  The minimum
     * possible inner input cost, regardless of rescan and materialization
     * considerations, is inner_run_cost.  We include that in
     * workspace->total_cost, but not yet in run_cost.
     */

    /* CPU costs left for later */

    /* Public result fields */
    workspace->startup_cost = startup_cost;
    workspace->total_cost = startup_cost + run_cost + inner_run_cost;
    /* Save private data for final_cost_mergejoin */
    workspace->run_cost = run_cost;
    workspace->inner_run_cost = inner_run_cost;
    workspace->outer_rows = outer_rows;
    workspace->inner_rows = inner_rows;
    workspace->outer_skip_rows = outer_skip_rows;
    workspace->inner_skip_rows = inner_skip_rows;
}

void final_cost_mergejoin_1p(
    PlannerInfo *root,
    MergePath *path,
    JoinCostWorkspace *workspace,
    const JoinPathExtraData *extra
) {
    Path *outer_path = path->jpath.outerjoinpath;
    Path *inner_path = path->jpath.innerjoinpath;
    /*
     * Use per-round sampled row estimates for base scan paths (Seq/Index).
     * In single-sample mode (sample_count == 1) or non-scan paths,
     * fall back to the planner's deterministic row estimate.
     */
    double inner_path_rows = inner_path->rows;

    List *mergeclauses = path->path_mergeclauses;
    List *innersortkeys = path->innersortkeys;
    Cost startup_cost = workspace->startup_cost;
    Cost run_cost = workspace->run_cost;
    Cost inner_run_cost = workspace->inner_run_cost;
    double outer_rows = workspace->outer_rows;
    double inner_rows = workspace->inner_rows;
    double outer_skip_rows = workspace->outer_skip_rows;
    double inner_skip_rows = workspace->inner_skip_rows;
    Cost cpu_per_tuple,
            bare_inner_cost,
            mat_inner_cost;
    QualCost merge_qual_cost;
    QualCost qp_qual_cost;
    double mergejointuples,
            rescannedtuples;
    double rescanratio;

    /* Protect some assumptions below that rowcounts aren't zero */
    if (inner_path_rows <= 0)
        inner_path_rows = 1;

    /* Mark the path with the correct row estimate */
    if (path->jpath.path.param_info) {
        path->jpath.path.rows = path->jpath.path.param_info->ppi_rows;
    } else {
        path->jpath.path.rows = path->jpath.path.parent->rows;
    }
    path->jpath.path.rows_sample = NULL;

    /* For partial paths, scale row estimate. */
    if (path->jpath.path.parallel_workers > 0) {
        double parallel_divisor = get_parallel_divisor(&path->jpath.path);

        path->jpath.path.rows =
                clamp_row_est(path->jpath.path.rows / parallel_divisor);
    }

    /*
     * We could include disable_cost in the preliminary estimate, but that
     * would amount to optimizing for the case where the join method is
     * disabled, which doesn't seem like the way to bet.
     */
    if (!enable_mergejoin)
        startup_cost += disable_cost;

    /*
     * Compute cost of the mergequals and qpquals (other restriction clauses)
     * separately.
     */
    cost_qual_eval(&merge_qual_cost, mergeclauses, root);
    cost_qual_eval(&qp_qual_cost, path->jpath.joinrestrictinfo, root);
    qp_qual_cost.startup -= merge_qual_cost.startup;
    qp_qual_cost.per_tuple -= merge_qual_cost.per_tuple;

    /*
     * With a SEMI or ANTI join, or if the innerrel is known unique, the
     * executor will stop scanning for matches after the first match.  When
     * all the joinclauses are merge clauses, this means we don't ever need to
     * back up the merge, and so we can skip mark/restore overhead.
     */
    if ((path->jpath.jointype == JOIN_SEMI ||
         path->jpath.jointype == JOIN_ANTI ||
         extra->inner_unique) &&
        (list_length(path->jpath.joinrestrictinfo) ==
         list_length(path->path_mergeclauses)))
        path->skip_mark_restore = true;
    else
        path->skip_mark_restore = false;

    /*
     * Get approx # tuples passing the mergequals.  We use approx_tuple_count
     * here because we need an estimate done with JOIN_INNER semantics.
     */
    mergejointuples = approx_tuple_count(root, &path->jpath, mergeclauses);

    /*
     * When there are equal merge keys in the outer relation, the mergejoin
     * must rescan any matching tuples in the inner relation. This means
     * re-fetching inner tuples; we have to estimate how often that happens.
     *
     * For regular inner and outer joins, the number of re-fetches can be
     * estimated approximately as size of merge join output minus size of
     * inner relation. Assume that the distinct key values are 1, 2, ..., and
     * denote the number of values of each key in the outer relation as m1,
     * m2, ...; in the inner relation, n1, n2, ...  Then we have
     *
     * size of join = m1 * n1 + m2 * n2 + ...
     *
     * number of rescanned tuples = (m1 - 1) * n1 + (m2 - 1) * n2 + ... = m1 *
     * n1 + m2 * n2 + ... - (n1 + n2 + ...) = size of join - size of inner
     * relation
     *
     * This equation works correctly for outer tuples having no inner match
     * (nk = 0), but not for inner tuples having no outer match (mk = 0); we
     * are effectively subtracting those from the number of rescanned tuples,
     * when we should not.  Can we do better without expensive selectivity
     * computations?
     *
     * The whole issue is moot if we are working from a unique-ified outer
     * input, or if we know we don't need to mark/restore at all.
     */
    if (IsA(outer_path, UniquePath) || path->skip_mark_restore)
        rescannedtuples = 0;
    else {
        rescannedtuples = mergejointuples - inner_path_rows;
        /* Must clamp because of possible underestimate */
        if (rescannedtuples < 0)
            rescannedtuples = 0;
    }

    /*
     * We'll inflate various costs this much to account for rescanning.  Note
     * that this is to be multiplied by something involving inner_rows, or
     * another number related to the portion of the inner rel we'll scan.
     */
    rescanratio = 1.0 + (rescannedtuples / inner_rows);

    /*
     * Decide whether we want to materialize the inner input to shield it from
     * mark/restore and performing re-fetches.  Our cost model for regular
     * re-fetches is that a re-fetch costs the same as an original fetch,
     * which is probably an overestimate; but on the other hand we ignore the
     * bookkeeping costs of mark/restore.  Not clear if it's worth developing
     * a more refined model.  So we just need to inflate the inner run cost by
     * rescanratio.
     */
    bare_inner_cost = inner_run_cost * rescanratio;

    /*
     * When we interpose a Material node the re-fetch cost is assumed to be
     * just cpu_operator_cost per tuple, independently of the underlying
     * plan's cost; and we charge an extra cpu_operator_cost per original
     * fetch as well.  Note that we're assuming the materialize node will
     * never spill to disk, since it only has to remember tuples back to the
     * last mark.  (If there are a huge number of duplicates, our other cost
     * factors will make the path so expensive that it probably won't get
     * chosen anyway.)	So we don't use cost_rescan here.
     *
     * Note: keep this estimate in sync with create_mergejoin_plan's labeling
     * of the generated Material node.
     */
    mat_inner_cost = inner_run_cost +
                     cpu_operator_cost * inner_rows * rescanratio;

    /*
     * If we don't need mark/restore at all, we don't need materialization.
     */
    if (path->skip_mark_restore)
        path->materialize_inner = false;

        /*
         * Prefer materializing if it looks cheaper, unless the user has asked to
         * suppress materialization.
         */
    else if (enable_material && mat_inner_cost < bare_inner_cost)
        path->materialize_inner = true;

        /*
         * Even if materializing doesn't look cheaper, we *must* do it if the
         * inner path is to be used directly (without sorting) and it doesn't
         * support mark/restore.
         *
         * Since the inner side must be ordered, and only Sorts and IndexScans can
         * create order to begin with, and they both support mark/restore, you
         * might think there's no problem --- but you'd be wrong.  Nestloop and
         * merge joins can *preserve* the order of their inputs, so they can be
         * selected as the input of a mergejoin, and they don't support
         * mark/restore at present.
         *
         * We don't test the value of enable_material here, because
         * materialization is required for correctness in this case, and turning
         * it off does not entitle us to deliver an invalid plan.
         */
    else if (innersortkeys == NIL &&
             !ExecSupportsMarkRestore(inner_path))
        path->materialize_inner = true;

        /*
         * Also, force materializing if the inner path is to be sorted and the
         * sort is expected to spill to disk.  This is because the final merge
         * pass can be done on-the-fly if it doesn't have to support mark/restore.
         * We don't try to adjust the cost estimates for this consideration,
         * though.
         *
         * Since materialization is a performance optimization in this case,
         * rather than necessary for correctness, we skip it if enable_material is
         * off.
         */
    else if (enable_material && innersortkeys != NIL &&
             relation_byte_size(inner_path_rows,
                                inner_path->pathtarget->width) >
             (work_mem * 1024L))
        path->materialize_inner = true;
    else
        path->materialize_inner = false;

    /* Charge the right incremental cost for the chosen case */
    if (path->materialize_inner)
        run_cost += mat_inner_cost;
    else
        run_cost += bare_inner_cost;

    /* CPU costs */

    /*
     * The number of tuple comparisons needed is approximately number of outer
     * rows plus number of inner rows plus number of rescanned tuples (can we
     * refine this?).  At each one, we need to evaluate the mergejoin quals.
     */
    startup_cost += merge_qual_cost.startup;
    startup_cost += merge_qual_cost.per_tuple *
            (outer_skip_rows + inner_skip_rows * rescanratio);
    run_cost += merge_qual_cost.per_tuple *
    ((outer_rows - outer_skip_rows) +
     (inner_rows - inner_skip_rows) * rescanratio);

    /*
     * For each tuple that gets through the mergejoin proper, we charge
     * cpu_tuple_cost plus the cost of evaluating additional restriction
     * clauses that are to be applied at the join.  (This is pessimistic since
     * not all of the quals may get evaluated at each tuple.)
     *
     * Note: we could adjust for SEMI/ANTI joins skipping some qual
     * evaluations here, but it's probably not worth the trouble.
     */
    startup_cost += qp_qual_cost.startup;
    cpu_per_tuple = cpu_tuple_cost + qp_qual_cost.per_tuple;
    run_cost += cpu_per_tuple * mergejointuples;

    /* tlist eval costs are paid per output row, not per tuple scanned */
    startup_cost += path->jpath.path.pathtarget->cost.startup;
    run_cost += path->jpath.path.pathtarget->cost.per_tuple * path->jpath.path.rows;

    path->jpath.path.startup_cost = startup_cost;
    path->jpath.path.total_cost = startup_cost + run_cost;
}

/* ==== ==== ==== ==== ==== ==== 1-PASS HASH JOIN COST MODEL ==== ==== ==== ==== ==== ==== */

void initial_cost_hashjoin_1p(
    PlannerInfo *root,
    JoinCostWorkspace *workspace,
    JoinType jointype,
    const List *hashclauses,
    Path *outer_path,
    Path *inner_path,
    JoinPathExtraData *extra,
    bool parallel_hash
) {
    Cost startup_cost = 0;
    Cost run_cost = 0;

    /* Use per-round sampled costs for base scans; otherwise deterministic costs */
    const Cost outer_startup_cost = outer_path->startup_cost;
    const Cost outer_total_cost = outer_path->total_cost;
    // const Cost inner_startup_cost = inner_path->startup_cost;
    const Cost inner_total_cost = inner_path->total_cost;

    /*
     * Use per-round sampled row estimates for base scan paths (Seq/Index).
     * In single-sample mode (sample_count == 1) or non-scan paths,
     * fall back to the planner's deterministic row estimate.
     */
    const double outer_path_rows = outer_path->rows;
    /*
     * Same logic for inner path. Use sampled rows only for scan paths
     * when running in multi-sample mode.
     */
    const double inner_path_rows = inner_path->rows;

    double inner_path_rows_total = inner_path_rows;
    int num_hashclauses = list_length(hashclauses);
    int numbuckets;
    int numbatches;
    int num_skew_mcvs;
    size_t space_allowed; /* unused */

    /* cost of source data */
    startup_cost += outer_startup_cost;
    run_cost += outer_total_cost - outer_startup_cost;
    startup_cost += inner_total_cost;

    /*
     * Cost of computing hash function: must do it once per input tuple. We
     * charge one cpu_operator_cost for each column's hash function.  Also,
     * tack on one cpu_tuple_cost per inner row, to model the costs of
     * inserting the row into the hashtable.
     *
     * XXX when a hashclause is more complex than a single operator, we really
     * should charge the extra eval costs of the left or right side, as
     * appropriate, here.  This seems more work than it's worth at the moment.
     */
    startup_cost += (cpu_operator_cost * num_hashclauses + cpu_tuple_cost)
            * inner_path_rows;
    run_cost += cpu_operator_cost * num_hashclauses * outer_path_rows;

    /*
     * If this is a parallel hash build, then the value we have for
     * inner_rows_total currently refers only to the rows returned by each
     * participant.  For shared hash table size estimation, we need the total
     * number, so we need to undo the division.
     */
    if (parallel_hash)
        inner_path_rows_total *= get_parallel_divisor(inner_path);

    /*
     * Get hash table size that executor would use for inner relation.
     *
     * XXX for the moment, always assume that skew optimization will be
     * performed.  As long as SKEW_HASH_MEM_PERCENT is small, it's not worth
     * trying to determine that for sure.
     *
     * XXX at some point it might be interesting to try to account for skew
     * optimization in the cost estimate, but for now, we don't.
     */
    ExecChooseHashTableSize(
        inner_path_rows_total,
        inner_path->pathtarget->width,
        true, /* useskew */
        parallel_hash, /* try_combined_hash_mem */
        outer_path->parallel_workers,
        &space_allowed,
        &numbuckets,
        &numbatches,
        &num_skew_mcvs
    );

    /*
     * If inner relation is too big then we will need to "batch" the join,
     * which implies writing and reading most of the tuples to disk an extra
     * time.  Charge seq_page_cost per page, since the I/O should be nice and
     * sequential.  Writing the inner rel counts as startup cost, all the rest
     * as run cost.
     */
    if (numbatches > 1) {
        double outerpages = page_size(outer_path_rows,
                                      outer_path->pathtarget->width);
        double innerpages = page_size(inner_path_rows,
                                      inner_path->pathtarget->width);

        startup_cost += seq_page_cost * innerpages;
        run_cost += seq_page_cost * (innerpages + 2 * outerpages);
    }

    /* CPU costs left for later */

    /* Public result fields */
    workspace->startup_cost = startup_cost;
    workspace->total_cost = startup_cost + run_cost;
    /* Save private data for final_cost_hashjoin */
    workspace->run_cost = run_cost;
    workspace->numbuckets = numbuckets;
    workspace->numbatches = numbatches;
    workspace->inner_rows_total = inner_path_rows_total;
}

void final_cost_hashjoin_1p(
    PlannerInfo *root,
    HashPath *path,
    JoinCostWorkspace *workspace,
    const JoinPathExtraData *extra
) {
    Path *outer_path = path->jpath.outerjoinpath;
    Path *inner_path = path->jpath.innerjoinpath;
    /*
     * Use per-round sampled row estimates for base scan paths (Seq/Index).
     * In single-sample mode (sample_count == 1) or non-scan paths,
     * fall back to the planner's deterministic row estimate.
     */
    const double outer_path_rows = outer_path->rows;
    /*
     * Same logic for inner path. Use sampled rows only for scan paths
     * when running in multi-sample mode.
     */
    const double inner_path_rows = inner_path->rows;

    double inner_path_rows_total = workspace->inner_rows_total;
    List *hashclauses = path->path_hashclauses;
    Cost startup_cost = workspace->startup_cost;
    Cost run_cost = workspace->run_cost;
    int numbuckets = workspace->numbuckets;
    int numbatches = workspace->numbatches;
    Cost cpu_per_tuple;
    QualCost hash_qual_cost;
    QualCost qp_qual_cost;
    double hashjointuples;
    double virtualbuckets;
    Selectivity innerbucketsize;
    Selectivity innermcvfreq;
    ListCell *hcl;

    /* Mark the path with the correct row estimate */
    if (path->jpath.path.param_info) {
        path->jpath.path.rows = path->jpath.path.param_info->ppi_rows;
    } else {
        path->jpath.path.rows = path->jpath.path.parent->rows;
    }
    path->jpath.path.rows_sample = NULL;

    /* For partial paths, scale row estimate. */
    if (path->jpath.path.parallel_workers > 0) {
        double parallel_divisor = get_parallel_divisor(&path->jpath.path);

        path->jpath.path.rows =
                clamp_row_est(path->jpath.path.rows / parallel_divisor);
    }

    /*
     * We could include disable_cost in the preliminary estimate, but that
     * would amount to optimizing for the case where the join method is
     * disabled, which doesn't seem like the way to bet.
     */
    if (!enable_hashjoin)
        startup_cost += disable_cost;

    /* mark the path with estimated # of batches */
    path->num_batches = numbatches;

    /* store the total number of tuples (sum of partial row estimates) */
    path->inner_rows_total = inner_path_rows_total;

    /* and compute the number of "virtual" buckets in the whole join */
    virtualbuckets = (double) numbuckets * (double) numbatches;

    /*
     * Determine bucketsize fraction and MCV frequency for the inner relation.
     * We use the smallest bucketsize or MCV frequency estimated for any
     * individual hashclause; this is undoubtedly conservative.
     *
     * BUT: if inner relation has been unique-ified, we can assume it's good
     * for hashing.  This is important both because it's the right answer, and
     * because we avoid contaminating the cache with a value that's wrong for
     * non-unique-ified paths.
     */
    if (IsA(inner_path, UniquePath)) {
        innerbucketsize = 1.0 / virtualbuckets;
        innermcvfreq = 0.0;
    } else {
        innerbucketsize = 1.0;
        innermcvfreq = 1.0;
        foreach(hcl, hashclauses) {
            RestrictInfo *restrictinfo = lfirst_node(RestrictInfo, hcl);
            Selectivity thisbucketsize;
            Selectivity thismcvfreq;

            /*
             * First we have to figure out which side of the hashjoin clause
             * is the inner side.
             *
             * Since we tend to visit the same clauses over and over when
             * planning a large query, we cache the bucket stats estimates in
             * the RestrictInfo node to avoid repeated lookups of statistics.
             */
            if (bms_is_subset(restrictinfo->right_relids,
                              inner_path->parent->relids)) {
                /* righthand side is inner */
                thisbucketsize = restrictinfo->right_bucketsize;
                if (thisbucketsize < 0) {
                    /* not cached yet */
                    estimate_hash_bucket_stats(
                        root,
                        get_rightop(restrictinfo->clause),
                        virtualbuckets,
                        &restrictinfo->right_mcvfreq,
                        &restrictinfo->right_bucketsize
                    );
                    thisbucketsize = restrictinfo->right_bucketsize;
                }
                thismcvfreq = restrictinfo->right_mcvfreq;
            } else {
                Assert(bms_is_subset(restrictinfo->left_relids,
                    inner_path->parent->relids));
                /* lefthand side is inner */
                thisbucketsize = restrictinfo->left_bucketsize;
                if (thisbucketsize < 0) {
                    /* not cached yet */
                    estimate_hash_bucket_stats(
                        root,
                        get_leftop(restrictinfo->clause),
                        virtualbuckets,
                        &restrictinfo->left_mcvfreq,
                        &restrictinfo->left_bucketsize
                    );
                    thisbucketsize = restrictinfo->left_bucketsize;
                }
                thismcvfreq = restrictinfo->left_mcvfreq;
            }

            if (innerbucketsize > thisbucketsize)
                innerbucketsize = thisbucketsize;
            if (innermcvfreq > thismcvfreq)
                innermcvfreq = thismcvfreq;
        }
    }

    /*
     * If the bucket holding the inner MCV would exceed hash_mem, we don't
     * want to hash unless there is really no other alternative, so apply
     * disable_cost.  (The executor normally copes with excessive memory usage
     * by splitting batches, but obviously it cannot separate equal values
     * that way, so it will be unable to drive the batch size below hash_mem
     * when this is true.)
     */
    if (relation_byte_size(clamp_row_est(inner_path_rows * innermcvfreq),
                           inner_path->pathtarget->width) > get_hash_memory_limit())
        startup_cost += disable_cost;

    /*
     * Compute cost of the hashquals and qpquals (other restriction clauses)
     * separately.
     */
    cost_qual_eval(&hash_qual_cost, hashclauses, root);
    cost_qual_eval(&qp_qual_cost, path->jpath.joinrestrictinfo, root);
    qp_qual_cost.startup -= hash_qual_cost.startup;
    qp_qual_cost.per_tuple -= hash_qual_cost.per_tuple;

    /* CPU costs */

    if (path->jpath.jointype == JOIN_SEMI ||
        path->jpath.jointype == JOIN_ANTI ||
        extra->inner_unique) {
        double outer_matched_rows;
        Selectivity inner_scan_frac;

        /*
         * With a SEMI or ANTI join, or if the innerrel is known unique, the
         * executor will stop after the first match.
         *
         * For an outer-rel row that has at least one match, we can expect the
         * bucket scan to stop after a fraction 1/(match_count+1) of the
         * bucket's rows, if the matches are evenly distributed.  Since they
         * probably aren't quite evenly distributed, we apply a fuzz factor of
         * 2.0 to that fraction.  (If we used a larger fuzz factor, we'd have
         * to clamp inner_scan_frac to at most 1.0; but since match_count is
         * at least 1, no such clamp is needed now.)
         */
        outer_matched_rows = rint(outer_path_rows * extra->semifactors.outer_match_frac);
        inner_scan_frac = 2.0 / (extra->semifactors.match_count + 1.0);

        startup_cost += hash_qual_cost.startup;
        run_cost += hash_qual_cost.per_tuple * outer_matched_rows *
                clamp_row_est(inner_path_rows * innerbucketsize * inner_scan_frac) * 0.5;

        /*
         * For unmatched outer-rel rows, the picture is quite a lot different.
         * In the first place, there is no reason to assume that these rows
         * preferentially hit heavily-populated buckets; instead assume they
         * are uncorrelated with the inner distribution and so they see an
         * average bucket size of inner_path_rows / virtualbuckets.  In the
         * second place, it seems likely that they will have few if any exact
         * hash-code matches and so very few of the tuples in the bucket will
         * actually require eval of the hash quals.  We don't have any good
         * way to estimate how many will, but for the moment assume that the
         * effective cost per bucket entry is one-tenth what it is for
         * matchable tuples.
         */
        run_cost += hash_qual_cost.per_tuple *
                (outer_path_rows - outer_matched_rows) *
                clamp_row_est(inner_path_rows / virtualbuckets) * 0.05;

        /* Get # of tuples that will pass the basic join */
        if (path->jpath.jointype == JOIN_ANTI)
            hashjointuples = outer_path_rows - outer_matched_rows;
        else
            hashjointuples = outer_matched_rows;
    } else {
        /*
         * The number of tuple comparisons needed is the number of outer
         * tuples times the typical number of tuples in a hash bucket, which
         * is the inner relation size times its bucketsize fraction.  At each
         * one, we need to evaluate the hashjoin quals.  But actually,
         * charging the full qual eval cost at each tuple is pessimistic,
         * since we don't evaluate the quals unless the hash values match
         * exactly.  For lack of a better idea, halve the cost estimate to
         * allow for that.
         */
        startup_cost += hash_qual_cost.startup;
        run_cost += hash_qual_cost.per_tuple * outer_path_rows *
                clamp_row_est(inner_path_rows * innerbucketsize) * 0.5;

        /*
         * Get approx # tuples passing the hashquals.  We use
         * approx_tuple_count here because we need an estimate done with
         * JOIN_INNER semantics.
         */
        hashjointuples = approx_tuple_count(root, &path->jpath, hashclauses);
    }

    /*
     * For each tuple that gets through the hashjoin proper, we charge
     * cpu_tuple_cost plus the cost of evaluating additional restriction
     * clauses that are to be applied at the join.  (This is pessimistic since
     * not all of the quals may get evaluated at each tuple.)
     */
    startup_cost += qp_qual_cost.startup;
    cpu_per_tuple = cpu_tuple_cost + qp_qual_cost.per_tuple;
    run_cost += cpu_per_tuple * hashjointuples;

    /* tlist eval costs are paid per output row, not per tuple scanned */
    startup_cost += path->jpath.path.pathtarget->cost.startup;
    run_cost += path->jpath.path.pathtarget->cost.per_tuple * path->jpath.path.rows;

    path->jpath.path.startup_cost = startup_cost;
    path->jpath.path.total_cost = startup_cost + run_cost;
}

/* ==== ==== ==== ==== ==== ==== 2-PASS NEST LOOP COST MODEL ==== ==== ==== ==== ==== ==== */

void initial_cost_nestloop_2p(
    PlannerInfo *root,
    JoinCostWorkspace *workspace,
    JoinType jointype,
    const Path *outer_path,
    Path *inner_path,
    const JoinPathExtraData *extra
) {
    /* ------------------------------- 1) Resolve samples & loop count ------------------------------- */
    const Sample *outer_rows_samp = outer_path->rows_sample;
    const Sample *inner_rows_samp = inner_path->rows_sample;

    const Sample *outer_startup_samp = outer_path->startup_cost_sample;
    const Sample *outer_total_samp = outer_path->total_cost_sample;

    const Sample *inner_startup_samp = inner_path->startup_cost_sample;
    const Sample *inner_total_samp = inner_path->total_cost_sample;

    Assert(outer_rows_samp != NULL && outer_rows_samp->sample_count > 0);
    Assert(inner_rows_samp != NULL && inner_rows_samp->sample_count > 0);
    const bool outer_rows_is_const = outer_rows_samp->sample_count == 1;
    const bool inner_rows_is_const = inner_rows_samp->sample_count == 1;

    Assert(outer_startup_samp != NULL && outer_startup_samp->sample_count > 0);
    Assert(outer_total_samp != NULL && outer_total_samp->sample_count > 0);
    const bool outer_startup_is_const = outer_startup_samp->sample_count == 1;
    const bool outer_total_is_const = outer_total_samp->sample_count == 1;

    Assert(inner_startup_samp != NULL && inner_startup_samp->sample_count > 0);
    Assert(inner_total_samp != NULL && inner_total_samp->sample_count > 0);
    const bool inner_startup_is_const = inner_startup_samp->sample_count == 1;
    const bool inner_total_is_const = inner_total_samp->sample_count == 1;

    /* Decide loop sample_count: if all have N>1, require same N; otherwise use the non-1 side. */
    int sample_count;
    if (outer_rows_is_const && inner_rows_is_const
        && outer_startup_is_const && outer_total_is_const
        && inner_startup_is_const && inner_total_is_const) {
        sample_count = 1;
    } else {
        sample_count = error_sample_count;
    }
    workspace->sample_count = sample_count;

    /* ------------------------------- 2) Init accumulators & private fields ------------------------------- */
    workspace->startup_cost = 0.0;
    workspace->total_cost = 0.0;

    workspace->run_cost = 0.0;
    workspace->inner_run_cost = 0.0;
    workspace->inner_rescan_run_cost = 0.0;

    /* ------------------------------- 3) Inner rescan costs (scalar, phase-1 LB) ------------------------------- */
    /*
     * cost_rescan currently returns scalar start/total costs.  We use them for
     * all samples as a conservative lower bound (do not overestimate in phase-1).
     */
    Cost inner_rescan_start_cost = 0.0;
    Cost inner_rescan_total_cost = 0.0;

    cost_rescan(root, inner_path,
                &inner_rescan_start_cost,
                &inner_rescan_total_cost);

    const Cost inner_rescan_run_cost_scalar =
            (inner_rescan_total_cost - inner_rescan_start_cost);

    /* ------------------------------- 4) Per-sample evaluation ------------------------------- */
    for (int i = 0; i < sample_count; ++i) {
        /* 4.1) Rows per sample (guard against <= 0) */
        double outer_rows_i = GET_ROW(outer_rows_samp, i, outer_path->rows, outer_rows_is_const);
        if (outer_rows_i <= 0) outer_rows_i = 1;

        /* 4.2) Per-sample input costs */
        const Cost outer_startup_i = GET_COST(outer_startup_samp, i, outer_path->startup_cost, outer_startup_is_const);
        const Cost outer_total_i = GET_COST(outer_total_samp, i, outer_path->total_cost, outer_total_is_const);

        const Cost inner_startup_i = GET_COST(inner_startup_samp, i, inner_path->startup_cost, inner_startup_is_const);
        const Cost inner_total_i = GET_COST(inner_total_samp, i, inner_path->total_cost, inner_total_is_const);

        /* 4.3) Phase-1 startup/run base (as in upstream semantics) */
        Cost startup_cost = 0.0;
        Cost run_cost = 0.0;

        /* Must pay both sides' startup before producing tuples. */
        startup_cost += outer_startup_i + inner_startup_i;

        /* Outer scan run part (always paid). */
        run_cost += (outer_total_i - outer_startup_i);

        /* Inner rescan startup part for (outer_rows_i - 1) rescans (if any). */
        if (outer_rows_i > 1)
            run_cost += (outer_rows_i - 1) * inner_rescan_start_cost;

        /* Inner side run parts (handled differently per join type/uniqueness). */
        const Cost inner_run_cost_i = (inner_total_i - inner_startup_i);
        const Cost inner_rescan_run_cost_i = inner_rescan_run_cost_scalar;

        if (jointype == JOIN_SEMI || jointype == JOIN_ANTI || extra->inner_unique) {
            /*
             * For SEMI/ANTI or known-unique inner, executor stops at first match.
             * We defer detailed CPU/qual-based estimation to final_cost_nestloop.
             * Phase-1 stores inner costs for phase-2 but does not add them to run_cost.
             */
            workspace->inner_run_cost_sample[i] = inner_run_cost_i;
            workspace->inner_rescan_run_cost_sample[i] = inner_rescan_run_cost_i;

            workspace->inner_run_cost += inner_run_cost_i;
            workspace->inner_rescan_run_cost += inner_rescan_run_cost_i;
        } else {
            /* Normal case: scan full inner for each outer row (with rescans). */
            run_cost += inner_run_cost_i;
            if (outer_rows_i > 1)
                run_cost += (outer_rows_i - 1) * inner_rescan_run_cost_i;
        }

        /* 4.4) Record per-sample lower bounds, accumulate scalars */
        workspace->startup_cost_sample[i] = startup_cost;
        workspace->total_cost_sample[i] = startup_cost + run_cost;
        workspace->run_cost_sample[i] = run_cost;

        workspace->startup_cost += startup_cost;
        workspace->total_cost += startup_cost + run_cost;
        workspace->run_cost += run_cost;
    }

    /* ------------------------------- 5) Finalize scalar lower bounds (means) ------------------------------- */
    const double invN = 1.0 / (double) sample_count;

    workspace->startup_cost *= invN;
    workspace->total_cost *= invN;
    workspace->run_cost *= invN;
    workspace->inner_run_cost *= invN;
    workspace->inner_rescan_run_cost *= invN;
}

void final_cost_nestloop_2p(
    PlannerInfo *root,
    NestPath *path,
    JoinCostWorkspace *workspace,
    const JoinPathExtraData *extra
) {
    /* ------------------------------- 1) Resolve paths & samples ------------------------------- */
    const Path *outer_path = path->jpath.outerjoinpath;
    const Path *inner_path = path->jpath.innerjoinpath;

    /* Row samples */
    const Sample *outer_rows_samp = outer_path->rows_sample;
    const Sample *inner_rows_samp = inner_path->rows_sample;

    Assert(outer_rows_samp != NULL && outer_rows_samp->sample_count > 0);
    Assert(inner_rows_samp != NULL && inner_rows_samp->sample_count > 0);
    const bool outer_rows_is_const = outer_rows_samp->sample_count == 1;
    const bool inner_rows_is_const = inner_rows_samp->sample_count == 1;

    /* Authoritative loop size comes from the workspace */
    Assert(workspace->sample_count > 0);
    const int sample_count = workspace->sample_count;

    /* ------------------------------- 2) Determine scalar output rows ------------------------------- */
    /* Mark rows (scalar) per PG semantics: param_info or parent->rows */
    if (path->jpath.path.param_info) {
        path->jpath.path.rows = path->jpath.path.param_info->ppi_rows;
        path->jpath.path.rows_sample = make_sample_by_single_value(path->jpath.path.param_info->ppi_rows);
    } else {
        path->jpath.path.rows = path->jpath.path.parent->rows;
        path->jpath.path.rows_sample = duplicate_sample(path->jpath.path.parent->rows_sample);
    }

    /* Parallel: rows represent per-worker output rows */
    if (path->jpath.path.parallel_workers > 0) {
        const double parallel_divisor = get_parallel_divisor(&path->jpath.path);
        path->jpath.path.rows = clamp_row_est(path->jpath.path.rows / parallel_divisor);
        Sample *new_sample = make_sample_by_scale_factor(
            path->jpath.path.rows_sample, 1.0 / parallel_divisor
        );
        pfree(path->jpath.path.rows_sample);
        path->jpath.path.rows_sample = new_sample;
    }

    /* Initialize the startup and total cost samples */
    path->jpath.path.startup_cost = 0.0;
    path->jpath.path.startup_cost_sample = initialize_sample(sample_count);
    path->jpath.path.total_cost = 0.0;
    path->jpath.path.total_cost_sample = initialize_sample(sample_count);

    /* Optional: match original behavior of penalizing when nestloop is disabled. */
    if (!enable_nestloop) {
        workspace->startup_cost += disable_cost;
        for (int i = 0; i < sample_count; ++i) {
            workspace->startup_cost_sample[i] += disable_cost;
        }
    }

    /* ------------------------------- 3) Qual/tlist costs (sample-invariant) ------------------------------- */
    QualCost rq_cost;
    cost_qual_eval(&rq_cost, path->jpath.joinrestrictinfo, root);

    const Cost cpu_per_tuple = cpu_tuple_cost + rq_cost.per_tuple;
    const Cost tlist_startup = path->jpath.path.pathtarget->cost.startup;
    const Cost tlist_per_tuple = path->jpath.path.pathtarget->cost.per_tuple;

    /* ------------------------------- 4) Accumulators for scalar (means) ------------------------------- */
    Cost startup_accum = 0.0;
    Cost total_accum = 0.0;

    /* If you maintain per-sample outputs on Path, ensure storage is allocated by caller. */

    /* ------------------------------- 5) Per-sample evaluation ------------------------------- */
    for (int i = 0; i < sample_count; ++i) {
        /* 5.1) Start from phase-1 lower bounds */
        Cost startup_cost_i = workspace->startup_cost_sample[i];
        Cost run_cost_i = workspace->run_cost_sample[i];

        /* 5.2) Per-sample input rows (guard against <= 0) */
        double outer_rows_i = GET_ROW(outer_rows_samp, i, outer_path->rows, outer_rows_is_const);
        double inner_rows_i = GET_ROW(inner_rows_samp, i, inner_path->rows, inner_rows_is_const);
        if (outer_rows_i <= 0) outer_rows_i = 1;
        if (inner_rows_i <= 0) inner_rows_i = 1;

        /* 5.3) Refine inner-source costs for SEMI/ANTI/inner_unique (per sample) */
        double ntuples_i; /* processed tuples (not emitted count) */

        if (path->jpath.jointype == JOIN_SEMI ||
            path->jpath.jointype == JOIN_ANTI ||
            extra->inner_unique) {
            /* Assumed to exist per your request */
            const Cost inner_run_cost_i = workspace->inner_run_cost_sample[i];
            const Cost inner_rescan_run_cost_i = workspace->inner_rescan_run_cost_sample[i];

            double outer_matched_rows = rint(outer_rows_i * extra->semifactors.outer_match_frac);
            double outer_unmatched_rows = outer_rows_i - outer_matched_rows;

            /* Early-stop fraction with fuzz */
            const Selectivity inner_scan_frac = 2.0 / (extra->semifactors.match_count + 1.0);

            /* Matched processed tuples */
            ntuples_i = outer_matched_rows * inner_rows_i * inner_scan_frac;

            if (has_indexed_join_quals(path)) {
                /* Matched rows scan a fraction of inner */
                run_cost_i += inner_run_cost_i * inner_scan_frac;
                if (outer_matched_rows > 1)
                    run_cost_i += (outer_matched_rows - 1) * inner_rescan_run_cost_i * inner_scan_frac;

                /* Unmatched: cost approximates returning first tuple of a nonempty scan (rescans) */
                run_cost_i += outer_unmatched_rows * (inner_rescan_run_cost_i / inner_rows_i);
                /* No quals for unmatched => ntuples_i unchanged */
            } else {
                /* Count all unmatched processed tuples */
                ntuples_i += outer_unmatched_rows * inner_rows_i;

                /* Force one full inner first-scan run cost */
                run_cost_i += inner_run_cost_i;
                if (outer_unmatched_rows >= 1)
                    outer_unmatched_rows -= 1;
                else
                    outer_matched_rows -= 1;

                /* Additional matched: rescans with early stop */
                if (outer_matched_rows > 0)
                    run_cost_i += outer_matched_rows * inner_rescan_run_cost_i * inner_scan_frac;

                /* Additional unmatched: full rescans */
                if (outer_unmatched_rows > 0)
                    run_cost_i += outer_unmatched_rows * inner_rescan_run_cost_i;
            }
        } else {
            /* Normal case: processed tuples = outer * inner */
            ntuples_i = outer_rows_i * inner_rows_i;
        }

        /* 5.4) CPU (restrict quals) */
        startup_cost_i += rq_cost.startup;
        run_cost_i += cpu_per_tuple * ntuples_i;

        /* 5.5) Tlist costs: startup once; per-tuple on output rows (scalar) */
        startup_cost_i += tlist_startup;
        run_cost_i += tlist_per_tuple * path->jpath.path.rows;

        /* 5.6) Aggregate results */
        const Cost total_cost_i = startup_cost_i + run_cost_i;

        startup_accum += startup_cost_i;
        total_accum += total_cost_i;

        /* If Path has per-sample outputs allocated, fill them here. */
        if (path->jpath.path.startup_cost_sample && path->jpath.path.total_cost_sample) {
            path->jpath.path.startup_cost_sample->sample[i] = startup_cost_i;
            path->jpath.path.total_cost_sample->sample[i] = total_cost_i;
        }
    }

    /* ------------------------------- 6) Finalize scalar outputs (means) ------------------------------- */
    const double invN = 1.0 / (double) (sample_count > 0 ? sample_count : 1);
    path->jpath.path.startup_cost = startup_accum * invN;
    path->jpath.path.total_cost = total_accum * invN;
}

/* ==== ==== ==== ==== ==== ==== 2-PASS MERGE JOIN COST MODEL ==== ==== ==== ==== ==== ==== */

void initial_cost_mergejoin_2p(
    PlannerInfo *root,
    JoinCostWorkspace *workspace,
    JoinType jointype,
    List *mergeclauses,
    Path *outer_path,
    Path *inner_path,
    List *outersortkeys,
    List *innersortkeys,
    JoinPathExtraData *extra
) {
    /* ----------------------------------------------------------------------
     * 0) Resolve per-path row samples and costs
     * ---------------------------------------------------------------------- */
    const Sample *outer_rows_samp = outer_path->rows_sample;
    const Sample *inner_rows_samp = inner_path->rows_sample;

    const Sample *outer_startup_samp = outer_path->startup_cost_sample;
    const Sample *outer_total_samp = outer_path->total_cost_sample;

    const Sample *inner_startup_samp = inner_path->startup_cost_sample;
    const Sample *inner_total_samp = inner_path->total_cost_sample;

    Assert(outer_rows_samp != NULL && outer_rows_samp->sample_count > 0);
    Assert(inner_rows_samp != NULL && inner_rows_samp->sample_count > 0);
    const bool outer_rows_is_const = outer_rows_samp->sample_count == 1;
    const bool inner_rows_is_const = inner_rows_samp->sample_count == 1;

    Assert(outer_startup_samp != NULL && outer_startup_samp->sample_count > 0);
    Assert(outer_total_samp != NULL && outer_total_samp->sample_count > 0);
    const bool outer_startup_is_const = outer_startup_samp->sample_count == 1;
    const bool outer_total_is_const = outer_total_samp->sample_count == 1;

    Assert(inner_startup_samp != NULL && inner_startup_samp->sample_count > 0);
    Assert(inner_total_samp != NULL && inner_total_samp->sample_count > 0);
    const bool inner_startup_is_const = inner_startup_samp->sample_count == 1;
    const bool inner_total_is_const = inner_total_samp->sample_count == 1;

    /* Decide loop sample_count: if all have N>1, require same N; otherwise use the non-1 side. */
    int sample_count;
    if (outer_rows_is_const && inner_rows_is_const
        && outer_startup_is_const && outer_total_is_const
        && inner_startup_is_const && inner_total_is_const) {
        sample_count = 1;
    } else {
        sample_count = error_sample_count;
    }
    workspace->sample_count = sample_count;

    /* ----------------------------------------------------------------------
     * 1) Compute merge-scan selectivities from the first mergeclause
     * ----------------------------------------------------------------------
     * These selectivities are properties of ordering and clause, not samples.
     * Later we'll convert them to per-sample row counts and then re-normalize.
     */
    Selectivity outerstartsel, outerendsel, innerstartsel, innerendsel;

    if (mergeclauses && jointype != JOIN_FULL) {
        RestrictInfo *firstclause = (RestrictInfo *) linitial(mergeclauses);
        List *opathkeys = outersortkeys ? outersortkeys : outer_path->pathkeys;
        List *ipathkeys = innersortkeys ? innersortkeys : inner_path->pathkeys;
        PathKey *opathkey;
        PathKey *ipathkey;
        MergeScanSelCache *cache;

        Assert(opathkeys && ipathkeys);
        opathkey = (PathKey *) linitial(opathkeys);
        ipathkey = (PathKey *) linitial(ipathkeys);

        /* Debug check: ordering compatibility for mergejoin. */
        if (opathkey->pk_opfamily != ipathkey->pk_opfamily ||
            opathkey->pk_eclass->ec_collation != ipathkey->pk_eclass->ec_collation ||
            opathkey->pk_nulls_first != ipathkey->pk_nulls_first)
            elog(ERROR, "left and right pathkeys do not match in mergejoin");

        cache = cached_scansel(root, firstclause, opathkey);

        if (bms_is_subset(firstclause->left_relids, outer_path->parent->relids)) {
            /* left side of clause is outer */
            outerstartsel = cache->leftstartsel;
            outerendsel = cache->leftendsel;
            innerstartsel = cache->rightstartsel;
            innerendsel = cache->rightendsel;
        } else {
            /* left side of clause is inner */
            outerstartsel = cache->rightstartsel;
            outerendsel = cache->rightendsel;
            innerstartsel = cache->leftstartsel;
            innerendsel = cache->leftendsel;
        }

        if (jointype == JOIN_LEFT || jointype == JOIN_ANTI) {
            outerstartsel = 0.0;
            outerendsel = 1.0;
        } else if (jointype == JOIN_RIGHT || jointype == JOIN_RIGHT_ANTI) {
            innerstartsel = 0.0;
            innerendsel = 1.0;
        }
    } else {
        /* Clauseless or FULL mergejoin: scan entire ranges. */
        outerstartsel = innerstartsel = 0.0;
        outerendsel = innerendsel = 1.0;
    }

    /* ----------------------------------------------------------------------
     * 2) Initialize accumulators for scalar (mean) lower bounds
     * ---------------------------------------------------------------------- */
    Cost startup_accum = 0;
    Cost total_accum = 0;
    Cost run_accum = 0;
    Cost inner_run_accum = 0;
    double outer_rows_accum = 0;
    double inner_rows_accum = 0;
    double outer_skip_accum = 0;
    double inner_skip_accum = 0;

    /* ----------------------------------------------------------------------
     * 3) Per-sample evaluation (point-to-point)
     * ----------------------------------------------------------------------
     * For each sample i, fetch outer/inner rows and costs from the i-th sample
     * (or from sample[0]/scalars if const), then compute the fractional
     * contribution to startup/run cost as in upstream logic.
     */
    for (int i = 0; i < sample_count; ++i) {
        /* 3.1) Fetch per-sample input rows (guard against <= 0) */
        double outer_path_rows = GET_ROW(outer_rows_samp, i, outer_path->rows, outer_rows_is_const);
        double inner_path_rows = GET_ROW(inner_rows_samp, i, inner_path->rows, inner_rows_is_const);
        if (outer_path_rows <= 0) outer_path_rows = 1;
        if (inner_path_rows <= 0) inner_path_rows = 1;

        /* 3.2) Convert selectivities to per-sample row counts (lower bounds) */
        double outer_skip_rows = rint(outer_path_rows * outerstartsel);
        double inner_skip_rows = rint(inner_path_rows * innerstartsel);
        double outer_rows = clamp_row_est(outer_path_rows * outerendsel);
        double inner_rows = clamp_row_est(inner_path_rows * innerendsel);

        Assert(outer_skip_rows <= outer_rows);
        Assert(inner_skip_rows <= inner_rows);

        /* 3.3) Renormalize selectivities after rounding (small-N stability) */
        double outerstartsel_i = outer_skip_rows / outer_path_rows;
        double innerstartsel_i = inner_skip_rows / inner_path_rows;
        double outerendsel_i = outer_rows / outer_path_rows;
        double innerendsel_i = inner_rows / inner_path_rows;

        Assert(outerstartsel_i <= outerendsel_i);
        Assert(innerstartsel_i <= innerendsel_i);

        /* 3.4) Build per-sample startup/run costs for each side
         *      If sorting is required, cost it per sample using that sample's rows
         *      and the sample's upstream input costs. Otherwise, use upstream
         *      path costs per sample and slice them by selectivity.
         */
        Cost startup_cost = 0;
        Cost run_cost = 0;
        Cost inner_run_cost;

        Path sort_path; /* dummy holder for cost_sort result */

        /* ---- OUTER side ---- */
        if (outersortkeys) {
            /* Per-sample: feed cost_sort with this sample's outer rows.
             * We also pass the input total cost of this sample to reflect
             * the upstream cost in the sort pipeline.
             */
            Cost outer_total_cost_i =
                    GET_COST(outer_total_samp, i, outer_path->total_cost, outer_total_is_const);

            cost_sort(&sort_path,
                      root,
                      outersortkeys,
                      outer_total_cost_i, /* input total cost */
                      outer_path_rows, /* input rows for this sample */
                      outer_path->pathtarget->width,
                      0.0,
                      work_mem,
                      -1.0);

            /* Slice sort cost into startup & run fractions per upstream logic */
            startup_cost += sort_path.startup_cost;
            startup_cost += (sort_path.total_cost - sort_path.startup_cost) * outerstartsel_i;
            run_cost += (sort_path.total_cost - sort_path.startup_cost) * (outerendsel_i - outerstartsel_i);
        } else {
            /* Use outer path's own per-sample costs */
            Cost outer_startup_cost_i =
                    GET_COST(outer_startup_samp, i, outer_path->startup_cost, outer_startup_is_const);
            Cost outer_total_cost_i =
                    GET_COST(outer_total_samp, i, outer_path->total_cost, outer_total_is_const);

            startup_cost += outer_startup_cost_i;
            startup_cost += (outer_total_cost_i - outer_startup_cost_i) * outerstartsel_i;
            run_cost += (outer_total_cost_i - outer_startup_cost_i) * (outerendsel_i - outerstartsel_i);
        }

        /* ---- INNER side ---- */
        if (innersortkeys) {
            Cost inner_total_cost_i =
                    GET_COST(inner_total_samp, i, inner_path->total_cost, inner_total_is_const);

            cost_sort(&sort_path,
                      root,
                      innersortkeys,
                      inner_total_cost_i, /* input total cost */
                      inner_path_rows, /* input rows for this sample */
                      inner_path->pathtarget->width,
                      0.0,
                      work_mem,
                      -1.0);

            startup_cost += sort_path.startup_cost;
            startup_cost += (sort_path.total_cost - sort_path.startup_cost) * innerstartsel_i;
            inner_run_cost = (sort_path.total_cost - sort_path.startup_cost) * (innerendsel_i - innerstartsel_i);
        } else {
            Cost inner_startup_cost_i =
                    GET_COST(inner_startup_samp, i, inner_path->startup_cost, inner_startup_is_const);
            Cost inner_total_cost_i =
                    GET_COST(inner_total_samp, i, inner_path->total_cost, inner_total_is_const);

            startup_cost += inner_startup_cost_i;
            startup_cost += (inner_total_cost_i - inner_startup_cost_i) * innerstartsel_i;
            inner_run_cost = (inner_total_cost_i - inner_startup_cost_i) * (innerendsel_i - innerstartsel_i);
        }

        /* 3.5) Phase-1 rule: we cannot yet decide on rescans/materialization.
         *      The minimum inner-side cost to include in total is inner_run_cost.
         *      Do NOT add it to run_cost yet (leave that to phase-2 decisions).
         */

        /* 3.6) Record per-sample lower bounds into workspace */
        workspace->startup_cost_sample[i] = startup_cost;
        workspace->total_cost_sample[i] = startup_cost + run_cost + inner_run_cost;

        workspace->run_cost_sample[i] = run_cost;
        workspace->inner_run_cost_sample[i] = inner_run_cost;
        workspace->outer_rows_sample[i] = outer_rows;
        workspace->inner_rows_sample[i] = inner_rows;
        workspace->outer_skip_rows_sample[i] = outer_skip_rows;
        workspace->inner_skip_rows_sample[i] = inner_skip_rows;

        /* 3.7) Accumulate for scalar (mean) outputs */
        startup_accum += startup_cost;
        total_accum += startup_cost + run_cost + inner_run_cost;
        run_accum += run_cost;
        inner_run_accum += inner_run_cost;
        outer_rows_accum += outer_rows;
        inner_rows_accum += inner_rows;
        outer_skip_accum += outer_skip_rows;
        inner_skip_accum += inner_skip_rows;
    }

    /* ----------------------------------------------------------------------
     * 4) Finalize scalar lower bounds as per-sample means
     * ---------------------------------------------------------------------- */
    const double invN = 1.0 / (double) sample_count;

    workspace->startup_cost = startup_accum * invN;
    workspace->total_cost = total_accum * invN;

    workspace->run_cost = run_accum * invN;
    workspace->inner_run_cost = inner_run_accum * invN;

    workspace->outer_rows = outer_rows_accum * invN;
    workspace->inner_rows = inner_rows_accum * invN;
    workspace->outer_skip_rows = outer_skip_accum * invN;
    workspace->inner_skip_rows = inner_skip_accum * invN;
}

void final_cost_mergejoin_2p(
    PlannerInfo *root,
    MergePath *path,
    JoinCostWorkspace *workspace,
    const JoinPathExtraData *extra
) {
    /* ------------------------------- 1) Resolve paths & row samples ------------------------------- */
    const Path *outer_path = path->jpath.outerjoinpath;
    Path *inner_path = path->jpath.innerjoinpath;

    /* Row samples */
    const Sample *outer_rows_samp = outer_path->rows_sample;
    const Sample *inner_rows_samp = inner_path->rows_sample;

    Assert(outer_rows_samp != NULL && outer_rows_samp->sample_count > 0);
    Assert(inner_rows_samp != NULL && inner_rows_samp->sample_count > 0);
    const bool outer_rows_is_const = outer_rows_samp->sample_count == 1;
    const bool inner_rows_is_const = inner_rows_samp->sample_count == 1;

    /* Authoritative loop size comes from the workspace */
    Assert(workspace->sample_count > 0);
    const int sample_count = workspace->sample_count;

    /* ------------------------------- 2) Set scalar output rows (+ optional rows_sample) ------------------------------- */
    if (path->jpath.path.param_info) {
        path->jpath.path.rows = path->jpath.path.param_info->ppi_rows;
        path->jpath.path.rows_sample = make_sample_by_single_value(path->jpath.path.param_info->ppi_rows);
    } else {
        path->jpath.path.rows = path->jpath.path.parent->rows;
        path->jpath.path.rows_sample = duplicate_sample(path->jpath.path.parent->rows_sample);
    }

    /* For partial paths, scale row estimate. */
    if (path->jpath.path.parallel_workers > 0) {
        const double parallel_divisor = get_parallel_divisor(&path->jpath.path);

        path->jpath.path.rows = clamp_row_est(path->jpath.path.rows / parallel_divisor);
        Sample *new_sample = make_sample_by_scale_factor(
            path->jpath.path.rows_sample, 1.0 / parallel_divisor
        );
        pfree(path->jpath.path.rows_sample);
        path->jpath.path.rows_sample = new_sample;
    }

    /* ------------------------------- 3) Split merge-qual vs. other-qual costs (sample-invariant) ------------------------------- */
    QualCost merge_qual_cost, qp_qual_cost;
    cost_qual_eval(&merge_qual_cost, path->path_mergeclauses, root);
    cost_qual_eval(&qp_qual_cost, path->jpath.joinrestrictinfo, root);
    qp_qual_cost.startup -= merge_qual_cost.startup;
    qp_qual_cost.per_tuple -= merge_qual_cost.per_tuple;

    const Cost tlist_startup = path->jpath.path.pathtarget->cost.startup;
    const Cost tlist_per_tuple = path->jpath.path.pathtarget->cost.per_tuple;

    /* ------------------------------- 4) Decide skip_mark_restore & materialize (DO THIS ONCE) ------------------------------- */
    /* 4.1) Decide if mark/restore can be skipped (same rule as upstream) */
    if ((path->jpath.jointype == JOIN_SEMI ||
         path->jpath.jointype == JOIN_ANTI ||
         extra->inner_unique) &&
        (list_length(path->jpath.joinrestrictinfo) ==
         list_length(path->path_mergeclauses)))
        path->skip_mark_restore = true;
    else
        path->skip_mark_restore = false;

    /* 4.2) Compute rescannedtuples / rescanratio using SCALARS (stable across samples) */
    double inner_path_rows_scalar = inner_path->rows;
    if (inner_path_rows_scalar <= 0)
        inner_path_rows_scalar = 1;

    /* INNER-semantics estimate for tuples passing merge quals (scalar) */
    const double mergejointuples_scalar =
            approx_tuple_count(root, &path->jpath, path->path_mergeclauses);

    /* From workspace (scalar means) */
    const double inner_rows_scalar = Max(workspace->inner_rows, 1.0);
    // const double outer_rows_scalar = Max(workspace->outer_rows, 1.0);

    double rescannedtuples_scalar;
    if (IsA(outer_path, UniquePath) || path->skip_mark_restore)
        rescannedtuples_scalar = 0.0;
    else {
        rescannedtuples_scalar = mergejointuples_scalar - inner_path_rows_scalar;
        if (rescannedtuples_scalar < 0) rescannedtuples_scalar = 0.0;
    }

    const double rescanratio = 1.0 + (rescannedtuples_scalar / inner_rows_scalar);

    /* 4.3) Decide whether to materialize inner (use scalar costs; DO NOT flip per-sample) */
    const Cost inner_run_cost_scalar = workspace->inner_run_cost; /* from initial phase (scalar mean) */
    const Cost bare_inner_cost =
            inner_run_cost_scalar * rescanratio;

    const Cost mat_inner_cost =
            inner_run_cost_scalar + cpu_operator_cost * inner_rows_scalar * rescanratio;

    if (path->skip_mark_restore) {
        path->materialize_inner = false; /* no mark/restore needed */
    } else if (enable_material && mat_inner_cost < bare_inner_cost) {
        path->materialize_inner = true; /* cheaper to materialize */
    } else if (path->innersortkeys == NIL && !ExecSupportsMarkRestore(inner_path)) {
        path->materialize_inner = true; /* required for correctness */
    } else if (enable_material && path->innersortkeys != NIL &&
               relation_byte_size(inner_path_rows_scalar, inner_path->pathtarget->width) >
               (work_mem * 1024L)) {
        path->materialize_inner = true; /* avoid sort’s mark/restore with spill risk */
    } else {
        path->materialize_inner = false;
    }

    /* ------------------------------- 5) Initialize per-sample outputs ------------------------------- */
    path->jpath.path.startup_cost = 0.0;
    path->jpath.path.startup_cost_sample = initialize_sample(sample_count);
    path->jpath.path.total_cost = 0.0;
    path->jpath.path.total_cost_sample = initialize_sample(sample_count);

    /* Optional: match original behavior of penalizing when mergejoin is disabled. */
    if (!enable_mergejoin) {
        workspace->startup_cost += disable_cost;
        for (int i = 0; i < sample_count; ++i) {
            workspace->startup_cost_sample[i] += disable_cost;
        }
    }

    /* ------------------------------- 6) Per-sample refinement ------------------------------- */
    Cost startup_accum = 0.0, total_accum = 0.0;

    for (int i = 0; i < sample_count; ++i) {
        /* 6.1) Start from initial lower bounds for this sample */
        Cost startup_cost_i = workspace->startup_cost_sample[i];
        Cost run_cost_i = workspace->run_cost_sample[i];
        const Cost inner_run_cost_i = workspace->inner_run_cost_sample[i];

        /* 6.2) Sample-specific row counts (protect against <= 0) */
        double outer_rows_i = GET_ROW(outer_rows_samp, i, outer_path->rows, outer_rows_is_const);
        double inner_rows_i = GET_ROW(inner_rows_samp, i, inner_path->rows, inner_rows_is_const);
        const double outer_skip_i = workspace->outer_skip_rows_sample[i];
        const double inner_skip_i = workspace->inner_skip_rows_sample[i];
        if (outer_rows_i <= 0) outer_rows_i = 1;
        if (inner_rows_i <= 0) inner_rows_i = 1;

        /* 6.3) Charge the right incremental inner cost based on chosen strategy (fixed rescanratio) */
        if (path->materialize_inner)
            run_cost_i += inner_run_cost_i + cpu_operator_cost * inner_rows_i * rescanratio;
        else
            run_cost_i += inner_run_cost_i * rescanratio;

        /* 6.4) CPU for merge quals: compare cost on skipped vs scanned portions */
        startup_cost_i += merge_qual_cost.startup;
        startup_cost_i += merge_qual_cost.per_tuple *
                (outer_skip_i + inner_skip_i * rescanratio);
        run_cost_i += merge_qual_cost.per_tuple *
        ((outer_rows_i - outer_skip_i) +
         (inner_rows_i - inner_skip_i) * rescanratio);

        /* 6.5) CPU for qp quals (tuples that pass the merge join proper) */
        const double mergejointuples_i = mergejointuples_scalar; /* keep scalar (inner semantics) */
        startup_cost_i += qp_qual_cost.startup;
        run_cost_i += (cpu_tuple_cost + qp_qual_cost.per_tuple) * mergejointuples_i;

        /* 6.6) Tlist costs: startup once; per-tuple on output rows (scalar rows) */
        startup_cost_i += tlist_startup;
        run_cost_i += tlist_per_tuple * path->jpath.path.rows;

        /* 6.7) Store results */
        const Cost total_cost_i = startup_cost_i + run_cost_i;
        path->jpath.path.startup_cost_sample->sample[i] = startup_cost_i;
        path->jpath.path.total_cost_sample->sample[i] = total_cost_i;

        startup_accum += startup_cost_i;
        total_accum += total_cost_i;
    }

    /* ------------------------------- 7) Finalize scalar outputs (means) ------------------------------- */
    const double invN = 1.0 / (double) sample_count;
    path->jpath.path.startup_cost = startup_accum * invN;
    path->jpath.path.total_cost = total_accum * invN;
}

/* ==== ==== ==== ==== ==== ==== 2-PASS HASH JOIN COST MODEL ==== ==== ==== ==== ==== ==== */

void initial_cost_hashjoin_2p(
    PlannerInfo *root,
    JoinCostWorkspace *workspace,
    JoinType jointype,
    const List *hashclauses,
    Path *outer_path,
    Path *inner_path,
    JoinPathExtraData *extra,
    bool parallel_hash
) {
    /* ------------------------------------------------------------------
     * 1) Resolve samples and pick loop count
     * ------------------------------------------------------------------ */
    const Sample *outer_rows_samp = outer_path->rows_sample;
    const Sample *inner_rows_samp = inner_path->rows_sample;

    const Sample *outer_startup_samp = outer_path->startup_cost_sample;
    const Sample *outer_total_samp = outer_path->total_cost_sample;

    const Sample *inner_startup_samp = inner_path->startup_cost_sample;
    const Sample *inner_total_samp = inner_path->total_cost_sample;

    Assert(outer_rows_samp != NULL && outer_rows_samp->sample_count > 0);
    Assert(inner_rows_samp != NULL && inner_rows_samp->sample_count > 0);
    const bool outer_rows_is_const = outer_rows_samp->sample_count == 1;
    const bool inner_rows_is_const = inner_rows_samp->sample_count == 1;

    Assert(outer_startup_samp != NULL && outer_startup_samp->sample_count > 0);
    Assert(outer_total_samp != NULL && outer_total_samp->sample_count > 0);
    const bool outer_startup_is_const = outer_startup_samp->sample_count == 1;
    const bool outer_total_is_const = outer_total_samp->sample_count == 1;

    Assert(inner_startup_samp != NULL && inner_startup_samp->sample_count > 0);
    Assert(inner_total_samp != NULL && inner_total_samp->sample_count > 0);
    const bool inner_startup_is_const = inner_startup_samp->sample_count == 1;
    const bool inner_total_is_const = inner_total_samp->sample_count == 1;

    /* Decide loop sample_count: if all have N>1, require same N; otherwise use the non-1 side. */
    int sample_count;
    if (outer_rows_is_const && inner_rows_is_const
        && outer_startup_is_const && outer_total_is_const
        && inner_startup_is_const && inner_total_is_const) {
        sample_count = 1;
    } else {
        sample_count = error_sample_count;
    }
    workspace->sample_count = sample_count;

    /* ------------------------------------------------------------------
     * 2) Precompute constants used across samples
     * ------------------------------------------------------------------ */
    const int num_hashclauses = list_length(hashclauses);
    /* Note: CPU costs per row are applied per-sample using sampled row counts. */

    /* ------------------------------------------------------------------
     * 3) Accumulators for scalar (mean) lower bounds
     * ------------------------------------------------------------------ */
    Cost startup_accum = 0;
    Cost total_accum = 0;
    Cost run_accum = 0;

    /* Optional: keep representative scalars for hash params */
    double numbuckets_acc = 0.0;
    double numbatches_acc = 0.0;
    double inner_rows_total_acc = 0.0;

    /* Ensure sample arrays in workspace are considered initialized elsewhere. */

    /* ------------------------------------------------------------------
     * 4) Per-sample evaluation (point-to-point)
     * ------------------------------------------------------------------ */
    for (int i = 0; i < sample_count; ++i) {
        /* 4.1) Fetch per-sample rows (guard against <= 0) */
        double outer_rows_i = GET_ROW(outer_rows_samp, i, outer_path->rows, outer_rows_is_const);
        double inner_rows_i = GET_ROW(inner_rows_samp, i, inner_path->rows, inner_rows_is_const);
        if (outer_rows_i <= 0) outer_rows_i = 1;
        if (inner_rows_i <= 0) inner_rows_i = 1;

        /* 4.2) Fetch per-sample input costs for both sides */
        const Cost outer_startup_i = GET_COST(outer_startup_samp, i, outer_path->startup_cost, outer_startup_is_const);
        const Cost outer_total_i = GET_COST(outer_total_samp, i, outer_path->total_cost, outer_total_is_const);
        // const Cost inner_startup_i = GET_COST(inner_startup_samp, i, inner_path->startup_cost, inner_startup_is_const);
        const Cost inner_total_i = GET_COST(inner_total_samp, i, inner_path->total_cost, inner_total_is_const);

        /* 4.3) Source data costs (lower bound): outer contributes startup+run; inner contributes total as startup */
        Cost startup_cost = 0;
        Cost run_cost = 0;

        startup_cost += outer_startup_i;
        run_cost += (outer_total_i - outer_startup_i);
        startup_cost += inner_total_i;

        /* 4.4) Hash function + build costs (phase-1 CPU lower bound)
         *   - inner: (hash per clause + tuple insert) * inner_rows
         *   - outer: (hash per clause) * outer_rows
         */
        startup_cost += (cpu_operator_cost * num_hashclauses + cpu_tuple_cost) * inner_rows_i;
        run_cost += cpu_operator_cost * num_hashclauses * outer_rows_i;

        /* 4.5) Parallel hash: adjust inner_rows_total for table sizing */
        double inner_rows_total_i = inner_rows_i;
        if (parallel_hash)
            inner_rows_total_i *= get_parallel_divisor(inner_path);

        /* 4.6) Choose hash table size for this sample */
        int numbuckets_i, numbatches_i, num_skew_mcvs_i;
        size_t space_allowed_dummy;

        ExecChooseHashTableSize(
            inner_rows_total_i,
            inner_path->pathtarget->width,
            true, /* useskew */
            parallel_hash, /* try_combined_hash_mem */
            outer_path->parallel_workers,
            &space_allowed_dummy,
            &numbuckets_i,
            &numbatches_i,
            &num_skew_mcvs_i
        );

        /* 4.7) If batching, account for I/O costs per sample (sequential) */
        if (numbatches_i > 1) {
            const double outerpages = page_size(outer_rows_i, outer_path->pathtarget->width);
            const double innerpages = page_size(inner_rows_i, inner_path->pathtarget->width);

            /* Write inner once (startup), then read inner + write+read outer (run) */
            startup_cost += seq_page_cost * innerpages;
            run_cost += seq_page_cost * (innerpages + 2 * outerpages);
        }

        /* 4.8) Record per-sample lower bounds into workspace */
        workspace->startup_cost_sample[i] = startup_cost;
        workspace->total_cost_sample[i] = startup_cost + run_cost;
        workspace->run_cost_sample[i] = run_cost;

        workspace->numbuckets_sample[i] = numbuckets_i;
        workspace->numbatches_sample[i] = numbatches_i;
        workspace->inner_rows_total_sample[i] = inner_rows_total_i;

        /* 4.9) Accumulate for scalar outputs */
        startup_accum += startup_cost;
        total_accum += startup_cost + run_cost;
        run_accum += run_cost;

        numbuckets_acc += (double) numbuckets_i;
        numbatches_acc += (double) numbatches_i;
        inner_rows_total_acc += inner_rows_total_i;
    }

    /* ------------------------------------------------------------------
     * 5) Finalize scalar lower bounds as per-sample means
     * ------------------------------------------------------------------ */
    const double invN = 1.0 / (double) sample_count;

    workspace->startup_cost = startup_accum * invN;
    workspace->total_cost = total_accum * invN;
    workspace->run_cost = run_accum * invN;

    /* Representative scalar params for phase-2. */
    workspace->numbuckets = (int) rint(numbuckets_acc * invN);
    workspace->numbuckets = pg_nextpower2_32(workspace->numbuckets);

    workspace->numbatches = (int) rint(numbatches_acc * invN);
    workspace->numbatches = pg_nextpower2_32(workspace->numbatches);

    workspace->inner_rows_total = inner_rows_total_acc * invN;

    /* CPU costs for join quals etc. are left to final_cost_hashjoin (phase-2). */
}

void final_cost_hashjoin_2p(
    PlannerInfo *root,
    HashPath *path,
    JoinCostWorkspace *workspace,
    const JoinPathExtraData *extra
) {
    /* ------------------------------- 1) Resolve paths & input samples ------------------------------- */
    const Path *outer_path = path->jpath.outerjoinpath;
    const Path *inner_path = path->jpath.innerjoinpath;

    /* Row samples */
    const Sample *outer_rows_samp = outer_path->rows_sample;
    const Sample *inner_rows_samp = inner_path->rows_sample;

    Assert(outer_rows_samp != NULL && outer_rows_samp->sample_count > 0);
    Assert(inner_rows_samp != NULL && inner_rows_samp->sample_count > 0);
    const bool outer_rows_is_const = outer_rows_samp->sample_count == 1;
    const bool inner_rows_is_const = inner_rows_samp->sample_count == 1;

    /* Authoritative loop size comes from the workspace */
    Assert(workspace->sample_count > 0);
    const int sample_count = workspace->sample_count;

    /* ------------------------------- 2) Determine scalar output rows (+ rows_sample) ------------------------------- */
    /* Mark rows (scalar) per PG semantics: param_info or parent->rows; also copy rows_sample */
    if (path->jpath.path.param_info) {
        path->jpath.path.rows = path->jpath.path.param_info->ppi_rows;
        path->jpath.path.rows_sample = make_sample_by_single_value(path->jpath.path.param_info->ppi_rows);
    } else {
        path->jpath.path.rows = path->jpath.path.parent->rows;
        path->jpath.path.rows_sample = duplicate_sample(path->jpath.path.parent->rows_sample);
    }

    /* For partial paths, scale row estimate. */
    if (path->jpath.path.parallel_workers > 0) {
        const double parallel_divisor = get_parallel_divisor(&path->jpath.path);

        path->jpath.path.rows = clamp_row_est(path->jpath.path.rows / parallel_divisor);
        Sample *new_sample = make_sample_by_scale_factor(
            path->jpath.path.rows_sample, 1.0 / parallel_divisor
        );
        pfree(path->jpath.path.rows_sample);
        path->jpath.path.rows_sample = new_sample;
    }

    /* ------------------------------- 3) Initialize per-sample output slots ------------------------------- */
    path->jpath.path.startup_cost = 0.0;
    path->jpath.path.startup_cost_sample = initialize_sample(sample_count);
    path->jpath.path.total_cost = 0.0;
    path->jpath.path.total_cost_sample = initialize_sample(sample_count);

    /* Optional: match original behavior of penalizing when hashjoin is disabled. */
    if (!enable_hashjoin) {
        workspace->startup_cost += disable_cost;
        for (int i = 0; i < sample_count; ++i) {
            workspace->startup_cost_sample[i] += disable_cost;
        }
    }

    /* ------------------------------- 4) Precompute sample-invariant pieces ------------------------------- */
    List *hashclauses = path->path_hashclauses;

    /* Qual costs: split hash join quals from the rest to avoid double-charging */
    QualCost hash_qual_cost;
    QualCost qp_qual_cost;
    cost_qual_eval(&hash_qual_cost, hashclauses, root);
    cost_qual_eval(&qp_qual_cost, path->jpath.joinrestrictinfo, root);
    qp_qual_cost.startup -= hash_qual_cost.startup;
    qp_qual_cost.per_tuple -= hash_qual_cost.per_tuple;

    const Cost tlist_startup = path->jpath.path.pathtarget->cost.startup;
    const Cost tlist_per_tuple = path->jpath.path.pathtarget->cost.per_tuple;

    /* For INNER-like semantics, this scalar is OK for non-SEMI/ANTI branch */
    const double hashjointuples_inner = approx_tuple_count(root, &path->jpath, hashclauses);

    /* ------------------------------- 5) Accumulators for scalar (means) ------------------------------- */
    Cost startup_accum = 0.0;
    Cost total_accum = 0.0;

    /* ------------------------------- 6) Per-sample evaluation ------------------------------- */
    for (int i = 0; i < sample_count; ++i) {
        /* 6.1) Start from phase-1 lower bounds (per sample) */
        Cost startup_cost_i = workspace->startup_cost_sample[i];
        Cost run_cost_i = workspace->run_cost_sample[i];

        /* 6.2) Per-sample input rows (guard against <= 0) */
        double outer_rows_i = GET_ROW(outer_rows_samp, i, outer_path->rows, outer_rows_is_const);
        double inner_rows_i = GET_ROW(inner_rows_samp, i, inner_path->rows, inner_rows_is_const);
        if (outer_rows_i <= 0) outer_rows_i = 1;
        if (inner_rows_i <= 0) inner_rows_i = 1;

        /* 6.3) Per-sample geometry from workspace (buckets, batches, inner total rows) */
        const int numbuckets_i = workspace->numbuckets_sample[i];
        const int numbatches_i = workspace->numbatches_sample[i];

        const double virtualbuckets_i = (double) Max(numbuckets_i, 1) * (double) Max(numbatches_i, 1);

        /* 6.4) Bucketsize fraction & MCV frequency for inner relation (conservative minima) */
        Selectivity innerbucketsize, innermcvfreq;

        if (IsA(inner_path, UniquePath)) {
            /* Unique-ified inner relation hashes uniformly */
            innerbucketsize = 1.0 / virtualbuckets_i;
            innermcvfreq = 0.0;
        } else {
            innerbucketsize = 1.0;
            innermcvfreq = 1.0;

            ListCell *hcl;
            foreach(hcl, hashclauses) {
                RestrictInfo *rinfo = lfirst_node(RestrictInfo, hcl);
                Selectivity thisbucketsize, thismcvfreq;

                if (bms_is_subset(rinfo->right_relids, inner_path->parent->relids)) {
                    /* RHS is inner */
                    estimate_hash_bucket_stats(
                        root,
                        get_rightop(rinfo->clause),
                        virtualbuckets_i,
                        &rinfo->right_mcvfreq,
                        &rinfo->right_bucketsize
                    );
                    thisbucketsize = rinfo->right_bucketsize;
                    thismcvfreq = rinfo->right_mcvfreq;
                } else {
                    Assert(bms_is_subset(rinfo->left_relids, inner_path->parent->relids));
                    /* LHS is inner */
                    estimate_hash_bucket_stats(
                        root,
                        get_leftop(rinfo->clause),
                        virtualbuckets_i,
                        &rinfo->left_mcvfreq,
                        &rinfo->left_bucketsize
                    );
                    thisbucketsize = rinfo->left_bucketsize;
                    thismcvfreq = rinfo->left_mcvfreq;
                }

                if (innerbucketsize > thisbucketsize) {
                    innerbucketsize = thisbucketsize;
                }
                if (innermcvfreq > thismcvfreq) {
                    innermcvfreq = thismcvfreq;
                }
            }
        }

        /* 6.5) If MCV bucket would exceed hash_mem, penalize (disable_cost) */
        if (relation_byte_size(clamp_row_est(inner_rows_i * innermcvfreq),
                               inner_path->pathtarget->width) > get_hash_memory_limit())
            startup_cost_i += disable_cost;

        /* 6.6) Hash-qual CPU + comparisons, then qp-qual & tlist */
        double hashjointuples_i;

        if (path->jpath.jointype == JOIN_SEMI ||
            path->jpath.jointype == JOIN_ANTI ||
            extra->inner_unique) {
            const double outer_matched_rows = rint(outer_rows_i * extra->semifactors.outer_match_frac);
            const Selectivity inner_scan_frac = 2.0 / (extra->semifactors.match_count + 1.0);

            /* Hash-qual cost: matched rows see a fraction of bucket; halve for exact-hash prefilter */
            startup_cost_i += hash_qual_cost.startup;
            run_cost_i += hash_qual_cost.per_tuple *
                    outer_matched_rows *
                    clamp_row_est(inner_rows_i * innerbucketsize * inner_scan_frac) * 0.5;

            /* Unmatched rows: avg bucket size, few exact matches -> small factor */
            run_cost_i += hash_qual_cost.per_tuple *
                    (outer_rows_i - outer_matched_rows) *
                    clamp_row_est(inner_rows_i / Max(virtualbuckets_i, 1.0)) * 0.05;

            /* Tuples passing the hash join proper */
            hashjointuples_i = (path->jpath.jointype == JOIN_ANTI)
                                   ? (outer_rows_i - outer_matched_rows)
                                   : outer_matched_rows;
        } else {
            /* Non-SEMI/ANTI: outer * (inner*bucketsize), halve for exact-hash prefilter */
            startup_cost_i += hash_qual_cost.startup;
            run_cost_i += hash_qual_cost.per_tuple *
                    outer_rows_i *
                    clamp_row_est(inner_rows_i * innerbucketsize) * 0.5;

            /* Passing tuples estimated once with INNER semantics (scalar) */
            hashjointuples_i = hashjointuples_inner;
        }

        /* qp-qual CPU + tlist costs */
        startup_cost_i += qp_qual_cost.startup;
        run_cost_i += (cpu_tuple_cost + qp_qual_cost.per_tuple) * hashjointuples_i;

        startup_cost_i += tlist_startup;
        run_cost_i += tlist_per_tuple * path->jpath.path.rows;

        /* 6.7) Finalize this sample and accumulate means */
        const Cost total_cost_i = startup_cost_i + run_cost_i;

        path->jpath.path.startup_cost_sample->sample[i] = startup_cost_i;
        path->jpath.path.total_cost_sample->sample[i] = total_cost_i;

        startup_accum += startup_cost_i;
        total_accum += total_cost_i;
    }

    /* ------------------------------- 7) Finalize scalar outputs (means) ------------------------------- */
    const double invN = 1.0 / (double) sample_count;

    path->jpath.path.startup_cost = startup_accum * invN;
    path->jpath.path.total_cost = total_accum * invN;

    /* Save representative #batches and total inner rows (rounded/mean) */
    path->num_batches = workspace->numbatches;
    path->inner_rows_total = workspace->inner_rows_total;
}
