//
// Created by Xuan Chen on 2025/11/2.
//

#include "postgres.h"

#include <math.h>

#include "access/amapi.h"
#include "executor/executor.h"
#include "optimizer/cost.h"
#include "optimizer/cost_ext.h"
#include "optimizer/sample.h"
#include "optimizer/optimizer.h"
#include "optimizer/paths.h"
#include "utils/lsyscache.h"
#include "utils/spccache.h"

/* ==== ==== ==== ==== ==== ==== SCAN COST HELPERS ==== ==== ==== ==== ==== ==== */

static List *extract_nonindex_conditions(
    List *qual_clauses,
    List *indexclauses
);

/*
 * get_restriction_qual_cost
 *	  Compute evaluation costs of a baserel's restriction quals, plus any
 *	  movable join quals that have been pushed down to the scan.
 *	  Results are returned into *qpqual_cost.
 *
 * This is a convenience subroutine that works for seqscans and other cases
 * where all the given quals will be evaluated the hard way.  It's not useful
 * for cost_index(), for example, where the index machinery takes care of
 * some of the quals.  We assume baserestrictcost was previously set by
 * set_baserel_size_estimates().
 */
void get_restriction_qual_cost(
    PlannerInfo *root,
    const RelOptInfo *baserel,
    const ParamPathInfo *param_info,
    QualCost *qpqual_cost
) {
    if (param_info) {
        /* Include costs of pushed-down clauses */
        cost_qual_eval(qpqual_cost, param_info->ppi_clauses, root);

        qpqual_cost->startup += baserel->baserestrictcost.startup;
        qpqual_cost->per_tuple += baserel->baserestrictcost.per_tuple;
    } else {
        *qpqual_cost = baserel->baserestrictcost;
    }
}

/*
 * extract_nonindex_conditions
 *
 * Given a list of quals to be enforced in an indexscan, extract the ones that
 * will have to be applied as qpquals (ie, the index machinery won't handle
 * them).  Here we detect only whether a qual clause is directly redundant
 * with some indexclause.  If the index path is chosen for use, createplan.c
 * will try a bit harder to get rid of redundant qual conditions; specifically
 * it will see if quals can be proven to be implied by the indexquals.  But
 * it does not seem worth the cycles to try to factor that in at this stage,
 * since we're only trying to estimate qual eval costs.  Otherwise this must
 * match the logic in create_indexscan_plan().
 *
 * qual_clauses, and the result, are lists of RestrictInfos.
 * indexclauses is a list of IndexClauses.
 */
static List *extract_nonindex_conditions(
    List *qual_clauses,
    List *indexclauses
) {
    List *result = NIL;
    ListCell *lc;

    foreach(lc, qual_clauses) {
        RestrictInfo *rinfo = lfirst_node(RestrictInfo, lc);

        if (rinfo->pseudoconstant)
            continue; /* we may drop pseudoconstants here */
        if (is_redundant_with_indexclauses(rinfo, indexclauses))
            continue; /* dup or derived from same EquivalenceClass */
        /* ... skip the predicate proof attempt createplan.c will try ... */
        result = lappend(result, rinfo);
    }
    return result;
}

/* ==== ==== ==== ==== ==== ==== SEQ SCAN COST MODEL ==== ==== ==== ==== ==== ==== */

void cost_seqscan_1p(
    Path *path,
    PlannerInfo *root,
    const RelOptInfo *baserel,
    const ParamPathInfo *param_info
) {
    Cost startup_cost = 0;
    double spc_seq_page_cost;
    QualCost qpqual_cost;

    /* Should only be applied to base relations */
    Assert(baserel->relid > 0);
    Assert(baserel->rtekind == RTE_RELATION);

    /* Mark the path with the correct row estimate */
    if (param_info) {
        path->rows = param_info->ppi_rows;
    } else {
        path->rows = baserel->rows;
    }
    path->rows_sample = NULL;

    if (!enable_seqscan) {
        startup_cost += disable_cost;
    }

    /* fetch estimated page cost for tablespace containing table */
    get_tablespace_page_costs(
        baserel->reltablespace, NULL, &spc_seq_page_cost
    );

    /*
     * disk costs
     */
    const Cost disk_run_cost = spc_seq_page_cost * baserel->pages;

    /* CPU costs */
    get_restriction_qual_cost(root, baserel, param_info, &qpqual_cost);

    startup_cost += qpqual_cost.startup;
    const Cost cpu_per_tuple = cpu_tuple_cost + qpqual_cost.per_tuple;
    Cost cpu_run_cost = cpu_per_tuple * baserel->tuples;
    /* tlist eval costs are paid per output row, not per tuple scanned */
    startup_cost += path->pathtarget->cost.startup;
    cpu_run_cost += path->pathtarget->cost.per_tuple * path->rows;

    /* Adjust costing for parallelism, if used. */
    if (path->parallel_workers > 0) {
        const double parallel_divisor = get_parallel_divisor(path);

        /* The CPU cost is divided among all the workers. */
        cpu_run_cost /= parallel_divisor;

        /*
         * It may be possible to amortize some of the I/O cost, but probably
         * not very much, because most operating systems already do aggressive
         * prefetching.  For now, we assume that the disk run cost can't be
         * amortized at all.
         */

        /*
         * In the case of a parallel plan, the row count needs to represent
         * the number of tuples processed per worker.
         */
        path->rows = clamp_row_est(path->rows / parallel_divisor);
    }

    path->startup_cost = startup_cost;
    path->total_cost = startup_cost + cpu_run_cost + disk_run_cost;
}

void cost_seqscan_2p(
    Path *path,
    PlannerInfo *root,
    const RelOptInfo *baserel,
    const ParamPathInfo *param_info
) {
    /* ----------------------------------------------------------------------
     * 0) Preconditions & invariants
     * ----------------------------------------------------------------------
     * This costing function applies only to base relations.
     */
    Assert(baserel->relid > 0);
    Assert(baserel->rtekind == RTE_RELATION);

    /* ----------------------------------------------------------------------
     * 1) Resolve base estimates and (optionally) sample distributions
     * ----------------------------------------------------------------------
     * Set the point estimate for rows (original behavior), and attempt
     * to obtain a per-sample distribution if available. We never mutate
     * the input distribution in-place.
     */
    if (param_info) {
        path->rows = param_info->ppi_rows;
        path->rows_sample = make_sample_by_single_value(param_info->ppi_rows);
    } else {
        path->rows = baserel->rows;
        path->rows_sample = duplicate_sample(baserel->rows_sample);
    }

    /* ----------------------------------------------------------------------
     * 2) Allocate per-sample outputs and initialize accumulators
     * ----------------------------------------------------------------------
     * We compute startup/total costs per sample, then average for the
     * scalar fields to remain backward-compatible with callers that
     * only read the scalar costs. We also average per-worker rows.
     */
    const int sample_count = path->rows_sample->sample_count;
    Assert(sample_count > 0);

    path->startup_cost_sample = initialize_sample(sample_count);
    path->total_cost_sample = initialize_sample(sample_count);

    double rows_accum = 0.0; /* per-worker rows average accumulator */
    Cost startup_accum = 0.0; /* startup cost average accumulator     */
    Cost total_accum = 0.0; /* total cost average accumulator       */

    /* ----------------------------------------------------------------------
     * 3) Precompute terms that do not vary across samples
     * ----------------------------------------------------------------------
     * This reduces overhead and avoids subtle inconsistencies.
     */
    Cost startup_cost_bias = 0;
    if (!enable_seqscan) {
        startup_cost_bias += disable_cost;
    }

    double spc_seq_page_cost;
    get_tablespace_page_costs(baserel->reltablespace, NULL, &spc_seq_page_cost);
    const Cost disk_run_cost = spc_seq_page_cost * baserel->pages;

    QualCost qpqual_cost;
    get_restriction_qual_cost(root, baserel, param_info, &qpqual_cost);
    const Cost cpu_per_tuple = cpu_tuple_cost + qpqual_cost.per_tuple;

    const Cost tlist_startup = path->pathtarget->cost.startup;
    const Cost tlist_per_tuple = path->pathtarget->cost.per_tuple;

    const bool use_parallel = (path->parallel_workers > 0);
    const double parallel_divisor = use_parallel ? get_parallel_divisor(path) : 1.0;

    /* ----------------------------------------------------------------------
     * 4) Per-sample evaluation
     * ----------------------------------------------------------------------
     * For each sample row estimate:
     *  - compute startup cost (constant across samples except for the tlist)
     *  - compute CPU cost split into scan (per tuple scanned) and
     *    tlist (per output row)
     *  - adjust CPU cost and effective per-worker rows under parallelism
     *  - record per-sample totals and accumulate means for scalar outputs
     */
    for (int i = 0; i < sample_count; ++i) {
        const double sample_rows_global = path->rows_sample->sample[i];

        /* 4.1) Startup cost for this sample (no sample-specific addends here). */
        Cost startup_cost = 0;
        startup_cost += startup_cost_bias;
        startup_cost += qpqual_cost.startup;
        startup_cost += tlist_startup;

        /* 4.2) CPU run cost: scan all tuples + tlist per *output* row. */
        Cost cpu_run_cost = cpu_per_tuple * baserel->tuples;

        /* 4.3) Parallel adjustment: divide CPU by workers; rows become per-worker. */
        double worker_rows = sample_rows_global;
        if (use_parallel) {
            cpu_run_cost /= parallel_divisor;
            worker_rows = clamp_row_est(sample_rows_global / parallel_divisor);
        }

        /* 4.4) Add tlist per-output-row cost based on worker_rows. */
        cpu_run_cost += tlist_per_tuple * worker_rows;

        /* 4.5) Total cost for this sample. Disk cost is not amortized. */
        const Cost total_cost = startup_cost + cpu_run_cost + disk_run_cost;

        /* 4.6) Record per-sample results and update accumulators. */
        path->rows_sample->sample[i] = worker_rows;
        path->startup_cost_sample->sample[i] = startup_cost;
        path->total_cost_sample->sample[i] = total_cost;

        rows_accum += worker_rows; /* average per-worker rows across samples */
        startup_accum += startup_cost;
        total_accum += total_cost;
    }

    /* ----------------------------------------------------------------------
     * 5) Finalize scalar outputs by averaging across samples
     * ----------------------------------------------------------------------
     * Keep the classic semantics:
     *  - path->rows is per-worker row estimate under parallelism
     *  - startup_cost/total_cost are the mean of the sample costs
     */
    const double invN = 1.0 / (double) sample_count;
    path->rows = rows_accum * invN;
    path->startup_cost = startup_accum * invN;
    path->total_cost = total_accum * invN;

    /* ----------------------------------------------------------------------
     * 6) Notes on memory ownership
     * ----------------------------------------------------------------------
     * The helper routines duplicate_sample()/initialize_sample() are assumed
     * to allocate in the Planner context so that the samples live as long as
     * the Path. Callers owning shorter-lived contexts must refcount/move if
     * needed.
     */
}

/* ==== ==== ==== ==== ==== ==== INDEX SCAN COST MODEL ==== ==== ==== ==== ==== ==== */

extern void cost_index_1p(
    IndexPath *path,
    PlannerInfo *root,
    const double loop_count,
    const bool partial_path
) {
    IndexOptInfo *index = path->indexinfo;
    RelOptInfo *baserel = index->rel;
    bool indexonly = (path->path.pathtype == T_IndexOnlyScan);
    amcostestimate_function amcostestimate;
    List *qpquals;
    Cost startup_cost = 0;
    Cost run_cost = 0;
    Cost cpu_run_cost = 0;
    Cost indexStartupCost;
    Cost indexTotalCost;
    Selectivity indexSelectivity;
    double indexCorrelation,
            csquared;
    double spc_seq_page_cost,
            spc_random_page_cost;
    Cost min_IO_cost,
            max_IO_cost;
    QualCost qpqual_cost;
    Cost cpu_per_tuple;
    double tuples_fetched;
    double pages_fetched;
    double rand_heap_pages;
    double index_pages;

    /* Should only be applied to base relations */
    Assert(IsA(baserel, RelOptInfo) &&
        IsA(index, IndexOptInfo));
    Assert(baserel->relid > 0);
    Assert(baserel->rtekind == RTE_RELATION);

    /*
     * Mark the path with the correct row estimate, and identify which quals
     * will need to be enforced as qpquals.  We need not check any quals that
     * are implied by the index's predicate, so we can use indrestrictinfo not
     * baserestrictinfo as the list of relevant restriction clauses for the
     * rel.
     */
    if (path->path.param_info) {
        path->path.rows = path->path.param_info->ppi_rows;
        /* qpquals come from the rel's restriction clauses and ppi_clauses */
        qpquals = list_concat(
            extract_nonindex_conditions(path->indexinfo->indrestrictinfo, path->indexclauses),
            extract_nonindex_conditions(path->path.param_info->ppi_clauses, path->indexclauses)
        );
    } else {
        path->path.rows = baserel->rows;
        /* qpquals come from just the rel's restriction clauses */
        qpquals = extract_nonindex_conditions(path->indexinfo->indrestrictinfo, path->indexclauses);
    }
    path->path.rows_sample = NULL;

    if (!enable_indexscan) {
        startup_cost += disable_cost;
    }
    /* we don't need to check enable_indexonlyscan; indxpath.c does that */

    /*
     * Call index-access-method-specific code to estimate the processing cost
     * for scanning the index, as well as the selectivity of the index (ie,
     * the fraction of main-table tuples we will have to retrieve) and its
     * correlation to the main-table tuple order.  We need a cast here because
     * pathnodes.h uses a weak function type to avoid including amapi.h.
     */
    amcostestimate = (amcostestimate_function) index->amcostestimate;
    amcostestimate(
        root, path, loop_count,
        &indexStartupCost, &indexTotalCost,
        &indexSelectivity, &indexCorrelation,
        &index_pages
    );

    /*
     * Save amcostestimate's results for possible use in bitmap scan planning.
     * We don't bother to save indexStartupCost or indexCorrelation, because a
     * bitmap scan doesn't care about either.
     */
    path->indextotalcost = indexTotalCost;
    path->indexselectivity = indexSelectivity;

    /* all costs for touching index itself included here */
    startup_cost += indexStartupCost;
    run_cost += indexTotalCost - indexStartupCost;

    /* estimate number of main-table tuples fetched */
    tuples_fetched = clamp_row_est(indexSelectivity * baserel->tuples);

    /* fetch estimated page costs for tablespace containing table */
    get_tablespace_page_costs(
        baserel->reltablespace,
        &spc_random_page_cost,
        &spc_seq_page_cost
    );

    /*----------
     * Estimate number of main-table pages fetched, and compute I/O cost.
     *
     * When the index ordering is uncorrelated with the table ordering,
     * we use an approximation proposed by Mackert and Lohman (see
     * index_pages_fetched() for details) to compute the number of pages
     * fetched, and then charge spc_random_page_cost per page fetched.
     *
     * When the index ordering is exactly correlated with the table ordering
     * (just after a CLUSTER, for example), the number of pages fetched should
     * be exactly selectivity * table_size.  What's more, all but the first
     * will be sequential fetches, not the random fetches that occur in the
     * uncorrelated case.  So if the number of pages is more than 1, we
     * ought to charge
     *		spc_random_page_cost + (pages_fetched - 1) * spc_seq_page_cost
     * For partially-correlated indexes, we ought to charge somewhere between
     * these two estimates.  We currently interpolate linearly between the
     * estimates based on the correlation squared (XXX is that appropriate?).
     *
     * If it's an index-only scan, then we will not need to fetch any heap
     * pages for which the visibility map shows all tuples are visible.
     * Hence, reduce the estimated number of heap fetches accordingly.
     * We use the measured fraction of the entire heap that is all-visible,
     * which might not be particularly relevant to the subset of the heap
     * that this query will fetch; but it's not clear how to do better.
     *----------
     */
    if (loop_count > 1) {
        /*
         * For repeated indexscans, the appropriate estimate for the
         * uncorrelated case is to scale up the number of tuples fetched in
         * the Mackert and Lohman formula by the number of scans, so that we
         * estimate the number of pages fetched by all the scans; then
         * pro-rate the costs for one scan.  In this case we assume all the
         * fetches are random accesses.
         */
        pages_fetched = index_pages_fetched(
            tuples_fetched * loop_count,
            baserel->pages,
            (double) index->pages,
            root
        );

        if (indexonly)
            pages_fetched = ceil(pages_fetched * (1.0 - baserel->allvisfrac));

        rand_heap_pages = pages_fetched;

        max_IO_cost = (pages_fetched * spc_random_page_cost) / loop_count;

        /*
         * In the perfectly correlated case, the number of pages touched by
         * each scan is selectivity * table_size, and we can use the Mackert
         * and Lohman formula at the page level to estimate how much work is
         * saved by caching across scans.  We still assume all the fetches are
         * random, though, which is an overestimate that's hard to correct for
         * without double-counting the cache effects.  (But in most cases
         * where such a plan is actually interesting, only one page would get
         * fetched per scan anyway, so it shouldn't matter much.)
         */
        pages_fetched = ceil(indexSelectivity * (double) baserel->pages);

        pages_fetched = index_pages_fetched(
            pages_fetched * loop_count,
            baserel->pages,
            (double) index->pages,
            root
        );

        if (indexonly)
            pages_fetched = ceil(pages_fetched * (1.0 - baserel->allvisfrac));

        min_IO_cost = (pages_fetched * spc_random_page_cost) / loop_count;
    } else {
        /*
         * Normal case: apply the Mackert and Lohman formula, and then
         * interpolate between that and the correlation-derived result.
         */
        pages_fetched = index_pages_fetched(
            tuples_fetched,
            baserel->pages,
            (double) index->pages,
            root
        );

        if (indexonly) {
            pages_fetched = ceil(pages_fetched * (1.0 - baserel->allvisfrac));
        }

        rand_heap_pages = pages_fetched;

        /* max_IO_cost is for the perfectly uncorrelated case (csquared=0) */
        max_IO_cost = pages_fetched * spc_random_page_cost;

        /* min_IO_cost is for the perfectly correlated case (csquared=1) */
        pages_fetched = ceil(indexSelectivity * (double) baserel->pages);

        if (indexonly) {
            pages_fetched = ceil(pages_fetched * (1.0 - baserel->allvisfrac));
        }

        if (pages_fetched > 0) {
            min_IO_cost = spc_random_page_cost;
            if (pages_fetched > 1) {
                min_IO_cost += (pages_fetched - 1) * spc_seq_page_cost;
            }
        } else {
            min_IO_cost = 0;
        }
    }

    if (partial_path) {
        /*
         * For index only scans compute workers based on number of index pages
         * fetched; the number of heap pages we fetch might be so small as to
         * effectively rule out parallelism, which we don't want to do.
         */
        if (indexonly) {
            rand_heap_pages = -1;
        }

        /*
         * Estimate the number of parallel workers required to scan index. Use
         * the number of heap pages computed considering heap fetches won't be
         * sequential as for parallel scans the pages are accessed in random
         * order.
         */
        path->path.parallel_workers = compute_parallel_worker(
            baserel,
            rand_heap_pages,
            index_pages,
            max_parallel_workers_per_gather
        );

        /*
         * Fall out if workers can't be assigned for parallel scan, because in
         * such a case this path will be rejected.  So there is no benefit in
         * doing extra computation.
         */
        if (path->path.parallel_workers <= 0) {
            return;
        }

        path->path.parallel_aware = true;
    }

    /*
     * Now interpolate based on estimated index order correlation to get total
     * disk I/O cost for main table accesses.
     */
    csquared = indexCorrelation * indexCorrelation;

    run_cost += max_IO_cost + csquared * (min_IO_cost - max_IO_cost);

    /*
     * Estimate CPU costs per tuple.
     *
     * What we want here is cpu_tuple_cost plus the evaluation costs of any
     * qual clauses that we have to evaluate as qpquals.
     */
    cost_qual_eval(&qpqual_cost, qpquals, root);

    startup_cost += qpqual_cost.startup;
    cpu_per_tuple = cpu_tuple_cost + qpqual_cost.per_tuple;

    cpu_run_cost += cpu_per_tuple * tuples_fetched;

    /* tlist eval costs are paid per output row, not per tuple scanned */
    startup_cost += path->path.pathtarget->cost.startup;
    cpu_run_cost += path->path.pathtarget->cost.per_tuple * path->path.rows;

    /* Adjust costing for parallelism, if used. */
    if (path->path.parallel_workers > 0) {
        const double parallel_divisor = get_parallel_divisor(&path->path);

        path->path.rows = clamp_row_est(path->path.rows / parallel_divisor);

        /* The CPU cost is divided among all the workers. */
        cpu_run_cost /= parallel_divisor;
    }

    run_cost += cpu_run_cost;

    path->path.startup_cost = startup_cost;
    path->path.total_cost = startup_cost + run_cost;
}

extern void cost_index_2p(
    IndexPath *path,
    PlannerInfo *root,
    const double loop_count,
    const bool partial_path
) {
    /* ----------------------------------------------------------------------
     * 0) Preconditions & basic objects
     * ----------------------------------------------------------------------
     * Applies only to base relations; fetch key structs early.
     */
    IndexOptInfo *index = path->indexinfo;
    RelOptInfo *baserel = index->rel;
    bool indexonly = (path->path.pathtype == T_IndexOnlyScan);

    Assert(IsA(baserel, RelOptInfo) && IsA(index, IndexOptInfo));
    Assert(baserel->relid > 0);
    Assert(baserel->rtekind == RTE_RELATION);

    /* ----------------------------------------------------------------------
     * 1) Resolve row estimates and qpquals; clone (don't mutate) samples
     * ----------------------------------------------------------------------
     * Use param_info (if any) for rows; qpquals are "non-index" quals.
     */
    List *qpquals;
    if (path->path.param_info) {
        path->path.rows = path->path.param_info->ppi_rows;
        path->path.rows_sample = make_sample_by_single_value(path->path.param_info->ppi_rows);

        qpquals = list_concat(
            extract_nonindex_conditions(path->indexinfo->indrestrictinfo, path->indexclauses),
            extract_nonindex_conditions(path->path.param_info->ppi_clauses, path->indexclauses)
        );
    } else {
        path->path.rows = baserel->rows;
        path->path.rows_sample = duplicate_sample(baserel->rows_sample);

        qpquals = extract_nonindex_conditions(path->indexinfo->indrestrictinfo, path->indexclauses);
    }

    /* ----------------------------------------------------------------------
     * 2) Allocate per-sample outputs; set accumulators for scalar means
     * ----------------------------------------------------------------------
     * We will keep per-sample startup/total costs and average into scalars.
     */
    const int sample_count = path->path.rows_sample->sample_count;
    Assert(sample_count > 0);

    path->path.startup_cost_sample = initialize_sample(sample_count);
    path->path.total_cost_sample = initialize_sample(sample_count);

    double rows_accum = 0.0; /* per-worker rows */
    Cost startup_accum = 0.0;
    Cost total_accum = 0.0;

    /* ----------------------------------------------------------------------
     * 3) Precompute sample-invariant terms (I/O, quals, AM costs, etc.)
     * ----------------------------------------------------------------------
     * Avoid recomputing inside the loop; prevents subtle inconsistencies.
     */
    amcostestimate_function amcostestimate = (amcostestimate_function) index->amcostestimate;

    Cost indexStartupCost;
    Cost indexTotalCost;
    Selectivity indexSelectivity;
    double indexCorrelation;
    double index_pages; /* double in API */

    amcostestimate(
        root, path, loop_count,
        &indexStartupCost, &indexTotalCost,
        &indexSelectivity, &indexCorrelation,
        &index_pages
    );

    /* Save for bitmap planning (as in upstream). */
    path->indextotalcost = indexTotalCost;
    path->indexselectivity = indexSelectivity;

    /* Costs for touching the index itself. */
    const Cost run_cost_index = indexTotalCost - indexStartupCost;
    Cost startup_bias = 0;
    if (!enable_indexscan)
        startup_bias += disable_cost;

    /* Fetch tablespace page costs (heap I/O). */
    double spc_seq_page_cost, spc_random_page_cost;
    get_tablespace_page_costs(
        baserel->reltablespace,
        &spc_random_page_cost,
        &spc_seq_page_cost
    );

    /* Estimate heap tuples fetched (selectivity * table tuples). */
    const double tuples_fetched = clamp_row_est(indexSelectivity * baserel->tuples);

    /* Compute heap page I/O costs per original logic (independent of samples). */
    Cost max_IO_cost, min_IO_cost;
    double pages_fetched, rand_heap_pages;

    if (loop_count > 1) {
        pages_fetched = index_pages_fetched(
            tuples_fetched * loop_count,
            baserel->pages,
            (double) index->pages,
            root
        );
        if (indexonly) {
            pages_fetched = ceil(pages_fetched * (1.0 - baserel->allvisfrac));
        }

        rand_heap_pages = pages_fetched;
        max_IO_cost = (pages_fetched * spc_random_page_cost) / loop_count;

        pages_fetched = ceil(indexSelectivity * (double) baserel->pages);
        pages_fetched = index_pages_fetched(
            pages_fetched * loop_count,
            baserel->pages,
            (double) index->pages,
            root
        );
        if (indexonly)
            pages_fetched = ceil(pages_fetched * (1.0 - baserel->allvisfrac));

        min_IO_cost = (pages_fetched * spc_random_page_cost) / loop_count;
    } else {
        pages_fetched = index_pages_fetched(
            tuples_fetched,
            baserel->pages,
            (double) index->pages,
            root
        );
        if (indexonly) {
            pages_fetched = ceil(pages_fetched * (1.0 - baserel->allvisfrac));
        }

        rand_heap_pages = pages_fetched;
        max_IO_cost = pages_fetched * spc_random_page_cost;

        pages_fetched = ceil(indexSelectivity * (double) baserel->pages);
        if (indexonly) {
            pages_fetched = ceil(pages_fetched * (1.0 - baserel->allvisfrac));
        }

        if (pages_fetched > 0) {
            min_IO_cost = spc_random_page_cost;
            if (pages_fetched > 1) {
                min_IO_cost += (pages_fetched - 1) * spc_seq_page_cost;
            }
        } else {
            min_IO_cost = 0;
        }
    }

    /* Optionally determine parallel workers (upstream semantics). */
    if (partial_path) {
        if (indexonly) {
            rand_heap_pages = -1;
        }

        path->path.parallel_workers = compute_parallel_worker(
            baserel,
            rand_heap_pages,
            index_pages,
            max_parallel_workers_per_gather
        );

        if (path->path.parallel_workers <= 0) {
            return;
        }

        path->path.parallel_aware = true;
    }

    /* Interpolate I/O cost based on correlation. */
    const double csquared = indexCorrelation * indexCorrelation;
    const Cost heap_io_cost = max_IO_cost + csquared * (min_IO_cost - max_IO_cost);

    /* Qual eval costs (sample-invariant). */
    QualCost qpqual_cost;
    cost_qual_eval(&qpqual_cost, qpquals, root);

    const Cost tlist_startup = path->path.pathtarget->cost.startup;
    const Cost tlist_per_tuple = path->path.pathtarget->cost.per_tuple;
    const Cost cpu_per_tuple = cpu_tuple_cost + qpqual_cost.per_tuple;

    const bool use_parallel = (path->path.parallel_workers > 0);
    const double parallel_divisor = use_parallel ? get_parallel_divisor(&path->path) : 1.0;

    /* Base startup cost (no sample dependence). */
    const Cost startup_base = startup_bias + indexStartupCost + qpqual_cost.startup + tlist_startup;

    /* Base run cost parts independent of samples. */
    const Cost run_cost_base = run_cost_index + heap_io_cost;

    /* ----------------------------------------------------------------------
     * 4) Per-sample evaluation (do NOT mutate input samples)
     * ----------------------------------------------------------------------
     * For each sample row estimate:
     *  - compute per-sample CPU run cost (tlist per output row)
     *  - apply parallel divisor to CPU & rows (not to heap I/O)
     *  - record sample startup/total; accumulate means for scalars
     */
    for (int i = 0; i < sample_count; ++i) {
        const double sample_rows_global = path->path.rows_sample->sample[i];

        /* 5.1) Per-sample startup cost: identical across samples here. */
        const Cost startup_cost = startup_base;

        /* 5.2) CPU run cost = scan-all-tuples + tlist per output row. */
        Cost cpu_run_cost = cpu_per_tuple * tuples_fetched;
        double worker_rows = sample_rows_global;

        if (use_parallel) {
            cpu_run_cost /= parallel_divisor;
            worker_rows = clamp_row_est(sample_rows_global / parallel_divisor);
        }

        cpu_run_cost += tlist_per_tuple * worker_rows;

        /* 5.3) Total run cost = base (index + heap I/O) + per-sample CPU. */
        const Cost run_cost = run_cost_base + cpu_run_cost;
        const Cost total_cost = startup_cost + run_cost;

        /* 5.4) Record per-sample outputs and update accumulators. */
        path->path.rows_sample->sample[i] = worker_rows;
        path->path.startup_cost_sample->sample[i] = startup_cost;
        path->path.total_cost_sample->sample[i] = total_cost;

        rows_accum += worker_rows; /* per-worker rows mean */
        startup_accum += startup_cost;
        total_accum += total_cost;
    }

    /* ----------------------------------------------------------------------
     * 5) Finalize scalar outputs by averaging across samples
     * ----------------------------------------------------------------------
     * Keep classic semantics (rows are per-worker under parallelism).
     */
    const double invN = 1.0 / (double) sample_count;
    path->path.rows = rows_accum * invN;
    path->path.startup_cost = startup_accum * invN;
    path->path.total_cost = total_accum * invN;

    /* ----------------------------------------------------------------------
     * 6) Notes
     * ----------------------------------------------------------------------
     * - We do not amortize heap I/O under parallelism (same as upstream).
     * - Helpers duplicate_sample()/initialize_sample() are assumed to
     *   allocate in the Planner context so samples live as long as the Path.
     */
}

/* ==== ==== ==== ==== ==== ==== GATHER COST MODEL ==== ==== ==== ==== ==== ==== */

extern void cost_gather_1p(
    GatherPath *path,
    PlannerInfo *root,
    const RelOptInfo *rel,
    const ParamPathInfo *param_info,
    const double *rows
) {
    Cost startup_cost = 0;
    Cost run_cost = 0;

    /* Mark the path with the correct row estimate */
    if (rows) {
        path->path.rows = *rows;
    } else if (param_info) {
        path->path.rows = param_info->ppi_rows;
    } else {
        path->path.rows = rel->rows;
    }
    path->path.rows_sample = NULL;

    startup_cost = path->subpath->startup_cost;

    run_cost = path->subpath->total_cost - path->subpath->startup_cost;

    /* Parallel setup and communication cost. */
    startup_cost += parallel_setup_cost;
    run_cost += parallel_tuple_cost * path->path.rows;

    path->path.startup_cost = startup_cost;
    path->path.total_cost = (startup_cost + run_cost);
}

extern void cost_gather_2p(
    GatherPath *path,
    PlannerInfo *root,
    const RelOptInfo *rel,
    const ParamPathInfo *param_info,
    const double *rows
) {
    /* ------------------------------- 1) Resolve rows & sample ------------------------------- */
    /* Path rows (scalar) per PG semantics; also prepare a rows_sample for per-sample loop */
    if (rows) {
        path->path.rows = *rows;
        path->path.rows_sample = make_sample_by_single_value(*rows);
    } else if (param_info) {
        path->path.rows = param_info->ppi_rows;
        path->path.rows_sample = make_sample_by_single_value(param_info->ppi_rows);
    } else {
        path->path.rows = rel->rows;
        path->path.rows_sample = duplicate_sample(rel->rows_sample);
    }

    /* ------------------------------- 2) Subpath costs (scalar + sample) ------------------------------- */
    const Path *subpath = path->subpath;

    const Sample *rows_samp = path->path.rows_sample; /* may be NULL */
    const Sample *sub_startup_samp = subpath->startup_cost_sample;
    const Sample *sub_total_samp = subpath->total_cost_sample;

    const bool rows_is_const
            = rows_samp == NULL || rows_samp->sample_count <= 1;
    const bool sub_startup_is_const
            = sub_startup_samp == NULL || sub_startup_samp->sample_count <= 1;
    const bool sub_total_is_const
            = sub_total_samp == NULL || sub_total_samp->sample_count <= 1;

    /* Decide loop sample_count: if all have N>1, require same N; otherwise use the non-1 side. */
    int sample_count;
    if (sub_startup_is_const && sub_total_is_const && rows_is_const) {
        sample_count = 1;
    } else {
        sample_count = error_sample_count;
    }

    path->path.startup_cost = 0.0;
    path->path.startup_cost_sample = initialize_sample(sample_count);
    path->path.total_cost = 0.0;
    path->path.total_cost_sample = initialize_sample(sample_count);

    /* ------------------------------- 4) Per-sample aggregation ------------------------------- */
    for (int i = 0; i < sample_count; ++i) {
        /* 4.1) Read subpath costs for this sample (fallback to scalar if no samples) */
        const Cost sub_startup_i
                = GET_COST(sub_startup_samp, i, path->subpath->startup_cost, sub_startup_is_const);
        const Cost sub_total_i
                = GET_COST(sub_total_samp, i, path->subpath->total_cost, sub_total_is_const);
        Cost startup_cost = sub_startup_i;
        Cost run_cost = sub_total_i - sub_startup_i;

        /* 4.2) Rows for parallel tuple cost (use sample-or-scalar rows) */
        double rows_i = GET_ROW(rows_samp, i, path->path.rows, rows_is_const);
        if (rows_i < 0) rows_i = 0;

        /* 4.3) Gather-specific overhead: setup once, per-tuple comm on output */
        startup_cost += parallel_setup_cost;
        run_cost += parallel_tuple_cost * rows_i;

        /* 4.4) Write per-sample outputs and accumulate means */
        const Cost total_cost = startup_cost + run_cost;

        path->path.startup_cost_sample->sample[i] = startup_cost;
        path->path.total_cost_sample->sample[i] = total_cost;

        path->path.startup_cost += startup_cost;
        path->path.total_cost += total_cost;
    }

    /* ------------------------------- 5) Finalize scalar outputs (means) ------------------------------- */
    const double invN = 1.0 / (double) sample_count;
    path->path.startup_cost *= invN;
    path->path.total_cost *= invN;
}

/* ==== ==== ==== ==== ==== ==== GATHER MERGE COST MODEL ==== ==== ==== ==== ==== ==== */

extern void cost_gather_merge_1p(
    GatherMergePath *path,
    PlannerInfo *root,
    const RelOptInfo *rel,
    const ParamPathInfo *param_info,
    const Cost input_startup_cost,
    const Cost input_total_cost,
    const double *rows
) {
    Cost startup_cost = 0;
    Cost run_cost = 0;

    /* Mark the path with the correct row estimate */
    if (rows) {
        path->path.rows = *rows;
    } else if (param_info) {
        path->path.rows = param_info->ppi_rows;
    } else {
        path->path.rows = rel->rows;
    }
    path->path.rows_sample = NULL;

    if (!enable_gathermerge) {
        startup_cost += disable_cost;
    }

    /*
     * Add one to the number of workers to account for the leader.  This might
     * be overgenerous since the leader will do less work than other workers
     * in typical cases, but we'll go with it for now.
     */
    Assert(path->num_workers > 0);
    const double N = (double) path->num_workers + 1;
    const double logN = LOG2(N);

    /* Assumed cost per tuple comparison */
    const Cost comparison_cost = 2.0 * cpu_operator_cost;

    /* Heap creation cost */
    startup_cost += comparison_cost * N * logN;

    /* Per-tuple heap maintenance cost */
    run_cost += path->path.rows * comparison_cost * logN;

    /* small cost for heap management, like cost_merge_append */
    run_cost += cpu_operator_cost * path->path.rows;

    /*
     * Parallel setup and communication cost.  Since Gather Merge, unlike
     * Gather, requires us to block until a tuple is available from every
     * worker, we bump the IPC cost up a little bit as compared with Gather.
     * For lack of a better idea, charge an extra 5%.
     */
    startup_cost += parallel_setup_cost;
    run_cost += parallel_tuple_cost * path->path.rows * 1.05;

    path->path.startup_cost = startup_cost + input_startup_cost;
    path->path.total_cost = (startup_cost + run_cost + input_total_cost);
}

extern void cost_gather_merge_2p(
    GatherMergePath *path,
    PlannerInfo *root,
    const RelOptInfo *rel,
    const ParamPathInfo *param_info,
    const Cost input_startup_cost,
    const Cost input_total_cost,
    const double *rows
) {
    /* ------------------------------- 1) Resolve rows & rows_sample ------------------------------- */
    if (rows) {
        path->path.rows = *rows;
        path->path.rows_sample = make_sample_by_single_value(*rows);
    } else if (param_info) {
        path->path.rows = param_info->ppi_rows;
        path->path.rows_sample = make_sample_by_single_value(param_info->ppi_rows);
    } else {
        path->path.rows = rel->rows;
        path->path.rows_sample = duplicate_sample(rel->rows_sample);
    }

    const Sample *rows_samp = path->path.rows_sample; /* may be NULL */
    const int rows_sc = rows_samp ? rows_samp->sample_count : 0;
    const bool rows_is_const = (rows_sc <= 1);
    const int sample_count = (rows_sc > 1) ? rows_sc : 1;

    path->path.startup_cost = 0.0;
    path->path.startup_cost_sample = initialize_sample(sample_count);
    path->path.total_cost = 0.0;
    path->path.total_cost_sample = initialize_sample(sample_count);

    /* ------------------------------- 2) Precompute GM constants ------------------------------- */
    /* Apply disable penalty once (we'll fold it into each sample's startup). */
    const bool add_disable_penalty = !enable_gathermerge;

    /* +1 worker for leader (same as upstream) */
    Assert(path->num_workers > 0);
    const double N = (double) path->num_workers + 1.0;
    const double logN = LOG2(N);

    /* Cost per tuple comparison (same as upstream) */
    const Cost comparison_cost = 2.0 * cpu_operator_cost;

    /* ------------------------------- 3) Init per-sample outputs ------------------------------- */
    path->path.startup_cost = 0.0;
    path->path.total_cost = 0.0;
    path->path.startup_cost_sample = initialize_sample(sample_count);
    path->path.total_cost_sample = initialize_sample(sample_count);

    /* ------------------------------- 4) Per-sample evaluation ------------------------------- */
    for (int i = 0; i < sample_count; ++i) {
        /* 4.1) Output row count for this sample (fallback to scalar rows) */
        double rows_i = GET_ROW(rows_samp, i, path->path.rows, rows_is_const);
        if (rows_i < 0.0)
            rows_i = 0.0; /* defensive: never negative */

        /* 4.2) Gather Merge startup costs (for this sample) */
        Cost gm_startup_i = 0.0;

        if (add_disable_penalty)
            gm_startup_i += disable_cost;

        /* Heap creation: ~ N * log2(N) tuple comparisons */
        gm_startup_i += comparison_cost * N * logN;

        /* Parallel setup once */
        gm_startup_i += parallel_setup_cost;

        /* 4.3) Gather Merge run costs (for this sample) */
        Cost gm_run_i = 0.0;

        /* Per-tuple heap maintenance: rows_i * log2(N) comparisons */
        gm_run_i += rows_i * comparison_cost * logN;

        /* Small management overhead per output tuple (like MergeAppend) */
        gm_run_i += cpu_operator_cost * rows_i;

        /* IPC per tuple, with 5% bump vs Gather */
        gm_run_i += parallel_tuple_cost * rows_i * 1.05;

        /* 4.4) Combine with input costs (same shape as upstream) */
        const Cost startup_i = gm_startup_i + input_startup_cost;
        const Cost total_i = gm_startup_i + gm_run_i + input_total_cost;

        /* 4.5) Write per-sample results & accumulate means */
        path->path.startup_cost_sample->sample[i] = startup_i;
        path->path.total_cost_sample->sample[i] = total_i;

        path->path.startup_cost += startup_i;
        path->path.total_cost += total_i;
    }

    /* ------------------------------- 5) Finalize scalar outputs (means) ------------------------------- */
    const double invN = 1.0 / (double) sample_count;
    path->path.startup_cost *= invN;
    path->path.total_cost *= invN;
}
