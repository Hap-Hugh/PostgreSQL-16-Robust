/*-------------------------------------------------------------------------
 *
 * costsize.c
 *	  Routines to compute (and set) relation sizes and path costs
 *
 * Path costs are measured in arbitrary units established by these basic
 * parameters:
 *
 *	seq_page_cost		Cost of a sequential page fetch
 *	random_page_cost	Cost of a non-sequential page fetch
 *	cpu_tuple_cost		Cost of typical CPU time to process a tuple
 *	cpu_index_tuple_cost  Cost of typical CPU time to process an index tuple
 *	cpu_operator_cost	Cost of CPU time to execute an operator or function
 *	parallel_tuple_cost Cost of CPU time to pass a tuple from worker to leader backend
 *	parallel_setup_cost Cost of setting up shared memory for parallelism
 *
 * We expect that the kernel will typically do some amount of read-ahead
 * optimization; this in conjunction with seek costs means that seq_page_cost
 * is normally considerably less than random_page_cost.  (However, if the
 * database is fully cached in RAM, it is reasonable to set them equal.)
 *
 * We also use a rough estimate "effective_cache_size" of the number of
 * disk pages in Postgres + OS-level disk cache.  (We can't simply use
 * NBuffers for this purpose because that would ignore the effects of
 * the kernel's disk cache.)
 *
 * Obviously, taking constants for these values is an oversimplification,
 * but it's tough enough to get any useful estimates even at this level of
 * detail.  Note that all of these parameters are user-settable, in case
 * the default values are drastically off for a particular platform.
 *
 * seq_page_cost and random_page_cost can also be overridden for an individual
 * tablespace, in case some data is on a fast disk and other data is on a slow
 * disk.  Per-tablespace overrides never apply to temporary work files such as
 * an external sort or a materialize node that overflows work_mem.
 *
 * We compute two separate costs for each path:
 *		total_cost: total estimated cost to fetch all tuples
 *		startup_cost: cost that is expended before first tuple is fetched
 * In some scenarios, such as when there is a LIMIT or we are implementing
 * an EXISTS(...) sub-select, it is not necessary to fetch all tuples of the
 * path's result.  A caller can estimate the cost of fetching a partial
 * result by interpolating between startup_cost and total_cost.  In detail:
 *		actual_cost = startup_cost +
 *			(total_cost - startup_cost) * tuples_to_fetch / path->rows;
 * Note that a base relation's rows count (and, by extension, plan_rows for
 * plan nodes below the LIMIT node) are set without regard to any LIMIT, so
 * that this equation works properly.  (Note: while path->rows is never zero
 * for ordinary relations, it is zero for paths for provably-empty relations,
 * so beware of division-by-zero.)	The LIMIT is applied as a top-level
 * plan node.
 *
 * For largely historical reasons, most of the routines in this module use
 * the passed result Path only to store their results (rows, startup_cost and
 * total_cost) into.  All the input data they need is passed as separate
 * parameters, even though much of it could be extracted from the Path.
 * An exception is made for the cost_XXXjoin() routines, which expect all
 * the other fields of the passed XXXPath to be filled in, and similarly
 * cost_index() assumes the passed IndexPath is valid except for its output
 * values.
 *
 *
 * Portions Copyright (c) 1996-2023, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 * IDENTIFICATION
 *	  src/backend/optimizer/path/costsize.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include <limits.h>
#include <math.h>

#include "access/amapi.h"
#include "access/htup_details.h"
#include "access/tsmapi.h"
#include "executor/executor.h"
#include "executor/nodeAgg.h"
#include "executor/nodeHash.h"
#include "miscadmin.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/clauses.h"
#include "optimizer/cost.h"
#include "optimizer/cost_ext.h"
#include "optimizer/sample.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/placeholder.h"
#include "optimizer/plancat.h"
#include "parser/parsetree.h"
#include "utils/lsyscache.h"
#include "utils/selfuncs.h"
#include "utils/spccache.h"
#include "utils/tuplesort.h"


#define LOG2(x)  (log(x) / 0.693147180559945)

/*
 * Append and MergeAppend nodes are less expensive than some other operations
 * which use cpu_tuple_cost; instead of adding a separate GUC, estimate the
 * per-tuple cost as cpu_tuple_cost multiplied by this value.
 */
#define APPEND_CPU_COST_MULTIPLIER 0.5

/*
 * Maximum value for row estimates.  We cap row estimates to this to help
 * ensure that costs based on these estimates remain within the range of what
 * double can represent.  add_path() wouldn't act sanely given infinite or NaN
 * cost values.
 */
#define MAXIMUM_ROWCOUNT 1e100

double seq_page_cost = DEFAULT_SEQ_PAGE_COST;
double random_page_cost = DEFAULT_RANDOM_PAGE_COST;
double cpu_tuple_cost = DEFAULT_CPU_TUPLE_COST;
double cpu_index_tuple_cost = DEFAULT_CPU_INDEX_TUPLE_COST;
double cpu_operator_cost = DEFAULT_CPU_OPERATOR_COST;
double parallel_tuple_cost = DEFAULT_PARALLEL_TUPLE_COST;
double parallel_setup_cost = DEFAULT_PARALLEL_SETUP_COST;
double recursive_worktable_factor = DEFAULT_RECURSIVE_WORKTABLE_FACTOR;

int effective_cache_size = DEFAULT_EFFECTIVE_CACHE_SIZE;

Cost disable_cost = 1.0e10;

int max_parallel_workers_per_gather = 2;

bool enable_seqscan = true;
bool enable_indexscan = true;
bool enable_indexonlyscan = true;
bool enable_bitmapscan = true;
bool enable_tidscan = true;
bool enable_sort = true;
bool enable_incremental_sort = true;
bool enable_hashagg = true;
bool enable_nestloop = true;
bool enable_material = true;
bool enable_memoize = true;
bool enable_mergejoin = true;
bool enable_hashjoin = true;
bool enable_gathermerge = true;
bool enable_partitionwise_join = false;
bool enable_partitionwise_aggregate = false;
bool enable_parallel_append = true;
bool enable_parallel_hash = true;
bool enable_partition_pruning = true;
bool enable_presorted_aggregate = true;
bool enable_async_append = true;

typedef struct {
    PlannerInfo *root;
    QualCost total;
} cost_qual_eval_context;

static bool cost_qual_eval_walker(
    Node *node, cost_qual_eval_context *context
);

static double calc_joinrel_size_estimate(
    PlannerInfo *root,
    RelOptInfo *joinrel,
    RelOptInfo *outer_rel,
    RelOptInfo *inner_rel,
    double outer_rows,
    double inner_rows,
    SpecialJoinInfo *sjinfo,
    List *restrictlist
);

static Selectivity get_foreign_key_join_selectivity(
    PlannerInfo *root,
    Relids outer_relids,
    Relids inner_relids,
    SpecialJoinInfo *sjinfo,
    List **restrictlist
);

static Cost append_nonpartial_cost(
    List *subpaths, int numpaths,
    int parallel_workers
);

static void set_rel_width(PlannerInfo *root, RelOptInfo *rel);

/*
 * clamp_row_est
 *		Force a row-count estimate to a sane value.
 */
double clamp_row_est(double nrows) {
    /*
     * Avoid infinite and NaN row estimates.  Costs derived from such values
     * are going to be useless.  Also force the estimate to be at least one
     * row, to make explain output look better and to avoid possible
     * divide-by-zero when interpolating costs.  Make it an integer, too.
     */
    if (nrows > MAXIMUM_ROWCOUNT || isnan(nrows))
        nrows = MAXIMUM_ROWCOUNT;
    else if (nrows <= 1.0)
        nrows = 1.0;
    else
        nrows = rint(nrows);

    return nrows;
}

/*
 * clamp_cardinality_to_long
 *		Cast a Cardinality value to a sane long value.
 */
long clamp_cardinality_to_long(Cardinality x) {
    /*
     * Just for paranoia's sake, ensure we do something sane with negative or
     * NaN values.
     */
    if (isnan(x))
        return LONG_MAX;
    if (x <= 0)
        return 0;

    /*
     * If "long" is 64 bits, then LONG_MAX cannot be represented exactly as a
     * double.  Casting it to double and back may well result in overflow due
     * to rounding, so avoid doing that.  We trust that any double value that
     * compares strictly less than "(double) LONG_MAX" will cast to a
     * representable "long" value.
     */
    return (x < (double) LONG_MAX) ? (long) x : LONG_MAX;
}

/*
 * cost_seqscan
 *   Determine and return the cost of scanning a relation sequentially.
 *
 * This version extends the classic PostgreSQL logic by supporting
 * per-sample distributions for rows and costs while keeping semantics
 * consistent, especially under parallelism (rows represent per-worker rows).
 */
void cost_seqscan(
    Path *path,
    PlannerInfo *root,
    const RelOptInfo *baserel,
    const ParamPathInfo *param_info
) {
    if (!enable_rows_dist || root->pass == 1) {
        cost_seqscan_1p(path, root, baserel, param_info);
    } else if (root->pass == 0 || root->pass == 2) {
        cost_seqscan_2p(path, root, baserel, param_info);
    } else {
        elog(ERROR, "bad pass number");
    }
}

/*
 * cost_samplescan
 *	  Determines and returns the cost of scanning a relation using sampling.
 *
 * 'baserel' is the relation to be scanned
 * 'param_info' is the ParamPathInfo if this is a parameterized path, else NULL
 */
void cost_samplescan(Path *path, PlannerInfo *root,
                     RelOptInfo *baserel, ParamPathInfo *param_info) {
    Cost startup_cost = 0;
    Cost run_cost = 0;
    RangeTblEntry *rte;
    TableSampleClause *tsc;
    TsmRoutine *tsm;
    double spc_seq_page_cost,
            spc_random_page_cost,
            spc_page_cost;
    QualCost qpqual_cost;
    Cost cpu_per_tuple;

    /* Should only be applied to base relations with tablesample clauses */
    Assert(baserel->relid > 0);
    rte = planner_rt_fetch(baserel->relid, root);
    Assert(rte->rtekind == RTE_RELATION);
    tsc = rte->tablesample;
    Assert(tsc != NULL);
    tsm = GetTsmRoutine(tsc->tsmhandler);

    /* Mark the path with the correct row estimate */
    if (param_info)
        path->rows = param_info->ppi_rows;
    else
        path->rows = baserel->rows;

    /* fetch estimated page cost for tablespace containing table */
    get_tablespace_page_costs(baserel->reltablespace,
                              &spc_random_page_cost,
                              &spc_seq_page_cost);

    /* if NextSampleBlock is used, assume random access, else sequential */
    spc_page_cost = (tsm->NextSampleBlock != NULL) ? spc_random_page_cost : spc_seq_page_cost;

    /*
     * disk costs (recall that baserel->pages has already been set to the
     * number of pages the sampling method will visit)
     */
    run_cost += spc_page_cost * baserel->pages;

    /*
     * CPU costs (recall that baserel->tuples has already been set to the
     * number of tuples the sampling method will select).  Note that we ignore
     * execution cost of the TABLESAMPLE parameter expressions; they will be
     * evaluated only once per scan, and in most usages they'll likely be
     * simple constants anyway.  We also don't charge anything for the
     * calculations the sampling method might do internally.
     */
    get_restriction_qual_cost(root, baserel, param_info, &qpqual_cost);

    startup_cost += qpqual_cost.startup;
    cpu_per_tuple = cpu_tuple_cost + qpqual_cost.per_tuple;
    run_cost += cpu_per_tuple * baserel->tuples;
    /* tlist eval costs are paid per output row, not per tuple scanned */
    startup_cost += path->pathtarget->cost.startup;
    run_cost += path->pathtarget->cost.per_tuple * path->rows;

    path->startup_cost = startup_cost;
    path->total_cost = startup_cost + run_cost;
}

/*
 * cost_gather
 *	  Determines and returns the cost of gather path.
 *
 * 'rel' is the relation to be operated upon
 * 'param_info' is the ParamPathInfo if this is a parameterized path, else NULL
 * 'rows' may be used to point to a row estimate; if non-NULL, it overrides
 * both 'rel' and 'param_info'.  This is useful when the path doesn't exactly
 * correspond to any particular RelOptInfo.
 */
void cost_gather(
    GatherPath *path, PlannerInfo *root,
    const RelOptInfo *rel, const ParamPathInfo *param_info,
    const double *rows
) {
    if (!enable_rows_dist || root->pass == 0 || root->pass == 1 || root->pass == 3) {
        cost_gather_1p(path, root, rel, param_info, rows);
    } else if (root->pass == 2) {
        cost_gather_2p(path, root, rel, param_info, rows);
    } else {
        elog(ERROR, "bad pass number");
    }
}

/*
 * cost_gather_merge
 *	  Determines and returns the cost of gather merge path.
 *
 * GatherMerge merges several pre-sorted input streams, using a heap that at
 * any given instant holds the next tuple from each stream. If there are N
 * streams, we need about N*log2(N) tuple comparisons to construct the heap at
 * startup, and then for each output tuple, about log2(N) comparisons to
 * replace the top heap entry with the next tuple from the same stream.
 */
/*
 * cost_gather_merge (samples version)
 *   Determine and return the cost of a Gather Merge path, with per-sample
 *   accounting when row samples are available.
 *
 * Model (same as upstream):
 *   - GatherMerge merges N pre-sorted streams using a heap:
 *       * heap build at startup ~ N * log2(N) tuple comparisons
 *       * per-output tuple maintenance ~ log2(N) comparisons
 *   - Parallel setup once; IPC per output tuple (5% more than Gather)
 *   - Final costs are: (GM_startup + input_startup) and
 *                      (GM_startup + GM_run + input_total)
 *
 * Notes:
 *   - We do not rely on subpath cost samples here because input costs are
 *     explicitly passed in as scalars (input_startup_cost / input_total_cost).
 *   - If rows_sample is present, we compute GM per-sample costs using each
 *     sample row count; otherwise we fall back to a single scalar pass.
 *   - We allocate per-sample outputs (startup_cost_sample / total_cost_sample).
 *
 * Assumes GET_ROW/GET_COST macros are defined like:
 *   #define GET_ROW(s,i,scalar,is_const) \
 *     ( ((s) == NULL || (s)->sample_count <= 0) ? (scalar) \
 *       : ((is_const) ? (s)->sample[0] : (s)->sample[(i)]) )
 */
void cost_gather_merge(
    GatherMergePath *path,
    PlannerInfo *root,
    const RelOptInfo *rel,
    const ParamPathInfo *param_info,
    const Cost input_startup_cost,
    const Cost input_total_cost,
    const double *rows
) {
    if (!enable_rows_dist || root->pass == 0 || root->pass == 1 || root->pass == 3) {
        cost_gather_merge_1p(
            path, root, rel, param_info, input_startup_cost, input_total_cost, rows
        );
    } else if (root->pass == 2) {
        cost_gather_merge_2p(
            path, root, rel, param_info, input_startup_cost, input_total_cost, rows
        );
    } else {
        elog(ERROR, "bad pass number");
    }
}

/*
 * cost_index
 *   Determine and return the cost of scanning a relation using an index.
 *
 * 'path' describes the indexscan under consideration, and is complete
 *		except for the fields to be set by this routine
 * 'loop_count' is the number of repetitions of the indexscan to factor into
 *		estimates of caching behavior
 *
 * In addition to rows, startup_cost and total_cost, cost_index() sets the
 * path's indextotalcost and indexselectivity fields.  These values will be
 * needed if the IndexPath is used in a BitmapIndexScan.
 *
 * NOTE: path->indexquals must contain only clauses usable as index
 * restrictions.  Any additional quals evaluated as qpquals may reduce the
 * number of returned tuples, but they won't reduce the number of tuples
 * we have to fetch from the table, so they don't reduce the scan cost.
 *
 * This version extends the classic PostgreSQL logic by supporting
 * per-sample distributions for rows and costs, while preserving original
 * semantics (esp. under parallelism: rows represent per-worker rows).
 */
void cost_index(
    IndexPath *path,
    PlannerInfo *root,
    const double loop_count,
    const bool partial_path
) {
    if (!enable_rows_dist || root->pass == 1) {
        cost_index_1p(path, root, loop_count, partial_path);
    } else if (root->pass == 0 || root->pass == 2) {
        cost_index_2p(path, root, loop_count, partial_path);
    } else {
        elog(ERROR, "bad pass number");
    }
}

/*
 * index_pages_fetched
 *	  Estimate the number of pages actually fetched after accounting for
 *	  cache effects.
 *
 * We use an approximation proposed by Mackert and Lohman, "Index Scans
 * Using a Finite LRU Buffer: A Validated I/O Model", ACM Transactions
 * on Database Systems, Vol. 14, No. 3, September 1989, Pages 401-424.
 * The Mackert and Lohman approximation is that the number of pages
 * fetched is
 *	PF =
 *		min(2TNs/(2T+Ns), T)			when T <= b
 *		2TNs/(2T+Ns)					when T > b and Ns <= 2Tb/(2T-b)
 *		b + (Ns - 2Tb/(2T-b))*(T-b)/T	when T > b and Ns > 2Tb/(2T-b)
 * where
 *		T = # pages in table
 *		N = # tuples in table
 *		s = selectivity = fraction of table to be scanned
 *		b = # buffer pages available (we include kernel space here)
 *
 * We assume that effective_cache_size is the total number of buffer pages
 * available for the whole query, and pro-rate that space across all the
 * tables in the query and the index currently under consideration.  (This
 * ignores space needed for other indexes used by the query, but since we
 * don't know which indexes will get used, we can't estimate that very well;
 * and in any case counting all the tables may well be an overestimate, since
 * depending on the join plan not all the tables may be scanned concurrently.)
 *
 * The product Ns is the number of tuples fetched; we pass in that
 * product rather than calculating it here.  "pages" is the number of pages
 * in the object under consideration (either an index or a table).
 * "index_pages" is the amount to add to the total table space, which was
 * computed for us by make_one_rel.
 *
 * Caller is expected to have ensured that tuples_fetched is greater than zero
 * and rounded to integer (see clamp_row_est).  The result will likewise be
 * greater than zero and integral.
 */
double index_pages_fetched(
    double tuples_fetched, BlockNumber pages,
    double index_pages, PlannerInfo *root
) {
    double pages_fetched;
    double total_pages;
    double T, b;

    /* T is # pages in table, but don't allow it to be zero */
    T = (pages > 1) ? (double) pages : 1.0;

    /* Compute number of pages assumed to be competing for cache space */
    total_pages = root->total_table_pages + index_pages;
    total_pages = Max(total_pages, 1.0);
    Assert(T <= total_pages);

    /* b is pro-rated share of effective_cache_size */
    b = (double) effective_cache_size * T / total_pages;

    /* force it positive and integral */
    if (b <= 1.0)
        b = 1.0;
    else
        b = ceil(b);

    /* This part is the Mackert and Lohman formula */
    if (T <= b) {
        pages_fetched =
                (2.0 * T * tuples_fetched) / (2.0 * T + tuples_fetched);
        if (pages_fetched >= T)
            pages_fetched = T;
        else
            pages_fetched = ceil(pages_fetched);
    } else {
        double lim;

        lim = (2.0 * T * b) / (2.0 * T - b);
        if (tuples_fetched <= lim) {
            pages_fetched =
                    (2.0 * T * tuples_fetched) / (2.0 * T + tuples_fetched);
        } else {
            pages_fetched =
                    b + (tuples_fetched - lim) * (T - b) / T;
        }
        pages_fetched = ceil(pages_fetched);
    }
    return pages_fetched;
}

/*
 * get_indexpath_pages
 *		Determine the total size of the indexes used in a bitmap index path.
 *
 * Note: if the same index is used more than once in a bitmap tree, we will
 * count it multiple times, which perhaps is the wrong thing ... but it's
 * not completely clear, and detecting duplicates is difficult, so ignore it
 * for now.
 */
static double get_indexpath_pages(Path *bitmapqual) {
    double result = 0;
    ListCell *l;

    if (IsA(bitmapqual, BitmapAndPath)) {
        BitmapAndPath *apath = (BitmapAndPath *) bitmapqual;

        foreach(l, apath->bitmapquals) {
            result += get_indexpath_pages((Path *) lfirst(l));
        }
    } else if (IsA(bitmapqual, BitmapOrPath)) {
        BitmapOrPath *opath = (BitmapOrPath *) bitmapqual;

        foreach(l, opath->bitmapquals) {
            result += get_indexpath_pages((Path *) lfirst(l));
        }
    } else if (IsA(bitmapqual, IndexPath)) {
        IndexPath *ipath = (IndexPath *) bitmapqual;

        result = (double) ipath->indexinfo->pages;
    } else
        elog(ERROR, "unrecognized node type: %d", nodeTag(bitmapqual));

    return result;
}

/*
 * cost_bitmap_heap_scan
 *	  Determines and returns the cost of scanning a relation using a bitmap
 *	  index-then-heap plan.
 *
 * 'baserel' is the relation to be scanned
 * 'param_info' is the ParamPathInfo if this is a parameterized path, else NULL
 * 'bitmapqual' is a tree of IndexPaths, BitmapAndPaths, and BitmapOrPaths
 * 'loop_count' is the number of repetitions of the indexscan to factor into
 *		estimates of caching behavior
 *
 * Note: the component IndexPaths in bitmapqual should have been costed
 * using the same loop_count.
 */
void cost_bitmap_heap_scan(
    Path *path, PlannerInfo *root, RelOptInfo *baserel,
    ParamPathInfo *param_info,
    Path *bitmapqual, double loop_count
) {
    Cost startup_cost = 0;
    Cost run_cost = 0;
    Cost indexTotalCost;
    QualCost qpqual_cost;
    Cost cpu_per_tuple;
    Cost cost_per_page;
    Cost cpu_run_cost;
    double tuples_fetched;
    double pages_fetched;
    double spc_seq_page_cost,
            spc_random_page_cost;
    double T;

    /* Should only be applied to base relations */
    Assert(IsA(baserel, RelOptInfo));
    Assert(baserel->relid > 0);
    Assert(baserel->rtekind == RTE_RELATION);

    /* Mark the path with the correct row estimate */
    if (param_info)
        path->rows = param_info->ppi_rows;
    else
        path->rows = baserel->rows;

    if (!enable_bitmapscan)
        startup_cost += disable_cost;

    pages_fetched = compute_bitmap_pages(root, baserel, bitmapqual,
                                         loop_count, &indexTotalCost,
                                         &tuples_fetched);

    startup_cost += indexTotalCost;
    T = (baserel->pages > 1) ? (double) baserel->pages : 1.0;

    /* Fetch estimated page costs for tablespace containing table. */
    get_tablespace_page_costs(baserel->reltablespace,
                              &spc_random_page_cost,
                              &spc_seq_page_cost);

    /*
     * For small numbers of pages we should charge spc_random_page_cost
     * apiece, while if nearly all the table's pages are being read, it's more
     * appropriate to charge spc_seq_page_cost apiece.  The effect is
     * nonlinear, too. For lack of a better idea, interpolate like this to
     * determine the cost per page.
     */
    if (pages_fetched >= 2.0)
        cost_per_page = spc_random_page_cost -
                        (spc_random_page_cost - spc_seq_page_cost)
                        * sqrt(pages_fetched / T);
    else
        cost_per_page = spc_random_page_cost;

    run_cost += pages_fetched * cost_per_page;

    /*
     * Estimate CPU costs per tuple.
     *
     * Often the indexquals don't need to be rechecked at each tuple ... but
     * not always, especially not if there are enough tuples involved that the
     * bitmaps become lossy.  For the moment, just assume they will be
     * rechecked always.  This means we charge the full freight for all the
     * scan clauses.
     */
    get_restriction_qual_cost(root, baserel, param_info, &qpqual_cost);

    startup_cost += qpqual_cost.startup;
    cpu_per_tuple = cpu_tuple_cost + qpqual_cost.per_tuple;
    cpu_run_cost = cpu_per_tuple * tuples_fetched;

    /* Adjust costing for parallelism, if used. */
    if (path->parallel_workers > 0) {
        double parallel_divisor = get_parallel_divisor(path);

        /* The CPU cost is divided among all the workers. */
        cpu_run_cost /= parallel_divisor;

        path->rows = clamp_row_est(path->rows / parallel_divisor);
    }


    run_cost += cpu_run_cost;

    /* tlist eval costs are paid per output row, not per tuple scanned */
    startup_cost += path->pathtarget->cost.startup;
    run_cost += path->pathtarget->cost.per_tuple * path->rows;

    path->startup_cost = startup_cost;
    path->total_cost = startup_cost + run_cost;

    path->rows_sample = initialize_sample(error_sample_count);
    path->startup_cost_sample = initialize_sample(error_sample_count);
    path->total_cost_sample = initialize_sample(error_sample_count);

    for (int i = 0; i < error_sample_count; ++i) {
        path->rows_sample->sample[i] = path->rows;
        path->startup_cost_sample->sample[i] = path->startup_cost;
        path->total_cost_sample->sample[i] = path->total_cost;
    }
}

/*
 * cost_bitmap_tree_node
 *		Extract cost and selectivity from a bitmap tree node (index/and/or)
 */
void cost_bitmap_tree_node(Path *path, Cost *cost, Selectivity *selec) {
    if (IsA(path, IndexPath)) {
        *cost = ((IndexPath *) path)->indextotalcost;
        *selec = ((IndexPath *) path)->indexselectivity;

        /*
         * Charge a small amount per retrieved tuple to reflect the costs of
         * manipulating the bitmap.  This is mostly to make sure that a bitmap
         * scan doesn't look to be the same cost as an indexscan to retrieve a
         * single tuple.
         */
        *cost += 0.1 * cpu_operator_cost * path->rows;
    } else if (IsA(path, BitmapAndPath)) {
        *cost = path->total_cost;
        *selec = ((BitmapAndPath *) path)->bitmapselectivity;
    } else if (IsA(path, BitmapOrPath)) {
        *cost = path->total_cost;
        *selec = ((BitmapOrPath *) path)->bitmapselectivity;
    } else {
        elog(ERROR, "unrecognized node type: %d", nodeTag(path));
        *cost = *selec = 0; /* keep compiler quiet */
    }
}

/*
 * cost_bitmap_and_node
 *		Estimate the cost of a BitmapAnd node
 *
 * Note that this considers only the costs of index scanning and bitmap
 * creation, not the eventual heap access.  In that sense the object isn't
 * truly a Path, but it has enough path-like properties (costs in particular)
 * to warrant treating it as one.  We don't bother to set the path rows field,
 * however.
 */
void cost_bitmap_and_node(BitmapAndPath *path, PlannerInfo *root) {
    Cost totalCost;
    Selectivity selec;
    ListCell *l;

    /*
     * We estimate AND selectivity on the assumption that the inputs are
     * independent.  This is probably often wrong, but we don't have the info
     * to do better.
     *
     * The runtime cost of the BitmapAnd itself is estimated at 100x
     * cpu_operator_cost for each tbm_intersect needed.  Probably too small,
     * definitely too simplistic?
     */
    totalCost = 0.0;
    selec = 1.0;
    foreach(l, path->bitmapquals) {
        Path *subpath = (Path *) lfirst(l);
        Cost subCost;
        Selectivity subselec;

        cost_bitmap_tree_node(subpath, &subCost, &subselec);

        selec *= subselec;

        totalCost += subCost;
        if (l != list_head(path->bitmapquals))
            totalCost += 100.0 * cpu_operator_cost;
    }
    path->bitmapselectivity = selec;
    path->path.rows = 0; /* per above, not used */
    path->path.startup_cost = totalCost;
    path->path.total_cost = totalCost;

    path->path.rows_sample = initialize_sample(error_sample_count);
    path->path.startup_cost_sample = initialize_sample(error_sample_count);
    path->path.total_cost_sample = initialize_sample(error_sample_count);

    for (int i = 0; i < error_sample_count; ++i) {
        path->path.rows_sample->sample[i] = path->path.rows;
        path->path.startup_cost_sample->sample[i] = path->path.startup_cost;
        path->path.total_cost_sample->sample[i] = path->path.total_cost;
    }
}

/*
 * cost_bitmap_or_node
 *		Estimate the cost of a BitmapOr node
 *
 * See comments for cost_bitmap_and_node.
 */
void cost_bitmap_or_node(BitmapOrPath *path, PlannerInfo *root) {
    Cost totalCost;
    Selectivity selec;
    ListCell *l;

    /*
     * We estimate OR selectivity on the assumption that the inputs are
     * non-overlapping, since that's often the case in "x IN (list)" type
     * situations.  Of course, we clamp to 1.0 at the end.
     *
     * The runtime cost of the BitmapOr itself is estimated at 100x
     * cpu_operator_cost for each tbm_union needed.  Probably too small,
     * definitely too simplistic?  We are aware that the tbm_unions are
     * optimized out when the inputs are BitmapIndexScans.
     */
    totalCost = 0.0;
    selec = 0.0;
    foreach(l, path->bitmapquals) {
        Path *subpath = (Path *) lfirst(l);
        Cost subCost;
        Selectivity subselec;

        cost_bitmap_tree_node(subpath, &subCost, &subselec);

        selec += subselec;

        totalCost += subCost;
        if (l != list_head(path->bitmapquals) &&
            !IsA(subpath, IndexPath))
            totalCost += 100.0 * cpu_operator_cost;
    }
    path->bitmapselectivity = Min(selec, 1.0);
    path->path.rows = 0; /* per above, not used */
    path->path.startup_cost = totalCost;
    path->path.total_cost = totalCost;

    path->path.rows_sample = initialize_sample(error_sample_count);
    path->path.startup_cost_sample = initialize_sample(error_sample_count);
    path->path.total_cost_sample = initialize_sample(error_sample_count);

    for (int i = 0; i < error_sample_count; ++i) {
        path->path.rows_sample->sample[i] = path->path.rows;
        path->path.startup_cost_sample->sample[i] = path->path.startup_cost;
        path->path.total_cost_sample->sample[i] = path->path.total_cost;
    }
}

/*
 * cost_tidscan
 *	  Determines and returns the cost of scanning a relation using TIDs.
 *
 * 'baserel' is the relation to be scanned
 * 'tidquals' is the list of TID-checkable quals
 * 'param_info' is the ParamPathInfo if this is a parameterized path, else NULL
 */
void cost_tidscan(Path *path, PlannerInfo *root,
                  RelOptInfo *baserel, List *tidquals, ParamPathInfo *param_info) {
    Cost startup_cost = 0;
    Cost run_cost = 0;
    bool isCurrentOf = false;
    QualCost qpqual_cost;
    Cost cpu_per_tuple;
    QualCost tid_qual_cost;
    int ntuples;
    ListCell *l;
    double spc_random_page_cost;

    /* Should only be applied to base relations */
    Assert(baserel->relid > 0);
    Assert(baserel->rtekind == RTE_RELATION);

    /* Mark the path with the correct row estimate */
    if (param_info)
        path->rows = param_info->ppi_rows;
    else
        path->rows = baserel->rows;

    /* Count how many tuples we expect to retrieve */
    ntuples = 0;
    foreach(l, tidquals) {
        RestrictInfo *rinfo = lfirst_node(RestrictInfo, l);
        Expr *qual = rinfo->clause;

        if (IsA(qual, ScalarArrayOpExpr)) {
            /* Each element of the array yields 1 tuple */
            ScalarArrayOpExpr *saop = (ScalarArrayOpExpr *) qual;
            Node *arraynode = (Node *) lsecond(saop->args);

            ntuples += estimate_array_length(arraynode);
        } else if (IsA(qual, CurrentOfExpr)) {
            /* CURRENT OF yields 1 tuple */
            isCurrentOf = true;
            ntuples++;
        } else {
            /* It's just CTID = something, count 1 tuple */
            ntuples++;
        }
    }

    /*
     * We must force TID scan for WHERE CURRENT OF, because only nodeTidscan.c
     * understands how to do it correctly.  Therefore, honor enable_tidscan
     * only when CURRENT OF isn't present.  Also note that cost_qual_eval
     * counts a CurrentOfExpr as having startup cost disable_cost, which we
     * subtract off here; that's to prevent other plan types such as seqscan
     * from winning.
     */
    if (isCurrentOf) {
        Assert(baserel->baserestrictcost.startup >= disable_cost);
        startup_cost -= disable_cost;
    } else if (!enable_tidscan)
        startup_cost += disable_cost;

    /*
     * The TID qual expressions will be computed once, any other baserestrict
     * quals once per retrieved tuple.
     */
    cost_qual_eval(&tid_qual_cost, tidquals, root);

    /* fetch estimated page cost for tablespace containing table */
    get_tablespace_page_costs(baserel->reltablespace,
                              &spc_random_page_cost,
                              NULL);

    /* disk costs --- assume each tuple on a different page */
    run_cost += spc_random_page_cost * ntuples;

    /* Add scanning CPU costs */
    get_restriction_qual_cost(root, baserel, param_info, &qpqual_cost);

    /* XXX currently we assume TID quals are a subset of qpquals */
    startup_cost += qpqual_cost.startup + tid_qual_cost.per_tuple;
    cpu_per_tuple = cpu_tuple_cost + qpqual_cost.per_tuple -
                    tid_qual_cost.per_tuple;
    run_cost += cpu_per_tuple * ntuples;

    /* tlist eval costs are paid per output row, not per tuple scanned */
    startup_cost += path->pathtarget->cost.startup;
    run_cost += path->pathtarget->cost.per_tuple * path->rows;

    path->startup_cost = startup_cost;
    path->total_cost = startup_cost + run_cost;
}

/*
 * cost_tidrangescan
 *	  Determines and sets the costs of scanning a relation using a range of
 *	  TIDs for 'path'
 *
 * 'baserel' is the relation to be scanned
 * 'tidrangequals' is the list of TID-checkable range quals
 * 'param_info' is the ParamPathInfo if this is a parameterized path, else NULL
 */
void cost_tidrangescan(Path *path, PlannerInfo *root,
                       RelOptInfo *baserel, List *tidrangequals,
                       ParamPathInfo *param_info) {
    Selectivity selectivity;
    double pages;
    Cost startup_cost = 0;
    Cost run_cost = 0;
    QualCost qpqual_cost;
    Cost cpu_per_tuple;
    QualCost tid_qual_cost;
    double ntuples;
    double nseqpages;
    double spc_random_page_cost;
    double spc_seq_page_cost;

    /* Should only be applied to base relations */
    Assert(baserel->relid > 0);
    Assert(baserel->rtekind == RTE_RELATION);

    /* Mark the path with the correct row estimate */
    if (param_info)
        path->rows = param_info->ppi_rows;
    else
        path->rows = baserel->rows;

    /* Count how many tuples and pages we expect to scan */
    selectivity = clauselist_selectivity(root, tidrangequals, baserel->relid,
                                         JOIN_INNER, NULL);
    pages = ceil(selectivity * baserel->pages);

    if (pages <= 0.0)
        pages = 1.0;

    /*
     * The first page in a range requires a random seek, but each subsequent
     * page is just a normal sequential page read. NOTE: it's desirable for
     * TID Range Scans to cost more than the equivalent Sequential Scans,
     * because Seq Scans have some performance advantages such as scan
     * synchronization and parallelizability, and we'd prefer one of them to
     * be picked unless a TID Range Scan really is better.
     */
    ntuples = selectivity * baserel->tuples;
    nseqpages = pages - 1.0;

    if (!enable_tidscan)
        startup_cost += disable_cost;

    /*
     * The TID qual expressions will be computed once, any other baserestrict
     * quals once per retrieved tuple.
     */
    cost_qual_eval(&tid_qual_cost, tidrangequals, root);

    /* fetch estimated page cost for tablespace containing table */
    get_tablespace_page_costs(baserel->reltablespace,
                              &spc_random_page_cost,
                              &spc_seq_page_cost);

    /* disk costs; 1 random page and the remainder as seq pages */
    run_cost += spc_random_page_cost + spc_seq_page_cost * nseqpages;

    /* Add scanning CPU costs */
    get_restriction_qual_cost(root, baserel, param_info, &qpqual_cost);

    /*
     * XXX currently we assume TID quals are a subset of qpquals at this
     * point; they will be removed (if possible) when we create the plan, so
     * we subtract their cost from the total qpqual cost.  (If the TID quals
     * can't be removed, this is a mistake and we're going to underestimate
     * the CPU cost a bit.)
     */
    startup_cost += qpqual_cost.startup + tid_qual_cost.per_tuple;
    cpu_per_tuple = cpu_tuple_cost + qpqual_cost.per_tuple -
                    tid_qual_cost.per_tuple;
    run_cost += cpu_per_tuple * ntuples;

    /* tlist eval costs are paid per output row, not per tuple scanned */
    startup_cost += path->pathtarget->cost.startup;
    run_cost += path->pathtarget->cost.per_tuple * path->rows;

    path->startup_cost = startup_cost;
    path->total_cost = startup_cost + run_cost;
}

/*
 * cost_subqueryscan
 *	  Determines and returns the cost of scanning a subquery RTE.
 *
 * 'baserel' is the relation to be scanned
 * 'param_info' is the ParamPathInfo if this is a parameterized path, else NULL
 * 'trivial_pathtarget' is true if the pathtarget is believed to be trivial.
 */
void cost_subqueryscan(
    SubqueryScanPath *path, PlannerInfo *root,
    const RelOptInfo *baserel, const ParamPathInfo *param_info,
    const bool trivial_pathtarget
) {
    if (!enable_rows_dist || root->pass == 1) {
        cost_subqueryscan_1p(path, root, baserel, param_info, trivial_pathtarget);
    } else if (root->pass == 0 || root->pass == 2) {
        cost_subqueryscan_2p(path, root, baserel, param_info, trivial_pathtarget);
    } else {
        elog(ERROR, "bad pass number");
    }
}

/*
 * cost_functionscan
 *	  Determines and returns the cost of scanning a function RTE.
 *
 * 'baserel' is the relation to be scanned
 * 'param_info' is the ParamPathInfo if this is a parameterized path, else NULL
 */
void cost_functionscan(
    Path *path, PlannerInfo *root,
    RelOptInfo *baserel, ParamPathInfo *param_info
) {
    Cost startup_cost = 0;
    Cost run_cost = 0;
    QualCost qpqual_cost;
    Cost cpu_per_tuple;
    RangeTblEntry *rte;
    QualCost exprcost;

    /* Should only be applied to base relations that are functions */
    Assert(baserel->relid > 0);
    rte = planner_rt_fetch(baserel->relid, root);
    Assert(rte->rtekind == RTE_FUNCTION);

    /* Mark the path with the correct row estimate */
    if (param_info)
        path->rows = param_info->ppi_rows;
    else
        path->rows = baserel->rows;

    /*
     * Estimate costs of executing the function expression(s).
     *
     * Currently, nodeFunctionscan.c always executes the functions to
     * completion before returning any rows, and caches the results in a
     * tuplestore.  So the function eval cost is all startup cost, and per-row
     * costs are minimal.
     *
     * XXX in principle we ought to charge tuplestore spill costs if the
     * number of rows is large.  However, given how phony our rowcount
     * estimates for functions tend to be, there's not a lot of point in that
     * refinement right now.
     */
    cost_qual_eval_node(&exprcost, (Node *) rte->functions, root);

    startup_cost += exprcost.startup + exprcost.per_tuple;

    /* Add scanning CPU costs */
    get_restriction_qual_cost(root, baserel, param_info, &qpqual_cost);

    startup_cost += qpqual_cost.startup;
    cpu_per_tuple = cpu_tuple_cost + qpqual_cost.per_tuple;
    run_cost += cpu_per_tuple * baserel->tuples;

    /* tlist eval costs are paid per output row, not per tuple scanned */
    startup_cost += path->pathtarget->cost.startup;
    run_cost += path->pathtarget->cost.per_tuple * path->rows;

    path->startup_cost = startup_cost;
    path->total_cost = startup_cost + run_cost;
}

/*
 * cost_tablefuncscan
 *	  Determines and returns the cost of scanning a table function.
 *
 * 'baserel' is the relation to be scanned
 * 'param_info' is the ParamPathInfo if this is a parameterized path, else NULL
 */
void cost_tablefuncscan(
    Path *path, PlannerInfo *root,
    RelOptInfo *baserel, ParamPathInfo *param_info
) {
    Cost startup_cost = 0;
    Cost run_cost = 0;
    QualCost qpqual_cost;
    Cost cpu_per_tuple;
    RangeTblEntry *rte;
    QualCost exprcost;

    /* Should only be applied to base relations that are functions */
    Assert(baserel->relid > 0);
    rte = planner_rt_fetch(baserel->relid, root);
    Assert(rte->rtekind == RTE_TABLEFUNC);

    /* Mark the path with the correct row estimate */
    if (param_info)
        path->rows = param_info->ppi_rows;
    else
        path->rows = baserel->rows;

    /*
     * Estimate costs of executing the table func expression(s).
     *
     * XXX in principle we ought to charge tuplestore spill costs if the
     * number of rows is large.  However, given how phony our rowcount
     * estimates for tablefuncs tend to be, there's not a lot of point in that
     * refinement right now.
     */
    cost_qual_eval_node(&exprcost, (Node *) rte->tablefunc, root);

    startup_cost += exprcost.startup + exprcost.per_tuple;

    /* Add scanning CPU costs */
    get_restriction_qual_cost(root, baserel, param_info, &qpqual_cost);

    startup_cost += qpqual_cost.startup;
    cpu_per_tuple = cpu_tuple_cost + qpqual_cost.per_tuple;
    run_cost += cpu_per_tuple * baserel->tuples;

    /* tlist eval costs are paid per output row, not per tuple scanned */
    startup_cost += path->pathtarget->cost.startup;
    run_cost += path->pathtarget->cost.per_tuple * path->rows;

    path->startup_cost = startup_cost;
    path->total_cost = startup_cost + run_cost;
}

/*
 * cost_valuesscan
 *	  Determines and returns the cost of scanning a VALUES RTE.
 *
 * 'baserel' is the relation to be scanned
 * 'param_info' is the ParamPathInfo if this is a parameterized path, else NULL
 */
void cost_valuesscan(
    Path *path, PlannerInfo *root,
    RelOptInfo *baserel, ParamPathInfo *param_info
) {
    Cost startup_cost = 0;
    Cost run_cost = 0;
    QualCost qpqual_cost;
    Cost cpu_per_tuple;

    /* Should only be applied to base relations that are values lists */
    Assert(baserel->relid > 0);
    Assert(baserel->rtekind == RTE_VALUES);

    /* Mark the path with the correct row estimate */
    if (param_info)
        path->rows = param_info->ppi_rows;
    else
        path->rows = baserel->rows;

    /*
     * For now, estimate list evaluation cost at one operator eval per list
     * (probably pretty bogus, but is it worth being smarter?)
     */
    cpu_per_tuple = cpu_operator_cost;

    /* Add scanning CPU costs */
    get_restriction_qual_cost(root, baserel, param_info, &qpqual_cost);

    startup_cost += qpqual_cost.startup;
    cpu_per_tuple += cpu_tuple_cost + qpqual_cost.per_tuple;
    run_cost += cpu_per_tuple * baserel->tuples;

    /* tlist eval costs are paid per output row, not per tuple scanned */
    startup_cost += path->pathtarget->cost.startup;
    run_cost += path->pathtarget->cost.per_tuple * path->rows;

    path->startup_cost = startup_cost;
    path->total_cost = startup_cost + run_cost;
}

/*
 * cost_ctescan
 *	  Determines and returns the cost of scanning a CTE RTE.
 *
 * Note: this is used for both self-reference and regular CTEs; the
 * possible cost differences are below the threshold of what we could
 * estimate accurately anyway.  Note that the costs of evaluating the
 * referenced CTE query are added into the final plan as initplan costs,
 * and should NOT be counted here.
 */
void cost_ctescan(
    Path *path, PlannerInfo *root,
    const RelOptInfo *baserel, const ParamPathInfo *param_info
) {
    if (!enable_rows_dist || root->pass == 1) {
        cost_ctescan_1p(path, root, baserel, param_info);
    } else if (root->pass == 0 || root->pass == 2) {
        cost_ctescan_2p(path, root, baserel, param_info);
    } else {
        elog(ERROR, "bad pass number");
    }
}

/*
 * cost_namedtuplestorescan
 *	  Determines and returns the cost of scanning a named tuplestore.
 */
void cost_namedtuplestorescan(
    Path *path, PlannerInfo *root,
    RelOptInfo *baserel, ParamPathInfo *param_info
) {
    Cost startup_cost = 0;
    Cost run_cost = 0;
    QualCost qpqual_cost;
    Cost cpu_per_tuple;

    /* Should only be applied to base relations that are Tuplestores */
    Assert(baserel->relid > 0);
    Assert(baserel->rtekind == RTE_NAMEDTUPLESTORE);

    /* Mark the path with the correct row estimate */
    if (param_info)
        path->rows = param_info->ppi_rows;
    else
        path->rows = baserel->rows;

    /* Charge one CPU tuple cost per row for tuplestore manipulation */
    cpu_per_tuple = cpu_tuple_cost;

    /* Add scanning CPU costs */
    get_restriction_qual_cost(root, baserel, param_info, &qpqual_cost);

    startup_cost += qpqual_cost.startup;
    cpu_per_tuple += cpu_tuple_cost + qpqual_cost.per_tuple;
    run_cost += cpu_per_tuple * baserel->tuples;

    path->startup_cost = startup_cost;
    path->total_cost = startup_cost + run_cost;
}

/*
 * cost_resultscan
 *	  Determines and returns the cost of scanning an RTE_RESULT relation.
 */
void cost_resultscan(
    Path *path, PlannerInfo *root,
    RelOptInfo *baserel, ParamPathInfo *param_info
) {
    Cost startup_cost = 0;
    Cost run_cost = 0;
    QualCost qpqual_cost;
    Cost cpu_per_tuple;

    /* Should only be applied to RTE_RESULT base relations */
    Assert(baserel->relid > 0);
    Assert(baserel->rtekind == RTE_RESULT);

    /* Mark the path with the correct row estimate */
    if (param_info) {
        path->rows = param_info->ppi_rows;
        path->rows_sample = duplicate_sample(param_info->ppi_rows_sample);
    } else {
        path->rows = baserel->rows;
        path->rows_sample = duplicate_sample(baserel->rows_sample);
    }

    /* We charge qual cost plus cpu_tuple_cost */
    get_restriction_qual_cost(root, baserel, param_info, &qpqual_cost);

    startup_cost += qpqual_cost.startup;
    cpu_per_tuple = cpu_tuple_cost + qpqual_cost.per_tuple;
    run_cost += cpu_per_tuple * baserel->tuples;

    path->startup_cost = startup_cost;
    path->total_cost = startup_cost + run_cost;

    // TODO: Do we need cost sampling for result scan?
}

/*
 * cost_recursive_union
 *	  Determines and returns the cost of performing a recursive union,
 *	  and also the estimated output size.
 *
 * We are given Paths for the nonrecursive and recursive terms.
 */
void cost_recursive_union(Path *runion, Path *nrterm, Path *rterm) {
    Cost startup_cost;
    Cost total_cost;
    double total_rows;

    /* We probably have decent estimates for the non-recursive term */
    startup_cost = nrterm->startup_cost;
    total_cost = nrterm->total_cost;
    total_rows = nrterm->rows;

    /*
     * We arbitrarily assume that about 10 recursive iterations will be
     * needed, and that we've managed to get a good fix on the cost and output
     * size of each one of them.  These are mighty shaky assumptions but it's
     * hard to see how to do better.
     */
    total_cost += 10 * rterm->total_cost;
    total_rows += 10 * rterm->rows;

    /*
     * Also charge cpu_tuple_cost per row to account for the costs of
     * manipulating the tuplestores.  (We don't worry about possible
     * spill-to-disk costs.)
     */
    total_cost += cpu_tuple_cost * total_rows;

    runion->startup_cost = startup_cost;
    runion->total_cost = total_cost;
    runion->rows = total_rows;
    runion->pathtarget->width = Max(nrterm->pathtarget->width,
                                    rterm->pathtarget->width);
}

/*
 * cost_tuplesort
 *	  Determines and returns the cost of sorting a relation using tuplesort,
 *    not including the cost of reading the input data.
 *
 * If the total volume of data to sort is less than sort_mem, we will do
 * an in-memory sort, which requires no I/O and about t*log2(t) tuple
 * comparisons for t tuples.
 *
 * If the total volume exceeds sort_mem, we switch to a tape-style merge
 * algorithm.  There will still be about t*log2(t) tuple comparisons in
 * total, but we will also need to write and read each tuple once per
 * merge pass.  We expect about ceil(logM(r)) merge passes where r is the
 * number of initial runs formed and M is the merge order used by tuplesort.c.
 * Since the average initial run should be about sort_mem, we have
 *		disk traffic = 2 * relsize * ceil(logM(p / sort_mem))
 *		cpu = comparison_cost * t * log2(t)
 *
 * If the sort is bounded (i.e., only the first k result tuples are needed)
 * and k tuples can fit into sort_mem, we use a heap method that keeps only
 * k tuples in the heap; this will require about t*log2(k) tuple comparisons.
 *
 * The disk traffic is assumed to be 3/4ths sequential and 1/4th random
 * accesses (XXX can't we refine that guess?)
 *
 * By default, we charge two operator evals per tuple comparison, which should
 * be in the right ballpark in most cases.  The caller can tweak this by
 * specifying nonzero comparison_cost; typically that's used for any extra
 * work that has to be done to prepare the inputs to the comparison operators.
 *
 * 'tuples' is the number of tuples in the relation
 * 'width' is the average tuple width in bytes
 * 'comparison_cost' is the extra cost per comparison, if any
 * 'sort_mem' is the number of kilobytes of work memory allowed for the sort
 * 'limit_tuples' is the bound on the number of output tuples; -1 if no bound
 */
static void cost_tuplesort(
    Cost *startup_cost, Cost *run_cost,
    double tuples, int width,
    Cost comparison_cost, int sort_mem,
    double limit_tuples
) {
    double input_bytes = relation_byte_size(tuples, width);
    double output_bytes;
    double output_tuples;
    long sort_mem_bytes = sort_mem * 1024L;

    /*
     * We want to be sure the cost of a sort is never estimated as zero, even
     * if passed-in tuple count is zero.  Besides, mustn't do log(0)...
     */
    if (tuples < 2.0)
        tuples = 2.0;

    /* Include the default cost-per-comparison */
    comparison_cost += 2.0 * cpu_operator_cost;

    /* Do we have a useful LIMIT? */
    if (limit_tuples > 0 && limit_tuples < tuples) {
        output_tuples = limit_tuples;
        output_bytes = relation_byte_size(output_tuples, width);
    } else {
        output_tuples = tuples;
        output_bytes = input_bytes;
    }

    if (output_bytes > sort_mem_bytes) {
        /*
         * We'll have to use a disk-based sort of all the tuples
         */
        double npages = ceil(input_bytes / BLCKSZ);
        double nruns = input_bytes / sort_mem_bytes;
        double mergeorder = tuplesort_merge_order(sort_mem_bytes);
        double log_runs;
        double npageaccesses;

        /*
         * CPU costs
         *
         * Assume about N log2 N comparisons
         */
        *startup_cost = comparison_cost * tuples * LOG2(tuples);

        /* Disk costs */

        /* Compute logM(r) as log(r) / log(M) */
        if (nruns > mergeorder)
            log_runs = ceil(log(nruns) / log(mergeorder));
        else
            log_runs = 1.0;
        npageaccesses = 2.0 * npages * log_runs;
        /* Assume 3/4ths of accesses are sequential, 1/4th are not */
        *startup_cost += npageaccesses *
                (seq_page_cost * 0.75 + random_page_cost * 0.25);
    } else if (tuples > 2 * output_tuples || input_bytes > sort_mem_bytes) {
        /*
         * We'll use a bounded heap-sort keeping just K tuples in memory, for
         * a total number of tuple comparisons of N log2 K; but the constant
         * factor is a bit higher than for quicksort.  Tweak it so that the
         * cost curve is continuous at the crossover point.
         */
        *startup_cost = comparison_cost * tuples * LOG2(2.0 * output_tuples);
    } else {
        /* We'll use plain quicksort on all the input tuples */
        *startup_cost = comparison_cost * tuples * LOG2(tuples);
    }

    /*
     * Also charge a small amount (arbitrarily set equal to operator cost) per
     * extracted tuple.  We don't charge cpu_tuple_cost because a Sort node
     * doesn't do qual-checking or projection, so it has less overhead than
     * most plan nodes.  Note it's correct to use tuples not output_tuples
     * here --- the upper LIMIT will pro-rate the run cost so we'd be double
     * counting the LIMIT otherwise.
     */
    *run_cost = cpu_operator_cost * tuples;
}

/*
 * cost_incremental_sort
 * 	Determines and returns the cost of sorting a relation incrementally, when
 *  the input path is presorted by a prefix of the pathkeys.
 *
 * 'presorted_keys' is the number of leading pathkeys by which the input path
 * is sorted.
 *
 * We estimate the number of groups into which the relation is divided by the
 * leading pathkeys, and then calculate the cost of sorting a single group
 * with tuplesort using cost_tuplesort().
 */
void cost_incremental_sort(
    Path *path,
    PlannerInfo *root, List *pathkeys, int presorted_keys,
    Cost input_startup_cost, Cost input_total_cost,
    double input_tuples, int width, Cost comparison_cost, int sort_mem,
    double limit_tuples
) {
    Cost startup_cost,
            run_cost,
            input_run_cost = input_total_cost - input_startup_cost;
    double group_tuples,
            input_groups;
    Cost group_startup_cost,
            group_run_cost,
            group_input_run_cost;
    List *presortedExprs = NIL;
    ListCell *l;
    bool unknown_varno = false;

    Assert(presorted_keys > 0 && presorted_keys < list_length(pathkeys));

    /*
     * We want to be sure the cost of a sort is never estimated as zero, even
     * if passed-in tuple count is zero.  Besides, mustn't do log(0)...
     */
    if (input_tuples < 2.0)
        input_tuples = 2.0;

    /* Default estimate of number of groups, capped to one group per row. */
    input_groups = Min(input_tuples, DEFAULT_NUM_DISTINCT);

    /*
     * Extract presorted keys as list of expressions.
     *
     * We need to be careful about Vars containing "varno 0" which might have
     * been introduced by generate_append_tlist, which would confuse
     * estimate_num_groups (in fact it'd fail for such expressions). See
     * recurse_set_operations which has to deal with the same issue.
     *
     * Unlike recurse_set_operations we can't access the original target list
     * here, and even if we could it's not very clear how useful would that be
     * for a set operation combining multiple tables. So we simply detect if
     * there are any expressions with "varno 0" and use the default
     * DEFAULT_NUM_DISTINCT in that case.
     *
     * We might also use either 1.0 (a single group) or input_tuples (each row
     * being a separate group), pretty much the worst and best case for
     * incremental sort. But those are extreme cases and using something in
     * between seems reasonable. Furthermore, generate_append_tlist is used
     * for set operations, which are likely to produce mostly unique output
     * anyway - from that standpoint the DEFAULT_NUM_DISTINCT is defensive
     * while maintaining lower startup cost.
     */
    foreach(l, pathkeys) {
        PathKey *key = (PathKey *) lfirst(l);
        EquivalenceMember *member = (EquivalenceMember *)
                linitial(key->pk_eclass->ec_members);

        /*
         * Check if the expression contains Var with "varno 0" so that we
         * don't call estimate_num_groups in that case.
         */
        if (bms_is_member(0, pull_varnos(root, (Node *) member->em_expr))) {
            unknown_varno = true;
            break;
        }

        /* expression not containing any Vars with "varno 0" */
        presortedExprs = lappend(presortedExprs, member->em_expr);

        if (foreach_current_index(l) + 1 >= presorted_keys)
            break;
    }

    /* Estimate the number of groups with equal presorted keys. */
    if (!unknown_varno)
        input_groups = estimate_num_groups(root, presortedExprs, input_tuples,
                                           NULL, NULL);

    group_tuples = input_tuples / input_groups;
    group_input_run_cost = input_run_cost / input_groups;

    /*
     * Estimate the average cost of sorting of one group where presorted keys
     * are equal.
     */
    cost_tuplesort(&group_startup_cost, &group_run_cost,
                   group_tuples, width, comparison_cost, sort_mem,
                   limit_tuples);

    /*
     * Startup cost of incremental sort is the startup cost of its first group
     * plus the cost of its input.
     */
    startup_cost = group_startup_cost + input_startup_cost +
                   group_input_run_cost;

    /*
     * After we started producing tuples from the first group, the cost of
     * producing all the tuples is given by the cost to finish processing this
     * group, plus the total cost to process the remaining groups, plus the
     * remaining cost of input.
     */
    run_cost = group_run_cost + (group_run_cost + group_startup_cost) *
               (input_groups - 1) + group_input_run_cost * (input_groups - 1);

    /*
     * Incremental sort adds some overhead by itself. Firstly, it has to
     * detect the sort groups. This is roughly equal to one extra copy and
     * comparison per tuple.
     */
    run_cost += (cpu_tuple_cost + comparison_cost) * input_tuples;

    /*
     * Additionally, we charge double cpu_tuple_cost for each input group to
     * account for the tuplesort_reset that's performed after each group.
     */
    run_cost += 2.0 * cpu_tuple_cost * input_groups;

    path->rows = input_tuples;
    path->startup_cost = startup_cost;
    path->total_cost = startup_cost + run_cost;

    // FIXME
    path->rows_sample = initialize_sample(error_sample_count);
    for (int i = 0; i < error_sample_count; ++i) {
        path->rows_sample->sample[i] = path->rows;
    }
    path->startup_cost_sample = initialize_sample(error_sample_count);
    for (int i = 0; i < error_sample_count; ++i) {
        path->startup_cost_sample->sample[i] = path->startup_cost;
    }
    path->total_cost_sample = initialize_sample(error_sample_count);
    for (int i = 0; i < error_sample_count; ++i) {
        path->total_cost_sample->sample[i] = path->total_cost;
    }
}

/*
 * cost_sort
 *	  Determines and returns the cost of sorting a relation, including
 *	  the cost of reading the input data.
 *
 * NOTE: some callers currently pass NIL for pathkeys because they
 * can't conveniently supply the sort keys.  Since this routine doesn't
 * currently do anything with pathkeys anyway, that doesn't matter...
 * but if it ever does, it should react gracefully to lack of key data.
 * (Actually, the thing we'd most likely be interested in is just the number
 * of sort keys, which all callers *could* supply.)
 */
void cost_sort(
    Path *path, PlannerInfo *root,
    List *pathkeys, Cost input_cost, double tuples, int width,
    Cost comparison_cost, int sort_mem,
    double limit_tuples
) {
    Cost startup_cost;
    Cost run_cost;

    cost_tuplesort(&startup_cost, &run_cost,
                   tuples, width,
                   comparison_cost, sort_mem,
                   limit_tuples);

    if (!enable_sort)
        startup_cost += disable_cost;

    startup_cost += input_cost;

    path->rows = tuples;
    path->startup_cost = startup_cost;
    path->total_cost = startup_cost + run_cost;

    // FIXME: Do we need to make `cost_tuplesort` also sample-based?

    // FIXME
    path->rows_sample = initialize_sample(error_sample_count);
    for (int i = 0; i < error_sample_count; ++i) {
        path->rows_sample->sample[i] = path->rows;
    }
    path->startup_cost_sample = initialize_sample(error_sample_count);
    for (int i = 0; i < error_sample_count; ++i) {
        path->startup_cost_sample->sample[i] = path->startup_cost;
    }
    path->total_cost_sample = initialize_sample(error_sample_count);
    for (int i = 0; i < error_sample_count; ++i) {
        path->total_cost_sample->sample[i] = path->total_cost;
    }
}

/*
 * append_nonpartial_cost
 *	  Estimate the cost of the non-partial paths in a Parallel Append.
 *	  The non-partial paths are assumed to be the first "numpaths" paths
 *	  from the subpaths list, and to be in order of decreasing cost.
 */
static Cost append_nonpartial_cost(
    List *subpaths, int numpaths, int parallel_workers
) {
    Cost *costarr;
    int arrlen;
    ListCell *l;
    ListCell *cell;
    int path_index;
    int min_index;
    int max_index;

    if (numpaths == 0)
        return 0;

    /*
     * Array length is number of workers or number of relevant paths,
     * whichever is less.
     */
    arrlen = Min(parallel_workers, numpaths);
    costarr = (Cost *) palloc(sizeof(Cost) * arrlen);

    /* The first few paths will each be claimed by a different worker. */
    path_index = 0;
    foreach(cell, subpaths) {
        Path *subpath = (Path *) lfirst(cell);

        if (path_index == arrlen)
            break;
        costarr[path_index++] = subpath->total_cost;
    }

    /*
     * Since subpaths are sorted by decreasing cost, the last one will have
     * the minimum cost.
     */
    min_index = arrlen - 1;

    /*
     * For each of the remaining subpaths, add its cost to the array element
     * with minimum cost.
     */
    for_each_cell(l, subpaths, cell) {
        Path *subpath = (Path *) lfirst(l);

        /* Consider only the non-partial paths */
        if (path_index++ == numpaths)
            break;

        costarr[min_index] += subpath->total_cost;

        /* Update the new min cost array index */
        min_index = 0;
        for (int i = 0; i < arrlen; i++) {
            if (costarr[i] < costarr[min_index])
                min_index = i;
        }
    }

    /* Return the highest cost from the array */
    max_index = 0;
    for (int i = 0; i < arrlen; i++) {
        if (costarr[i] > costarr[max_index])
            max_index = i;
    }

    return costarr[max_index];
}

/*
 * cost_append
 *	  Determines and returns the cost of an Append node.
 */
void cost_append(AppendPath *apath) {
    ListCell *l;

    apath->path.startup_cost = 0;
    apath->path.total_cost = 0;
    apath->path.rows = 0;

    if (apath->subpaths == NIL)
        return;

    if (!apath->path.parallel_aware) {
        List *pathkeys = apath->path.pathkeys;

        if (pathkeys == NIL) {
            Path *firstsubpath = (Path *) linitial(apath->subpaths);

            /*
             * For an unordered, non-parallel-aware Append we take the startup
             * cost as the startup cost of the first subpath.
             */
            apath->path.startup_cost = firstsubpath->startup_cost;

            /* Compute rows and costs as sums of subplan rows and costs. */
            foreach(l, apath->subpaths) {
                Path *subpath = (Path *) lfirst(l);

                apath->path.rows += subpath->rows;
                apath->path.total_cost += subpath->total_cost;
            }
        } else {
            /*
             * For an ordered, non-parallel-aware Append we take the startup
             * cost as the sum of the subpath startup costs.  This ensures
             * that we don't underestimate the startup cost when a query's
             * LIMIT is such that several of the children have to be run to
             * satisfy it.  This might be overkill --- another plausible hack
             * would be to take the Append's startup cost as the maximum of
             * the child startup costs.  But we don't want to risk believing
             * that an ORDER BY LIMIT query can be satisfied at small cost
             * when the first child has small startup cost but later ones
             * don't.  (If we had the ability to deal with nonlinear cost
             * interpolation for partial retrievals, we would not need to be
             * so conservative about this.)
             *
             * This case is also different from the above in that we have to
             * account for possibly injecting sorts into subpaths that aren't
             * natively ordered.
             */
            foreach(l, apath->subpaths) {
                Path *subpath = (Path *) lfirst(l);
                Path sort_path; /* dummy for result of cost_sort */

                if (!pathkeys_contained_in(pathkeys, subpath->pathkeys)) {
                    /*
                     * We'll need to insert a Sort node, so include costs for
                     * that.  We can use the parent's LIMIT if any, since we
                     * certainly won't pull more than that many tuples from
                     * any child.
                     */
                    cost_sort(&sort_path,
                              NULL, /* doesn't currently need root */
                              pathkeys,
                              subpath->total_cost,
                              subpath->rows,
                              subpath->pathtarget->width,
                              0.0,
                              work_mem,
                              apath->limit_tuples);
                    subpath = &sort_path;
                }

                apath->path.rows += subpath->rows;
                apath->path.startup_cost += subpath->startup_cost;
                apath->path.total_cost += subpath->total_cost;
            }
        }
    } else /* parallel-aware */
    {
        int i = 0;
        double parallel_divisor = get_parallel_divisor(&apath->path);

        /* Parallel-aware Append never produces ordered output. */
        Assert(apath->path.pathkeys == NIL);

        /* Calculate startup cost. */
        foreach(l, apath->subpaths) {
            Path *subpath = (Path *) lfirst(l);

            /*
             * Append will start returning tuples when the child node having
             * lowest startup cost is done setting up. We consider only the
             * first few subplans that immediately get a worker assigned.
             */
            if (i == 0)
                apath->path.startup_cost = subpath->startup_cost;
            else if (i < apath->path.parallel_workers)
                apath->path.startup_cost = Min(apath->path.startup_cost,
                                               subpath->startup_cost);

            /*
             * Apply parallel divisor to subpaths.  Scale the number of rows
             * for each partial subpath based on the ratio of the parallel
             * divisor originally used for the subpath to the one we adopted.
             * Also add the cost of partial paths to the total cost, but
             * ignore non-partial paths for now.
             */
            if (i < apath->first_partial_path)
                apath->path.rows += subpath->rows / parallel_divisor;
            else {
                double subpath_parallel_divisor;

                subpath_parallel_divisor = get_parallel_divisor(subpath);
                apath->path.rows += subpath->rows * (subpath_parallel_divisor /
                                                     parallel_divisor);
                apath->path.total_cost += subpath->total_cost;
            }

            apath->path.rows = clamp_row_est(apath->path.rows);

            i++;
        }

        /* Add cost for non-partial subpaths. */
        apath->path.total_cost +=
                append_nonpartial_cost(apath->subpaths,
                                       apath->first_partial_path,
                                       apath->path.parallel_workers);
    }

    /*
     * Although Append does not do any selection or projection, it's not free;
     * add a small per-tuple overhead.
     */
    apath->path.total_cost +=
            cpu_tuple_cost * APPEND_CPU_COST_MULTIPLIER * apath->path.rows;
}

/*
 * cost_merge_append
 *	  Determines and returns the cost of a MergeAppend node.
 *
 * MergeAppend merges several pre-sorted input streams, using a heap that
 * at any given instant holds the next tuple from each stream.  If there
 * are N streams, we need about N*log2(N) tuple comparisons to construct
 * the heap at startup, and then for each output tuple, about log2(N)
 * comparisons to replace the top entry.
 *
 * (The effective value of N will drop once some of the input streams are
 * exhausted, but it seems unlikely to be worth trying to account for that.)
 *
 * The heap is never spilled to disk, since we assume N is not very large.
 * So this is much simpler than cost_sort.
 *
 * As in cost_sort, we charge two operator evals per tuple comparison.
 *
 * 'pathkeys' is a list of sort keys
 * 'n_streams' is the number of input streams
 * 'input_startup_cost' is the sum of the input streams' startup costs
 * 'input_total_cost' is the sum of the input streams' total costs
 * 'tuples' is the number of tuples in all the streams
 */
void cost_merge_append(
    Path *path, PlannerInfo *root,
    List *pathkeys, int n_streams,
    Cost input_startup_cost, Cost input_total_cost,
    double tuples
) {
    Cost startup_cost = 0;
    Cost run_cost = 0;
    Cost comparison_cost;
    double N;
    double logN;

    /*
     * Avoid log(0)...
     */
    N = (n_streams < 2) ? 2.0 : (double) n_streams;
    logN = LOG2(N);

    /* Assumed cost per tuple comparison */
    comparison_cost = 2.0 * cpu_operator_cost;

    /* Heap creation cost */
    startup_cost += comparison_cost * N * logN;

    /* Per-tuple heap maintenance cost */
    run_cost += tuples * comparison_cost * logN;

    /*
     * Although MergeAppend does not do any selection or projection, it's not
     * free; add a small per-tuple overhead.
     */
    run_cost += cpu_tuple_cost * APPEND_CPU_COST_MULTIPLIER * tuples;

    path->startup_cost = startup_cost + input_startup_cost;
    path->total_cost = startup_cost + run_cost + input_total_cost;
}

/*
 * cost_material
 *	  Determines and returns the cost of materializing a relation, including
 *	  the cost of reading the input data.
 *
 * If the total volume of data to materialize exceeds work_mem, we will need
 * to write it to disk, so the cost is much higher in that case.
 *
 * Note that here we are estimating the costs for the first scan of the
 * relation, so the materialization is all overhead --- any savings will
 * occur only on rescan, which is estimated in cost_rescan.
 */
void cost_material(
    Path *path,
    Cost input_startup_cost, Cost input_total_cost,
    double tuples, int width
) {
    Cost startup_cost = input_startup_cost;
    Cost run_cost = input_total_cost - input_startup_cost;
    double nbytes = relation_byte_size(tuples, width);
    long work_mem_bytes = work_mem * 1024L;

    path->rows = tuples;

    /*
     * Whether spilling or not, charge 2x cpu_operator_cost per tuple to
     * reflect bookkeeping overhead.  (This rate must be more than what
     * cost_rescan charges for materialize, ie, cpu_operator_cost per tuple;
     * if it is exactly the same then there will be a cost tie between
     * nestloop with A outer, materialized B inner and nestloop with B outer,
     * materialized A inner.  The extra cost ensures we'll prefer
     * materializing the smaller rel.)	Note that this is normally a good deal
     * less than cpu_tuple_cost; which is OK because a Material plan node
     * doesn't do qual-checking or projection, so it's got less overhead than
     * most plan nodes.
     */
    run_cost += 2 * cpu_operator_cost * tuples;

    /*
     * If we will spill to disk, charge at the rate of seq_page_cost per page.
     * This cost is assumed to be evenly spread through the plan run phase,
     * which isn't exactly accurate but our cost model doesn't allow for
     * nonuniform costs within the run phase.
     */
    if (nbytes > work_mem_bytes) {
        double npages = ceil(nbytes / BLCKSZ);

        run_cost += seq_page_cost * npages;
    }

    path->startup_cost = startup_cost;
    path->total_cost = startup_cost + run_cost;
}

/*
 * cost_agg
 *		Determines and returns the cost of performing an Agg plan node,
 *		including the cost of its input.
 *
 * aggcosts can be NULL when there are no actual aggregate functions (i.e.,
 * we are using a hashed Agg node just to do grouping).
 *
 * Note: when aggstrategy == AGG_SORTED, caller must ensure that input costs
 * are for appropriately-sorted input.
 */
void cost_agg(
    Path *path, PlannerInfo *root,
    const AggStrategy aggstrategy, const AggClauseCosts *aggcosts,
    const int numGroupCols, const double numGroups, List *quals,
    const Cost input_startup_cost, const Cost input_total_cost,
    const double input_tuples, const double input_width
) {
    double output_tuples;
    Cost startup_cost;
    Cost total_cost;
    AggClauseCosts dummy_aggcosts;

    /* Use all-zero per-aggregate costs if NULL is passed */
    if (aggcosts == NULL) {
        Assert(aggstrategy == AGG_HASHED);
        MemSet(&dummy_aggcosts, 0, sizeof(AggClauseCosts));
        aggcosts = &dummy_aggcosts;
    }

    /*
     * The transCost.per_tuple component of aggcosts should be charged once
     * per input tuple, corresponding to the costs of evaluating the aggregate
     * transfns and their input expressions. The finalCost.per_tuple component
     * is charged once per output tuple, corresponding to the costs of
     * evaluating the finalfns.  Startup costs are of course charged but once.
     *
     * If we are grouping, we charge an additional cpu_operator_cost per
     * grouping column per input tuple for grouping comparisons.
     *
     * We will produce a single output tuple if not grouping, and a tuple per
     * group otherwise.  We charge cpu_tuple_cost for each output tuple.
     *
     * Note: in this cost model, AGG_SORTED and AGG_HASHED have exactly the
     * same total CPU cost, but AGG_SORTED has lower startup cost.  If the
     * input path is already sorted appropriately, AGG_SORTED should be
     * preferred (since it has no risk of memory overflow).  This will happen
     * as long as the computed total costs are indeed exactly equal --- but if
     * there's roundoff error we might do the wrong thing.  So be sure that
     * the computations below form the same intermediate values in the same
     * order.
     */
    if (aggstrategy == AGG_PLAIN) {
        startup_cost = input_total_cost;
        startup_cost += aggcosts->transCost.startup;
        startup_cost += aggcosts->transCost.per_tuple * input_tuples;
        startup_cost += aggcosts->finalCost.startup;
        startup_cost += aggcosts->finalCost.per_tuple;
        /* we aren't grouping */
        total_cost = startup_cost + cpu_tuple_cost;
        output_tuples = 1;
    } else if (aggstrategy == AGG_SORTED || aggstrategy == AGG_MIXED) {
        /* Here we are able to deliver output on-the-fly */
        startup_cost = input_startup_cost;
        total_cost = input_total_cost;
        if (aggstrategy == AGG_MIXED && !enable_hashagg) {
            startup_cost += disable_cost;
            total_cost += disable_cost;
        }
        /* calcs phrased this way to match HASHED case, see note above */
        total_cost += aggcosts->transCost.startup;
        total_cost += aggcosts->transCost.per_tuple * input_tuples;
        total_cost += (cpu_operator_cost * numGroupCols) * input_tuples;
        total_cost += aggcosts->finalCost.startup;
        total_cost += aggcosts->finalCost.per_tuple * numGroups;
        total_cost += cpu_tuple_cost * numGroups;
        output_tuples = numGroups;
    } else {
        /* must be AGG_HASHED */
        startup_cost = input_total_cost;
        if (!enable_hashagg)
            startup_cost += disable_cost;
        startup_cost += aggcosts->transCost.startup;
        startup_cost += aggcosts->transCost.per_tuple * input_tuples;
        /* cost of computing hash value */
        startup_cost += (cpu_operator_cost * numGroupCols) * input_tuples;
        startup_cost += aggcosts->finalCost.startup;

        total_cost = startup_cost;
        total_cost += aggcosts->finalCost.per_tuple * numGroups;
        /* cost of retrieving from hash table */
        total_cost += cpu_tuple_cost * numGroups;
        output_tuples = numGroups;
    }

    /*
     * Add the disk costs of hash aggregation that spills to disk.
     *
     * Groups that go into the hash table stay in memory until finalized, so
     * spilling and reprocessing tuples doesn't incur additional invocations
     * of transCost or finalCost. Furthermore, the computed hash value is
     * stored with the spilled tuples, so we don't incur extra invocations of
     * the hash function.
     *
     * Hash Agg begins returning tuples after the first batch is complete.
     * Accrue writes (spilled tuples) to startup_cost and to total_cost;
     * accrue reads only to total_cost.
     */
    if (aggstrategy == AGG_HASHED || aggstrategy == AGG_MIXED) {
        double pages;
        double pages_written = 0.0;
        double pages_read = 0.0;
        double spill_cost;
        double hashentrysize;
        double nbatches;
        Size mem_limit;
        uint64 ngroups_limit;
        int num_partitions;
        int depth;

        /*
         * Estimate number of batches based on the computed limits. If less
         * than or equal to one, all groups are expected to fit in memory;
         * otherwise we expect to spill.
         */
        hashentrysize = hash_agg_entry_size(list_length(root->aggtransinfos),
                                            input_width,
                                            aggcosts->transitionSpace);
        hash_agg_set_limits(hashentrysize, numGroups, 0, &mem_limit,
                            &ngroups_limit, &num_partitions);

        nbatches = Max((numGroups * hashentrysize) / mem_limit,
                       numGroups / ngroups_limit);

        nbatches = Max(ceil(nbatches), 1.0);
        num_partitions = Max(num_partitions, 2);

        /*
         * The number of partitions can change at different levels of
         * recursion; but for the purposes of this calculation assume it stays
         * constant.
         */
        depth = ceil(log(nbatches) / log(num_partitions));

        /*
         * Estimate number of pages read and written. For each level of
         * recursion, a tuple must be written and then later read.
         */
        pages = relation_byte_size(input_tuples, input_width) / BLCKSZ;
        pages_written = pages_read = pages * depth;

        /*
         * HashAgg has somewhat worse IO behavior than Sort on typical
         * hardware/OS combinations. Account for this with a generic penalty.
         */
        pages_read *= 2.0;
        pages_written *= 2.0;

        startup_cost += pages_written * random_page_cost;
        total_cost += pages_written * random_page_cost;
        total_cost += pages_read * seq_page_cost;

        /* account for CPU cost of spilling a tuple and reading it back */
        spill_cost = depth * input_tuples * 2.0 * cpu_tuple_cost;
        startup_cost += spill_cost;
        total_cost += spill_cost;
    }

    /*
     * If there are quals (HAVING quals), account for their cost and
     * selectivity.
     */
    if (quals) {
        QualCost qual_cost;

        cost_qual_eval(&qual_cost, quals, root);
        startup_cost += qual_cost.startup;
        total_cost += qual_cost.startup + output_tuples * qual_cost.per_tuple;

        output_tuples = clamp_row_est(
            output_tuples * clauselist_selectivity(
                root, quals, 0, JOIN_INNER, NULL
            )
        );
    }

    path->rows = output_tuples;
    path->startup_cost = startup_cost;
    path->total_cost = total_cost;

    path->rows_sample = make_sample_by_single_value(output_tuples);
    path->startup_cost_sample = make_sample_by_single_value(startup_cost);
    path->total_cost_sample = make_sample_by_single_value(total_cost);
}

/*
 * cost_windowagg
 *		Determines and returns the cost of performing a WindowAgg plan node,
 *		including the cost of its input.
 *
 * Input is assumed already properly sorted.
 */
void cost_windowagg(
    Path *path, PlannerInfo *root,
    List *windowFuncs, const int numPartCols, const int numOrderCols,
    const Cost input_startup_cost, const Cost input_total_cost,
    const double input_tuples
) {
    ListCell *lc;

    Cost startup_cost = input_startup_cost;
    Cost total_cost = input_total_cost;

    /*
     * Window functions are assumed to cost their stated execution cost, plus
     * the cost of evaluating their input expressions, per tuple.  Since they
     * may in fact evaluate their inputs at multiple rows during each cycle,
     * this could be a drastic underestimate; but without a way to know how
     * many rows the window function will fetch, it's hard to do better.  In
     * any case, it's a good estimate for all the built-in window functions,
     * so we'll just do this for now.
     */
    foreach(lc, windowFuncs) {
        WindowFunc *wfunc = lfirst_node(WindowFunc, lc);
        QualCost argcosts;

        argcosts.startup = argcosts.per_tuple = 0;
        add_function_cost(root, wfunc->winfnoid, (Node *) wfunc,
                          &argcosts);
        startup_cost += argcosts.startup;
        Cost wfunccost = argcosts.per_tuple;

        /* also add the input expressions' cost to per-input-row costs */
        cost_qual_eval_node(&argcosts, (Node *) wfunc->args, root);
        startup_cost += argcosts.startup;
        wfunccost += argcosts.per_tuple;

        /*
         * Add the filter's cost to per-input-row costs.  XXX We should reduce
         * input expression costs according to filter selectivity.
         */
        cost_qual_eval_node(&argcosts, (Node *) wfunc->aggfilter, root);
        startup_cost += argcosts.startup;
        wfunccost += argcosts.per_tuple;

        total_cost += wfunccost * input_tuples;
    }

    /*
     * We also charge cpu_operator_cost per grouping column per tuple for
     * grouping comparisons, plus cpu_tuple_cost per tuple for general
     * overhead.
     *
     * XXX this neglects costs of spooling the data to disk when it overflows
     * work_mem.  Sooner or later that should get accounted for.
     */
    total_cost += cpu_operator_cost * (numPartCols + numOrderCols) * input_tuples;
    total_cost += cpu_tuple_cost * input_tuples;

    path->rows = input_tuples;
    path->startup_cost = startup_cost;
    path->total_cost = total_cost;
}

/*
 * cost_group
 *		Determines and returns the cost of performing a Group plan node,
 *		including the cost of its input.
 *
 * Note: caller must ensure that input costs are for appropriately-sorted
 * input.
 */
void cost_group(
    Path *path, PlannerInfo *root,
    int numGroupCols, double numGroups,
    List *quals,
    Cost input_startup_cost, Cost input_total_cost,
    double input_tuples
) {
    double output_tuples;
    Cost startup_cost;
    Cost total_cost;

    output_tuples = numGroups;
    startup_cost = input_startup_cost;
    total_cost = input_total_cost;

    /*
     * Charge one cpu_operator_cost per comparison per input tuple. We assume
     * all columns get compared at most of the tuples.
     */
    total_cost += cpu_operator_cost * input_tuples * numGroupCols;

    /*
     * If there are quals (HAVING quals), account for their cost and
     * selectivity.
     */
    if (quals) {
        QualCost qual_cost;

        cost_qual_eval(&qual_cost, quals, root);
        startup_cost += qual_cost.startup;
        total_cost += qual_cost.startup + output_tuples * qual_cost.per_tuple;

        output_tuples = clamp_row_est(
            output_tuples * clauselist_selectivity(
                root, quals, 0, JOIN_INNER, NULL
            ));
    }

    path->rows = output_tuples;
    path->startup_cost = startup_cost;
    path->total_cost = total_cost;
}

/*
 * initial_cost_nestloop
 *	  Preliminary estimate of the cost of a nestloop join path.
 *
 * This must quickly produce lower-bound estimates of the path's startup and
 * total costs.  If we are unable to eliminate the proposed path from
 * consideration using the lower bounds, final_cost_nestloop will be called
 * to obtain the final estimates.
 *
 * The exact division of labor between this function and final_cost_nestloop
 * is private to them, and represents a tradeoff between speed of the initial
 * estimate and getting a tight lower bound.  We choose to not examine the
 * join quals here, since that's by far the most expensive part of the
 * calculations.  The end result is that CPU-cost considerations must be
 * left for the second phase; and for SEMI/ANTI joins, we must also postpone
 * incorporation of the inner path's run cost.
 *
 * 'workspace' is to be filled with startup_cost, total_cost, and perhaps
 *		other data to be used by final_cost_nestloop
 * 'jointype' is the type of join to be performed
 * 'outer_path' is the outer input to the join
 * 'inner_path' is the inner input to the join
 * 'extra' contains miscellaneous information about the join
 */
void initial_cost_nestloop(
    PlannerInfo *root,
    JoinCostWorkspace *workspace,
    const JoinType jointype,
    const Path *outer_path,
    Path *inner_path,
    const JoinPathExtraData *extra
) {
    if (root->pass == 1) {
        initial_cost_nestloop_1p(
            root, workspace, jointype, outer_path, inner_path, extra
        );
    } else if (root->pass == 2) {
        initial_cost_nestloop_2p(
            root, workspace, jointype, outer_path, inner_path, extra
        );
    } else {
        elog(ERROR, "bad pass number");
    }
}

/*
 * final_cost_nestloop
 *	  Final estimate of the cost and result size of a nestloop join path.
 *
 * 'path' is already filled in except for the rows and cost fields
 * 'workspace' is the result from initial_cost_nestloop
 * 'extra' contains miscellaneous information about the join
 */
void final_cost_nestloop(
    PlannerInfo *root, NestPath *path,
    JoinCostWorkspace *workspace,
    const JoinPathExtraData *extra
) {
    if (root->pass == 1) {
        final_cost_nestloop_1p(root, path, workspace, extra);
    } else if (root->pass == 2) {
        final_cost_nestloop_2p(root, path, workspace, extra);
    } else {
        elog(ERROR, "bad pass number");
    }
}

/*
 * initial_cost_mergejoin
 *	  Preliminary estimate of the cost of a mergejoin path.
 *
 * This must quickly produce lower-bound estimates of the path's startup and
 * total costs.  If we are unable to eliminate the proposed path from
 * consideration using the lower bounds, final_cost_mergejoin will be called
 * to obtain the final estimates.
 *
 * The exact division of labor between this function and final_cost_mergejoin
 * is private to them, and represents a tradeoff between speed of the initial
 * estimate and getting a tight lower bound.  We choose to not examine the
 * join quals here, except for obtaining the scan selectivity estimate which
 * is really essential (but fortunately, use of caching keeps the cost of
 * getting that down to something reasonable).
 * We also assume that cost_sort is cheap enough to use here.
 *
 * 'workspace' is to be filled with startup_cost, total_cost, and perhaps
 *		other data to be used by final_cost_mergejoin
 * 'jointype' is the type of join to be performed
 * 'mergeclauses' is the list of joinclauses to be used as merge clauses
 * 'outer_path' is the outer input to the join
 * 'inner_path' is the inner input to the join
 * 'outersortkeys' is the list of sort keys for the outer path
 * 'innersortkeys' is the list of sort keys for the inner path
 * 'extra' contains miscellaneous information about the join
 *
 * Note: outersortkeys and innersortkeys should be NIL if no explicit
 * sort is needed because the respective source path is already ordered.
 */
/*
 * initial_cost_mergejoin
 *   Preliminary (lower-bound) cost estimate for a merge join, extended with
 *   per-sample "point-to-point" evaluation.
 *
 * Phase-1 must be fast and produce lower bounds on startup_cost/total_cost.
 * If the path is not rejected by these bounds, phase-2 (final_cost_mergejoin)
 * will refine them. We purposely examine only the first mergeclause to compute
 * scan selectivities and assume cost_sort is cheap enough here.
 *
 * This version evaluates outer/inner costs per sample: each sample index i
 * uses the i-th rows and costs (startup/total) from the corresponding paths,
 * falling back to scalar values (or sample[0]) when needed.
 *
 * Contract:
 *   - For every sample i, startup_cost_sample[i] and total_cost_sample[i]
 *     must be <= their final counterparts computed in phase-2.
 *   - Scalar fields (startup_cost/total_cost/...) store the averages of the
 *     per-sample lower bounds to preserve compatibility with legacy callers.
 */
void initial_cost_mergejoin(
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
    if (root->pass == 1) {
        initial_cost_mergejoin_1p(
            root, workspace, jointype, mergeclauses,
            outer_path, inner_path, outersortkeys, innersortkeys, extra
        );
    } else if (root->pass == 2) {
        initial_cost_mergejoin_2p(
            root, workspace, jointype, mergeclauses,
            outer_path, inner_path, outersortkeys, innersortkeys, extra
        );
    } else {
        elog(ERROR, "bad pass number");
    }
}

/*
 * final_cost_mergejoin
 *	  Final estimate of the cost and result size of a mergejoin path.
 *
 * Unlike other costsize functions, this routine makes two actual decisions:
 * whether the executor will need to do mark/restore, and whether we should
 * materialize the inner path.  It would be logically cleaner to build
 * separate paths testing these alternatives, but that would require repeating
 * most of the cost calculations, which are not all that cheap.  Since the
 * choice will not affect output pathkeys or startup cost, only total cost,
 * there is no possibility of wanting to keep more than one path.  So it seems
 * best to make the decisions here and record them in the path's
 * skip_mark_restore and materialize_inner fields.
 *
 * Mark/restore overhead is usually required, but can be skipped if we know
 * that the executor need find only one match per outer tuple, and that the
 * mergeclauses are sufficient to identify a match.
 *
 * We materialize the inner path if we need mark/restore and either the inner
 * path can't support mark/restore, or it's cheaper to use an interposed
 * Material node to handle mark/restore.
 *
 * 'path' is already filled in except for the rows and cost fields and
 *		skip_mark_restore and materialize_inner
 * 'workspace' is the result from initial_cost_mergejoin
 * 'extra' contains miscellaneous information about the join
 */
void final_cost_mergejoin(
    PlannerInfo *root, MergePath *path,
    JoinCostWorkspace *workspace,
    const JoinPathExtraData *extra
) {
    if (root->pass == 1) {
        final_cost_mergejoin_1p(root, path, workspace, extra);
    } else if (root->pass == 2) {
        final_cost_mergejoin_2p(root, path, workspace, extra);
    } else {
        elog(ERROR, "bad pass number");
    }
}

/*
 * initial_cost_hashjoin
 *	  Preliminary estimate of the cost of a hashjoin path.
 *
 * This must quickly produce lower-bound estimates of the path's startup and
 * total costs.  If we are unable to eliminate the proposed path from
 * consideration using the lower bounds, final_cost_hashjoin will be called
 * to obtain the final estimates.
 *
 * The exact division of labor between this function and final_cost_hashjoin
 * is private to them, and represents a tradeoff between speed of the initial
 * estimate and getting a tight lower bound.  We choose to not examine the
 * join quals here (other than by counting the number of hash clauses),
 * so we can't do much with CPU costs.  We do assume that
 * ExecChooseHashTableSize is cheap enough to use here.
 *
 * 'workspace' is to be filled with startup_cost, total_cost, and perhaps
 *		other data to be used by final_cost_hashjoin
 * 'jointype' is the type of join to be performed
 * 'hashclauses' is the list of joinclauses to be used as hash clauses
 * 'outer_path' is the outer input to the join
 * 'inner_path' is the inner input to the join
 * 'extra' contains miscellaneous information about the join
 * 'parallel_hash' indicates that inner_path is partial and that a shared
 *		hash table will be built in parallel
 */
void initial_cost_hashjoin(
    PlannerInfo *root,
    JoinCostWorkspace *workspace,
    JoinType jointype,
    const List *hashclauses,
    Path *outer_path,
    Path *inner_path,
    JoinPathExtraData *extra,
    bool parallel_hash
) {
    if (root->pass == 1) {
        initial_cost_hashjoin_1p(
            root, workspace, jointype, hashclauses,
            outer_path, inner_path, extra, parallel_hash
        );
    } else if (root->pass == 2) {
        initial_cost_hashjoin_2p(
            root, workspace, jointype, hashclauses,
            outer_path, inner_path, extra, parallel_hash
        );
    } else {
        elog(ERROR, "bad pass number");
    }
}

/*
 * final_cost_hashjoin
 *	  Final estimate of the cost and result size of a hashjoin path.
 *
 * Note: the numbatches estimate is also saved into 'path' for use later
 *
 * 'path' is already filled in except for the rows and cost fields and
 *		num_batches
 * 'workspace' is the result from initial_cost_hashjoin
 * 'extra' contains miscellaneous information about the join
 */
void final_cost_hashjoin(
    PlannerInfo *root, HashPath *path,
    JoinCostWorkspace *workspace,
    const JoinPathExtraData *extra
) {
    if (root->pass == 1) {
        final_cost_hashjoin_1p(root, path, workspace, extra);
    } else if (root->pass == 2) {
        final_cost_hashjoin_2p(root, path, workspace, extra);
    } else {
        elog(ERROR, "bad pass number");
    }
}

/*
 * cost_subplan
 *		Figure the costs for a SubPlan (or initplan).
 *
 * Note: we could dig the subplan's Plan out of the root list, but in practice
 * all callers have it handy already, so we make them pass it.
 */
void cost_subplan(
    PlannerInfo *root, SubPlan *subplan, Plan *plan
) {
    QualCost sp_cost;

    /* Figure any cost for evaluating the testexpr */
    cost_qual_eval(&sp_cost,
                   make_ands_implicit((Expr *) subplan->testexpr),
                   root);

    if (subplan->useHashTable) {
        /*
         * If we are using a hash table for the subquery outputs, then the
         * cost of evaluating the query is a one-time cost.  We charge one
         * cpu_operator_cost per tuple for the work of loading the hashtable,
         * too.
         */
        sp_cost.startup += plan->total_cost +
                cpu_operator_cost * plan->plan_rows;

        /*
         * The per-tuple costs include the cost of evaluating the lefthand
         * expressions, plus the cost of probing the hashtable.  We already
         * accounted for the lefthand expressions as part of the testexpr, and
         * will also have counted one cpu_operator_cost for each comparison
         * operator.  That is probably too low for the probing cost, but it's
         * hard to make a better estimate, so live with it for now.
         */
    } else {
        /*
         * Otherwise we will be rescanning the subplan output on each
         * evaluation.  We need to estimate how much of the output we will
         * actually need to scan.  NOTE: this logic should agree with the
         * tuple_fraction estimates used by make_subplan() in
         * plan/subselect.c.
         */
        Cost plan_run_cost = plan->total_cost - plan->startup_cost;

        if (subplan->subLinkType == EXISTS_SUBLINK) {
            /* we only need to fetch 1 tuple; clamp to avoid zero divide */
            sp_cost.per_tuple += plan_run_cost / clamp_row_est(plan->plan_rows);
        } else if (subplan->subLinkType == ALL_SUBLINK ||
                   subplan->subLinkType == ANY_SUBLINK) {
            /* assume we need 50% of the tuples */
            sp_cost.per_tuple += 0.50 * plan_run_cost;
            /* also charge a cpu_operator_cost per row examined */
            sp_cost.per_tuple += 0.50 * plan->plan_rows * cpu_operator_cost;
        } else {
            /* assume we need all tuples */
            sp_cost.per_tuple += plan_run_cost;
        }

        /*
         * Also account for subplan's startup cost. If the subplan is
         * uncorrelated or undirect correlated, AND its topmost node is one
         * that materializes its output, assume that we'll only need to pay
         * its startup cost once; otherwise assume we pay the startup cost
         * every time.
         */
        if (subplan->parParam == NIL &&
            ExecMaterializesOutput(nodeTag(plan)))
            sp_cost.startup += plan->startup_cost;
        else
            sp_cost.per_tuple += plan->startup_cost;
    }

    subplan->startup_cost = sp_cost.startup;
    subplan->per_call_cost = sp_cost.per_tuple;
}

/*
 * cost_qual_eval
 *		Estimate the CPU costs of evaluating a WHERE clause.
 *		The input can be either an implicitly-ANDed list of boolean
 *		expressions, or a list of RestrictInfo nodes.  (The latter is
 *		preferred since it allows caching of the results.)
 *		The result includes both a one-time (startup) component,
 *		and a per-evaluation component.
 */
void cost_qual_eval(QualCost *cost, List *quals, PlannerInfo *root) {
    cost_qual_eval_context context;
    ListCell *l;

    context.root = root;
    context.total.startup = 0;
    context.total.per_tuple = 0;

    /* We don't charge any cost for the implicit ANDing at top level ... */

    foreach(l, quals) {
        Node *qual = (Node *) lfirst(l);

        cost_qual_eval_walker(qual, &context);
    }

    *cost = context.total;
}

/*
 * cost_qual_eval_node
 *		As above, for a single RestrictInfo or expression.
 */
void cost_qual_eval_node(QualCost *cost, Node *qual, PlannerInfo *root) {
    cost_qual_eval_context context;

    context.root = root;
    context.total.startup = 0;
    context.total.per_tuple = 0;

    cost_qual_eval_walker(qual, &context);

    *cost = context.total;
}

static bool cost_qual_eval_walker(Node *node, cost_qual_eval_context *context) {
    if (node == NULL)
        return false;

    /*
     * RestrictInfo nodes contain an eval_cost field reserved for this
     * routine's use, so that it's not necessary to evaluate the qual clause's
     * cost more than once.  If the clause's cost hasn't been computed yet,
     * the field's startup value will contain -1.
     */
    if (IsA(node, RestrictInfo)) {
        RestrictInfo *rinfo = (RestrictInfo *) node;

        if (rinfo->eval_cost.startup < 0) {
            cost_qual_eval_context locContext;

            locContext.root = context->root;
            locContext.total.startup = 0;
            locContext.total.per_tuple = 0;

            /*
             * For an OR clause, recurse into the marked-up tree so that we
             * set the eval_cost for contained RestrictInfos too.
             */
            if (rinfo->orclause)
                cost_qual_eval_walker((Node *) rinfo->orclause, &locContext);
            else
                cost_qual_eval_walker((Node *) rinfo->clause, &locContext);

            /*
             * If the RestrictInfo is marked pseudoconstant, it will be tested
             * only once, so treat its cost as all startup cost.
             */
            if (rinfo->pseudoconstant) {
                /* count one execution during startup */
                locContext.total.startup += locContext.total.per_tuple;
                locContext.total.per_tuple = 0;
            }
            rinfo->eval_cost = locContext.total;
        }
        context->total.startup += rinfo->eval_cost.startup;
        context->total.per_tuple += rinfo->eval_cost.per_tuple;
        /* do NOT recurse into children */
        return false;
    }

    /*
     * For each operator or function node in the given tree, we charge the
     * estimated execution cost given by pg_proc.procost (remember to multiply
     * this by cpu_operator_cost).
     *
     * Vars and Consts are charged zero, and so are boolean operators (AND,
     * OR, NOT). Simplistic, but a lot better than no model at all.
     *
     * Should we try to account for the possibility of short-circuit
     * evaluation of AND/OR?  Probably *not*, because that would make the
     * results depend on the clause ordering, and we are not in any position
     * to expect that the current ordering of the clauses is the one that's
     * going to end up being used.  The above per-RestrictInfo caching would
     * not mix well with trying to re-order clauses anyway.
     *
     * Another issue that is entirely ignored here is that if a set-returning
     * function is below top level in the tree, the functions/operators above
     * it will need to be evaluated multiple times.  In practical use, such
     * cases arise so seldom as to not be worth the added complexity needed;
     * moreover, since our rowcount estimates for functions tend to be pretty
     * phony, the results would also be pretty phony.
     */
    if (IsA(node, FuncExpr)) {
        add_function_cost(context->root, ((FuncExpr *) node)->funcid, node,
                          &context->total);
    } else if (IsA(node, OpExpr) ||
               IsA(node, DistinctExpr) ||
               IsA(node, NullIfExpr)) {
        /* rely on struct equivalence to treat these all alike */
        set_opfuncid((OpExpr *) node);
        add_function_cost(context->root, ((OpExpr *) node)->opfuncid, node,
                          &context->total);
    } else if (IsA(node, ScalarArrayOpExpr)) {
        ScalarArrayOpExpr *saop = (ScalarArrayOpExpr *) node;
        Node *arraynode = (Node *) lsecond(saop->args);
        QualCost sacosts;
        QualCost hcosts;
        int estarraylen = estimate_array_length(arraynode);

        set_sa_opfuncid(saop);
        sacosts.startup = sacosts.per_tuple = 0;
        add_function_cost(context->root, saop->opfuncid, NULL,
                          &sacosts);

        if (OidIsValid(saop->hashfuncid)) {
            /* Handle costs for hashed ScalarArrayOpExpr */
            hcosts.startup = hcosts.per_tuple = 0;

            add_function_cost(context->root, saop->hashfuncid, NULL, &hcosts);
            context->total.startup += sacosts.startup + hcosts.startup;

            /* Estimate the cost of building the hashtable. */
            context->total.startup += estarraylen * hcosts.per_tuple;

            /*
             * XXX should we charge a little bit for sacosts.per_tuple when
             * building the table, or is it ok to assume there will be zero
             * hash collision?
             */

            /*
             * Charge for hashtable lookups.  Charge a single hash and a
             * single comparison.
             */
            context->total.per_tuple += hcosts.per_tuple + sacosts.per_tuple;
        } else {
            /*
             * Estimate that the operator will be applied to about half of the
             * array elements before the answer is determined.
             */
            context->total.startup += sacosts.startup;
            context->total.per_tuple += sacosts.per_tuple *
                    estimate_array_length(arraynode) * 0.5;
        }
    } else if (IsA(node, Aggref) ||
               IsA(node, WindowFunc)) {
        /*
         * Aggref and WindowFunc nodes are (and should be) treated like Vars,
         * ie, zero execution cost in the current model, because they behave
         * essentially like Vars at execution.  We disregard the costs of
         * their input expressions for the same reason.  The actual execution
         * costs of the aggregate/window functions and their arguments have to
         * be factored into plan-node-specific costing of the Agg or WindowAgg
         * plan node.
         */
        return false; /* don't recurse into children */
    } else if (IsA(node, GroupingFunc)) {
        /* Treat this as having cost 1 */
        context->total.per_tuple += cpu_operator_cost;
        return false; /* don't recurse into children */
    } else if (IsA(node, CoerceViaIO)) {
        CoerceViaIO *iocoerce = (CoerceViaIO *) node;
        Oid iofunc;
        Oid typioparam;
        bool typisvarlena;

        /* check the result type's input function */
        getTypeInputInfo(iocoerce->resulttype,
                         &iofunc, &typioparam);
        add_function_cost(context->root, iofunc, NULL,
                          &context->total);
        /* check the input type's output function */
        getTypeOutputInfo(exprType((Node *) iocoerce->arg),
                          &iofunc, &typisvarlena);
        add_function_cost(context->root, iofunc, NULL,
                          &context->total);
    } else if (IsA(node, ArrayCoerceExpr)) {
        ArrayCoerceExpr *acoerce = (ArrayCoerceExpr *) node;
        QualCost perelemcost;

        cost_qual_eval_node(&perelemcost, (Node *) acoerce->elemexpr,
                            context->root);
        context->total.startup += perelemcost.startup;
        if (perelemcost.per_tuple > 0)
            context->total.per_tuple += perelemcost.per_tuple *
                    estimate_array_length((Node *) acoerce->arg);
    } else if (IsA(node, RowCompareExpr)) {
        /* Conservatively assume we will check all the columns */
        RowCompareExpr *rcexpr = (RowCompareExpr *) node;
        ListCell *lc;

        foreach(lc, rcexpr->opnos) {
            Oid opid = lfirst_oid(lc);

            add_function_cost(context->root, get_opcode(opid), NULL,
                              &context->total);
        }
    } else if (IsA(node, MinMaxExpr) ||
               IsA(node, SQLValueFunction) ||
               IsA(node, XmlExpr) ||
               IsA(node, CoerceToDomain) ||
               IsA(node, NextValueExpr)) {
        /* Treat all these as having cost 1 */
        context->total.per_tuple += cpu_operator_cost;
    } else if (IsA(node, CurrentOfExpr)) {
        /* Report high cost to prevent selection of anything but TID scan */
        context->total.startup += disable_cost;
    } else if (IsA(node, SubLink)) {
        /* This routine should not be applied to un-planned expressions */
        elog(ERROR, "cannot handle unplanned sub-select");
    } else if (IsA(node, SubPlan)) {
        /*
         * A subplan node in an expression typically indicates that the
         * subplan will be executed on each evaluation, so charge accordingly.
         * (Sub-selects that can be executed as InitPlans have already been
         * removed from the expression.)
         */
        SubPlan *subplan = (SubPlan *) node;

        context->total.startup += subplan->startup_cost;
        context->total.per_tuple += subplan->per_call_cost;

        /*
         * We don't want to recurse into the testexpr, because it was already
         * counted in the SubPlan node's costs.  So we're done.
         */
        return false;
    } else if (IsA(node, AlternativeSubPlan)) {
        /*
         * Arbitrarily use the first alternative plan for costing.  (We should
         * certainly only include one alternative, and we don't yet have
         * enough information to know which one the executor is most likely to
         * use.)
         */
        AlternativeSubPlan *asplan = (AlternativeSubPlan *) node;

        return cost_qual_eval_walker((Node *) linitial(asplan->subplans),
                                     context);
    } else if (IsA(node, PlaceHolderVar)) {
        /*
         * A PlaceHolderVar should be given cost zero when considering general
         * expression evaluation costs.  The expense of doing the contained
         * expression is charged as part of the tlist eval costs of the scan
         * or join where the PHV is first computed (see set_rel_width and
         * add_placeholders_to_joinrel).  If we charged it again here, we'd be
         * double-counting the cost for each level of plan that the PHV
         * bubbles up through.  Hence, return without recursing into the
         * phexpr.
         */
        return false;
    }

    /* recurse into children */
    return expression_tree_walker(node, cost_qual_eval_walker,
                                  (void *) context);
}

/*
 * compute_semi_anti_join_factors
 *	  Estimate how much of the inner input a SEMI, ANTI, or inner_unique join
 *	  can be expected to scan.
 *
 * In a hash or nestloop SEMI/ANTI join, the executor will stop scanning
 * inner rows as soon as it finds a match to the current outer row.
 * The same happens if we have detected the inner rel is unique.
 * We should therefore adjust some of the cost components for this effect.
 * This function computes some estimates needed for these adjustments.
 * These estimates will be the same regardless of the particular paths used
 * for the outer and inner relation, so we compute these once and then pass
 * them to all the join cost estimation functions.
 *
 * Input parameters:
 *	joinrel: join relation under consideration
 *	outerrel: outer relation under consideration
 *	innerrel: inner relation under consideration
 *	jointype: if not JOIN_SEMI or JOIN_ANTI, we assume it's inner_unique
 *	sjinfo: SpecialJoinInfo relevant to this join
 *	restrictlist: join quals
 * Output parameters:
 *	*semifactors is filled in (see pathnodes.h for field definitions)
 */
void compute_semi_anti_join_factors(
    PlannerInfo *root,
    RelOptInfo *joinrel,
    RelOptInfo *outerrel,
    RelOptInfo *innerrel,
    JoinType jointype,
    SpecialJoinInfo *sjinfo,
    List *restrictlist,
    SemiAntiJoinFactors *semifactors
) {
    Selectivity jselec;
    Selectivity nselec;
    Selectivity avgmatch;
    SpecialJoinInfo norm_sjinfo;
    List *joinquals;
    ListCell *l;

    /*
     * In an ANTI join, we must ignore clauses that are "pushed down", since
     * those won't affect the match logic.  In a SEMI join, we do not
     * distinguish joinquals from "pushed down" quals, so just use the whole
     * restrictinfo list.  For other outer join types, we should consider only
     * non-pushed-down quals, so that this devolves to an IS_OUTER_JOIN check.
     */
    if (IS_OUTER_JOIN(jointype)) {
        joinquals = NIL;
        foreach(l, restrictlist) {
            RestrictInfo *rinfo = lfirst_node(RestrictInfo, l);

            if (!RINFO_IS_PUSHED_DOWN(rinfo, joinrel->relids))
                joinquals = lappend(joinquals, rinfo);
        }
    } else
        joinquals = restrictlist;

    /*
     * Get the JOIN_SEMI or JOIN_ANTI selectivity of the join clauses.
     */
    jselec = clauselist_selectivity(root,
                                    joinquals,
                                    0,
                                    (jointype == JOIN_ANTI) ? JOIN_ANTI : JOIN_SEMI,
                                    sjinfo);

    /*
     * Also get the normal inner-join selectivity of the join clauses.
     */
    norm_sjinfo.type = T_SpecialJoinInfo;
    norm_sjinfo.min_lefthand = outerrel->relids;
    norm_sjinfo.min_righthand = innerrel->relids;
    norm_sjinfo.syn_lefthand = outerrel->relids;
    norm_sjinfo.syn_righthand = innerrel->relids;
    norm_sjinfo.jointype = JOIN_INNER;
    norm_sjinfo.ojrelid = 0;
    norm_sjinfo.commute_above_l = NULL;
    norm_sjinfo.commute_above_r = NULL;
    norm_sjinfo.commute_below_l = NULL;
    norm_sjinfo.commute_below_r = NULL;
    /* we don't bother trying to make the remaining fields valid */
    norm_sjinfo.lhs_strict = false;
    norm_sjinfo.semi_can_btree = false;
    norm_sjinfo.semi_can_hash = false;
    norm_sjinfo.semi_operators = NIL;
    norm_sjinfo.semi_rhs_exprs = NIL;

    nselec = clauselist_selectivity(root,
                                    joinquals,
                                    0,
                                    JOIN_INNER,
                                    &norm_sjinfo);

    /* Avoid leaking a lot of ListCells */
    if (IS_OUTER_JOIN(jointype))
        list_free(joinquals);

    /*
     * jselec can be interpreted as the fraction of outer-rel rows that have
     * any matches (this is true for both SEMI and ANTI cases).  And nselec is
     * the fraction of the Cartesian product that matches.  So, the average
     * number of matches for each outer-rel row that has at least one match is
     * nselec * inner_rows / jselec.
     *
     * Note: it is correct to use the inner rel's "rows" count here, even
     * though we might later be considering a parameterized inner path with
     * fewer rows.  This is because we have included all the join clauses in
     * the selectivity estimate.
     */
    if (jselec > 0) /* protect against zero divide */
    {
        avgmatch = nselec * innerrel->rows / jselec;
        /* Clamp to sane range */
        avgmatch = Max(1.0, avgmatch);
    } else
        avgmatch = 1.0;

    semifactors->outer_match_frac = jselec;
    semifactors->match_count = avgmatch;
}

/*
 * set_baserel_size_estimates
 *		Set the size estimates for the given base relation.
 *
 * The rel's targetlist and restrictinfo list must have been constructed
 * already, and rel->tuples must be set.
 *
 * We set the following fields of the rel node:
 *	rows: the estimated number of output tuples (after applying
 *		  restriction clauses).
 *	width: the estimated average output tuple width in bytes.
 *	baserestrictcost: estimated cost of evaluating baserestrictinfo clauses.
 */
void set_baserel_size_estimates(PlannerInfo *root, RelOptInfo *rel) {
    double nrows, est_sel;

    /* Should only be applied to base relations */
    Assert(rel->relid > 0);

    est_sel = clauselist_selectivity(
        root, rel->baserestrictinfo, 0, JOIN_INNER, NULL
    );

    nrows = rel->tuples * est_sel;

    rel->rows = clamp_row_est(nrows);

    cost_qual_eval(&rel->baserestrictcost, rel->baserestrictinfo, root);

    set_rel_width(root, rel);

    set_baserel_rows_sample(root, rel, est_sel);
}

/*
 * get_parameterized_baserel_size
 *		Make a size estimate for a parameterized scan of a base relation.
 *
 * 'param_clauses' lists the additional join clauses to be used.
 *
 * set_baserel_size_estimates must have been applied already.
 */
double get_parameterized_baserel_size(
    PlannerInfo *root, RelOptInfo *rel,
    List *param_clauses
) {
    List *allclauses;
    double nrows;

    /*
     * Estimate the number of rows returned by the parameterized scan, knowing
     * that it will apply all the extra join clauses as well as the rel's own
     * restriction clauses.  Note that we force the clauses to be treated as
     * non-join clauses during selectivity estimation.
     */
    allclauses = list_concat_copy(param_clauses, rel->baserestrictinfo);
    nrows = rel->tuples *
            clauselist_selectivity(root,
                                   allclauses,
                                   rel->relid, /* do not use 0! */
                                   JOIN_INNER,
                                   NULL);
    nrows = clamp_row_est(nrows);
    /* For safety, make sure result is not more than the base estimate */
    if (nrows > rel->rows)
        nrows = rel->rows;
    return nrows;
}

/*
 * set_joinrel_size_estimates
 *		Set the size estimates for the given join relation.
 *
 * The rel's targetlist must have been constructed already, and a
 * restriction clause list that matches the given component rels must
 * be provided.
 *
 * Since there is more than one way to make a joinrel for more than two
 * base relations, the results we get here could depend on which component
 * rel pair is provided.  In theory we should get the same answers no matter
 * which pair is provided; in practice, since the selectivity estimation
 * routines don't handle all cases equally well, we might not.  But there's
 * not much to be done about it.  (Would it make sense to repeat the
 * calculations for each pair of input rels that's encountered, and somehow
 * average the results?  Probably way more trouble than it's worth, and
 * anyway we must keep the rowcount estimate the same for all paths for the
 * joinrel.)
 *
 * We set only the rows field here.  The reltarget field was already set by
 * build_joinrel_tlist, and baserestrictcost is not used for join rels.
 */
void set_joinrel_size_estimates(
    PlannerInfo *root, RelOptInfo *rel,
    RelOptInfo *outer_rel,
    RelOptInfo *inner_rel,
    SpecialJoinInfo *sjinfo,
    List *restrictlist
) {
    rel->rows = calc_joinrel_size_estimate(
        root,
        rel,
        outer_rel,
        inner_rel,
        outer_rel->rows,
        inner_rel->rows,
        sjinfo,
        restrictlist
    );

    /* Estimated selectivity for conditioning p(true_sel | est_sel=e0).
     * Notes: we assume that the join type is always JOIN_INNER. */
    const double sel_est = clauselist_selectivity(
        root,
        restrictlist,
        0,
        sjinfo->jointype,
        sjinfo
    );

    set_joinrel_rows_sample(
        root, rel, outer_rel, inner_rel, restrictlist, sel_est
    );

    if (root->pass == 1) {
        /* Overwrite: we only need rows at a particular sample point. */
        Assert(rel->rows_sample != NULL);
        if (rel->rows_sample->sample_count != 1) {
            Assert(rel->rows_sample->sample_count == error_sample_count);
            rel->rows = rel->rows_sample->sample[root->round];
        }
    }
}

/*
 * get_parameterized_joinrel_size
 *		Make a size estimate for a parameterized scan of a join relation.
 *
 * 'rel' is the joinrel under consideration.
 * 'outer_path', 'inner_path' are (probably also parameterized) Paths that
 *		produce the relations being joined.
 * 'sjinfo' is any SpecialJoinInfo relevant to this join.
 * 'restrict_clauses' lists the join clauses that need to be applied at the
 * join node (including any movable clauses that were moved down to this join,
 * and not including any movable clauses that were pushed down into the
 * child paths).
 *
 * set_joinrel_size_estimates must have been applied already.
 */
double get_parameterized_joinrel_size(
    PlannerInfo *root, RelOptInfo *rel,
    Path *outer_path,
    Path *inner_path,
    SpecialJoinInfo *sjinfo,
    List *restrict_clauses
) {
    double nrows;

    /*
     * Estimate the number of rows returned by the parameterized join as the
     * sizes of the input paths times the selectivity of the clauses that have
     * ended up at this join node.
     *
     * As with set_joinrel_size_estimates, the rowcount estimate could depend
     * on the pair of input paths provided, though ideally we'd get the same
     * estimate for any pair with the same parameterization.
     */
    nrows = calc_joinrel_size_estimate(
        root,
        rel,
        outer_path->parent,
        inner_path->parent,
        outer_path->rows,
        inner_path->rows,
        sjinfo,
        restrict_clauses
    );
    /* For safety, make sure result is not more than the base estimate */
    if (nrows > rel->rows)
        nrows = rel->rows;
    return nrows;
}

/*
 * calc_joinrel_size_estimate
 *		Workhorse for set_joinrel_size_estimates and
 *		get_parameterized_joinrel_size.
 *
 * outer_rel/inner_rel are the relations being joined, but they should be
 * assumed to have sizes outer_rows/inner_rows; those numbers might be less
 * than what rel->rows says, when we are considering parameterized paths.
 */
static double calc_joinrel_size_estimate(
    PlannerInfo *root,
    RelOptInfo *joinrel,
    RelOptInfo *outer_rel,
    RelOptInfo *inner_rel,
    double outer_rows,
    double inner_rows,
    SpecialJoinInfo *sjinfo,
    List *restrictlist
) {
    JoinType jointype = sjinfo->jointype;
    Selectivity fkselec;
    Selectivity jselec;
    Selectivity pselec;
    double nrows;

    /*
     * Compute joinclause selectivity.  Note that we are only considering
     * clauses that become restriction clauses at this join level; we are not
     * double-counting them because they were not considered in estimating the
     * sizes of the component rels.
     *
     * First, see whether any of the joinclauses can be matched to known FK
     * constraints.  If so, drop those clauses from the restrictlist, and
     * instead estimate their selectivity using FK semantics.  (We do this
     * without regard to whether said clauses are local or "pushed down".
     * Probably, an FK-matching clause could never be seen as pushed down at
     * an outer join, since it would be strict and hence would be grounds for
     * join strength reduction.)  fkselec gets the net selectivity for
     * FK-matching clauses, or 1.0 if there are none.
     */
    fkselec = get_foreign_key_join_selectivity(
        root,
        outer_rel->relids,
        inner_rel->relids,
        sjinfo,
        &restrictlist
    );

    /*
     * For an outer join, we have to distinguish the selectivity of the join's
     * own clauses (JOIN/ON conditions) from any clauses that were "pushed
     * down".  For inner joins we just count them all as joinclauses.
     */
    if (IS_OUTER_JOIN(jointype)) {
        List *joinquals = NIL;
        List *pushedquals = NIL;
        ListCell *l;

        /* Grovel through the clauses to separate into two lists */
        foreach(l, restrictlist) {
            RestrictInfo *rinfo = lfirst_node(RestrictInfo, l);

            if (RINFO_IS_PUSHED_DOWN(rinfo, joinrel->relids))
                pushedquals = lappend(pushedquals, rinfo);
            else
                joinquals = lappend(joinquals, rinfo);
        }

        /* Get the separate selectivities */
        jselec = clauselist_selectivity(
            root,
            joinquals,
            0,
            jointype,
            sjinfo
        );
        pselec = clauselist_selectivity(
            root,
            pushedquals,
            0,
            jointype,
            sjinfo
        );

        /* Avoid leaking a lot of ListCells */
        list_free(joinquals);
        list_free(pushedquals);
    } else {
        jselec = clauselist_selectivity(
            root,
            restrictlist,
            0,
            jointype,
            sjinfo
        );
        pselec = 0.0; /* not used, keep compiler quiet */
    }

    /*
     * Basically, we multiply size of Cartesian product by selectivity.
     *
     * If we are doing an outer join, take that into account: the joinqual
     * selectivity has to be clamped using the knowledge that the output must
     * be at least as large as the non-nullable input.  However, any
     * pushed-down quals are applied after the outer join, so their
     * selectivity applies fully.
     *
     * For JOIN_SEMI and JOIN_ANTI, the selectivity is defined as the fraction
     * of LHS rows that have matches, and we apply that straightforwardly.
     */
    switch (jointype) {
        case JOIN_INNER:
            nrows = outer_rows * inner_rows * fkselec * jselec;
            /* pselec not used */
            break;
        case JOIN_LEFT:
            nrows = outer_rows * inner_rows * fkselec * jselec;
            if (nrows < outer_rows)
                nrows = outer_rows;
            nrows *= pselec;
            break;
        case JOIN_FULL:
            nrows = outer_rows * inner_rows * fkselec * jselec;
            if (nrows < outer_rows)
                nrows = outer_rows;
            if (nrows < inner_rows)
                nrows = inner_rows;
            nrows *= pselec;
            break;
        case JOIN_SEMI:
            nrows = outer_rows * fkselec * jselec;
            /* pselec not used */
            break;
        case JOIN_ANTI:
            nrows = outer_rows * (1.0 - fkselec * jselec);
            nrows *= pselec;
            break;
        default:
            /* other values not expected here */
            elog(ERROR, "unrecognized join type: %d", (int) jointype);
            nrows = 0; /* keep compiler quiet */
            break;
    }

    return clamp_row_est(nrows);
}

/*
 * get_foreign_key_join_selectivity
 *		Estimate join selectivity for foreign-key-related clauses.
 *
 * Remove any clauses that can be matched to FK constraints from *restrictlist,
 * and return a substitute estimate of their selectivity.  1.0 is returned
 * when there are no such clauses.
 *
 * The reason for treating such clauses specially is that we can get better
 * estimates this way than by relying on clauselist_selectivity(), especially
 * for multi-column FKs where that function's assumption that the clauses are
 * independent falls down badly.  But even with single-column FKs, we may be
 * able to get a better answer when the pg_statistic stats are missing or out
 * of date.
 */
static Selectivity get_foreign_key_join_selectivity(
    PlannerInfo *root,
    Relids outer_relids,
    Relids inner_relids,
    SpecialJoinInfo *sjinfo,
    List **restrictlist
) {
    Selectivity fkselec = 1.0;
    JoinType jointype = sjinfo->jointype;
    List *worklist = *restrictlist;
    ListCell *lc;

    /* Consider each FK constraint that is known to match the query */
    foreach(lc, root->fkey_list) {
        ForeignKeyOptInfo *fkinfo = (ForeignKeyOptInfo *) lfirst(lc);
        bool ref_is_outer;
        List *removedlist;
        ListCell *cell;

        /*
         * This FK is not relevant unless it connects a baserel on one side of
         * this join to a baserel on the other side.
         */
        if (bms_is_member(fkinfo->con_relid, outer_relids) &&
            bms_is_member(fkinfo->ref_relid, inner_relids))
            ref_is_outer = false;
        else if (bms_is_member(fkinfo->ref_relid, outer_relids) &&
                 bms_is_member(fkinfo->con_relid, inner_relids))
            ref_is_outer = true;
        else
            continue;

        /*
         * If we're dealing with a semi/anti join, and the FK's referenced
         * relation is on the outside, then knowledge of the FK doesn't help
         * us figure out what we need to know (which is the fraction of outer
         * rows that have matches).  On the other hand, if the referenced rel
         * is on the inside, then all outer rows must have matches in the
         * referenced table (ignoring nulls).  But any restriction or join
         * clauses that filter that table will reduce the fraction of matches.
         * We can account for restriction clauses, but it's too hard to guess
         * how many table rows would get through a join that's inside the RHS.
         * Hence, if either case applies, punt and ignore the FK.
         */
        if ((jointype == JOIN_SEMI || jointype == JOIN_ANTI) &&
            (ref_is_outer || bms_membership(inner_relids) != BMS_SINGLETON))
            continue;

        /*
         * Modify the restrictlist by removing clauses that match the FK (and
         * putting them into removedlist instead).  It seems unsafe to modify
         * the originally-passed List structure, so we make a shallow copy the
         * first time through.
         */
        if (worklist == *restrictlist)
            worklist = list_copy(worklist);

        removedlist = NIL;
        foreach(cell, worklist) {
            RestrictInfo *rinfo = (RestrictInfo *) lfirst(cell);
            bool remove_it = false;
            int i;

            /* Drop this clause if it matches any column of the FK */
            for (i = 0; i < fkinfo->nkeys; i++) {
                if (rinfo->parent_ec) {
                    /*
                     * EC-derived clauses can only match by EC.  It is okay to
                     * consider any clause derived from the same EC as
                     * matching the FK: even if equivclass.c chose to generate
                     * a clause equating some other pair of Vars, it could
                     * have generated one equating the FK's Vars.  So for
                     * purposes of estimation, we can act as though it did so.
                     *
                     * Note: checking parent_ec is a bit of a cheat because
                     * there are EC-derived clauses that don't have parent_ec
                     * set; but such clauses must compare expressions that
                     * aren't just Vars, so they cannot match the FK anyway.
                     */
                    if (fkinfo->eclass[i] == rinfo->parent_ec) {
                        remove_it = true;
                        break;
                    }
                } else {
                    /*
                     * Otherwise, see if rinfo was previously matched to FK as
                     * a "loose" clause.
                     */
                    if (list_member_ptr(fkinfo->rinfos[i], rinfo)) {
                        remove_it = true;
                        break;
                    }
                }
            }
            if (remove_it) {
                worklist = foreach_delete_current(worklist, cell);
                removedlist = lappend(removedlist, rinfo);
            }
        }

        /*
         * If we failed to remove all the matching clauses we expected to
         * find, chicken out and ignore this FK; applying its selectivity
         * might result in double-counting.  Put any clauses we did manage to
         * remove back into the worklist.
         *
         * Since the matching clauses are known not outerjoin-delayed, they
         * would normally have appeared in the initial joinclause list.  If we
         * didn't find them, there are two possibilities:
         *
         * 1. If the FK match is based on an EC that is ec_has_const, it won't
         * have generated any join clauses at all.  We discount such ECs while
         * checking to see if we have "all" the clauses.  (Below, we'll adjust
         * the selectivity estimate for this case.)
         *
         * 2. The clauses were matched to some other FK in a previous
         * iteration of this loop, and thus removed from worklist.  (A likely
         * case is that two FKs are matched to the same EC; there will be only
         * one EC-derived clause in the initial list, so the first FK will
         * consume it.)  Applying both FKs' selectivity independently risks
         * underestimating the join size; in particular, this would undo one
         * of the main things that ECs were invented for, namely to avoid
         * double-counting the selectivity of redundant equality conditions.
         * Later we might think of a reasonable way to combine the estimates,
         * but for now, just punt, since this is a fairly uncommon situation.
         */
        if (removedlist == NIL ||
            list_length(removedlist) !=
            (fkinfo->nmatched_ec - fkinfo->nconst_ec + fkinfo->nmatched_ri)) {
            worklist = list_concat(worklist, removedlist);
            continue;
        }

        /*
         * Finally we get to the payoff: estimate selectivity using the
         * knowledge that each referencing row will match exactly one row in
         * the referenced table.
         *
         * XXX that's not true in the presence of nulls in the referencing
         * column(s), so in principle we should derate the estimate for those.
         * However (1) if there are any strict restriction clauses for the
         * referencing column(s) elsewhere in the query, derating here would
         * be double-counting the null fraction, and (2) it's not very clear
         * how to combine null fractions for multiple referencing columns. So
         * we do nothing for now about correcting for nulls.
         *
         * XXX another point here is that if either side of an FK constraint
         * is an inheritance parent, we estimate as though the constraint
         * covers all its children as well.  This is not an unreasonable
         * assumption for a referencing table, ie the user probably applied
         * identical constraints to all child tables (though perhaps we ought
         * to check that).  But it's not possible to have done that for a
         * referenced table.  Fortunately, precisely because that doesn't
         * work, it is uncommon in practice to have an FK referencing a parent
         * table.  So, at least for now, disregard inheritance here.
         */
        if (jointype == JOIN_SEMI || jointype == JOIN_ANTI) {
            /*
             * For JOIN_SEMI and JOIN_ANTI, we only get here when the FK's
             * referenced table is exactly the inside of the join.  The join
             * selectivity is defined as the fraction of LHS rows that have
             * matches.  The FK implies that every LHS row has a match *in the
             * referenced table*; but any restriction clauses on it will
             * reduce the number of matches.  Hence we take the join
             * selectivity as equal to the selectivity of the table's
             * restriction clauses, which is rows / tuples; but we must guard
             * against tuples == 0.
             */
            RelOptInfo *ref_rel = find_base_rel(root, fkinfo->ref_relid);
            double ref_tuples = Max(ref_rel->tuples, 1.0);

            fkselec *= ref_rel->rows / ref_tuples;
        } else {
            /*
             * Otherwise, selectivity is exactly 1/referenced-table-size; but
             * guard against tuples == 0.  Note we should use the raw table
             * tuple count, not any estimate of its filtered or joined size.
             */
            RelOptInfo *ref_rel = find_base_rel(root, fkinfo->ref_relid);
            double ref_tuples = Max(ref_rel->tuples, 1.0);

            fkselec *= 1.0 / ref_tuples;
        }

        /*
         * If any of the FK columns participated in ec_has_const ECs, then
         * equivclass.c will have generated "var = const" restrictions for
         * each side of the join, thus reducing the sizes of both input
         * relations.  Taking the fkselec at face value would amount to
         * double-counting the selectivity of the constant restriction for the
         * referencing Var.  Hence, look for the restriction clause(s) that
         * were applied to the referencing Var(s), and divide out their
         * selectivity to correct for this.
         */
        if (fkinfo->nconst_ec > 0) {
            for (int i = 0; i < fkinfo->nkeys; i++) {
                EquivalenceClass *ec = fkinfo->eclass[i];

                if (ec && ec->ec_has_const) {
                    EquivalenceMember *em = fkinfo->fk_eclass_member[i];
                    RestrictInfo *rinfo = find_derived_clause_for_ec_member(ec,
                                                                            em);

                    if (rinfo) {
                        Selectivity s0;

                        s0 = clause_selectivity(root,
                                                (Node *) rinfo,
                                                0,
                                                jointype,
                                                sjinfo);
                        if (s0 > 0)
                            fkselec /= s0;
                    }
                }
            }
        }
    }

    *restrictlist = worklist;
    CLAMP_PROBABILITY(fkselec);
    return fkselec;
}

/*
 * set_subquery_size_estimates
 *		Set the size estimates for a base relation that is a subquery.
 *
 * The rel's targetlist and restrictinfo list must have been constructed
 * already, and the Paths for the subquery must have been completed.
 * We look at the subquery's PlannerInfo to extract data.
 *
 * We set the same fields as set_baserel_size_estimates.
 */
void set_subquery_size_estimates(PlannerInfo *root, RelOptInfo *rel) {
    PlannerInfo *subroot = rel->subroot;
    RelOptInfo *sub_final_rel;
    ListCell *lc;

    /* Should only be applied to base relations that are subqueries */
    Assert(rel->relid > 0);
    Assert(planner_rt_fetch(rel->relid, root)->rtekind == RTE_SUBQUERY);

    /*
     * Copy raw number of output rows from subquery.  All of its paths should
     * have the same output rowcount, so just look at cheapest-total.
     */
    sub_final_rel = fetch_upper_rel(subroot, UPPERREL_FINAL, NULL);
    rel->tuples = sub_final_rel->cheapest_total_path->rows;

    /*
     * Compute per-output-column width estimates by examining the subquery's
     * targetlist.  For any output that is a plain Var, get the width estimate
     * that was made while planning the subquery.  Otherwise, we leave it to
     * set_rel_width to fill in a datatype-based default estimate.
     */
    foreach(lc, subroot->parse->targetList) {
        TargetEntry *te = lfirst_node(TargetEntry, lc);
        Node *texpr = (Node *) te->expr;
        int32 item_width = 0;

        /* junk columns aren't visible to upper query */
        if (te->resjunk)
            continue;

        /*
         * The subquery could be an expansion of a view that's had columns
         * added to it since the current query was parsed, so that there are
         * non-junk tlist columns in it that don't correspond to any column
         * visible at our query level.  Ignore such columns.
         */
        if (te->resno < rel->min_attr || te->resno > rel->max_attr)
            continue;

        /*
         * XXX This currently doesn't work for subqueries containing set
         * operations, because the Vars in their tlists are bogus references
         * to the first leaf subquery, which wouldn't give the right answer
         * even if we could still get to its PlannerInfo.
         *
         * Also, the subquery could be an appendrel for which all branches are
         * known empty due to constraint exclusion, in which case
         * set_append_rel_pathlist will have left the attr_widths set to zero.
         *
         * In either case, we just leave the width estimate zero until
         * set_rel_width fixes it.
         */
        if (IsA(texpr, Var) &&
            subroot->parse->setOperations == NULL) {
            Var *var = (Var *) texpr;
            RelOptInfo *subrel = find_base_rel(subroot, var->varno);

            item_width = subrel->attr_widths[var->varattno - subrel->min_attr];
        }
        rel->attr_widths[te->resno - rel->min_attr] = item_width;
    }

    /* Now estimate number of output rows, etc */
    set_baserel_size_estimates(root, rel);
}

/*
 * set_function_size_estimates
 *		Set the size estimates for a base relation that is a function call.
 *
 * The rel's targetlist and restrictinfo list must have been constructed
 * already.
 *
 * We set the same fields as set_baserel_size_estimates.
 */
void set_function_size_estimates(PlannerInfo *root, RelOptInfo *rel) {
    RangeTblEntry *rte;
    ListCell *lc;

    /* Should only be applied to base relations that are functions */
    Assert(rel->relid > 0);
    rte = planner_rt_fetch(rel->relid, root);
    Assert(rte->rtekind == RTE_FUNCTION);

    /*
     * Estimate number of rows the functions will return. The rowcount of the
     * node is that of the largest function result.
     */
    rel->tuples = 0;
    foreach(lc, rte->functions) {
        RangeTblFunction *rtfunc = (RangeTblFunction *) lfirst(lc);
        double ntup = expression_returns_set_rows(root, rtfunc->funcexpr);

        if (ntup > rel->tuples)
            rel->tuples = ntup;
    }

    /* Now estimate number of output rows, etc */
    set_baserel_size_estimates(root, rel);
}

/*
 * set_function_size_estimates
 *		Set the size estimates for a base relation that is a function call.
 *
 * The rel's targetlist and restrictinfo list must have been constructed
 * already.
 *
 * We set the same fields as set_tablefunc_size_estimates.
 */
void set_tablefunc_size_estimates(PlannerInfo *root, RelOptInfo *rel) {
    /* Should only be applied to base relations that are functions */
    Assert(rel->relid > 0);
    Assert(planner_rt_fetch(rel->relid, root)->rtekind == RTE_TABLEFUNC);

    rel->tuples = 100;

    /* Now estimate number of output rows, etc */
    set_baserel_size_estimates(root, rel);
}

/*
 * set_values_size_estimates
 *		Set the size estimates for a base relation that is a values list.
 *
 * The rel's targetlist and restrictinfo list must have been constructed
 * already.
 *
 * We set the same fields as set_baserel_size_estimates.
 */
void set_values_size_estimates(PlannerInfo *root, RelOptInfo *rel) {
    RangeTblEntry *rte;

    /* Should only be applied to base relations that are values lists */
    Assert(rel->relid > 0);
    rte = planner_rt_fetch(rel->relid, root);
    Assert(rte->rtekind == RTE_VALUES);

    /*
     * Estimate number of rows the values list will return. We know this
     * precisely based on the list length (well, barring set-returning
     * functions in list items, but that's a refinement not catered for
     * anywhere else either).
     */
    rel->tuples = list_length(rte->values_lists);

    /* Now estimate number of output rows, etc */
    set_baserel_size_estimates(root, rel);
}

/*
 * set_cte_size_estimates
 *		Set the size estimates for a base relation that is a CTE reference.
 *
 * The rel's targetlist and restrictinfo list must have been constructed
 * already, and we need an estimate of the number of rows returned by the CTE
 * (if a regular CTE) or the non-recursive term (if a self-reference).
 *
 * We set the same fields as set_baserel_size_estimates.
 */
void set_cte_size_estimates(PlannerInfo *root, RelOptInfo *rel, double cte_rows) {
    RangeTblEntry *rte;

    /* Should only be applied to base relations that are CTE references */
    Assert(rel->relid > 0);
    rte = planner_rt_fetch(rel->relid, root);
    Assert(rte->rtekind == RTE_CTE);

    if (rte->self_reference) {
        /*
         * In a self-reference, we assume the average worktable size is a
         * multiple of the nonrecursive term's size.  The best multiplier will
         * vary depending on query "fan-out", so make its value adjustable.
         */
        rel->tuples = clamp_row_est(recursive_worktable_factor * cte_rows);
    } else {
        /* Otherwise just believe the CTE's rowcount estimate */
        rel->tuples = cte_rows;
    }

    /* Now estimate number of output rows, etc */
    set_baserel_size_estimates(root, rel);
}

/*
 * set_namedtuplestore_size_estimates
 *		Set the size estimates for a base relation that is a tuplestore reference.
 *
 * The rel's targetlist and restrictinfo list must have been constructed
 * already.
 *
 * We set the same fields as set_baserel_size_estimates.
 */
void set_namedtuplestore_size_estimates(PlannerInfo *root, RelOptInfo *rel) {
    RangeTblEntry *rte;

    /* Should only be applied to base relations that are tuplestore references */
    Assert(rel->relid > 0);
    rte = planner_rt_fetch(rel->relid, root);
    Assert(rte->rtekind == RTE_NAMEDTUPLESTORE);

    /*
     * Use the estimate provided by the code which is generating the named
     * tuplestore.  In some cases, the actual number might be available; in
     * others the same plan will be re-used, so a "typical" value might be
     * estimated and used.
     */
    rel->tuples = rte->enrtuples;
    if (rel->tuples < 0)
        rel->tuples = 1000;

    /* Now estimate number of output rows, etc */
    set_baserel_size_estimates(root, rel);
}

/*
 * set_result_size_estimates
 *		Set the size estimates for an RTE_RESULT base relation
 *
 * The rel's targetlist and restrictinfo list must have been constructed
 * already.
 *
 * We set the same fields as set_baserel_size_estimates.
 */
void set_result_size_estimates(PlannerInfo *root, RelOptInfo *rel) {
    /* Should only be applied to RTE_RESULT base relations */
    Assert(rel->relid > 0);
    Assert(planner_rt_fetch(rel->relid, root)->rtekind == RTE_RESULT);

    /* RTE_RESULT always generates a single row, natively */
    rel->tuples = 1;

    /* Now estimate number of output rows, etc */
    set_baserel_size_estimates(root, rel);
}

/*
 * set_foreign_size_estimates
 *		Set the size estimates for a base relation that is a foreign table.
 *
 * There is not a whole lot that we can do here; the foreign-data wrapper
 * is responsible for producing useful estimates.  We can do a decent job
 * of estimating baserestrictcost, so we set that, and we also set up width
 * using what will be purely datatype-driven estimates from the targetlist.
 * There is no way to do anything sane with the rows value, so we just put
 * a default estimate and hope that the wrapper can improve on it.  The
 * wrapper's GetForeignRelSize function will be called momentarily.
 *
 * The rel's targetlist and restrictinfo list must have been constructed
 * already.
 */
void set_foreign_size_estimates(PlannerInfo *root, RelOptInfo *rel) {
    /* Should only be applied to base relations */
    Assert(rel->relid > 0);

    rel->rows = 1000; /* entirely bogus default estimate */

    cost_qual_eval(&rel->baserestrictcost, rel->baserestrictinfo, root);

    set_rel_width(root, rel);
}


/*
 * set_rel_width
 *		Set the estimated output width of a base relation.
 *
 * The estimated output width is the sum of the per-attribute width estimates
 * for the actually-referenced columns, plus any PHVs or other expressions
 * that have to be calculated at this relation.  This is the amount of data
 * we'd need to pass upwards in case of a sort, hash, etc.
 *
 * This function also sets reltarget->cost, so it's a bit misnamed now.
 *
 * NB: this works best on plain relations because it prefers to look at
 * real Vars.  For subqueries, set_subquery_size_estimates will already have
 * copied up whatever per-column estimates were made within the subquery,
 * and for other types of rels there isn't much we can do anyway.  We fall
 * back on (fairly stupid) datatype-based width estimates if we can't get
 * any better number.
 *
 * The per-attribute width estimates are cached for possible re-use while
 * building join relations or post-scan/join pathtargets.
 */
static void set_rel_width(PlannerInfo *root, RelOptInfo *rel) {
    Oid reloid = planner_rt_fetch(rel->relid, root)->relid;
    int32 tuple_width = 0;
    bool have_wholerow_var = false;
    ListCell *lc;

    /* Vars are assumed to have cost zero, but other exprs do not */
    rel->reltarget->cost.startup = 0;
    rel->reltarget->cost.per_tuple = 0;

    foreach(lc, rel->reltarget->exprs) {
        Node *node = (Node *) lfirst(lc);

        /*
         * Ordinarily, a Var in a rel's targetlist must belong to that rel;
         * but there are corner cases involving LATERAL references where that
         * isn't so.  If the Var has the wrong varno, fall through to the
         * generic case (it doesn't seem worth the trouble to be any smarter).
         */
        if (IsA(node, Var) &&
            ((Var *) node)->varno == rel->relid) {
            Var *var = (Var *) node;
            int ndx;
            int32 item_width;

            Assert(var->varattno >= rel->min_attr);
            Assert(var->varattno <= rel->max_attr);

            ndx = var->varattno - rel->min_attr;

            /*
             * If it's a whole-row Var, we'll deal with it below after we have
             * already cached as many attr widths as possible.
             */
            if (var->varattno == 0) {
                have_wholerow_var = true;
                continue;
            }

            /*
             * The width may have been cached already (especially if it's a
             * subquery), so don't duplicate effort.
             */
            if (rel->attr_widths[ndx] > 0) {
                tuple_width += rel->attr_widths[ndx];
                continue;
            }

            /* Try to get column width from statistics */
            if (reloid != InvalidOid && var->varattno > 0) {
                item_width = get_attavgwidth(reloid, var->varattno);
                if (item_width > 0) {
                    rel->attr_widths[ndx] = item_width;
                    tuple_width += item_width;
                    continue;
                }
            }

            /*
             * Not a plain relation, or can't find statistics for it. Estimate
             * using just the type info.
             */
            item_width = get_typavgwidth(var->vartype, var->vartypmod);
            Assert(item_width > 0);
            rel->attr_widths[ndx] = item_width;
            tuple_width += item_width;
        } else if (IsA(node, PlaceHolderVar)) {
            /*
             * We will need to evaluate the PHV's contained expression while
             * scanning this rel, so be sure to include it in reltarget->cost.
             */
            PlaceHolderVar *phv = (PlaceHolderVar *) node;
            PlaceHolderInfo *phinfo = find_placeholder_info(root, phv);
            QualCost cost;

            tuple_width += phinfo->ph_width;
            cost_qual_eval_node(&cost, (Node *) phv->phexpr, root);
            rel->reltarget->cost.startup += cost.startup;
            rel->reltarget->cost.per_tuple += cost.per_tuple;
        } else {
            /*
             * We could be looking at an expression pulled up from a subquery,
             * or a ROW() representing a whole-row child Var, etc.  Do what we
             * can using the expression type information.
             */
            int32 item_width;
            QualCost cost;

            item_width = get_typavgwidth(exprType(node), exprTypmod(node));
            Assert(item_width > 0);
            tuple_width += item_width;
            /* Not entirely clear if we need to account for cost, but do so */
            cost_qual_eval_node(&cost, node, root);
            rel->reltarget->cost.startup += cost.startup;
            rel->reltarget->cost.per_tuple += cost.per_tuple;
        }
    }

    /*
     * If we have a whole-row reference, estimate its width as the sum of
     * per-column widths plus heap tuple header overhead.
     */
    if (have_wholerow_var) {
        int32 wholerow_width = MAXALIGN(SizeofHeapTupleHeader);

        if (reloid != InvalidOid) {
            /* Real relation, so estimate true tuple width */
            wholerow_width += get_relation_data_width(reloid,
                                                      rel->attr_widths - rel->min_attr);
        } else {
            /* Do what we can with info for a phony rel */
            AttrNumber i;

            for (i = 1; i <= rel->max_attr; i++)
                wholerow_width += rel->attr_widths[i - rel->min_attr];
        }

        rel->attr_widths[0 - rel->min_attr] = wholerow_width;

        /*
         * Include the whole-row Var as part of the output tuple.  Yes, that
         * really is what happens at runtime.
         */
        tuple_width += wholerow_width;
    }

    Assert(tuple_width >= 0);
    rel->reltarget->width = tuple_width;
}

/*
 * set_pathtarget_cost_width
 *		Set the estimated eval cost and output width of a PathTarget tlist.
 *
 * As a notational convenience, returns the same PathTarget pointer passed in.
 *
 * Most, though not quite all, uses of this function occur after we've run
 * set_rel_width() for base relations; so we can usually obtain cached width
 * estimates for Vars.  If we can't, fall back on datatype-based width
 * estimates.  Present early-planning uses of PathTargets don't need accurate
 * widths badly enough to justify going to the catalogs for better data.
 */
PathTarget *set_pathtarget_cost_width(PlannerInfo *root, PathTarget *target) {
    int32 tuple_width = 0;
    ListCell *lc;

    /* Vars are assumed to have cost zero, but other exprs do not */
    target->cost.startup = 0;
    target->cost.per_tuple = 0;

    foreach(lc, target->exprs) {
        Node *node = (Node *) lfirst(lc);

        tuple_width += get_expr_width(root, node);

        /* For non-Vars, account for evaluation cost */
        if (!IsA(node, Var)) {
            QualCost cost;

            cost_qual_eval_node(&cost, node, root);
            target->cost.startup += cost.startup;
            target->cost.per_tuple += cost.per_tuple;
        }
    }

    Assert(tuple_width >= 0);
    target->width = tuple_width;

    return target;
}

/*
 * compute_bitmap_pages
 *
 * compute number of pages fetched from heap in bitmap heap scan.
 */
double compute_bitmap_pages(PlannerInfo *root, RelOptInfo *baserel, Path *bitmapqual,
                            int loop_count, Cost *cost, double *tuple) {
    Cost indexTotalCost;
    Selectivity indexSelectivity;
    double T;
    double pages_fetched;
    double tuples_fetched;
    double heap_pages;
    long maxentries;

    /*
     * Fetch total cost of obtaining the bitmap, as well as its total
     * selectivity.
     */
    cost_bitmap_tree_node(bitmapqual, &indexTotalCost, &indexSelectivity);

    /*
     * Estimate number of main-table pages fetched.
     */
    tuples_fetched = clamp_row_est(indexSelectivity * baserel->tuples);

    T = (baserel->pages > 1) ? (double) baserel->pages : 1.0;

    /*
     * For a single scan, the number of heap pages that need to be fetched is
     * the same as the Mackert and Lohman formula for the case T <= b (ie, no
     * re-reads needed).
     */
    pages_fetched = (2.0 * T * tuples_fetched) / (2.0 * T + tuples_fetched);

    /*
     * Calculate the number of pages fetched from the heap.  Then based on
     * current work_mem estimate get the estimated maxentries in the bitmap.
     * (Note that we always do this calculation based on the number of pages
     * that would be fetched in a single iteration, even if loop_count > 1.
     * That's correct, because only that number of entries will be stored in
     * the bitmap at one time.)
     */
    heap_pages = Min(pages_fetched, baserel->pages);
    maxentries = tbm_calculate_entries(work_mem * 1024L);

    if (loop_count > 1) {
        /*
         * For repeated bitmap scans, scale up the number of tuples fetched in
         * the Mackert and Lohman formula by the number of scans, so that we
         * estimate the number of pages fetched by all the scans. Then
         * pro-rate for one scan.
         */
        pages_fetched = index_pages_fetched(tuples_fetched * loop_count,
                                            baserel->pages,
                                            get_indexpath_pages(bitmapqual),
                                            root);
        pages_fetched /= loop_count;
    }

    if (pages_fetched >= T)
        pages_fetched = T;
    else
        pages_fetched = ceil(pages_fetched);

    if (maxentries < heap_pages) {
        double exact_pages;
        double lossy_pages;

        /*
         * Crude approximation of the number of lossy pages.  Because of the
         * way tbm_lossify() is coded, the number of lossy pages increases
         * very sharply as soon as we run short of memory; this formula has
         * that property and seems to perform adequately in testing, but it's
         * possible we could do better somehow.
         */
        lossy_pages = Max(0, heap_pages - maxentries / 2);
        exact_pages = heap_pages - lossy_pages;

        /*
         * If there are lossy pages then recompute the number of tuples
         * processed by the bitmap heap node.  We assume here that the chance
         * of a given tuple coming from an exact page is the same as the
         * chance that a given page is exact.  This might not be true, but
         * it's not clear how we can do any better.
         */
        if (lossy_pages > 0)
            tuples_fetched =
                    clamp_row_est(indexSelectivity *
                                  (exact_pages / heap_pages) * baserel->tuples +
                                  (lossy_pages / heap_pages) * baserel->tuples);
    }

    if (cost)
        *cost = indexTotalCost;
    if (tuple)
        *tuple = tuples_fetched;

    return pages_fetched;
}
