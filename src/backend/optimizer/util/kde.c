//
// Created by Xuan Chen on 2025/9/22.
// Modified by Xuan Chen on 2025/10/2.
//

#include "postgres.h"

#include <math.h>
#include <stdlib.h>
#include "optimizer/sample.h"
#include "optimizer/ep.h"
#include "optimizer/kde.h"

static const double EPS = 1e-12;
double error_kde_sigma_span = 4.0;

/* GUC Parameters */
int error_bin_count = 1;
double error_sample_kde_bandwidth = 0.0;

/* ---------------- Utilities ---------------- */

/* Comparator for sorting by sel_est ascending */
static int cmp_sel_est_asc(const void *a, const void *b) {
    const ErrorProfileRaw *pa = a;
    const ErrorProfileRaw *pb = b;
    if (pa->sel_est < pb->sel_est) return -1;
    if (pa->sel_est > pb->sel_est) return 1;
    return 0;
}

/* Safe log ratio: log(max(sel_true,EPS)/max(sel_est,EPS)) */
static double safe_log_ratio(const double sel_true, const double sel_est) {
    const double t = fmax(sel_true, EPS);
    const double e = fmax(sel_est, EPS);
    return log(t / e);
}

/* ---------------- RNG (LCG + Box-Muller) ----------------
   Uses the same LCG update as sample_from_distribution.
   - rng_uniform01(): uniform in [0,1)
   - rng_std_normal(): standard normal N(0,1) via Box–Muller
*/
static unsigned int lcg_next(const unsigned int s) {
    return s * 1664525u + 1013904223u;
}

static double rng_uniform01(unsigned int *seed) {
    /* Generate u in [0,1). We mimic your style (mod 1e6) to keep consistency. */
    const double u = (double) (*seed % 1000000u) / 1000000.0;
    *seed = lcg_next(*seed);
    return u;
}

static double rng_std_normal(unsigned int *seed) {
    /* Box–Muller transform */
    double u1 = rng_uniform01(seed);
    const double u2 = rng_uniform01(seed);
    if (u1 <= 0.0) {
        u1 = 1e-12; /* guard log(0) */
    }
    const double r = sqrt(-2.0 * log(u1));
    const double theta = 2.0 * M_PI * u2;
    return r * cos(theta); /* one normal draw */
}

/* ---------------- KDE helpers (deterministic, no RNG) ---------------- */

/*
 * Draw M samples from a Gaussian KDE defined by data x[0..n-1] and bandwidth h.
 *
 * KDE sampling scheme (mixture-of-Gaussians view):
 *   1) Pick an index i ~ Uniform{0, ..., n-1}
 *   2) Draw Z ~ N(0, 1)
 *   3) Emit y = x[i] + h * Z
 *
 * Output:
 *   - out->sample[k] holds the k-th sampled value
 *   - out->sample_count = number of samples written (<= DIST_MAX_SAMPLE)
 *
 * Notes:
 *   - No probabilities are produced; this function returns *unweighted samples*.
 *   - h <= 0 is clamped to a small positive value to avoid degeneration.
 *   - Uses rng_uniform01(seed) and rng_std_normal(seed) provided elsewhere.
 */
static void discretize_kde_by_sampling(
    const double *x, const int n, double h,
    const int sample_count,
    unsigned int *seed,
    Sample *out
) {
    /* Basic argument checks */
    if (out == NULL || x == NULL || n <= 0 || sample_count <= 0) {
        return;
    }
    /* Initialize output count */
    out->sample_count = 0;
    /* Guard bandwidth to avoid zero or negative scale */
    if (h <= 0.0) {
        h = 1e-6;
    }
    /* Cap to fixed capacity */
    int M = sample_count;
    if (M > DIST_MAX_SAMPLE) {
        M = DIST_MAX_SAMPLE;
    }

    /* Generate M i.i.d. samples from the KDE */
    for (int k = 0; k < M; ++k) {
        /* Choose a component uniformly among the n samples */
        const double u = rng_uniform01(seed);
        int idx = (int) (u * n);
        if (idx >= n) {
            idx = n - 1; /* fencepost guard */
        }

        /* Draw standard normal noise and form the KDE sample */
        const double z = rng_std_normal(seed);
        const double y = x[idx] + h * z;

        /* Store */
        out->sample[k] = y;
    }

    out->sample_count = M;
}

/* ---------------- Build thresholds + params + dists ---------------- */

/* Build per-bin distributions and store thresholds/params into ep. */
void make_error_sample(ErrorProfile *ep) {
    Assert(ep != NULL);
    Assert(ep->sample_count > 0);
    Assert(error_bin_count >= 1 && error_bin_count <= EP_MAX_BIN);

    /* 1) sort by sel_est */
    qsort(ep->data, (size_t)ep->sample_count, sizeof(ErrorProfileRaw), cmp_sel_est_asc);

    const int N = ep->sample_count;
    int start = 0;
    double prev_hi = -INFINITY;

    for (int b = 0; b < error_bin_count; ++b) {
        /* 2) equal-count binning on sel_est */
        int end = (int) llround(((long long) (b + 1) * (long long) N) / (double) error_bin_count);
        if (end > N) {
            end = N;
        }
        if (b == error_bin_count - 1) {
            end = N;
        }
        const int count = end - start;

        const double sel_lo = (b == 0) ? -INFINITY : ep->error_sample_thresh[b - 1];
        const double sel_hi = (count > 0) ? ep->data[end - 1].sel_est : ((b == 0) ? INFINITY : prev_hi);

        ep->error_sample_thresh[b] = sel_hi;
        prev_hi = sel_hi;

        /* 3) collect x = log(sel_true/sel_est) for this bin */
        double *x = NULL;
        if (count > 0) {
            x = (double *) malloc(sizeof(double) * count);
            Assert(x != NULL);
            for (int i = 0; i < count; ++i) {
                const ErrorProfileRaw *s = &ep->data[start + i];
                x[i] = safe_log_ratio(s->sel_true, s->sel_est);
            }
        }

        /* 4) stats of x */
        double mean_x = 0.0, std_x = 0.0;
        if (count > 0) {
            calc_mean_std(x, count, &mean_x, &std_x);
        }

        /* 5) record params */
        const double h = (error_sample_kde_bandwidth > 0.0) ? error_sample_kde_bandwidth : 1e-6;
        ep->params[b].bin_index = b;
        ep->params[b].n_points = count;
        ep->params[b].bandwidth_h = h;
        ep->params[b].mean_logratio = mean_x;
        ep->params[b].std_logratio = std_x;
        ep->params[b].sel_est_lo = sel_lo;
        ep->params[b].sel_est_hi = sel_hi;

        /* 6) build per-bin distribution BY SAMPLING (writes into fixed arrays) */
        ep->error_sample[b].sample_count = 0; /* reset */
        if (error_sample_count > 0 && count > 0) {
            /* decorrelate bins a bit */
            unsigned int bin_seed = error_sample_seed ^ (0x9E3779B9u * (unsigned int) b);
            discretize_kde_by_sampling(
                x, count, h,
                error_sample_count,
                &bin_seed,
                &ep->error_sample[b]
            );
        }

        if (x) free(x);
        start = end;
    }

    /* clear remaining bins (if any) */
    for (int b = error_bin_count; b < EP_MAX_BIN; ++b) {
        ep->error_sample_thresh[b] = NAN;
        ep->params[b].bin_index = -1;
        ep->params[b].n_points = 0;
        ep->params[b].bandwidth_h = 0.0;
        ep->params[b].mean_logratio = 0.0;
        ep->params[b].std_logratio = 0.0;
        ep->params[b].sel_est_lo = NAN;
        ep->params[b].sel_est_hi = NAN;
        ep->error_sample[b].sample_count = 0;
        /* vals/probs arrays remain as-is; sample_count=0 means "empty" */
    }
}

/* ------------------------------- Sampling ------------------------------- */

/* Optional helper: find bin by sel_est using (thresh[b-1], thresh[b]].
   Returns -1 if none found. */
int find_bin_by_sel_est(const ErrorProfile *ep, const double sel_est) {
    if (error_bin_count <= 0) {
        return -1;
    }
    for (int b = 0; b < error_bin_count; ++b) {
        const double lo = (b == 0) ? -INFINITY : ep->error_sample_thresh[b - 1];
        const double hi = ep->error_sample_thresh[b];
        if (sel_est > lo && sel_est <= hi) return b;
    }
    /* fallback: the last non-empty bin */
    for (int b = error_bin_count - 1; b >= 0; --b) {
        if (ep->params[b].n_points > 0) return b;
    }
    return -1;
}
