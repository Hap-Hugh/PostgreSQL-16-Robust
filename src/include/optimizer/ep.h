//
// Created by Xuan Chen on 2025/10/2.
//

#ifndef EP_H
#define EP_H

#include "postgres.h"
#include "nodes/pathnodes.h"
#include "optimizer/sample.h"

#define EP_MAX_SAMPLE     1024
#define EP_MAX_BIN        8

/* Forward Declarations */
typedef struct Sample Sample;

typedef struct ErrorProfileRaw ErrorProfileRaw;

typedef struct ErrorSampleParams ErrorSampleParams;

typedef struct ErrorProfile ErrorProfile;

struct ErrorProfileRaw {
    double sel_true;
    double sel_est;
};

/* Parameters of each bin’s error sample */
struct ErrorSampleParams {
    int bin_index;
    int n_points; /* number of raw samples in this bin */
    double bandwidth_h; /* KDE bandwidth (if needed later) */
    double mean_logratio; /* mean of log(sel_true/sel_est) */
    double std_logratio; /* std of log(sel_true/sel_est) */
    double sel_est_lo; /* left boundary of sel_est */
    double sel_est_hi; /* right boundary of sel_est */
};

struct ErrorProfile {
    int sample_count;
    double std_true;
    double std_est;
    double error_sample_thresh[EP_MAX_BIN]; /* sel_est threshold of each bin */
    Sample error_sample[EP_MAX_BIN]; /* samples */
    ErrorSampleParams params[EP_MAX_BIN]; /* parameters of each bin */
    ErrorProfileRaw data[EP_MAX_SAMPLE]; /* input raw data */
};


/* ------------------------------- Aliases ------------------------------- */
char *get_alias(
    const PlannerInfo *root,
    Index relid
);

char *get_std_alias(
    const PlannerInfo *root,
    Index relid
);

/* ------------------------------- Error Profiles ------------------------------- */
int read_error_profile(
    const char *filename,
    ErrorProfile *ep
);

bool get_error_profile(
    const char *alias,
    const char *alias_fallback,
    ErrorProfile **ep
);

#endif // EP_H
