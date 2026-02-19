//
// Created by Xuan Chen on 2025/9/22.
// Created by Xuan Chen on 2025/9/28.
// Modified by Xuan Chen on 2025/10/2.
//

#ifndef KDE_H
#define KDE_H

#include "optimizer/sample.h"

/* GUC Parameters */
extern double error_sample_kde_bandwidth;
extern int error_bin_count;

/* Forward Declarations */
typedef Sample Sample;

typedef ErrorProfile ErrorProfile;

/* ------------------------------- KDE Estimation ------------------------------- */
void make_error_sample(
    ErrorProfile *ep
);

/* ------------------------------- Sampling ------------------------------- */
int find_bin_by_sel_est(
    const ErrorProfile *ep,
    double sel_est
);

#endif // KDE_H
