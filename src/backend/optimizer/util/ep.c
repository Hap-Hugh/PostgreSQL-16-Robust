//
// Created by Xuan Chen on 2025/10/2.
//

#include "optimizer/sample.h"
#include "optimizer/optimizer.h"
#include "utils/smem.h"

/* ------------------------------- Aliases ------------------------------- */
char *get_alias(
    const PlannerInfo *root,
    const Index relid
) {
    Assert(relid > 0);
    const RangeTblEntry *rte = root->simple_rte_array[relid];
    const char *alias = rte->eref->aliasname;
    return pstrdup(alias);
}

char *get_std_alias(
    const PlannerInfo *root,
    const Index relid
) {
    Assert(relid > 0);
    const RangeTblEntry *rte = root->simple_rte_array[relid];
    const char *alias = rte->eref->aliasname;
    char *std_alias = pstrdup(alias);
    char *ch = std_alias;
    while (*ch != '\0') {
        if (*ch >= '0' && *ch <= '9') {
            *ch = '\0'; // *ch is a digit, break now
            break;
        }
        ++ch;
    }
    return std_alias;
}

/* ------------------------------- Error Profiles ------------------------------- */
int read_error_profile(
    const char *filename,
    ErrorProfile *ep
) {
    if (!filename || filename[0] == '\0' || ep == NULL) {
        elog(WARNING, "Invalid filename");
        return -1;
    }

    FILE *fp = fopen(filename, "r");
    if (!fp) {
        elog(WARNING, "Could not open file %s", filename);
        return -2;
    }

    memset(ep, 0, sizeof(ErrorProfile));
    double *sel_true_array = palloc0(sizeof(double) * EP_MAX_SAMPLE);
    double *sel_est_array = palloc0(sizeof(double) * EP_MAX_SAMPLE);

    int sample_count = 0;
    while (true) {
        double sel_true, sel_est;
        const int result = fscanf(fp, "%lf %lf", &sel_true, &sel_est);
        if (result == EOF) {
            break;
        }
        if (result == 2) {
            if (sample_count >= EP_MAX_SAMPLE) {
                break;
            }
            sel_true = clamp01(sel_true);
            sel_est = clamp01(sel_est);
            ep->data[sample_count].sel_true = sel_true;
            ep->data[sample_count].sel_est = sel_est;
            sel_true_array[sample_count] = sel_true;
            sel_est_array[sample_count] = sel_est;
            ++sample_count;
        } else {
            int c;
            do {
                c = fgetc(fp);
            } while (c != '\n' && c != EOF);
        }
    }
    fclose(fp);
    ep->sample_count = sample_count;

    if (sample_count > 0) {
        double mean_est, std_est, mean_true, std_true;
        calc_mean_std(sel_est_array, sample_count, &mean_est, &std_est);
        calc_mean_std(sel_true_array, sample_count, &mean_true, &std_true);
        ep->std_est = std_est;
        ep->std_true = std_true;
    } else {
        ep->std_est = 0.0;
        ep->std_true = 0.0;
    }

    pfree(sel_est_array);
    pfree(sel_true_array);
    return 0;
}

bool get_error_profile(
    const char *alias,
    const char *alias_fallback,
    ErrorProfile **ep
) {
    // Try the alias + suffix first
    char ep_key[SM_KEY_LEN];
    sprintf(ep_key, "%s%s", alias, ".txt");
    bool found = SessionMemFind(ep_key, ep);

    // If we find the error profile, return immediately
    if (ep != NULL && found) {
        return true;
    }

    // Otherwise, fallback to alias_fallback + suffix
    if (strcmp(alias, alias_fallback) == 0) {
        // We check whether the alias and alias fallback is the same
        return false;
    }
    char ep_key_fallback[SM_KEY_LEN];
    sprintf(ep_key_fallback, "%s%s", alias_fallback, ".txt");
    found = SessionMemFind(ep_key_fallback, ep);

    if (ep != NULL && found) {
        return true;
    }
    return false;
}
