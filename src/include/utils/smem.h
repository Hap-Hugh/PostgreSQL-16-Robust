//
// Created by Xuan Chen on 2025/9/22.
// Modified by Xuan Chen on 2025/9/24.
// Modified by Xuan Chen on 2025/10/2.
//

#ifndef SMEM_H
#define SMEM_H

#include "hsearch.h"
#include "palloc.h"
#include "optimizer/ep.h"

#define SM_KEY_LEN 64

/* GUC Parameters */
extern char *error_profile_path;

/* Forward Declarations */
typedef struct SessionMemKey SessionMemKey;

typedef struct SessionMemEntry SessionMemEntry;

typedef struct SessionMem SessionMem;

typedef struct ErrorProfile ErrorProfile;

/* Definitions */
struct SessionMemKey {
    char key[SM_KEY_LEN];
};

struct SessionMemEntry {
    SessionMemKey key;
    ErrorProfile ep;
};

struct SessionMem {
    HTAB *table;
    MemoryContext mcxt;
    bool initialized;
};

SessionMem *GetSessionMem(void);

void SessionMemInit(void);

bool SessionMemFind(const char *key, ErrorProfile **ep);

bool SessionMemSave(const char *key, const ErrorProfile *ep);

void SessionMemFree(void);

void SessionMemLoadAll(const char *dirname, void *extra);

#endif // SMEM_H
