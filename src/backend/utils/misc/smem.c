//
// Created by Xuan Chen on 2025/9/22.
// Modified by Xuan Chen on 2025/9/24.
// Modified by Xuan Chen on 2025/10/2.
//

#include "postgres.h"

#include <dirent.h>
#include "utils/memutils.h"
#include "utils/smem.h"
#include "optimizer/kde.h"

/* GUC Parameters */
char *error_profile_path = NULL;

/* Session Memory Pointer */
static SessionMem *session_mem = NULL;

SessionMem *GetSessionMem() {
    if (session_mem == NULL) {
        SessionMemInit();
    }
    return session_mem;
}

void SessionMemInit() {
    if (session_mem != NULL) {
        return;
    }
    MemoryContext old = MemoryContextSwitchTo(TopMemoryContext);

    session_mem = (SessionMem *) palloc0(sizeof(SessionMem));
    session_mem->mcxt = AllocSetContextCreate(
        TopMemoryContext, "SessionMemoryData", ALLOCSET_DEFAULT_SIZES
    );

    HASHCTL hashctl;
    memset(&hashctl, 0, sizeof(HASHCTL));
    hashctl.keysize = sizeof(SessionMemKey);
    hashctl.entrysize = sizeof(SessionMemEntry);
    hashctl.hcxt = session_mem->mcxt;

    session_mem->table = hash_create(
        "SessionMemHashTable", 128, &hashctl,
        HASH_ELEM | HASH_CONTEXT | HASH_BLOBS
    );
    session_mem->initialized = true;

    MemoryContextSwitchTo(old);
    elog(DEBUG1, "Session memory initialized");
}

bool SessionMemFind(const char *key, ErrorProfile **ep) {
    const SessionMem *mem = GetSessionMem();
    if (mem == NULL || !mem->initialized) {
        elog(WARNING, "Session memory not initialized");
        return false;
    }

    SessionMemKey session_mem_key;
    memset(&session_mem_key, 0, sizeof(SessionMemKey));
    strlcpy(session_mem_key.key, key, sizeof(session_mem_key.key));

    SessionMemEntry *entry = hash_search(
        mem->table, &session_mem_key, HASH_FIND, NULL
    );
    if (entry == NULL) {
        elog(WARNING, "Session memory key %s not found", key);
        return false;
    }
    *ep = &entry->ep;
    elog(DEBUG1, "Error profile found, key = %s", key);
    return true;
}

bool SessionMemSave(const char *key, const ErrorProfile *ep) {
    SessionMem *mem = GetSessionMem();
    if (mem == NULL || !mem->initialized) {
        elog(WARNING, "Session memory not initialized");
        return false;
    }

    bool found;
    SessionMemKey session_mem_key;
    memset(&session_mem_key, 0, sizeof(SessionMemKey));
    strlcpy(session_mem_key.key, key, sizeof(session_mem_key.key));

    SessionMemEntry *entry = hash_search(
        mem->table, &session_mem_key, HASH_ENTER, &found
    );
    if (entry == NULL) {
        elog(WARNING, "Session memory key %s upsert failed", key);
        return false;
    }
    memcpy(&entry->ep, ep, sizeof(ErrorProfile));
    elog(DEBUG1, "Error profile saved, key = %s", key);
    return true;
}

void SessionMemFree(void) {
    if (session_mem == NULL) {
        return;
    }
    MemoryContextDelete(session_mem->mcxt);
    session_mem->initialized = false;
    pfree(session_mem);
    session_mem = NULL;
}

void SessionMemLoadAll(const char *dirname, void *extra) {
    (void) extra;

    // Prepare the session memory for saving error profiles
    SessionMemFree();
    SessionMemInit();
    SessionMem *mem = GetSessionMem();
    if (mem == NULL || !mem->initialized) {
        elog(WARNING, "Session memory not initialized");
        return;
    }

    // Check the GUC parameter: dirname
    if (!dirname || *dirname == '\0') {
        elog(WARNING, "Invalid directory");
        return;
    }

    // Open the newly set directory for error profiles
    DIR *dir = opendir(dirname);
    if (!dir) {
        elog(WARNING, "Cannot open directory %s", dirname);
        return;
    }

    // Traverse the error prpfiles in the directory
    int ep_loaded_count = 0;
    struct dirent *dirent;
    while ((dirent = readdir(dir)) != NULL) {
        const char *filename = dirent->d_name;

        // Ignore hidden files
        if (filename[0] == '.') {
            continue;
        }

        // Only accept *.txt files
        const char *dot = strrchr(filename, '.');
        if (!dot || strcmp(dot, ".txt") != 0) {
            continue;
        }

        // Concat the full path for read_error_profile function
        const size_t fullpath_len = strlen(dirname) + strlen(filename) + 2;
        char *fullpath = palloc0(fullpath_len);
        pg_snprintf(fullpath, fullpath_len, "%s/%s", dirname, filename);

        // Read the error profile and cache the ErrorProfile
        elog(LOG, "Loading file %s", fullpath);
        ErrorProfile *ep = palloc0(sizeof(ErrorProfile));
        read_error_profile(fullpath, ep);
        make_error_sample(ep);

        // Save the error profiles to session memory
        SessionMemSave(filename, ep);
        ++ep_loaded_count;

        // Done with current data structure
        pfree(ep);
        pfree(fullpath);
    }
    closedir(dir);

    elog(LOG, "%d error profiles loaded", ep_loaded_count);
}
