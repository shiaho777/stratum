/* Local score-row pointers for one GQA group.
 * Slot i is query head (h0 + i). The absolute head id is not an index:
 * a group with h0 >= 64 (Llama-3.1-405B has 128 query heads) used to
 * write past a 64-slot stack array. nslots is the length of lg and must
 * be at least the group width (h1 - h0), not the absolute head id.
 * Returns 0, or -1 without writing if the group does not fit.
 */
#ifndef STRATUM_ATTN_LG_H
#define STRATUM_ATTN_LG_H

#include <stddef.h>

#define STRATUM_ATTN_LG_SLOTS 256

static inline int stratum_attn_bind_lg(float** lg, int nslots,
                                       float* blg, size_t blg_row,
                                       int h0, int h1) {
    if (!lg || h0 > h1) return -1;
    int n = h1 - h0;
    if (n > nslots || (n > 0 && !blg)) return -1;
    for (int i = 0; i < n; i++)
        lg[i] = blg + (size_t)i * blg_row;
    return 0;
}

#endif
