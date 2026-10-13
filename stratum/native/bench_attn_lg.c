/* Calls the shipped stratum_attn_bind_lg at Nq=80, Nk=8.
 * Group width is 10, so the last groups start at h0 >= 64. The slot
 * count passed in is that width, not 64. The old absolute-head index
 * writes lg[h] with h >= 64 and does not fit in a bgs-length array.
 */
#include "stratum_attn_lg.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const int Nq = 80, Nk = 8;
    const int bgs = Nq / Nk;
    const size_t row = 4;
    int saw_high = 0;
    float* blg = (float*)calloc((size_t)bgs * row, sizeof(float));
    if (!blg || bgs < 1) return 2;
    float* poison = (float*)(uintptr_t)1;

    for (int kv = 0; kv < Nk; kv++) {
        int h0 = kv * bgs;
        int h1 = h0 + bgs;
        if (h0 >= 64) saw_high = 1;
        float* lg[16];
        for (int i = 0; i < 16; i++) lg[i] = poison;
        int rc = stratum_attn_bind_lg(lg, bgs, blg, row, h0, h1);
        if (rc != 0 || lg[0] != blg
            || lg[bgs - 1] != blg + (size_t)(bgs - 1) * row
            || lg[bgs] != poison) {
            printf("attn_lg_mismatch h0=%d rc=%d\n", h0, rc);
            free(blg);
            return 2;
        }
    }
    float* small[4];
    if (stratum_attn_bind_lg(small, 4, blg, row, 70, 80) == 0 || !saw_high) {
        printf("attn_lg_reject_miss saw_high=%d\n", saw_high);
        free(blg);
        return 2;
    }
    printf("attn_lg_ok nq=%d nk=%d bgs=%d h0_ge_64=1\n", Nq, Nk, bgs);
    free(blg);
    return 0;
}
