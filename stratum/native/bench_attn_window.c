/* Attendance microbench. Calls the shipped predicate in stratum_attn_window.h
 * and, beside it, the pre-change sink+ring rule (local, so the before-number
 * stays reproducible). No model file, no large allocation.
 *
 *   legacy  cap=4096 last=4999  visible=4096  token100=0
 *   shipped cap=8192 last=4999  visible=5000  token100=1
 */
#include "stratum_attn_window.h"
#include <stdio.h>

static int ring_visible(int token, int last, int capacity, int sink) {
    int ring;
    if (token < 0 || token > last || capacity < 1) return 0;
    if (last < capacity) return 1;
    if (sink < 0) sink = 0;
    if (sink > capacity) sink = capacity;
    if (token < sink) return 1;
    ring = capacity - sink;
    if (ring < 1) return 0;
    return token + ring > last;
}

static int ring_count(int last, int capacity, int sink) {
    int n = 0, t;
    for (t = 0; t <= last; t++) n += ring_visible(t, last, capacity, sink);
    return n;
}

int main(void) {
    const int last = 4999;
    const int token = 100;
    int leg_n = ring_count(last, 4096, 4);
    int leg_has = ring_visible(token, last, 4096, 4);
    int ship_n = stratum_attn_visible_count(last, 8192);
    int ship_has = stratum_attn_token_visible(token, last, 8192);
    int ship_short = stratum_attn_visible_count(15, 4096);
    printf("legacy_cap4096_visible=%d legacy_has_100=%d\n", leg_n, leg_has);
    printf("shipped_cap8192_visible=%d shipped_has_100=%d\n", ship_n, ship_has);
    printf("shipped_short16_visible=%d\n", ship_short);
    if (leg_n != 4096 || leg_has != 0) return 2;
    if (ship_n != 5000 || ship_has != 1) return 3;
    if (ship_short != 16) return 4;
    printf("attn_window_ok\n");
    return 0;
}
