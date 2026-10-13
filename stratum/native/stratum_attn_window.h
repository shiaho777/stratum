#ifndef STRATUM_ATTN_WINDOW_H
#define STRATUM_ATTN_WINDOW_H

/* Full-prefix KV attendance.
 *
 * A token stays visible for every later step until the KV capacity is
 * exhausted. The previous qwen35 sink+ring (capacity 4096, sink 4) kept
 * 4096 buffer slots but, once the context passed the capacity, overwrote
 * early tokens: at last-written index 4999, context token 100 was gone.
 *
 * stratum_attn_token_visible is the predicate the engine now implements
 * when the write cursor still fits. Past capacity the engine returns an
 * error instead of wrapping, so a too-small buffer cannot silently drop
 * the middle of the prompt.
 */

static inline int stratum_attn_token_visible(int token_index, int last_written,
                                            int capacity) {
    if (capacity < 1) return 0;
    if (token_index < 0 || last_written < 0) return 0;
    if (token_index > last_written) return 0;
    if (last_written >= capacity) return 0;
    return 1;
}

static inline int stratum_attn_visible_count(int last_written, int capacity) {
    if (capacity < 1 || last_written < 0) return 0;
    if (last_written >= capacity) return 0;
    return last_written + 1;
}

#endif
