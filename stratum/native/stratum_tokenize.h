/*
 * stratum_tokenize.h — lightweight detokenizer for GGUF models.
 *
 * Reads `tokenizer.ggml.tokens` (array<string>) from GGUF metadata,
 * builds a zero-copy offset table over the mmap, and provides
 * id→string lookup for human-readable output.
 *
 * Handles GPT-2 byte-level BPE display encoding: "Ġ" → space,
 * "Ċ" → newline, "ĉ" → tab, "Ĝ" → tab alternative.
 *
 * No tokenization (text→ids) — input remains pre-tokenized IDs.
 * Zero memory overhead beyond the offset table (~8 bytes per entry).
 */
#ifndef STRATUM_TOKENIZE_H
#define STRATUM_TOKENIZE_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    const uint8_t* base;    /* mmap base for zero-copy reads */
    uint32_t       count;   /* number of tokens */
    uint32_t*      offs;    /* offset into base for each token string */
    uint16_t*      lens;    /* length of each token string */
    int            available;
} StratumVocab;

/* Scan GGUF metadata for tokenizer.ggml.tokens; build offset table.
 * Returns 0 on success, -1 if no tokenizer data found. */
static inline int stratum_vocab_init(const Gguf* gguf, const uint8_t* mmap_base,
                                      StratumVocab* vocab) {
    memset(vocab, 0, sizeof(*vocab));
    vocab->available = 0;

    /* Find tokenizer.ggml.tokens KV entry */
    const GgufKV* kv = NULL;
    for (uint64_t i = 0; i < gguf->n_kv; i++) {
        if (strcmp(gguf->kv[i].key, "tokenizer.ggml.tokens") == 0) {
            kv = &gguf->kv[i];
            break;
        }
    }
    if (!kv || kv->vtype != 9) return -1;  /* not array<string> */

    /* Parse array structure: u32 etype(=8), u64 count, then strings */
    const uint8_t* p = kv->bytes;
    uint32_t etype;
    memcpy(&etype, p, 4); p += 4;
    if (etype != 8) return -1;  /* element type must be STRING */

    uint64_t cnt;
    memcpy(&cnt, p, 8); p += 8;
    if (cnt == 0 || cnt > 1000000) return -1;

    vocab->base  = (const uint8_t*)p;  /* strings start here */
    vocab->count = (uint32_t)cnt;

    /* Build offset+length table */
    vocab->offs = (uint32_t*)malloc(sizeof(uint32_t) * cnt);
    vocab->lens = (uint16_t*)malloc(sizeof(uint16_t) * cnt);
    if (!vocab->offs || !vocab->lens) {
        free(vocab->offs); free(vocab->lens);
        vocab->offs = NULL; vocab->lens = NULL;
        return -1;
    }

    const uint8_t* q = vocab->base;
    for (uint64_t i = 0; i < cnt; i++) {
        uint64_t slen;
        memcpy(&slen, q, 8); q += 8;
        vocab->offs[i] = (uint32_t)(q - vocab->base);
        vocab->lens[i] = (uint16_t)(slen > 65535 ? 65535 : slen);
        q += slen;
    }

    vocab->available = 1;
    return 0;
}

static inline void stratum_vocab_free(StratumVocab* v) {
    free(v->offs); free(v->lens);
    v->offs = NULL; v->lens = NULL;
    v->available = 0;
}

/* Decode one token into a writable buffer (handles Ġ→space etc.).
 * Returns bytes written. Buffer should be at least 256 bytes. */
static inline int stratum_decode_token(const StratumVocab* v, uint32_t id,
                                        char* buf, int buf_size) {
    if (!v->available || id >= v->count) return snprintf(buf, buf_size, "[%u]", id);

    const uint8_t* src = v->base + v->offs[id];
    int len = v->lens[id];
    if (len > buf_size - 1) len = buf_size - 1;

    int wi = 0;
    for (int ri = 0; ri < len && wi < buf_size - 1; ri++) {
        uint8_t c = src[ri];
        if (c == 0xC4 && ri + 1 < len && src[ri + 1] == 0xA0) {
            /* Ġ (U+0120) → space */
            buf[wi++] = ' '; ri++;
        } else if (c == 0xC4 && ri + 1 < len && src[ri + 1] == 0x8A) {
            /* Ċ (U+010A) → newline */
            buf[wi++] = '\n'; ri++;
        } else if (c == 0xC4 && ri + 1 < len && src[ri + 1] == 0x89) {
            /* ĉ (U+0109) → tab */
            buf[wi++] = '\t'; ri++;
        } else {
            buf[wi++] = (char)c;
        }
    }
    buf[wi] = '\0';
    return wi;
}

#endif /* STRATUM_TOKENIZE_H */
