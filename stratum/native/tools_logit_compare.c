// logit_compare — compare two STRATUM_LOGITS_DUMP files (see stratum_engine.h).
//
// Reports, per position and in summary: KL(base || candidate), top-1
// agreement, and max |Δlogit|. Pure measurement — the standard finer-grained
// regression behind "greedy argmax sequences are identical" (the gates only
// see a handful of tokens; this sees the whole distribution).
//
// Usage:   logit_compare <base.slog> <candidate.slog>
// Exit:    0 unless a dump is malformed or the two vocab sizes differ.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

typedef struct {
    FILE* f;
    uint32_t n_vocab;
    uint32_t token;
    float* logits;
    int eof;
} SLog;

static int slog_open(const char* path, SLog* s) {
    memset(s, 0, sizeof(*s));
    s->f = fopen(path, "rb");
    if (!s->f) { fprintf(stderr, "logit_compare: cannot open %s\n", path); return -1; }
    char magic[8];
    if (fread(magic, 1, 8, s->f) != 8 || memcmp(magic, "SLOG0001", 8) != 0) {
        fprintf(stderr, "logit_compare: %s: bad magic\n", path);
        fclose(s->f); return -1;
    }
    if (fread(&s->n_vocab, 4, 1, s->f) != 1 || s->n_vocab == 0 || s->n_vocab > (1u << 26)) {
        fprintf(stderr, "logit_compare: %s: bad n_vocab\n", path);
        fclose(s->f); return -1;
    }
    s->logits = (float*)malloc(sizeof(float) * (size_t)s->n_vocab);
    if (!s->logits) { fclose(s->f); return -1; }
    return 0;
}

static int slog_next(SLog* s) {
    if (fread(&s->token, 4, 1, s->f) != 1) { s->eof = 1; return 0; }
    if (fread(s->logits, sizeof(float), s->n_vocab, s->f) != s->n_vocab) {
        fprintf(stderr, "logit_compare: truncated record\n");
        s->eof = 1; return -1;
    }
    return 1;
}

static void slog_close(SLog* s) {
    if (s->f) fclose(s->f);
    free(s->logits);
}

/* log-sum-exp based KL(base||Q): stable for raw logits in float32 */
static double kl_divergence(const float* b, const float* q, uint32_t n) {
    double mb = b[0], mq = q[0];
    for (uint32_t i = 1; i < n; i++) {
        if (b[i] > mb) mb = b[i];
        if (q[i] > mq) mq = q[i];
    }
    double zb = 0, zq = 0;
    for (uint32_t i = 0; i < n; i++) { zb += exp((double)b[i] - mb); zq += exp((double)q[i] - mq); }
    double logZb = mb + log(zb), logZq = mq + log(zq);
    double kl = 0;
    for (uint32_t i = 0; i < n; i++) {
        double lp = (double)b[i] - logZb;          /* log p(i)          */
        double lq = (double)q[i] - logZq;          /* log q(i)          */
        double p = exp(lp);
        if (p > 1e-300) kl += p * (lp - lq);
    }
    return kl;
}

static int argmax_f32(const float* v, uint32_t n) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return (int)best;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <base.slog> <candidate.slog>\n", argv[0]);
        return 1;
    }
    SLog base, cand;
    if (slog_open(argv[1], &base) != 0) return 1;
    if (slog_open(argv[2], &cand) != 0) return 1;
    if (base.n_vocab != cand.n_vocab) {
        fprintf(stderr, "logit_compare: vocab mismatch (%u vs %u)\n", base.n_vocab, cand.n_vocab);
        return 1;
    }

    printf("  %-6s %-10s %-14s %-14s %-10s\n", "step", "token", "KL(base||cand)", "max|dl|", "top1 agree");
    double kl_sum = 0, kl_max = 0;
    uint64_t steps = 0, agree = 0;
    for (;;) {
        int rb = slog_next(&base);
        int rc = slog_next(&cand);
        if (rb < 0 || rc < 0) return 1;
        if (!rb || !rc) {
            if (rb != rc)
                fprintf(stderr, "logit_compare: length mismatch (%llu vs %llu records compared; extra records ignored)\n",
                        (unsigned long long)steps, (unsigned long long)steps);
            break;
        }
        double kl = kl_divergence(base.logits, cand.logits, base.n_vocab);
        double maxd = 0;
        for (uint32_t i = 0; i < base.n_vocab; i++) {
            double d = fabs((double)base.logits[i] - (double)cand.logits[i]);
            if (d > maxd) maxd = d;
        }
        int tb = argmax_f32(base.logits, base.n_vocab);
        int tc = argmax_f32(cand.logits, cand.n_vocab);
        int ok = tb == tc;
        agree += ok;
        kl_sum += kl;
        if (kl > kl_max) kl_max = kl;
        steps++;
        printf("  %-6llu %-10u %-14.6g %-14.6g %-10s\n",
               (unsigned long long)steps - 1, base.token, kl, maxd, ok ? "yes" : "NO");
    }

    if (steps == 0) {
        fprintf(stderr, "logit_compare: no records compared\n");
        return 1;
    }
    printf("  summary: steps=%llu  mean KL=%.6g  max KL=%.6g  top-1 agreement=%llu/%llu (%.2f%%)\n",
           (unsigned long long)steps, kl_sum / (double)steps, kl_max,
           (unsigned long long)agree, (unsigned long long)steps,
           100.0 * (double)agree / (double)steps);
    slog_close(&base);
    slog_close(&cand);
    return 0;
}
