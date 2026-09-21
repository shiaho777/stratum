/* stratum_q4k_w16_amx.h — AMX kernel for GGML_TYPE_Q4K_W16 (type 43).
 *
 *  The w16 blob stores the exact Q4_K integer n*sc in i16 lanes, k-major
 *  within 32-row tiles, so the matvec runs as rank-1 MATINT outer products:
 *    - 32-row x 32-seq tiles, i16 x i16 -> i32 accumulators (1024 MACs/op)
 *    - one drain per (32-row x 32-seq x 2048-k) tile
 *    - the Q4_K minterm runs on the same MATINT path: m16 i16 tiles x i16
 *      group sums -> exact-integer i32 (m<=63, vsum<=32736)
 *    - merge applies y = xs * (d*P - dm*M) with P,M both exact integers
 *
 *  Two independent AMX units exist on this machine (one per P-cluster,
 *  measured ~917M matint/s each, 2 threads scale exactly 2x) — a persistent
 *  pool thread owns unit 2; the caller runs unit 1. Prep is split by
 *  16-seq halves; a content-hash cache skips prep entirely when transformer
 *  sibling projections (q/k/v, gate/up) reuse the same activation buffers.
 *
 *  Activation quant is per-seq i16 10-bit (~56-60dB SNR, ~6-12x finer than
 *  the int8 path's ~45dB) — NOT bit-exact vs SDOT/f64; near-tie argmax can
 *  flip on random-weight models (SDOT itself flips there too). Bench gate:
 *  full-matrix argmax_mismatch=0 vs the f64 fallback.
 *
 *  Only reached for type-43 tensors, which exist only in gguf_w16_convert
 *  output — the model file itself is the opt-in. STRATUM_AMX_W16_OFF=1
 *  forces the portable fallback. Requires N%32==0 && K%256==0 (converter
 *  leaves other tensors as Q4_K).
 */
#ifndef STRATUM_Q4K_W16_AMX_H
#define STRATUM_Q4K_W16_AMX_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <setjmp.h>
#include <pthread.h>
#include <arm_neon.h>
#include <mach/mach.h>
#include <mach/thread_policy.h>

/* ---- raw AMX ops (undocumented; encodings validated on M4) ---- */
#define ST_AMX_NOP5(op,i5) __asm(".word (0x201000 + (%0 << 5) + %1)" : : "i"(op), "i"(i5) : "memory")
#define ST_AMX_OPG(op,g)   __asm(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" : : "i"(op), "r"((uint64_t)(g)) : "memory")
#define ST_AMX_SET()    ST_AMX_NOP5(17,0)
#define ST_AMX_CLR()    ST_AMX_NOP5(17,1)
#define ST_AMX_LDX(g)   ST_AMX_OPG(0,g)
#define ST_AMX_LDY(g)   ST_AMX_OPG(1,g)
#define ST_AMX_LDZ(g)   ST_AMX_OPG(4,g)
#define ST_AMX_STZ(g)   ST_AMX_OPG(5,g)
#define ST_AMX_MATINT(g) ST_AMX_OPG(20,g)
#define ST_AMX_MATFP(g)  ST_AMX_OPG(21,g)

#define ST_AMX_NW 2              /* AMX units (one per P-cluster, measured) */
#define ST_AMX_SLAB 8            /* drain slabs per worker (lag-7 pipeline) */
#define ST_AMX_LAG  (ST_AMX_SLAB - 1)

static inline uint64_t st_amx_lx1(int r,const void*p){ return ((uint64_t)r<<56)|(uint64_t)p; }
static inline uint64_t st_amx_lx4(int r,const void*p){ return ((uint64_t)r<<56)|(1ULL<<62)|(1ULL<<60)|(uint64_t)p; }
static inline uint64_t st_amx_lz2(int z,const void*p){ return ((uint64_t)z<<56)|(1ULL<<62)|(uint64_t)p; }
static inline uint64_t st_amx_stz(int z,void*p){ return ((uint64_t)z<<56)|(1ULL<<62)|(uint64_t)p; }
static inline uint64_t st_amx_mi(int xo,int yo){ return (1ULL<<63)|(3ULL<<42)|((uint64_t)xo<<10)|(uint64_t)yo; }
static inline uint64_t st_amx_mfp(int xo,int yo){ return (1ULL<<63)|(4ULL<<42)|((uint64_t)xo<<10)|(uint64_t)yo; }

/* one-time availability probe: SIGILL -> path permanently off */
static sigjmp_buf st_amx_jb;
static void st_amx_sigh(int s){ (void)s; siglongjmp(st_amx_jb,1); }
static int st_amx_available(void) {
    static int probed = 0, ok = 0;
    if (!probed) {
        probed = 1;
        struct sigaction sa = { .sa_handler = st_amx_sigh }, old;
        sigaction(SIGILL, &sa, &old);
        if (sigsetjmp(st_amx_jb, 1) == 0) { ST_AMX_SET(); ST_AMX_CLR(); ok = 1; }
        sigaction(SIGILL, &old, NULL);
    }
    return ok;
}

/* ---- blob layout (see tools_gguf_w16_convert.c) ---- */
typedef struct {
    const int16_t* w16;   /* [rt][k][32] tiles, N*K i16 */
    const float*   d;     /* [N] */
    const float*   dm;    /* [N] */
    const int16_t* m16;   /* [rt][g][32] i16 = m[rt*32+j][g] (minterm via MATINT) */
} StW16Blob;
static inline StW16Blob st_w16_blob(const uint8_t* p, int N, int K) {
    StW16Blob b;
    b.w16 = (const int16_t*)p;
    b.d   = (const float*)(p + (size_t)N * K * 2);
    b.dm  = b.d + N;
    b.m16 = (const int16_t*)(b.dm + N);
    return b;
}
/* ---- lazy buffers ---- */
static int16_t* st_amx_xw = NULL;    /* [c][k][32] per seq-chunk */
static int16_t* st_amx_vb = NULL;    /* [c][NG][32] i16 group sums (vsum) */
static int32_t* st_amx_ms = NULL;    /* [c][rt][64][16] i32 minsum drains */
static int32_t* st_amx_zb = NULL;    /* [SLAB][NCH][64][16] per-worker drains */
static float*   st_amx_xs1 = NULL;   /* [c*32+32] per-chunk scales */
static int16_t* st_amx_xq = NULL;    /* [c*32+b][K] */
static int32_t* st_amx_vsum = NULL;  /* [c*2+h][NG][16] group sums */
static size_t   st_amx_capN = 0, st_amx_capK = 0, st_amx_capC = 0;
static uint8_t  st_amx_zp[128] __attribute__((aligned(128)));

/* prep cache: transformer siblings (q/k/v, gate/up, SSM projections) share
 * the same xs buffers — if pointers AND content hash match the last call,
 * the whole quant+transpose+vb prep is skipped. Keyed per (chunk, half). */
static const float* st_amx_cxp[128];
static uint64_t     st_amx_chash[8];   /* [c*2+h], valid for st_amx_c* key */
static int          st_amx_cB = -1, st_amx_cK = 0, st_amx_cNC = 0;
static int          st_amx_hits = 0;   /* prof counter */

static int st_amx_ensure(int N, int K, int NC) {
    int NCH = (K + 2047) / 2048, NG = K / 32;
    if ((size_t)N > st_amx_capN || (size_t)K > st_amx_capK || (size_t)NC > st_amx_capC) {
        size_t cn = st_amx_capN > (size_t)N ? st_amx_capN : (size_t)N;
        size_t ck = st_amx_capK > (size_t)K ? st_amx_capK : (size_t)K;
        size_t cc = st_amx_capC > (size_t)NC ? st_amx_capC : (size_t)NC;
        free(st_amx_xw); free(st_amx_vb); free(st_amx_ms); free(st_amx_zb); free(st_amx_xs1); free(st_amx_xq);
        free(st_amx_vsum);
        st_amx_xw = malloc(ck * 32 * cc * sizeof(int16_t));
        st_amx_xq = malloc(ck * 32 * cc * sizeof(int16_t));
        st_amx_vb = malloc((size_t)NG * 32 * cc * sizeof(int16_t));
        st_amx_ms = malloc(cn * 32 * cc * sizeof(int32_t));  /* == cc*(cn/32)*1024 drains */
        st_amx_zb = malloc((size_t)ST_AMX_NW * ST_AMX_SLAB * NCH * 64 * 16 * sizeof(int32_t));
        st_amx_xs1 = malloc(32 * cc * sizeof(float));
        st_amx_vsum = malloc((size_t)cc * 2 * NG * 16 * sizeof(int32_t));
        st_amx_capN = cn; st_amx_capK = ck; st_amx_capC = cc;
        st_amx_cB = -1;   /* buffers reallocated — cached prep state is gone */
    }
    return st_amx_xw && st_amx_xq && st_amx_vb && st_amx_ms && st_amx_zb
           && st_amx_xs1 && st_amx_vsum;
}

/* raw issue: one 32-out x 32-seq tile, K_eff rank-1 steps, drained to zb
 * per 2048-step chunk (i32 bound: 2048*945*1023 = 1.98e9 < INT32_MAX at
 * 10-bit activations). wtile/xwt are [k][32] i16 row streams.
 * Doubles as the minterm kernel: m16 tiles + i16 vsum -> i32 ms drains
 * (m<=63, vsum<=32736 — the whole minterm is exact-integer now). */
static inline void st_amx_issue_raw(const int16_t* wtile, const int16_t* xwt,
                                    int K, int32_t* zb) {
    int NCH = (K + 2047) / 2048;
    for (int ch = 0; ch < NCH; ch++) {
        int c0 = ch * 2048, c1 = (c0 + 2048 < K) ? c0 + 2048 : K;
        for (int r = 0; r < 64; r += 2) ST_AMX_LDZ(st_amx_lz2(r, st_amx_zp));
        for (int c = c0; c < c1; c += 8) {
            ST_AMX_LDX(st_amx_lx4(0, xwt + (size_t)c * 32));
            ST_AMX_LDX(st_amx_lx4(4, xwt + (size_t)(c + 4) * 32));
            ST_AMX_LDY(st_amx_lx4(0, wtile + (size_t)c * 32));
            ST_AMX_LDY(st_amx_lx4(4, wtile + (size_t)(c + 4) * 32));
            for (int u = 0; u < 8 && c + u < c1; u++) ST_AMX_MATINT(st_amx_mi(u * 64, u * 64));
        }
        for (int r = 0; r < 64; r += 2) ST_AMX_STZ(st_amx_stz(r, zb + (size_t)ch * 1024 + r * 16));
    }
}
static inline void st_amx_issue(const StW16Blob* w, int rt, int K, int16_t* xwt,
                                int32_t* zb0, int zsel) {
    int NCH = (K + 2047) / 2048;
    st_amx_issue_raw(w->w16 + (size_t)rt * K * 32, xwt, K,
                     zb0 + (size_t)zsel * NCH * 1024);
}
static inline void st_amx_merge(const StW16Blob* w, int rt, int st32, int Bc,
                                int K, int N, const int32_t* zb0, int zsel,
                                float* const* ys,
                                const float* xs1c, const int32_t* mst) {
    int NCH = (K + 2047) / 2048;
    const int32_t* zb = zb0 + (size_t)zsel * NCH * 1024;
    /* When per-seq rows are 4KB-strided (N multiple of 1024), 32 interleaved
     * store streams alias to the same L1 set (~2.4us/tile stall) — stage into
     * an L1 tile then copy out per-seq. Otherwise store directly. */
    int stage = (N & 1023) == 0;
    float yt[32 * 32];
    for (int r = 0; r < 32; r++) {
        int row = rt * 32 + r;
        float32x4_t sc = vdupq_n_f32(w->d[row]);
        float32x4_t sd = vdupq_n_f32(w->dm[row]);
        const int32_t* re = zb + 2 * r * 16;
        const int32_t* ro = zb + (2 * r + 1) * 16;
        const int32_t* me = mst + 2 * r * 16;
        const int32_t* mo = mst + (2 * r + 1) * 16;
        for (int q = 0; q < 4; q++) {          /* 8 seqs per q: even/odd deinterleave */
            int32x4_t ie = vld1q_s32(re + q * 4), io = vld1q_s32(ro + q * 4);
            for (int ch = 1; ch < NCH; ch++) {
                const int32_t* zc = zb + (size_t)ch * 1024;
                ie = vaddq_s32(ie, vld1q_s32(zc + 2 * r * 16 + q * 4));
                io = vaddq_s32(io, vld1q_s32(zc + (2 * r + 1) * 16 + q * 4));
            }
            /* ms drains use the same [2r/2r+1][16] layout as zb */
            float32x4_t mf_e = vcvtq_f32_s32(vld1q_s32(me + q * 4));
            float32x4_t mf_o = vcvtq_f32_s32(vld1q_s32(mo + q * 4));
            float32x4x2_t xsv = vld2q_f32(xs1c + q * 8);            /* even xs / odd xs */
            /* y = xs*(d*P - dm*M): P=int dot, M=int minsum — both exact i32 */
            float32x4_t ve = vsubq_f32(vmulq_f32(sc, vcvtq_f32_s32(ie)),
                                       vmulq_f32(sd, mf_e));
            float32x4_t vo = vsubq_f32(vmulq_f32(sc, vcvtq_f32_s32(io)),
                                       vmulq_f32(sd, mf_o));
            float32x4_t ye_ = vmulq_f32(ve, xsv.val[0]);
            float32x4_t yo_ = vmulq_f32(vo, xsv.val[1]);
            float te[4], to[4]; vst1q_f32(te, ye_); vst1q_f32(to, yo_);
            int b0 = q * 8;
            if (stage) {
                for (int j = 0; j < 4; j++) {
                    yt[r * 32 + b0 + 2 * j]     = te[j];
                    yt[r * 32 + b0 + 2 * j + 1] = to[j];
                }
            } else {
                /* scatter even lanes to ys[2j], odd to ys[2j+1] */
                for (int j = 0; j < 4; j++) {
                    if (b0 + 2 * j < Bc)     ys[st32 + b0 + 2 * j][row]     = te[j];
                    if (b0 + 2 * j + 1 < Bc) ys[st32 + b0 + 2 * j + 1][row] = to[j];
                }
            }
        }
    }
    if (stage)
        for (int b = 0; b < Bc; b++) {
            float* yb = ys[st32 + b] + rt * 32;
            for (int r = 0; r < 32; r++) yb[r] = yt[r * 32 + b];
        }
}

/* ---- persistent AMX worker pool --------------------------------------
 * dispatch_apply bursts are too short (~150us) for the scheduler to migrate
 * blocks onto different P-clusters, so the 2nd AMX unit stayed idle. A
 * persistent thread settles on the 2nd cluster and only pays wake/join. */
static void st_amx_run_range(const StW16Blob* wb, int N, int K, int NC,
                             int rt0, int rt1, int wid, int B,
                             float* const* ys,
                             const float* const* xs, int solo, int maybe);

static pthread_t      st_amx_pool_th;
static pthread_mutex_t st_amx_pool_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  st_amx_pool_go = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  st_amx_pool_dn = PTHREAD_COND_INITIALIZER;
static volatile unsigned st_amx_pool_seq = 0;
static volatile int      st_amx_pool_done = 0;
static int            st_amx_pool_on = 0;   /* -1 once spawn failed */
static StW16Blob      st_amx_jwb;
static int            st_amx_jN, st_amx_jK, st_amx_jNC, st_amx_jB, st_amx_jr0, st_amx_jr1;
static float* const*        st_amx_jys;
static const float* const*  st_amx_jxs;
static int                  st_amx_jhit;
/* cross-worker readiness: [w*2]=prep-half done, [w*2+1]=minterm-half done */
static volatile int st_amx_fl[4];

static volatile int st_amx_pool_parked = 0;
static double st_amx_us(void);

static void* st_amx_pool_main(void* unused) {
    (void)unused;
    /* nudge onto the second P-cluster: distinct affinity tag */
    thread_affinity_policy_data_t pol = { .affinity_tag = 3 };
    thread_policy_set(mach_thread_self(), THREAD_AFFINITY_POLICY,
                      (thread_policy_t)&pol, THREAD_AFFINITY_POLICY_COUNT);
    unsigned seen = 0;
    for (;;) {
        /* spin-then-park: matmul calls arrive back-to-back during a forward,
         * so a short spin keeps the 2nd AMX unit hot; park if the caller
         * goes quiet (~150us) to avoid burning a P-core indefinitely. */
        double t0 = st_amx_us();
        while (__atomic_load_n(&st_amx_pool_seq, __ATOMIC_ACQUIRE) == seen) {
            if (st_amx_us() - t0 > 150.0) {
                pthread_mutex_lock(&st_amx_pool_mu);
                st_amx_pool_parked = 1;
                while (st_amx_pool_seq == seen)
                    pthread_cond_wait(&st_amx_pool_go, &st_amx_pool_mu);
                st_amx_pool_parked = 0;
                pthread_mutex_unlock(&st_amx_pool_mu);
                break;
            }
        }
        seen = st_amx_pool_seq;
        st_amx_run_range(&st_amx_jwb, st_amx_jN, st_amx_jK, st_amx_jNC,
                         st_amx_jr0, st_amx_jr1, 1, st_amx_jB, st_amx_jys,
                         st_amx_jxs, 0, st_amx_jhit);
        __atomic_store_n(&st_amx_pool_done, seen, __ATOMIC_RELEASE);
    }
    return NULL;
}

static int st_amx_pool_start(void) {
    if (st_amx_pool_on == 0) {
        if (pthread_create(&st_amx_pool_th, NULL, st_amx_pool_main, NULL) == 0) {
            pthread_detach(st_amx_pool_th);
            st_amx_pool_on = 1;
        } else st_amx_pool_on = -1;
    }
    return st_amx_pool_on > 0;
}

/* submit wid=1's range to the pool thread and run wid=0 on the caller */
static inline void st_amx_run_parallel(const StW16Blob* wb, int N, int K, int NC,
                                       int NT, int B, float* const* ys,
                                       const float* const* xs, int maybe) {
    int half = NT / 2;
    st_amx_jwb = *wb; st_amx_jN = N; st_amx_jK = K; st_amx_jNC = NC;
    st_amx_jB = B; st_amx_jr0 = half; st_amx_jr1 = NT; st_amx_jys = ys;
    st_amx_jxs = xs; st_amx_jhit = maybe;
    st_amx_fl[0] = st_amx_fl[1] = st_amx_fl[2] = st_amx_fl[3] = 0;
    unsigned s = __atomic_add_fetch(&st_amx_pool_seq, 1, __ATOMIC_RELEASE);
    if (st_amx_pool_parked) {           /* rare: worker is asleep */
        pthread_mutex_lock(&st_amx_pool_mu);
        pthread_cond_signal(&st_amx_pool_go);
        pthread_mutex_unlock(&st_amx_pool_mu);
    }
    st_amx_run_range(wb, N, K, NC, 0, half, 0, B, ys, xs, 0, maybe);
    while (__atomic_load_n(&st_amx_pool_done, __ATOMIC_ACQUIRE) != (int)s) { }
}

static double st_amx_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static int st_amx_prof = -1;
static double st_amx_wtime[ST_AMX_NW];

/* content hash of one (chunk, 16-seq half) — the same folding that
 * st_amx_prep_half computes as a side channel. Used to verify the prep
 * cache when xs pointers repeat (q/k/v, gate/up share one input). */
static uint64_t st_amx_hash_chalf(const float* const* xs, int c, int Bc,
                                  int K, int h) {
    uint32x4_t a0 = vdupq_n_u32(0x9e3779b9u + (unsigned)h * 97u);
    uint32x4_t a1 = vdupq_n_u32(0x85ebca6bu);
    int s0 = c * 32, bl = h * 16, bu = bl + 16 < Bc ? bl + 16 : Bc;
    for (int b = bl; b < bu; b++) {
        const uint32_t* p = (const uint32_t*)xs[s0 + b];
        for (int k = 0; k < K; k += 16) {
            a0 = vmlaq_u32(a0, vld1q_u32(p + k),      vdupq_n_u32(0x9e3779b1u));
            a1 = vmlaq_u32(a1, vld1q_u32(p + k + 4),  vdupq_n_u32(0xc2b2ae35u));
            a0 = vmlaq_u32(a0, vld1q_u32(p + k + 8),  vdupq_n_u32(0x27d4eb2fu));
            a1 = vmlaq_u32(a1, vld1q_u32(p + k + 12), vdupq_n_u32(0x165667b1u));
        }
    }
    uint32x4_t a = veorq_u32(a0, vshlq_n_u32(a1, 13));
    uint32_t t[4]; vst1q_u32(t, a);
    return (uint64_t)(t[0] ^ t[2]) * 0x9E3779B97F4A7C15ULL
         ^ ((uint64_t)(t[1] ^ t[3]) << 1);
}

/* activation prep for one 16-seq half of chunk c: i16 10-bit quant,
 * blocked 8x8 transpose into xw, per-seq scales, i16 group sums vsum[g][b]
 * (10-bit -> vsum<=32736 fits i16 -> the minterm runs as exact-int MATINT).
 * Plus a side-channel content hash folded into the amax pass (~free).
 * vsum accumulation is fused into the transpose pass (no extra xq scan).
 * Splits cleanly: each AMX worker owns one half (bh = 0 or 1). */
static uint64_t st_amx_prep_half(const float* const* xs, int c, int Bc,
                                 int K, int NG, int bh, int32_t* vsumg) {
    int s0 = c * 32, bl = bh * 16, bu = bl + 16;
    float*   xs1c = st_amx_xs1 + c * 32;
    int16_t* xqc  = st_amx_xq + (size_t)c * 32 * K;
    int16_t* xwc  = st_amx_xw + (size_t)c * K * 32;
    int16_t* vbc  = st_amx_vb + (size_t)c * NG * 32;
    uint32x4_t ha0 = vdupq_n_u32(0x9e3779b9u + (unsigned)bh * 97u);
    uint32x4_t ha1 = vdupq_n_u32(0x85ebca6bu);
    for (int b = bl; b < bu && b < Bc; b++) {
        const float* xp = xs[s0 + b];
        float32x4_t vm = vdupq_n_f32(1e-30f);
        for (int k = 0; k < K; k += 16) {
            float32x4_t v0 = vld1q_f32(xp + k),      v1 = vld1q_f32(xp + k + 4);
            float32x4_t v2 = vld1q_f32(xp + k + 8),  v3 = vld1q_f32(xp + k + 12);
            ha0 = vmlaq_u32(ha0, vreinterpretq_u32_f32(v0), vdupq_n_u32(0x9e3779b1u));
            ha1 = vmlaq_u32(ha1, vreinterpretq_u32_f32(v1), vdupq_n_u32(0xc2b2ae35u));
            ha0 = vmlaq_u32(ha0, vreinterpretq_u32_f32(v2), vdupq_n_u32(0x27d4eb2fu));
            ha1 = vmlaq_u32(ha1, vreinterpretq_u32_f32(v3), vdupq_n_u32(0x165667b1u));
            float32x4_t m0 = vmaxq_f32(vabsq_f32(v0), vabsq_f32(v1));
            float32x4_t m1 = vmaxq_f32(vabsq_f32(v2), vabsq_f32(v3));
            vm = vmaxq_f32(vm, vmaxq_f32(m0, m1));
        }
        float am = vmaxvq_f32(vm);
        float s = am / 1023.0f, inv = 1.0f / s;   /* 10-bit: vsum<=32736 fits i16 */
        xs1c[b] = s;
        int16_t* xq = xqc + (size_t)b * K;
        float32x4_t vi = vdupq_n_f32(inv);
        for (int k = 0; k < K; k += 8) {
            int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + k), vi));
            int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + k + 4), vi));
            vst1q_s16(xq + k, vcombine_s16(vmovn_s32(i0), vmovn_s32(i1)));
        }
    }
    memset(vsumg, 0, (size_t)NG * 16 * sizeof(int32_t));
    for (int k0 = 0; k0 < K; k0 += 8) {
        int g = k0 >> 5;
        int32_t* vs = vsumg + (size_t)g * 16;
        for (int b0 = bl; b0 < bu; b0 += 8) {
            if (b0 + 8 > Bc) break;   /* tail handled below */
            int16x8_t r0 = vld1q_s16(xqc + (size_t)(b0 + 0) * K + k0);
            int16x8_t r1 = vld1q_s16(xqc + (size_t)(b0 + 1) * K + k0);
            int16x8_t r2 = vld1q_s16(xqc + (size_t)(b0 + 2) * K + k0);
            int16x8_t r3 = vld1q_s16(xqc + (size_t)(b0 + 3) * K + k0);
            int16x8_t r4 = vld1q_s16(xqc + (size_t)(b0 + 4) * K + k0);
            int16x8_t r5 = vld1q_s16(xqc + (size_t)(b0 + 5) * K + k0);
            int16x8_t r6 = vld1q_s16(xqc + (size_t)(b0 + 6) * K + k0);
            int16x8_t r7 = vld1q_s16(xqc + (size_t)(b0 + 7) * K + k0);
            /* fused: accumulate per-seq sums of this 8-k block into vsumg
             * (feeds vb = xs_b * group-sum without a third xq pass) */
            vs[b0 - bl + 0] += vaddlvq_s16(r0); vs[b0 - bl + 1] += vaddlvq_s16(r1);
            vs[b0 - bl + 2] += vaddlvq_s16(r2); vs[b0 - bl + 3] += vaddlvq_s16(r3);
            vs[b0 - bl + 4] += vaddlvq_s16(r4); vs[b0 - bl + 5] += vaddlvq_s16(r5);
            vs[b0 - bl + 6] += vaddlvq_s16(r6); vs[b0 - bl + 7] += vaddlvq_s16(r7);
            int16x8_t t0 = vtrn1q_s16(r0, r1), t1 = vtrn2q_s16(r0, r1);
            int16x8_t t2 = vtrn1q_s16(r2, r3), t3 = vtrn2q_s16(r2, r3);
            int16x8_t t4 = vtrn1q_s16(r4, r5), t5 = vtrn2q_s16(r4, r5);
            int16x8_t t6 = vtrn1q_s16(r6, r7), t7 = vtrn2q_s16(r6, r7);
            int16x8_t u0 = (int16x8_t)vtrn1q_s32((int32x4_t)t0, (int32x4_t)t2);
            int16x8_t u1 = (int16x8_t)vtrn1q_s32((int32x4_t)t4, (int32x4_t)t6);
            int16x8_t u2 = (int16x8_t)vtrn2q_s32((int32x4_t)t0, (int32x4_t)t2);
            int16x8_t u3 = (int16x8_t)vtrn2q_s32((int32x4_t)t4, (int32x4_t)t6);
            int16x8_t u4 = (int16x8_t)vtrn1q_s32((int32x4_t)t1, (int32x4_t)t3);
            int16x8_t u5 = (int16x8_t)vtrn1q_s32((int32x4_t)t5, (int32x4_t)t7);
            int16x8_t u6 = (int16x8_t)vtrn2q_s32((int32x4_t)t1, (int32x4_t)t3);
            int16x8_t u7 = (int16x8_t)vtrn2q_s32((int32x4_t)t5, (int32x4_t)t7);
            int16x8_t o0 = (int16x8_t)vtrn1q_s64((int64x2_t)u0, (int64x2_t)u1);
            int16x8_t o1 = (int16x8_t)vtrn2q_s64((int64x2_t)u0, (int64x2_t)u1);
            int16x8_t o2 = (int16x8_t)vtrn1q_s64((int64x2_t)u2, (int64x2_t)u3);
            int16x8_t o3 = (int16x8_t)vtrn2q_s64((int64x2_t)u2, (int64x2_t)u3);
            int16x8_t o4 = (int16x8_t)vtrn1q_s64((int64x2_t)u4, (int64x2_t)u5);
            int16x8_t o5 = (int16x8_t)vtrn2q_s64((int64x2_t)u4, (int64x2_t)u5);
            int16x8_t o6 = (int16x8_t)vtrn1q_s64((int64x2_t)u6, (int64x2_t)u7);
            int16x8_t o7 = (int16x8_t)vtrn2q_s64((int64x2_t)u6, (int64x2_t)u7);
            int16_t* xw = xwc + (size_t)k0 * 32 + b0;
            vst1q_s16(xw + 0 * 32, o0); vst1q_s16(xw + 1 * 32, o4);
            vst1q_s16(xw + 2 * 32, o2); vst1q_s16(xw + 3 * 32, o6);
            vst1q_s16(xw + 4 * 32, o1); vst1q_s16(xw + 5 * 32, o5);
            vst1q_s16(xw + 6 * 32, o3); vst1q_s16(xw + 7 * 32, o7);
        }
        int tl = Bc & ~7; if (tl < bl) tl = bl;
        for (int b = tl; b < bu; b++) {
            if (b < Bc) { const int16_t* xq = xqc + (size_t)b * K;
                          int s = 0;
                          for (int j = 0; j < 8; j++) { int16_t v = xq[k0 + j];
                              xwc[(size_t)(k0 + j) * 32 + b] = v; s += v; }
                          vs[b - bl] += s; }
            else        { for (int j = 0; j < 8; j++) xwc[(size_t)(k0 + j) * 32 + b] = 0; }
        }
    }
    for (int b = bl; b < bu; b++) if (b >= Bc) xs1c[b] = 0;
    for (int g = 0; g < NG; g++)
        for (int i = 0; i < 16; i++) {
            int b = bl + i;
            vbc[(size_t)g * 32 + b] =
                (b < Bc) ? (int16_t)vsumg[(size_t)g * 16 + i] : 0;
        }
    uint32x4_t ha = veorq_u32(ha0, vshlq_n_u32(ha1, 13));
    uint32_t ht[4]; vst1q_u32(ht, ha);
    return (uint64_t)(ht[0] ^ ht[2]) * 0x9E3779B97F4A7C15ULL
         ^ ((uint64_t)(ht[1] ^ ht[3]) << 1);
}

/* worker: prep for its 16-seq half of every chunk, then run tiles
 * rt in [rt0,rt1) over all chunks on this thread's AMX. This machine has 2
 * independent AMX units (one per P-cluster, measured ~917M matint/s each,
 * 2 threads scale exactly 2x). AMX state is per-thread: each worker enables
 * its own context. solo=1 -> caller does both halves, no sync. */
static void st_amx_run_range(const StW16Blob* wb, int N, int K, int NC,
                             int rt0, int rt1, int wid, int B,
                             float* const* ys,
                             const float* const* xs, int solo, int maybe) {
    int NG = K / 32, NCH = (K + 2047) / 2048;
    int32_t* zb0 = st_amx_zb + (size_t)wid * ST_AMX_SLAB * NCH * 1024;
    int h0 = solo ? 0 : wid, h1 = solo ? 2 : wid + 1;
    /* phase 1 (pre-AMX): quant+transpose+vb for my seq-half of all chunks.
     * When the caller marked a cache candidate (same xs pointers/dims), the
     * content hash decides per (chunk,half) whether prep can be skipped. */
    for (int h = h0; h < h1; h++)
        for (int c = 0; c < NC; c++) {
            int Bc = (B - c * 32 < 32) ? B - c * 32 : 32;
            int ci = c * 2 + h;
            int32_t* vsg = st_amx_vsum + (size_t)ci * NG * 16;
            if (NC <= 4) {
                if (maybe && st_amx_hash_chalf(xs, c, Bc, K, h) == st_amx_chash[ci]) {
                    if (st_amx_prof) __atomic_add_fetch(&st_amx_hits, 1, __ATOMIC_RELAXED);
                    continue;
                }
                st_amx_chash[ci] = st_amx_prep_half(xs, c, Bc, K, NG, h, vsg);
            } else
                (void)st_amx_prep_half(xs, c, Bc, K, NG, h, vsg);
        }
    __atomic_store_n(&st_amx_fl[wid * 2], 1, __ATOMIC_RELEASE);
    if (!solo)   /* minterm needs the other half's vsum rows */
        while (!__atomic_load_n(&st_amx_fl[(1 - wid) * 2], __ATOMIC_ACQUIRE)) { }
    ST_AMX_SET();
    /* minterm is interleaved into the job lookahead: each job's ms tile is
     * issued (MATINT m16 x i16 vsum, exact-integer) right before its main
     * tile, so the first main drain isn't stuck behind a 112-tile burst.
     * Same drain layout as zb; AMX retires stores in program order, and the
     * merge only reads ms[i] three jobs after it was issued. */
    int NRT = N / 32;
    int NT = rt1 - rt0, NJ = NT * NC;
    for (int i = 0; i < ST_AMX_LAG && i < NJ; i++) {
        st_amx_issue_raw(wb->m16 + (size_t)(rt0 + i / NC) * NG * 32,
                         st_amx_vb + (size_t)(i % NC) * NG * 32, NG,
                         st_amx_ms + ((size_t)(i % NC) * NRT + rt0 + i / NC) * 1024);
        st_amx_issue(wb, rt0 + i / NC, K,
                     st_amx_xw + (size_t)(i % NC) * K * 32, zb0, i & (ST_AMX_SLAB - 1));
    }
    double w0 = st_amx_prof ? st_amx_us() : 0;
    for (int i = 0; i < NJ; i++) {
        if (i + ST_AMX_LAG < NJ) {
            int j = i + ST_AMX_LAG;
            st_amx_issue_raw(wb->m16 + (size_t)(rt0 + j / NC) * NG * 32,
                             st_amx_vb + (size_t)(j % NC) * NG * 32, NG,
                             st_amx_ms + ((size_t)(j % NC) * NRT + rt0 + j / NC) * 1024);
            st_amx_issue(wb, rt0 + j / NC, K,
                         st_amx_xw + (size_t)(j % NC) * K * 32, zb0, j & (ST_AMX_SLAB - 1));
        }
        int c = i % NC, rt = rt0 + i / NC;
        int s0 = c * 32, Bc = (B - s0 < 32) ? B - s0 : 32;
        /* prefetch ~2 jobs ahead of the LDY stream: the weight flow is
         * DRAM-latency-bound (~25GB/s vs 38 ceiling), so keeping 2-3 tiles
         * (~192KB) in flight hides it behind this tile's merge window. */
        for (int pa = 2; pa <= 3; pa++)
            if (i + pa < NJ) {
                const int16_t* nw = wb->w16 + (size_t)(rt0 + (i + pa) / NC) * K * 32;
                int beg = (pa - 2) * (K * 32);   /* half-tile per merge keeps it cheap */
                for (int pf = 0; pf < K * 32; pf += 4096)
                    __builtin_prefetch((const char*)nw + beg * 2 + pf, 0, 0);
            }
        st_amx_merge(wb, rt, s0, Bc, K, N, zb0, i & (ST_AMX_SLAB - 1), ys,
                     st_amx_xs1 + c * 32,
                     st_amx_ms + ((size_t)c * NRT + rt) * 1024);
    }
    if (st_amx_prof) st_amx_wtime[wid] = st_amx_us() - w0;
    ST_AMX_CLR();
}

/* Portable fallback (no AMX): correct but slow — used when AMX probe fails
 * or STRATUM_AMX_W16_OFF=1. Reconstructs f32 weights from the w16 blob. */
static void st_q4k_w16_matvec_fallback(const GgufTensor* w,
                                       const float* const* xs,
                                       float* const* ys,
                                       int B, int N, int K) {
    int NG = K / 32;
    StW16Blob wb = st_w16_blob(g_st.mmap_base + w->offset, N, K);
    for (int b = 0; b < B; b++)
        for (int r = 0; r < N; r++) {
            const int16_t* wrow = wb.w16 + (size_t)(r / 32) * K * 32 + (r % 32);
            double acc = 0;
            for (int k = 0; k < K; k++) acc += (double)wrow[(size_t)k * 32] * xs[b][k];
            /* minterm: dm[r] * sum_g m16[g][r] * sum_{k in g} x_k */
            double msum = 0;
            for (int g = 0; g < NG; g++) {
                double s = 0;
                for (int i = 0; i < 32; i++) s += xs[b][g * 32 + i];
                msum += (double)wb.m16[(size_t)(r / 32) * NG * 32 + g * 32 + r % 32] * s;
            }
            ys[b][r] = (float)(wb.d[r] * acc - wb.dm[r] * msum);
        }
}

/* AMX multiseq entry: process B seqs in chunks of 32 (one Z tile of seqs). */
static inline void st_q4k_w16_amx_multix(const GgufTensor* w,
                                         const float* const* xs,
                                         float* const* ys,
                                         int B, int N, int K) {
    int NG = K / 32;
    if (st_amx_prof < 0) st_amx_prof = getenv("STRATUM_AMX_PROF") ? 1 : 0;
    StW16Blob wb = st_w16_blob(g_st.mmap_base + w->offset, N, K);
    int NC = (B + 31) / 32;                 /* seq-chunks (32 seqs per Z tile) */
    /* minterm drains once per tile: NG*63*32736 < INT32_MAX needs NG<=1040 */
    if (NG > 1024 || !st_amx_ensure(N, K, NC)) { st_q4k_w16_matvec_fallback(w, xs, ys, B, N, K); return; }
    memset(st_amx_zp, 0, 128);
    double t_mm = 0, tt;
    tt = st_amx_us();
    /* prep-cache candidate: same xs pointer array + dims as last call
     * (q/k/v, gate/up, SSM projections share one input). Content hash is
     * verified per (chunk,half) inside the workers before skipping. */
    int maybe = 0;
    if (NC <= 4) {
        maybe = (B == st_amx_cB && K == st_amx_cK && NC == st_amx_cNC);
        if (maybe)
            for (int b = 0; b < B; b++)
                if (xs[b] != st_amx_cxp[b]) { maybe = 0; break; }
        for (int b = 0; b < B; b++) st_amx_cxp[b] = xs[b];
        st_amx_cB = B; st_amx_cK = K; st_amx_cNC = NC;
    } else st_amx_cB = -1;
    /* prep is split by 16-seq half across the persistent pool
     * (2nd P-cluster AMX unit, ~2x issue+drain rate measured) + caller. */
    int NT_ = N / 32;
    if (NT_ >= 4 && st_amx_pool_start()) {
        st_amx_run_parallel(&wb, N, K, NC, NT_, B, ys, xs, maybe);
    } else {
        st_amx_run_range(&wb, N, K, NC, 0, NT_, 0, B, ys, xs, 1, maybe);
    }
    t_mm += st_amx_us() - tt;
    if (st_amx_prof)
        fprintf(stderr, "[amxprof] total=%.0f us (w0_jobs=%.0f w1_jobs=%.0f hits=%d)\n",
                t_mm, st_amx_wtime[0], st_amx_wtime[1], st_amx_hits);
    (void)NG;
}
#endif
