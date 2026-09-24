/* stratum_q4k_w16_amx.h — AMX kernel for GGML_TYPE_Q4K_W16 (type 43).
 *
 *  The w16 blob stores the exact Q4_K integer n*sc in i16 lanes, k-major
 *  within 32-row tiles, so the matvec runs as rank-1 MATINT outer products:
 *    - 32-row x 32-seq tiles, i16 x i16 -> i32 accumulators (1024 MACs/op)
 *    - one drain per (32-row x 32-seq x 256-k) tile = exactly one Q4_K
 *      block, because Q4_K carries a separate d/dmin pair per 256 elements
 *    - the Q4_K minterm runs on the same MATINT path per block: m16 i16
 *      tiles x i16 group sums -> exact-integer i32 (m<=63, vsum<=32736)
 *    - merge applies y = xs * sum_i (d_i*P_i - dm_i*M_i) over the K/256
 *      blocks, with d/dm stored as f32[N*nb] (P,M both exact integers)
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
#define ST_AMX_SLAB 16            /* drain slabs per worker (lag-7 pipeline) */
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
    const float*   d;     /* [N*nb] per 256-block super-scale */
    const float*   dm;    /* [N*nb] per 256-block super-min (all-zero for q6k) */
    const int16_t* m16;   /* [rt][g][32] i16 = m[rt*32+j][g]; NULL for q6k (no minterm) */
    int q6;               /* 1 for type-44: minterm skipped entirely */
} StW16Blob;
static inline StW16Blob st_w16_blob(const uint8_t* p, int N, int K, int q6) {
    StW16Blob b;
    size_t nb = (size_t)K / 256;
    b.w16 = (const int16_t*)p;
    b.d   = (const float*)(p + (size_t)N * K * 2);
    b.dm  = b.d + (size_t)N * nb;
    b.m16 = q6 ? NULL : (const int16_t*)(b.dm + (size_t)N * nb);
    b.q6  = q6;
    return b;
}
/* ---- lazy buffers ---- */
static int16_t* st_amx_xw = NULL;    /* [c][k][32] per seq-chunk */
static int16_t* st_amx_q4k_wt = NULL;   /* [v][slot][K*32] runtime-unpacked */
static int16_t* st_amx_q4k_mt = NULL;   /* [v][slot][NG*32] group mins */
static float*   st_amx_q4k_dd = NULL;   /* [v][slot][64*NCH] d then dm */
static size_t   st_amx_q4k_capK = 0;
static int16_t* st_amx_vb = NULL;    /* [c][NG][32] i16 group sums (vsum) */
static int32_t* st_amx_ms = NULL;    /* [c][rt][64][16] i32 minsum drains */
static int32_t* st_amx_zb = NULL;    /* [SLAB][NCH][64][16] per-worker drains */
static float*   st_amx_xs1 = NULL;   /* [c*32+32] per-chunk scales */
static int16_t* st_amx_xq = NULL;    /* [c*32+b][K] */
static int32_t* st_amx_vsum = NULL;  /* [c*2+h][NG][16] group sums */
static size_t   st_amx_capN = 0, st_amx_capK = 0, st_amx_capC = 0;
static size_t   st_amx_capNMS = 0;   /* ms only sized by minterm (type-43) callers */
static uint8_t  st_amx_zp[128] __attribute__((aligned(128)));

/* prep cache: transformer siblings (q/k/v, gate/up, SSM projections) share
 * the same xs buffers — if pointers AND content hash match the last call,
 * the whole quant+transpose+vb prep is skipped. Keyed per (chunk, half). */
static const float* st_amx_cxp[128];
static uint64_t     st_amx_chash[8];   /* [c*2+h], valid for st_amx_c* key */
static int          st_amx_cB = -1, st_amx_cK = 0, st_amx_cNC = 0;
static int          st_amx_hits = 0;   /* prof counter */

static int st_amx_ensure(int N, int K, int NC, int need_ms) {
    if ((size_t)N > st_amx_capN || (size_t)K > st_amx_capK || (size_t)NC > st_amx_capC
        || (need_ms && (size_t)N > st_amx_capNMS)) {
        size_t cn = st_amx_capN > (size_t)N ? st_amx_capN : (size_t)N;
        size_t ck = st_amx_capK > (size_t)K ? st_amx_capK : (size_t)K;
        size_t cc = st_amx_capC > (size_t)NC ? st_amx_capC : (size_t)NC;
        size_t cm = st_amx_capNMS;
        if (need_ms && (size_t)N > cm) cm = (size_t)N;   /* q6 never touches ms */
        /* every buffer must be sized by the capacity maxima, not this call's
         * dims: vb/vsum are indexed by the caller's NG=K/32 and zb by NCH —
         * a smaller-K realloc would otherwise shrink them under the feet of
         * later larger-K calls (heap overflow -> corrupt xs1/neighbors). */
        size_t cng = ck / 32, cnch = (ck + 255) / 256;
        free(st_amx_xw); free(st_amx_vb); free(st_amx_ms); free(st_amx_zb); free(st_amx_xs1); free(st_amx_xq);
        free(st_amx_vsum);
        st_amx_xw = malloc(ck * 32 * cc * sizeof(int16_t));
        st_amx_xq = malloc(ck * 32 * cc * sizeof(int16_t));
        st_amx_vb = malloc(cng * 32 * cc * sizeof(int16_t));
        st_amx_ms = malloc(cm * cnch * 32 * cc * sizeof(int32_t));  /* per-block minterm drains (q4k only) */
        st_amx_zb = malloc((size_t)ST_AMX_NW * ST_AMX_SLAB * cnch * 64 * 16 * sizeof(int32_t));
        st_amx_xs1 = malloc(32 * cc * sizeof(float));
        st_amx_vsum = malloc(cc * 2 * cng * 16 * sizeof(int32_t));
        st_amx_capN = cn; st_amx_capK = ck; st_amx_capC = cc; st_amx_capNMS = cm;
        st_amx_cB = -1;   /* buffers reallocated — cached prep state is gone */
    }
    return st_amx_xw && st_amx_xq && st_amx_vb && st_amx_ms && st_amx_zb
           && st_amx_xs1 && st_amx_vsum;
}

/* drain-landing canary: STZ stores retire asynchronously in AMX program
 * order, so a CPU merge racing the queue can read a stale slab. The tail
 * pair is pre-armed with a value no real drain can contain — accumulator
 * bound is 2048*945*1023 ~ 1.98e9 and the minterm stays < 0x7F7F7F7F at
 * NG<=1024 — then the merge spins until it is overwritten. In-order
 * retirement makes tail-landed imply whole-tile-landed. */
static int st_amx_prof = -1;
static double st_amx_us(void);
static double st_amx_t_issue[ST_AMX_NW], st_amx_t_wait[ST_AMX_NW], st_amx_t_merge[ST_AMX_NW];
#define ST_AMX_CAN 0x7F7F7F7F7F7F7F7FULL
static inline void st_amx_drain_arm(int32_t* zb, int NCH) {
    uint64_t* t = (uint64_t*)(zb + (size_t)NCH * 1024);
    t[-2] = ST_AMX_CAN; t[-1] = ST_AMX_CAN;
}
static inline void st_amx_drain_wait(const int32_t* zb, int NCH) {
    const volatile uint64_t* t = (const volatile uint64_t*)(zb + (size_t)NCH * 1024);
    long spin = 0;
    while (t[-1] == ST_AMX_CAN || t[-2] == ST_AMX_CAN)
        if (++spin > 500000000L) { fprintf(stderr, "[amx] drain wait timeout\n"); abort(); }
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
}

/* raw issue: one 32-out x 32-seq tile, K_eff rank-1 steps, drained to zb
 * per 2048-step chunk (i32 bound: 2048*945*1023 = 1.98e9 < INT32_MAX at
 * 10-bit activations). wtile/xwt are [k][32] i16 row streams.
 * Doubles as the minterm kernel: m16 tiles + i16 vsum -> i32 ms drains
 * (m<=63, vsum<=32736 — the whole minterm is exact-integer now). */
static inline void st_amx_issue_raw(const int16_t* wtile, const int16_t* xwt,
                                    int K, int32_t* zb, int ysgn) {
    /* one drain per 256-k block: Q4_K d/dm are per-block, so the merge must
     * see each block's integer partial separately (y = sum_i d_i*P_i - dm_i*M_i).
     * ysgn selects the MATINT Y-operand sign: Q4K_W16 weights are n*sc in
     * [0,945] (unsigned is fine), Q6K_W16 stores sc*(q-32) in [-4096,4096]
     * and MUST run signed-Y (bit 26) or negatives wrap to ~65436. */
    int NCH = (K + 255) / 256;
    uint64_t ys = (uint64_t)ysgn << 26;
    st_amx_drain_arm(zb, NCH);
    for (int ch = 0; ch < NCH; ch++) {
        int c0 = ch * 256, c1 = (c0 + 256 < K) ? c0 + 256 : K;
        for (int r = 0; r < 64; r += 2) ST_AMX_LDZ(st_amx_lz2(r, st_amx_zp));
        for (int c = c0; c < c1; c += 8) {
            ST_AMX_LDX(st_amx_lx4(0, xwt + (size_t)c * 32));
            ST_AMX_LDX(st_amx_lx4(4, xwt + (size_t)(c + 4) * 32));
            ST_AMX_LDY(st_amx_lx4(0, wtile + (size_t)c * 32));
            ST_AMX_LDY(st_amx_lx4(4, wtile + (size_t)(c + 4) * 32));
            for (int u = 0; u < 8 && c + u < c1; u++) ST_AMX_MATINT(st_amx_mi(u * 64, u * 64) | ys);
        }
        for (int r = 0; r < 64; r += 2) ST_AMX_STZ(st_amx_stz(r, zb + (size_t)ch * 1024 + r * 16));
    }
}
static inline void st_amx_issue(const StW16Blob* w, int rt, int K, int16_t* xwt,
                                int32_t* zb0, int zsel) {
    int NCH = (K + 255) / 256;
    st_amx_issue_raw(w->w16 + (size_t)rt * K * 32, xwt, K,
                     zb0 + (size_t)zsel * NCH * 1024, w->q6);
}
static inline void st_amx_merge(int q6, const float* d0, const float* dm0,
                                int rt, int st32, int Bc,
                                int K, int N, const int32_t* zb0, int zsel,
                                float* const* ys,
                                const float* xs1c, const int32_t* mst, int wid) {
    int NCH = (K + 255) / 256;              /* == nb: per-block drains */
    const int32_t* zb = zb0 + (size_t)zsel * NCH * 1024;
    double _t0 = st_amx_prof > 1 ? st_amx_us() : 0;
    st_amx_drain_wait(zb, NCH);      /* main tile drains landed */
    if (!q6) st_amx_drain_wait(mst, NCH); /* minterm drains landed */
    double _t1 = st_amx_prof > 1 ? st_amx_us() : 0;
    if (st_amx_prof > 1) st_amx_t_wait[wid] += _t1 - _t0;
    /* When per-seq rows are 4KB-strided (N multiple of 1024), 32 interleaved
     * store streams alias to the same L1 set (~2.4us/tile stall) — stage into
     * an L1 tile then copy out per-seq. Otherwise store directly. */
    int stage = (N & 1023) == 0 && !getenv("STRATUM_AMX_NOSTAGE");
    float yt[32 * 32];
    for (int r = 0; r < 32; r++) {
        int row = rt * 32 + r;
        const float* drow = d0 + (size_t)r * NCH;
        const float* dmrow = dm0 + (size_t)r * NCH;
        for (int q = 0; q < 4; q++) {          /* 8 seqs per q: even/odd deinterleave */
            /* y = xs * sum_i (d_i*P_i - dm_i*M_i): per-block exact-i32 partials */
            float32x4_t ae = vdupq_n_f32(0), ao = vdupq_n_f32(0);
            if (q6) {
                for (int ch = 0; ch < NCH; ch++) {
                    const int32_t* zc = zb + (size_t)ch * 1024;
                    float32x4_t ie = vcvtq_f32_s32(vld1q_s32(zc + 2 * r * 16 + q * 4));
                    float32x4_t io = vcvtq_f32_s32(vld1q_s32(zc + (2 * r + 1) * 16 + q * 4));
                    float32x4_t sc = vdupq_n_f32(drow[ch]);
                    ae = vfmaq_f32(ae, sc, ie);
                    ao = vfmaq_f32(ao, sc, io);
                }
            } else
            for (int ch = 0; ch < NCH; ch++) {
                const int32_t* zc = zb + (size_t)ch * 1024;
                const int32_t* mc = mst + (size_t)ch * 1024;
                float32x4_t ie = vcvtq_f32_s32(vld1q_s32(zc + 2 * r * 16 + q * 4));
                float32x4_t io = vcvtq_f32_s32(vld1q_s32(zc + (2 * r + 1) * 16 + q * 4));
                float32x4_t mf_e = vcvtq_f32_s32(vld1q_s32(mc + 2 * r * 16 + q * 4));
                float32x4_t mf_o = vcvtq_f32_s32(vld1q_s32(mc + (2 * r + 1) * 16 + q * 4));
                float32x4_t sc = vdupq_n_f32(drow[ch]);
                float32x4_t sd = vdupq_n_f32(dmrow[ch]);
                ae = vsubq_f32(vfmaq_f32(ae, sc, ie), vmulq_f32(sd, mf_e));
                ao = vsubq_f32(vfmaq_f32(ao, sc, io), vmulq_f32(sd, mf_o));
            }
            float32x4x2_t xsv = vld2q_f32(xs1c + q * 8);            /* even xs / odd xs */
            float32x4_t ye_ = vmulq_f32(ae, xsv.val[0]);
            float32x4_t yo_ = vmulq_f32(ao, xsv.val[1]);
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
    if (st_amx_prof > 1) st_amx_t_merge[wid] += st_amx_us() - _t1;
}

/* ---- merge offload pool --------------------------------------------------
 * merge (drain_wait + NEON f32 rescale/scatter) costs ~5us/tile on the issue
 * thread — half the call for big-N tensors. Offload to dedicated lanes:
 * issue threads only push AMX work and gate on slot-busy backpressure;
 * merge lanes pull tickets per worker. Slot ring reuse is safe: a slot is
 * marked busy at issue and cleared only after its merge completed. */
#define ST_AMX_MW 3
static pthread_t        st_amx_mw_th[ST_AMX_MW];
static pthread_mutex_t  st_amx_mw_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t   st_amx_mw_cv = PTHREAD_COND_INITIALIZER;
static volatile unsigned st_amx_mw_seq = 0;
static volatile unsigned st_amx_mw_ack[ST_AMX_MW];
static volatile int     st_amx_mw_parked = 0;
static volatile int     st_amx_mw_next[ST_AMX_NW];
static volatile int     st_amx_mw_issued[ST_AMX_NW];   /* jobs issued per worker */
static volatile int     st_amx_mw_done[ST_AMX_NW];     /* merges completed */
static volatile int     st_amx_mw_on = 0;
static volatile unsigned st_amx_mw_myseq = 0;
static volatile int     st_amx_slot_busy[ST_AMX_NW][ST_AMX_SLAB];
static int              st_amx_mw_n = 0, st_amx_mw_dead = 0;
static double st_amx_p_wait[2], st_amx_p_upk[2], st_amx_p_iss[2];
/* q4k unpack producer state (dedicated threads; see st_amx_ux_main) */
static volatile int st_amx_q4k_unext[ST_AMX_NW];
static volatile int st_amx_q4k_uready[ST_AMX_NW][ST_AMX_SLAB];
static double st_amx_ux_claims = 0, st_amx_is_claims = 0;
/* merge job context (set once per call, read-only for merge lanes) */
static const StW16Blob* st_amx_mw_w;
static int   st_amx_mw_q4k = 0;   /* 1 = runtime-unpacked Q4_K source */
static const uint8_t* st_amx_mw_q4k_src = NULL;
static int   st_amx_mw_N, st_amx_mw_K, st_amx_mw_NC, st_amx_mw_B;
static int   st_amx_mw_NRT, st_amx_mw_NCH;
static int   st_amx_mw_rt0[ST_AMX_NW], st_amx_mw_NJ[ST_AMX_NW];
static float* const* st_amx_mw_ys;
static const float*  st_amx_mw_xs1;

static void* st_amx_mw_main(void* a) {
    int wid = (int)(intptr_t)a; (void)wid;
    unsigned seen = 0;
    for (;;) {
        int spin = 0;
        while (__atomic_load_n(&st_amx_mw_seq, __ATOMIC_ACQUIRE) == seen) {
            if (++spin > 300000) {
                pthread_mutex_lock(&st_amx_mw_mu);
                __atomic_add_fetch(&st_amx_mw_parked, 1, __ATOMIC_RELAXED);
                while (st_amx_mw_seq == seen)
                    pthread_cond_wait(&st_amx_mw_cv, &st_amx_mw_mu);
                __atomic_sub_fetch(&st_amx_mw_parked, 1, __ATOMIC_RELAXED);
                pthread_mutex_unlock(&st_amx_mw_mu);
                break;
            }
        }
        seen = st_amx_mw_seq;
        const StW16Blob* w = st_amx_mw_w;
        int NCH = st_amx_mw_NCH, NC = st_amx_mw_NC, NRT = st_amx_mw_NRT;
        for (;;) {
            int did = 0;
            for (int v = 0; v < ST_AMX_NW; v++) {
                /* claim only issued jobs (CAS so the ticket is never wasted) */
                int j = st_amx_mw_next[v];
                if (j >= st_amx_mw_NJ[v]
                    || j >= __atomic_load_n(&st_amx_mw_issued[v], __ATOMIC_ACQUIRE)
                    || !__atomic_compare_exchange_n(
                           (volatile int*)&st_amx_mw_next[v], &j, j + 1,
                           0, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
                    continue;
                did = 1;
                int rt = st_amx_mw_rt0[v] + j / NC, c = j % NC;
                int zsel = j & (ST_AMX_SLAB - 1);
                int32_t* zb0 = st_amx_zb + (size_t)v * ST_AMX_SLAB
                                        * ((st_amx_capK + 255) / 256) * 1024;
                int32_t* msj = st_amx_ms +
                    ((size_t)c * NRT + rt) * (size_t)NCH * 1024;
                const float *d0, *dm0;
                if (st_amx_mw_q4k) {
                    float* dd = st_amx_q4k_dd +
                        ((size_t)v * ST_AMX_SLAB + zsel) * 64 * NCH;
                    d0 = dd; dm0 = dd + (size_t)32 * NCH;
                } else {
                    d0  = w->d  + (size_t)rt * 32 * NCH;
                    dm0 = w->dm + (size_t)rt * 32 * NCH;
                }
                st_amx_merge(st_amx_mw_q4k ? 0 : w->q6, d0, dm0, rt, c * 32,
                             (st_amx_mw_B - c * 32 < 32) ? st_amx_mw_B - c * 32 : 32,
                             st_amx_mw_K, st_amx_mw_N, zb0, zsel,
                             st_amx_mw_ys, st_amx_mw_xs1 + c * 32, msj, v);
                __atomic_store_n(&st_amx_slot_busy[v][zsel], 0, __ATOMIC_RELEASE);
                __atomic_add_fetch(&st_amx_mw_done[v], 1, __ATOMIC_RELEASE);
            }
            /* done only when every issued job of both workers is merged —
             * workers may not have issued yet, so idle != finished */
            if (!did) {
                if (__atomic_load_n(&st_amx_mw_done[0], __ATOMIC_ACQUIRE) >= st_amx_mw_NJ[0]
                    && __atomic_load_n(&st_amx_mw_done[1], __ATOMIC_ACQUIRE) >= st_amx_mw_NJ[1])
                    break;
                /* q4k mode is unpack-bound: a hot-spinning lane steals the
                 * cores the ux threads need. Yield only in q4k mode — the
                 * w16 path keeps its latency-critical spin. */
                if (st_amx_mw_q4k) sched_yield();
            }
        }
        __atomic_store_n(&st_amx_mw_ack[wid], seen, __ATOMIC_RELEASE);
    }
    return NULL;
}

static int st_amx_mw_start(void) {
    if (st_amx_mw_dead) return 0;
    if (!st_amx_mw_n) {
        for (int i = 0; i < ST_AMX_MW; i++) {
            if (pthread_create(&st_amx_mw_th[i], NULL, st_amx_mw_main,
                               (void*)(intptr_t)i) != 0) { st_amx_mw_dead = 1; break; }
            st_amx_mw_n++;
        }
        if (!st_amx_mw_n) return 0;
    }
    return 1;
}

/* ---- dedicated q4k unpack workers -----------------------------------------
 * In mw mode the issue threads must not unpack: while an issuer spends ~23us
 * inside st_q4k_unpack_tile, the three merge lanes spin at full tilt and (on
 * a loaded machine) preempt the unpacker, collapsing the pipeline ~6x
 * (measured 144us/job vs 25us solo). These threads only unpack: claim the
 * next job index per worker, fill that job's slot scratch, publish uready.
 * They never block holding a claim (claim only when the slot is already
 * free) and the issuer self-unpacks as fallback — deadlock-free. */
#define ST_AMX_UX 4
static pthread_t st_amx_ux_th[ST_AMX_UX];
static int       st_amx_ux_n = 0, st_amx_ux_dead = 0;
static void st_q4k_unpack_tile(const uint8_t* tbase, int rt, int K,
                               int16_t* wt, int16_t* mt, float* dd);

static void* st_amx_ux_main(void* a) {
    (void)a;
    unsigned seen = 0;
    for (;;) {
        int spin = 0;
        while (__atomic_load_n(&st_amx_mw_seq, __ATOMIC_ACQUIRE) == seen) {
            if (++spin > 300000) {
                pthread_mutex_lock(&st_amx_mw_mu);
                __atomic_add_fetch(&st_amx_mw_parked, 1, __ATOMIC_RELAXED);
                while (st_amx_mw_seq == seen)
                    pthread_cond_wait(&st_amx_mw_cv, &st_amx_mw_mu);
                __atomic_sub_fetch(&st_amx_mw_parked, 1, __ATOMIC_RELAXED);
                pthread_mutex_unlock(&st_amx_mw_mu);
                break;
            }
        }
        seen = st_amx_mw_seq;
        if (!st_amx_mw_q4k) continue;
        int NC = st_amx_mw_NC, NCH = st_amx_mw_NCH, Kw = st_amx_mw_K;
        const uint8_t* src = st_amx_mw_q4k_src;
        for (;;) {
            if (__atomic_load_n(&st_amx_mw_seq, __ATOMIC_ACQUIRE) != seen)
                break;
            int did = 0;
            for (int v = 0; v < ST_AMX_NW; v++) {
                int u = __atomic_load_n(&st_amx_q4k_unext[v],
                                        __ATOMIC_ACQUIRE);
                int z = u & (ST_AMX_SLAB - 1);
                if (u < st_amx_mw_NJ[v]
                    && __atomic_load_n(&st_amx_q4k_uready[v][z],
                                       __ATOMIC_ACQUIRE) != u + 1
                    && !__atomic_load_n(&st_amx_slot_busy[v][z],
                                        __ATOMIC_ACQUIRE)
                    && __atomic_compare_exchange_n(
                           (volatile int*)&st_amx_q4k_unext[v],
                           &u, u + 1, 0, __ATOMIC_RELAXED,
                           __ATOMIC_RELAXED)) {
                    /* mark the slot occupied NOW — until merge frees it no
                     * other ux may claim the i+SLAB successor into this
                     * scratch (that was the deadlock: a second claim could
                     * clobber an unpacked-but-not-yet-issued slot) */
                    __atomic_store_n(&st_amx_slot_busy[v][z], 1,
                                     __ATOMIC_RELEASE);
                    size_t sb = (size_t)v * ST_AMX_SLAB + z;
                    st_q4k_unpack_tile(src, st_amx_mw_rt0[v] + u / NC, Kw,
                                       st_amx_q4k_wt + sb * (size_t)Kw * 32,
                                       st_amx_q4k_mt + sb * (size_t)(Kw / 32) * 32,
                                       st_amx_q4k_dd + sb * 64 * NCH);
                    __atomic_store_n(&st_amx_q4k_uready[v][z], u + 1,
                                     __ATOMIC_RELEASE);
                    if (st_amx_prof)
                        __atomic_add_fetch(&st_amx_ux_claims, 1,
                                           __ATOMIC_RELAXED);
                    did = 1;
                }
            }
            if (!did
                && st_amx_q4k_unext[0] >= st_amx_mw_NJ[0]
                && st_amx_q4k_unext[1] >= st_amx_mw_NJ[1])
                break;
        }
    }
    return NULL;
}

static int st_amx_ux_start(void) {
    if (st_amx_ux_dead) return 0;
    if (!st_amx_ux_n) {
        for (int i = 0; i < ST_AMX_UX; i++) {
            if (pthread_create(&st_amx_ux_th[i], NULL, st_amx_ux_main,
                               NULL) != 0) { st_amx_ux_dead = 1; break; }
            st_amx_ux_n++;
        }
        if (!st_amx_ux_n) return 0;
    }
    return 1;
}

/* ---- persistent AMX worker pool --------------------------------------
 * dispatch_apply bursts are too short (~150us) for the scheduler to migrate
 * blocks onto different P-clusters, so the 2nd AMX unit stayed idle. A
 * persistent thread settles on the 2nd cluster and only pays wake/join. */
static void st_amx_run_range(const StW16Blob* wb, int N, int K, int NC,
                             int rt0, int rt1, int wid, int B,
                             float* const* ys,
                             const float* const* xs, int solo, int maybe);
static void st_amx_run_range_q4k(const uint8_t* tbase, int N, int K, int NC,
                                 int rt0, int rt1, int wid, int B,
                                 float* const* ys,
                                 const float* const* xs, int solo, int maybe);
static const uint8_t* st_amx_jq4k = NULL;   /* q4k-mode tensor base for pool */

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
        if (st_amx_jq4k)
            st_amx_run_range_q4k(st_amx_jq4k, st_amx_jN, st_amx_jK,
                                 st_amx_jNC, st_amx_jr0, st_amx_jr1, 1,
                                 st_amx_jB, st_amx_jys, st_amx_jxs, 0,
                                 st_amx_jhit);
        else
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
    int NG = K / 32, NCH = (K + 255) / 256;   /* NCH == nb: one drain per Q4_K block */
    size_t cnch = (st_amx_capK + 255) / 256;  /* per-worker slab region sized by capacity */
    int32_t* zb0 = st_amx_zb + (size_t)wid * ST_AMX_SLAB * cnch * 1024;
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
    if (__atomic_load_n(&st_amx_mw_on, __ATOMIC_ACQUIRE) && !solo) {
        /* issue-only: merge lanes consume via slot-busy backpressure */
        double w0 = st_amx_prof ? st_amx_us() : 0;
        for (int i = 0; i < NJ; i++) {
            int tt = rt0 + i / NC, cc2 = i % NC;
            int32_t* msj = st_amx_ms + ((size_t)cc2 * NRT + tt) * (size_t)NCH * 1024;
            if (!wb->q6)
            for (int blk = 0; blk < NCH; blk++)
                st_amx_issue_raw(wb->m16 + (size_t)tt * NG * 32 + (size_t)blk * 8 * 32,
                                 st_amx_vb + (size_t)cc2 * NG * 32 + (size_t)blk * 8 * 32, 8,
                                 msj + (size_t)blk * 1024, 0);
            int zsel = i & (ST_AMX_SLAB - 1);
            while (__atomic_load_n(&st_amx_slot_busy[wid][zsel], __ATOMIC_ACQUIRE)) { }
            __atomic_store_n(&st_amx_slot_busy[wid][zsel], 1, __ATOMIC_RELAXED);
            st_amx_issue(wb, tt, K,
                         st_amx_xw + (size_t)cc2 * K * 32, zb0, zsel);
            __atomic_store_n(&st_amx_mw_issued[wid], i + 1, __ATOMIC_RELEASE);
        }
        while (__atomic_load_n(&st_amx_mw_done[wid], __ATOMIC_ACQUIRE) < NJ) { }
        if (st_amx_prof) st_amx_wtime[wid] = st_amx_us() - w0;
        ST_AMX_CLR();
        return;
    }
    for (int i = 0; i < ST_AMX_LAG && i < NJ; i++) {
        int tt = rt0 + i / NC, cc2 = i % NC;
        int32_t* msj = st_amx_ms + ((size_t)cc2 * NRT + tt) * (size_t)NCH * 1024;
        if (!wb->q6)
        for (int blk = 0; blk < NCH; blk++)
            st_amx_issue_raw(wb->m16 + (size_t)tt * NG * 32 + (size_t)blk * 8 * 32,
                             st_amx_vb + (size_t)cc2 * NG * 32 + (size_t)blk * 8 * 32, 8,
                             msj + (size_t)blk * 1024, 0);
        st_amx_issue(wb, tt, K,
                     st_amx_xw + (size_t)cc2 * K * 32, zb0, i & (ST_AMX_SLAB - 1));
    }
    double w0 = st_amx_prof ? st_amx_us() : 0;
    for (int i = 0; i < NJ; i++) {
        if (i + ST_AMX_LAG < NJ) {
            int j = i + ST_AMX_LAG;
            int tt = rt0 + j / NC, cc2 = j % NC;
            int32_t* msj = st_amx_ms + ((size_t)cc2 * NRT + tt) * (size_t)NCH * 1024;
            double _ti = st_amx_prof > 1 ? st_amx_us() : 0;
            if (!wb->q6)
            for (int blk = 0; blk < NCH; blk++)
                st_amx_issue_raw(wb->m16 + (size_t)tt * NG * 32 + (size_t)blk * 8 * 32,
                                 st_amx_vb + (size_t)cc2 * NG * 32 + (size_t)blk * 8 * 32, 8,
                                 msj + (size_t)blk * 1024, 0);
            st_amx_issue(wb, tt, K,
                         st_amx_xw + (size_t)cc2 * K * 32, zb0, j & (ST_AMX_SLAB - 1));
            if (st_amx_prof > 1) st_amx_t_issue[wid] += st_amx_us() - _ti;
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
                for (int pf = 0; pf < K * 32; pf += 2048)
                    __builtin_prefetch((const char*)nw + beg * 2 + pf, 0, 0);
            }
        st_amx_merge(wb->q6,
                     wb->d + (size_t)rt * 32 * NCH,
                     wb->dm + (size_t)rt * 32 * NCH,
                     rt, s0, Bc, K, N, zb0, i & (ST_AMX_SLAB - 1), ys,
                     st_amx_xs1 + c * 32,
                     st_amx_ms + ((size_t)c * NRT + rt) * (size_t)NCH * 1024, wid);
    }
    if (st_amx_prof) st_amx_wtime[wid] = st_amx_us() - w0;
    ST_AMX_CLR();
}

/* ==== runtime Q4_K -> w16-tile unpack (STRATUM_AMX_Q4K, values unchanged) ====
 * Same transform as type-43 (w16[k] = n*sc exact i16, d/dm per 256-block,
 * m16 group mins) but produced per 32-row tile at matmul time, so plain
 * Q4_K models can run the AMX engine without a converted file. Scratch is
 * per (worker, drain-slot): the slot ring already guarantees the previous
 * job using that slot has fully drained/merged before reuse. */
static int st_amx_q4k_ensure(int K) {
    if ((size_t)K <= st_amx_q4k_capK) return 1;
    int NG = K / 32, NCH = K / 256;
    int16_t* wt = (int16_t*)malloc((size_t)ST_AMX_NW * ST_AMX_SLAB * K * 32 * 2);
    int16_t* mt = (int16_t*)malloc((size_t)ST_AMX_NW * ST_AMX_SLAB * NG * 32 * 2);
    float*   dd = (float*)  malloc((size_t)ST_AMX_NW * ST_AMX_SLAB * 64 * NCH * 4);
    if (!wt || !mt || !dd) { free(wt); free(mt); free(dd); return 0; }
    free(st_amx_q4k_wt); free(st_amx_q4k_mt); free(st_amx_q4k_dd);
    st_amx_q4k_wt = wt; st_amx_q4k_mt = mt; st_amx_q4k_dd = dd;
    st_amx_q4k_capK = (size_t)K;
    return 1;
}

/* unpack 32 Q4_K rows (tile rt) into the w16-tile scratch layout:
 * wt[k*32+ri] = n*sc, mt[g*32+ri] = m, dd[ri*NCH+i] = d, dd[32*NCH + ri*NCH+i] = dm
 * Two-phase: rows dequant into a row-major temp (contiguous NEON stores),
 * then an 8x8 i16 transpose yields the k-major tile AMX streams. */
static void st_q4k_unpack_tile(const uint8_t* tbase, int rt, int K,
                               int16_t* wt, int16_t* mt, float* dd) {
    int nb = K / 256;
    int16_t wtmp[32 * 256];            /* one 256-block, 32 rows, row-major */
    for (int i = 0; i < nb; i++) {
        for (int ri = 0; ri < 32; ri++) {
            const block_q4_K* b = (const block_q4_K*)
                (tbase + ((size_t)rt * 32 + ri) * (size_t)nb * 144
                 + (size_t)i * 144);
            dd[ri * nb + i]           = q4k_fp16_to_fp32(b->d);
            dd[32 * nb + ri * nb + i] = q4k_fp16_to_fp32(b->dmin);
            int16_t* wr = wtmp + (size_t)ri * 256;
            for (int js = 0; js < 256; js += 64) {
                int is_ = js / 32;
                uint8_t sc1, m1, sc2, m2;
                q4k_get_scale_min(is_,     b->scales, &sc1, &m1);
                q4k_get_scale_min(is_ + 1, b->scales, &sc2, &m2);
                int g0 = i * 8 + is_, g1 = g0 + 1;
                mt[(size_t)g0 * 32 + ri] = (int16_t)m1;
                mt[(size_t)g1 * 32 + ri] = (int16_t)m2;
                const uint8_t* q = b->qs + (js / 64) * 32;
                int kb = js;
                if (getenv("STRATUM_UPK_SCALAR")) {
                    for (int j = 0; j < 32; j++) {
                        wr[kb + j]      = (int16_t)((q[j] & 0xF) * sc1);
                        wr[kb + j + 32] = (int16_t)((q[j] >> 4) * sc2);
                    }
                    continue;
                }
                uint8x16_t qv0 = vld1q_u8(q), qv1 = vld1q_u8(q + 16);
                uint8x16_t lo0 = vandq_u8(qv0, vdupq_n_u8(0xF));
                uint8x16_t lo1 = vandq_u8(qv1, vdupq_n_u8(0xF));
                uint8x16_t hi0 = vshrq_n_u8(qv0, 4);
                uint8x16_t hi1 = vshrq_n_u8(qv1, 4);
                int16x8_t s1 = vdupq_n_s16(sc1), s2 = vdupq_n_s16(sc2);
                int16x8_t a0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(lo0)));
                int16x8_t a1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(lo0)));
                int16x8_t a2 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(lo1)));
                int16x8_t a3 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(lo1)));
                int16x8_t b0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(hi0)));
                int16x8_t b1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(hi0)));
                int16x8_t b2 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(hi1)));
                int16x8_t b3 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(hi1)));
                vst1q_s16(wr + kb +  0, vmulq_s16(a0, s1));
                vst1q_s16(wr + kb +  8, vmulq_s16(a1, s1));
                vst1q_s16(wr + kb + 16, vmulq_s16(a2, s1));
                vst1q_s16(wr + kb + 24, vmulq_s16(a3, s1));
                vst1q_s16(wr + kb + 32, vmulq_s16(b0, s2));
                vst1q_s16(wr + kb + 40, vmulq_s16(b1, s2));
                vst1q_s16(wr + kb + 48, vmulq_s16(b2, s2));
                vst1q_s16(wr + kb + 56, vmulq_s16(b3, s2));
            }
        }
        /* transpose wtmp[32][256] -> wt[(i*256 + k)*32 + ri] */
        int16_t* wtb = wt + (size_t)i * 256 * 32;
        for (int k0 = 0; k0 < 256; k0 += 8)
            for (int r0 = 0; r0 < 32; r0 += 8) {
                int16x8_t r0v = vld1q_s16(wtmp + (size_t)(r0 + 0) * 256 + k0);
                int16x8_t r1v = vld1q_s16(wtmp + (size_t)(r0 + 1) * 256 + k0);
                int16x8_t r2v = vld1q_s16(wtmp + (size_t)(r0 + 2) * 256 + k0);
                int16x8_t r3v = vld1q_s16(wtmp + (size_t)(r0 + 3) * 256 + k0);
                int16x8_t r4v = vld1q_s16(wtmp + (size_t)(r0 + 4) * 256 + k0);
                int16x8_t r5v = vld1q_s16(wtmp + (size_t)(r0 + 5) * 256 + k0);
                int16x8_t r6v = vld1q_s16(wtmp + (size_t)(r0 + 6) * 256 + k0);
                int16x8_t r7v = vld1q_s16(wtmp + (size_t)(r0 + 7) * 256 + k0);
                /* 8x8 i16 transpose via 16->32->64 bit zips */
                int16x8_t t0 = vtrn1q_s16(r0v, r1v), t1 = vtrn2q_s16(r0v, r1v);
                int16x8_t t2 = vtrn1q_s16(r2v, r3v), t3 = vtrn2q_s16(r2v, r3v);
                int16x8_t t4 = vtrn1q_s16(r4v, r5v), t5 = vtrn2q_s16(r4v, r5v);
                int16x8_t t6 = vtrn1q_s16(r6v, r7v), t7 = vtrn2q_s16(r6v, r7v);
                int32x4_t u0 = vtrn1q_s32(vreinterpretq_s32_s16(t0), vreinterpretq_s32_s16(t2));
                int32x4_t u1 = vtrn1q_s32(vreinterpretq_s32_s16(t4), vreinterpretq_s32_s16(t6));
                int32x4_t u2 = vtrn2q_s32(vreinterpretq_s32_s16(t0), vreinterpretq_s32_s16(t2));
                int32x4_t u3 = vtrn2q_s32(vreinterpretq_s32_s16(t4), vreinterpretq_s32_s16(t6));
                int32x4_t u4 = vtrn1q_s32(vreinterpretq_s32_s16(t1), vreinterpretq_s32_s16(t3));
                int32x4_t u5 = vtrn1q_s32(vreinterpretq_s32_s16(t5), vreinterpretq_s32_s16(t7));
                int32x4_t u6 = vtrn2q_s32(vreinterpretq_s32_s16(t1), vreinterpretq_s32_s16(t3));
                int32x4_t u7 = vtrn2q_s32(vreinterpretq_s32_s16(t5), vreinterpretq_s32_s16(t7));
                int16x8_t o0 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s32(u0), vreinterpretq_s64_s32(u1)));
                int16x8_t o1 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s32(u4), vreinterpretq_s64_s32(u5)));
                int16x8_t o2 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s32(u2), vreinterpretq_s64_s32(u3)));
                int16x8_t o3 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s32(u6), vreinterpretq_s64_s32(u7)));
                int16x8_t o4 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s32(u0), vreinterpretq_s64_s32(u1)));
                int16x8_t o5 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s32(u4), vreinterpretq_s64_s32(u5)));
                int16x8_t o6 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s32(u2), vreinterpretq_s64_s32(u3)));
                int16x8_t o7 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s32(u6), vreinterpretq_s64_s32(u7)));
                vst1q_s16(wtb + (size_t)(k0 + 0) * 32 + r0, o0);
                vst1q_s16(wtb + (size_t)(k0 + 1) * 32 + r0, o1);
                vst1q_s16(wtb + (size_t)(k0 + 2) * 32 + r0, o2);
                vst1q_s16(wtb + (size_t)(k0 + 3) * 32 + r0, o3);
                vst1q_s16(wtb + (size_t)(k0 + 4) * 32 + r0, o4);
                vst1q_s16(wtb + (size_t)(k0 + 5) * 32 + r0, o5);
                vst1q_s16(wtb + (size_t)(k0 + 6) * 32 + r0, o6);
                vst1q_s16(wtb + (size_t)(k0 + 7) * 32 + r0, o7);
            }
    }
}

/* q4k twin of st_amx_run_range: identical pipeline, but each issue step
 * unpacks the tile's Q4_K rows into this slot's scratch instead of reading
 * a w16 blob. Merge lanes resolve d/dm via st_amx_q4k_dd[(v,slot)]. */
static void st_amx_run_range_q4k(const uint8_t* tbase, int N, int K, int NC,
                                 int rt0, int rt1, int wid, int B,
                                 float* const* ys,
                                 const float* const* xs, int solo, int maybe) {
    int NG = K / 32, NCH = (K + 255) / 256;
    size_t cnch = (st_amx_capK + 255) / 256;
    int32_t* zb0 = st_amx_zb + (size_t)wid * ST_AMX_SLAB * cnch * 1024;
    int h0 = solo ? 0 : wid, h1 = solo ? 2 : wid + 1;
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
    if (!solo)
        while (!__atomic_load_n(&st_amx_fl[(1 - wid) * 2], __ATOMIC_ACQUIRE)) { }
    ST_AMX_SET();
    int NRT = N / 32;
    int NT = rt1 - rt0, NJ = NT * NC;
    size_t sbase = (size_t)wid * ST_AMX_SLAB;
#define ST_Q4K_WT(z) (st_amx_q4k_wt + (sbase + (z)) * (size_t)K * 32)
#define ST_Q4K_MT(z) (st_amx_q4k_mt + (sbase + (z)) * (size_t)NG * 32)
#define ST_Q4K_DD(z) (st_amx_q4k_dd + (sbase + (z)) * 64 * NCH)
    /* issue one job: wait slot free, unpack tile into slot scratch, then
     * minterm + main issue. Shared by both mw and in-loop pipelines. */
#define ST_Q4K_ISSUE(i_)                                                   \
    do {                                                                   \
        int _i = (i_);                                                     \
        int _tt = rt0 + _i / NC, _cc = _i % NC;                            \
        int _z = _i & (ST_AMX_SLAB - 1);                                   \
        int32_t* _msj = st_amx_ms +                                        \
            ((size_t)_cc * NRT + _tt) * (size_t)NCH * 1024;                \
        double _t0 = st_amx_prof ? st_amx_us() : 0;                        \
        if (__atomic_load_n(&st_amx_mw_on, __ATOMIC_ACQUIRE) && !solo) {   \
            /* ux threads unpack ahead; while waiting for job i the       \
             * issuer steals ANY unclaimed job (not just i) — real work   \
             * instead of a spin that would starve the ux threads. */     \
            int _sp = 0;                                                   \
            for (;;) {                                                     \
                if (__atomic_load_n(&st_amx_q4k_uready[wid][_z],           \
                                    __ATOMIC_ACQUIRE) == _i + 1) break;    \
                int _u = st_amx_q4k_unext[wid];                            \
                int _uz = _u & (ST_AMX_SLAB - 1);                          \
                if (_u < NJ                                                \
                    && __atomic_load_n(&st_amx_q4k_uready[wid][_uz],       \
                                       __ATOMIC_ACQUIRE) != _u + 1         \
                    && !__atomic_load_n(&st_amx_slot_busy[wid][_uz],       \
                                        __ATOMIC_ACQUIRE)                  \
                    && __atomic_compare_exchange_n(                        \
                           (volatile int*)&st_amx_q4k_unext[wid],          \
                           &_u, _u + 1, 0, __ATOMIC_RELAXED,               \
                           __ATOMIC_RELAXED)) {                            \
                    __atomic_store_n(&st_amx_slot_busy[wid][_uz], 1,       \
                                     __ATOMIC_RELEASE);                    \
                    st_q4k_unpack_tile(tbase, rt0 + _u / NC, K,            \
                                       ST_Q4K_WT(_uz), ST_Q4K_MT(_uz),     \
                                       ST_Q4K_DD(_uz));                    \
                    __atomic_store_n(&st_amx_q4k_uready[wid][_uz], _u + 1, \
                                     __ATOMIC_RELEASE);                    \
                    if (st_amx_prof)                                       \
                        __atomic_add_fetch(&st_amx_is_claims, 1,           \
                                           __ATOMIC_RELAXED);              \
                    continue;                                              \
                }                                                          \
                if ((++_sp & 0x3FF) == 0) sched_yield();                   \
            }                                                              \
            __atomic_store_n(&st_amx_slot_busy[wid][_z], 1, __ATOMIC_RELAXED); \
        } else                                                             \
            st_q4k_unpack_tile(tbase, _tt, K, ST_Q4K_WT(_z),               \
                               ST_Q4K_MT(_z), ST_Q4K_DD(_z));              \
        double _t2 = st_amx_prof ? st_amx_us() : 0;                        \
        for (int blk = 0; blk < NCH; blk++)                                \
            st_amx_issue_raw(ST_Q4K_MT(_z) + (size_t)blk * 8 * 32,         \
                             st_amx_vb + (size_t)_cc * NG * 32             \
                                       + (size_t)blk * 8 * 32,             \
                             8, _msj + (size_t)blk * 1024, 0);             \
        st_amx_issue_raw(ST_Q4K_WT(_z),                                    \
                         st_amx_xw + (size_t)_cc * K * 32, K,              \
                         zb0 + (size_t)_z * NCH * 1024, 0);                \
        if (__atomic_load_n(&st_amx_mw_on, __ATOMIC_ACQUIRE) && !solo)     \
            __atomic_store_n(&st_amx_mw_issued[wid], _i + 1,               \
                             __ATOMIC_RELEASE);                            \
        if (st_amx_prof) {                                                 \
            st_amx_p_wait[wid] += _t2 - _t0;                               \
            st_amx_p_iss[wid]  += st_amx_us() - _t2;                       \
        }                                                                  \
    } while (0)

    if (__atomic_load_n(&st_amx_mw_on, __ATOMIC_ACQUIRE) && !solo) {
        double w0 = st_amx_prof ? st_amx_us() : 0;
        for (int i = 0; i < NJ; i++) ST_Q4K_ISSUE(i);
        while (__atomic_load_n(&st_amx_mw_done[wid], __ATOMIC_ACQUIRE) < NJ) { }
        if (st_amx_prof) st_amx_wtime[wid] = st_amx_us() - w0;
        ST_AMX_CLR();
        return;
    }
    for (int i = 0; i < ST_AMX_LAG && i < NJ; i++) ST_Q4K_ISSUE(i);
    double w0 = st_amx_prof ? st_amx_us() : 0;
    for (int i = 0; i < NJ; i++) {
        if (i + ST_AMX_LAG < NJ) ST_Q4K_ISSUE(i + ST_AMX_LAG);
        int c = i % NC, rt = rt0 + i / NC;
        int s0 = c * 32, Bc = (B - s0 < 32) ? B - s0 : 32;
        int zsel = i & (ST_AMX_SLAB - 1);
        /* prefetch the Q4_K source of a couple tiles ahead */
        if (i + 2 < NJ) {
            const uint8_t* nw = tbase +
                (size_t)(rt0 + (i + 2) / NC) * 32 * (size_t)NCH * 144;
            for (int pf = 0; pf < 32 * NCH * 144; pf += 2048)
                __builtin_prefetch((const char*)nw + pf, 0, 0);
        }
        float* dd = ST_Q4K_DD(zsel);
        st_amx_merge(0, dd, dd + (size_t)32 * NCH,
                     rt, s0, Bc, K, N, zb0, zsel, ys,
                     st_amx_xs1 + c * 32,
                     st_amx_ms + ((size_t)c * NRT + rt) * (size_t)NCH * 1024, wid);
    }
    if (st_amx_prof) st_amx_wtime[wid] = st_amx_us() - w0;
    ST_AMX_CLR();
#undef ST_Q4K_WT
#undef ST_Q4K_MT
#undef ST_Q4K_DD
#undef ST_Q4K_ISSUE
}

/* AMX multiseq entry for plain Q4_K tensors (runtime unpack).
 * Returns 0 when scratch can't be allocated — caller runs the SDOT path. */
static inline int st_q4k_amx_multix(const GgufTensor* w,
                                    const float* const* xs,
                                    float* const* ys,
                                    int B, int N, int K) {
    int NG = K / 32;
    if (st_amx_prof < 0) { const char* e = getenv("STRATUM_AMX_PROF"); st_amx_prof = e ? atoi(e) : 0; }
    const uint8_t* tbase = g_st.mmap_base + w->offset;
    int NC = (B + 31) / 32;
    if (NG > 1024 || !st_amx_ensure(N, K, NC, 1) || !st_amx_q4k_ensure(K))
        return 0;   /* SDOT caller path handles the fallback */
    memset(st_amx_zp, 0, 128);
    if (st_amx_prof) {
        st_amx_p_wait[0] = st_amx_p_wait[1] = 0;
        st_amx_p_iss[0] = st_amx_p_iss[1] = 0;
        st_amx_ux_claims = st_amx_is_claims = 0;
    }
    double t_mm = 0, tt;
    tt = st_amx_us();
    int maybe = 0;
    if (NC <= 4) {
        maybe = (B == st_amx_cB && K == st_amx_cK && NC == st_amx_cNC);
        if (maybe)
            for (int b = 0; b < B; b++)
                if (xs[b] != st_amx_cxp[b]) { maybe = 0; break; }
        for (int b = 0; b < B; b++) st_amx_cxp[b] = xs[b];
        st_amx_cB = B; st_amx_cK = K; st_amx_cNC = NC;
    } else st_amx_cB = -1;
    int NT_ = N / 32;
    int mp = 0;
    if (NT_ >= 4 && !getenv("STRATUM_AMX_SOLO") && !getenv("STRATUM_AMX_NOMW")
        && st_amx_pool_start() && st_amx_mw_start() && st_amx_ux_start()) {
        st_amx_mw_q4k = 1;
        st_amx_mw_q4k_src = tbase;
        st_amx_mw_N = N; st_amx_mw_K = K; st_amx_mw_NC = NC; st_amx_mw_B = B;
        st_amx_mw_NRT = N / 32; st_amx_mw_NCH = (K + 255) / 256;
        st_amx_mw_rt0[0] = 0;        st_amx_mw_rt0[1] = NT_ / 2;
        st_amx_mw_NJ[0] = (NT_ / 2) * NC;
        st_amx_mw_NJ[1] = (NT_ - NT_ / 2) * NC;
        st_amx_mw_ys = ys; st_amx_mw_xs1 = st_amx_xs1;
        for (int v = 0; v < ST_AMX_NW; v++) {
            __atomic_store_n(&st_amx_mw_next[v], 0, __ATOMIC_RELAXED);
            __atomic_store_n(&st_amx_mw_issued[v], 0, __ATOMIC_RELAXED);
            st_amx_mw_done[v] = 0;
            for (int z = 0; z < ST_AMX_SLAB; z++) {
                st_amx_slot_busy[v][z] = 0;
                st_amx_q4k_uready[v][z] = 0;
            }
            /* claim gate reset last: a ux claim that sees unext==0 must
             * already observe cleared slot state (release ordering) */
            __atomic_store_n(&st_amx_q4k_unext[v], 0, __ATOMIC_RELEASE);
        }
        __atomic_store_n(&st_amx_mw_on, 1, __ATOMIC_RELEASE);
        unsigned mseq = __atomic_add_fetch(&st_amx_mw_seq, 1, __ATOMIC_RELEASE);
        st_amx_mw_myseq = mseq;
        if (__atomic_load_n(&st_amx_mw_parked, __ATOMIC_ACQUIRE) > 0) {
            pthread_mutex_lock(&st_amx_mw_mu);
            pthread_cond_broadcast(&st_amx_mw_cv);
            pthread_mutex_unlock(&st_amx_mw_mu);
        }
        mp = 1;
    } else __atomic_store_n(&st_amx_mw_on, 0, __ATOMIC_RELEASE);
    if (NT_ >= 4 && !getenv("STRATUM_AMX_SOLO") && st_amx_pool_start()) {
        st_amx_jq4k = tbase;
        st_amx_jN = N; st_amx_jK = K; st_amx_jNC = NC;
        st_amx_jB = B; st_amx_jr0 = NT_ / 2; st_amx_jr1 = NT_;
        st_amx_jys = ys; st_amx_jxs = xs; st_amx_jhit = maybe;
        st_amx_fl[0] = st_amx_fl[1] = st_amx_fl[2] = st_amx_fl[3] = 0;
        unsigned s = __atomic_add_fetch(&st_amx_pool_seq, 1, __ATOMIC_RELEASE);
        if (st_amx_pool_parked) {
            pthread_mutex_lock(&st_amx_pool_mu);
            pthread_cond_signal(&st_amx_pool_go);
            pthread_mutex_unlock(&st_amx_pool_mu);
        }
        st_amx_run_range_q4k(tbase, N, K, NC, 0, NT_ / 2, 0, B, ys, xs, 0, maybe);
        while (__atomic_load_n(&st_amx_pool_done, __ATOMIC_ACQUIRE) != (int)s) { }
    } else {
        st_amx_run_range_q4k(tbase, N, K, NC, 0, NT_, 0, B, ys, xs, 1, maybe);
    }
    if (mp)
        for (int i = 0; i < st_amx_mw_n; i++)
            while (__atomic_load_n(&st_amx_mw_ack[i], __ATOMIC_ACQUIRE)
                   != st_amx_mw_myseq) { }
    t_mm += st_amx_us() - tt;
    if (st_amx_prof)
        fprintf(stderr, "[amxprof-q4k] total=%.0f us (w0_jobs=%.0f w1_jobs=%.0f hits=%d | w0 wt%.0f is%.0f | w1 wt%.0f is%.0f | ux=%.0f self=%.0f)\n",
                t_mm, st_amx_wtime[0], st_amx_wtime[1], st_amx_hits,
                st_amx_p_wait[0], st_amx_p_iss[0],
                st_amx_p_wait[1], st_amx_p_iss[1],
                st_amx_ux_claims, st_amx_is_claims);
    (void)NG;
    return 1;
}

/* Portable fallback (no AMX): correct but slow — used when AMX probe fails
 * or STRATUM_AMX_W16_OFF=1. Reconstructs f32 weights from the w16 blob. */
static void st_q4k_w16_matvec_fallback(const GgufTensor* w,
                                       const float* const* xs,
                                       float* const* ys,
                                       int B, int N, int K) {
    int NG = K / 32, nb = K / 256;
    int q6 = (w->type == 44);   /* GGML_TYPE_Q6K_W16: no minterm */
    StW16Blob wb = st_w16_blob(g_st.mmap_base + w->offset, N, K, q6);
    for (int b = 0; b < B; b++)
        for (int r = 0; r < N; r++) {
            const int16_t* wrow = wb.w16 + (size_t)(r / 32) * K * 32 + (r % 32);
            double y = 0;
            for (int i = 0; i < nb; i++) {           /* per 256-block d/dm */
                double acc = 0;
                for (int k = i * 256; k < i * 256 + 256; k++)
                    acc += (double)wrow[(size_t)k * 32] * xs[b][k];
                double msum = 0;
                if (!q6)
                for (int g = i * 8; g < i * 8 + 8; g++) {
                    double s2 = 0;
                    for (int j = 0; j < 32; j++) s2 += xs[b][g * 32 + j];
                    msum += (double)wb.m16[(size_t)(r / 32) * NG * 32 + g * 32 + r % 32] * s2;
                }
                y += wb.d[(size_t)r * nb + i] * acc - wb.dm[(size_t)r * nb + i] * msum;
            }
            ys[b][r] = (float)y;
        }
}

/* AMX multiseq entry: process B seqs in chunks of 32 (one Z tile of seqs). */
static inline void st_q4k_w16_amx_multix(const GgufTensor* w,
                                         const float* const* xs,
                                         float* const* ys,
                                         int B, int N, int K) {
    int NG = K / 32;
    if (st_amx_prof < 0) { const char* e = getenv("STRATUM_AMX_PROF"); st_amx_prof = e ? atoi(e) : 0; }
    StW16Blob wb = st_w16_blob(g_st.mmap_base + w->offset, N, K,
                               w->type == 44 /* GGML_TYPE_Q6K_W16 */);
    int NC = (B + 31) / 32;                 /* seq-chunks (32 seqs per Z tile) */
    /* minterm drains once per tile: NG*63*32736 < INT32_MAX needs NG<=1040 */
    if (NG > 1024 || !st_amx_ensure(N, K, NC, w->type != 44)) {
        st_q4k_w16_matvec_fallback(w, xs, ys, B, N, K); return; }
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
    int mp = 0;
    if (NT_ >= 4 && !getenv("STRATUM_AMX_SOLO") && !getenv("STRATUM_AMX_NOMW")
        && st_amx_pool_start() && st_amx_mw_start()) {
        st_amx_mw_w = &wb;
        st_amx_mw_q4k = 0;
        st_amx_mw_N = N; st_amx_mw_K = K; st_amx_mw_NC = NC; st_amx_mw_B = B;
        st_amx_mw_NRT = N / 32; st_amx_mw_NCH = (K + 255) / 256;
        st_amx_mw_rt0[0] = 0;        st_amx_mw_rt0[1] = NT_ / 2;
        st_amx_mw_NJ[0] = (NT_ / 2) * NC;
        st_amx_mw_NJ[1] = (NT_ - NT_ / 2) * NC;
        st_amx_mw_ys = ys; st_amx_mw_xs1 = st_amx_xs1;
        for (int v = 0; v < ST_AMX_NW; v++) {
            __atomic_store_n(&st_amx_mw_next[v], 0, __ATOMIC_RELAXED);
            __atomic_store_n(&st_amx_mw_issued[v], 0, __ATOMIC_RELAXED);
            st_amx_mw_done[v] = 0;
            for (int z = 0; z < ST_AMX_SLAB; z++) st_amx_slot_busy[v][z] = 0;
        }
        __atomic_store_n(&st_amx_mw_on, 1, __ATOMIC_RELEASE);
        unsigned mseq = __atomic_add_fetch(&st_amx_mw_seq, 1, __ATOMIC_RELEASE);
        st_amx_mw_myseq = mseq;
        if (__atomic_load_n(&st_amx_mw_parked, __ATOMIC_ACQUIRE) > 0) {
            pthread_mutex_lock(&st_amx_mw_mu);
            pthread_cond_broadcast(&st_amx_mw_cv);
            pthread_mutex_unlock(&st_amx_mw_mu);
        }
        mp = 1;
    } else __atomic_store_n(&st_amx_mw_on, 0, __ATOMIC_RELEASE);
    if (NT_ >= 4 && !getenv("STRATUM_AMX_SOLO") && st_amx_pool_start()) {
        st_amx_jq4k = NULL;
        st_amx_run_parallel(&wb, N, K, NC, NT_, B, ys, xs, maybe);
    } else {
        st_amx_run_range(&wb, N, K, NC, 0, NT_, 0, B, ys, xs, 1, maybe);
    }
    /* merge-lane ack barrier: a lane may still be inside its inner job loop
     * (polling next/issued) after both workers' done counts hit NJ — if we
     * returned now, the next call would reset next/issued/done under it and
     * it would merge new jobs with a dead-stack `w` (wb is a local). Wait
     * for every lane to ack this seq (it acks only after exiting the loop)
     * before letting the stack frame die. */
    if (mp)
        for (int i = 0; i < st_amx_mw_n; i++)
            while (__atomic_load_n(&st_amx_mw_ack[i], __ATOMIC_ACQUIRE)
                   != st_amx_mw_myseq) { }
    t_mm += st_amx_us() - tt;
    if (st_amx_prof)
        fprintf(stderr, "[amxprof] total=%.0f us (w0_jobs=%.0f w1_jobs=%.0f hits=%d)\n",
                t_mm, st_amx_wtime[0], st_amx_wtime[1], st_amx_hits);
    if (st_amx_prof > 1)
        fprintf(stderr, "[amxprof2] w0: issue=%.0f wait=%.0f merge=%.0f | w1: %.0f %.0f %.0f us\n",
                st_amx_t_issue[0], st_amx_t_wait[0], st_amx_t_merge[0],
                st_amx_t_issue[1], st_amx_t_wait[1], st_amx_t_merge[1]);
    (void)NG;
}
#endif
