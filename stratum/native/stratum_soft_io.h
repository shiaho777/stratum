#ifndef STRATUM_SOFT_IO_H
#define STRATUM_SOFT_IO_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <time.h>

typedef struct {
    size_t start;
    size_t end;
} StratumLayerSpan;

typedef struct {
    const void* mmap_base;
    size_t mmap_size;
    int fd;
    int keep_resident;

    int soft_warm;
    size_t prefix_end;
    size_t prefix_init;

    size_t hot_end;
    double hot_end_t;
    size_t stage_horizon;
    double stage_horizon_t;

    double last_main_s;
    int main_scans;

    long reaffirm_n;
    long reaffirm_bytes;
    int reaffirm_last_li;
    double reaffirm_last_t;

    long frontier_advances;
    long frontier_bytes;
    int frontier_li;

    long cold_stage;
    long truth_clamp_n;
    long horizon_heal_n;
    long horizon_stage_n;

    size_t cache_mres;
    double cache_mres_t;
    size_t cache_free;
    double cache_free_t;
} StratumSoftIo;

static inline size_t stratum_soft_page_align_down(size_t x) {
    size_t pgsz = 16384;
    long ps = 0; size_t pl = sizeof(ps);
    if (sysctlbyname("hw.pagesize", &ps, &pl, NULL, 0) == 0 && ps > 0) pgsz = (size_t)ps;
    return x & ~(pgsz - 1);
}

static inline size_t stratum_soft_page_align_up(size_t x) {
    size_t pgsz = 16384;
    long ps = 0; size_t pl = sizeof(ps);
    if (sysctlbyname("hw.pagesize", &ps, &pl, NULL, 0) == 0 && ps > 0) pgsz = (size_t)ps;
    return (x + pgsz - 1) & ~(pgsz - 1);
}

static inline void stratum_soft_io_init(StratumSoftIo* s,
                                       const void* mmap_base,
                                       size_t mmap_size,
                                       int fd) {
    if (!s) return;
    memset(s, 0, sizeof(*s));
    s->mmap_base = mmap_base;
    s->mmap_size = mmap_size;
    s->fd = fd;
    s->frontier_li = -1;
    s->reaffirm_last_li = -1;
}

static inline void stratum_soft_io_set_keep_resident(StratumSoftIo* s, int kr) {
    if (s) s->keep_resident = kr ? 1 : 0;
}

static inline void stratum_soft_io_enable(StratumSoftIo* s, size_t prefix_end) {
    if (!s) return;
    s->soft_warm = 1;
    s->prefix_end = prefix_end;
    if (s->prefix_init == 0) s->prefix_init = prefix_end;
}

static inline size_t stratum_soft_model_resident_bytes(const StratumSoftIo* s) {
    if (!s || !s->mmap_base || s->mmap_size == 0) return 0;
    double t = 0.0;
    {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        t = now.tv_sec + now.tv_nsec / 1e9;
    }
    StratumSoftIo* mut = (StratumSoftIo*)s;
    if (mut->cache_mres_t > 0.0 && (t - mut->cache_mres_t) < 0.085
        && mut->cache_mres > 0)
        return mut->cache_mres;
    size_t pgsz = 16384;
    long ps = 0; size_t pl = sizeof(ps);
    if (sysctlbyname("hw.pagesize", &ps, &pl, NULL, 0) == 0 && ps > 0) pgsz = (size_t)ps;
    size_t sz = s->mmap_size;
    size_t np = (sz + pgsz - 1) / pgsz;
    if (np == 0) return 0;
    int samples = 96;
    if ((size_t)samples > np) samples = (int)np;
    int hot = 0;
    for (int i = 0; i < samples; i++) {
        size_t pi = ((size_t)i * np) / (size_t)samples;
        if (pi >= np) pi = np - 1;
        char vec = 0;
        const char* addr = (const char*)s->mmap_base + pi * pgsz;
        if (mincore((void*)addr, pgsz, &vec) == 0 && (vec & 1)) hot++;
    }
    if (samples <= 0) return 0;
    size_t res = (size_t)(((double)hot / (double)samples) * (double)sz);
    mut->cache_mres = res;
    mut->cache_mres_t = t;
    return res;
}

static inline size_t stratum_soft_free_reclaim_bytes(void) {
    size_t free_reclaim = 0;
    size_t pgsz = 16384;
    long ps = 0; size_t pl = sizeof(ps);
    if (sysctlbyname("hw.pagesize", &ps, &pl, NULL, 0) == 0 && ps > 0) pgsz = (size_t)ps;
    vm_statistics64_data_t vm; mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          (host_info64_t)&vm, &cnt) == KERN_SUCCESS) {
        free_reclaim = ((size_t)vm.free_count + (size_t)vm.inactive_count
                      + (size_t)vm.purgeable_count
                      + (size_t)vm.speculative_count) * pgsz;
    }
    return free_reclaim;
}

static inline size_t stratum_soft_free_pages_bytes(void) {
    size_t pgsz = 16384;
    long ps = 0; size_t pl = sizeof(ps);
    if (sysctlbyname("hw.pagesize", &ps, &pl, NULL, 0) == 0 && ps > 0) pgsz = (size_t)ps;
    vm_statistics64_data_t vm; mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          (host_info64_t)&vm, &cnt) == KERN_SUCCESS)
        return ((size_t)vm.free_count + (size_t)vm.speculative_count) * pgsz;
    return 0;
}


static inline int stratum_soft_range_pages_hot(const StratumSoftIo* s,
                                              size_t start, size_t len, int samples) {
    if (!s || !s->mmap_base || len == 0) return 0;
    size_t pgsz = 16384;
    long ps = 0; size_t pl = sizeof(ps);
    if (sysctlbyname("hw.pagesize", &ps, &pl, NULL, 0) == 0 && ps > 0) pgsz = (size_t)ps;
    size_t end = start + len;
    if (end > s->mmap_size) end = s->mmap_size;
    if (end <= start) return 0;
    if (samples < 3) samples = 3;
    if (samples > 12) samples = 12;
    int hot = 0, total = 0;
    for (int i = 0; i < samples; i++) {
        size_t off = start;
        if (i == 0) off = start;
        else if (i == samples - 1) off = (end > pgsz) ? (end - pgsz) : start;
        else {
            size_t span = end - start;
            off = start + (span * (size_t)i) / (size_t)samples;
        }
        off = off - (off % pgsz);
        if (off >= s->mmap_size) continue;
        char v = 0;
        if (mincore((void*)((const char*)s->mmap_base + off), pgsz, &v) != 0) continue;
        total++;
        if (v & 1) hot++;
    }
    if (total <= 0) return 0;
    return (hot * 5 >= total * 4);
}

static inline size_t stratum_soft_leading_hot_end(const StratumSoftIo* s, size_t limit) {
    if (!s || !s->mmap_base || limit == 0) return 0;
    size_t ms = s->mmap_size;
    if (limit > ms) limit = ms;
    size_t chunk = (size_t)(96ULL << 20);
    size_t hot_end = 0;
    int miss = 0;
    for (size_t off = 0; off < limit; ) {
        size_t clen = chunk;
        if (off + clen > limit) clen = limit - off;
        if (clen < (size_t)(1ULL << 20)) break;
        if (stratum_soft_range_pages_hot(s, off, clen, 5)) {
            hot_end = off + clen;
            miss = 0;
        } else {
            miss++;
            if (miss >= 2) break;
        }
        off += clen;
    }
    return hot_end;
}

static inline double stratum_soft_now_s(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec + now.tv_nsec / 1e9;
}

static inline size_t stratum_soft_refresh_hot_end(StratumSoftIo* s) {
    if (!s || !s->mmap_base || s->mmap_size == 0) return 0;
    double t = stratum_soft_now_s();
    if (s->hot_end_t > 0.0 && (t - s->hot_end_t) < 0.055 && s->hot_end > 0)
        return s->hot_end;
    size_t lim = s->prefix_end > 0 ? s->prefix_end : s->mmap_size;
    if (lim > s->mmap_size) lim = s->mmap_size;
    if (lim < (size_t)(64ULL << 20)) lim = s->mmap_size;
    size_t he = stratum_soft_leading_hot_end(s, lim);
    size_t mres = stratum_soft_model_resident_bytes(s);
    if (mres > s->mmap_size) mres = s->mmap_size;
    if (mres > he + (size_t)(256ULL << 20)) {
        size_t floor = (size_t)((mres * 84) / 100);
        if (floor > lim) floor = lim;
        if (he < floor) he = floor;
    } else if (he < (size_t)(32ULL << 20) && mres > he) {
        he = mres < lim ? mres : lim;
    }
    s->hot_end = he;
    s->hot_end_t = t;
    return he;
}

static inline size_t stratum_soft_stage_horizon(StratumSoftIo* s) {
    if (!s || !s->soft_warm) return 0;
    size_t pe = s->prefix_end;
    size_t ms = s->mmap_size;
    if (pe == 0) return 0;
    if (pe > ms) pe = ms;
    double t = stratum_soft_now_s();
    if (s->stage_horizon_t > 0.0 && (t - s->stage_horizon_t) < 0.040
        && s->stage_horizon > 0)
        return s->stage_horizon;
    size_t he = stratum_soft_refresh_hot_end(s);
    size_t mres = stratum_soft_model_resident_bytes(s);
    if (mres > ms) mres = ms;
    double rf = (ms > 0) ? ((double)mres / (double)ms) : 0.0;
    size_t hz = pe;
    size_t gap = (pe > he) ? (pe - he) : 0;
    size_t floor = mres;
    if (he > floor) floor = he;
    if (floor < (size_t)((pe * 50) / 100))
        floor = (size_t)((pe * 50) / 100);
    if (floor > pe) floor = pe;
    int lie = (gap > (size_t)(768ULL << 20) && rf < 0.70)
           || (pe > 0 && he + (size_t)(1024ULL << 20) < pe && rf < 0.68)
           || (pe > 0 && mres + (size_t)(2048ULL << 20) < pe && rf < 0.65);
    int slow = (s->last_main_s > 0.0 && s->last_main_s > 2.95);
    int pressure = (s->last_main_s > 0.0 && s->last_main_s > 3.10)
                || rf < 0.55;
    if (lie || (slow && gap > (size_t)(256ULL << 20))) {
        size_t slack = pressure ? (size_t)(160ULL << 20) : (size_t)(256ULL << 20);
        if (slow && !pressure) slack = (size_t)(224ULL << 20);
        hz = floor + slack;
        if (hz > pe) hz = pe;
        size_t min_hz = (size_t)((pe * 55) / 100);
        if (pressure) min_hz = (size_t)((pe * 48) / 100);
        if (mres + (size_t)(128ULL << 20) > min_hz)
            min_hz = mres + (size_t)(128ULL << 20);
        if (min_hz > pe) min_hz = pe;
        if (hz < min_hz) hz = min_hz;
        s->horizon_stage_n++;
    }
    s->stage_horizon = hz;
    s->stage_horizon_t = t;
    return hz;
}

static inline int stratum_soft_tensor_in_prefix(const StratumSoftIo* s,
                                               size_t offset, size_t nbytes) {
    if (!s || !s->soft_warm) return 0;
    if (s->prefix_end == 0) return 1;
    size_t hz = ((StratumSoftIo*)s)->stage_horizon;
    if (hz == 0 || (s->stage_horizon_t <= 0.0))
        hz = stratum_soft_stage_horizon((StratumSoftIo*)s);
    if (hz == 0) hz = s->prefix_end;
    return offset + nbytes <= hz + (size_t)(64ULL << 20);
}

static inline int stratum_soft_tensor_past_prefix(const StratumSoftIo* s, size_t offset) {
    if (!s || !s->soft_warm || s->prefix_end == 0) return 0;
    size_t hz = ((StratumSoftIo*)s)->stage_horizon;
    if (hz == 0 || (s->stage_horizon_t <= 0.0))
        hz = stratum_soft_stage_horizon((StratumSoftIo*)s);
    if (hz == 0) hz = s->prefix_end;
    return offset >= hz;
}

static inline int stratum_soft_mostly_hot(const StratumSoftIo* s) {
    if (!s || !s->soft_warm) return 0;
    size_t ms = s->mmap_size;
    if (ms == 0) return 0;
    if (s->last_main_s > 3.10) return 0;
    size_t mres = stratum_soft_model_resident_bytes(s);
    if (mres < (size_t)((ms * 72) / 100)) return 0;
    size_t he = s->hot_end;
    if (he == 0 || s->hot_end_t <= 0.0)
        he = stratum_soft_refresh_hot_end((StratumSoftIo*)s);
    if (he + (size_t)(512ULL << 20) < s->prefix_end
        && mres < (size_t)((ms * 80) / 100))
        return 0;
    if (s->last_main_s <= 0.0) {
        if (mres >= (size_t)((ms * 85) / 100)) return 1;
        if (mres >= (size_t)((ms * 80) / 100)
            && s->prefix_end > 0
            && he + (size_t)(384ULL << 20) >= s->prefix_end)
            return 1;
        return 0;
    }
    if (s->last_main_s < 2.90
        && mres >= (size_t)((ms * 80) / 100))
        return 1;
    if (s->last_main_s < 2.40
        && mres >= (size_t)((ms * 75) / 100))
        return 1;
    if (s->prefix_end > 0
        && mres >= (size_t)((s->prefix_end * 82) / 100)
        && mres >= (size_t)((ms * 58) / 100)
        && s->last_main_s < 2.95)
        return 1;
    if (s->prefix_end > 0
        && mres >= (size_t)((s->prefix_end * 88) / 100)
        && mres >= (size_t)((ms * 55) / 100)
        && s->last_main_s < 2.70)
        return 1;
    if (s->prefix_end >= (size_t)((ms * 90) / 100)
        && mres >= (size_t)((ms * 78) / 100)
        && he >= (size_t)((s->prefix_end * 85) / 100)
        && s->last_main_s < 3.00)
        return 1;
    if (s->frontier_li >= 0
        && mres >= (size_t)((ms * 80) / 100)
        && he >= (size_t)((ms * 70) / 100)
        && s->last_main_s < 2.95)
        return 1;
    return 0;
}

static inline int stratum_soft_is_full_hot(const StratumSoftIo* s) {
    if (!s || !s->soft_warm) return 0;
    size_t ms = s->mmap_size;
    if (ms == 0) return 0;
    size_t mres = stratum_soft_model_resident_bytes(s);
    if (mres > ms) mres = ms;
    double rf = (double)mres / (double)ms;
    if (rf < 0.80) return 0;
    if (s->last_main_s <= 0.0) return rf >= 0.88;
    if (rf >= 0.92 && s->last_main_s < 2.70) return 1;
    if (rf >= 0.88 && s->last_main_s < 2.45) return 1;
    if (rf >= 0.85 && s->last_main_s < 2.30) return 1;
    if (rf >= 0.82 && s->last_main_s < 2.15) return 1;
    if (rf >= 0.80 && s->last_main_s < 2.05) return 1;
    return 0;
}

static inline int stratum_soft_is_thrash(const StratumSoftIo* s) {
    if (!s || !s->soft_warm) return 0;
    size_t ms = s->mmap_size;
    if (ms == 0) return 0;
    size_t mres = stratum_soft_model_resident_bytes(s);
    if (mres > ms) mres = ms;
    double rf = (double)mres / (double)ms;
    if (rf >= 0.78) return 0;
    if (s->last_main_s > 3.20 && rf < 0.75) return 1;
    if (s->last_main_s > 0.0 && s->last_main_s < 2.55 && rf >= 0.62)
        return 0;
    if (s->last_main_s > 0.0 && s->last_main_s > 2.55 && rf < 0.72)
        return 1;
    if (rf < 0.55) return 1;
    size_t free_true = stratum_soft_free_pages_bytes();
    if (free_true > 0 && free_true < (size_t)(192ULL << 20) && rf < 0.62
        && (s->last_main_s <= 0.0 || s->last_main_s > 2.55))
        return 1;
    return 0;
}

static inline int stratum_soft_io_pressure(const StratumSoftIo* s) {
    if (!s || !s->soft_warm) return 0;
    size_t ms = s->mmap_size;
    size_t mres = stratum_soft_model_resident_bytes(s);
    if (mres > ms && ms > 0) mres = ms;
    double rf = (ms > 0) ? ((double)mres / (double)ms) : 0.0;
    if (rf >= 0.78) return 0;
    if (rf >= 0.72 && (s->last_main_s <= 0.0 || s->last_main_s < 2.70))
        return 0;
    if (s->last_main_s > 0.0 && s->last_main_s > 2.95 && rf < 0.75) return 1;
    if (rf < 0.58) return 1;
    if (s->prefix_end > 0 && s->hot_end + (size_t)(1024ULL << 20) < s->prefix_end
        && rf < 0.68)
        return 1;
    size_t free_true = stratum_soft_free_pages_bytes();
    if (free_true > 0 && free_true < (size_t)(192ULL << 20) && rf < 0.62
        && s->last_main_s > 2.40)
        return 1;
    return 0;
}





static inline void stratum_soft_reaffirm_range(StratumSoftIo* s,
                                              size_t start, size_t end, int force_madv) {
    if (!s || !s->mmap_base || end <= start) return;
    size_t ms = s->mmap_size;
    if (start >= ms) return;
    if (end > ms) end = ms;
    size_t a = stratum_soft_page_align_down(start);
    size_t b = stratum_soft_page_align_up(end);
    if (b > ms) b = ms;
    if (b <= a) return;
    size_t len = b - a;
    if (len < (size_t)(1ULL << 20)) return;
    int thrash = stratum_soft_is_thrash(s);
    int full_hot = stratum_soft_is_full_hot(s);
    if (full_hot) {
#ifdef F_RDADVISE
        if (s->fd >= 0 && force_madv) {
            struct radvisory ra;
            ra.ra_offset = (off_t)a;
            size_t adv = len;
            if (adv > (size_t)(24ULL << 20)) adv = (size_t)(24ULL << 20);
            ra.ra_count = (int)((adv > (size_t)0x7fffffff) ? 0x7fffffff : adv);
            fcntl(s->fd, F_RDADVISE, &ra);
            s->reaffirm_n++;
            s->reaffirm_bytes += (long)adv;
        }
#endif
        return;
    }
    if (!thrash && (force_madv || len <= (size_t)(128ULL << 20)))
        madvise((void*)((char*)s->mmap_base + a), len, MADV_WILLNEED);
#ifdef F_RDADVISE
    if (s->fd >= 0) {
        struct radvisory ra;
        ra.ra_offset = (off_t)a;
        size_t adv = len;
        if (thrash && adv > (size_t)(40ULL << 20)) adv = (size_t)(40ULL << 20);
        ra.ra_count = (int)((adv > (size_t)0x7fffffff) ? 0x7fffffff : adv);
        fcntl(s->fd, F_RDADVISE, &ra);
    }
#endif
    s->reaffirm_n++;
    s->reaffirm_bytes += (long)len;
}

static inline void stratum_soft_reaffirm_holes(StratumSoftIo* s,
                                              size_t start, size_t end, size_t budget) {
    if (!s || !s->mmap_base || end <= start || budget == 0) return;
    size_t ms = s->mmap_size;
    if (start >= ms) return;
    if (end > ms) end = ms;
    size_t pgsz = 16384;
    long ps = 0; size_t pl = sizeof(ps);
    if (sysctlbyname("hw.pagesize", &ps, &pl, NULL, 0) == 0 && ps > 0) pgsz = (size_t)ps;
    size_t chunk = (size_t)(48ULL << 20);
    size_t used = 0;
    for (size_t off = start; off < end && used < budget; ) {
        size_t clen = chunk;
        if (off + clen > end) clen = end - off;
        if (clen < pgsz) break;
        if (!stratum_soft_range_pages_hot(s, off, clen, 5)) {
            stratum_soft_reaffirm_range(s, off, off + clen, 1);
            used += clen;
        }
        off += clen;
    }
}

static inline void stratum_soft_reaffirm_bulk(StratumSoftIo* s, size_t end, size_t cap) {
    if (!s || !s->mmap_base || end == 0 || cap == 0) return;
    size_t ms = s->mmap_size;
    if (end > ms) end = ms;
    if (end > cap) end = cap;
    if (end <= (size_t)(16ULL << 20)) return;
    stratum_soft_reaffirm_range(s, 0, end, 1);
}

static inline void stratum_soft_reaffirm_throttle(StratumSoftIo* s, int li,
                                                 const StratumLayerSpan* ranges,
                                                 int n_layers) {
    if (!s || !s->soft_warm || s->keep_resident) return;
    if (s->prefix_end == 0 || !s->mmap_base || s->mmap_size == 0) return;
    if (stratum_soft_is_full_hot(s)) return;
    if (s->main_scans > 0 && s->last_main_s > 3.45) return;
    if (s->main_scans > 0 && s->last_main_s > 0.0 && s->last_main_s < 2.40
        && li > 0 && (li & 15) != 0) return;
    if (li < 0) return;
    int step = 8;
    if (s->last_main_s > 0.0 && s->last_main_s > 3.10) step = 16;
    if (s->last_main_s > 0.0 && s->last_main_s < 2.85) step = 6;
    if (s->reaffirm_last_li >= 0 && (li - s->reaffirm_last_li) < step) return;
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double t = now.tv_sec + now.tv_nsec / 1e9;
    if (s->reaffirm_last_t > 0.0 && (t - s->reaffirm_last_t) < 0.040) return;
    size_t fr = stratum_soft_free_reclaim_bytes();
    if (fr < (size_t)(288ULL << 20)) return;
    size_t free_true = stratum_soft_free_pages_bytes();
    if (free_true > 0 && free_true < (size_t)(128ULL << 20)) return;
    size_t ms = s->mmap_size;
    size_t prefix = s->prefix_end;
    if (prefix > ms) prefix = ms;
    size_t mres = stratum_soft_model_resident_bytes(s);
    if (mres > ms) mres = ms;
    double res_frac = (ms > 0) ? ((double)mres / (double)ms) : 0.0;
    int deep = (s->last_main_s > 0.0 && s->last_main_s > 3.15);
    int hold = (!deep) && res_frac >= 0.62
            && (s->last_main_s <= 0.0 || s->last_main_s < 3.05)
            && fr >= (size_t)(480ULL << 20);
    if (ranges && n_layers > 0 && li >= 0 && li < n_layers) {
        int li0 = li;
        int li1 = li + (hold ? 3 : 2);
        if (li1 >= n_layers) li1 = n_layers - 1;
        size_t cur0 = ranges[li0].start;
        size_t cur1 = ranges[li1].end;
        if (cur0 > (size_t)(80ULL << 20)) cur0 -= (size_t)(80ULL << 20);
        else cur0 = 0;
        cur1 += (size_t)(40ULL << 20);
        if (cur1 > prefix) cur1 = prefix;
        if (cur1 > cur0 + (size_t)(8ULL << 20)) {
            if (hold) {
                size_t span = cur1 - cur0;
                if (span > (size_t)(128ULL << 20)) span = (size_t)(128ULL << 20);
                stratum_soft_reaffirm_range(s, cur0, cur0 + span, 1);
            } else {
                size_t bud = deep ? (size_t)(48ULL << 20) : (size_t)(96ULL << 20);
                if (fr < (size_t)(480ULL << 20)) bud = (size_t)(32ULL << 20);
                stratum_soft_reaffirm_holes(s, cur0, cur1, bud);
            }
        }
    }
    if (hold && fr >= (size_t)(640ULL << 20)) {
        size_t win = (size_t)(192ULL << 20);
        if (res_frac >= 0.75 && fr >= (size_t)(1024ULL << 20)
            && s->last_main_s > 0.0 && s->last_main_s < 2.90)
            win = (size_t)(320ULL << 20);
        size_t lead = prefix < win ? prefix : win;
        stratum_soft_reaffirm_bulk(s, lead, win);
    } else if (!deep && fr >= (size_t)(896ULL << 20) && res_frac < 0.68) {
        size_t win = (size_t)(128ULL << 20);
        size_t lead = prefix < win ? prefix : win;
        if (lead > (size_t)(16ULL << 20))
            stratum_soft_reaffirm_holes(s, 0, lead, win);
    }
    s->reaffirm_last_li = li;
    s->reaffirm_last_t = t;
}

static inline int stratum_soft_main_is_fast(const StratumSoftIo* s) {
    if (!s) return 0;
    return s->last_main_s > 0.0 && s->last_main_s < 2.55;
}

static inline int stratum_soft_main_is_ok(const StratumSoftIo* s) {
    if (!s) return 0;
    return s->last_main_s > 0.0 && s->last_main_s < 3.00;
}


static inline int stratum_soft_should_extend(const StratumSoftIo* s, float accept_ema) {
    if (!s || s->keep_resident) return 1;
    if (!s->soft_warm) return 1;
    if (accept_ema > 0.0f && accept_ema < 1.85f) return 0;
    if (s->main_scans <= 1) {
        return accept_ema >= 1.85f;
    }
    if (s->last_main_s <= 0.0) return 0;
    size_t ms = s->mmap_size;
    size_t mres = stratum_soft_model_resident_bytes(s);
    double rf = (ms > 0) ? ((double)mres / (double)ms) : 0.0;
    if (s->last_main_s > 3.40) {
        if (accept_ema >= 1.95f) return 1;
        return accept_ema >= 2.20f;
    }
    if (s->last_main_s > 3.05) {
        return accept_ema >= 1.90f;
    }
    if (s->last_main_s > 2.70) {
        return accept_ema >= 1.90f;
    }
    if (s->last_main_s > 2.45 && s->last_main_s <= 2.70 && accept_ema >= 1.90f)
        return 1;
    if (s->last_main_s > 2.45 && rf < 0.40 && accept_ema < 2.10f) return 0;
    if (s->last_main_s <= 2.50 && accept_ema >= 1.90f) return 1;
    if (rf >= 0.50 && accept_ema >= 1.90f) return 1;
    if (rf >= 0.45 && accept_ema >= 2.00f) return 1;
    return accept_ema >= 2.05f;
}

static inline int stratum_soft_extend_max_rounds(const StratumSoftIo* s) {
    if (!s || !s->soft_warm) return 3;
    if (s->last_main_s > 3.20) return 1;
    if (s->last_main_s > 2.55) return 1;
    size_t ms = s->mmap_size;
    size_t mres = stratum_soft_model_resident_bytes(s);
    if (ms > 0 && mres >= (size_t)((ms * 75) / 100) && s->last_main_s < 2.30)
        return 2;
    if (s->last_main_s < 2.20) return 2;
    return 1;
}

static inline void stratum_soft_reaffirm_scan_near(StratumSoftIo* s, size_t scan_off) {
    if (!s || !s->soft_warm || s->keep_resident || !s->mmap_base) return;
    if (stratum_soft_is_full_hot(s)) return;
    if (s->last_main_s > 0.0 && s->last_main_s > 2.85) return;
    if (stratum_soft_io_pressure(s) && s->last_main_s > 2.55) return;
    int healthy = (s->last_main_s > 0.0 && s->last_main_s < 2.20);
    if (!healthy && s->main_scans > 0 && (s->main_scans % 2) == 1) return;
    size_t free_true = stratum_soft_free_pages_bytes();
    if (free_true > 0 && free_true < (size_t)(160ULL << 20)) return;
    size_t fr = stratum_soft_free_reclaim_bytes();
    if (fr < (size_t)(288ULL << 20)) return;
    size_t ms = s->mmap_size;
    size_t pe = s->prefix_end;
    if (pe > ms) pe = ms;
    if (pe < (size_t)(64ULL << 20)) return;
    size_t he = s->hot_end;
    if (he == 0) he = stratum_soft_refresh_hot_end(s);
    if (he + (size_t)(768ULL << 20) >= pe) return;
    size_t mres = stratum_soft_model_resident_bytes(s);
    if (mres > ms) mres = ms;
    double rf = (ms > 0) ? ((double)mres / (double)ms) : 0.0;
    if (rf >= 0.70) return;
    size_t a = 0;
    if (scan_off > (size_t)(48ULL << 20)) a = scan_off - (size_t)(48ULL << 20);
    size_t b = scan_off + (size_t)(96ULL << 20);
    if (b > pe) b = pe;
    if (a > pe) a = pe;
    if (b <= a + (size_t)(16ULL << 20)) return;
    size_t bud = (size_t)(64ULL << 20);
    if (s->last_main_s > 0.0 && s->last_main_s > 2.70) bud = (size_t)(32ULL << 20);
    if (fr < (size_t)(384ULL << 20)) bud = (size_t)(24ULL << 20);
    if (rf < 0.50 && fr >= (size_t)(480ULL << 20) && s->last_main_s <= 2.55)
        bud = (size_t)(96ULL << 20);
    stratum_soft_reaffirm_holes(s, a, b, bud);
}

static inline void stratum_soft_reaffirm_pre_main(StratumSoftIo* s) {
    if (!s || !s->soft_warm || s->keep_resident) return;
    if (s->main_scans != 0 || s->prefix_end == 0 || !s->mmap_base) return;
    if (stratum_soft_is_full_hot(s)) return;
    size_t fr = stratum_soft_free_reclaim_bytes();
    size_t ms = s->mmap_size;
    size_t pe = s->prefix_end;
    if (pe > ms) pe = ms;
    size_t mres = stratum_soft_model_resident_bytes(s);
    double rf = (ms > 0) ? ((double)mres / (double)ms) : 0.0;
    size_t he = stratum_soft_refresh_hot_end(s);
    size_t lead = pe < (size_t)(384ULL << 20) ? pe : (size_t)(384ULL << 20);
    if (rf < 0.55 && fr >= (size_t)(640ULL << 20))
        lead = pe < (size_t)(768ULL << 20) ? pe : (size_t)(768ULL << 20);
    else if (rf >= 0.70)
        lead = pe < (size_t)(256ULL << 20) ? pe : (size_t)(256ULL << 20);
    if (he + (size_t)(256ULL << 20) < pe && fr >= (size_t)(256ULL << 20)) {
        size_t bud = rf < 0.50 ? (size_t)(384ULL << 20) : (size_t)(256ULL << 20);
        size_t span = pe - he;
        if (span > (size_t)(512ULL << 20)) span = (size_t)(512ULL << 20);
        stratum_soft_reaffirm_holes(s, he, he + span, bud);
        s->horizon_heal_n++;
    }
    if (fr >= (size_t)(320ULL << 20) && lead > (size_t)(16ULL << 20)) {
        if (rf >= 0.65 && he + (size_t)(128ULL << 20) >= lead)
            stratum_soft_reaffirm_bulk(s, lead, lead);
        else
            stratum_soft_reaffirm_holes(s, 0, lead,
                rf < 0.45 ? (size_t)(512ULL << 20) : (size_t)(256ULL << 20));
    } else if (fr >= (size_t)(192ULL << 20) && lead > (size_t)(16ULL << 20)) {
        stratum_soft_reaffirm_holes(s, 0, lead, (size_t)(128ULL << 20));
    }
    (void)stratum_soft_stage_horizon(s);
}

static inline void stratum_soft_reaffirm_tree_lead(StratumSoftIo* s) {
    if (!s || !s->soft_warm || s->keep_resident || !s->mmap_base) return;
    if (stratum_soft_is_full_hot(s)) return;
    if (s->last_main_s > 0.0 && s->last_main_s > 3.20) return;
    size_t fr = stratum_soft_free_reclaim_bytes();
    if (fr < (size_t)(256ULL << 20)) return;
    size_t ms = s->mmap_size;
    size_t pe = s->prefix_end;
    if (pe > ms) pe = ms;
    size_t lead = pe < (size_t)(160ULL << 20) ? pe : (size_t)(160ULL << 20);
    if (stratum_soft_main_is_fast(s) && fr >= (size_t)(512ULL << 20))
        lead = pe < (size_t)(256ULL << 20) ? pe : (size_t)(256ULL << 20);
    if (lead > (size_t)(16ULL << 20))
        stratum_soft_reaffirm_bulk(s, lead, lead);
}

static inline void stratum_soft_reaffirm_after_main(StratumSoftIo* s) {
    if (!s || !s->soft_warm || s->keep_resident || !s->mmap_base) return;
    if (stratum_soft_is_full_hot(s)) { s->reaffirm_last_li = -1; return; }
    size_t ms = s->mmap_size;
    size_t mres = stratum_soft_model_resident_bytes(s);
    if (mres > ms) mres = ms;
    double rf = (ms > 0) ? ((double)mres / (double)ms) : 0.0;
    if (stratum_soft_main_is_fast(s) && rf >= 0.65) {
        s->reaffirm_last_li = -1;
        return;
    }
    if (s->last_main_s > 3.10 && rf < 0.55) {
        s->reaffirm_last_li = -1;
        return;
    }
    if (s->prefix_end > 0 && s->last_main_s > 0.0 && s->last_main_s < 3.25) {
        size_t fr = stratum_soft_free_reclaim_bytes();
        size_t pe = s->prefix_end;
        if (pe > ms) pe = ms;
        size_t lead = pe < (size_t)(96ULL << 20) ? pe : (size_t)(96ULL << 20);
        if (s->last_main_s > 2.95) lead = pe < (size_t)(64ULL << 20) ? pe : (size_t)(64ULL << 20);
        if (stratum_soft_io_pressure(s) && s->last_main_s > 2.90) {
            lead = pe < (size_t)(48ULL << 20) ? pe : (size_t)(48ULL << 20);
            if (fr >= (size_t)(320ULL << 20) && lead > (size_t)(16ULL << 20)
                && (s->main_scans % 2) == 0)
                stratum_soft_reaffirm_holes(s, 0, lead, (size_t)(24ULL << 20));
        } else if (fr >= (size_t)(256ULL << 20) && lead > (size_t)(16ULL << 20)) {
            if (rf >= 0.70 && s->last_main_s < 2.70)
                stratum_soft_reaffirm_bulk(s, lead, lead);
            else
                stratum_soft_reaffirm_holes(s, 0, lead,
                    s->last_main_s > 2.95 ? (size_t)(32ULL << 20) : (size_t)(64ULL << 20));
        }
    }
    s->reaffirm_last_li = -1;
}

static inline void stratum_soft_heal_horizon(StratumSoftIo* s) {
    if (!s || !s->soft_warm || s->keep_resident || !s->mmap_base) return;
    if (stratum_soft_is_full_hot(s)) return;
    size_t fr = stratum_soft_free_reclaim_bytes();
    if (fr < (size_t)(192ULL << 20)) return;
    size_t pe = s->prefix_end;
    size_t ms = s->mmap_size;
    if (pe > ms) pe = ms;
    if (pe < (size_t)(64ULL << 20)) return;
    size_t he = stratum_soft_refresh_hot_end(s);
    size_t hz = stratum_soft_stage_horizon(s);
    size_t gap_from = he;
    if (hz > 0 && hz < he) gap_from = hz;
    int main_heavy = (s->last_main_s > 3.05);
    if (main_heavy && s->horizon_heal_n >= 2 && (s->horizon_heal_n % 2) == 1)
        return;
    if (gap_from + (size_t)(128ULL << 20) >= pe) {
        size_t lead = pe < (size_t)(160ULL << 20) ? pe : (size_t)(160ULL << 20);
        if (main_heavy)
            lead = pe < (size_t)(96ULL << 20) ? pe : (size_t)(96ULL << 20);
        if (lead > (size_t)(16ULL << 20)) {
            size_t bud = main_heavy ? (size_t)(64ULL << 20) : (size_t)(96ULL << 20);
            if (fr < (size_t)(320ULL << 20)) bud = (size_t)(48ULL << 20);
            stratum_soft_reaffirm_holes(s, 0, lead, bud);
            s->horizon_heal_n++;
        }
        return;
    }
    size_t span = pe - gap_from;
    size_t win = span;
    if (win > (size_t)(256ULL << 20)) win = (size_t)(256ULL << 20);
    if (main_heavy && win > (size_t)(128ULL << 20))
        win = (size_t)(128ULL << 20);
    size_t bud = (size_t)(96ULL << 20);
    if (main_heavy) bud = (size_t)(64ULL << 20);
    if (s->last_main_s > 0.0 && s->last_main_s < 2.70) bud = (size_t)(80ULL << 20);
    if (fr < (size_t)(320ULL << 20)) bud = (size_t)(48ULL << 20);
    if (win > (size_t)(16ULL << 20)) {
        stratum_soft_reaffirm_holes(s, gap_from, gap_from + win, bud);
        s->horizon_heal_n++;
    }
}

static inline void stratum_soft_reaffirm_inter_step(StratumSoftIo* s) {
    if (!s || !s->soft_warm || s->keep_resident) return;
    if (stratum_soft_is_full_hot(s)) return;
    size_t fr = stratum_soft_free_reclaim_bytes();
    size_t pe = s->prefix_end;
    size_t ms = s->mmap_size;
    if (pe > ms) pe = ms;
    if (s->last_main_s > 2.90) {
        size_t lead = pe < (size_t)(64ULL << 20) ? pe : (size_t)(64ULL << 20);
        if (fr >= (size_t)(224ULL << 20) && lead > (size_t)(16ULL << 20)
            && (s->main_scans <= 1 || (s->horizon_heal_n % 2) == 0))
            stratum_soft_reaffirm_holes(s, 0, lead, (size_t)(32ULL << 20));
        return;
    }
    if (s->last_main_s > 2.70) {
        size_t lead = pe < (size_t)(96ULL << 20) ? pe : (size_t)(96ULL << 20);
        if (fr >= (size_t)(256ULL << 20) && lead > (size_t)(16ULL << 20))
            stratum_soft_reaffirm_holes(s, 0, lead, (size_t)(48ULL << 20));
        return;
    }
    stratum_soft_heal_horizon(s);
    if (s->last_main_s > 0.0 && s->last_main_s < 2.70)
        stratum_soft_reaffirm_tree_lead(s);
}


static inline void stratum_soft_frontier_note(StratumSoftIo* s, int li,
                                             const StratumLayerSpan* ranges,
                                             int n_layers) {
    if (!s || !s->soft_warm || s->keep_resident) return;
    if (!ranges || li < 0 || li >= n_layers) return;
    if (s->main_scans > 0 && s->last_main_s > 3.15) return;
    StratumLayerSpan r = ranges[li];
    if (r.end <= r.start) return;
    if (r.start > s->prefix_end + (size_t)(48ULL << 20)) return;
    if (r.end <= s->prefix_end) return;
    int contig = (r.start <= s->prefix_end + (size_t)(8ULL << 20));
    if (!contig && !stratum_soft_range_pages_hot(s, r.start, r.end - r.start, 6)) return;
    if (!contig && s->main_scans > 0 && s->last_main_s > 3.00) return;
    if (s->main_scans > 0 && s->last_main_s > 3.05) return;
    size_t grow = r.end - s->prefix_end;
    if (grow > (size_t)(384ULL << 20)
        && !stratum_soft_range_pages_hot(s, r.start, r.end - r.start, 6))
        return;
    size_t mres = stratum_soft_model_resident_bytes(s);
    size_t ms = s->mmap_size;
    if (mres > ms) mres = ms;
    size_t cap = mres + (size_t)(640ULL << 20);
    if (s->last_main_s > 3.00) cap = mres + (size_t)(320ULL << 20);
    if (cap < s->prefix_init) cap = s->prefix_init;
    if (cap > ms) cap = ms;
    if (r.end > cap) return;
    s->prefix_end = r.end;
    s->frontier_advances++;
    s->frontier_bytes += (long)grow;
    s->frontier_li = li;
}

static inline void stratum_soft_frontier_bulk_walk(StratumSoftIo* s, int from_li,
                                                  int max_layers,
                                                  const StratumLayerSpan* ranges,
                                                  int n_layers,
                                                  int (*skip_layer)(int li, void* ud),
                                                  void* ud) {
    if (!s || !s->soft_warm || s->keep_resident || !ranges) return;
    if (s->main_scans > 0 && s->last_main_s > 3.35) return;
    if (max_layers < 1) max_layers = 1;
    size_t mres = stratum_soft_model_resident_bytes(s);
    size_t ms = s->mmap_size;
    if (mres > ms) mres = ms;
    size_t cap = mres + (size_t)(640ULL << 20);
    int press = stratum_soft_io_pressure(s);
    if (!press && s->last_main_s > 0.0 && s->last_main_s < 1.70)
        cap = mres + (size_t)(1400ULL << 20);
    else if (!press && s->last_main_s > 0.0 && s->last_main_s < 1.90)
        cap = mres + (size_t)(1280ULL << 20);
    else if (!press && s->last_main_s > 0.0 && s->last_main_s < 2.40)
        cap = mres + (size_t)(960ULL << 20);
    if (s->last_main_s > 3.00 || press) cap = mres + (size_t)(320ULL << 20);
    if (cap < s->prefix_init) cap = s->prefix_init;
    if (cap > ms) cap = ms;
    if (!press && s->last_main_s > 0.0 && s->last_main_s < 1.70 && max_layers < 16)
        max_layers = 16;
    else if (!press && s->last_main_s > 0.0 && s->last_main_s < 1.90 && max_layers < 14)
        max_layers = 14;
    else if (!press && s->last_main_s > 0.0 && s->last_main_s < 2.40 && max_layers < 10)
        max_layers = 10;
    int seen = 0;
    int li0 = from_li >= 0 ? from_li : 0;
    for (int li = li0; li < n_layers && seen < max_layers; li++) {
        if (skip_layer && skip_layer(li, ud)) continue;
        StratumLayerSpan r = ranges[li];
        if (r.end <= r.start) continue;
        if (r.end <= s->prefix_end) { seen++; continue; }
        if (r.start > s->prefix_end + (size_t)(64ULL << 20)) break;
        if (!stratum_soft_range_pages_hot(s, r.start, r.end - r.start, 8)) break;
        if (r.end > cap) break;
        size_t grow = r.end - s->prefix_end;
        s->prefix_end = r.end;
        s->frontier_advances++;
        s->frontier_bytes += (long)grow;
        s->frontier_li = li;
        seen++;
    }
}

static inline void stratum_soft_note_main(StratumSoftIo* s, double main_s) {
    if (!s) return;
    s->last_main_s = main_s;
    s->main_scans++;
    s->hot_end_t = 0.0;
    s->stage_horizon_t = 0.0;
    s->cache_mres_t = 0.0;
}

static inline int stratum_soft_multiseq_pick_b(const StratumSoftIo* s,
                                              size_t free_reclaim,
                                              size_t anon_per_stream,
                                              int b_max) {
    size_t msz = s && s->mmap_size ? s->mmap_size : 0;
    size_t mres = s ? stratum_soft_model_resident_bytes(s) : 0;
    if (mres > msz && msz > 0) mres = msz;
    double res_frac = (msz > 0) ? ((double)mres / (double)msz) : 0.0;
    int thrashy = (s && s->soft_warm) || res_frac < 0.58
               || (s && s->last_main_s > 0.0 && s->last_main_s > 3.10);
    int hot = (!thrashy) && res_frac >= 0.80
           && (!s || s->last_main_s <= 0.0 || s->last_main_s < 2.95);
    int ms_B = 4;
    if (thrashy) {
        if (free_reclaim < (size_t)(256ULL << 20)) ms_B = 32;
        else if (free_reclaim < (size_t)(768ULL << 20)) ms_B = 80;
        else if (free_reclaim < (size_t)(1536ULL << 20)) ms_B = 160;
        else ms_B = 256;
    } else if (hot) {
        if (free_reclaim >= (size_t)(3072ULL << 20)) ms_B = 512;
        else if (free_reclaim >= (size_t)(2048ULL << 20)) ms_B = 384;
        else ms_B = 192;
    } else {
        if (free_reclaim >= (size_t)(2048ULL << 20)) ms_B = 256;
        else if (msz > 0 && free_reclaim < (size_t)((msz * 40) / 100)) ms_B = 96;
        else if (msz > 0 && free_reclaim < (size_t)((msz * 70) / 100)) ms_B = 160;
        else ms_B = 160;
    }
    if (anon_per_stream > 0) {
        size_t anon_budget = free_reclaim > (size_t)(768ULL << 20)
                           ? free_reclaim - (size_t)(512ULL << 20)
                           : (size_t)(256ULL << 20);
        int bcap = (int)(anon_budget / anon_per_stream);
        if (bcap < 2) bcap = 2;
        if (ms_B > bcap) ms_B = bcap;
    }
    if (ms_B < 2) ms_B = 2;
    if (b_max > 0 && ms_B > b_max) ms_B = b_max;
    return ms_B;
}

static inline int stratum_soft_slot_pressure(const StratumSoftIo* s,
                                            long fail_slot, long start_try) {
    if (!s || !s->soft_warm) return 0;
    if (fail_slot <= 0 || start_try <= 80) return 0;
    return ((fail_slot * 100) / start_try) >= 14;
}




static inline void stratum_soft_report(const StratumSoftIo* s, FILE* out) {
    if (!s || !out) return;
    fprintf(out,
            "  V136 soft_io: warm=%d prefix=%.2fGB hot=%.2fGB hz=%.2fGB init=%.2fGB main=%.2fs scans=%d\n"
            "  V136 reaffirm: n=%ld bytes=%.1fMB heal=%ld hz_stage=%ld\n"
            "  V136 frontier: advances=%ld +%.2fGB end=%.2fGB li=%d\n",
            s->soft_warm,
            s->prefix_end / (1024.0 * 1024.0 * 1024.0),
            s->hot_end / (1024.0 * 1024.0 * 1024.0),
            s->stage_horizon / (1024.0 * 1024.0 * 1024.0),
            s->prefix_init / (1024.0 * 1024.0 * 1024.0),
            s->last_main_s, s->main_scans,
            s->reaffirm_n, s->reaffirm_bytes / (1024.0 * 1024.0),
            s->horizon_heal_n, s->horizon_stage_n,
            s->frontier_advances,
            s->frontier_bytes / (1024.0 * 1024.0 * 1024.0),
            s->prefix_end / (1024.0 * 1024.0 * 1024.0),
            s->frontier_li);
}

#endif
