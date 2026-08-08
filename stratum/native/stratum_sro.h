#ifndef STRATUM_SRO_H
#define STRATUM_SRO_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t model_sz;
    size_t model_res;
    size_t free_reclaim;
    int thrash;
    int stage_sticky;
    double last_main_s;
    double last_io_stall_s;
} stratum_sro_sample_t;

typedef struct {
    int mild;
    int res_hot;
    int res_ultra;
    int hot_io;
    int hfw;
    int shs;
    int pressure;
    size_t hot_budget;
    size_t pf_budget;
    int pf_layers;
    int wave_multi;
    int retire_ok;
    int pure_compute;
} stratum_sro_policy_t;

static inline void stratum_sro_decide(const stratum_sro_sample_t* s,
                                     stratum_sro_policy_t* o)
{
    size_t msz = s->model_sz;
    size_t fr = s->free_reclaim;
    size_t mres = s->model_res;
    if (mres > msz && msz) mres = msz;

    o->mild = (msz > 0 && fr >= (size_t)((msz * 55) / 100)) ? 1 : 0;
    o->res_hot = (msz > 0 && mres >= (size_t)((msz * 92) / 100)) ? 1 : 0;
    o->res_ultra = (msz > 0 && mres >= (size_t)((msz * 95) / 100)) ? 1 : 0;

    o->hot_io = (o->mild
        || o->res_hot
        || s->last_main_s <= 1e-6
        || (s->last_main_s > 0.0 && s->last_main_s < 2.20
            && s->last_io_stall_s < 0.05)) ? 1 : 0;
    if ((s->thrash || s->stage_sticky) && !o->res_hot && !o->mild)
        o->hot_io = 0;
    if (!o->hot_io) o->shs = 0;
    else o->shs = o->res_ultra ? 1 : 0;
    o->hfw = (o->hot_io && (o->mild || o->res_hot)) ? 1 : 0;

    {
        double free_r = msz ? (double)fr / (double)msz : 0.0;
        double res_r = msz ? (double)mres / (double)msz : 0.0;
        double eff = free_r + res_r * 0.65;
        int p;
        if (eff >= 1.20) p = 8;
        else if (eff >= 0.95) p = 18;
        else if (eff >= 0.70) p = 32;
        else if (eff >= 0.55) p = 45;
        else if (eff >= 0.35) p = 62;
        else if (eff >= 0.20) p = 78;
        else p = 92;
        if ((s->thrash || s->stage_sticky) && res_r < 0.92) {
            if (p < 75) p = 75;
        }
        if (free_r < 0.12 && res_r < 0.85) p = 95;
        if (s->last_main_s > 3.50) {
            if (p < 82) p = 82;
        }
        if (s->last_main_s > 0.0 && s->last_main_s < 1.80
            && s->last_io_stall_s < 0.05) {
            if (p > 30) p = 30;
        }
        if (o->hot_io && o->hfw && p > 40) p = 40;
        o->pressure = p;
    }

    if (o->mild) {
        o->hot_budget = msz;
        o->pf_budget = (fr > msz) ? (fr - (msz / 2)) : (fr / 6);
        o->pf_layers = 4;
        o->wave_multi = 3;
    } else if (o->res_hot && o->hot_io) {
        o->hot_budget = mres;
        o->pf_budget = fr / 5;
        o->pf_layers = 2;
        o->wave_multi = 2;
    } else if (o->hot_io) {
        o->hot_budget = mres + (fr / 3);
        o->pf_budget = fr / 6;
        o->pf_layers = 2;
        o->wave_multi = 2;
    } else {
        o->hot_budget = mres;
        o->pf_budget = (fr > (size_t)(384ULL << 20)) ? (fr / 10) : 0;
        o->pf_layers = o->pf_budget ? 1 : 0;
        o->wave_multi = 1;
    }

    o->retire_ok = (!o->hot_io && (s->thrash || s->stage_sticky)
        && fr < (size_t)(512ULL << 20)) ? 1 : 0;
    o->pure_compute = (o->hfw && o->pressure <= 40) ? 1 : 0;
}

#endif
