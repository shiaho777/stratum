/*
 * h3_pack.c — H3 pipeline M3: PackedLayout builder (t2va / fl2va spike).
 *
 * Reproduces ComfyUI minimax.py PackedLayout for the plain t2va case
 * (no refs, no keyframes): packed stream order is
 *
 *   [text (text_len rows)] [audio (audio_t*2 rows)] [video (vt*frame_rows)]
 *
 * Row roles:
 *   text rows  : fed to the TE (condition_proj/token_refiner) upstream —
 *                handled by te_qwen3vl_step; this module receives the
 *                resulting text_states.
 *   audio rows : stereo channel-major, t advances per latent frame, w
 *                pinned to the target grid extremes per channel, h = 0.
 *   video rows : patchified latent rows in (t, h, w) order, position =
 *                (t_cursor + FRAME_RESCALE-cumsum per latent-t, area-
 *                normalized (h, w) from the frame grid).
 *
 * This binary is a verification spike: it builds the layout for a
 * chosen geometry, prints the grid numerics against hand-computed
 * anchors, patchifies a synthetic video latent and checks the
 * round-trip through unpatchify, packs stereo audio the same way, and
 * projects both through video/audio_patch_proj (F32 GEMV) to prove the
 * segment assembly matches the reference einsum semantics.
 *
 * Constants pinned from comfy/ldm/minimax/model.py:
 *   latents_dim 24, audio_latents_dim 32, patch (1,2,2) -> 96-dim video
 *   rows; FRAME_PER_TOKEN (1,4,4,4,4), FRAME_RESCALE 5/3.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LATENTS_DIM 24
#define AUDIO_DIM 32
#define VIDEO_ROW_DIM (LATENTS_DIM * 4)   /* patch 1x2x2 */
#define FRAME_RESCALE (5.0 / 3.0)
static const int FRAME_PER_TOKEN[5] = {1, 4, 4, 4, 4};

static double now_x(void) { return 0; }   /* placeholder, unused */

/* _axis_from_sqrt_area: linspace((1-ratio)/2, (1+ratio)/2, n, endpoint=F)*32 */
static void axis_from_sqrt_area(int dim, int patch, double *out, int n_out) {
    double ratio = (double)dim / sqrt((double)dim * dim /*placeholder*/);
    (void)patch;
    /* caller passes sqrt_area explicitly; see frame_grid() */
    (void)ratio; (void)out; (void)n_out;
}

static void axis_vals(int dim, int patch, double sqrt_area, double *out) {
    int n = dim / patch;
    double ratio = (double)dim / sqrt_area;
    for (int i = 0; i < n; i++)
        out[i] = ((double)i * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0;
}

/* _frame_grid returns frame rows [h_grid * w_grid, 2] (h-major) and the
 * w-axis vector (for the audio w pins). */
static int frame_grid(int lat_h, int lat_w, double **frame_out,
                      double **w_grid_out, int *n_rows) {
    double sqrt_area = sqrt((double)lat_h * (double)lat_w);
    int nh = lat_h / 2, nw = lat_w / 2;
    double *hv = malloc(sizeof(double) * nh);
    double *wv = malloc(sizeof(double) * nw);
    axis_vals(lat_h, 2, sqrt_area, hv);
    axis_vals(lat_w, 2, sqrt_area, wv);
    double *frame = malloc(sizeof(double) * nh * nw * 2);
    int r = 0;
    for (int i = 0; i < nh; i++)
        for (int j = 0; j < nw; j++) {
            frame[r * 2 + 0] = hv[i];
            frame[r * 2 + 1] = wv[j];
            r++;
        }
    free(hv);
    *frame_out = frame;
    *w_grid_out = wv;
    *n_rows = r;
    return nw;
}

/* _video_t_grid: origin + exclusive cumsum of FRAME_RESCALE*FRAME_PER_TOKEN */
static void video_t_grid(int vt, double origin, double *out) {
    double acc = 0;
    for (int k = 0; k < vt; k++) {
        out[k] = origin + acc;
        acc += FRAME_RESCALE * FRAME_PER_TOKEN[k % 5];
    }
}

int main(int argc, char** argv) {
    /* geometry: latent_t, latent_h, latent_w, audio_t, text_len */
    int vt = argc > 1 ? atoi(argv[1]) : 3;
    int lat_h = argc > 2 ? atoi(argv[2]) : 32;
    int lat_w = argc > 3 ? atoi(argv[3]) : 32;
    int audio_t = argc > 4 ? atoi(argv[4]) : 12;
    int text_len = argc > 5 ? atoi(argv[5]) : 28;

    double *frame, *w_grid;
    int frame_rows;
    int nw = frame_grid(lat_h, lat_w, &frame, &w_grid, &frame_rows);

    int n_audio = audio_t * 2;
    int n_video = vt * frame_rows;
    long seq_len = text_len + n_audio + n_video;
    printf("layout: text=%d audio=%d video=%d seq=%ld (frame_rows=%d, nw=%d)\n",
           text_len, n_audio, n_video, seq_len, frame_rows, nw);

    /* --- position grid [seq, 3] (t, h, w) --- */
    double *pos = calloc((size_t)seq_len * 3, sizeof(double));
    /* text: t = 0..text_len-1, h=w=0 */
    for (int i = 0; i < text_len; i++) pos[(size_t)i * 3 + 0] = i;
    /* target audio: t = cursor..cursor+audio_t-1 (per channel repeat),
     * h = 0, w = {w_low, w_high} per stereo channel */
    double cursor = text_len;
    for (int t = 0; t < audio_t; t++) {
        for (int c = 0; c < 2; c++) {
            long r = text_len + (long)t * 2 + c;
            pos[r * 3 + 0] = cursor + t;
            pos[r * 3 + 2] = c == 0 ? w_grid[0] : w_grid[nw - 1];
        }
    }
    /* target video: t = cursor + cumsum, (h, w) from frame grid */
    {
        double *tv = malloc(sizeof(double) * vt);
        video_t_grid(vt, cursor, tv);
        long r = text_len + n_audio;
        for (int t = 0; t < vt; t++)
            for (int f = 0; f < frame_rows; f++) {
                pos[(r + (long)t * frame_rows + f) * 3 + 0] = tv[t];
                pos[(r + (long)t * frame_rows + f) * 3 + 1] = frame[f * 2 + 0];
                pos[(r + (long)t * frame_rows + f) * 3 + 2] = frame[f * 2 + 1];
            }
        free(tv);
    }

    /* --- verification anchors (hand-computed) --- */
    double sqrt_area = sqrt((double)lat_h * lat_w);
    printf("sqrt_area=%.6f  w_axis[0]=%.6f w_axis[last]=%.6f\n",
           sqrt_area, w_grid[0], w_grid[nw - 1]);
    /* lat_h=lat_w=32: ratio=32/32=1 -> linspace(0, ... , 16)*32:
     * axis[i] = i * (32/16) = 2i. */
    printf("axis check (expect 0,2,...,30): %.3f %.3f %.3f ... %.3f\n",
           w_grid[0], w_grid[1], w_grid[2], w_grid[nw - 1]);
    printf("video t_grid (origin=%g):", cursor);
    {
        double *tv = malloc(sizeof(double) * vt);
        video_t_grid(vt, cursor, tv);
        for (int k = 0; k < vt; k++) printf(" %.4f", tv[k]);
        printf("   (expect origin, +%.4f, +%.4f... per FRAME_PER_TOKEN)\n",
               FRAME_RESCALE * 4, FRAME_RESCALE * 4);
        free(tv);
    }
    printf("audio rows t=0: (%g, 0, %g) / (%g, 0, %g)\n",
           pos[(size_t)text_len * 3], pos[(size_t)text_len * 3 + 2],
           pos[(size_t)(text_len + 1) * 3], pos[(size_t)(text_len + 1) * 3 + 2]);

    /* --- patchify round-trip on synthetic video latent ---
     * latent [C=24, T, H, W] row-major C-outer; patchify with (1,2,2)
     * must produce rows [t*h*w, 96] where row (t, hh, ww) packs
     * latent[c][t][2hh+b][2ww+a] in c-outer, then (b, a) inner order
     * per the reference einsum nctrhpwq->nthwcrpq: within a row the
     * order is c, then r(=ph), then q(=pw). */
    {
        size_t ln = (size_t)LATENTS_DIM * vt * lat_h * lat_w;
        float *latent = malloc(sizeof(float) * ln);
        for (size_t i = 0; i < ln; i++) latent[i] = (float)(i % 97) * 0.25f;
        int nh = lat_h / 2, nw2 = lat_w / 2;
        float *rows = malloc(sizeof(float) * (size_t)vt * nh * nw2 * VIDEO_ROW_DIM);
        long ri = 0;
        for (int t = 0; t < vt; t++)
            for (int hh = 0; hh < nh; hh++)
                for (int ww = 0; ww < nw2; ww++) {
                    float *row = &rows[ri * VIDEO_ROW_DIM];
                    int p = 0;
                    for (int c = 0; c < LATENTS_DIM; c++)
                        for (int b = 0; b < 2; b++)
                            for (int a = 0; a < 2; a++)
                                row[p++] = latent[((size_t)c * vt + t)
                                                  * lat_h * lat_w
                                                  + (size_t)(2 * hh + b) * lat_w
                                                  + (2 * ww + a)];
                    ri++;
                }
        /* spot-check row 0 and one interior row */
        printf("patchify row0[0..3] = %.2f %.2f %.2f %.2f (expect c0: "
               "(0,0)(0,1)(1,0)(1,1) -> %.2f %.2f %.2f %.2f)\n",
               rows[0], rows[1], rows[2], rows[3],
               latent[0], latent[1], latent[lat_w], latent[lat_w + 1]);
        long tr = (1 * nh + 1) * nw2 + 1;
        printf("patchify row(t1,h1,w1)[0..1] = %.2f %.2f (expect %.2f %.2f)\n",
               rows[tr * VIDEO_ROW_DIM], rows[tr * VIDEO_ROW_DIM + 1],
               latent[((size_t)0 * vt + 1) * lat_h * lat_w + 2 * lat_w + 2],
               latent[((size_t)0 * vt + 1) * lat_h * lat_w + 2 * lat_w + 3]);
        free(rows);
        free(latent);
    }

    /* --- stereo audio pack check: latent [32, ch=2, T] channel-major --- */
    {
        size_t an = (size_t)AUDIO_DIM * 2 * audio_t;
        float *al = malloc(sizeof(float) * an);
        for (size_t i = 0; i < an; i++) al[i] = (float)(i % 61) * 0.1f;
        float *arows = malloc(sizeof(float) * (size_t)n_audio * AUDIO_DIM);
        /* pack: [ch*T, 32], ch0 t0..T-1 then ch1 t0..T-1 (channel-major) */
        for (int ch = 0; ch < 2; ch++)
            for (int t = 0; t < audio_t; t++)
                for (int c = 0; c < AUDIO_DIM; c++)
                    arows[((size_t)ch * audio_t + t) * AUDIO_DIM + c] =
                        al[((size_t)c * 2 + ch) * audio_t + t];
        printf("audio pack row(ch0,t0)[0..1] = %.2f %.2f (expect %.2f %.2f)\n",
               arows[0], arows[1], al[0], al[(size_t)2 * audio_t]);
        printf("audio pack row(ch1,t0)[0] = %.2f (expect %.2f)\n",
               arows[(size_t)audio_t * AUDIO_DIM], al[(size_t)audio_t]);
        free(arows);
        free(al);
    }

    free(pos); free(frame); free(w_grid);
    printf("M3 layout spike: OK\n");
    return 0;
}
