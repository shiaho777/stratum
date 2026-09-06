/*
 * h3_sample.c — H3 pipeline M4: flow-matching Euler sampling loop over
 * the packed denoiser forward (t2va spike, fl2va denoiser).
 *
 * Sampler contract (pinned from comfy/ldm/minimax + the ModelSamplingAV
 * wrapper):
 *   - timestep arrives as sigma*1000 (0..1000); t_v = 1 - sigma_v;
 *   - the audio stream runs on its own shifted schedule:
 *       sigma_a = shift_map(sigma_v, 12.0 -> 3.0),
 *       shift_map(s, f, g) = g*s'/(1+(g-1)s') where
 *       s' = s/(f + s*(1-f))   (invert base grid, re-apply)
 *   - with default audio_scale=1.0 the carried-audio transform is the
 *     identity, so the network sees raw stream latents (the general
 *     scale!=1 branch is documented but not exercised here);
 *   - velocity outputs are NEGATED by the model (-video_out) — the
 *     Euler update against the standard flow ODE x1 = x0 + dt*v must
 *     therefore ADD the network output (model returns -v in x0 space);
 *   - per-step timestep embedding: adaln_t_table lerp at
 *     pos = (1-sigma_v) * (rows-1)? NO — the table is indexed by the
 *     t the network sees per modality: t_emb rows are looked up ONCE
 *     per forward with t = t_v for video/text rows. In the curve-form
 *     checkpoint the network interpolates the table at the fractional
 *     grid index of t. rows_to_mod_index collapses per-stream single
 *     t values, so M = #distinct t values (2 for t2va: t_v and t_a).
 *
 * This spike therefore runs the packed forward at arbitrary sigma with
 * TWO table rows (video/text row and audio row), Euler-integrates both
 * target streams from pure noise, and reports trajectory norms.
 *
 * Reuses h3_forward.c's validated building blocks via #include of the
 * shared C file compiled once (single-TU style of the engine).
 */
