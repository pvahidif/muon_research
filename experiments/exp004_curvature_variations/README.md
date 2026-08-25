# exp004_curvature_variations

Tests whether the gamma-vs-sigma shape difference exp003 found between
`attn.v.weight` and `mlp.fc.weight` -- `|gamma_ii| ~ sigma_i^beta` with a
noticeably different `beta` per matrix (analyzed in
`articles/01_the_geometry/article.ipynb`, section 4, "geometry of singular
value components") -- comes from those two matrices' own **aspect ratio**
rather than from being an attention vs. MLP matrix per se. `attn.v.weight`
has shape `(num_heads * head_dim, model_dim)`; `mlp.fc.weight` has shape
`(expansion_ratio * model_dim, model_dim)`. With the base config's
`model_dim=192`, `head_dim=64`, `num_heads=3`, `expansion_ratio=4.0`,
`attn.v.weight` is square (192x192, aspect ratio 1) while `mlp.fc.weight` is
768x192 (aspect ratio 4) -- exactly the two ratios whose `beta` differed in
exp003. Three variations on that base config each push one matrix's aspect
ratio to match the other's, via a different architectural knob each time:

- `head_dim_256` (`head_dim=256`, `num_heads=3` unchanged): widens
  `attn.v.weight` to 768x192 (ratio 4) by making each head wider.
- `num_heads_12` (`num_heads=12`, `head_dim=64` unchanged): widens
  `attn.v.weight` to the same 768x192 (ratio 4) by adding more heads
  instead -- a different route to the same shape, to check it's the shape
  and not the head count/structure that matters.
- `expansion_ratio_1` (`expansion_ratio=1.0`): narrows `mlp.fc.weight` to
  192x192 (ratio 1) instead, matching it down to `attn.v.weight`'s own
  ratio rather than stretching `attn.v.weight` up to `mlp.fc.weight`'s.

`head_dim_64` is the unmodified base config, included as the fourth arm so
all four (two ratio-1 matrices, two ratio-4 matrices, split across both
matrix types) are profiled the same way side by side. If `beta` converges
whenever the two matrices share an aspect ratio -- regardless of which one
moved or how -- that supports aspect ratio as the driver; if `attn.v.weight`
and `mlp.fc.weight` keep their exp003 `beta`s even after being reshaped to
match, the difference isn't really about aspect ratio.

(A separate, much larger sweep over aspect ratio and model width, run
outside this repo, is analyzed in `articles/01_the_geometry/article.ipynb`,
section 5, "different aspect ratios" -- this experiment is the earlier,
narrower check on the same question, using only the two matrices/config
knobs exp003 already profiled.)

## Subdirectories

- **`muon_create/`** -- trains four fresh 1-layer models from scratch (not
  forked from any other experiment), one per aspect-ratio variation above,
  seed `1337`, `train_steps=1001`, checkpointing once at `step=1000`. Same
  `two_d`-on-Muon rule setup as `exp000_muon_and_adamw/muon`.
- **`muon_curvature/`** -- `run_curv_profile.py` profiles `blocks.*.attn.v.weight`
  and `blocks.*.mlp.fc.weight` (SVD decomposition, `compute_gamma`,
  `compute_phi`) at each arm's `step_1000` checkpoint, against a held-out
  validation batch resampled fresh each time (`profile_batch_size=32768`,
  `profile_batch_resample: true`) -- same profiling setup exp003's
  `muon_curvature/` uses.

## Running

```bash
experiments/exp004_curvature_variations/muon_create/run.sh
experiments/exp004_curvature_variations/muon_curvature/run.sh
```

Requires `data/fineweb10B_v500` (see the repo-root README's Data section).
`muon_create/run.sh` invokes `run_optim_rules.py`, sweeping `override_args`
into one run per arm under `seed_1337/<override>/`; `muon_curvature/run.sh`
invokes `run_curv_profile.py` and must run after `muon_create/`. Both
`torchrun` against the directory's own `config.yaml`.
