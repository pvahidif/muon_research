# exp003_curvature

Profiles the curvature (SVD decomposition of the update signal, plus
gamma/phi statistics) around points on Muon's own training trajectory, to
look at how the geometry of individual singular-value components relates to
training dynamics. Analyzed in `articles/01_the_geometry/article.ipynb`,
section 4, "geometry of singular value components." Finding: away from the
very start of training, the profiled directions are approximately
Hessian-orthogonal within the profiled subspace, but not Hessian eigenvectors
because substantial curvature leaks outside that subspace —
`|gamma_ii|` (mode curvature) fits a clean `sigma_i^beta` power law
(`attn.v.weight`: beta 0.72–1.18, R²=0.60–0.92;
`mlp.fc.weight`: beta 1.41–1.77, R²=0.51 at step 0 and 0.94–0.99 from step 10
on), gamma's off-diagonal terms shrink by roughly an order of magnitude from
the earliest fork to the latest, and the real gradient's own projection onto
each mode is noisy but scales with sigma_i too — the biggest, most-curved
modes carry the most real gradient signal.

## Subdirectories

- **`muon_continue/`** — forks
  `exp000_muon_and_adamw/muon/seed_1337/rules.two_d.lr_0.02` at steps
  `{0, 1, 10, 100, 500, 1000, 2000, 4000, 6000}`, each into a single
  `continue` branch (plain Muon, i.e. no actual override — same rule the
  trunk was already using), checkpointing every step for 10 steps after
  each fork. This produces the dense, per-step checkpoints the profiling
  runs below read from.
- **`muon_curvature/`** — `run_curv_profile.py` profiles
  `blocks.*.attn.v.weight` and `blocks.*.mlp.fc.weight` (SVD decomposition,
  `compute_gamma`/`compute_phi`) at hand-picked steps from each
  `muon_continue` fork's `continue` branch: just the fork points
  themselves early on (`fork_000000`: step 0, `fork_000001`: 1-3, `fork_000010`: 10–14,
  where the Hessian moves fast but sample noise is low), and a full window
  of consecutive post-fork steps later on (`fork_000100` through
  `fork_006000`: ~7–11 steps each, where the Hessian moves slowly but
  per-checkpoint noise is relatively higher, so later analysis can average
  across the window). Gamma/phi are estimated against a held-out
  validation batch (`profile_batch_size=32768`) that's freshly resampled
  at every profiling event (`profile_batch_resample: true`), kept
  independent of the real (training-data) batch used for the SVD itself.
- **`muon_curvature_fixed_pool_b/`** — identical run to `muon_curvature/`,
  except `profile_batch_resample: false` — gamma/phi are estimated from a
  single held-out batch reused across every profiling event instead of a
  fresh one each time, to check the results in `muon_curvature/` aren't an
  artifact of that resampling.
- **`muon_curvature_same_pools/`** — identical run again, but with
  `profile_batch_size` left unset entirely, so gamma/phi fall back to the
  same batch used for the real gradient/SVD rather than an independent
  held-out one — checks that using a genuinely separate pool at all (as
  the two runs above do) isn't itself what's driving the results.

## Running

Requires `exp000_muon_and_adamw/muon` to already be run. `muon_continue/`
must run before any of the curvature-profiling jobs:

```bash
experiments/exp003_curvature/muon_continue/run.sh
experiments/exp003_curvature/muon_curvature/run.sh
experiments/exp003_curvature/muon_curvature_fixed_pool_b/run.sh
experiments/exp003_curvature/muon_curvature_same_pools/run.sh
```

`muon_continue/run.sh` invokes
`src/muon_research/scripts/run_branch_compare.py`; the three curvature jobs
invoke `run_curv_profile.py`. All `torchrun` against the directory's own
`config.yaml`.
