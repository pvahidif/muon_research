# exp001_adamw_then_muon

Tests whether Muon's advantage over AdamW is a per-step property (every Muon
step is at least as good as the AdamW step it replaces) or a state-dependent
one (Muon needs to reach some region of parameter space AdamW's steps don't).
Forks `exp000_muon_and_adamw/adamw`'s best run (`lr=0.003`) at several points
and, at each fork, runs two branches from the identical checkpoint reading
the identical data: one continuing with AdamW, one switching to Muon (its
own tuned `lr=0.02`). Any difference is purely the update rule, not the data.

Analyzed in `articles/01_the_geometry/article.ipynb`, section 2. Finding:
the AdamW-vs-Muon val-loss diff always starts near zero, rises as Muon's
update destabilizes the loss relative to continuing AdamW, peaks, then falls
back through zero as Muon pulls ahead — and how long that catch-up takes
(`iter_zero`) grows in absolute terms with the fork step, staying in the
27–38% range (of the fork step) past ~500 steps. Muon isn't winning
step-for-step; it's winning by reaching a
state that makes subsequent training more effective.

## Subdirectories

- **`(adamw_then_muon config.yaml/run.sh)`** — main comparison: forks
  `exp000_muon_and_adamw/adamw/seed_1337/rules.two_d.lr_0.003` at steps
  `{100, 200, 500, 1000, 2000, 4000, 6000, 8000}` into `adamw`/`muon`
  branches, run via `run_branch_compare.py`.
- **`muon_checkpoints/`** — same branch-compare mechanism, but at fork
  points `{10, 100, 200, 500, 1000}`, with the `muon` branch KL-matched to
  `adamw`'s own update size at every step (`kl_matched: true`), and with
  fine-grained `checkpoint_after_steps` (`{1, 4, 16, 64}`) saved right after
  each fork, feeding `muon_continue/` below.
- **`muon_continue/`** — `run_branch_continue.py` re-forks
  (`adamw_no_reset`) `muon_checkpoints/adamw`'s branches at checkpoint
  offsets off each of its 5 fork points (e.g. steps `{11, 14, 26}` off the
  step-10 fork), continuing 256 more steps under the same AdamW-vs-Muon
  override without resetting optimizer state or the data cursor.

## Running

Requires `exp000_muon_and_adamw/adamw` to already be run (these fork its
checkpoints). Then, in order:

```bash
experiments/exp001_adamw_then_muon/adamw_then_muon/run.sh
experiments/exp001_adamw_then_muon/muon_checkpoints/run.sh   # before muon_continue
experiments/exp001_adamw_then_muon/muon_continue/run.sh
```

`adamw_then_muon/run.sh` and `muon_checkpoints/run.sh` invoke
`src/muon_research/scripts/run_branch_compare.py`; `muon_continue/run.sh`
invokes `src/muon_research/scripts/run_branch_continue.py`. All `torchrun`
against the directory's own `config.yaml`.
