# exp002_compare_muon_pow

Muon's update is the polar factor of the Adam-momentum signal — equivalent
to raising every singular value of that signal to power 0 (pure
orthogonalization). This experiment checks whether that power is special,
by forking Muon's own best run and comparing power 0 (regular Muon, via
Newton-Schulz, and an exact-SVD reference `svdp_z000`) against `-0.25`
(over-weights smaller singular directions) and `+0.25` (over-weights
dominant ones). Because changing the update's shape also changes its size,
every branch is per-param KL-matched at every step (binary search rescales
each variant's own direction to reproduce the exact same token-mean KL
divergence Muon's own real update produced) — isolating *direction* as the
only thing differing between branches.

Analyzed in `articles/01_the_geometry/article.ipynb`, section 3. Finding:
the power matters, and *which* direction helps flips with training
progress — early on (`divergence_step` ≤ 100), `-0.25` opens a fast
advantage that fades on reunification; from step 500 on, the roles swap and
`+0.25` wins instead. The advantage carries in the parameters, not leftover
optimizer state — resetting momentum at reunification doesn't change the
qualitative picture.

## Subdirectories

- **`01_pow_checkpoint/`** — main sweep: forks
  `exp000_muon_and_adamw/muon/seed_1337/rules.two_d.lr_0.02` at steps
  `{10, 100, 500, 2000, 4000}` into 4 branches (`muon` reference,
  `svdp_m025`, `svdp_z000`, `svdp_p025`), KL-matched, with
  `checkpoint_after_steps={1, 4, 16, 64}`.
- **`01_pow_continue/`** — re-forks `01_pow_checkpoint`'s branches at those
  checkpoint offsets (e.g. `{11, 14, 26}` off the step-10 fork) and
  reunites all four under plain Muon (no more KL-matching) for 256–512
  more steps, to watch what happens to the gap that opened during
  divergence. Each fork point is run twice — once resetting optimizer
  state on reunification (`muon_with_reset`) and once carrying it over
  (`muon_no_reset`) — to check whether the advantage lives in the
  parameters or in leftover optimizer state.
- **`02_p025_run/`** — forks the step-2000 Muon run for a longer,
  per param kl-matched 129-step window running only `svdp_p025`
  (power +0.25, not KL-matched to Muon globally, just per-param) —
  builds a training path that isn't pure Muon, to feed into the next fork.
- **`02_pow_from_p025/`** — second fork in the "power tree": picks
  `02_p025_run`'s endpoint back up at steps `{2064, 2128}` and forks again
  into all three KL-matched powers, checkpointing at offsets `{16, 64}`
  steps later.
- **`02_pow_continue/`** — reunites `02_pow_from_p025`'s three power
  branches, at both its fork points, under plain Muon for 512 more steps —
  same reunification test as `01_pow_continue/`, but starting from a path
  that already ran under `p=+0.25` rather than pure Muon.

## Running

Requires `exp000_muon_and_adamw/muon` to already be run. Order matters —
each stage forks the previous one's output:

```bash
experiments/exp002_compare_muon_pow/01_pow_checkpoint/run.sh
experiments/exp002_compare_muon_pow/01_pow_continue/run.sh        # forks 01_pow_checkpoint

experiments/exp002_compare_muon_pow/02_p025_run/run.sh
experiments/exp002_compare_muon_pow/02_pow_from_p025/run.sh    # forks 02_p025_run
experiments/exp002_compare_muon_pow/02_pow_continue/run.sh     # forks 02_pow_from_p025
```

`01_pow_checkpoint/`, `02_p025_run/`, and `02_pow_from_p025/` invoke
`src/muon_research/scripts/run_branch_compare.py`; `01_pow_continue/` and
`02_pow_continue/` invoke `run_branch_continue.py`. All `torchrun` against
the directory's own `config.yaml`.
