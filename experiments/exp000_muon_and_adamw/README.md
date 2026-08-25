# exp000_muon_and_adamw

Baseline Muon-vs-AdamW comparison and learning-rate tuning. Establishes the
config used as the starting point for every later experiment: a 1-layer,
192-dim transformer (3 heads, 64 head-dim) trained on `data/fineweb10B_v500`
(vocab 500, BBPE), global batch 32768 tokens, `lr_cooldown_frac=0.7`. Only the
`two_d` rule's `update` (attention/MLP weight matrices: `muon` vs `adamw`)
differs between arms — embed, proj, 1D params, schedule, data, and seed
(`1337`) are identical. The `two_d` rule also uses the same `betas: [0.9,
0.95]` in both arms, and Muon's `nesterov` is turned off, so the two arms
share the same momentum dynamics and differ only in the update direction
(Newton-Schulz-orthogonalized vs raw Adam) applied to that momentum.

Analyzed in `articles/01_the_geometry/article.ipynb`, section 1.

## Subdirectories

- **`muon/`** — sweeps `rules.two_d.lr` over `{0.005, 0.01, 0.02, 0.04, 0.08}`
  with the `two_d` group on Muon, 8000 steps. Best: `lr=0.02`
  (final val loss 2.6966).
- **`adamw/`** — sweeps `rules.two_d.lr` over `{0.00075, 0.0015, 0.003,
  0.006, 0.012}` with the `two_d` group on AdamW, 12000 steps. Best:
  `lr=0.003` (final val loss 2.7016).
- **`muon_steps/`** — reruns Muon's best config (`lr=0.02`) at
  `train_steps={4000, 16000}` (the 8000-step point comes from `muon/`
  itself) to show 8000 steps is a convenient checkpoint for comparing
  dynamics, not a claim about convergence — val loss is still dropping
  fast at that point.

`muon/seed_1337/rules.two_d.lr_0.02` and
`adamw/seed_1337/rules.two_d.lr_0.003` are the two runs every other
experiment in this repo forks from.

## Running

Each subdirectory is a `run_optim_rules.py` job:

```bash
experiments/exp000_muon_and_adamw/muon/run.sh
experiments/exp000_muon_and_adamw/adamw/run.sh
experiments/exp000_muon_and_adamw/muon_steps/run.sh
```

Requires `data/fineweb10B_v500` (see the repo-root README's Data section).
Each `run.sh` `torchrun`s `src/muon_research/scripts/run_optim_rules.py`
against its own `config.yaml`, sweeping `override_args` into one run per
combination under `seed_<seed>/rules.two_d.lr_<lr>/`.
