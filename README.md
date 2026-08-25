# muon-research

The goal of this repository is to understand the dynamics and geometry of training with Muon and its variants.

## Geon

Every run in this repo trains under a single optimizer, `Geon` (`src/muon_research/optim/geon.py`). Geon keeps Adam-style state (`step`, `m`, `v`) for **every** parameter regardless of which update rule that parameter is currently using, and only decides at step time, per parameter, how to turn that state into an update. Four kinds of update rule are allowed, chosen independently per parameter and per step:

- **skip** — no update at all that step, but the parameter's `m`/`v` state keeps accumulating anyway, so it stays warm for whenever a real update rule resumes.
- **adamw** — the usual AdamW-style, per-element normalized update.
- **muon** — Muon's Newton-Schulz orthogonalization of the same underlying momentum signal.
- **exact SVD powers** — an exact-SVD variant of that same signal, reweighting its singular values by a power `p` (`p=0` recovers Muon exactly; other `p` over- or under-weight the dominant singular directions).

Because the state itself never depends on which rule is active, a parameter (or the whole model) can switch update rules mid-run without losing or resetting its accumulated optimizer state. That's what makes the experiments in this repo possible in the first place: forking a checkpoint and continuing under a different rule, switching back and forth, or KL-matching several rules' updates against each other at the same step, all rely on this.

Geon is a research tool, not a production optimizer: it favors simplicity and generality over speed (no sharding, no caching — every rank computes every update from scratch). We know this makes it inefficient, but since Geon is only ever used here to understand the underlying dynamics and patterns, not to train models quickly, that tradeoff is fine.

## Install and Testing

```bash
pip install -e ".[dev]"
pytest
```

## Data

Training and validation data is pretokenized [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) text, stored as flat `.bin` shards (a 256-`int32` header followed by `uint16` token ids) under `data/` at the repo root. `data/` is expected to be a symlink to wherever the actual shards live on disk (they're large and machine-specific, so they aren't committed) — point it at your own data store, e.g.:

```bash
ln -s /path/to/your/fineweb/data data
```

`data_source` in a config's `train:` block names a subdirectory under there (e.g. `data/fineweb10B_v500`), holding `fineweb_train_*.bin`/`fineweb_val_*.bin` shards plus, for custom-vocab data, the `tokenizer.json` that produced them.

Two prep jobs, both under `src/muon_research/download_data/`:

- **`cached_fineweb10B.py`** (or `cached_fineweb100B.py` / `cached_finewebedu10B.py`) — downloads the pre-tokenized, GPT-2-vocab (50257) FineWeb cache from the Hugging Face Hub straight into `data/fineweb10B/` (skips re-tokenizing from raw text, which otherwise takes about an hour):

  ```bash
  python src/muon_research/download_data/cached_fineweb10B.py        # full 10B tokens
  python src/muon_research/download_data/cached_fineweb10B.py 20     # first 20 shards only (~2B tokens)
  ```

- **`cached_fineweb10B_vocab.py`** — builds a smaller custom-vocab version (what the experiments in this repo actually train on, e.g. `fineweb10B_v500`): reconstructs documents from that GPT-2 cache, trains a byte-level BPE tokenizer of the requested size on a sample of the training documents, then retokenizes everything into new shards plus the `tokenizer.json`:

  ```bash
  python src/muon_research/download_data/cached_fineweb10B_vocab.py --vocab-size 500
  python src/muon_research/download_data/cached_fineweb10B_vocab.py --vocab-size 5000 --num-chunks 20
  ```

  Output goes to `data/fineweb10B_v{vocab_size}/` by default. `cached_fineweb10B_test.py` sanity-checks a custom-vocab directory against the GPT-2 cache it was built from (decodes both and diffs the document text).

## Experiments

`experiments/` holds the actual runs behind the articles: one directory per experiment (`expNNN_name/`), each with its own `README.md` explaining what it tests, which article section analyzes it, and how its subdirectories relate. Every run subdirectory pairs a `config.yaml` (train/model/rules, and any per-run `override_args` sweep) with a `run.sh` that `torchrun`s the corresponding script under `src/muon_research/scripts/` (`run_optim_rules.py`, `run_branch_compare.py`, `run_branch_continue.py`, `run_curv_profile.py`, ...) against it.

Later experiments generally fork checkpoints from earlier ones (e.g. `exp001`/`exp002`/`exp003` all fork from `exp000`'s baseline Muon/AdamW runs), so each experiment's own `README.md` documents the order its `run.sh` scripts need to run in — start there before launching anything. Reproducing a given article figure/finding just means running the experiment its `README.md` points to.
