"""Same run as run_geon.py / run_geon_category_rules.py, except every param's
full optimizer spec — update kind, sizing kind, lr, betas, nesterov, wd_raw — is
driven by an explicit list of ``rules`` instead of a handful of hardcoded
groups (embed/proj/1D/2D-block). Supports both GPT and GPL (``train.model_type``).

Each rule is ``{patterns, steps, update, sizing, lr, betas, nesterov, wd_raw[,
coeff][, warmup_steps][, name]}``:

- ``patterns`` — one or more fnmatch-style globs (same convention as
  ``ProfileConfig.param_name_patterns`` in curv.py) matched against
  each param's full dotted name from ``model.named_parameters()``, e.g.
  ``"blocks.*.attn.q.weight"`` or ``"*.bias"``. A single string is also
  accepted.
- ``steps`` — ``[start, end)`` (half-open, 0-indexed — the same ``step``
  loop variable the training loop uses, so a param's rules must partition
  ``[0, train_steps)``). ``end: null`` means "through ``train_steps``" --
  ``Rule.end`` stays ``None`` forever (never resolved to a concrete value),
  so it still means "through train_steps" even for a combo whose own
  ``override_args`` change ``train_steps`` -- not the base config's
  original value.
- ``update`` — ``"adamw"`` | ``"muon"`` | ``"skip"``, or ``[name, power]``
  for ``("svd_pow", power)``, or ``[name, q1, q2]`` for ``("svd_band", q1,
  q2)`` (Geon's phase-2 update kind; see muon_research/optim/geon.py's module
  docstring).
- ``sizing`` — ``"learning_rate"`` | ``"kl_match"`` | ``"fro_match"`` |
  ``"op_match"`` (Geon's phase-3 sizing kind; see muon_research/optim/geon.py's
  module docstring). ``kl_match`` needs ``model=``/``batches=``, so every
  step's microbatches are kept around and passed to ``optimizer.step()``
  unconditionally (harmless if no ``kl_match`` rule is in play). How often
  its binary search actually re-probes for a fresh coefficient, versus
  reusing the last one found, is controlled by the top-level
  ``klmatch_schedule`` config key (optional; see ``load_klmatch_schedule``
  and Geon's own ``Schedule``) — default is to always recompute, which is
  correct but can be expensive over a long run with many ``kl_match`` rules.
- ``lr``/``betas``/``nesterov``/``wd_raw`` — this rule's *base* (unscheduled)
  values; the usual stable-then-cooldown schedule (``lr_cooldown_frac``)
  still scales ``lr``/``wd_raw`` by a single global ``eta(step)``, same as
  run_geon.py, on top of whatever the active rule says. ``wd_raw`` is named
  for what Geon actually does with it: a raw multiplier applied directly
  (``p *= 1 - wd_raw``), NOT scaled by ``lr`` the way AdamW-style decoupled
  weight decay is — see muon_research/optim/geon.py's module docstring.
- ``coeff`` (default ``1.0``) — multiplies the lr fed into this rule's
  *sizing* entry specifically (not the group's own bookkeeping lr) — same
  convention as run_geon_category_rules.py.
- ``warmup_steps`` (default ``0``, i.e. no warmup) — linear lr warmup over
  this many steps, counted from this rule's own ``start`` (not absolute
  step 0), so a rule that only activates mid-training still gets a fresh
  warmup when it does; multiplies ``lr`` on top of ``eta(step)`` (``wd_raw``
  is untouched by it) -- see ``Rule.warmup_factor``.

Every param belongs to its own Geon param_group (see ``fork.build_model_and_geon``)
so its betas/nesterov/wd_raw/lr can be swapped independently as training crosses
a rule's step boundary — Geon reads all four live off ``group_of(p)`` every
``step()`` call, so this needs no special "switch groups" machinery, just
overwriting that group's fields in place each step (see
``RuleSet.apply_for_step``); Geon's own per-param momentum/variance state is
untouched by the switch (it's keyed by param identity, not group).

Rules for the *same* params sharing the *same* ``sizing`` at a given step are
matched jointly (one probe, one shared scale) whenever they're literally the
same rule — two different rules never get merged even if they happen to pick
the same sizing kind, mirroring run_geon_category_rules.py's ``build_sizings``.

Before training starts, ``fork.build_model_and_geon`` calls
``RuleSet.resolve``, which requires that, for *every* model parameter,
the matching rules' step ranges exactly partition ``[0, train_steps)`` — no
gaps, no overlaps — and that every rule matches at least one param; otherwise
it raises one combined ``ValueError`` listing every problem param/rule, and
training never starts.

Example config:

    train:
      data_source: ...
      train_steps: 2000
      report_steps: 100
      seq_len: 1024
      val_size: 1048576
      batch_size: 65536
      mbs: 16
      vocab_size: 50257
      num_layers: 12
      model_dim: 768
      model_type: gpt        # or: gpl
      head_dim: 64            # gpt only
      num_heads: 12           # gpt only
      # embed_dim / num_tokens instead, for model_type: gpl

    rules:
      - name: embed
        patterns: ["embed.weight"]
        steps: [0, null]
        update: adamw
        sizing: learning_rate
        lr: 0.3
        betas: [0.8, 0.95]
        nesterov: false
        wd_raw: 0.0

      - name: proj
        patterns: ["proj.weight"]
        steps: [0, null]
        update: adamw
        sizing: learning_rate
        lr: 0.003125
        betas: [0.8, 0.95]
        nesterov: false
        wd_raw: 0.0

      - name: 1d
        patterns: ["*.bias", "*.gains"]
        steps: [0, null]
        update: adamw
        sizing: learning_rate
        lr: 0.01
        betas: [0.8, 0.95]
        nesterov: false
        wd_raw: 0.0

      - name: blocks_early
        patterns: ["blocks.*.attn.*.weight", "blocks.*.mlp.*.weight"]
        steps: [0, 1000]
        update: adamw
        sizing: learning_rate
        lr: 0.03
        betas: [0.9, 0.95]
        nesterov: true
        wd_raw: 0.0003

      - name: blocks_late
        patterns: ["blocks.*.attn.*.weight", "blocks.*.mlp.*.weight"]
        steps: [1000, null]
        update: muon
        sizing: learning_rate
        lr: 0.03
        betas: [0.9, 0.95]
        nesterov: true
        wd_raw: 0.0003

    klmatch_schedule:          # optional; default: always recompute (see above)
      _type: schedule          # or {_type: ap, k: <int>} for a flat "every k steps"
      schedule:
        - [0, 100, 1]          # every step for the first 100 steps after a
                                # kl_match rule activates (see Rule.steps)
        - [100, 1000, 10]      # then every 10th step
        - [1000, 2000, 50]     # then every 50th -- last entry's end must
                                # reach this run's own train_steps, or later
                                # steps never re-probe at all (a step outside
                                # every entry's range is simply never due)

    override_args:            # optional; sweeps -- one run per combination
      seed: [0, 1, 2]          # any TrainConfig field name works here
      rules.blocks_early.lr: [0.01, 0.03]   # "rules.<rule name>.<field>"
      # targets one specific rule's field -- any Rule field except 'name'.
      # cartesian product across every listed key (here: 3x2 = 6 runs),
      # each with its own <run_path>/<key>_<value>/... subdirectory. See
      # load_override_args for the list-of-dicts form (separate
      # sweeps concatenated instead of cross-producted).
"""

# pylint: disable=all

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import yaml

from muon_research.data import DistributedDataCursor, distributed_data_generator
from muon_research.fork import (
    _makedirs_robust,
    build_model_and_geon,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
    train_config_num_params,
)
from muon_research.optim.geon import Schedule
from muon_research.paths import resolve_repo_path
from muon_research.rules import (
    RuleSet,
    TrainConfig,
    apply_overrides,
    load_override_args,
)
from muon_research.constants import (
    FILENAME_CONFIGS,
    FILENAME_DONE,
    FILENAME_LOGS,
    FILENAME_METRICS,
    ROLLING_LOSS_BETA,
)

CHECKPOINTS_DIRNAME = "checkpoints"


########################################
#                Config                #
########################################


def load_klmatch_schedule(path: str) -> Schedule:
    """Read the optional top-level 'klmatch_schedule' key -- how often a
    ``sizing: kl_match`` rule's binary search actually re-probes for its
    ``coeff`` versus reusing the last one found for that param (see
    ``Geon``'s own docstring and ``Schedule``) -- and wrap it into a
    ``Schedule``. Missing/``null`` (the default for configs that don't set
    it, e.g. ``payload.get(...)`` returning ``None``) becomes
    ``Schedule(None)``, "always recompute" -- today's behavior for every
    config that predates this option, unchanged.

    Applies to the *whole* run (there's only one model/optimizer here,
    unlike run_branch_compare.py's per-branch ``branch_config.
    klmatch_schedule``) -- not swept per ``override_args`` combo.
    """
    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    try:
        return Schedule(payload.get("klmatch_schedule"))
    except ValueError as e:
        raise ValueError(f"config {path!r}: klmatch_schedule invalid: {e}") from e


def _format_override_value(v) -> str:
    """Filesystem-friendly stringification for a run_path subdirectory name."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return "-".join(_format_override_value(x) for x in v)
    return str(v)


########################################
#            Training loop             #
########################################


def _should_report(step: int, train_config: TrainConfig) -> bool:
    """Validate/log at the final step, every ``report_steps``, and right
    before a checkpoint (so its saved state has a matching logged val_loss).
    ``step`` is "steps completed so far", the same convention checkpoint_steps
    uses -- see ``_should_checkpoint``."""
    return (
        step == train_config.train_steps
        or step % train_config.report_steps == 0
        or step in (train_config.checkpoint_steps or ())
    )


def _should_checkpoint(completed_steps: int, train_config: TrainConfig) -> bool:
    """Whether to checkpoint now, having completed ``completed_steps`` steps
    so far (0 before any training) -- the same "steps completed so far"
    convention ``checkpoint_steps`` uses."""
    return completed_steps in (train_config.checkpoint_steps or ())


def run_geon_rules(
    train_config: TrainConfig,
    rule_set: RuleSet,
    run_path: str,
    klmatch_schedule: Schedule = Schedule(None),
) -> None:
    """One run. Assumes the process group is already initialized (see
    ``main`` -- repeatedly init/destroy-ing a process group within one
    process is unsupported by PyTorch, so that happens once for the whole
    sweep, not once per run)."""
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))

    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    torch.cuda.manual_seed_all(train_config.seed)

    file_logs = os.path.join(run_path, FILENAME_LOGS)
    file_metrics = os.path.join(run_path, FILENAME_METRICS)
    file_configs = os.path.join(run_path, FILENAME_CONFIGS)
    file_done = os.path.join(run_path, FILENAME_DONE)
    checkpoint_dir = os.path.join(run_path, CHECKPOINTS_DIRNAME)

    # Every rank checks this identically -- single-node torchrun means every
    # rank sees the same local filesystem, so this needs no coordination.
    # This is deliberately NOT gated on rank==0 (unlike the old file_logs
    # check it replaces): if only rank 0 decided to skip, every other rank
    # would hang forever on its next collective call (dist.barrier/broadcast/
    # all_reduce) since it alone would never reach one. file_logs is no
    # longer used for this at all -- it can legitimately exist mid-run (e.g.
    # after a crash-and-resume), so its mere presence no longer means
    # "done"; only file_done, written once every step has run, does.
    if os.path.exists(file_done):
        if dist.get_rank() == 0:
            print(f"{file_done} exists, run already completed, skipping ...")
        # barrier so every rank leaves this run in lockstep before the next
        # run (if any) starts probing run_path/checkpoints -- the process
        # group is shared across the whole sweep, so this run's stragglers
        # would otherwise bleed into the next run's collectives.
        dist.barrier()
        return

    if dist.get_rank() == 0:
        _makedirs_robust(run_path)
        print("logs:    ", file_logs)
        print("metrics: ", file_metrics)
        print("configs: ", file_configs)
        config_payload = asdict(train_config)
        config_payload["rules"] = [asdict(rule) for rule in rule_set.rules]
        with open(file_configs, "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2)
    # Every rank looks for a checkpoint under run_path later (independently,
    # no coordination -- see find_latest_checkpoint); this barrier just
    # removes any doubt that rank 0 has finished creating run_path (and thus
    # that checkpoint_dir's absence really means "no checkpoints", not "not
    # created yet") before anyone looks.
    dist.barrier()

    def print0(s, console=False, log=True):
        if dist.get_rank() == 0:
            s = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] {s}"
            if console:
                print(s)
            if log:
                with open(file_logs, "a") as f:
                    print(s, file=f)

    def log_metric(**metric):
        if dist.get_rank() == 0:
            with open(file_metrics, "a", encoding="utf-8") as f:
                print(json.dumps(metric), file=f)

    print0("=" * 100)
    print0(f"Config: {train_config}")
    print0(f"Rules: {rule_set}")
    print0("=" * 100)
    print0(
        f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
    )
    print0("=" * 100)

    # DistributedDataCursor (not the simpler distributed_data_generator) --
    # its state_dict()/load_state_dict() are exactly the "fast skip tokens"
    # primitive checkpointing needs: resuming jumps straight to the saved
    # (file_idx, pos), no re-reading of already-consumed batches. The val
    # loader stays a one-shot generator read -- it's the same deterministic
    # batch every time regardless of resume, nothing to checkpoint there.
    train_loader = DistributedDataCursor(
        os.path.join(
            resolve_repo_path(train_config.data_source), train_config.train_data_pattern
        ),
        train_config.batch_size,
        vocab_size=train_config.vocab_size,
        seq_len=train_config.seq_len,
    )
    val_inputs, val_targets = next(
        distributed_data_generator(
            os.path.join(
                resolve_repo_path(train_config.data_source), train_config.val_data_pattern
            ),
            train_config.val_size,
            vocab_size=train_config.vocab_size,
            seq_len=train_config.seq_len,
        )
    )

    # Validates every rule covers exactly what it should (see
    # RuleSet.resolve) before any GPU memory / training step happens.
    built = build_model_and_geon(train_config, rule_set, compile_model=True)
    model, optimizer, rule_set = built["model"], built["optimizer"], built["rule_set"]
    # Governs how often any sizing: kl_match rule's binary search actually
    # re-probes vs. reuses its last coeff -- see load_klmatch_schedule.
    # Set before training starts (not passed into build_model_and_geon
    # itself, which every other caller -- fork_branch, load_checkpoint --
    # also uses without one) so it applies from step 0.
    optimizer.set_kl_match_cache_schedule(klmatch_schedule)
    named_params = list(model.named_parameters())
    print0(
        f"rules OK: {len(rule_set)} rules uniquely cover {len(named_params)} params "
        f"x [0, {train_config.train_steps}) steps",
        console=True,
    )

    # Resume: every rank independently finds + reads the same checkpoint
    # file off the shared filesystem (no broadcast needed -- model/optimizer/
    # train_loader state is bit-identical across ranks by construction, see
    # save_checkpoint) and restores it. Falls back to a fresh start (step 0,
    # freshly-initialized model broadcast from rank 0 below) if none exists.
    start_step = 0
    training_time = 0.0
    rolling_loss = 0.0
    rolling_loss_step = 0
    # Bias-corrected EMA train loss as of the last completed step -- set
    # inside the training section below, None until the first one runs (no
    # training loss to report yet at a fresh start's step-0 eval).
    unbiased_rolling_loss = None
    checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is not None:
        ckpt = load_checkpoint(checkpoint_path, model, optimizer, device=device)
        train_loader.load_state_dict(ckpt["train_loader"])
        start_step = ckpt["step"]
        training_time = ckpt["training_time"]
        rolling_loss = ckpt["rolling_loss"]
        rolling_loss_step = ckpt["rolling_loss_step"]
        print0(
            f"resuming from {checkpoint_path} at step {start_step}"
            f"/{train_config.train_steps}",
            console=True,
        )

    # No-op when resuming (every rank already agrees, having loaded the same
    # checkpoint above) -- needed only to sync each rank's own fresh random
    # init when starting from scratch.
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)
    # start the clock
    dist.barrier()
    t0 = time.perf_counter()
    for step in range(start_step, train_config.train_steps + 1):

        # --------------- VALIDATION SECTION -----------------
        if _should_report(step, train_config):
            # stop the clock
            dist.barrier()
            training_time += time.perf_counter() - t0
            model.eval()
            val_loss = 0
            with torch.no_grad():
                assert len(val_inputs) % train_config.mbs == 0
                for i in range(len(val_inputs) // train_config.mbs):
                    val_loss += model(
                        val_inputs[i * train_config.mbs : (i + 1) * train_config.mbs],
                        val_targets[i * train_config.mbs : (i + 1) * train_config.mbs],
                    )
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_loss /= train_config.val_size
            rolling_loss_str = (
                f"rolling_loss:{unbiased_rolling_loss:.5f} "
                if unbiased_rolling_loss is not None
                else ""
            )
            print0(
                f"step:{step}/{train_config.train_steps} {rolling_loss_str}"
                f"val_loss:{val_loss:.5f} "
                f"train_time:{training_time:.3f}s"
                + f" step_avg:{1000*training_time/max(step, 1):.2f}ms",
                console=True,
            )
            log_metric(
                kind="eval",
                step=step,
                val_loss=val_loss.item(),
                train_time=training_time,
            )
            model.train()
            # start the clock again
            dist.barrier()
            t0 = time.perf_counter()

        # step == 0 only happens on the loop's very first iteration, and
        # only when actually starting fresh (start_step == 0) -- a resumed
        # run's range() starts at start_step > 0, so step is never 0 there.
        # Checkpoints the freshly initialized model (0 steps completed, no
        # training yet) right after the validation section above has
        # already logged its matching val_loss -- same "checkpoint once its
        # state has a matching logged val_loss" convention every other
        # checkpoint below follows.
        if step == 0 and _should_checkpoint(0, train_config):
            save_checkpoint(
                checkpoint_dir,
                0,
                model,
                optimizer,
                train_loader.state_dict(),
                training_time=training_time,
                rolling_loss=rolling_loss,
                rolling_loss_step=rolling_loss_step,
            )
            print0("[checkpoint] saved step 0 (initialized model)", console=True)
            dist.barrier()

        if step == train_config.train_steps:
            break

        # --------------- TRAINING SECTION -----------------
        inputs, targets = next(train_loader)
        # accumulate across microbatches in case we are running with fewer than 8 gpus
        assert len(inputs) % train_config.mbs == 0
        train_loss = torch.zeros((), device=device)
        step_batches = []
        for i in range(len(inputs) // train_config.mbs):
            x_mb = inputs[i * train_config.mbs : (i + 1) * train_config.mbs]
            y_mb = targets[i * train_config.mbs : (i + 1) * train_config.mbs]
            loss = model(x_mb, y_mb)
            train_loss += loss.detach()
            loss.backward()
            step_batches.append((x_mb, y_mb))
        for name, p in model.named_parameters():
            assert p.grad is not None, name
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

        # resolve this step's per-param updates/sizings from the active
        # rules, and set every param's own group's lr/betas/nesterov/wd_raw
        eta = train_config.eta_of(step)
        updates, sizings = rule_set.apply_for_step(step, eta, named_params, optimizer)
        # model=/batches= only actually used if a kl_match rule is in play;
        # harmless to pass unconditionally otherwise.
        optimizer.step(updates, sizings, model=model, batches=step_batches)
        model.zero_grad(set_to_none=True)

        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        train_loss /= train_config.batch_size
        rolling_loss_step += 1
        rolling_loss = (
            ROLLING_LOSS_BETA * rolling_loss + (1 - ROLLING_LOSS_BETA) * train_loss
        )
        unbiased_rolling_loss = rolling_loss / (
            1 - ROLLING_LOSS_BETA**rolling_loss_step
        )

        approx_training_time = training_time + (time.perf_counter() - t0)
        print0(
            f"step:{step+1}/{train_config.train_steps} train_time:{approx_training_time:.3f}s"
            + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms",
            console=True,
            log=False,
        )
        log_metric(
            kind="train",
            step=step + 1,
            train_loss=train_loss.item(),
            rolling_loss=unbiased_rolling_loss.item(),
            train_time=approx_training_time,
        )

        if _should_checkpoint(step + 1, train_config):
            save_checkpoint(
                checkpoint_dir,
                step + 1,
                model,
                optimizer,
                train_loader.state_dict(),
                training_time=approx_training_time,
                rolling_loss=rolling_loss,
                rolling_loss_step=rolling_loss_step,
            )
            print0(f"[checkpoint] saved step {step + 1}", console=True)
            dist.barrier()

    # Every rank reaches here only after the loop's own final-iteration
    # dist.barrier() (in the validation section, right before the
    # step==train_steps break) -- so it's safe for rank 0 alone to write
    # file_done; every other rank is guaranteed to have finished too.
    dist.barrier()
    if dist.get_rank() == 0:
        with open(file_done, "w", encoding="utf-8") as f:
            f.write(f"{train_config.train_steps}\n")
    # barrier so every rank leaves this run in lockstep before the next run
    # (if any) starts -- see the process-group note at the top of this
    # function.
    dist.barrier()


########################################
#                  CLI                 #
########################################


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run_path",
        default="logs",
        help="Directory for run logs.",
    )
    p.add_argument(
        "--config_path",
        default=None,
        help="Path to a training config YAML (needs 'train' and 'rules'; "
        "'override_args'/'klmatch_schedule' are optional). Defaults to "
        "<run_path>/config.yaml. 'override_args' sweeps any TrainConfig field "
        "or 'rules.<name>.<field>' (cartesian product per dict, list-of-dicts "
        "concatenated) -- see load_override_args. Every resulting combo is run "
        "separately, with its own <run_path>/<key>_<value>/... subdirectory. "
        "'klmatch_schedule' (a muon_research.optim.geon Schedule spec, e.g. "
        "{_type: ap, k: 10}; default null, i.e. always recompute) controls how "
        "often any 'sizing: kl_match' rule's binary search actually re-probes "
        "vs. reuses its last coeff -- see load_klmatch_schedule; applies to "
        "every combo in this sweep alike, not swept itself.",
    )
    p.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Optional, for running a sweep across multiple parallel nodes: "
        "total number of shards this sweep's runs are split across. Requires "
        "--shard_index. All combinations are sorted by model size (ascending, "
        "see train_config_num_params) and dealt round-robin (i %% num_shards "
        "== shard_index), so every shard gets an even mix of small and large "
        "models instead of a contiguous size-sorted chunk.",
    )
    p.add_argument(
        "--shard_index",
        type=int,
        default=None,
        help="Optional: this invocation's shard index, in [0, num_shards). "
        "Requires --num_shards.",
    )
    args = p.parse_args()
    print("run_optim_rules ", args)

    if (args.num_shards is None) != (args.shard_index is None):
        p.error("--num_shards and --shard_index must be given together")
    if args.num_shards is not None:
        if args.num_shards <= 0:
            p.error(f"--num_shards must be positive, got {args.num_shards}")
        if not 0 <= args.shard_index < args.num_shards:
            p.error(
                f"--shard_index must be in [0, {args.num_shards}), "
                f"got {args.shard_index}"
            )

    config_path = args.config_path or os.path.join(args.run_path, "config.yaml")
    train_config = TrainConfig.load_from_config(config_path)
    rule_set = RuleSet.load_from_config(config_path)
    combos = load_override_args(config_path)
    klmatch_schedule = load_klmatch_schedule(config_path)

    runs = []
    for overrides in combos:
        combo_train_config, combo_rule_set = apply_overrides(
            train_config, rule_set, overrides
        )
        run_path = args.run_path
        for key, value in overrides.items():
            run_path = os.path.join(run_path, f"{key}_{_format_override_value(value)}")
        runs.append((combo_train_config, combo_rule_set, run_path))
    if not runs:
        runs = [(train_config, rule_set, args.run_path)]

    if args.num_shards is not None:
        runs.sort(key=lambda run: train_config_num_params(run[0]))
        runs = [
            run for i, run in enumerate(runs) if i % args.num_shards == args.shard_index
        ]
        print(
            f"shard {args.shard_index}/{args.num_shards}: "
            f"{len(runs)}/{max(len(combos), 1)} run(s)"
        )

    # Initialized once for the whole sweep, not once per run: PyTorch
    # explicitly documents repeated init/destroy of a process group within
    # one process as unsupported/untested, and requires external (i.e. not
    # torch.distributed) synchronization between a destroy and the next
    # init -- which a plain Python loop calling run_geon_rules per run
    # can't provide. One process group, torn down once at the very end,
    # sidesteps that entirely.
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    # this code can be run equivalently with 1, 2, 4, or 8 gpus.
    assert 8 % dist.get_world_size() == 0
    dist.barrier()
    try:
        for tc, combo_rule_set, run_path in runs:
            run_geon_rules(
                tc, combo_rule_set, run_path=run_path, klmatch_schedule=klmatch_schedule
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    # torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) ...
    main()
