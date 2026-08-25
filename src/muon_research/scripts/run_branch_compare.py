"""N-branch KL-matched comparison of a set of overrides, from a
run_optim_rules.py checkpoint.

For each ``(path, step)`` job (``path`` is a run_optim_rules.py ``run_path``,
``step`` one of its saved ``checkpoints/step_<n>.pt``): load that run's
``config.json`` (train fields + rules) and checkpoint *unmodified* -- same
``train_steps``, same rules, same schedule as the original run, exactly the
regular resume path ``run_optim_rules.py`` itself would take.

1. **Continue.** Resume training normally (real ``Geon.step()``, same shape
   as ``run_optim_rules.run_geon_rules``, train-loss logging included) for
   ``prefork_steps`` steps (a run's own ``prefork_steps`` if it gave one,
   else ``branch_config.prefork_steps``), up through ``end_step = step +
   prefork_steps`` (which must not exceed the checkpoint's own
   ``train_steps`` -- the config only defines the schedule/rules that far,
   checked up front).

2. **Fork, once, at end_step.** Clone the trunk's ``(model, optimizer)``
   once per entry in the run's ``branch_specs`` (a list of ``BranchSpec``,
   each a ``(name, override, kl_matched)`` -- see ``BranchSpec``) -- via
   ``fork.fork_branch`` (the shared forking helper every script that clones
   a trunk uses): each a fresh
   ``fork.build_model_and_geon`` construction (ties each branch's own
   model params and optimizer param_groups together correctly by
   construction) loaded from the trunk's own ``state_dict()``s, not
   ``copy.deepcopy``: ``Optimizer``'s pickling protocol only preserves
   ``{state, param_groups}``, silently dropping Geon's own
   ``_step_count``/``s_min``/``s_max``/etc, so a deep-copied optimizer
   would be missing them entirely. Model/optimizer state is loaded
   carefully to avoid aliasing the trunk's own live tensors -- see
   ``fork.fork_branch``'s own docstring. Each branch's own directory is
   named after its ``BranchSpec.name`` (must be unique within the run).

3. **fork_steps.** One shared batch per outer-loop iteration (``fork_steps``
   of them; a run's own ``fork_steps`` if it gave one, else
   ``branch_config.fork_steps``), read off a clone of the trunk's cursor
   (cheap -- shares the already-loaded shard tensor, no disk re-read) so
   exploration never advances the trunk's own cursor. Every branch's
   rule-derived "typical" update (kind/lr/warmup) is resolved at
   ``end_step + i`` (iteration ``i``'s real corresponding step, *not* frozen
   at ``end_step`` -- eta and rule/warmup selection track the schedule
   exactly as continued real training would, so ``end_step + fork_steps``
   must not exceed ``train_steps`` either, checked up front alongside
   ``end_step`` itself) via Geon's own phases
   (``_refresh_state``, ``_direction``, ``_resolve_sizes`` -- reached into
   directly: the "shared params now, the rest later, differently" split
   can't go through ``Geon.step()`` twice, since ``_refresh_state`` advances
   *every* param's Adam step/momentum unconditionally on every call).
   ``branch_config.shared_patterns``-matched params always get each
   branch's own typical update, unconditionally, no KL matching; call the
   complement "the rest":

   * The **branch at index 0** (the *reference*; ``kl_matched`` must be
     ``False``, enforced at load time) applies its own typical update to
     the rest for real, unscaled; that update's own KL (measured on
     ``rest`` alone) is ``target_kl`` for this iteration, needed by every
     ``kl_matched`` branch below regardless of whether this iteration is
     logged.
   * Every **other branch with ``kl_matched=False``** applies its own
     typical update to the rest for real, unscaled -- its natural,
     unconstrained trajectory.
   * Every **other branch with ``kl_matched=True``** applies its own
     typical *direction* to the rest, scaled (binary search, log-space,
     mirroring ``Geon._kl_matched_size``) to match the reference branch's
     ``target_kl`` -- i.e. this override's direction under the same KL
     budget the reference branch used.

   Comparing a regular branch against a kl_matched branch of the same
   override isolates "does this override do better/worse because it moves
   more/less in KL terms" from comparing it against the reference branch,
   which isolates "how does this override's direction compare to the
   reference override's, given the same KL budget."

   Every branch's real update above always applies, every iteration.
   ``branch_config.metric_schedule`` (a ``Schedule``, see its own docstring
   -- default unrestricted, i.e. every iteration) only controls which
   iterations bother *measuring/logging* it: iteration ``i`` (0-based,
   local to this job's own fork window, *not* the absolute train step) is
   logged if ``metric_schedule.should_do(i)``.

   Independently, ``branch_config.checkpoint_after_steps`` (default
   ``None`` -- none) is a plain list of *durations* (number of fork-window
   steps to train before checkpointing -- 1 means "after the first
   iteration", not "at iteration index 1"; contrast with ``metric_schedule``
   above, which *is* a 0-based iteration index) at which every branch gets
   its ``(model, optimizer)`` plus the shared data-cursor position
   checkpointed to
   ``<run_path>/branches/<branch_spec.name>/checkpoints/step_<n>.pt`` (``n =
   fork_step + d`` for duration ``d`` -- "steps completed so far", matching
   every other checkpoint/logged-metric step in this codebase) -- in the
   same format run_optim_rules.py's own checkpoints use, so a later job can
   load a few of these back, verify they share the same data-cursor
   position (they will, since every branch reads off the same cloned
   cursor), and continue them under one new, shared override -- see
   run_branch_continue.py. ``d=0`` is allowed too, writing a checkpoint
   right at the fork point (before any of this branch's own training) --
   a verbatim, if redundant, copy of ``source_path``'s own checkpoint at
   ``fork_step``, useful purely so a downstream config can address every
   duration the same way instead of special-casing 0.

4. **Discard.** Every branch that ran is dropped; nothing else happens to
   this job afterward (the fork is the last thing that happens) beyond any
   checkpoints written above.

Metrics: trunk logs ``kind="train"`` exactly like run_optim_rules.py, to
``<run_path>/metrics.jsonl``. Each branch gets its own directory --
``<run_path>/branches/<branch_spec.name>/`` -- with one ``kind="applied"``
metric entry per logged iteration (see ``metric_schedule`` above):
``init_val_loss`` (this branch's val_loss right after the fork),
``post_val_loss`` (this iteration's val_loss after its real update), ``kl``,
``train_loss``, and ``scale``/``target_kl``. That same directory also gets a
``checkpoints/`` subdirectory when ``checkpoint_after_steps`` triggered any
(see above).

Example config:

    runs:
      - name: muon
        path: experiments/post/000_lr_tune/seed_1337/rules.two_d.lr_0.04
        steps: [10, 100, 500, 1000, 2000, 3000]
        branch_specs:
          - name: adamw_ref
            override:
                rules.two_d.update: adamw
                rules.two_d.lr: 0.006
                rules.two_d.betas: [0.9, 0.95]
                rules.two_d.nesterov: false
                rules.two_d.wd_raw: 0.0002
                rules.two_d.warmup_steps: 40
                rules.two_d.coeff: 1.0
          - name: muon_regular
            override:
                rules.two_d.update: muon
                rules.two_d.lr: 0.04
                rules.two_d.betas: [0.9, 0.95]
                rules.two_d.nesterov: true
                rules.two_d.wd_raw: 0.0002
                rules.two_d.warmup_steps: 0
                rules.two_d.coeff: 1.0
          - name: muon_kl_matched
            override:
                rules.two_d.update: muon
                rules.two_d.lr: 0.04
                rules.two_d.betas: [0.9, 0.95]
                rules.two_d.nesterov: true
                rules.two_d.wd_raw: 0.0002
                rules.two_d.warmup_steps: 0
                rules.two_d.coeff: 1.0
            kl_matched: true
        # optional per-run overrides of the branch_config fields below;
        # omit to fall back to branch_config's value.
        fork_steps: 10
        prefork_steps: 5
        checkpoint_after_steps: [5]  # optional; re-fork point(s) for a later job
        metric_schedule: {_type: schedule, schedule: [[0, 10, 1], [10, 500, 10]]}  # dense early, sparse later
        klmatch_schedule: {_type: ap, k: 10}  # optional; re-fit kl_match coeffs every 10 steps

    branch_config:
      prefork_steps: 20
      fork_steps: 5
      shared_patterns: ["embed.weight", "*.bias", "*.gains"]
      mbs: 16  # optional; overrides every job's own checkpoint mbs if given
      kl_batch_size: 8192  # optional; restricts KL computations to this many
                            # leading tokens of each batch (must divide mbs * seq_len)
      checkpoint_after_steps: []  # optional; default checkpoints nothing
      metric_schedule: {_type: ap, k: 1}  # optional; default (omit, or null) logs every iteration
      klmatch_schedule: null  # optional; default always re-fits kl_match coeffs

# metric_schedule/klmatch_schedule both take a muon_research/optim/geon.py Schedule
# spec -- None (log/refit every iteration), {_type: ap, k: <int>} (every k
# steps), or {_type: schedule, schedule: [[start, end, k], ...]} (dense
# early, sparse later -- see Schedule's own docstring).
"""

# pylint: disable=all

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from fnmatch import fnmatch

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

from muon_research.data import DistributedDataCursor, distributed_data_generator
from muon_research.optim.geon import BinarySearch, Geon, Schedule
from muon_research.paths import resolve_repo_path
from muon_research import fork
from muon_research.constants import (
    FILENAME_CONFIGS,
    FILENAME_DONE,
    FILENAME_LOGS,
    FILENAME_METRICS,
    ROLLING_LOSS_BETA,
)
from muon_research.rules import (
    RuleSet,
    TrainConfig,
    apply_overrides,
    load_checkpoint_config,
)
from muon_research.scripts.run_optim_rules import CHECKPOINTS_DIRNAME

########################################
#                Config                #
########################################


@dataclass
class BranchSpec:
    """One forked branch. ``branch_index`` is assigned by position in the
    owning run's ``branch_specs`` list (0-based, in list order) -- not
    user-set; any ``branch_index`` given in YAML is ignored.

    ``branch_index == 0`` is always the *reference* branch: applies its own
    (``override``) typical update to ``rest`` for real, unscaled, every
    iteration; that update's own KL (on ``rest`` alone) is ``target_kl`` for
    every other branch this iteration that sets ``kl_matched=True``. Must
    have ``kl_matched=False`` (enforced by ``_parse_branch_specs``).

    Every other branch (``branch_index >= 1``) applies its own typical
    *direction* to ``rest``, either:

    * ``kl_matched=False`` -- for real, unscaled (its own natural,
      unconstrained trajectory).
    * ``kl_matched=True`` -- scaled (binary search, log-space, mirroring
      ``Geon._kl_matched_size``) to match the reference branch's
      ``target_kl`` for this iteration.

    ``branch_config.shared_patterns``-matched params always get each
    branch's own typical update unconditionally, regardless of
    ``kl_matched``. ``name`` becomes this branch's own directory name
    (``<run_path>/branches/<name>/``), so it must be unique within the run.
    """

    name: str
    override: dict
    kl_matched: bool = False
    branch_index: int = 0


@dataclass
class RunSpec:
    name: str
    # A run_optim_rules.py run_path (this checkpoint's source). Kept exactly
    # as given in the YAML (relative or absolute) -- resolved against the
    # repo root on demand, at each actual file-I/O call site, so it stays
    # portable across checkouts and round-trips unchanged into
    # config.json/logs.
    path: str
    steps: list[int]
    branch_specs: list[BranchSpec]
    # Per-run overrides of the same-named branch_config fields; None (default)
    # falls back to branch_config's value.
    fork_steps: int | None = None
    prefork_steps: int | None = None
    checkpoint_after_steps: list[int] | None = None
    metric_schedule: Schedule | None = None
    klmatch_schedule: Schedule | None = None


@dataclass
class BranchConfig:
    prefork_steps: int
    fork_steps: int
    shared_patterns: list[str]
    kl_search_iters: int = 16
    kl_scale_init: float = 1.0
    kl_search_expand_factor: float = 10.0
    # ---- TrainConfig field overrides ----
    # mbs/batch_size/val_size each override that field of every job's own
    # (checkpoint config's) TrainConfig when set; None (default) leaves it
    # as the checkpoint's own config.json has it. Applied once, in
    # run_checkpoint_branch, so every downstream use -- the trunk's own
    # training loop and everything fork_and_explore does -- picks it up
    # automatically, no further plumbing needed.
    # microbatch granularity; must evenly divide batch_size/val_size
    mbs: int | None = None
    batch_size: int | None = None
    val_size: int | None = None
    # ---- KL-matching-only override ----
    # Restricts fork_and_explore's own KL computations (not the typical
    # direction/size resolution, which still sees the full batch) to the
    # first kl_batch_size tokens of each iteration's shared batch, when set.
    # None (default) means the full batch. Must be divisible by (effective
    # mbs * seq_len) -- one mbs chunk's worth of tokens -- checked once
    # train_config (and any mbs override above) is resolved, since
    # seq_len/mbs vary per job; see run_checkpoint_branch.
    kl_batch_size: int | None = None
    # ---- Branch checkpointing (for a later re-fork) ----
    # Durations (number of fork-window steps to train before checkpointing
    # -- 1 means "after the first iteration", *not* "at iteration index 1";
    # unlike metric_schedule, this is a count, not a 0-based iteration
    # index) at which every branch (see the run's own branch_specs) gets
    # its (model, optimizer, shared data-cursor position) checkpointed to
    # ``<run_path>/branches/<branch_spec.name>/checkpoints/step_<n>.pt``
    # (``n`` is the absolute step, ``fork_step + d`` -- "steps completed so
    # far", matching every other checkpoint's naming in this codebase). ``0``
    # is allowed too -- "right at the fork point, before any of this
    # branch's own training" -- writing a checkpoint that's a verbatim (if
    # redundant) copy of source_path's own checkpoint at fork_step, purely
    # so a downstream job can address every duration uniformly (e.g.
    # run_branch_continue.py's own path/branch_names/checkpoint_step,
    # without special-casing 0 to mean "read from source_path instead").
    # None/empty (default) checkpoints nothing.
    # Independent of metric_schedule -- checkpointing and metric logging are
    # separate concerns. Written in the same format run_optim_rules.py's own
    # checkpoints use, so a later job can resume from one exactly the same
    # way (see e.g. run_branch_continue.py).
    checkpoint_after_steps: list[int] | None = None
    # ---- Metric sparsification ----
    # A Schedule (see its own docstring for the accepted spec shapes --
    # {"_type": "ap"/"schedule", ...} or None) controlling which fork
    # iterations (0-based, local to this job's own fork window -- not the
    # absolute train step) actually compute+log branch metrics: iteration i
    # is logged if ``.should_do(i)``. Default (an unrestricted Schedule)
    # logs every iteration. Only trims *diagnostic* work (post-update
    # val_loss eval, logging itself) -- every branch's real update still
    # applies every iteration regardless, so skipped iterations don't
    # change training, only what gets measured about them. Always resolved
    # to an actual Schedule by load_branch_yaml -- never None itself
    # (unlike RunSpec's own same-named field, which uses None to mean "not
    # given, fall back to this").
    metric_schedule: Schedule = Schedule(None)
    # ---- KL-match coefficient caching ----
    # A Schedule (see its own docstring) controlling how often kl_match
    # sizing actually re-searches its coefficient vs. reuses a cached one --
    # both for each branch's own Geon (any rule-level "sizing: kl_match"
    # entry, via Geon.set_kl_match_cache_schedule -- see fork_branch) and
    # for fork_and_explore's own separate kl_matched=True branch-spec
    # matching (via a second, independent Schedule instance, driven by the
    # fork loop's own 0-based iteration index -- Geon's internal _step_count
    # never advances during fork exploration, since branches never call
    # optimizer.step() there, only its phases directly; see
    # fork_and_explore). Default (an unrestricted Schedule) means always
    # recompute, in both places -- today's behavior, unchanged unless opted
    # in. Always resolved to an actual Schedule by load_branch_yaml -- never
    # None itself (unlike RunSpec's own same-named field).
    klmatch_schedule: Schedule = Schedule(None)
    # ---- Compilation ----
    # model.compile(dynamic=False, fullgraph=True) on the trunk and every
    # branch (safe now that branches are freshly constructed + load_state_dict,
    # not copy.deepcopy'd from an already-compiled trunk). Only speeds up
    # forward()/__call__ (the real _forward_backward update path) -- the
    # KL probes (_probe_kl/_counterfactual_measure/_kl_matched_scale) call
    # model.logits() directly, bypassing __call__, so they stay eager
    # regardless. Compilation only actually happens (and costs time) on
    # first invocation, not at the .compile() call site; dynamo's cache is
    # keyed on the traced graph (shapes/dtypes), not the model instance or
    # its weight values, so only the very first model built in this process
    # pays the real cost -- every later one (branch or trunk, this job or
    # the next one in the same --num_shards process) reuses it almost for
    # free. Each compile's wall time is measured (a throwaway warmup
    # forward+backward, discarded via zero_grad right after) and printed.
    compile_models: bool = True


def _parse_checkpoint_after_steps(raw, *, where: str) -> list[int] | None:
    """``[d, ...]`` (durations -- see ``BranchConfig.checkpoint_after_steps``;
    ``0`` means "right at the fork point, before any training") -> sorted,
    deduped ``[d, ...]``, or None if ``raw`` is None/missing."""
    if raw is None:
        return None
    durations = sorted({int(x) for x in raw})
    for d in durations:
        if d < 0:
            raise ValueError(f"{where}: checkpoint_after_steps entry {d} must be >= 0")
    return durations


def _parse_schedule_spec(raw, *, where: str, field_name: str) -> Schedule:
    """Validates ``raw`` as a ``Schedule`` spec (see its docstring) and
    wraps it into an actual ``Schedule`` -- ``None``/missing becomes
    ``Schedule(None)`` (its own "always due" value), since ``BranchConfig``'s
    resolved fields have no further fallback and so are never ``None``
    themselves. Shared by ``metric_schedule`` and ``klmatch_schedule`` -- see
    ``BranchConfig``. RunSpec's own per-run overrides need a genuine
    None-means-not-given sentinel instead -- see
    ``_parse_run_schedule_override``."""
    try:
        return Schedule(raw)
    except ValueError as e:
        raise ValueError(f"{where}: {field_name} invalid: {e}") from e


def _parse_run_schedule_override(
    raw, *, where: str, field_name: str
) -> Schedule | None:
    """Same as ``_parse_schedule_spec``, but preserves ``None`` (missing
    from this run's own YAML entry -- "not given, fall back to
    branch_config's value") instead of collapsing it to ``Schedule(None)``
    ("given, and unrestricted"). See ``RunSpec``."""
    if raw is None:
        return None
    return _parse_schedule_spec(raw, where=where, field_name=field_name)


def _parse_branch_specs(raw, *, where: str) -> list[BranchSpec]:
    """``[{name, override, kl_matched}, ...]`` -> ``[BranchSpec, ...]``, with
    ``branch_index`` assigned by position (0-based, matching list order).
    See ``BranchSpec``."""
    if not raw:
        raise ValueError(f"{where}: branch_specs must be a non-empty list")
    specs = []
    seen_names = set()
    for i, entry in enumerate(raw):
        name = str(entry["name"])
        if name in seen_names:
            raise ValueError(f"{where}: duplicate branch_spec name {name!r}")
        seen_names.add(name)
        override = dict(entry["override"])
        if not override:
            raise ValueError(f"{where}: branch_spec {name!r} has an empty override")
        kl_matched = bool(entry.get("kl_matched", False))
        specs.append(
            BranchSpec(
                name=name, override=override, kl_matched=kl_matched, branch_index=i
            )
        )
    if specs[0].kl_matched:
        raise ValueError(
            f"{where}: first branch_spec ({specs[0].name!r}) must have "
            f"kl_matched=False -- it's the reference branch that defines target_kl"
        )
    return specs


def load_branch_yaml(path: str) -> tuple[list[RunSpec], BranchConfig]:
    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    raw_runs = payload.get("runs") or []
    if not raw_runs:
        raise ValueError(f"config {path!r} has no 'runs' entries")
    runs = []
    # (name, step) is what actually determines each job's output directory
    # (see _job_run_path) -- so that's the uniqueness constraint, not name
    # alone: two runs sharing a name but with disjoint steps never collide.
    job_dirs_seen = set()
    for entry in raw_runs:
        name = str(entry["name"])
        steps = [int(s) for s in entry["steps"]]
        if not steps:
            raise ValueError(f"run {name!r} has an empty 'steps' list")
        for step in steps:
            job_dir = (name, step)
            if job_dir in job_dirs_seen:
                raise ValueError(
                    f"duplicate (name, step) = {job_dir!r} -- these determine "
                    f"each job's output directory (<run_path>/{name}/"
                    f"step_{step:06d}), which must be unique"
                )
            job_dirs_seen.add(job_dir)
        runs.append(
            RunSpec(
                name=name,
                path=str(entry["path"]),
                steps=steps,
                branch_specs=_parse_branch_specs(
                    entry.get("branch_specs"), where=f"run {name!r}"
                ),
                fork_steps=(
                    int(entry["fork_steps"])
                    if entry.get("fork_steps") is not None
                    else None
                ),
                prefork_steps=(
                    int(entry["prefork_steps"])
                    if entry.get("prefork_steps") is not None
                    else None
                ),
                checkpoint_after_steps=_parse_checkpoint_after_steps(
                    entry.get("checkpoint_after_steps"), where=f"run {name!r}"
                ),
                metric_schedule=_parse_run_schedule_override(
                    entry.get("metric_schedule"),
                    where=f"run {name!r}",
                    field_name="metric_schedule",
                ),
                klmatch_schedule=_parse_run_schedule_override(
                    entry.get("klmatch_schedule"),
                    where=f"run {name!r}",
                    field_name="klmatch_schedule",
                ),
            )
        )

    bc = payload.get("branch_config") or {}
    for key in ("prefork_steps", "fork_steps", "shared_patterns"):
        if key not in bc:
            raise ValueError(f"config {path!r} branch_config is missing key {key!r}")

    branch_config = BranchConfig(
        prefork_steps=int(bc["prefork_steps"]),
        fork_steps=int(bc["fork_steps"]),
        shared_patterns=[str(p) for p in bc["shared_patterns"]],
        kl_search_iters=int(bc.get("kl_search_iters", 16)),
        kl_scale_init=float(bc.get("kl_scale_init", 1.0)),
        kl_search_expand_factor=float(bc.get("kl_search_expand_factor", 10.0)),
        mbs=(int(bc["mbs"]) if "mbs" in bc and bc["mbs"] is not None else None),
        batch_size=(
            int(bc["batch_size"])
            if "batch_size" in bc and bc["batch_size"] is not None
            else None
        ),
        val_size=(
            int(bc["val_size"])
            if "val_size" in bc and bc["val_size"] is not None
            else None
        ),
        kl_batch_size=(
            int(bc["kl_batch_size"])
            if "kl_batch_size" in bc and bc["kl_batch_size"] is not None
            else None
        ),
        checkpoint_after_steps=_parse_checkpoint_after_steps(
            bc.get("checkpoint_after_steps"), where="branch_config"
        ),
        metric_schedule=_parse_schedule_spec(
            bc.get("metric_schedule"),
            where="branch_config",
            field_name="metric_schedule",
        ),
        klmatch_schedule=_parse_schedule_spec(
            bc.get("klmatch_schedule"),
            where="branch_config",
            field_name="klmatch_schedule",
        ),
        compile_models=bool(bc.get("compile_models", True)),
    )
    if branch_config.prefork_steps < 0:
        raise ValueError("branch_config.prefork_steps must not be negative")
    if branch_config.fork_steps <= 0:
        raise ValueError("branch_config.fork_steps must be positive")
    if branch_config.mbs is not None and branch_config.mbs <= 0:
        raise ValueError("branch_config.mbs must be positive if given")
    if branch_config.kl_batch_size is not None and branch_config.kl_batch_size <= 0:
        raise ValueError("branch_config.kl_batch_size must be positive if given")
    if branch_config.batch_size is not None and branch_config.batch_size <= 0:
        raise ValueError("branch_config.batch_size must be positive if given")
    if branch_config.val_size is not None and branch_config.val_size <= 0:
        raise ValueError("branch_config.val_size must be positive if given")

    for run in runs:
        if run.fork_steps is not None and run.fork_steps <= 0:
            raise ValueError(f"run {run.name!r}: fork_steps must be positive if given")
        if run.prefork_steps is not None and run.prefork_steps < 0:
            raise ValueError(
                f"run {run.name!r}: prefork_steps must not be negative " f"if given"
            )
    return runs, branch_config


def load_compare_job_config(job_run_path: str) -> tuple[TrainConfig, RuleSet]:
    """Read one of *this* script's own job-level config.json files (the
    nested ``train=asdict(...)``/``rules=[...]`` shape ``run_checkpoint_branch``
    writes -- not run_optim_rules.py's flat one, see ``load_checkpoint_config``)
    back into ``(TrainConfig, RuleSet)``. Used by run_branch_continue.py
    to recover the *original* (pre-override) train_config/rules a job's
    branches forked from, so a new shared override can be applied to that
    same baseline rather than to whichever override a branch happened to be
    using."""
    with open(os.path.join(job_run_path, FILENAME_CONFIGS), encoding="utf-8") as f:
        payload = json.load(f)
    train_fields = {f.name for f in fields(TrainConfig)}
    train_config = TrainConfig(
        **{k: v for k, v in payload["train"].items() if k in train_fields}
    )
    return train_config, RuleSet.load_from_payload(payload["rules"])


def _validate_branch_specs(branch_specs: list[BranchSpec], rule_set: RuleSet) -> None:
    for spec in branch_specs:
        rule_set.validate_override(spec.override)


########################################
#        Forward / eval helpers        #
########################################


def _forward_backward(
    model, train_config: TrainConfig, x: torch.Tensor, y: torch.Tensor
):
    """mbs-chunked forward+backward, grad all_reduce -- returns
    ``(train_loss, mbs_batches)``."""
    assert len(x) % train_config.mbs == 0
    train_loss = torch.zeros((), device=x.device)
    mbs_batches = []
    for i in range(len(x) // train_config.mbs):
        x_mb = x[i * train_config.mbs : (i + 1) * train_config.mbs]
        y_mb = y[i * train_config.mbs : (i + 1) * train_config.mbs]
        loss = model(x_mb, y_mb)
        train_loss += loss.detach()
        loss.backward()
        mbs_batches.append((x_mb, y_mb))
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
    train_loss /= train_config.batch_size
    return train_loss, mbs_batches


@torch.no_grad()
def _evaluate(
    model, val_inputs, val_targets, train_config: TrainConfig
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    assert len(val_inputs) % train_config.mbs == 0
    val_loss = torch.zeros((), device=val_inputs.device)
    for i in range(len(val_inputs) // train_config.mbs):
        val_loss += model(
            val_inputs[i * train_config.mbs : (i + 1) * train_config.mbs],
            val_targets[i * train_config.mbs : (i + 1) * train_config.mbs],
        )
    dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
    val_loss /= train_config.val_size
    model.train(was_training)
    return val_loss


########################################
#      Update application / KL probes  #
########################################


@torch.no_grad()
def _apply_update(
    params, optimizer: Geon, directions: dict, sizes: dict, scale: float = 1.0
) -> None:
    """``p <- (1 - wd_raw) * p - scale * size * direction`` -- same write
    ``Geon.step()`` does; ``wd_raw`` is always applied at its configured
    strength, never scaled by ``scale`` (only the size*direction term is)."""
    for p in params:
        wd_raw = float(optimizer.group_of(p)["wd_raw"])
        if wd_raw != 0.0:
            p.mul_(1.0 - wd_raw)
        p.add_(directions[p].to(dtype=p.dtype), alpha=-scale * float(sizes[p]))


@torch.no_grad()
def _probe_kl(model, mbs_batches, params, apply_fn) -> float:
    """Token-mean KL(p_before || p_after) that ``apply_fn()`` (an in-place
    mutation of ``params``) would cause on ``mbs_batches`` -- snapshots
    ``params`` first and restores them after, so this never permanently
    changes the model. Same probe/restore shape as the KL probe Geon runs
    internally for its own ``kl_match`` sizing, generalized to an
    arbitrary mutation (needed here so weight decay -- which an
    additive-delta-only design can't express -- is included in the
    measurement)."""
    saved = {p: p.data.clone() for p in params}
    device = next(model.parameters()).device
    kl_sum = torch.zeros((), device=device, dtype=torch.float64)
    ntok = torch.zeros((), device=device, dtype=torch.float64)
    was_training = model.training
    model.eval()
    try:
        for x, y in mbs_batches:
            for p in params:
                p.copy_(saved[p])
            logp_before = F.log_softmax(model.logits(x).float(), dim=-1)
            apply_fn()
            logp_after = F.log_softmax(model.logits(x).float(), dim=-1)
            kl_sum = kl_sum + (logp_before.exp() * (logp_before - logp_after)).sum()
            ntok = ntok + float(y.numel())
        t = torch.stack([kl_sum, ntok])
        if dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t[0].item() / max(t[1].item(), 1.0))
    finally:
        model.train(was_training)
        for p, w in saved.items():
            p.copy_(w)


def _kl_matched_scale(
    model,
    mbs_batches,
    params,
    optimizer: Geon,
    directions: dict,
    sizes: dict,
    target_kl: float,
    *,
    kl_search_iters: int,
    kl_scale_init: float,
    kl_search_expand_factor: float,
) -> tuple[float, float]:
    """Binary-search (mirrors ``Geon._kl_matched_size``, via the same
    shared ``BinarySearch.bracketed_binary_search``) a ``scale`` for
    ``_apply_update(params, ..., scale)`` on ``params`` matching
    ``target_kl`` on ``mbs_batches``. Returns ``(scale, kl_at_scale)``."""
    if not params:
        return 1.0, 0.0
    if target_kl <= 0.0 or not math.isfinite(target_kl):
        kl = _probe_kl(
            model,
            mbs_batches,
            params,
            lambda: _apply_update(params, optimizer, directions, sizes, scale=1.0),
        )
        return 1.0, kl
    return BinarySearch.bracketed_binary_search(
        lambda s: _probe_kl(
            model,
            mbs_batches,
            params,
            lambda: _apply_update(params, optimizer, directions, sizes, scale=s),
        ),
        target_kl,
        init=kl_scale_init,
        iters=kl_search_iters,
        expand_factor=kl_search_expand_factor,
    )


########################################
#           Fork / explore             #
########################################


def _select_active_shared_rest(named_params, updates_typical, shared_patterns):
    active_named = [(n, p) for n, p in named_params if updates_typical[p] != "skip"]
    active = [p for _n, p in active_named]
    shared = [
        p for n, p in active_named if any(fnmatch(n, pat) for pat in shared_patterns)
    ]
    shared_set = set(shared)
    rest = [p for p in active if p not in shared_set]
    return active, shared, rest


def _resolve_typical(
    model,
    optimizer: Geon,
    named_params,
    rule_set: RuleSet,
    frozen_step: int,
    frozen_eta: float,
    shared_patterns,
    mbs_batches,
):
    """Resolves ``rule_set`` into ``optimizer``'s own param_groups at
    ``(frozen_step, frozen_eta)`` (mutates lr/betas/nesterov/wd_raw in
    place -- callers that need a *different* rule set active afterward
    must re-resolve, see ``fork_and_explore``), then computes that rule
    set's typical (unscaled) per-param direction/size. Returns ``(shared,
    rest, directions, sizes)``."""
    updates_typical, sizings_typical = rule_set.apply_for_step(
        frozen_step, frozen_eta, named_params, optimizer
    )
    active, shared, rest = _select_active_shared_rest(
        named_params, updates_typical, shared_patterns
    )
    directions = {p: optimizer._direction(p, updates_typical[p]) for p in active}
    sizes = optimizer._resolve_sizes(
        active, sizings_typical, directions, model=model, batches=mbs_batches
    )
    return shared, rest, directions, sizes


def _make_log_metric(file_metrics: str):
    def log_metric(**metric):
        if dist.get_rank() == 0:
            with open(file_metrics, "a", encoding="utf-8") as f:
                print(json.dumps(metric), file=f)

    return log_metric


def _make_branch_dir(
    run_path: str,
    branch_name: str,
    *,
    role: str,
    branch_index: int,
    kl_matched: bool,
    override: dict,
    fork_step: int,
    branch_config: BranchConfig,
    b_train_config: TrainConfig,
    b_rule_set: RuleSet,
):
    """Creates ``<run_path>/branches/<branch_name>/``, writes its
    config.json, and returns a ready-to-use ``log_metric`` closure."""
    branch_run_path = os.path.join(run_path, "branches", branch_name)
    file_metrics = None
    if dist.get_rank() == 0:
        fork._makedirs_robust(branch_run_path)
        file_metrics = os.path.join(branch_run_path, FILENAME_METRICS)
        file_configs = os.path.join(branch_run_path, FILENAME_CONFIGS)
        config_payload = dict(
            branch_name=branch_name,
            branch_index=branch_index,
            kl_matched=kl_matched,
            role=role,
            fork_step=fork_step,
            override=override,
            fork_steps=branch_config.fork_steps,
            shared_patterns=branch_config.shared_patterns,
            train=asdict(b_train_config),
            rules=[asdict(r) for r in b_rule_set.rules],
        )
        with open(file_configs, "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2)
    return _make_log_metric(file_metrics)


def fork_and_explore(
    *,
    trunk_model,
    trunk_optimizer,
    trunk_rule_set: RuleSet,
    train_config: TrainConfig,
    trunk_cursor: DistributedDataCursor,
    val_inputs,
    val_targets,
    branch_config: BranchConfig,
    branch_specs: list[BranchSpec],
    fork_step: int,
    run_path: str,
    print0,
) -> None:
    fork_val_loss = _evaluate(trunk_model, val_inputs, val_targets, train_config)
    print0(
        f"  [branch @ step {fork_step}] pre-fork val_loss={fork_val_loss.item():.5f}",
        console=True,
    )

    def fork_branch(b_rule_set, b_train_config, label):
        """One independent clone of the trunk, ready to run under
        b_rule_set -- see fork.fork_branch for the actual construction
        (verified independence: tests/test_fork.py). The klmatch_schedule
        passed through here governs any rule-level "sizing: kl_match" entry
        this branch's own Geon resolves internally (via _resolve_typical ->
        _resolve_sizes -> _kl_matched_size) -- separate from
        fork_and_explore's own kl_matched=True branch-spec matching below,
        which never goes through Geon.step() (only its phases, directly)
        and so needs its own, separately-scheduled cache -- see
        fork_and_explore. For this schedule to actually take effect (not
        just always-recompute), fork_and_explore's own loop also advances
        b_optimizer._step_count once per iteration -- Geon.step() is the
        only thing that normally does this, and it's never called on a
        branch optimizer here, so without that, _step_count (and hence
        "which step is it") would stay frozen at 0 for this branch's entire
        life."""
        return fork.fork_branch(
            trunk_model,
            trunk_optimizer,
            b_train_config,
            b_rule_set,
            compile_models=branch_config.compile_models,
            label=label,
            print0=print0,
            klmatch_schedule=branch_config.klmatch_schedule,
        )

    branches = []
    for spec in branch_specs:
        tc_s, rule_set_s = apply_overrides(train_config, trunk_rule_set, spec.override)
        print0(
            f"  [branch @ step {fork_step}] {spec.name!r} "
            f"(index={spec.branch_index}"
            f"{', kl_matched' if spec.kl_matched else ''}) -> {spec.override}",
            console=True,
        )
        b = fork_branch(rule_set_s, tc_s, spec.name)
        b["named_params"] = list(b["model"].named_parameters())
        b["spec"] = spec
        b["train_config"] = tc_s
        branches.append(b)

    for b in branches:
        spec = b["spec"]
        if spec.branch_index == 0:
            role = f"{spec.name} (branch_index=0, reference; defines target_kl)"
        elif spec.kl_matched:
            role = (
                f"{spec.name} (branch_index={spec.branch_index}, "
                f"KL-matched to {branch_specs[0].name!r})"
            )
        else:
            role = f"{spec.name} (branch_index={spec.branch_index}, regular/unscaled)"
        b["log_metric"] = _make_branch_dir(
            run_path,
            spec.name,
            role=role,
            branch_index=spec.branch_index,
            kl_matched=spec.kl_matched,
            override=spec.override,
            fork_step=fork_step,
            branch_config=branch_config,
            b_train_config=b["train_config"],
            b_rule_set=b["rule_set"],
        )

    # KL computations (not the typical direction/size resolution, which
    # still sees every mbs chunk) are restricted to this many leading mbs
    # chunks of each iteration's shared batch when branch_config.kl_batch_size
    # is set -- validated divisible by mbs * seq_len in run_checkpoint_branch,
    # so this is always a whole number of chunks.
    num_kl_chunks = (
        branch_config.kl_batch_size // (train_config.mbs * train_config.seq_len)
        if branch_config.kl_batch_size is not None
        else None
    )

    def kl_slice(mbs_batches):
        return mbs_batches[:num_kl_chunks] if num_kl_chunks is not None else mbs_batches

    checkpoint_after_steps_set = set(branch_config.checkpoint_after_steps or [])

    # A second, independent Schedule instance (same spec as each branch's
    # own Geon.kl_match_cache_schedule, set in fork_branch above, but a
    # distinct object -- this one is driven by the fork loop's own 0-based
    # iteration index i below, not Geon._step_count, which never advances
    # here since branches only ever have their phases called directly, not
    # optimizer.step() itself) governing how often kl_matched=True branch
    # specs' own _kl_matched_scale search (not Geon-internal, see its call
    # site below) actually re-searches vs. reuses a cached (scale, kl),
    # keyed per branch name. Same "first call always computes" contract as
    # Geon's own cache: a schedule "skip" only takes effect once a cached
    # entry already exists for that branch.
    kl_matched_refit_schedule = Schedule(branch_config.klmatch_schedule)
    kl_matched_scale_cache: dict[str, tuple[float, float]] = {}

    # A fresh instance (branch_config.metric_schedule is itself already a
    # Schedule -- Schedule(...) accepts and unwraps one, see its docstring)
    # -- gates *diagnostic* work only (post-update val_loss eval, logging
    # itself), driven by this same loop's own i, same convention as above.
    metric_log_schedule = Schedule(branch_config.metric_schedule)

    def log_and_print(b, i, **metric):
        b["log_metric"](fork_step=fork_step, iter=i, **metric)
        kind = metric["kind"]
        parts = [
            f"kl={metric['kl']:.6g}",
            f"post_val_loss={metric['post_val_loss']:.5f}",
        ]
        if "scale" in metric:
            parts.insert(1, f"scale={metric['scale']:.4f}")
        print0(
            f"    [branch @ step {fork_step}] '{b['spec'].name}' iter={i} {kind} "
            f"train_loss={metric['train_loss']:.5f} " + " ".join(parts),
            console=True,
        )

    # Reads off a clone of the trunk's cursor (cheap: shares the already-
    # loaded shard tensor, no disk re-read) rather than trunk_cursor itself
    # -- even though nothing reads trunk_cursor again after this fork (it's
    # the last thing that happens to this job), advancing the live cursor
    # here would still make any real continuation elsewhere (e.g. a rerun,
    # or an equivalent single-branch job) train on non-matching data, for no
    # benefit. Each iteration's batch is fetched exactly once (outer loop)
    # and handed to every branch as-is (inner loop): all branches see
    # identical data, not just identically distributed data.
    def _checkpoint_branches_at(absolute_step: int, cursor_state: dict) -> None:
        for b in branches:
            fork.save_checkpoint(
                os.path.join(run_path, "branches", b["spec"].name, CHECKPOINTS_DIRNAME),
                absolute_step,
                b["model"],
                b["optimizer"],
                cursor_state,
            )
        print0(
            f"  [branch @ step {fork_step}] checkpointed branches at "
            f"step={absolute_step}",
            console=True,
        )

    fork_cursor = trunk_cursor.clone()
    # duration 0: the branches' own state right at the fork point, before
    # any of their own training -- a verbatim (if redundant) copy of
    # source_path's own checkpoint at fork_step, since fork_branch's whole
    # guarantee is that a branch starts as an independent copy of that
    # state (see tests/test_fork.py). Written anyway, on request, purely so
    # a downstream job (e.g. run_branch_continue.py) can address every
    # duration in checkpoint_after_steps uniformly, without special-casing
    # 0 to mean "read from source_path instead".
    if 0 in checkpoint_after_steps_set:
        _checkpoint_branches_at(fork_step, fork_cursor.state_dict())
    for i in range(branch_config.fork_steps):
        x, y = fork_cursor.next_batch()
        # Not frozen at fork_step: iteration i's real corresponding step is
        # fork_step + i (fork_step itself is "steps completed so far" at the
        # fork point, same convention run_checkpoint_branch's own trunk loop
        # uses for its own eta_of calls), so eta/rule/warmup selection here
        # track exactly what continued real training would have used.
        step_i = fork_step + i
        # Gates *diagnostic* work only (post-update val_loss eval, logging)
        # -- every branch's real update below still applies every iteration
        # regardless, so skipped iterations don't change training, only
        # what gets measured/logged about them.
        log_this_iter = metric_log_schedule.should_do(i)

        target_kl = None
        for b in branches:
            spec = b["spec"]
            tc_s = b["train_config"]
            eta_i = tc_s.eta_of(step_i)

            train_loss_s, mbs_batches_s = _forward_backward(b["model"], tc_s, x, y)
            kl_batches_s = kl_slice(mbs_batches_s)
            b["optimizer"]._refresh_state()

            shared_s, rest_s, directions_s, sizes_s = _resolve_typical(
                b["model"],
                b["optimizer"],
                b["named_params"],
                b["rule_set"],
                step_i,
                eta_i,
                branch_config.shared_patterns,
                mbs_batches_s,
            )
            # `shared` is always applied first, uniformly across every
            # branch regardless of role -- unlike the pre-branch_specs
            # version of this script, which applied it first for the
            # regular/kl_matched branches but *after* probing target_kl for
            # the reference branch (branch_1), an inconsistency left over
            # from when branch_1 was special-cased code, not data. Every
            # KL probe below still only perturbs `rest` (never `shared`),
            # so it isolates rest's own marginal KL contribution on top of
            # an identically-prepared model state across every branch --
            # a fair, apples-to-apples comparison, and the same state
            # `rest`'s real update lands on a moment later regardless.
            _apply_update(shared_s, b["optimizer"], directions_s, sizes_s, scale=1.0)

            if spec.branch_index == 0:
                # Reference: applies its own typical update to `rest` for
                # real, unscaled. Its own KL on `rest` alone is `target_kl`
                # for every later kl_matched branch this iteration, needed
                # regardless of whether this iteration is logged.
                target_kl = (
                    _probe_kl(
                        b["model"],
                        kl_batches_s,
                        rest_s,
                        lambda: _apply_update(
                            rest_s, b["optimizer"], directions_s, sizes_s, scale=1.0
                        ),
                    )
                    if rest_s
                    else 0.0
                )
                applied_scale_s, applied_kl_s, logged_target_kl = (
                    1.0,
                    target_kl,
                    target_kl,
                )
                _apply_update(rest_s, b["optimizer"], directions_s, sizes_s, scale=1.0)
            elif spec.kl_matched:
                # Cached per branch name (see kl_matched_refit_schedule
                # above); the very first call for a branch always searches,
                # regardless of the schedule, same contract as Geon's own
                # kl_match cache.
                if (
                    spec.name in kl_matched_scale_cache
                    and not kl_matched_refit_schedule.should_do(i)
                ):
                    applied_scale_s, applied_kl_s = kl_matched_scale_cache[spec.name]
                else:
                    applied_scale_s, applied_kl_s = _kl_matched_scale(
                        b["model"],
                        kl_batches_s,
                        rest_s,
                        b["optimizer"],
                        directions_s,
                        sizes_s,
                        target_kl,
                        kl_search_iters=branch_config.kl_search_iters,
                        kl_scale_init=branch_config.kl_scale_init,
                        kl_search_expand_factor=branch_config.kl_search_expand_factor,
                    )
                    kl_matched_scale_cache[spec.name] = (applied_scale_s, applied_kl_s)
                logged_target_kl = target_kl
                _apply_update(
                    rest_s, b["optimizer"], directions_s, sizes_s, scale=applied_scale_s
                )
            else:
                applied_scale_s = 1.0
                if log_this_iter:
                    applied_kl_s = (
                        _probe_kl(
                            b["model"],
                            kl_batches_s,
                            rest_s,
                            lambda: _apply_update(
                                rest_s, b["optimizer"], directions_s, sizes_s, scale=1.0
                            ),
                        )
                        if rest_s
                        else 0.0
                    )
                    logged_target_kl = applied_kl_s
                _apply_update(rest_s, b["optimizer"], directions_s, sizes_s, scale=1.0)

            b["model"].zero_grad(set_to_none=True)
            # fork_and_explore never calls optimizer.step() on a branch (see
            # fork_branch's own docstring/the module docstring -- only its
            # phases are reached into directly), so _step_count would
            # otherwise stay frozen at 0 for this branch's entire life,
            # making its own kl_match_cache_schedule (see fork_branch) see
            # a constant "step" forever instead of one that actually
            # advances per fork iteration. Advanced here, once per branch
            # per iteration, mirroring what Geon.step() itself would have
            # done -- nothing else reads/depends on it (param_sync_every's
            # own check lives inside step(), never reached here either).
            b["optimizer"]._step_count += 1
            if log_this_iter:
                post_val_s = _evaluate(b["model"], val_inputs, val_targets, tc_s)
                log_and_print(
                    b,
                    i,
                    kind="applied",
                    train_loss=train_loss_s.item(),
                    scale=applied_scale_s,
                    kl=applied_kl_s,
                    target_kl=logged_target_kl,
                    init_val_loss=fork_val_loss.item(),
                    post_val_loss=post_val_s.item(),
                )

        # Every branch's (model, optimizer) checkpointed together, plus one
        # shared cursor snapshot (they all read off the same fork_cursor, so
        # it's identical for all of them at this point) -- for a later job
        # to load a few of these back and continue them under a new, shared
        # override (see run_branch_continue.py). checkpoint_after_steps
        # holds *durations* (i+1 iterations done, or 0 for "right at the
        # fork point" -- see above), not 0-based iteration indices -- i==0
        # is the 1st iteration.
        if (i + 1) in checkpoint_after_steps_set:
            _checkpoint_branches_at(step_i + 1, fork_cursor.state_dict())

    for b in branches:
        del b["model"], b["optimizer"]


########################################
#            Training loop             #
########################################


def run_checkpoint_branch(
    name: str,
    source_path: str,
    checkpoint_step: int,
    branch_specs: list[BranchSpec],
    branch_config: BranchConfig,
    run_path: str,
) -> None:
    """One job. Assumes the process group is already initialized (see
    ``main``)."""
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))

    file_logs = os.path.join(run_path, FILENAME_LOGS)
    file_metrics = os.path.join(run_path, FILENAME_METRICS)
    file_configs = os.path.join(run_path, FILENAME_CONFIGS)
    file_done = os.path.join(run_path, FILENAME_DONE)

    # Every rank checks this identically -- see run_optim_rules.py's own note
    # on why this can't be rank-0-gated (every other rank would hang on its
    # next collective call).
    if os.path.exists(file_done):
        if dist.get_rank() == 0:
            print(f"{file_done} exists, job already completed, skipping ...")
        dist.barrier()
        return

    # Loaded and used as-is -- same train_steps, same rules, same schedule as
    # the original run; this is exactly run_optim_rules.py's own resume path,
    # just stopping at end_step instead of train_steps. Nothing here is
    # extended or modified to accommodate a longer horizon: end_step must
    # already fit inside the schedule/rule coverage the checkpoint's own
    # config defines (checked below), same as it would for any ordinary
    # resume.
    train_config, rule_set = load_checkpoint_config(resolve_repo_path(source_path))
    # Overridden once, here, rather than at each forward/backward/eval call
    # site -- every downstream use (the trunk's own training loop below, and
    # everything fork_and_explore does with the same train_config) picks it
    # up automatically. Purely a microbatch-chunking knob (see BranchConfig.mbs).
    if branch_config.mbs is not None:
        train_config = replace(train_config, mbs=branch_config.mbs)
    if branch_config.batch_size is not None:
        train_config = replace(train_config, batch_size=branch_config.batch_size)
    if branch_config.val_size is not None:
        train_config = replace(train_config, val_size=branch_config.val_size)
    if branch_config.kl_batch_size is not None:
        tokens_per_mbs_chunk = train_config.mbs * train_config.seq_len
        if branch_config.kl_batch_size % tokens_per_mbs_chunk != 0:
            raise ValueError(
                f"branch_config.kl_batch_size ({branch_config.kl_batch_size}) must "
                f"be divisible by this job's mbs * seq_len "
                f"({train_config.mbs} * {train_config.seq_len} = "
                f"{tokens_per_mbs_chunk}) -- KL computations are restricted to a "
                f"whole number of the same mbs chunks the typical update uses"
            )
    _validate_branch_specs(branch_specs, rule_set)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    torch.cuda.manual_seed_all(train_config.seed)

    end_step = checkpoint_step + branch_config.prefork_steps
    if end_step > train_config.train_steps:
        raise ValueError(
            f"checkpoint_step ({checkpoint_step}) + prefork_steps "
            f"({branch_config.prefork_steps}) = {end_step} exceeds "
            f"{source_path!r}'s own train_steps ({train_config.train_steps}) -- "
            f"the checkpoint's config only defines the schedule/rules up to "
            f"train_steps, so a job can't continue past it"
        )
    # fork_and_explore's own eta/rule/warmup selection isn't frozen at
    # end_step -- it tracks end_step + i for iteration i (the real
    # corresponding step), so the fork window needs schedule coverage just
    # as much as the trunk continuation does.
    if end_step + branch_config.fork_steps > train_config.train_steps:
        raise ValueError(
            f"end_step ({end_step}) + branch_config.fork_steps "
            f"({branch_config.fork_steps}) = {end_step + branch_config.fork_steps} "
            f"exceeds {source_path!r}'s own train_steps ({train_config.train_steps}) "
            f"-- fork_and_explore resolves each iteration's schedule at its real "
            f"corresponding step (not frozen at end_step), so the fork window needs "
            f"to fit inside train_steps too"
        )
    if branch_config.checkpoint_after_steps:
        for d in branch_config.checkpoint_after_steps:
            if not 0 <= d <= branch_config.fork_steps:
                raise ValueError(
                    f"branch_config.checkpoint_after_steps entry {d} must be in "
                    f"[0, fork_steps={branch_config.fork_steps}]"
                )

    if dist.get_rank() == 0:
        fork._makedirs_robust(run_path)
        print("logs:    ", file_logs)
        print("metrics: ", file_metrics)
        print("configs: ", file_configs)
        config_payload = dict(
            name=name,
            source_path=source_path,
            checkpoint_step=checkpoint_step,
            branch_specs=[
                dict(
                    name=s.name,
                    branch_index=s.branch_index,
                    override=s.override,
                    kl_matched=s.kl_matched,
                )
                for s in branch_specs
            ],
            end_step=end_step,
            train=asdict(train_config),
            rules=[asdict(r) for r in rule_set.rules],
            branch_config=dict(
                prefork_steps=branch_config.prefork_steps,
                fork_steps=branch_config.fork_steps,
                shared_patterns=branch_config.shared_patterns,
                checkpoint_after_steps=branch_config.checkpoint_after_steps,
                # .cache_schedule unwraps back to the raw spec (a plain
                # dict or None) -- branch_config's own fields are Schedule
                # objects, not JSON-serializable as-is.
                metric_schedule=branch_config.metric_schedule.cache_schedule,
                klmatch_schedule=branch_config.klmatch_schedule.cache_schedule,
                compile_models=branch_config.compile_models,
            ),
        )
        with open(file_configs, "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2)
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
    print0(
        f"Branching from: {source_path} step={checkpoint_step} "
        f"branch_specs={[s.name for s in branch_specs]!r}"
    )
    print0(f"Config: {train_config}")
    print0(f"Rules: {rule_set}")
    print0(f"end_step: {end_step}")
    print0("=" * 100)

    trunk_cursor = DistributedDataCursor(
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

    built = fork.build_model_and_geon(train_config, rule_set)
    model, optimizer, rule_set = built["model"], built["optimizer"], built["rule_set"]
    if branch_config.compile_models:
        fork.compile_and_warmup(model, train_config, print0, "trunk")
    named_params = list(model.named_parameters())

    checkpoint_path = os.path.join(
        source_path, CHECKPOINTS_DIRNAME, f"step_{checkpoint_step}.pt"
    )
    ckpt = fork.load_checkpoint(
        resolve_repo_path(checkpoint_path), model, optimizer, device=device
    )
    trunk_cursor.load_state_dict(ckpt["train_loader"])
    assert int(ckpt["step"]) == checkpoint_step, (ckpt["step"], checkpoint_step)
    print0(
        f"loaded checkpoint {checkpoint_path} at step {checkpoint_step}", console=True
    )

    training_time = 0.0
    rolling_loss = 0.0
    rolling_loss_step = 0

    dist.barrier()
    t0 = time.perf_counter()

    # ---------------- 1) continue training, plain, up to end_step ----------
    for step in range(checkpoint_step, end_step):
        inputs, targets = trunk_cursor.next_batch()
        train_loss, mbs_batches = _forward_backward(
            model, train_config, inputs, targets
        )

        eta = train_config.eta_of(step)
        updates, sizings = rule_set.apply_for_step(step, eta, named_params, optimizer)
        optimizer.step(updates, sizings, model=model, batches=mbs_batches)
        model.zero_grad(set_to_none=True)

        rolling_loss_step += 1
        rolling_loss = (
            ROLLING_LOSS_BETA * rolling_loss + (1 - ROLLING_LOSS_BETA) * train_loss
        )
        unbiased_rolling_loss = rolling_loss / (
            1 - ROLLING_LOSS_BETA**rolling_loss_step
        )

        approx_training_time = training_time + (time.perf_counter() - t0)
        # Every step, on console (live progress); also to log.txt every
        # report_steps (plus the last one), with the rolling loss, so the
        # persisted log isn't one line per step but still shows progress.
        is_report_step = (step + 1) % train_config.report_steps == 0 or (
            step + 1 == end_step
        )
        print0(
            f"step:{step+1}/{end_step} "
            + (f"rolling_loss:{unbiased_rolling_loss:.5f} " if is_report_step else "")
            + f"train_time:{approx_training_time:.3f}s",
            console=True,
            log=is_report_step,
        )
        log_metric(
            kind="train",
            step=step + 1,
            train_loss=train_loss.item(),
            rolling_loss=unbiased_rolling_loss.item(),
            train_time=approx_training_time,
        )

    # ---------------- 2) fork into branches, once, at end_step -------------
    dist.barrier()
    training_time += time.perf_counter() - t0
    print0(f"forking at step {end_step} ...", console=True)
    fork_and_explore(
        trunk_model=model,
        trunk_optimizer=optimizer,
        trunk_rule_set=rule_set,
        train_config=train_config,
        trunk_cursor=trunk_cursor,
        val_inputs=val_inputs,
        val_targets=val_targets,
        branch_config=branch_config,
        branch_specs=branch_specs,
        fork_step=end_step,
        run_path=run_path,
        print0=print0,
    )
    dist.barrier()

    if dist.get_rank() == 0:
        with open(file_done, "w", encoding="utf-8") as f:
            f.write(f"{end_step}\n")
    dist.barrier()


########################################
#                  CLI                 #
########################################


def _job_run_path(base_run_path: str, name: str, step: int) -> str:
    return os.path.join(base_run_path, name, f"step_{step:06d}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run_path",
        default="logs",
        help="Directory for job logs.",
    )
    p.add_argument(
        "--config_path",
        default=None,
        help="Path to a branch config YAML ('runs' + 'branch_config'). "
        "Defaults to <run_path>/config.yaml.",
    )
    p.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Optional, for running a sweep across multiple parallel nodes: "
        "total number of shards this job list is split across. Requires "
        "--shard_index. Jobs are sorted by their source run's model size "
        "(ascending) and dealt round-robin (i %% num_shards == shard_index).",
    )
    p.add_argument(
        "--shard_index",
        type=int,
        default=None,
        help="Optional: this invocation's shard index, in [0, num_shards). "
        "Requires --num_shards.",
    )
    args = p.parse_args()
    print("run_branch_compare ", args)

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
    run_specs, branch_config = load_branch_yaml(config_path)

    jobs = [
        (
            run_spec.path,
            step,
            run_spec.branch_specs,
            run_spec.name,
            replace(
                branch_config,
                fork_steps=(
                    run_spec.fork_steps
                    if run_spec.fork_steps is not None
                    else branch_config.fork_steps
                ),
                prefork_steps=(
                    run_spec.prefork_steps
                    if run_spec.prefork_steps is not None
                    else branch_config.prefork_steps
                ),
                checkpoint_after_steps=(
                    run_spec.checkpoint_after_steps
                    if run_spec.checkpoint_after_steps is not None
                    else branch_config.checkpoint_after_steps
                ),
                metric_schedule=(
                    run_spec.metric_schedule
                    if run_spec.metric_schedule is not None
                    else branch_config.metric_schedule
                ),
                klmatch_schedule=(
                    run_spec.klmatch_schedule
                    if run_spec.klmatch_schedule is not None
                    else branch_config.klmatch_schedule
                ),
            ),
        )
        for run_spec in run_specs
        for step in run_spec.steps
    ]

    if args.num_shards is not None:
        num_params_by_path = {}
        for source_path, _step, _branch_specs, _name, _bc in jobs:
            if source_path not in num_params_by_path:
                tc, _rules = load_checkpoint_config(resolve_repo_path(source_path))
                num_params_by_path[source_path] = fork.train_config_num_params(tc)

        def _job_cost(job) -> int:
            """Rough FLOPs proxy for load-balancing shards: num_params *
            steps_to_do, where steps_to_do is this job's own real update
            count -- prefork_steps (trunk) plus fork_steps once per branch
            (len(branch_specs))."""
            source_path, _step, branch_specs, _name, bc = job
            steps_to_do = bc.prefork_steps + bc.fork_steps * len(branch_specs)
            return num_params_by_path[source_path] * steps_to_do

        jobs.sort(key=_job_cost)
        jobs = [
            j for i, j in enumerate(jobs) if i % args.num_shards == args.shard_index
        ]
        print(f"shard {args.shard_index}/{args.num_shards}: {len(jobs)} job(s)")

    # Initialized once for the whole job list, not once per job -- see
    # run_optim_rules.py's identical note (repeated init/destroy of a process
    # group within one process is unsupported by PyTorch).
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    assert 8 % dist.get_world_size() == 0
    dist.barrier()
    try:
        for (
            source_path,
            step,
            branch_specs,
            name,
            job_branch_config,
        ) in jobs:
            job_run_path = _job_run_path(args.run_path, name, step)
            run_checkpoint_branch(
                name,
                source_path,
                step,
                branch_specs,
                job_branch_config,
                run_path=job_run_path,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    # torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) ...
    main()
