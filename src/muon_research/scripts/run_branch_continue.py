"""Continue a group of branch checkpoints -- from a run_branch_compare.py
job, or from a *previous run of this same script* (see point 1 below) --
under one new, shared override -- a "re-fork": load a few branches' saved
states (see ``BranchConfig.checkpoint_after_steps`` there) from the *same*
fork point of the *same* job, verify they really do share that data-cursor
position (they should, since every branch in a job reads off one shared
cloned cursor), then continue each of them -- same weights (optimizer state
too, unless ``reset_optimizer_state``), but the *original* job's rules with
one new override applied on top, replacing whichever override that branch
had been using -- for a fixed number of further steps, real ``Geon.step()``
training throughout, logging train/eval loss per branch.

For each ``runs`` entry, ``checkpoint_step`` may be a single int or a list
of them -- a list expands into one ``ContinueSpec`` ("job") per step, all
sharing that entry's other fields (``path``/``branch_names``/``override``/
``continue_steps``), same convention run_branch_compare.py's own
``steps`` list uses. Each job's own output dir is
``<run_path>/<name>/step_<checkpoint_step>/`` -- so ``(name, checkpoint_step)``
must be unique, not ``name`` alone.

For each job:

1. **Load.** ``path`` is a job dir produced by either script: a
   run_branch_compare.py job (``<run_path>/<name>/step_<checkpoint_step>/``,
   what that script's own ``_job_run_path`` produces) or a *previous*
   run_branch_continue.py job (this script's own ``_job_run_path`` --
   e.g. re-forking a checkpoint that was itself produced by an earlier
   re-fork, to chain several rounds of "diverge under different rules,
   then reunite" together). Either way, ``path/config.json`` gives the
   job's *original* (pre-override) ``train_config``/``rules`` --
   ``apply_overrides`` applies this run's ``override`` to *that* baseline
   (not to a branch's own override), so every continued branch starts its
   new schedule from the same common rules, exactly the same way
   ``override_1``/``override_2`` did at the first fork.

   For each of ``branch_names`` (e.g. ``["branch_1", "branch_2"]``), reads
   ``<path>/branches/<branch_name>/checkpoints/step_<checkpoint_step>.pt``
   (run_branch_compare.py's own layout -- what
   ``branch_config.checkpoint_after_steps`` writes) when
   ``source_nested_branches`` is true (the default), else
   ``<path>/<branch_name>/checkpoints/step_<checkpoint_step>.pt`` (this
   script's own layout, lacking the extra ``branches/`` level -- see
   ``run_branch_continue``'s own output path below) -- set
   ``source_nested_branches: false`` in a ``runs`` entry when its ``path``
   is itself a previous run_branch_continue.py job. Both layouts use the
   same payload shape (model/optimizer/train_loader/step, same as
   run_optim_rules.py's own checkpoints).

2. **Verify.** Every loaded checkpoint's ``train_loader`` (data-cursor
   state) must be identical -- asserted, not just assumed, since a
   mismatched fork point or branch name here would silently compare
   branches that saw different data. One shared ``DistributedDataCursor``
   is then resumed from that single, verified position, so every branch
   continues reading identical batches, same convention
   ``fork_and_explore`` itself uses.

3. **Continue.** Plain per-branch training (own model/optimizer, same
   shared batch each step, no forking/KL-matching) from ``checkpoint_step``
   through ``checkpoint_step + continue_steps``, logging ``kind="train"``
   every step (train_loss/rolling_loss come free from the forward/backward
   training already does) and ``kind="eval"`` whenever ``metric_schedule``
   says so (default: every step, same as before this field existed --
   unlike fork_and_explore's ``metric_schedule``, there's no KL probe
   riding along to amortize here, so dense-by-default is cheap and val_loss
   is the whole point of comparing branches; pass a sparser
   ``metric_schedule`` to cut the extra eval forward pass on long
   ``continue_steps`` runs) to
   ``<run_path>/<name>/step_<checkpoint_step>/<branch_name>/metrics.jsonl``.

   ``warmup_steps`` (default 0 -- none) linearly warms ``eta`` up from
   (near) 0 to the schedule's own ``train_config.eta_of(step)`` over the
   first ``warmup_steps`` steps of the continuation -- same shape as
   ``Rule.warmup_steps``/``Rule.warmup_factor``, just relative to this job's
   own ``checkpoint_step`` instead of a rule's ``start``.

   ``reset_optimizer_state`` (default ``False``) clears every branch's
   optimizer state (``Geon``'s per-param momentum EMAs + step count) right
   after its checkpoint loads, before the first continuation step -- the
   model still starts from that branch's own weights, but momentum starts
   fresh, isolating how much of a branch's continued trajectory comes from
   its momentum state vs. just its weights.

   ``compile_models`` (default ``True``) runs ``model.compile(dynamic=False,
   fullgraph=True)`` on every branch -- only speeds up the real
   ``_forward_backward`` update path, not ``_evaluate``. Compilation only
   actually happens (and costs time) on first invocation; dynamo's cache is
   keyed on the traced graph (shapes/dtypes), not the model instance or its
   weight values, so only the very first branch built in this process pays
   the real cost -- every later one reuses it almost for free. Each
   compile's wall time is measured (a throwaway warmup forward+backward,
   discarded right after) and printed.

Example config:

    runs:
      - name: adamw_muon
        path: logs/adamw/step_000100
        checkpoint_step: [105, 110, 120]  # or a single int
        branch_names: [branch_1, branch_2]
        override: {rules.two_d.update: muon, rules.two_d.lr: 0.04, ...}
        continue_steps: 200
        warmup_steps: 20  # optional; default 0 (no warmup)
        source_nested_branches: true  # optional; default true (see above)
        reset_optimizer_state: true  # optional; default false
        compile_models: true  # optional; default true
        metric_schedule: {_type: ap, k: 10}  # optional; default (omit, or null) evals every step

# metric_schedule takes a muon_research/optim/geon.py Schedule spec -- None
# (eval every step), {_type: ap, k: <int>} (every k steps), or
# {_type: schedule, schedule: [[start, end, k], ...]} (dense early, sparse
# later -- see Schedule's own docstring). Only gates the kind="eval" val_loss
# computation; kind="train" is always logged every step.
"""

# pylint: disable=all

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.distributed as dist
import yaml

from muon_research.data import DistributedDataCursor, distributed_data_generator
from muon_research.optim.geon import Schedule
from muon_research.rules import apply_overrides
from muon_research.paths import resolve_repo_path
from muon_research import fork
from muon_research.scripts.run_branch_compare import (
    _forward_backward,
    _evaluate,
    _make_log_metric,
    _parse_schedule_spec,
    load_compare_job_config,
)
from muon_research.constants import (
    FILENAME_CONFIGS,
    FILENAME_DONE,
    FILENAME_LOGS,
    FILENAME_METRICS,
    ROLLING_LOSS_BETA,
)
from muon_research.scripts.run_optim_rules import CHECKPOINTS_DIRNAME

########################################
#                Config                #
########################################


@dataclass
class ContinueSpec:
    name: str
    # A run_branch_compare.py job dir (this checkpoint's source). Kept
    # exactly as given in the YAML (relative or absolute) -- resolved
    # against the repo root on demand, at each actual file-I/O call site,
    # so it stays portable across checkouts and round-trips unchanged into
    # config.json/logs.
    path: str
    checkpoint_step: int
    branch_names: list[str]
    override: dict
    continue_steps: int
    # Linear eta warmup (same shape as Rule.warmup_steps/Rule.warmup_factor,
    # just relative to this job's own checkpoint_step instead of a rule's
    # start): 0 (default) means no warmup. Otherwise, for the first
    # warmup_steps steps of the continuation, eta is scaled by
    # min(1.0, (steps_since_start + 1) / warmup_steps) -- 0 (exclusive) up
    # to the schedule's own train_config.eta_of(step) by the end of
    # warmup_steps, then unscaled from there on.
    warmup_steps: int = 0
    # True (default): `path` is a run_branch_compare.py job dir, so each
    # branch's checkpoint lives at `path/branches/<branch_name>/checkpoints/
    # step_<n>.pt` (that script's own layout). False: `path` is a *previous*
    # run_branch_continue.py job dir instead, whose own branches live one
    # level shallower, directly at `path/<branch_name>/checkpoints/
    # step_<n>.pt` (no `branches/` in between) -- see this module's own
    # docstring for chaining several rounds of re-forking together.
    source_nested_branches: bool = True
    # When True, every branch's optimizer state (m/v momentum EMAs, step
    # count -- see Geon._refresh_state) is cleared right after its
    # checkpoint loads, before the first continuation step -- weights still
    # carry that branch's full history, but momentum starts fresh. Lets a
    # comparison isolate how much of a branch's continued trajectory is
    # driven by its momentum state vs. just its weights.
    reset_optimizer_state: bool = False
    # model.compile(dynamic=False, fullgraph=True) on every branch. Only
    # speeds up forward()/__call__ (the real _forward_backward update
    # path) -- _evaluate stays eager either way, since it never calls
    # model(...)/model.logits() through anything that would need it here.
    # Compilation only actually happens (and costs time) on first
    # invocation, not at the .compile() call site; dynamo's cache is keyed
    # on the traced graph (shapes/dtypes), not the model instance or its
    # weight values, so only the very first branch built in this process
    # pays the real cost -- every later one (this job or the next one in
    # the same --num_shards process) reuses it almost for free. Each
    # compile's wall time is measured (a throwaway warmup forward+backward,
    # discarded via zero_grad right after) and printed.
    compile_models: bool = True
    # Controls which continuation iterations bother computing/logging
    # kind="eval" (the val_loss _evaluate() call) -- default (None) evals
    # every step, same as this script's behavior before this field existed.
    # kind="train" is unaffected: always logged every step, since
    # train_loss/rolling_loss come free from the forward/backward training
    # already does, unlike the extra forward pass eval needs. Iteration i
    # is 0-based, local to this job's own continuation window
    # (i = step - checkpoint_step), same convention run_branch_compare.py's
    # own metric_schedule uses. See Schedule's own docstring for the
    # accepted shapes.
    metric_schedule: Schedule = Schedule(None)


def load_continue_yaml(path: str) -> list[ContinueSpec]:
    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    raw_runs = payload.get("runs") or []
    if not raw_runs:
        raise ValueError(f"config {path!r} has no 'runs' entries")

    specs = []
    # (name, checkpoint_step) is what actually determines each job's output
    # directory (see _job_run_path) -- so that's the uniqueness constraint,
    # not name alone, same convention run_branch_compare.py's own
    # (name, step) check uses.
    job_dirs_seen = set()
    for entry in raw_runs:
        name = str(entry["name"])

        branch_names = [str(b) for b in entry["branch_names"]]
        if not branch_names:
            raise ValueError(f"run {name!r} has an empty 'branch_names' list")
        if len(set(branch_names)) != len(branch_names):
            raise ValueError(f"run {name!r}: 'branch_names' has duplicates")

        # A single int, or a list of them -- one spec per checkpoint_step,
        # all sharing this entry's other fields.
        raw_steps = entry["checkpoint_step"]
        checkpoint_steps = (
            [int(s) for s in raw_steps]
            if isinstance(raw_steps, list)
            else [int(raw_steps)]
        )
        if not checkpoint_steps:
            raise ValueError(f"run {name!r} has an empty 'checkpoint_step' list")
        for checkpoint_step in checkpoint_steps:
            if checkpoint_step < 0:
                raise ValueError(f"run {name!r}: checkpoint_step must not be negative")
            job_dir = (name, checkpoint_step)
            if job_dir in job_dirs_seen:
                raise ValueError(
                    f"duplicate (name, checkpoint_step) = {job_dir!r} -- these "
                    f"determine each job's output directory (<run_path>/{name}/"
                    f"step_{checkpoint_step:06d}), which must be unique"
                )
            job_dirs_seen.add(job_dir)

        continue_steps = int(entry["continue_steps"])
        if continue_steps <= 0:
            raise ValueError(f"run {name!r}: continue_steps must be positive")

        warmup_steps = int(entry.get("warmup_steps", 0))
        if warmup_steps < 0:
            raise ValueError(f"run {name!r}: warmup_steps must not be negative")

        metric_schedule = _parse_schedule_spec(
            entry.get("metric_schedule"),
            where=f"run {name!r}",
            field_name="metric_schedule",
        )

        for checkpoint_step in checkpoint_steps:
            specs.append(
                ContinueSpec(
                    name=name,
                    path=str(entry["path"]),
                    checkpoint_step=checkpoint_step,
                    branch_names=branch_names,
                    override=dict(entry.get("override") or {}),
                    continue_steps=continue_steps,
                    warmup_steps=warmup_steps,
                    source_nested_branches=bool(
                        entry.get("source_nested_branches", True)
                    ),
                    reset_optimizer_state=bool(
                        entry.get("reset_optimizer_state", False)
                    ),
                    compile_models=bool(entry.get("compile_models", True)),
                    metric_schedule=metric_schedule,
                )
            )
    return specs


########################################
#            Training loop             #
########################################


def _job_run_path(base_run_path: str, name: str, checkpoint_step: int) -> str:
    return os.path.join(base_run_path, name, f"step_{checkpoint_step:06d}")


def run_branch_continue(spec: ContinueSpec, run_path: str) -> None:
    """One job. Assumes the process group is already initialized (see
    ``main``)."""
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))

    file_logs = os.path.join(run_path, FILENAME_LOGS)
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

    # The job's *original* (pre-override) rules -- spec.override replaces
    # whichever override a branch itself had been using, so every continued
    # branch starts its new schedule from this same common baseline, not
    # from each other's possibly-different prior override.
    job_train_config, job_rule_set = load_compare_job_config(
        resolve_repo_path(spec.path)
    )
    train_config, rule_set = apply_overrides(
        job_train_config, job_rule_set, spec.override
    )

    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    torch.cuda.manual_seed_all(train_config.seed)

    end_step = spec.checkpoint_step + spec.continue_steps
    if end_step > train_config.train_steps:
        raise ValueError(
            f"checkpoint_step ({spec.checkpoint_step}) + continue_steps "
            f"({spec.continue_steps}) = {end_step} exceeds "
            f"{spec.path!r}'s own train_steps "
            f"({train_config.train_steps}) -- the job's config only defines "
            f"the schedule/rules up to train_steps, so a job can't continue "
            f"past it"
        )

    if dist.get_rank() == 0:
        fork._makedirs_robust(run_path)
        print("logs:    ", file_logs)
        print("configs: ", file_configs)
        config_payload = dict(
            name=spec.name,
            path=spec.path,
            checkpoint_step=spec.checkpoint_step,
            branch_names=spec.branch_names,
            override=spec.override,
            continue_steps=spec.continue_steps,
            warmup_steps=spec.warmup_steps,
            source_nested_branches=spec.source_nested_branches,
            reset_optimizer_state=spec.reset_optimizer_state,
            compile_models=spec.compile_models,
            # .cache_schedule unwraps back to the raw spec (a plain dict or
            # None) -- spec.metric_schedule is a Schedule object, not
            # JSON-serializable as-is. Same pattern run_branch_compare.py's
            # own config_payload uses.
            metric_schedule=spec.metric_schedule.cache_schedule,
            end_step=end_step,
            train=asdict(train_config),
            rules=[asdict(r) for r in rule_set.rules],
        )
        with open(file_configs, "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2)
    dist.barrier()

    def print0(s, console=False, log=True):
        if dist.get_rank() == 0:
            if console:
                print(s)
            if log:
                with open(file_logs, "a") as f:
                    print(s, file=f)

    print0("=" * 100)
    print0(
        f"Continuing: {spec.path} step={spec.checkpoint_step} "
        f"branches={spec.branch_names} override={spec.override}"
    )
    print0(f"Config: {train_config}")
    print0(f"Rules: {rule_set}")
    print0(f"end_step: {end_step}")
    print0("=" * 100)

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

    # Every rank independently reads the same checkpoint files off the
    # shared filesystem -- model/optimizer/train_loader state is
    # bit-identical across ranks by construction (see
    # run_optim_rules.save_checkpoint), so no broadcast is needed, same
    # convention run_checkpoint_branch's own trunk-loading uses.
    branches: dict[str, dict] = {}
    cursor_states = []
    for branch_name in spec.branch_names:
        # spec.source_nested_branches picks which of the two layouts
        # `spec.path` uses -- see ContinueSpec's own docstring.
        branch_dir = (
            os.path.join(spec.path, "branches", branch_name)
            if spec.source_nested_branches
            else os.path.join(spec.path, branch_name)
        )
        checkpoint_path = os.path.join(
            branch_dir, CHECKPOINTS_DIRNAME, f"step_{spec.checkpoint_step}.pt"
        )
        # Each branch gets its own RuleSet instance (apply_overrides with an
        # empty override is a cheap way to get an unresolved copy) -- they
        # all start from the same rule_set here, but build_model_and_geon
        # resolves whichever instance it's given in place, so sharing one
        # object across every branch's own build would have each new
        # resolve() silently overwrite what an earlier branch's (same
        # object, aliased) rule_set already resolved to.
        _train_config_b, rule_set_b = apply_overrides(train_config, rule_set, {})
        built = fork.build_model_and_geon(train_config, rule_set_b)
        model, optimizer, rule_set_b = (
            built["model"],
            built["optimizer"],
            built["rule_set"],
        )
        # In-place (nn.Module.compile(), not the torch.compile(model)
        # wrapper) -- same call run_optim_rules.py itself uses, and safe to
        # do before load_checkpoint below since it doesn't rename params or
        # wrap the module (unlike torch.compile, which can prefix state_dict
        # keys with "_orig_mod."). This script never copy.deepcopy's a model
        # after construction (each branch is built once, loaded once, then
        # trained in a stable loop), so compiling here has no
        # cloning-a-compiled-module hazard to worry about.
        if spec.compile_models:
            fork.compile_and_warmup(model, train_config, print0, branch_name)
        named_params = list(model.named_parameters())
        ckpt = fork.load_checkpoint(
            resolve_repo_path(checkpoint_path), model, optimizer, device=device
        )
        assert int(ckpt["step"]) == spec.checkpoint_step, (
            branch_name,
            ckpt["step"],
            spec.checkpoint_step,
        )
        cursor_states.append(ckpt["train_loader"])
        if spec.reset_optimizer_state:
            # Weights above still carry this branch's full history; only
            # the optimizer's own momentum EMAs/step count (Geon.state,
            # keyed by param -- see _refresh_state) are dropped, so the
            # next _refresh_state() call re-initializes them from scratch.
            optimizer.state.clear()

        branch_run_path = os.path.join(run_path, branch_name)
        file_metrics = None
        if dist.get_rank() == 0:
            fork._makedirs_robust(branch_run_path)
            file_metrics = os.path.join(branch_run_path, FILENAME_METRICS)
        branches[branch_name] = dict(
            model=model,
            optimizer=optimizer,
            named_params=named_params,
            rule_set=rule_set_b,
            log_metric=_make_log_metric(file_metrics),
            rolling_loss=0.0,
            rolling_loss_step=0,
        )
        print0(
            f"loaded {checkpoint_path} at step {spec.checkpoint_step}"
            + (" (optimizer state reset)" if spec.reset_optimizer_state else ""),
            console=True,
        )

    # All branches in a job read off one shared cloned cursor (see
    # fork_and_explore), so every checkpoint taken at the same fork
    # iteration must agree on data-cursor position -- verified here, not
    # just assumed, since a mismatched checkpoint_step or branch_name would
    # otherwise silently compare branches that saw different data from here
    # on.
    for branch_name, cursor_state in zip(spec.branch_names, cursor_states):
        assert cursor_state == cursor_states[0], (
            f"data-cursor state mismatch: {spec.branch_names[0]!r} vs "
            f"{branch_name!r} at checkpoint_step={spec.checkpoint_step} -- "
            f"these branches didn't fork from the same point"
        )

    cursor = DistributedDataCursor(
        os.path.join(
            resolve_repo_path(train_config.data_source), train_config.train_data_pattern
        ),
        train_config.batch_size,
        vocab_size=train_config.vocab_size,
        seq_len=train_config.seq_len,
    )
    cursor.load_state_dict(cursor_states[0])

    training_time = 0.0

    dist.barrier()
    t0 = time.perf_counter()

    for step in range(spec.checkpoint_step, end_step):
        # One shared batch per step, handed to every branch as-is -- same
        # "identical data, not just identically distributed data" design
        # fork_and_explore itself uses.
        inputs, targets = cursor.next_batch()
        eta = train_config.eta_of(step)
        # 0-based, local to this job's own continuation window -- same
        # convention run_branch_compare.py's own metric_schedule uses.
        steps_since_start = step - spec.checkpoint_step
        if spec.warmup_steps > 0:
            # Same shape as Rule.warmup_factor, relative to this job's own
            # checkpoint_step instead of a rule's start.
            eta *= min(1.0, (steps_since_start + 1) / spec.warmup_steps)
        log_this_iter = spec.metric_schedule.should_do(steps_since_start)

        for branch_name in spec.branch_names:
            b = branches[branch_name]
            train_loss, mbs_batches = _forward_backward(
                b["model"], train_config, inputs, targets
            )
            updates, sizings = b["rule_set"].apply_for_step(
                step, eta, b["named_params"], b["optimizer"]
            )
            b["optimizer"].step(updates, sizings, model=b["model"], batches=mbs_batches)
            b["model"].zero_grad(set_to_none=True)

            b["rolling_loss_step"] += 1
            b["rolling_loss"] = (
                ROLLING_LOSS_BETA * b["rolling_loss"]
                + (1 - ROLLING_LOSS_BETA) * train_loss
            )
            unbiased_rolling_loss = b["rolling_loss"] / (
                1 - ROLLING_LOSS_BETA ** b["rolling_loss_step"]
            )
            approx_training_time = training_time + (time.perf_counter() - t0)

            log_line = (
                f"step:{step+1}/{end_step} {branch_name} "
                f"rolling_loss:{unbiased_rolling_loss:.5f} "
            )
            if log_this_iter:
                val_loss = _evaluate(b["model"], val_inputs, val_targets, train_config)
                log_line += f"val_loss:{val_loss:.5f} "
            print0(
                log_line + f"train_time:{approx_training_time:.3f}s",
                console=True,
            )
            b["log_metric"](
                kind="train",
                step=step + 1,
                train_loss=train_loss.item(),
                rolling_loss=unbiased_rolling_loss.item(),
                train_time=approx_training_time,
            )
            if log_this_iter:
                b["log_metric"](
                    kind="eval",
                    step=step + 1,
                    val_loss=val_loss.item(),
                    train_time=approx_training_time,
                )

    dist.barrier()

    if dist.get_rank() == 0:
        with open(file_done, "w", encoding="utf-8") as f:
            f.write(f"{end_step}\n")
    dist.barrier()


########################################
#                  CLI                 #
########################################


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
        help="Path to a continue config YAML ('runs'). Defaults to "
        "<run_path>/config.yaml.",
    )
    p.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Optional, for running a sweep across multiple parallel nodes: "
        "total number of shards this job list is split across. Requires "
        "--shard_index. Jobs are sorted by their source job's model size "
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
    print("run_branch_continue ", args)

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
    specs = load_continue_yaml(config_path)

    if args.num_shards is not None:
        num_params_by_path = {}
        for spec in specs:
            if spec.path not in num_params_by_path:
                tc, _rules = load_compare_job_config(resolve_repo_path(spec.path))
                num_params_by_path[spec.path] = fork.train_config_num_params(tc)

        def _spec_cost(spec) -> int:
            """Rough FLOPs proxy for load-balancing shards: num_params *
            steps_to_do, where steps_to_do is continue_steps once per
            branch being continued."""
            steps_to_do = spec.continue_steps * len(spec.branch_names)
            return num_params_by_path[spec.path] * steps_to_do

        specs.sort(key=_spec_cost)
        specs = [
            s for i, s in enumerate(specs) if i % args.num_shards == args.shard_index
        ]
        print(f"shard {args.shard_index}/{args.num_shards}: {len(specs)} job(s)")

    # Initialized once for the whole job list, not once per job -- see
    # run_optim_rules.py's identical note (repeated init/destroy of a process
    # group within one process is unsupported by PyTorch).
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    assert 8 % dist.get_world_size() == 0
    dist.barrier()
    try:
        for spec in specs:
            job_run_path = _job_run_path(args.run_path, spec.name, spec.checkpoint_step)
            run_branch_continue(spec, run_path=job_run_path)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    # torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) ...
    main()
