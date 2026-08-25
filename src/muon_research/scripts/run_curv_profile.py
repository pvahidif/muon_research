"""One-shot curvature/KL-match profiling of an already-checkpointed model
(model, optimizer, and data-loader cursor state), with no training in
progress. Works on either of the two checkpoint shapes this codebase
produces, freely mixed in the same config (see ``RunSpec.path`` and
``run_optim_rules.load_checkpoint_config``):

- a plain run_optim_rules.py run (``<run_path>/checkpoints/step_<n>.pt``,
  ``<run_path>/config.json`` flat), or
- one branch of a run_branch_compare.py job
  (``<job_run_path>/branches/<branch_name>/checkpoints/step_<n>.pt``,
  that branch's own nested ``config.json`` -- ``path`` below must be the
  branch's own directory, e.g. ``.../branches/svdp_p025``, not the job dir
  above it).

Uses the shared profiling machinery (``ProfileConfig``, ``profile_matrix``,
``run_profile_capture``, ``decompose_matrix``, ``hvp_curvatures``,
``sample_gradient_projections``, ...) from ``muon_research/curv.py`` --
see that module's own docstring. This script loads a real checkpoint of
either shape above (rules-based model/optimizer construction via
``fork.build_model_and_geon``/``fork.load_checkpoint``, config via
``run_optim_rules.load_checkpoint_config``), resumes its saved data
cursor to pull the batch(es) the original run would have processed next
(pool A -- see ``muon_research/curv.py``), does one forward+backward to
get real gradients at that state, and profiles once.

For each ``(path, step)`` job (``path`` is either checkpoint source above,
``step`` one of its saved ``checkpoints/step_<n>.pt``):

1. Load ``train_config``/``rules`` from the checkpoint's own ``config.json``
   (``run_optim_rules.load_checkpoint_config``),
   build a fresh model+Geon from them, and load the checkpoint's model/
   optimizer/data-loader state -- exactly the regular resume path
   run_optim_rules.py itself would take.
2. Data: pool A (see ``muon_research/curv.py``'s own pool A/B docstring) is
   ALWAYS the checkpoint's own resumed train-data cursor, pulling the next
   mbs-chunked batch(es) the original run would have processed next, had
   it kept training -- see ``_load_pool_batches``. Pool B (what every
   quantity besides ``sigma``/``D_i`` themselves -- gamma, phi, ... -- is
   estimated from) reuses pool A's own batch(es) when
   ``profile.profile_batch_size`` is unset (the documented approximation,
   modeled as independent rather than actually independent); if set,
   pool B is instead a genuinely separate batch from held-out val data --
   one FIXED batch reused across every profiling event by default, or
   (``profile.profile_batch_resample``) a fresh deterministic slice per
   event, keyed off its own checkpoint step.
3. One forward+backward over pool A's own batches, all-reduced -- real
   gradients at this exact checkpoint state (needed for
   ``profile_source="signal"``/``"grad"``, which read them off ``p.grad``;
   harmless overhead for ``"weight"``/``"prev_momentum"``, which don't
   touch ``p.grad`` at all). Pool B's own batches are never used for this
   -- they only ever feed ``run_profile_capture``, so that ``p.grad``
   (hence pool A's own ``D_i``/``sigma_i`` for ``"signal"``/``"grad"``)
   never depends on them, even when pool B happens to alias pool A.
4. ``run_profile_capture``, once, for every 2D block weight
   (``model.blocks.*``, plus GPL's own ``embed_proj.weight``) selected by
   ``profile.param_name_patterns``. Results saved to
   ``<run_path>/<name>/step_<step>/profiles/step_<step>/svd_curv.pt``
   (``iter_num`` in the payload is this checkpoint's own step, not a
   training-loop step).

Example config:

    runs:
      - name: svdp_p025             # one branch of a run_branch_compare.py job --
        path: experiments/exp002_compare_muon_pow/pow_checkpoint/muon/step_002000/branches/svdp_p025
        steps: [2016]               # path is that branch's own dir, not the job dir above it

    profile:
      profile_source: signal
      profile_decomposition: svd
      compute_gamma: true
      compute_phi: true
      max_modes: 16
      param_name_patterns: null   # optional; omit/null = every 2D block weight
      mbs: null   # optional; omit/null = the checkpoint's own saved mbs

Run with, e.g.:
    torchrun --standalone --nproc_per_node=1 \\
        src/muon_research/scripts/run_curv_profile.py \\
        --run_path ... --config_path ...
"""

# pylint: disable=all

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import yaml

from muon_research.curv import (
    ProfileConfig,
    _makedirs_robust,
    load_profile_config,
    run_profile_capture,
    select_matrix_params,
)
from muon_research.data import DistributedDataCursor
from muon_research.paths import resolve_repo_path
from muon_research import fork
from muon_research.constants import FILENAME_CONFIGS, FILENAME_DONE, FILENAME_LOGS
from muon_research.rules import load_checkpoint_config
from muon_research.scripts.run_optim_rules import CHECKPOINTS_DIRNAME

########################################
#                Config                #
########################################


@dataclass
class RunSpec:
    name: str
    # This checkpoint's source -- either a plain run_optim_rules.py run_path
    # or one branch's own directory from a run_branch_compare.py job (see
    # this module's own docstring); load_checkpoint_config accepts either
    # shape, so nothing else here needs to know which one it is. Kept
    # exactly as given in the YAML (relative or absolute) -- resolved
    # against the repo root on demand, at each actual file-I/O call site,
    # so it stays portable across checkouts and round-trips unchanged into
    # config.json/logs.
    path: str
    steps: list[int]


def load_run_yaml(path: str) -> tuple[list[RunSpec], ProfileConfig]:
    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    raw_runs = payload.get("runs") or []
    if not raw_runs:
        raise ValueError(f"config {path!r} has no 'runs' entries")
    runs = []
    # (name, step) is what actually determines each job's output directory
    # (see _job_run_path) -- so that's the uniqueness constraint, not name
    # alone.
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
        runs.append(RunSpec(name=name, path=str(entry["path"]), steps=steps))

    profile_config = load_profile_config(path)
    return runs, profile_config


def _job_run_path(base_run_path: str, name: str, step: int) -> str:
    return os.path.join(base_run_path, name, f"step_{step:06d}")


########################################
#              One job                 #
########################################


def _split_into_microbatches(inputs, targets, mbs: int):
    assert len(inputs) % mbs == 0
    return [
        (inputs[i * mbs : (i + 1) * mbs], targets[i * mbs : (i + 1) * mbs])
        for i in range(len(inputs) // mbs)
    ]


def _load_pool_batches(
    train_config,
    profile_config: ProfileConfig,
    ckpt: dict,
    checkpoint_step: int,
) -> tuple[
    list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]
]:
    """Pool A / pool B batches (see ``muon_research/curv.py``'s own
    docstring), each already split into ``train_config.mbs``-sized
    microbatches. Pool A is ALWAYS the checkpoint's own resumed
    train-data cursor -- the real batch the original run would have
    processed next, had it kept training -- regardless of
    ``profile_config``: it's what the caller's forward+backward populates
    ``p.grad`` from, hence what ``profile_source="signal"``/``"grad"``
    read off it.

    Pool B must never be derived from that same forward+backward, or
    ``profile_source="signal"``/``"grad"`` (which read ``p.grad``, itself
    a function of pool A's batch) would end up correlated with whatever
    pool B measures gamma/phi from -- exactly what
    ``ProfileConfig.profile_batch_size``'s own "genuinely independent"
    promise requires NOT happening. So: with ``profile_batch_size`` unset,
    pool B is pool A's own batches, reused as-is (the documented
    approximation -- modeled as independent, not actually independent);
    with it set, pool B is instead pulled from a SEPARATE cursor over
    held-out val data (past ``val_size``, so it never overlaps eval),
    fixed across every profiling event by default, or
    (``profile_batch_resample``) a fresh deterministic slice per event,
    keyed off ``checkpoint_step``.
    """
    cursor_a = DistributedDataCursor(
        os.path.join(
            resolve_repo_path(train_config.data_source), train_config.train_data_pattern
        ),
        train_config.batch_size,
        vocab_size=train_config.vocab_size,
        seq_len=train_config.seq_len,
    )
    cursor_a.load_state_dict(ckpt["train_loader"])
    pool_a = _split_into_microbatches(*cursor_a.next_batch(), train_config.mbs)

    if profile_config.profile_batch_size is None:
        return pool_a, pool_a

    cursor_b = DistributedDataCursor(
        os.path.join(
            resolve_repo_path(train_config.data_source), train_config.val_data_pattern
        ),
        profile_config.profile_batch_size,
        vocab_size=train_config.vocab_size,
        seq_len=train_config.seq_len,
    )
    cursor_b.advance_tokens(train_config.val_size)
    if profile_config.profile_batch_resample:
        cursor_b.advance_tokens(checkpoint_step * profile_config.profile_batch_size)
    pool_b = _split_into_microbatches(*cursor_b.next_batch(), train_config.mbs)
    return pool_a, pool_b


def run_checkpoint_profile(
    name: str,
    source_path: str,
    checkpoint_step: int,
    profile_config: ProfileConfig,
    run_path: str,
) -> None:
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

    train_config, rule_set = load_checkpoint_config(resolve_repo_path(source_path))
    # Overridden once, here, rather than at each forward/backward call site
    # -- every downstream use (pool A/B chunking, hvp_curvatures,
    # sample_gradient_projections) picks it up automatically. Purely a
    # microbatch-chunking knob (see ProfileConfig.mbs); the checkpoint's own
    # saved mbs is used unless profile_config.mbs overrides it.
    if profile_config.mbs is not None:
        train_config = replace(train_config, mbs=profile_config.mbs)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    torch.cuda.manual_seed_all(train_config.seed)

    if dist.get_rank() == 0:
        _makedirs_robust(run_path)
        print("logs:    ", file_logs)
        print("configs: ", file_configs)
        config_payload = dict(
            name=name,
            source_path=source_path,
            checkpoint_step=checkpoint_step,
            train=asdict(train_config),
            profile=asdict(profile_config),
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

    print0("=" * 100)
    print0(f"Profiling checkpoint: {source_path} step={checkpoint_step}")
    print0(f"Config: {train_config}")
    print0(f"Profile: {profile_config}")
    print0("=" * 100)

    built = fork.build_model_and_geon(train_config, rule_set)
    model, optimizer = built["model"], built["optimizer"]
    named_params = list(model.named_parameters())

    checkpoint_path = os.path.join(
        source_path, CHECKPOINTS_DIRNAME, f"step_{checkpoint_step}.pt"
    )
    ckpt = fork.load_checkpoint(
        resolve_repo_path(checkpoint_path), model, optimizer, device=device
    )
    assert int(ckpt["step"]) == checkpoint_step, (ckpt["step"], checkpoint_step)
    print0(
        f"loaded checkpoint {checkpoint_path} at step {checkpoint_step}", console=True
    )

    # Every 2D weight inside a transformer block (attn/mlp matrices), plus
    # GPL's own embed_proj -- derived from param names, since a simplified
    # model/optimizer construction isn't available for a rules-based
    # checkpoint.
    matrix_params_named = [
        (n, p)
        for n, p in named_params
        if p.ndim >= 2 and (n.startswith("blocks.") or n == "embed_proj.weight")
    ]
    matrix_params_named = select_matrix_params(
        matrix_params_named, profile_config.param_name_patterns
    )
    print0(
        f"profile matrices ({len(matrix_params_named)}): "
        f"{[n for n, _p in matrix_params_named]}",
        console=True,
    )

    pool_a_batches, pool_b_batches = _load_pool_batches(
        train_config, profile_config, ckpt, checkpoint_step
    )
    print0("pool A: resumed checkpoint's own train cursor", console=True)
    if profile_config.profile_batch_size is None:
        print0(
            "pool B: reusing pool A's own batch(es) (profile_batch_size unset)",
            console=True,
        )
    else:
        resample = "resampled" if profile_config.profile_batch_resample else "fixed"
        print0(
            f"pool B: {resample} held-out val batch "
            f"(profile_batch_size={profile_config.profile_batch_size}, past val_size)",
            console=True,
        )

    # Only pool A's own batches ever populate p.grad -- profile_source=
    # "signal"/"grad" (profiling_tensor, see curv.py) read pool A's D_i/
    # sigma_i off exactly this; pool B's batches are passed through
    # untouched to run_profile_capture below and never go through this
    # forward+backward, so p.grad can't end up depending on them.
    model.zero_grad(set_to_none=True)
    for x, y in pool_a_batches:
        loss = model(x, y)
        loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

    run_profile_capture(
        model=model,
        optimizer=optimizer,
        matrix_params_named=matrix_params_named,
        pool_b_batches=pool_b_batches,
        device=device,
        profile_config=profile_config,
        train_config=train_config,
        optim=name,
        iter_num=checkpoint_step,
        run_path=run_path,
        print0=print0,
    )

    model.zero_grad(set_to_none=True)
    dist.barrier()
    if dist.get_rank() == 0:
        with open(file_done, "w", encoding="utf-8") as f:
            f.write(f"{checkpoint_step}\n")
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
        help="Path to a profile config YAML ('runs' + 'profile'). Defaults "
        "to <run_path>/config.yaml.",
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
    print("run_curv_profile ", args)

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
    run_specs, profile_config = load_run_yaml(config_path)

    jobs = [
        (run_spec.path, step, run_spec.name)
        for run_spec in run_specs
        for step in run_spec.steps
    ]

    if args.num_shards is not None:
        num_params_by_path = {}
        for source_path, _step, _name in jobs:
            if source_path not in num_params_by_path:
                tc, _rules = load_checkpoint_config(resolve_repo_path(source_path))
                num_params_by_path[source_path] = fork.train_config_num_params(tc)

        jobs.sort(key=lambda job: num_params_by_path[job[0]])
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
        for source_path, step, name in jobs:
            job_run_path = _job_run_path(args.run_path, name, step)
            run_checkpoint_profile(
                name, source_path, step, profile_config, run_path=job_run_path
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    # torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) ...
    main()
