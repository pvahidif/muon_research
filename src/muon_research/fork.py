"""The shared model/optimizer/checkpoint/fork lifecycle for every script
that trains, resumes, or forks a Geon-optimized model (run_optim_rules.py,
run_branch_compare.py, run_branch_continue.py, run_curv_profile.py) -- one
copy of this logic, used identically everywhere.

- ``build_model_and_geon(train_config, rule_set)`` builds a fresh GPT/GPL
  model and Geon optimizer with **one param_group per parameter**, and
  returns ``dict(model=..., optimizer=..., rule_set=...)`` -- the same
  ``rule_set``, now ``resolve``'d against these param names. Doesn't
  include ``named_parameters()`` or a name-by-param mapping -- both are
  one-liners off ``model`` (``list(model.named_parameters())``), so a
  caller that needs one just computes it, rather than this function
  caching and handing back something already sitting on ``model``.
- ``compile_and_warmup`` triggers and times ``torch.compile`` once, up
  front, rather than silently inside the first real training step.
- ``save_checkpoint``/``load_checkpoint`` write/read
  ``<dir>/step_<n>.pt`` atomically, rank-0-only (every rank computes
  bit-identical state, so one writer is enough; every rank reads its own
  copy back independently on resume).
- ``fork_branch(trunk_model, trunk_optimizer, b_train_config, b_rule_set,
  ...)`` clones a live ``(model, optimizer)`` pair into an independent
  branch ready to train under its own rules, returning the same
  ``dict(model=..., optimizer=..., rule_set=...)`` shape as
  ``build_model_and_geon`` (it's built via one, then has its state loaded
  in place). The source doesn't have to be "the trunk" -- any live
  ``(model, optimizer)`` can be forked from, including a
  previously-forked branch.

A branch is never ``copy.deepcopy(trunk_optimizer)``'d -- that would
silently drop Geon's own instance attributes (``_step_count``,
``kl_matched_coeffs``, ``s_min``, ``s_max``, ``kl_search_*``), which
aren't part of its ``state_dict()``, leaving a "branch" that's actually a
different, broken optimizer. It's built fresh instead (via
``build_model_and_geon``), then loaded from the trunk's own live
``state_dict()``s -- see ``fork_branch`` for why the optimizer state_dict
specifically needs ``copy.deepcopy``-ing first.

This design -- and ``Geon.group_of`` doing a fresh lookup on every call
rather than any caller keeping its own {param: group} mapping -- exists
because an earlier version of this logic, duplicated independently across
scripts, had two real bugs that were only caught by running real CUDA
training and diffing tensors, not by reading source. See
tests/test_fork.py for that history and the regression tests guarding it.
"""

import copy
import glob
import os
import re
import time

import torch
import torch.distributed as dist

from muon_research.model import build_model
from muon_research.optim.geon import Geon
from muon_research.rules import RuleSet, TrainConfig

_CHECKPOINT_STEP_RE = re.compile(r"^step_(\d+)\.pt$")

# Distinguishes "caller didn't pass klmatch_schedule" (skip the call
# entirely) from "caller passed klmatch_schedule=None" (a real, valid
# Schedule spec meaning "always due" -- see Schedule's own docstring) --
# None can't double as the "not given" sentinel here.
_UNSET = object()


def _makedirs_robust(path: str, retries: int = 10, delay: float = 0.2) -> None:
    """``os.makedirs(path, exist_ok=True)``, tolerant of a transient
    ``FileNotFoundError`` that can surface when many concurrent processes
    (e.g. separate ``--num_shards`` jobs, possibly on different nodes) create
    overlapping directory trees on a network filesystem (NFS/EFS/FSx-style):
    one node's creation of a shared intermediate directory can race another's
    in a way ``exist_ok=True`` alone doesn't fully cover, since it only
    guards the final ``mkdir`` call, not every step of the walk up to it."""
    for attempt in range(retries):
        try:
            os.makedirs(path, exist_ok=True)
            return
        except FileNotFoundError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


########################################
#          Model & Optimizer           #
########################################


def train_config_num_params(train_config: TrainConfig) -> int:
    """Total parameter count for ``train_config``'s model -- used to sort
    combos by size for the --num_shards/--shard_index sweep sharding in
    run_optim_rules.py's ``main()``. Only ``.numel()`` (shape) is needed, so
    this builds the model on the ``"meta"`` device: same param shapes/count
    as a real ``muon_research.model.build_model`` call, but no storage is
    ever allocated and no init/RNG work happens -- effectively free
    regardless of model size (measured: ~550MB / ~1.6s for a real CPU build
    of a ~137M-param config, vs. ~0MB / ~3.5ms on meta), unlike
    ``build_model_and_geon``'s use of it, which needs real
    (random-initialized, later ``.cuda()``'d) weights and so can't use
    meta.
    """
    with torch.device("meta"):
        model = build_model(train_config)
    return sum(p.numel() for p in model.parameters())


def build_model_and_geon(
    train_config: TrainConfig, rule_set: RuleSet, *, compile_model: bool = False
):
    """Build a fresh GPT/GPL model and a Geon optimizer with **one
    param_group per parameter**, so each param's betas/nesterov/wd_raw/lr
    can be swapped independently as training crosses a rule's step
    boundary -- Geon reads all four live off ``group_of(p)`` every
    ``step()`` call (see ``muon_research/optim/geon.py``), so overwriting a
    group's fields in place (see ``RuleSet.apply_for_step``) is enough; no
    group-membership juggling needed. Resolves ``rule_set`` (see
    ``RuleSet.resolve``) in place before moving the model to CUDA, so a
    broken rule set fails before any GPU memory is spent.

    ``compile_model=True`` compiles immediately (what a script training
    this exact model wants, e.g. run_optim_rules.py's own trunk); the
    default, ``False``, skips it -- for a caller that's about to overwrite
    these weights entirely via ``load_state_dict`` anyway (every
    ``fork_branch``/``load_checkpoint`` caller: a fresh branch or trunk,
    always immediately loaded from the real trunk/checkpoint state),
    compiling here would be wasted work; ``compile_and_warmup`` is the
    explicit, opt-in way to compile after that load instead.

    Returns ``dict(model=..., optimizer=..., rule_set=...)`` -- the same
    ``rule_set`` passed in, now resolved.
    """
    cpu_model = build_model(train_config)
    rule_set.resolve(
        [name for name, _p in cpu_model.named_parameters()],
        train_config.train_steps,
    )

    model = cpu_model.cuda()
    for name, p in model.named_parameters():
        # Zero-init the residual-branch output projections (e.g. attn.proj,
        # mlp.proj, the final proj) -- segment match, not substring, so
        # GPL's embed_proj (a distinct "embed_proj" segment) isn't swept
        # in too.
        if train_config.zero_proj_init and "proj" in name.split("."):
            p.data.zero_()

    named_params = list(model.named_parameters())
    initial = {p: rule_set.rule_for_step(name, 0) for name, p in named_params}
    optimizer = Geon(
        [
            dict(
                params=[p],
                lr=initial[p].lr,
                betas=initial[p].betas,
                nesterov=initial[p].nesterov,
                wd_raw=initial[p].wd_raw,
            )
            for _name, p in named_params
        ],
        eps=train_config.geon_eps,
        s_min=train_config.geon_s_min,
        s_max=train_config.geon_s_max,
    )

    if compile_model:
        model.compile(dynamic=False, fullgraph=True)
    return dict(model=model, optimizer=optimizer, rule_set=rule_set)


def compile_and_warmup(model, train_config: TrainConfig, print0, label: str) -> None:
    """``model.compile()`` itself returns immediately -- tracing+codegen
    only actually happens (and costs time) on first invocation -- so timing
    it means triggering one now, with a throwaway dummy batch (same
    (mbs, seq_len) shape real training uses, so the resulting compiled
    artifact is the one real training reuses; the *values* don't matter,
    only shape/dtype affect the compiled graph). Discarded via zero_grad
    right after, so this has no effect on training. Dynamo's compile cache
    is keyed on the traced graph, not the model instance or its weight
    values, so this also warms the cache for every subsequent model built
    with the same mbs/seq_len in this process -- typically ~free after the
    first call, which is what's actually being measured/reported here."""
    model.compile(dynamic=False, fullgraph=True)
    device = next(model.parameters()).device
    x = torch.randint(
        0,
        train_config.vocab_size,
        (train_config.mbs, train_config.seq_len),
        device=device,
    )
    y = torch.randint(
        0,
        train_config.vocab_size,
        (train_config.mbs, train_config.seq_len),
        device=device,
    )
    t0 = time.perf_counter()
    loss = model(x, y)
    loss.backward()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    model.zero_grad(set_to_none=True)
    print0(f"  [compile] {label}: {t1 - t0:.2f}s", console=True)


########################################
#              Checkpoints             #
########################################


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """The highest-step ``step_<n>.pt`` under ``checkpoint_dir``, or
    ``None`` if there isn't one (including if ``checkpoint_dir`` doesn't
    exist yet -- ``glob`` on a missing directory just yields nothing).
    Every rank calls this independently and gets the same answer (single
    shared filesystem, no coordination needed).
    """
    best_step = -1
    best_path = None
    for path in glob.glob(os.path.join(checkpoint_dir, "step_*.pt")):
        m = _CHECKPOINT_STEP_RE.match(os.path.basename(path))
        if not m:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step, best_path = step, path
    return best_path


def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    model: torch.nn.Module,
    optimizer: Geon,
    train_loader_state: dict,
    **extra,
) -> None:
    """Write ``<checkpoint_dir>/step_<step>.pt`` -- model/optimizer state are
    bit-identical across ranks by construction (every rank computes every
    update from the same all-reduced gradients, see
    ``muon_research/optim/geon.py``'s module docstring), and
    ``train_loader_state`` (just a shared shard/token position, not
    per-rank -- see ``DistributedDataCursor``) is too, so rank 0 alone
    saving is enough (this function no-ops on every other rank); every
    rank reads it back independently on resume (see ``load_checkpoint``),
    no broadcast needed either way.

    ``**extra`` (e.g. ``training_time``/``rolling_loss``/
    ``rolling_loss_step`` for run_optim_rules.py's own trunk checkpoints;
    nothing extra for a branch checkpoint) is merged into the saved
    payload as-is, so different callers can save what they track without
    this function needing to know about all of them.

    Written to a ``.tmp`` path and atomically renamed into place, so a
    crash mid-write never leaves a corrupt checkpoint for the next
    invocation to (fail to) resume from.

    RNG state (torch/cuda/numpy/random) is deliberately NOT saved -- a
    resumed run won't be bit-identical to an uninterrupted one, only
    statistically equivalent, which is the standard tradeoff for not
    overcomplicating this.
    """
    if dist.get_rank() != 0:
        return
    _makedirs_robust(checkpoint_dir)
    payload = {
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_loader": train_loader_state,
        **extra,
    }
    final_path = os.path.join(checkpoint_dir, f"step_{step}.pt")
    tmp_path = final_path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Geon,
    *,
    device,
) -> dict:
    """``torch.load(checkpoint_path)`` and restore its ``model``/
    ``optimizer`` state into the given (already-constructed, e.g. via
    ``build_model_and_geon``) ``model``/``optimizer`` in place.

    Returns the raw loaded payload; callers pull whatever extra fields
    they need (``step``, ``train_loader``, and (for a run_optim_rules.py
    trunk checkpoint only) ``training_time``/``rolling_loss``/
    ``rolling_loss_step`` -- shape varies by which script's
    ``save_checkpoint`` call wrote it, see its own docstring) directly off
    it.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


########################################
#                Forking                #
########################################


def fork_branch(
    trunk_model,
    trunk_optimizer,
    b_train_config: TrainConfig,
    b_rule_set: RuleSet,
    *,
    compile_models: bool,
    label: str,
    print0,
    klmatch_schedule=_UNSET,
) -> dict:
    """One independent clone of the trunk, ready to run under
    ``b_rule_set``.

    Freshly constructed (not ``copy.deepcopy``'d: ``torch.optim.Optimizer``'s
    pickling protocol only preserves ``{state, param_groups}``, silently
    dropping Geon's own ``_step_count``/``s_min``/``s_max``/etc, so a
    deep-copied optimizer would be missing them entirely) then loaded from
    the trunk's own live ``state_dict()``s.

    Model weights: ``Module.load_state_dict`` copies values into the
    target's own already-allocated tensors, so this is safe as-is.

    Optimizer state is NOT safe to load the same naive way. Unlike
    ``Module.load_state_dict``, ``Optimizer.state_dict()`` returns the
    *live* state tensors by reference, and ``Optimizer.load_state_dict``
    installs them as-is with no clone. Loading straight from
    ``trunk_optimizer.state_dict()`` would leave every branch's Adam state
    (``m``/``v``) aliasing the trunk's own tensors -- and every other
    branch forked from the same trunk snapshot, since they'd all reference
    the same underlying storage -- so any branch's in-place state update
    (``Geon._refresh_state``'s ``.lerp_()``/``.addcmul_()``) would silently
    corrupt the trunk and every sibling branch. ``copy.deepcopy`` the
    state_dict first so each branch gets its own independent tensors.

    ``klmatch_schedule``, if given (some callers never pass it, leaving
    Geon's own constructed default in place), sets the branch's own
    ``Geon.kl_match_cache_schedule`` right after construction, before
    load_state_dict.

    Returns the same ``dict(model=..., optimizer=..., rule_set=...)``
    shape as ``build_model_and_geon``.
    """
    built = build_model_and_geon(b_train_config, b_rule_set, compile_model=False)
    b_model, b_optimizer = built["model"], built["optimizer"]
    if klmatch_schedule is not _UNSET:
        b_optimizer.set_kl_match_cache_schedule(klmatch_schedule)
    if compile_models:
        compile_and_warmup(b_model, b_train_config, print0, label)
    b_model.load_state_dict(trunk_model.state_dict())
    b_optimizer.load_state_dict(copy.deepcopy(trunk_optimizer.state_dict()))
    return built
