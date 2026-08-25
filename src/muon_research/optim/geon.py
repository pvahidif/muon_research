"""Geon: generalized optimizer

Geon keeps Adam-style state (``step``, ``m``, ``v``) for **every** parameter
and defers all update decisions to step time: which update rule each param
uses and how each update is sized are passed as arguments to :meth:`Geon.step`,
so the same instance can play different roles. Simplicity and generality over
speed by design: no sharding, no caching — every rank computes every update
from the (already all-reduced, hence rank-identical) gradients.

One step = four small phases:

1. **State.** Every param's ``step`` / ``m`` / ``v`` are refreshed from
   ``p.grad`` (Adam EMAs with the group's ``betas``) — including params whose
   update is ``"skip"``, so moments stay warm for later steps.

2. **Direction.** Per param, from ``updates[p]``. The *signal* is the
   bias-corrected ``m_hat = m / (1 - beta1^step)``, or its Nesterov version
   ``lerp(grad, m_hat, beta1)`` if the group sets ``nesterov=True``.

   * ``"skip"``        — no weight write at all (no weight decay either)
   * ``"adamw"``       — ``signal / (sqrt(v_hat) + eps)``
   * ``"muon"``        — Newton-Schulz polar of the signal, x aspect scale
   * ``("svd_pow", p)``  — exact-SVD ``U S^p Vᵀ`` of the signal with
     median-normalized, clamped singular values, x aspect scale
   * ``("svd_band", q1, q2)`` — exact-SVD ``U diag(s) Vᵀ`` of the signal,
     ``s = 1`` for rank in ``[q1·n, q2·n)`` (rank 0 = largest singular
     value) and ``s = 0`` elsewhere -- equal weight within the band, not
     scaled by the signal's own singular values, x aspect scale

3. **Size.** From ``sizings``, a list of entries covering disjoint sets of
   params:

   * ``("learning_rate", params, lr)`` — each param in ``params`` uses ``lr``.
   * ``("kl_match", params, lr)`` — probe a joint **Muon** step at ``lr`` on
     exactly these params (all other weights untouched), measure the token-mean
     output KL vs the pre-step model, then binary-search one common scale for
     these params' *actual* directions (from phase 2) that reproduces that KL.
   * ``("fro_match", params, lr)`` — same Muon probe, but match the *joint*
     Frobenius norm (``sqrt(Σ ‖·‖_F²)`` across ``params``, i.e. treating them
     as one block-diagonal update) instead of KL. Norms scale linearly with
     the size, so this is a closed form, not a search.
   * ``("op_match", params, lr)`` — same Muon probe, matching the *joint*
     operator norm instead (``max ‖·‖_op`` across ``params`` — the operator
     norm of a block-diagonal update equals the max of its blocks'). Also
     closed form.

   Every non-``"skip"`` param must be covered by exactly one entry; a
   ``ValueError`` is raised otherwise.

4. **Write.** ``p ← (1 - wd_raw) · p - size · direction``. ``wd_raw`` is the
   group's **raw** decay multiplier — unlike AdamW-style decoupled weight
   decay, it is NOT multiplied by the lr, hence the name — and is not
   applied to ``"skip"`` params.

All learning rates — group ``lr`` and the ``lr`` inside sizing entries — must
already include any schedule / decay; Geon applies no schedule of its own.

Every ``param_sync_every`` steps (default 10), after the write, every
param is all-reduced (averaged) across ranks. Gradients are already
all-reduced each step, so ranks should already compute identical updates;
this is cheap insurance against them drifting apart from per-rank
floating-point nondeterminism (e.g. kernel selection) accumulating over
many steps. Set ``param_sync_every<=0`` to disable. No-op unless
``torch.distributed`` is initialized with more than one rank.

``model`` and ``batches`` (a list of ``(inputs, targets)`` microbatches) are
required whenever a ``kl_match`` sizing is present; ``model`` must expose
``model.logits(x)`` (as ``geon1.model.GPT`` does). KL sums are all-reduced,
so all ranks arrive at the same matched scale. ``fro_match`` / ``op_match``
need neither — norms are a pure function of the (already rank-identical)
tensors, no forward pass required.

Changing a param's group ``betas`` after the fact — e.g. a schedule that
swaps hyperparameters as training crosses a step boundary — must NOT be
done by writing ``group["betas"]`` directly. Bias correction (phase 2's
``m_hat``/``v_hat``, ``1 / (1 - beta**step)``) assumes the EMA has run
under one constant ``beta`` since ``step`` 1; switching ``beta`` without
adjusting ``m``/``v`` would apply the new ``beta``'s correction factor to
an EMA that wasn't actually accumulated under it. Use
:meth:`Geon.set_betas` instead, which rescales the existing ``m``/``v`` in
place so the very next bias correction (still at the same ``step`` --
this never resets it) reproduces exactly the unbiased estimate the old
``beta`` would have given, then lets the EMA continue under the new
``beta`` from there with no discontinuity. No-ops when the betas given
are unchanged, so it's safe to call unconditionally every step, the way a
per-step hyperparameter schedule does.

Depends on nothing else in this codebase,
so every script that builds a Geon (or reuses its Schedule/
BinarySearch helpers) can import this module without a
circular-import risk.
"""

from __future__ import annotations

import math
import warnings
from typing import Sequence

import torch
from torch import Tensor
import torch.distributed as dist
import torch.nn.functional as F

# "skip" | "adamw" | "muon" | ("svd_pow", p) | ("svd_band", q1, q2)
UpdateKind = str | tuple[str, float] | tuple[str, float, float]
# ("learning_rate" | "kl_match" | "fro_match" | "op_match", params, lr)
SizingEntry = tuple[str, Sequence[torch.nn.Parameter], float]


def _aspect_scale(p: Tensor) -> float:
    return float(max(1, p.size(-2) / p.size(-1)) ** 0.5)


def _newton_schulz_polar(G: Tensor, iters: int = 12) -> Tensor:
    """Approximate polar factor of ``G`` (Newton-Schulz, bfloat16)."""
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 2, -1.5, 0.5
    for _ in range(iters):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def _svd_power(
    M: Tensor,
    p: float,
    *,
    s_min: float = 1e-5,
    s_max: float = 1e5,
) -> Tensor:
    """``U S_eff^p Vᵀ`` with median-normalized, clamped singular values."""
    # pylint: disable=not-callable
    U, S, Vh = torch.linalg.svd(M.float(), full_matrices=False)
    if float(p) == 0.0:
        return U @ Vh
    S = S / S.median().clamp_min(1e-12)
    S = S.clamp_min(float(s_min))
    S = S.clamp_max(float(s_max))
    return (U * S.pow(float(p)).unsqueeze(-2)) @ Vh


def _svd_band(
    M: Tensor,
    q1: float,
    q2: float,
) -> Tensor:
    """``U diag(s) Vᵀ`` with ``s = 1`` for rank in ``[q1·n, q2·n)`` (rank 0
    = largest, ``n`` = num singular values) and ``s = 0`` elsewhere -- equal
    weight for every direction kept in the band, not scaled by the signal's
    own singular value magnitude.
    """
    # pylint: disable=not-callable
    U, S, Vh = torch.linalg.svd(M.float(), full_matrices=False)
    n = S.size(-1)
    rank = torch.arange(n, device=S.device, dtype=S.dtype)
    band = (rank >= float(q1) * n) & (rank < float(q2) * n)
    S = band.to(S.dtype)
    return (U * S.unsqueeze(-2)) @ Vh


@torch.no_grad()
def _kl_of_deltas(
    model: torch.nn.Module,
    batches: list[tuple[Tensor, Tensor]],
    deltas: dict[torch.nn.Parameter, Tensor],
    scale: float,
) -> float:
    """Token-mean KL(p_θ || p_{θ - scale·Δ}) on ``batches``; restores weights."""
    saved = {p: p.data.clone() for p in deltas}
    device = next(model.parameters()).device
    kl_sum = torch.zeros((), device=device, dtype=torch.float64)
    ntok = torch.zeros((), device=device, dtype=torch.float64)
    was_training = model.training
    model.eval()
    try:
        for x, y in batches:
            for p in deltas:
                p.copy_(saved[p])
            logp_b = F.log_softmax(model.logits(x).float(), dim=-1)
            for p, d in deltas.items():
                p.add_(d.to(dtype=p.dtype), alpha=-float(scale))
            logp_a = F.log_softmax(model.logits(x).float(), dim=-1)
            kl_sum = kl_sum + (logp_b.exp() * (logp_b - logp_a)).sum()
            ntok = ntok + float(y.numel())
        t = torch.stack([kl_sum, ntok])
        if dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t[0].item() / max(t[1].item(), 1.0))
    finally:
        model.train(was_training)
        for p, w in saved.items():
            p.copy_(w)


class BinarySearch:
    """Namespace for the log-space bracketed binary search shared by every
    KL-matching search in this codebase (``Geon._kl_matched_size``, the
    ``_kl_matched_scale`` in run_branch_compare.py) -- same algorithm, just
    a different ``probe`` closure per caller. Stateless
    (a plain function grouped under a class for discoverability, not an
    object with instance data) -- ``bracketed_binary_search`` needs no
    ``self``.
    """

    @staticmethod
    def bracketed_binary_search(
        probe,
        target: float,
        *,
        init: float,
        iters: int,
        expand_factor: float,
    ) -> tuple[float, float]:
        """Log-space binary search for ``x`` such that ``probe(x) ~=
        target``, assuming ``probe`` is monotonically increasing in ``x``.
        Costs exactly ``iters`` calls to ``probe``, always -- one per
        iteration, whether that iteration is spent expanding the bracket or
        narrowing it.

        Starts from ``lo = hi = init`` (a single guess, not a pre-guessed
        bracket) and doesn't trust even that blindly: each iteration first
        makes sure ``lo`` doesn't already overshoot ``target`` (if it does,
        that same probe is a valid -- and, being inside the bracket, at
        least as tight as whatever ``hi`` we had -- upper bound, so it's
        promoted to ``hi`` before ``lo /= expand_factor`` and the iteration
        ends), then that ``hi`` doesn't already undershoot it
        (symmetrically promoted to ``lo`` before ``hi *= expand_factor``);
        only once both bounds actually bracket ``target`` does an iteration
        spend its probe on an ordinary bisection step instead. No probe is
        ever wasted: one that fails to establish its own side always
        tightens the other. A target far from ``init`` can still burn most
        or all of the ``iters`` budget on expansion, trading bisection
        precision for a predictable, fixed total cost -- callers whose
        ``init`` rarely needs much expansion (most of them) see no
        difference from plain bisection.

        Returns ``(x, probe(x))`` at the last point probed -- either the
        final bisection midpoint, or (only if every iteration went to
        expansion) the last bound tried.
        """
        log_lo = log_hi = math.log(init)
        log_step = math.log(expand_factor)
        val_lo = val_hi = None
        x, val = math.exp(log_lo), None

        for _ in range(iters):
            if val_lo is None:
                x = math.exp(log_lo)
                val = probe(x)
                if val <= target:
                    val_lo = val
                else:
                    log_hi, val_hi = log_lo, val
                    log_lo -= log_step
                continue
            if val_hi is None:
                x = math.exp(log_hi)
                val = probe(x)
                if val >= target:
                    val_hi = val
                else:
                    log_lo, val_lo = log_hi, val
                    log_hi += log_step
                continue
            mid = 0.5 * (log_lo + log_hi)
            x = math.exp(mid)
            val = probe(x)
            if val < target:
                log_lo, val_lo = mid, val
            else:
                log_hi, val_hi = mid, val
        return x, val


class Schedule:
    """ "Should this happen at step i?" -- a small, reusable, 0-based step
    schedule. Not specific to Geon or to KL matching; used by
    ``Geon.kl_match_cache_schedule`` to control how often a slow-moving
    quantity gets recomputed vs reused, but stands on its own otherwise.
    Validated once, at construction (``_validate``, called from
    ``__init__``) -- ``should_do(step)`` is the only thing callers need
    afterward, and never raises for a config that passed construction.

    ``cache_schedule`` (``None``, a dict with a ``"_type"`` key, or another
    ``Schedule`` instance) accepts:

    * ``None`` -- always due: ``should_do`` always returns ``True``.
    * ``{"_type": "ap", "k": <int>}`` -- due every ``k`` steps
      (``step % k == 0``), a plain arithmetic progression.
    * ``{"_type": "schedule", "schedule": [(start, end, k), ...]}`` -- step
      ``i`` is due if some entry has ``start <= i < end`` and
      ``(i - start) % k == 0``; a step outside every entry's range is never
      due. Use e.g. ``[(0, 100, 1), (100, 1000, 10), (1000, 10**9, 100)]``
      for "every step at first, then every 10th, then every 100th" --
      dense early, sparse later.
    * Another ``Schedule`` -- unwrapped to its own ``cache_schedule`` first
      (``Schedule(Schedule(spec))`` == ``Schedule(spec)``), so a caller that
      already holds a ``Schedule`` (e.g. a resolved config field) can hand
      it anywhere a raw spec is expected -- ``Geon.__init__``,
      ``set_kl_match_cache_schedule``, or a fresh ``Schedule(...)`` of its
      own -- without unwrapping it manually first.
    """

    def __init__(self, cache_schedule: dict | Schedule | None):
        if isinstance(cache_schedule, Schedule):
            cache_schedule = cache_schedule.cache_schedule
        self._validate(cache_schedule)
        self.cache_schedule = cache_schedule

    @staticmethod
    def _validate(cache_schedule) -> None:
        if cache_schedule is None:
            return
        if not isinstance(cache_schedule, dict) or "_type" not in cache_schedule:
            raise ValueError("Schedule spec must be None or a dict with a '_type' key")
        kind = cache_schedule["_type"]
        if kind == "ap":
            if "k" not in cache_schedule:
                raise ValueError("Schedule _type='ap' needs a 'k' field")
            k = cache_schedule["k"]
            if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
                raise ValueError(f"Schedule 'ap' k must be a positive int, got {k!r}")
        elif kind == "schedule":
            if "schedule" not in cache_schedule:
                raise ValueError("Schedule _type='schedule' needs a 'schedule' field")
            entries = cache_schedule["schedule"]
            if not entries:
                raise ValueError("Schedule 'schedule' must be non-empty")
            for entry in entries:
                start, end, k = entry
                if start >= end:
                    raise ValueError(f"Schedule entry {entry!r} needs start < end")
                if k <= 0:
                    raise ValueError(f"Schedule entry {entry!r} needs k > 0")
        else:
            raise ValueError(f"Schedule _type must be 'ap' or 'schedule', got {kind!r}")

    def should_do(self, step: int) -> bool:
        """Whether this schedule is due at ``step`` (0-based)."""
        if self.cache_schedule is None:
            return True
        kind = self.cache_schedule["_type"]
        if kind == "ap":
            return step % int(self.cache_schedule["k"]) == 0
        if kind == "schedule":
            return any(
                start <= step < end and (step - start) % k == 0
                for start, end, k in self.cache_schedule["schedule"]
            )
        raise ValueError(f"unknown Schedule _type {kind!r}")


class Geon(torch.optim.Optimizer):
    """Per-step-pluggable update rules + sizing; see module docstring.

    Group hyperparameters: ``betas``, ``eps``, ``wd_raw`` (raw
    multiplier, not lr-scaled), ``nesterov``. ``lr`` is stored for external
    bookkeeping (e.g. schedules) but sizes come solely from ``sizings``.

    ``kl_match_cache_schedule`` (a raw dict/None, wrapped into a
    ``Schedule`` -- see its docstring for the accepted shapes) controls how
    often ``_kl_matched_size`` (the ``kl_match`` sizing kind's binary search
    -- 1 + ``kl_search_iters`` forward-pass probes per call, by default)
    actually re-searches for ``coeff`` versus reusing the last one found for
    that call's exact param group (keyed by ``frozenset(params)``, so a
    freshly-constructed model's optimizer -- e.g. a new branch -- always
    starts with an empty cache; no manual invalidation needed). A cache
    *hit* skips every probe entirely (not just the search: the reference
    Muon probe too) and returns ``coeff_cached * lr`` -- ``lr`` is read
    fresh every call, not cached, so a stale ``coeff`` still tracks the
    current step's schedule-driven ``lr`` correctly. The very first call
    for a given param group always computes, regardless of the schedule.
    ``None`` (default) means always recompute -- today's behavior,
    unchanged unless opted in.

    The ``step`` fed to ``Schedule.should_do`` is ``self._step_count`` at
    call time: since it's incremented once at the very end of ``step()``,
    after ``_resolve_sizes`` (hence ``_kl_matched_size``) already ran, its
    value there is exactly "steps completed before this one" -- the same
    0-based convention ``metric_schedule``/``checkpoint_after_steps`` use
    elsewhere.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-10,
        wd_raw: float = 0.0,
        nesterov: bool = False,
        *,
        kl_search_iters: int = 16,
        kl_scale_init: float = 1.0,
        kl_search_expand_factor: float = 10.0,
        kl_match_cache_schedule: dict | Schedule | None = None,
        s_min: float = 1e-5,
        s_max: float = 1e5,
        param_sync_every: int = 10,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, wd_raw=wd_raw, nesterov=nesterov)
        super().__init__(params, defaults)
        self.kl_search_iters = int(kl_search_iters)
        self.kl_scale_init = float(kl_scale_init)
        # See BinarySearch.bracketed_binary_search: a bound that doesn't yet
        # bracket the target is expanded outward by this factor, one
        # iteration of the shared kl_search_iters budget at a time.
        self.kl_search_expand_factor = float(kl_search_expand_factor)
        # Wraps (and validates) the raw dict/None here; the instance
        # attribute is a Schedule, not the raw input -- see Schedule and
        # _kl_matched_size.
        self.kl_match_cache_schedule = Schedule(kl_match_cache_schedule)
        # Public: the coefficient last found (or reused) for each kl_match
        # sizing group, keyed by frozenset(params) -- see _kl_matched_size.
        # Exposed so callers (e.g. run_branch_compare.py) can
        # report what's currently in effect.
        self.kl_matched_coeffs: dict[frozenset, float] = {}
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.param_sync_every = int(param_sync_every)
        self._step_count = 0

    # ------------------------------------------------------------- helpers

    def group_of(self, p: torch.nn.Parameter) -> dict:
        """``p``'s own param_group dict (Geon gives every parameter its
        own group, so this always returns exactly one). A plain linear
        search, not cached anywhere -- callers that need a param's
        current lr/betas/nesterov/wd_raw (e.g. ``RuleSet.apply_for_step``)
        call this fresh each time rather than keeping their own {param:
        group} mapping, which would need explicit invalidation whenever
        ``load_state_dict`` replaces ``param_groups`` wholesale. Cheap
        enough at the param-tensor counts this codebase works with (one
        group per tensor, not per scalar) that this has never been a
        bottleneck."""
        for group in self.param_groups:
            if any(p is q for q in group["params"]):
                return group
        raise KeyError("parameter not in Geon")

    def _iter_params(self):
        for group in self.param_groups:
            for p in group["params"]:
                yield group, p

    def set_kl_match_cache_schedule(
        self, cache_schedule: dict | Schedule | None
    ) -> None:
        """Replace ``kl_match_cache_schedule`` after construction (e.g. once
        a per-run override is resolved, when the caller isn't the one
        constructing this ``Geon``) -- re-wraps and re-validates via
        ``Schedule``, same as ``__init__``. Does NOT clear
        ``kl_matched_coeffs``: a schedule swap doesn't itself invalidate
        already-cached coefficients (the new schedule just governs whether
        future calls treat them as fresh)."""
        self.kl_match_cache_schedule = Schedule(cache_schedule)

    def set_betas(self, p: torch.nn.Parameter, betas: tuple[float, float]) -> None:
        """Set ``p``'s own param_group ``betas`` -- the only safe way to
        change them after construction (see module docstring).

        For whichever of the two betas actually changes, rescales the
        corresponding EMA (``m`` for beta1, ``v`` for beta2) in place so
        the next bias correction -- still at this same ``step``, never
        reset -- reproduces exactly the unbiased estimate the *old* beta
        would have given: for beta ``b`` at ``step`` ``t``, ``m_hat = m /
        (1 - b**t)`` is that estimate, so solving ``m_new / (1 -
        b_new**t) == m_old / (1 - b_old**t)`` for ``m_new`` gives
        ``m_new = m_old * (1 - b_new**t) / (1 - b_old**t)``. This keeps
        the corrected estimate continuous across the switch -- no
        discontinuity, no discarded momentum -- and is exact assuming the
        underlying quantity being averaged hasn't itself changed at this
        exact step.

        No-op (beyond the assignment itself) if ``betas`` is unchanged, or
        if ``step == 0`` (``m``/``v`` are still exactly zero, nothing to
        rescale) -- safe to call unconditionally every step, e.g. from a
        per-step hyperparameter schedule that re-asserts the active rule's
        betas whether or not they actually moved since last step.
        """
        group = self.group_of(p)
        old_betas = tuple(group["betas"])
        new_betas = tuple(betas)
        if old_betas != new_betas:
            state = self.state.get(p)
            if state and state["step"] > 0:
                step = state["step"]
                for key, b_old, b_new in zip(("m", "v"), old_betas, new_betas):
                    if b_old != b_new:
                        state[key].mul_((1.0 - b_new**step) / (1.0 - b_old**step))
        group["betas"] = new_betas

    def _refresh_state(self) -> None:
        """Phase 1: Adam EMAs for every param (state may grow keys later)."""
        for group, p in self._iter_params():
            assert p.grad is not None, "Geon.step requires grads on all params"
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["m"] = torch.zeros_like(p)
                state["v"] = torch.zeros_like(p)
            b1, b2 = group["betas"]
            state["step"] += 1
            state["m"].lerp_(p.grad, 1.0 - float(b1))
            state["v"].mul_(float(b2)).addcmul_(p.grad, p.grad, value=1.0 - float(b2))

    def _bias_corrected(self, raw: Tensor, beta: float, step: int) -> Tensor:
        """Adam-style bias correction: ``raw / (1 - beta**step)``, undoing
        the zero-init bias in the ``m``/``v`` EMAs (see ``_refresh_state``).
        Shared by ``_signal`` (``m`` -> ``mhat``, via ``betas[0]``) and
        ``_direction``'s ``adamw`` branch (``v`` -> ``vhat``, via
        ``betas[1]``) -- same formula, different EMA/beta.
        """
        return raw.float() / (1.0 - beta**step)

    def _signal(self, p: torch.nn.Parameter) -> Tensor:
        """Bias-corrected momentum, or its Nesterov version if the group asks."""
        group = self.group_of(p)
        state = self.state[p]
        b1 = float(group["betas"][0])
        mhat = self._bias_corrected(state["m"], b1, state["step"])
        if group["nesterov"]:
            return torch.lerp(p.grad.float(), mhat, b1)
        return mhat

    def _direction(self, p: torch.nn.Parameter, kind: UpdateKind) -> Tensor:
        """Phase 2: unscaled update direction (float32); ``ΔW = -size · dir``."""
        signal = self._signal(p)
        if kind == "adamw":
            group = self.group_of(p)
            state = self.state[p]
            b2 = float(group["betas"][1])
            vhat = self._bias_corrected(state["v"], b2, state["step"])
            return signal / (vhat.sqrt() + float(group["eps"]))
        if kind == "muon":
            return _newton_schulz_polar(signal).float() * _aspect_scale(p)
        if isinstance(kind, tuple) and len(kind) == 2 and kind[0] == "svd_pow":
            return _svd_power(
                signal,
                float(kind[1]),
                s_min=self.s_min,
                s_max=self.s_max,
            ) * _aspect_scale(p)
        if isinstance(kind, tuple) and len(kind) == 3 and kind[0] == "svd_band":
            return _svd_band(
                signal,
                float(kind[1]),
                float(kind[2]),
            ) * _aspect_scale(p)
        raise ValueError(f"unknown update kind {kind!r}")

    def _kl_matched_size(
        self,
        params: list[torch.nn.Parameter],
        lr: float,
        directions: dict[torch.nn.Parameter, Tensor],
        *,
        model: torch.nn.Module,
        batches: list[tuple[Tensor, Tensor]],
    ) -> float:
        """Phase 3, ``kl_match``: one size for ``params`` matching a Muon probe.

        Reference: a joint Muon step at ``lr`` on exactly ``params`` (other
        weights untouched). Binary-search ``coeff`` in log-space so the actual
        ``directions`` at ``coeff·lr`` give the same token-mean output KL;
        returns ``coeff·lr``.

        ``coeff`` is cached per exact ``params`` group (see
        ``kl_match_cache_schedule``) -- a cache hit skips every probe here
        (the reference Muon one included) and returns ``coeff_cached * lr``
        against this call's own ``lr``, not a frozen one.
        """
        cache_key = frozenset(params)
        if (
            cache_key in self.kl_matched_coeffs
            and not self.kl_match_cache_schedule.should_do(self._step_count)
        ):
            return float(self.kl_matched_coeffs[cache_key] * lr)

        reference = {p: self._direction(p, "muon") for p in params}
        target_kl = _kl_of_deltas(model, batches, reference, scale=lr)
        if target_kl <= 0.0 or not math.isfinite(target_kl):
            warnings.warn(
                f"Geon._kl_matched_size: reference Muon probe's target_kl="
                f"{target_kl!r} is <= 0 or non-finite -- skipping the "
                f"KL-matched search and returning the unscaled lr ({lr!r})"
            )
            return float(lr)
        actual = {p: directions[p] for p in params}
        coeff, _kl = BinarySearch.bracketed_binary_search(
            lambda c: _kl_of_deltas(model, batches, actual, scale=c * lr),
            target_kl,
            init=self.kl_scale_init,
            iters=self.kl_search_iters,
            expand_factor=self.kl_search_expand_factor,
        )
        self.kl_matched_coeffs[cache_key] = coeff
        return float(coeff * lr)

    def _fro_matched_size(
        self,
        params: list[torch.nn.Parameter],
        lr: float,
        directions: dict[torch.nn.Parameter, Tensor],
    ) -> float:
        """Phase 3, ``fro_match``: one size matching a Muon probe's joint Frobenius norm.

        Reference: a joint Muon step at ``lr`` on exactly ``params``. The
        joint Frobenius norm (``sqrt(Σ ‖·‖_F²)``, i.e. ``params`` treated as
        one block-diagonal update) scales linearly with the size, so the
        matching scale is closed-form rather than searched.
        """
        target_fro = (
            sum(float(self._direction(p, "muon").float().norm() ** 2) for p in params)
            ** 0.5
        )
        actual_fro = (
            sum(float(directions[p].float().norm() ** 2) for p in params) ** 0.5
        )
        if actual_fro <= 0.0 or not math.isfinite(actual_fro):
            warnings.warn(
                f"Geon._fro_matched_size: actual_fro={actual_fro!r} is <= 0 "
                f"or non-finite -- skipping the fro-matched scale and "
                f"returning the unscaled lr ({lr!r})"
            )
            return float(lr)
        return float(target_fro / actual_fro * lr)

    def _op_matched_size(
        self,
        params: list[torch.nn.Parameter],
        lr: float,
        directions: dict[torch.nn.Parameter, Tensor],
    ) -> float:
        """Phase 3, ``op_match``: one size matching a Muon probe's joint operator norm.

        Reference: a joint Muon step at ``lr`` on exactly ``params``. The
        joint operator norm (``max ‖·‖_op`` across ``params`` — the operator
        norm of a block-diagonal update equals the max of its blocks') scales
        linearly with the size, so the matching scale is closed-form.
        """
        # pylint: disable=not-callable
        target_op = max(
            float(torch.linalg.matrix_norm(self._direction(p, "muon").float(), ord=2))
            for p in params
        )
        actual_op = max(
            float(torch.linalg.matrix_norm(directions[p].float(), ord=2))
            for p in params
        )
        if actual_op <= 0.0 or not math.isfinite(actual_op):
            warnings.warn(
                f"Geon._op_matched_size: actual_op={actual_op!r} is <= 0 "
                f"or non-finite -- skipping the op-matched scale and "
                f"returning the unscaled lr ({lr!r})"
            )
            return float(lr)
        return float(target_op / actual_op * lr)

    def _resolve_sizes(
        self,
        active: list[torch.nn.Parameter],
        sizings: Sequence[SizingEntry] | None,
        directions: dict[torch.nn.Parameter, Tensor],
        *,
        model: torch.nn.Module | None,
        batches: list[tuple[Tensor, Tensor]] | None,
    ) -> dict[torch.nn.Parameter, float]:
        """Phase 3: per-param size; every active param needs exactly one entry."""
        active_set = set(active)
        sizes: dict[torch.nn.Parameter, float] = {}
        for entry in sizings or ():
            kind, entry_params, lr = entry
            entry_params = [p for p in entry_params if p in active_set]
            for p in entry_params:
                if p in sizes:
                    raise ValueError("param covered by two sizing entries")
            if not entry_params:
                continue
            if kind == "learning_rate":
                for p in entry_params:
                    sizes[p] = float(lr)
            elif kind == "kl_match":
                if model is None or batches is None:
                    raise ValueError("kl_match sizing needs model= and batches=")
                size = self._kl_matched_size(
                    entry_params, float(lr), directions, model=model, batches=batches
                )
                for p in entry_params:
                    sizes[p] = size
            elif kind == "fro_match":
                size = self._fro_matched_size(entry_params, float(lr), directions)
                for p in entry_params:
                    sizes[p] = size
            elif kind == "op_match":
                size = self._op_matched_size(entry_params, float(lr), directions)
                for p in entry_params:
                    sizes[p] = size
            else:
                raise ValueError(f"unknown sizing kind {kind!r}")
        uncovered = [p for p in active if p not in sizes]
        if uncovered:
            raise ValueError(
                f"sizings must cover every non-skip param "
                f"({len(uncovered)} uncovered)"
            )
        return sizes

    # ---------------------------------------------------------------- step

    # pylint: disable=arguments-differ,arguments-renamed
    @torch.no_grad()
    def step(
        self,
        updates: dict[torch.nn.Parameter, UpdateKind],
        sizings: Sequence[SizingEntry] | None = None,
        *,
        model: torch.nn.Module | None = None,
        batches: list[tuple[Tensor, Tensor]] | None = None,
        closure=None,
    ) -> None:
        """One step; ``updates`` must name every param (see module docstring)."""
        if closure is not None:
            raise RuntimeError("closure not supported")
        all_params = [p for _g, p in self._iter_params()]
        missing = [p for p in all_params if p not in updates]
        if missing:
            raise ValueError(f"updates must cover every param ({len(missing)} missing)")

        self._refresh_state()

        active = [p for p in all_params if updates[p] != "skip"]
        directions = {p: self._direction(p, updates[p]) for p in active}
        sizes = self._resolve_sizes(
            active, sizings, directions, model=model, batches=batches
        )

        for p in active:
            wd_raw = float(self.group_of(p)["wd_raw"])
            if wd_raw != 0.0:
                p.mul_(1.0 - wd_raw)
            p.add_(directions[p].to(dtype=p.dtype), alpha=-sizes[p])

        self._step_count += 1
        if (
            self.param_sync_every > 0
            and self._step_count % self.param_sync_every == 0
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            world_size = dist.get_world_size()
            for _group, p in self._iter_params():
                dist.all_reduce(p, op=dist.ReduceOp.SUM)
                p.div_(world_size)
