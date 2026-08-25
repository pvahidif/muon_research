"""Shared curvature profiling machinery for a 2D (Muon-shaped) weight
matrix ``p``, decomposed into per-mode directions ``D_i`` -- used by
``muon_research/scripts/run_curv_profile.py`` (profiles a single
already-checkpointed model, once, with no training in progress).

Everything is conditioned on the current weight ``p`` (fixed, given).
Past that, two deliberately separate data pools feed everything else:

- **Pool A** determines ``D_i``/``sigma_i`` themselves (via
  ``profiling_tensor``/``decompose_matrix``) -- it's whatever batch(es)
  produced ``p.grad`` (for ``profile_source="grad"``/``"signal"``) or
  just training history (``"weight"``/``"prev_momentum"``).
- **Pool B** is the data every other quantity below is estimated from --
  the ``pool_b_batches`` argument every function below takes. If the
  caller has a genuinely independent held-out batch
  (``ProfileConfig.profile_batch_size``), that's pool B. Otherwise, pool B
  *is* pool A's own batch, reused -- a deliberate approximation, modeled
  as independent rather than given its own joint model with pool A.

Every quantity below only ever reads ``pool_b_batches``, never ``p.grad``/
``p``/optimizer state directly.

Everything here is model/optimizer-construction-agnostic: it operates on
whatever ``(model, optimizer, param)`` the caller already has (a ``Geon``
optimizer, a ``GPT``/``GPL`` model, real gradients already populated on
``model.parameters()``). It's also ``TrainConfig``-schema-agnostic --
every function here that needs one only reads ``.batch_size``/``.seq_len``,
so any object with those two attributes works.

For each profiled matrix ``p``, given its profiling tensor ``M`` (see
``profiling_tensor``):

1. Decompose ``M = Σ_i sigma_i D_i``, ``D_i = u_i v_iᵀ`` (unit Frobenius norm,
   mutually orthogonal), per ``ProfileConfig.profile_decomposition`` -- see
   ``decompose_matrix``.
2. If ``compute_gamma``: the exact double-backward-HVP decomposition
   ``H_W[D_i] = Σ_j gamma_ji D_j + alpha_i Q_i`` (``gamma``/``gamma_diag``,
   ``alpha``, ``Q_i``, ``eta_perp``, ``gamma_perp_diag``) -- see
   ``hvp_curvatures``.
3. If ``compute_phi``: raw per-example samples of ``<D_i, grad>`` (and,
   if ``compute_gamma`` also ran, ``<Q_i, grad>``) -- see
   ``sample_gradient_projections``. Statistics (mean/variance/...) are
   computed later, from these raw samples, not here.

Assembled by ``profile_matrix`` and saved (via ``run_profile_capture``) to
``<run_path>/profiles/step_<iter>/svd_curv.pt``:

- ``iter_num`` -- this checkpoint's own step.
- ``optim`` -- caller-supplied label (e.g. "muon"/"adamw"), or ``None``.
- ``profile_source`` / ``profile_decomposition`` -- the ``ProfileConfig``
  choices this payload was produced with.
- ``matrix_names`` -- the profiled params' full dotted names.
- ``matrices`` -- ``{name: {...}}``, one entry per profiled 2D weight, each
  shaped by ``r`` = however many modes were kept (``max_modes``, or the
  tensor's own rank if smaller):

  - ``sigma`` -- ``sigma_i``, shape ``(r,)``. Pool A only (the
    ``M = Σ_i sigma_i D_i`` decomposition's own singular values, or
    row/column norms for ``row_wise``/``col_wise``) -- independent of
    Pool B.
  - ``gamma`` -- ``gamma[i,j] = <D_i, H_W[D_j]>``, shape ``(r, r)``.
    ``None`` unless ``compute_gamma``. Extrapolated up to
    ``train_config.batch_size`` tokens (one real batch's worth of summed
    loss), regardless of how many tokens ``pool_b_batches`` actually
    covered -- see ``hvp_curvatures``.
  - ``gamma_diag`` -- ``gamma``'s own diagonal, shape ``(r,)`` -- curvature
    along each ``D_i`` itself. Same extrapolated scale as ``gamma``.
  - ``alpha`` -- ``alpha[j] = ||H_W[D_j] - Σ_i gamma[i,j] D_i||_F``, shape
    ``(r,)`` -- the leftover HVP's own size. Same extrapolated scale as
    ``gamma``.
  - ``eta_perp`` -- ``eta_perp[i,j] = <Q_i, Q_j>``, shape ``(r, r)`` -- a
    cosine similarity in ``[-1, 1]`` between the unit-norm leftover
    directions (``Q_i`` itself is NOT saved, too large -- only quantities
    derived from it). Scale-free; no extrapolation applies.
  - ``gamma_perp_diag`` -- ``gamma_perp_diag[j] = <Q_j, H_W[Q_j]>``, shape
    ``(r,)`` -- the same kind of quantity as ``gamma_diag``, but along
    ``Q_j`` instead of ``D_j``. Same extrapolated scale as ``gamma``.
  - ``phi_samples`` -- raw per-example ``phi_i = <D_i, ∇_W L(example)>``,
    shape ``(r, n_examples)``. ``None`` unless ``compute_phi``. NOT
    extrapolated: each value is one example's own gradient projection, at
    that example's own scale, not the real batch's -- see
    ``sample_gradient_projections`` for the reconstruction factor.
  - ``phi_perp_samples`` -- ``phi_perp_i = <Q_i, ∇_W L(example)>`` from
    that same per-example gradient, shape ``(r, n_examples)``. ``None``
    unless both ``compute_phi`` AND ``compute_gamma`` ran (needs that
    pass's own ``Q_i``).
  - ``real_batch_examples`` -- ``train_config.batch_size /
    train_config.seq_len``, a fixed config-derived count, not measured --
    how many examples make up one real training batch. The factor
    ``phi_samples``/``phi_perp_samples`` need their MEAN multiplied by
    (not summed, not divided) to estimate the real batch's own
    ``<D_i, grad>``/``<Q_i, grad>``.
  - ``lr`` -- this param's current learning rate
    (``optimizer.group_of(p)["lr"]``) at profiling time; a hyperparameter
    value, not a measured quantity.
- ``gradient_scale_fixed`` -- always ``True`` (every quantity above is
  already at one fixed gradient scale, not one that drifts across the
  saved fields).
"""

# pylint: disable=all

import os
from dataclasses import MISSING, dataclass, fields
from fnmatch import fnmatch

import torch
import torch.distributed as dist
import yaml
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

from muon_research.fork import _makedirs_robust
from muon_research.model import GPL, GPT
from muon_research.optim.geon import Geon

########################################
#                Config                #
########################################


@dataclass
class ProfileConfig:
    profile_source: str  # "weight" | "signal" | "grad" | "prev_momentum"
    # fnmatch-style glob patterns (matched against each param's full dotted
    # name, e.g. "blocks.3.attn.q.weight") selecting which weights to
    # profile. None (default) = every 2D Muon-type block weight. Use
    # wildcards to span layers, e.g. "blocks.*.attn.q.weight".
    param_name_patterns: list[str] | None = None
    # Whether to compute gamma/gamma_diag/alpha/eta_perp/gamma_perp_diag
    # (see hvp_curvatures) / phi_samples,phi_perp_samples (see
    # sample_gradient_projections). Skipping either saves real time:
    # compute_gamma needs a double-backward HVP per mode, and
    # compute_phi is the priciest of all -- one forward+backward per
    # *training example* (not just per microbatch). phi_perp_samples is
    # only produced when compute_gamma also ran (it needs that pass's
    # own Q_i); with compute_gamma off, compute_phi still gives you
    # phi_samples alone.
    compute_gamma: bool = True
    compute_phi: bool = True
    max_modes: int | None = None
    # How to decompose each profiled matrix M into M = Σ_i sigma_i D_i (unit
    # Frobenius norm, mutually orthogonal D_i): "svd" (economy SVD, the
    # default — D_i = outer(u_i, v_i) from torch.linalg.svd), "row_wise"
    # (D_i = outer(b_i, row_i/‖row_i‖) — mode i is M's own i-th row), or
    # "col_wise" (D_i = outer(col_i/‖col_i‖, b_i) — mode i is M's own i-th
    # column). b_i is the standard basis (row/column) selector.
    profile_decomposition: str = "svd"
    # Pool B (see this module's own docstring). None (default): reuse
    # pool A's own batch(es), modeled as independent of it. If set: at
    # startup, set aside a FIXED batch of this many tokens from the val
    # data (right after the val_size block, so it never overlaps with
    # eval) and reuse that same batch for every profiling event instead --
    # a genuinely independent sample from pool A.
    profile_batch_size: int | None = None
    # Only meaningful when profile_batch_size is set. False (default): that
    # val batch is carved out once and reused, frozen, for every profiling
    # event. True: instead of the one frozen batch, re-derive a fresh,
    # deterministic slice of that same val stream for each profiling event,
    # keyed off its own iter_num -- so pool B varies from event to event
    # (still disjoint from eval, still reproducible) rather than staying
    # fixed.
    profile_batch_resample: bool = False
    # Override this checkpoint's own train_config.mbs (the microbatch size
    # each forward+backward chunk uses -- see run_curv_profile.py's
    # _split_into_microbatches) for this profiling run only. None (default)
    # keeps the checkpoint's own saved value. Lower this to trade speed for
    # peak GPU memory during hvp_curvatures'/sample_gradient_projections'
    # forward+backward passes, independent of whatever mbs the original
    # training run used -- must still evenly divide batch_size (and
    # profile_batch_size, if set).
    mbs: int | None = None

    def __post_init__(self):
        if self.profile_source not in ("weight", "signal", "grad", "prev_momentum"):
            raise ValueError(
                f"profile_source must be one of 'weight', 'signal', 'grad', "
                f"'prev_momentum', got {self.profile_source!r}"
            )
        if self.profile_decomposition not in ("svd", "row_wise", "col_wise"):
            raise ValueError(
                f"profile_decomposition must be 'svd', 'row_wise', or "
                f"'col_wise', got {self.profile_decomposition!r}"
            )
        if self.max_modes is not None:
            self.max_modes = int(self.max_modes)
        if self.param_name_patterns is not None:
            self.param_name_patterns = [str(c) for c in self.param_name_patterns]
        if self.profile_batch_size is not None:
            self.profile_batch_size = int(self.profile_batch_size)
            if self.profile_batch_size <= 0:
                raise ValueError(
                    f"profile_batch_size must be positive, got "
                    f"{self.profile_batch_size!r}"
                )
        if self.profile_batch_resample and self.profile_batch_size is None:
            raise ValueError("profile_batch_resample requires profile_batch_size")
        if self.mbs is not None:
            self.mbs = int(self.mbs)
            if self.mbs <= 0:
                raise ValueError(f"mbs must be positive, got {self.mbs!r}")


def load_profile_config(path: str) -> ProfileConfig:
    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    payload = payload["profile"]
    field_names = {field_.name for field_ in fields(ProfileConfig)}
    required = {
        field_.name
        for field_ in fields(ProfileConfig)
        if field_.default is MISSING and field_.default_factory is MISSING
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"config {path!r} is missing profile keys: {sorted(missing)}")
    unknown = payload.keys() - field_names
    if unknown:
        raise ValueError(f"config {path!r} has unknown profile keys: {sorted(unknown)}")
    return ProfileConfig(**payload)


def select_matrix_params(
    matrix_params_named: list[tuple[str, torch.nn.Parameter]],
    param_name_patterns: list[str] | None,
) -> list[tuple[str, torch.nn.Parameter]]:
    """Filter to just the params whose full name matches one of
    ``param_name_patterns`` (fnmatch globs, e.g. "blocks.*.attn.q.weight"),
    or every 2D block param if ``param_name_patterns`` is None."""
    if param_name_patterns is None:
        return matrix_params_named
    selected = [
        (name, p)
        for name, p in matrix_params_named
        if any(fnmatch(name, pattern) for pattern in param_name_patterns)
    ]
    unmatched = [
        pattern
        for pattern in param_name_patterns
        if not any(fnmatch(name, pattern) for name, _p in matrix_params_named)
    ]
    if unmatched:
        raise ValueError(f"param_name_patterns matched no 2D block param: {unmatched}")
    return selected


########################################
#          Profiling internals         #
########################################


def economy_svd(M: torch.Tensor, max_modes: int | None):
    # pylint: disable=not-callable
    U, sigma, Vh = torch.linalg.svd(M.float(), full_matrices=False)
    r = int(sigma.numel())
    if max_modes is not None:
        r = min(r, int(max_modes))
    return U[:, :r].contiguous(), sigma[:r].contiguous(), Vh[:r, :].contiguous()


def row_wise_decomposition(M: torch.Tensor, max_modes: int | None):
    """``M = Σ_i sigma_i D_i``, ``D_i = outer(b_i, row_i / ‖row_i‖)`` — mode
    ``i`` is just ``M``'s own ``i``-th row, Frobenius-normalized, placed in
    a matrix that's zero everywhere else (``sigma_i = ‖row_i‖``, the row's
    own norm; ``b_i`` the standard basis row-selector). Rows are disjoint,
    so this is already an exact, Frobenius-orthogonal decomposition — no
    factorization needed. Modes are sorted by descending ``sigma_i``
    (mirrors SVD's descending singular-value order) so ``max_modes``
    truncation keeps the largest-norm rows.
    """
    M = M.float()
    nrow = M.shape[-2]
    row_norms = M.norm(dim=-1)
    order = torch.argsort(row_norms, descending=True)
    r = nrow if max_modes is None else min(nrow, int(max_modes))
    order = order[:r]
    sigma = row_norms[order].contiguous()
    Vh = (M[order] / sigma.clamp_min(1e-12).unsqueeze(-1)).contiguous()
    U = torch.zeros(nrow, r, dtype=M.dtype, device=M.device)
    U[order, torch.arange(r, device=M.device)] = 1.0
    return U.contiguous(), sigma, Vh


def col_wise_decomposition(M: torch.Tensor, max_modes: int | None):
    """``M = Σ_i sigma_i D_i``, ``D_i = outer(col_i / ‖col_i‖, b_i)`` — the
    column-wise mirror of ``row_wise_decomposition``: mode ``i`` is just
    ``M``'s own ``i``-th column, Frobenius-normalized, placed in a matrix
    that's zero everywhere else (``sigma_i = ‖col_i‖``, the column's own
    norm; ``b_i`` the standard basis column-selector). Columns are
    disjoint, so this is already an exact, Frobenius-orthogonal
    decomposition — no factorization needed. Modes are sorted by
    descending ``sigma_i`` so ``max_modes`` truncation keeps the
    largest-norm columns.
    """
    M = M.float()
    ncol = M.shape[-1]
    col_norms = M.norm(dim=-2)
    order = torch.argsort(col_norms, descending=True)
    r = ncol if max_modes is None else min(ncol, int(max_modes))
    order = order[:r]
    sigma = col_norms[order].contiguous()
    U = (M[:, order] / sigma.clamp_min(1e-12).unsqueeze(-2)).contiguous()
    Vh = torch.zeros(r, ncol, dtype=M.dtype, device=M.device)
    Vh[torch.arange(r, device=M.device), order] = 1.0
    return U, sigma, Vh.contiguous()


def decompose_matrix(M: torch.Tensor, decomposition: str, max_modes: int | None):
    """``M ≈ Σ_i sigma_i D_i``, ``D_i = outer(u_i, v_i)`` — dispatches on
    ``decomposition`` (``"svd"`` | ``"row_wise"`` | ``"col_wise"``).
    Everything downstream (``hvp_curvatures``, the per-mode KL match) only
    ever consumes ``U`` (columns ``u_i``) / ``sigma`` (``sigma_i``) / ``Vh``
    (rows ``v_i``) in this outer-product form, so it's decomposition-agnostic.
    """
    if decomposition == "svd":
        return economy_svd(M, max_modes)
    if decomposition == "row_wise":
        return row_wise_decomposition(M, max_modes)
    if decomposition == "col_wise":
        return col_wise_decomposition(M, max_modes)
    raise ValueError(f"unknown profile_decomposition {decomposition!r}")


def _updated_signal(p: torch.nn.Parameter, optimizer: Geon) -> torch.Tensor:
    """Read-only mirror of ``Geon._refresh_state`` + ``Geon._signal`` *as if*
    this step's ``optimizer.step()`` had already folded ``p.grad`` into
    momentum — without touching ``optimizer.state``, so the real
    ``optimizer.step()`` right after profiling (if any -- irrelevant for a
    one-shot checkpoint profile, which never calls step() at all) is
    unaffected.

    ``_refresh_state`` does ``state["m"].lerp_(p.grad, 1 - beta1)`` and
    ``state["step"] += 1`` *before* ``_signal`` bias-corrects it; this
    computes that same next ``m``/``step`` into locals instead of mutating
    ``state``, then applies the identical bias-correction (and Nesterov
    blend, if the group uses it) on top. Requires ``p.grad`` to already be
    set (a real, already-all-reduced gradient at the current weights).
    """
    group = optimizer.group_of(p)
    state = optimizer.state[p]
    b1 = float(group["betas"][0])
    grad = p.grad.detach().float()
    if len(state) == 0:
        m_new = grad * (1.0 - b1)
        step_new = 1
    else:
        m_new = state["m"].float().lerp(grad, 1.0 - b1)
        step_new = int(state["step"]) + 1
    mhat = m_new / (1.0 - b1**step_new)
    if group["nesterov"]:
        return torch.lerp(grad, mhat, b1)
    return mhat


def _prev_momentum(p: torch.nn.Parameter, optimizer: Geon) -> torch.Tensor:
    """Bias-corrected momentum as of the LAST completed ``step()`` call --
    i.e. ``Geon._signal`` without folding in the current ``p.grad`` (no
    Nesterov blend either, since that would reintroduce the current
    gradient) and without mutating ``optimizer.state``. A never-stepped
    state (fresh/empty, e.g. a step-0 checkpoint saved before any
    ``step()`` call) has no prior momentum at all -- ``state["m"]`` would
    still be all zero and its bias-correction denominator
    (``1 - b1**0 == 0``) is undefined, so that case returns zeros directly
    rather than ``0/0``.
    """
    group = optimizer.group_of(p)
    state = optimizer.state[p]
    if len(state) == 0:
        return torch.zeros_like(p, dtype=torch.float32)
    b1 = float(group["betas"][0])
    return state["m"].float() / (1.0 - b1 ** state["step"])


def profiling_tensor(
    p: torch.nn.Parameter, optimizer: Geon, profile_source: str
) -> torch.Tensor:
    """The tensor whose SVD defines the profiled directions.

    ``"weight"``: the raw weight. ``"grad"``: the raw (already-all-reduced)
    gradient, unmodified. ``"signal"``: the signal Geon's real
    ``optimizer.step()`` would actually use for ``p`` this step -- i.e.
    bias-corrected momentum (Nesterov-blended if the group uses it) *with
    this step's gradient already folded in*, mirrors
    ``Geon._refresh_state``/``_signal`` combined (see ``_updated_signal``),
    computed read-only so it doesn't touch ``optimizer.state``.
    ``"prev_momentum"``: the bias-corrected momentum as of the last
    completed ``step()`` call, i.e. the same computation *without* folding
    in the current gradient (see ``_prev_momentum``) -- the "before"
    counterpart to ``"signal"``'s "after". All but ``"weight"`` require
    ``p.grad`` to already be a real, already-all-reduced gradient at the
    current weights (from a normal training step, or -- for a one-shot
    checkpoint profile -- one forward+backward the caller runs first).
    """
    if profile_source == "weight":
        return p.detach().float()
    if profile_source == "grad":
        return p.grad.detach().float()
    if profile_source == "prev_momentum":
        return _prev_momentum(p, optimizer).detach().float()
    if profile_source == "signal":
        return _updated_signal(p, optimizer).detach().float()
    raise ValueError(f"unknown profile_source {profile_source}")


def hvp_curvatures(
    model: GPT | GPL,
    W: torch.nn.Parameter,
    U: torch.Tensor,
    Vh: torch.Tensor,
    pool_b_batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    real_batch_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact double-backward-HVP decomposition of each mode's own Hessian
    action: ``H_W[D_j] = Σ_i gamma_ij D_i + alpha_j Q_j``, where
    ``Σ_i gamma_ij D_i`` is what ``span{D_i}`` already reconstructs of it
    and ``Q_j`` (unit Frobenius norm) is what's left over. One backward
    pass per mode ``j`` gives ``H_W[D_j]`` (the expensive part), which
    already projects onto every ``D_i`` at once -- so ``gamma`` (the full
    ``r x r`` matrix) costs the same as its diagonal alone
    (``gamma_diag[i] = gamma[i, i]``, the curvature along ``D_i`` itself).

    Returns ``(gamma, alpha, eta_perp, gamma_perp_diag, Q)``:

    - ``gamma[i, j] = <D_i, H_W[D_j]>``.
    - ``alpha[j] = ||H_W[D_j] - Σ_i gamma[i,j] D_i||_F`` (the leftover's
      size).
    - ``Q[j]``: that leftover, Frobenius-normalized to unit norm --
      returned (not part of the saved payload -- too large) so a caller
      can project other things onto it, e.g. ``sample_gradient_projections``.
    - ``eta_perp[i, j] = <Q_i, Q_j>`` -- already a cosine similarity in
      ``[-1, 1]`` (every ``Q_i`` is unit-norm by construction): near 0
      off-diagonal means the leftovers are close to orthogonal.
    - ``gamma_perp_diag[j] = <Q_j, H_W[Q_j]>`` -- the same kind of
      quantity ``gamma_diag`` is, but along ``Q_j`` instead of ``D_j``.
      Needs one more HVP per mode (``Q_j`` only exists once every
      ``H_W[D_j]`` has already been seen, so this is a second pass over
      ``pool_b_batches``, not reusing the first pass's graph).

    Everything here is a finite-sample estimate driven entirely by
    ``pool_b_batches`` (``U``/``Vh``, hence ``D_i``/``sigma_i``, are already
    given by the time this is called -- see this module's own docstring).
    Summed over microbatches and all-reduced across ranks, then scaled by
    ``real_batch_tokens / n_tokens`` to extrapolate from however many
    tokens ``pool_b_batches`` actually covers up to a real training
    batch's scale (a no-op when it already is one). ``eta_perp`` needs no
    such scaling (both its factors are already scale-invariant unit
    vectors); ``Q`` doesn't either (normalizing a residual by its own norm
    cancels whatever scale it was measured at).

    Calls ``model.zero_grad()`` internally -- caller is responsible for
    snapshotting/restoring real training grads.
    """
    r = U.shape[1]
    n_mb = len(pool_b_batches)
    gamma = torch.zeros(r, r, dtype=torch.float32, device=device)
    U = U.to(device=device, dtype=torch.float32)
    Vh = Vh.to(device=device, dtype=torch.float32)
    hvp_sum = torch.zeros(r, *W.shape, dtype=torch.float32, device=device)
    n_tokens = sum(int(y.numel()) for _x, y in pool_b_batches) * dist.get_world_size()
    was_training = model.training
    model.eval()
    with sdpa_kernel(SDPBackend.MATH):
        with tqdm(
            total=r * n_mb,
            desc="hvp_curvatures: H[D_j]",
            leave=False,
            disable=dist.get_rank() != 0,
        ) as pbar:
            for x, y in pool_b_batches:
                model.zero_grad(set_to_none=True)
                loss_mb = model(x, y)
                (gW,) = torch.autograd.grad(loss_mb, W, create_graph=True)
                gW = gW.float()
                for j in range(r):
                    u, v = U[:, j], Vh[j, :]
                    gdot = torch.dot(u, gW @ v)
                    (hvp,) = torch.autograd.grad(gdot, W, retain_graph=(j + 1 < r))
                    hvp = hvp.float()
                    # <D_i, H[D_j]> for every i, all from this one HVP: the
                    # diagonal of U^T @ H[D_j] @ Vhᵀ.
                    gamma[:, j] += torch.diagonal(U.T @ hvp @ Vh.T)
                    hvp_sum[j] += hvp
                    pbar.update(1)
                model.zero_grad(set_to_none=True)
                del loss_mb, gW
    model.train(was_training)
    dist.all_reduce(gamma, op=dist.ReduceOp.SUM)
    dist.all_reduce(hvp_sum, op=dist.ReduceOp.SUM)
    scale = float(real_batch_tokens) / float(n_tokens)

    # Uses gamma's own (raw, pre-scale/pre-symmetrize) column j -- the
    # actual <D_i, H[D_j]> this hvp_sum[j] was measured to have, so the
    # reconstruction is self-consistent with it.
    resid = torch.empty_like(hvp_sum)
    for j in range(r):
        recon = (U * gamma[:, j].unsqueeze(0)) @ Vh
        resid[j] = hvp_sum[j] - recon
    resid *= scale

    alpha = torch.linalg.matrix_norm(resid, ord="fro").clamp_min(1e-12)
    Q = resid / alpha.view(r, *([1] * (resid.dim() - 1)))
    Q_flat = Q.reshape(r, -1)
    eta_perp = Q_flat @ Q_flat.T

    # gamma_perp_diag[j] = <Q_j, H[Q_j]> -- a second, independent HVP
    # pass: Q_j only exists once every H[D_j] has already been seen.
    gamma_perp_sum = torch.zeros(r, dtype=torch.float32, device=device)
    was_training = model.training
    model.eval()
    with sdpa_kernel(SDPBackend.MATH):
        with tqdm(
            total=r * n_mb,
            desc="hvp_curvatures: H[Q_j]",
            leave=False,
            disable=dist.get_rank() != 0,
        ) as pbar:
            for x, y in pool_b_batches:
                model.zero_grad(set_to_none=True)
                loss_mb = model(x, y)
                (gW,) = torch.autograd.grad(loss_mb, W, create_graph=True)
                gW = gW.float()
                for j in range(r):
                    gdot_q = torch.sum(Q[j] * gW)
                    (hvp_q,) = torch.autograd.grad(gdot_q, W, retain_graph=(j + 1 < r))
                    gamma_perp_sum[j] += torch.sum(Q[j] * hvp_q.float())
                    pbar.update(1)
                model.zero_grad(set_to_none=True)
                del loss_mb, gW
    model.train(was_training)
    dist.all_reduce(gamma_perp_sum, op=dist.ReduceOp.SUM)
    gamma_perp_diag = gamma_perp_sum * scale

    gamma *= scale
    # Analytically gamma[i,j] == gamma[j,i] (H is symmetric), but two
    # independently computed HVPs won't be bit-identical -- symmetrize.
    gamma = 0.5 * (gamma + gamma.T)
    return gamma, alpha, eta_perp, gamma_perp_diag, Q


def sample_gradient_projections(
    model: GPT | GPL,
    W: torch.nn.Parameter,
    U: torch.Tensor,
    Vh: torch.Tensor,
    Q: torch.Tensor | None,
    pool_b_batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Raw per-*example* gradient projections (not per-microbatch):
    ``phi_i = <D_i, ∇_W L(example)>`` for every example in
    ``pool_b_batches``, and -- if ``Q`` is given (the residual directions
    from ``hvp_curvatures``; ``None`` skips this) -- also
    ``phi_perp_i = <Q_i, ∇_W L(example)>``, from that same per-example
    gradient (no extra backward pass). Returns raw samples, shape
    ``(r, n_examples)``, gathered across every rank's own examples
    (data-parallel: each rank sees different ones) -- statistics
    (mean/variance/...) are for a caller to compute later, not here.

    ``grad`` is the raw, un-normalized gradient of one example's own
    SUM-over-its-``seq_len``-tokens loss (``model(x, y)`` uses
    ``reduction="sum"``) -- one additive term of the real training
    gradient, same scale as the ``profile_source="signal"`` tensor. ``D_i``/
    ``Q_i`` are unit-Frobenius-norm and carry no scale of their own, so
    ``phi_i``/``phi_perp_i`` inherit ``grad``'s scale exactly: the
    per-example (not per-batch, not per-token) projection.

    The real per-step gradient is the *unscaled sum* of every example's own
    gradient in the batch (``run_optim_rules.py``'s training loop
    accumulates ``loss.backward()`` across microbatches and all-reduces
    ``p.grad`` with SUM, never dividing by ``batch_size`` -- only the
    logged ``train_loss`` gets that division, not the gradient actually
    used for the step). So these raw per-example samples are *not* yet at
    the real batch's own scale: an unbiased estimate of the real batch's
    ``<D_i, grad>`` is ``real_batch_examples * mean(phi_samples[i])`` (and
    likewise for ``phi_perp``), where ``real_batch_examples =
    train_config.batch_size / train_config.seq_len`` (see
    ``profile_matrix``) is how many examples actually make up one real
    batch -- summing ``real_batch_examples`` i.i.d. terms, not averaging
    them. A caller after variance, not just the mean, should scale by
    ``real_batch_examples`` too (not its square root): the variance of a
    sum of i.i.d. terms is ``real_batch_examples`` times the per-term
    variance.

    Calls ``model.zero_grad()`` internally -- caller is responsible for
    snapshotting/restoring real training grads.
    """
    r = U.shape[1]
    U = U.to(device=device, dtype=torch.float32)
    Vh = Vh.to(device=device, dtype=torch.float32)
    has_perp = Q is not None
    if has_perp:
        Q_flat = Q.to(device=device, dtype=torch.float32).reshape(r, -1)
    n_local = sum(int(x.shape[0]) for x, _y in pool_b_batches)
    phi_local = torch.empty(r, n_local, dtype=torch.float32, device=device)
    phi_perp_local = (
        torch.empty(r, n_local, dtype=torch.float32, device=device)
        if has_perp
        else None
    )
    was_training = model.training
    model.eval()
    idx = 0
    with sdpa_kernel(SDPBackend.MATH):
        with tqdm(
            total=n_local,
            desc="sample_gradient_projections",
            leave=False,
            disable=dist.get_rank() != 0,
        ) as pbar:
            for x, y in pool_b_batches:
                for j in range(x.shape[0]):
                    model.zero_grad(set_to_none=True)
                    loss_ex = model(x[j : j + 1], y[j : j + 1])
                    (grad,) = torch.autograd.grad(loss_ex, W)
                    grad = grad.float()
                    phi_local[:, idx] = torch.diagonal(U.T @ grad @ Vh.T)
                    if has_perp:
                        phi_perp_local[:, idx] = Q_flat @ grad.reshape(-1)
                    idx += 1
                    pbar.update(1)
    model.zero_grad(set_to_none=True)
    model.train(was_training)

    # Every rank profiles the same number of examples (same mbs, same
    # pool_b_batches shape everywhere), so an equal-shaped all_gather is
    # safe -- concatenate to get every rank's examples in one tensor.
    world_size = dist.get_world_size()
    phi_gathered = [torch.empty_like(phi_local) for _ in range(world_size)]
    dist.all_gather(phi_gathered, phi_local)
    phi_samples = torch.cat(phi_gathered, dim=1).detach().cpu()

    phi_perp_samples = None
    if has_perp:
        phi_perp_gathered = [
            torch.empty_like(phi_perp_local) for _ in range(world_size)
        ]
        dist.all_gather(phi_perp_gathered, phi_perp_local)
        phi_perp_samples = torch.cat(phi_perp_gathered, dim=1).detach().cpu()

    return phi_samples, phi_perp_samples


def profile_matrix(
    model: GPT | GPL,
    optimizer: Geon,
    p: torch.nn.Parameter,
    pool_b_batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    profile_config: ProfileConfig,
    train_config,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Decompose the profiling tensor for ``p`` (per
    ``profile_config.profile_decomposition``) and compute whatever
    ``profile_config`` asks for on top; return ``(payload, U, Vh)``.

    ``gamma``/``gamma_diag``/``alpha``/``eta_perp``/``gamma_perp_diag``
    (see ``hvp_curvatures``) are ``None`` unless ``compute_gamma``.
    ``phi_samples``/``phi_perp_samples`` (see
    ``sample_gradient_projections``) are ``None`` unless ``compute_phi``
    -- and ``phi_perp_samples`` specifically also needs
    ``compute_gamma`` (it projects onto ``Q_i``, which only exists as
    a byproduct of that pass).

    ``train_config`` is duck-typed -- only ``.batch_size``/``.seq_len`` are
    read -- so either caller's own ``TrainConfig`` class works.
    ``gamma``/``gamma_diag`` are extrapolated up to ``train_config``'s real
    per-step token scale regardless of how many tokens ``pool_b_batches``
    actually covers (see ``hvp_curvatures``); ``phi_samples``/
    ``phi_perp_samples`` are raw, per-example, and NOT extrapolated the
    same way -- ``real_batch_examples`` is saved alongside them precisely
    because that extrapolation is different (a sum over i.i.d. examples,
    not a token-count rescale) and is left to a later step: the real
    batch's own ``<D_i, grad>`` is estimated as ``real_batch_examples *
    mean(phi_samples[i])`` (see ``sample_gradient_projections``).
    """
    real_batch_tokens = int(train_config.batch_size)
    real_batch_examples = train_config.batch_size / train_config.seq_len
    M = profiling_tensor(p, optimizer, profile_config.profile_source)
    U, sigma, Vh = decompose_matrix(
        M, profile_config.profile_decomposition, profile_config.max_modes
    )
    lr = float(optimizer.group_of(p)["lr"])

    gamma = gamma_diag = alpha = eta_perp = gamma_perp_diag = None
    Q = None
    if profile_config.compute_gamma:
        gamma, alpha, eta_perp, gamma_perp_diag, Q = hvp_curvatures(
            model, p, U, Vh, pool_b_batches, device, real_batch_tokens
        )
        gamma = gamma.detach().cpu()
        gamma_diag = torch.diagonal(gamma).clone()
        alpha = alpha.detach().cpu()
        eta_perp = eta_perp.detach().cpu()
        gamma_perp_diag = gamma_perp_diag.detach().cpu()

    phi_samples = phi_perp_samples = None
    if profile_config.compute_phi:
        phi_samples, phi_perp_samples = sample_gradient_projections(
            model, p, U, Vh, Q, pool_b_batches, device
        )
        phi_samples = phi_samples.detach().cpu()
        if phi_perp_samples is not None:
            phi_perp_samples = phi_perp_samples.detach().cpu()

    payload = {
        "sigma": sigma.detach().cpu(),
        "gamma": gamma,
        "gamma_diag": gamma_diag,
        "alpha": alpha,
        "eta_perp": eta_perp,
        "gamma_perp_diag": gamma_perp_diag,
        "phi_samples": phi_samples,
        "phi_perp_samples": phi_perp_samples,
        "real_batch_examples": real_batch_examples,
        "lr": lr,
    }
    return payload, U, Vh


def _snapshot_grads(model: GPT | GPL) -> dict[int, torch.Tensor]:
    return {
        id(p): p.grad.detach().clone() for p in model.parameters() if p.grad is not None
    }


def _restore_grads(model: GPT | GPL, grads: dict[int, torch.Tensor]) -> None:
    for p in model.parameters():
        g = grads.get(id(p))
        p.grad = None if g is None else g.to(dtype=p.dtype, device=p.device)


def run_profile_capture(
    *,
    model: GPT | GPL,
    optimizer: Geon,
    matrix_params_named: list[tuple[str, torch.nn.Parameter]],
    pool_b_batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    profile_config: ProfileConfig,
    train_config,
    optim: str | None,
    iter_num: int,
    run_path: str,
    print0,
) -> None:
    """Profile every 2D matrix on pre-step weights; save one payload per step.

    Snapshots/restores every param's ``.grad`` **per matrix** (not once for
    the whole capture): ``hvp_curvatures`` calls ``model.zero_grad()``
    internally, which wipes gradients for the whole model, not just the
    matrix it's processing -- without a per-matrix restore, the next
    matrix's ``profiling_tensor(..., "signal")`` would see ``p.grad is
    None``.

    ``optim`` is purely descriptive (recorded in the saved payload, e.g.
    "muon"/"adamw" or a run name) -- pass ``None`` if there's no single
    natural label (e.g. a rules-based checkpoint with per-param update
    kinds).
    """
    matrices = {}
    for name, p in matrix_params_named:
        saved_grads = _snapshot_grads(model)
        print0(f"[curv] iter={iter_num} {name}", console=True)
        matrices[name], _U, _Vh = profile_matrix(
            model,
            optimizer,
            p,
            pool_b_batches,
            device,
            profile_config,
            train_config,
        )
        _restore_grads(model, saved_grads)

    if dist.get_rank() == 0:
        step_dir = os.path.join(run_path, "profiles", f"step_{iter_num:06d}")
        _makedirs_robust(step_dir)
        payload = {
            "iter_num": int(iter_num),
            "optim": optim,
            "profile_source": profile_config.profile_source,
            "profile_decomposition": profile_config.profile_decomposition,
            "matrix_names": [n for n, _p in matrix_params_named],
            "matrices": matrices,
            # gamma/gamma_diag are already extrapolated to train_config's
            # real per-step token scale (see profile_matrix); phi_samples/
            # phi_perp_samples are raw per-example values, not yet at the
            # real batch's scale -- multiply their mean by real_batch_examples
            # (saved per matrix) to estimate the real batch's own <D_i, grad>
            # (see sample_gradient_projections).
            "gradient_scale_fixed": True,
        }
        torch.save(payload, os.path.join(step_dir, "svd_curv.pt"))
        print0(
            f"[curv] iter={iter_num} done ({len(matrix_params_named)} matrices)",
            console=True,
        )
