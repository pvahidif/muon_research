# pylint: disable=all
from notebooks.utils import *
from muon_research.paths import *

from muon_research.scripts import analyze_curv_profile


def load_experiment_metrics(glob_pattern: str | list[str]) -> pl.DataFrame:
    """Load config.json + metrics.jsonl from every run dir matching ``glob_pattern``.

    ``glob_pattern`` is a single glob string, or a list of them — matches
    from every pattern are concatenated (duplicates, e.g. from overlapping
    patterns, are deduped).

    Each row is one metrics.jsonl entry (train or eval step), joined with all of
    that run's config fields, so no data is dropped up front.
    """
    patterns = [glob_pattern] if isinstance(glob_pattern, str) else list(glob_pattern)
    run_dirs = sorted({Path(p) for pattern in patterns for p in glob.glob(pattern)})
    rows = []
    for run_dir in run_dirs:
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.jsonl"
        if not config_path.exists() or not metrics_path.exists():
            continue

        with open(config_path) as f:
            config = json.load(f)

        with open(metrics_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                metric = json.loads(line)
                rows.append({"run_dir": str(run_dir), **config, **metric})
    assert len(rows) > 0, "read no data"
    df = pl.DataFrame(rows, infer_schema_length=1000000, strict=False)
    if "seed" not in df.columns:
        df = df.with_columns(seed=pl.lit(0).cast(pl.Int64))
    return df


def fit_loglog_ols(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS fit ``y = intercept + slope * x``; returns coeffs + r2/l1/l2 (MAE/RMSE)."""
    x = np.asarray(x)
    y = np.asarray(y)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "l1": float(np.mean(np.abs(resid))),
        "l2": float(np.sqrt(np.mean(resid**2))),
    }


def fit_loglog_quantile(
    x: np.ndarray, y: np.ndarray, tau: float = 0.5, iters: int = 100, eps: float = 1e-6
) -> dict:
    """Quantile (default: median, tau=0.5) regression ``y = intercept + slope * x``.

    No statsmodels/scipy in this env, so this is IRLS (iteratively reweighted
    least squares): each iteration solves a weighted OLS with weights
    ``tau/|resid|`` (resid>=0) or ``(1-tau)/|resid|`` (resid<0), which
    converges to the tau-quantile (least-absolute-deviation when tau=0.5)
    solution. Warm-started from the OLS fit.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    for _ in range(iters):
        resid = y - X @ beta
        w = np.where(resid >= 0, tau, 1.0 - tau) / np.maximum(np.abs(resid), eps)
        WX = X * w[:, None]
        beta = np.linalg.solve(WX.T @ X, WX.T @ y)
    resid = y - X @ beta
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "l1": float(np.mean(np.abs(resid))),
        "l2": float(np.sqrt(np.mean(resid**2))),
    }


def _iter_svd_curv_payloads(step_dir: str) -> list[dict]:
    """Every ``svd_curv.pt`` payload under one ``profiles/step_*`` dir.

    Normally that's just the single top-level file. When the run used
    ``profile.probe_num_batches`` (run_geon_curv_profile.py), profiling that
    step ran separately per fixed probe batch, and there's no top-level
    file — instead one lives under each ``probe_<j>/`` subdirectory, with
    its own ``probe_idx`` field already baked into the payload. Returned in
    ``probe_idx`` order (``[None]`` in the non-probe case).
    """
    top_level = os.path.join(step_dir, "svd_curv.pt")
    if os.path.exists(top_level):
        return [torch.load(top_level, map_location="cpu", weights_only=False)]
    return [
        torch.load(p, map_location="cpu", weights_only=False)
        for p in sorted(glob.glob(os.path.join(step_dir, "probe_*", "svd_curv.pt")))
    ]


# Per-mode payload fields ``load_curv_fits`` can regress one against
# another, and the one-letter quantity code (used in output column names)
# each maps to. ``l_i``/``curvatures``: diagonal HVP curvature. ``e_i``/
# ``noise_variance``: per-mode per-*example* gradient-projection variance
# (always non-negative). ``p_i``/``projection_mean``: per-mode per-example
# gradient-projection mean (signed) -- see ``projection_moments`` in
# run_geon_curv_profile.py for exactly what each measures. ``e``/``p`` are
# only present in a payload when that run had ``profile.compute_e`` on
# (``None`` otherwise, same as ``l`` with ``compute_l``). ``"s"`` (singular
# values themselves) is a valid quantity code too -- see
# ``_resolve_curv_qty`` -- but isn't in this map since it's always present,
# not gated behind any ``compute_*`` flag.
_CURV_FIT_FIELDS = {
    "l": "curvatures",
    "e": "noise_variance",
    "p": "projection_mean",
}


def _resolve_curv_qty(mat: dict, s_full: np.ndarray, qty: str) -> np.ndarray | None:
    """One named per-mode quantity out of a ``svd_curv.pt`` matrix payload,
    for ``load_curv_fits``'s regression pairs. ``"s"`` is ``s_full`` (the
    singular values, always present); anything else is looked up via
    ``_CURV_FIT_FIELDS`` and is ``None`` if that payload's ``compute_*``
    flag was off.
    """
    if qty == "s":
        return s_full
    val = mat.get(_CURV_FIT_FIELDS[qty])
    return None if val is None else val.numpy().astype(np.float64)


def _iter_curv_profile_matrices(run_dirs: list[Path]):
    """Shared walk behind ``load_curv_fits``/``load_curv_matrices``: yields
    ``(base_row, mat)`` for every profiled matrix across every
    (run, step, probe) -- ``base_row`` already has run_dir/config/matrix_name/
    iter_num/probe_idx/optim/profile_source merged in (``probe_idx`` is
    ``None`` for a normal, non-probe run); ``mat`` is that matrix's raw
    per-mode payload dict from ``svd_curv.pt``. Runs missing ``config.json``
    are skipped. Supports ``profile.profile_num_batches`` transparently via
    ``_iter_svd_curv_payloads``.
    """
    for run_dir in tqdm(run_dirs, desc="loading curv profile runs"):
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        with open(config_path) as f:
            config = json.load(f)

        for step_dir in sorted(glob.glob(str(run_dir / "profiles" / "step_*"))):
            for record in _iter_svd_curv_payloads(step_dir):
                for matrix_name, mat in record["matrices"].items():
                    base_row = {
                        "run_dir": str(run_dir),
                        **config,
                        "matrix_name": matrix_name,
                        "iter_num": int(record["iter_num"]),
                        "probe_idx": record.get("probe_idx"),
                        "optim": record.get("optim"),
                        "profile_source": record.get("profile_source"),
                    }
                    if "model_type" not in base_row:
                        base_row["model_type"] = "gpt"
                    yield base_row, mat


def load_curv_fits(
    glob_pattern: str | list[str],
    regressions: list[tuple[str, str]] = [("l", "s")],
) -> pl.DataFrame:
    """Load run_geon_curv_profile.py dumps + regress each ``(y, x)`` pair in
    ``regressions`` per (run, step, matrix[, probe]) -- e.g.
    ``regressions=[("l", "s"), ("e", "l")]`` fits ``l_i`` vs ``s_i`` (the
    original default) *and* ``e_i`` vs ``l_i``.

    ``glob_pattern`` is a single glob string, or a list of them — matches
    from every pattern are concatenated (duplicates deduped), same
    convention as ``load_experiment_metrics``.

    Each side of a pair is one of ``"l"`` (``curvatures``), ``"e"``
    (``noise_variance``), ``"p"`` (``projection_mean``), or ``"s"`` (the
    singular values themselves, always present); see ``_resolve_curv_qty``.
    For most pairs this is a log-log fit -- ``log|y| ~ log|x|``, via both
    OLS and median (tau=0.5) quantile regression -- contributing that
    pair's own ``{y}_on_{x}_{ols,qr}_{intercept,slope,r2,l1,l2}`` columns
    (coefficients + r2 + l1/l2 error -- mean absolute / root-mean-square
    residual -- plus ``{y}_on_{x}_n_modes_fit``, the number of modes that
    survived that pair's own filtering: finite, both sides ``!=0``).

    ``("p", "s")`` is the one exception: ``p_i`` is a signed per-example
    gradient-projection *mean* that legitimately crosses zero across modes,
    so a log-log power-law slope isn't the right model for it (unlike
    ``e_i``, always non-negative, or ``l_i``, whose sign is physically
    meaningful but rarely flips mode-to-mode). Instead it's fit as the
    ratio ``p_i / s_i`` itself (unlogged) against ``log(s_i)`` -- the same
    quantity as the ad hoc ``p/s`` vs ``log(s)`` scatter this mirrors (see
    curv_expr.ipynb) -- so ``slope``/``intercept`` read directly as how
    that ratio trends with scale.

    A pair missing from a given payload (either side's ``compute_*`` was
    off) just leaves that row's columns for it null rather than dropping
    the row -- so requesting multiple pairs together still returns every
    row *any* of them has data for. ``n_modes`` (unprefixed) is the raw
    per-matrix mode count *before* any pair-specific filtering.

    If the run used ``profile.probe_num_batches``, each probe batch was
    profiled (and saved) separately -- see ``_iter_svd_curv_payloads`` --
    so each one gets fit and returned as its own row here too, distinguished
    by the ``probe_idx`` column (``None`` for a normal, non-probe run).
    """
    valid = set(_CURV_FIT_FIELDS) | {"s"}
    for y_qty, x_qty in regressions:
        assert y_qty in valid and x_qty in valid, (
            f"unknown quantity in regression pair {(y_qty, x_qty)!r}, "
            f"each side must be one of {sorted(valid)}"
        )

    patterns = [glob_pattern] if isinstance(glob_pattern, str) else list(glob_pattern)
    run_dirs = sorted({Path(p) for pattern in patterns for p in glob.glob(pattern)})
    rows = []
    for base_row, mat in _iter_curv_profile_matrices(run_dirs):
        s_full = mat["singular_values"].numpy().astype(np.float64)
        row = {**base_row, "n_modes": int(s_full.size)}
        got_any = False
        for y_qty, x_qty in regressions:
            y_val = _resolve_curv_qty(mat, s_full, y_qty)
            x_val = _resolve_curv_qty(mat, s_full, x_qty)
            if y_val is None or x_val is None:
                continue
            keep = np.isfinite(x_val) & np.isfinite(y_val) & (x_val != 0) & (y_val != 0)
            xx, yy = x_val[keep], y_val[keep]
            if xx.size < 2:
                continue
            if (y_qty, x_qty) == ("p", "s"):
                x = np.log(xx)
                y = yy / xx
                y = np.clip(y, np.quantile(y, 0.0), np.quantile(y, 1.0 - 0.0))
            else:
                x = np.log(np.abs(xx))
                y = np.log(np.abs(yy))

            ols = fit_loglog_ols(x, y)
            qr = fit_loglog_quantile(x, y)
            prefix = f"{y_qty}_on_{x_qty}"
            row[f"{prefix}_n_modes_fit"] = int(xx.size)
            row[f"{prefix}_ols_intercept"] = ols["intercept"]
            row[f"{prefix}_ols_slope"] = ols["slope"]
            row[f"{prefix}_ols_r2"] = ols["r2"]
            row[f"{prefix}_ols_l1"] = ols["l1"]
            row[f"{prefix}_ols_l2"] = ols["l2"]
            row[f"{prefix}_qr_intercept"] = qr["intercept"]
            row[f"{prefix}_qr_slope"] = qr["slope"]
            row[f"{prefix}_qr_r2"] = qr["r2"]
            row[f"{prefix}_qr_l1"] = qr["l1"]
            row[f"{prefix}_qr_l2"] = qr["l2"]
            got_any = True
        if got_any:
            rows.append(row)

    assert len(rows) > 0, "read no data"
    return pl.DataFrame(rows, infer_schema_length=1000000, strict=False)


def _to_numpy(x):
    if x is None:
        return x
    return x.numpy().astype(np.float64)


def load_curv_matrices(glob_pattern: str | list[str]) -> pl.DataFrame:
    """Load run_geon_curv_profile.py's per-matrix profile payloads in full.

    ``glob_pattern`` is a single glob string, or a list of them — matches
    from every pattern are concatenated (duplicates deduped), same
    convention as ``load_experiment_metrics``.

    Unlike ``load_curv_fits`` (which reduces each per-mode quantity to a
    scalar log-log slope), this keeps every raw per-mode tensor from
    ``svd_curv.pt`` under its own meaningful column, un-fit — meant for e.g.
    plotting one matrix's curvature heatmap, or fitting/aggregating the raw
    values some other way:

    - ``curvature_matrix`` — full HVP curvature ``C[i,j] = <D_i, H_W[D_j]>``
      (``compute_l``); ``curvatures`` is its diagonal, ``l_i`` (kept
      separately too, for parity with ``load_curv_fits``'s ``"l"``).
    - ``kl_scales`` — per-mode KL-matched coefficient ``a_i`` (``compute_a``);
      ``target_kl`` is the scalar reference KL it was matched to.
    - ``noise_variance`` — per-mode per-example gradient-projection variance
      ``e_i`` (``compute_e``).
    - ``projection_mean`` — per-mode per-example gradient-projection mean
      ``p_i`` (signed; free byproduct of the same pass as ``e_i``, so also
      gated by ``compute_e``).

    Any quantity a given payload doesn't have (its ``compute_*`` flag was
    off) is simply ``None`` in that row rather than dropping the row — a
    row is only skipped if config.json is missing for its run, or the
    payload has none of the above at all.

    Same loading logic as ``load_curv_fits`` (``_iter_svd_curv_payloads``,
    so ``profile.profile_num_batches`` runs are supported too, one row per
    probe), just keeping the raw per-mode tensors instead of fitting them:
    one row per (run, step, profiled matrix, probe_idx) -- ``probe_idx`` is
    ``None`` for a normal, non-probe run, same convention as
    ``load_curv_fits``.
    """
    patterns = [glob_pattern] if isinstance(glob_pattern, str) else list(glob_pattern)
    run_dirs = sorted({Path(p) for pattern in patterns for p in glob.glob(pattern)})

    rows = []
    for base_row, mat in _iter_curv_profile_matrices(run_dirs):
        if all(
            mat.get(k) is None
            for k in (
                "curvature_matrix",
                "kl_scales",
                "noise_variance",
                "projection_mean",
            )
        ):
            continue
        rows.append(
            {
                **base_row,
                "n_modes": int(mat["singular_values"].numel()),
                "singular_values": _to_numpy(mat["singular_values"]),
                "lr": mat.get("lr"),
                # l_i
                "curvatures": _to_numpy(mat.get("curvatures")),
                "curvature_matrix": _to_numpy(mat.get("curvature_matrix")),
                # a_i
                "kl_scales": _to_numpy(mat.get("kl_scales")),
                "target_kl": mat.get("target_kl"),
                # e_i
                "noise_variance": _to_numpy(mat.get("noise_variance")),
                # p_i
                "projection_mean": _to_numpy(mat.get("projection_mean")),
            }
        )

    assert len(rows) > 0, "read no data"
    return pl.DataFrame(rows, infer_schema_length=1000000, strict=False)


def load_spectral_overlap(glob_pattern: str | list[str]) -> pl.DataFrame:
    """Load run_geon_spectral_overlap.py dumps: one row per
    (run, step, matrix, side, k, t).

    ``glob_pattern`` is a single glob string, or a list of them — matches
    from every pattern are concatenated (duplicates deduped), same
    convention as ``load_experiment_metrics``.

    Same "config.json + everything" loading convention as
    ``load_experiment_metrics``, except the per-event payload here is
    ``profiles/step_*/spectral_overlap.pt`` (nested
    ``matrices[name][side][k][t] -> singular values``, see
    run_geon_spectral_overlap.py) — flattened out to one row per (side, k, t)
    grid cell, keeping the singular-value tensor itself (length ``k*t``).
    """
    patterns = [glob_pattern] if isinstance(glob_pattern, str) else list(glob_pattern)
    run_dirs = sorted({Path(p) for pattern in patterns for p in glob.glob(pattern)})

    rows = []
    for run_dir in tqdm(run_dirs, desc="loading spectral overlap runs"):
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        with open(config_path) as f:
            config = json.load(f)

        for step_dir in sorted(glob.glob(str(run_dir / "profiles" / "step_*"))):
            payload_path = os.path.join(step_dir, "spectral_overlap.pt")
            if not os.path.exists(payload_path):
                continue
            record = torch.load(payload_path, map_location="cpu", weights_only=False)

            for matrix_name, by_side in record["matrices"].items():
                for side, by_k in by_side.items():
                    for k, by_t in by_k.items():
                        for t, sv in by_t.items():
                            rows.append(
                                {
                                    "run_dir": str(run_dir),
                                    **config,
                                    "matrix_name": matrix_name,
                                    "iter_num": int(record["iter_num"]),
                                    "side": side,
                                    "k": int(k),
                                    "t": int(t),
                                    "n_values": int(sv.numel()),
                                    "singular_values": sv.numpy(),
                                }
                            )

    assert len(rows) > 0, "read no data"
    return pl.DataFrame(rows, infer_schema_length=1000000, strict=False)


def load_future_grad(glob_pattern: str | list[str]) -> pl.DataFrame:
    """Load run_geon_curv_profile.py's ``f_ik`` (future-gradient projection)
    joined against the matching ``s_i``.

    ``glob_pattern`` is a single glob string, or a list of them — matches
    from every pattern are concatenated (duplicates deduped), same
    convention as ``load_experiment_metrics``.

    Reads ``profiles/step_*/future_grad.pt`` (written when
    ``profile.compute_f`` was on) joined against the ``svd_curv.pt``
    record(s) for the same ``iter_num``/matrix (same profiling event, so
    the ``D_i`` — and hence mode index ``i`` — line up) for
    ``singular_values``. ``f_ik`` is kept **signed** (not ``abs``), since it
    can be positive or negative and how to combine the two is left to the
    caller.

    Supports ``profile.profile_num_batches`` (requires
    ``profile_batch_size``): each profiling event's ``U``/``Vh`` directions
    (and hence ``f_ik``, which depends only on those, never on which batch
    was used) are always seeded from probe 0 alone, so there's exactly one
    ``future_grad.pt`` per step regardless of probing -- but ``singular_values``
    themselves are recomputed (deterministically, from the batch-independent
    weight/momentum signal -- see ``profiling_tensor`` -- so consistent
    across probes modulo floating-point noise) and saved once per probe
    under ``probe_<j>/svd_curv.pt`` (see ``_iter_svd_curv_payloads``, no
    top-level ``svd_curv.pt`` exists in that case). The same ``f_ik`` is
    joined against *every* available probe's own ``singular_values``, giving
    one row per (run, step, matrix, k, probe_idx) -- distinguished by the
    ``probe_idx`` column (``None`` for a normal, non-probe run, same
    convention as ``load_curv_fits``) -- so this can be joined against
    ``load_curv_fits``'s probe-indexed rows on
    (``iter_num``, ``matrix_name``, ``probe_idx``).
    """
    patterns = [glob_pattern] if isinstance(glob_pattern, str) else list(glob_pattern)
    run_dirs = sorted({Path(p) for pattern in patterns for p in glob.glob(pattern)})

    rows = []
    for run_dir in tqdm(run_dirs, desc="loading future_grad runs"):
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        with open(config_path) as f:
            config = json.load(f)

        svd_by_iter: dict[int, list[dict]] = {}
        for step_dir in sorted(glob.glob(str(run_dir / "profiles" / "step_*"))):
            payloads = _iter_svd_curv_payloads(step_dir)
            if payloads:
                svd_by_iter[int(payloads[0]["iter_num"])] = payloads

        for step_dir in sorted(glob.glob(str(run_dir / "profiles" / "step_*"))):
            fg_path = os.path.join(step_dir, "future_grad.pt")
            if not os.path.exists(fg_path):
                continue
            record = torch.load(fg_path, map_location="cpu", weights_only=False)
            svd_recs = svd_by_iter.get(int(record["iter_num"]))
            if not svd_recs:
                continue

            for svd_rec in svd_recs:
                for matrix_name, mat in record["matrices"].items():
                    svd_mat = svd_rec["matrices"].get(matrix_name)
                    if svd_mat is None:
                        continue
                    s = svd_mat["singular_values"].numpy().astype(np.float64)
                    for k, f in mat["f_ik"].items():
                        f = f.numpy().astype(np.float64)
                        rows.append(
                            {
                                "run_dir": str(run_dir),
                                **config,
                                "matrix_name": matrix_name,
                                "iter_num": int(record["iter_num"]),
                                "probe_idx": svd_rec.get("probe_idx"),
                                "k": int(k),
                                "n_modes": int(s.size),
                                "singular_values": s,
                                "f_ik": f,
                            }
                        )

    assert len(rows) > 0, "read no data"
    ret = pl.DataFrame(rows, infer_schema_length=1000000, strict=False)
    if "model_type" not in ret.columns:
        ret = ret.with_columns(pl.lit("gpt").alias("model_type"))
    return ret


def load_checkpoint_branch_metrics(glob_pattern: str | list[str]) -> pl.DataFrame:
    """Load run_geon_checkpoint_branch.py output: one row per branch
    metrics.jsonl entry (``kind="counterfactual"``/``"applied"``), joined
    with that branch's own config.json and its parent job's config.json.

    ``glob_pattern`` matches *job* dirs (run_geon_checkpoint_branch.py's
    ``<run_path>/<run name>/step_<checkpoint_step>/`` -- the trunk
    continuation, not a branch) -- a single glob string, or a list of them;
    matches from every pattern are concatenated (duplicates deduped), same
    convention as ``load_experiment_metrics``. Each matched job dir's own
    ``branches/step_*/<override name>/`` subdirectories are discovered and
    loaded internally, one row per line of that branch's own metrics.jsonl.

    Job-level config.json fields are prefixed ``job_`` (e.g. ``job_name``,
    ``job_source_path``, ``job_checkpoint_step``, ``job_source_name``,
    ``job_train``, ``job_rules``) since they'd otherwise collide with the
    branch's own ``train``/``rules``/``fork_steps``/``shared_patterns`` keys
    -- which differ from the job's (the branch's may be override-modified).
    """
    patterns = [glob_pattern] if isinstance(glob_pattern, str) else list(glob_pattern)
    job_dirs = sorted({Path(p) for pattern in patterns for p in glob.glob(pattern)})

    rows = []
    for job_dir in tqdm(job_dirs, desc="loading checkpoint branch jobs"):
        job_config_path = job_dir / "config.json"
        if not job_config_path.exists():
            continue
        with open(job_config_path) as f:
            job_config = json.load(f)

        for branch_dir in sorted(glob.glob(str(job_dir / "branches" / "step_*" / "*"))):
            branch_dir = Path(branch_dir)
            branch_config_path = branch_dir / "config.json"
            metrics_path = branch_dir / "metrics.jsonl"
            if not branch_config_path.exists() or not metrics_path.exists():
                continue
            with open(branch_config_path) as f:
                branch_config = json.load(f)

            with open(metrics_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    metric = json.loads(line)
                    rows.append(
                        {
                            "job_dir": str(job_dir),
                            "branch_dir": str(branch_dir),
                            **{f"job_{k}": v for k, v in job_config.items()},
                            **branch_config,
                            **metric,
                        }
                    )

    assert len(rows) > 0, "read no data"
    return pl.DataFrame(rows, infer_schema_length=1000000, strict=False)


def load_checkpoint_compare_metrics(glob_pattern: str | list[str]) -> pl.DataFrame:
    """Load run_geon_checkpoint_compare.py output: one row per branch
    metrics.jsonl entry (``kind="applied"``), joined with that branch's own
    config.json and its parent job's config.json.

    ``glob_pattern`` matches *job* dirs (run_geon_checkpoint_compare.py's
    ``<run_path>/<run name>/step_<checkpoint_step>/`` -- the trunk
    continuation, not a branch) -- a single glob string, or a list of them;
    matches from every pattern are concatenated (duplicates deduped), same
    convention as ``load_experiment_metrics``. Each matched job dir's own
    ``branches/<branch_spec.name>/`` subdirectories are discovered and
    loaded internally (one per entry in that job's own ``branch_specs`` --
    an arbitrary, per-run list of names, not a fixed ``branch_1``/``branch_2``/
    ``branch_3``) -- unlike ``load_checkpoint_branch_metrics``'s
    ``branches/step_*/<override name>/`` (many forks per job), this script
    forks exactly once per job, so there's no per-fork-step nesting -- one
    row per line of that branch's own metrics.jsonl.

    Job-level config.json fields are prefixed ``job_`` (e.g. ``job_name``,
    ``job_source_path``, ``job_checkpoint_step``, ``job_branch_specs``,
    ``job_end_step``, ``job_train``, ``job_rules``, ``job_branch_config``)
    since they'd otherwise collide with the branch's own
    ``train``/``rules``/``fork_steps``/``shared_patterns`` keys -- which can
    differ from the job's (the branch's may be override-modified). Each
    branch's own config.json contributes ``branch_name``, ``branch_index``,
    ``kl_matched``, ``role``, ``override`` (a struct, not a name -- see
    ``BranchSpec``), among others.
    """
    patterns = [glob_pattern] if isinstance(glob_pattern, str) else list(glob_pattern)
    job_dirs = sorted({Path(p) for pattern in patterns for p in glob.glob(pattern)})

    rows = []
    for job_dir in tqdm(job_dirs, desc="loading checkpoint compare jobs"):
        job_config_path = job_dir / "config.json"
        if not job_config_path.exists():
            continue
        with open(job_config_path) as f:
            job_config = json.load(f)

        for branch_dir in sorted(glob.glob(str(job_dir / "branches" / "*"))):
            branch_dir = Path(branch_dir)
            branch_config_path = branch_dir / "config.json"
            metrics_path = branch_dir / "metrics.jsonl"
            if not branch_config_path.exists() or not metrics_path.exists():
                continue
            with open(branch_config_path) as f:
                branch_config = json.load(f)

            with open(metrics_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    metric = json.loads(line)
                    rows.append(
                        {
                            "job_dir": str(job_dir),
                            "branch_dir": str(branch_dir),
                            **{f"job_{k}": v for k, v in job_config.items()},
                            **branch_config,
                            **metric,
                        }
                    )

    assert len(rows) > 0, "read no data"
    return pl.DataFrame(rows, infer_schema_length=1000000, strict=False)


def load_branch_continue_metrics(glob_pattern: str | list[str]) -> pl.DataFrame:
    """Load run_geon_branch_continue.py output: one row per branch
    metrics.jsonl entry (``kind="train"``/``"eval"``), joined with its job's
    config.json.

    ``glob_pattern`` matches *job* dirs (run_geon_branch_continue.py's
    ``<run_path>/<name>/step_<checkpoint_step>/``) -- a single glob string,
    or a list of them; matches from every pattern are concatenated
    (duplicates deduped), same convention as ``load_experiment_metrics``.
    Each matched job dir's own ``<branch_name>/metrics.jsonl`` files (one
    per name in that job's own ``branch_names``) are read in, one row per
    line -- unlike run_geon_checkpoint_compare.py's branches, every branch
    in one of these jobs shares the same override/train_config/rules (see
    ``ContinueSpec``), so there's no per-branch config.json to join in,
    just the job's own.

    Job-level config.json fields are prefixed ``job_`` (e.g. ``job_name``,
    ``job_path``, ``job_checkpoint_step``, ``job_branch_names``,
    ``job_override``, ``job_continue_steps``, ``job_warmup_steps``,
    ``job_reset_optimizer_state``, ``job_end_step``, ``job_train``,
    ``job_rules``) since they'd otherwise collide with the metric's own
    ``kind``/``step``/``train_loss``/``val_loss`` keys.
    """
    patterns = [glob_pattern] if isinstance(glob_pattern, str) else list(glob_pattern)
    job_dirs = sorted({Path(p) for pattern in patterns for p in glob.glob(pattern)})

    rows = []
    for job_dir in tqdm(job_dirs, desc="loading branch continue jobs"):
        job_config_path = job_dir / "config.json"
        if not job_config_path.exists():
            continue
        with open(job_config_path) as f:
            job_config = json.load(f)

        for branch_name in job_config.get("branch_names", []):
            metrics_path = job_dir / branch_name / "metrics.jsonl"
            if not metrics_path.exists():
                continue

            with open(metrics_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    metric = json.loads(line)
                    rows.append(
                        {
                            "job_dir": str(job_dir),
                            "branch_name": branch_name,
                            **{f"job_{k}": v for k, v in job_config.items()},
                            **metric,
                        }
                    )

    assert len(rows) > 0, "read no data"
    return pl.DataFrame(rows, infer_schema_length=1000000, strict=False)


def load_branch_full_history(glob_pattern: str | list[str]) -> pl.DataFrame:
    """Wrapper around ``load_branch_continue_metrics`` that also loads each
    continued branch's own history *before* being picked up there -- the
    run_geon_checkpoint_compare.py fork-exploration training (its own
    per-branch override, ``kind="applied"`` metrics) that produced the
    checkpoint being continued, from when forking started through the
    checkpointed iteration. Combined with ``load_branch_continue_metrics``'s
    own train/eval history, this gives one continuous per-branch loss
    history: fork exploration, then the shared-override continuation.

    ``glob_pattern`` is the same run_geon_branch_continue.py job-dir glob
    ``load_branch_continue_metrics`` takes. The preceding fork-exploration
    run is located automatically, no separate pattern needed: each continue
    job's own config.json already records ``path`` (the source
    run_geon_checkpoint_compare.py job dir) and ``checkpoint_step`` (which
    fork iteration was checkpointed) -- fork-phase rows are every "applied"
    row from that job dir with ``fork_step + iter + 1 <= checkpoint_step``
    (both sides use the same "steps completed so far" convention -- see
    ``BranchConfig.checkpoint_after_steps``).

    Adds ``phase`` (``"fork_explore"`` or ``"shared_continue"``) and a
    common ``step``/``val_loss`` -- fork-phase ``step`` is ``fork_step +
    iter + 1`` (steps-completed-so-far, matching run_geon_rules.py's own
    convention) and ``val_loss`` is that iteration's ``post_val_loss``;
    continue-phase keeps its own ``step``/``val_loss`` as-is. This gives a
    clean split, no overlap: fork-phase's last row lands exactly on
    ``checkpoint_step``, continue-phase's first row on ``checkpoint_step +
    1``. ``fork_step`` (native only to fork-phase rows) is broadcast across
    every row of the same branch history, so it's never null. Columns that
    only exist on one phase (e.g. ``kl``/``scale``/``override`` on
    ``"fork_explore"``, ``job_override``/``rolling_loss`` on
    ``"shared_continue"``) are null on rows from the other phase. The
    original fork job's own name is kept as ``fork_job_name`` (``job_name``
    is overwritten with the *continue* job's name on fork-phase rows too,
    so grouping/filtering by ``job_name`` sees one continuous history --
    note a single fork job can feed several continue jobs, so its rows are
    duplicated once per continue job that references it).
    """
    df_continue = load_branch_continue_metrics(glob_pattern).with_columns(
        phase=pl.lit("shared_continue")
    )

    fork_cache: dict[str, pl.DataFrame] = {}
    fork_frames = []
    jobs = df_continue.select("job_name", "job_path", "job_checkpoint_step").unique()
    for row in jobs.iter_rows(named=True):
        job_name, job_path, checkpoint_step = (
            row["job_name"],
            row["job_path"],
            row["job_checkpoint_step"],
        )
        if job_path not in fork_cache:
            fork_cache[job_path] = load_checkpoint_compare_metrics(
                [resolve_repo_path(job_path)]
            ).filter(pl.col("kind").eq("applied"))

        df_fork = (
            fork_cache[job_path]
            .filter((pl.col("fork_step") + pl.col("iter") + 1).le(checkpoint_step))
            .rename({"job_name": "fork_job_name"})
            .with_columns(
                step=pl.col("fork_step") + pl.col("iter") + 1,
                val_loss=pl.col("post_val_loss"),
                phase=pl.lit("fork_explore"),
                job_name=pl.lit(job_name),
                job_path=pl.lit(job_path),
                job_checkpoint_step=pl.lit(checkpoint_step),
            )
        )
        fork_frames.append(df_fork)

    df_fork_all = pl.concat(fork_frames, how="diagonal_relaxed")
    df = pl.concat([df_fork_all, df_continue], how="diagonal_relaxed")
    # fork_step only exists natively on fork-phase rows; broadcast it across
    # every row of the same branch history (constant within one) so it's
    # never null, even on shared_continue rows.
    df = df.with_columns(
        fork_step=pl.col("fork_step")
        .max()
        .over("job_name", "job_checkpoint_step", "branch_name")
    )
    return df.sort("job_name", "branch_name", "job_checkpoint_step", "step")
