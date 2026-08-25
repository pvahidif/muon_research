"""Offline plots for run_curv_profile.py output
(``<run_path>/<name>/step_<n>/profiles/step_<n>/svd_curv.pt`` -- see
run_curv_profile.py's own docstring).

``run_path`` is the base directory a run_curv_profile.py config's
``--run_path`` pointed at (e.g. ``experiments/exp003_curvature/
muon_curvature``). Every immediate subdirectory (one per ``RunSpec.name``,
e.g. ``fork_000000``, ``fork_000100``, ...) is a GROUP; every
``step_*/profiles/step_*/svd_curv.pt`` found inside one group is loaded and
POOLED together for most pages (see ``load_curv_profile_groups``/
``_pool_fields``). Pooling is deliberate, not just convenience: a group that
names several consecutive post-fork checkpoints (see
``experiments/exp003_curvature/muon_curvature/config.yaml``'s own comment)
is doing so specifically so its samples can be combined to bring down
noise, on the assumption the underlying Hessian barely moves across that
short a window -- a group with only a single checkpoint (an early,
fast-changing training stage where one sample is already precise enough)
pools trivially to just that one step's own values. Every page below is one
grid of subplots, one PER GROUP (in sorted-by-name order, e.g.
``fork_000000`` before ``fork_000100``), with the sole exception of the two
heatmap pages, which write one PDF PAGE PER GROUP instead (see
``plot_heatmap_pages``) -- an ``(r, r)`` matrix lives in that step's own
mode basis, so unlike a scalar per-mode field, it can't be pooled across
steps into one panel; every page shares the same grid size (sized to
whichever group has the most steps), so pages don't come out a different
physical size group to group. ``plot_heatmap_page`` (singular) is a
notebook-oriented alternative: one COMBINED figure, one panel per group,
each showing only that group's own fork step -- not wired into this CLI.

Pages (in order), all reading straight from the payload's own field names
(see ``muon_research/curv.py``'s own docstring for their definitions/scales):

1. ``sigma_vs_index`` -- ``sigma_i`` vs mode index ``i`` (spectral decay),
   ``sigma`` log-scale, index linear, no fit line.
2. ``gamma_diag_vs_sigma`` -- ``|gamma_ii|`` vs ``sigma_i``, log-log, with a
   log-log power-law fit line (see ``_log_fit``/``_plot_log_fit_line``).
   Points are colored by ``gamma_diag``'s own sign (blue +, red -) so
   negative-curvature modes stay visible even though the axis itself is a
   magnitude.
3. ``alpha_vs_sigma`` -- ``alpha_i`` vs ``sigma_i``, log-log, with a fit
   line. ``alpha`` is already non-negative (a Frobenius norm), so no sign
   coloring.
4. ``gamma_perp_diag_vs_sigma`` -- ``|gamma_perp_diag_i|`` vs ``sigma_i``,
   log-log, fit line, sign-colored (same rationale as ``gamma_diag``).
5. ``gamma_heatmap`` -- ``gamma_ij / sqrt(|gamma_ii * gamma_jj|)``, a
   covariance-to-correlation-style normalization of the full HVP curvature
   matrix (see ``normalize_gamma``). The diagonal is ``sign(gamma_ii)``
   (+-1), not always +1, since ``gamma_ii`` itself can be negative (a
   non-convex direction) -- the normalization uses ``|gamma_ii*gamma_jj|``
   under the square root precisely so a negative diagonal entry doesn't
   produce NaN. Colormap is 0-centered (0 always white, ``RdBu_r``) with a
   PER-PANEL dynamic range (at least +-1) rather than a fixed +-1, since
   this quantity -- unlike ``eta_perp`` below -- isn't provably bounded in
   ``[-1, 1]``.
6. ``eta_perp_heatmap`` -- ``eta_perp_ij = <Q_i, Q_j>``, already a cosine
   similarity in ``[-1, 1]`` as saved (every ``Q_i`` is unit Frobenius
   norm) -- 0-centered colormap, fixed +-1 range.
7. ``phi_ratio`` / ``phi_perp_ratio`` -- for ``phi_samples``/
   ``phi_perp_samples`` (raw per-example gradient projections as saved,
   not per-mode -- see ``sample_gradient_projections`` in curv.py): each
   record's own per-example samples are FIRST averaged over the example
   axis (one batch-mean point per mode, estimating that mode's true
   projection rather than any single example's noisy one -- see
   ``_pool_phi_samples``'s own docstring for why this matters), and those
   batch-mean points are what gets pooled across every step in the group,
   one point per (mode, record), paired with that mode's own ``sigma_i``.
   A power ``p`` is fit per group from ``log|phi| ~ p*log(sigma)`` (same
   log-log fit machinery as the pair pages above, just relabeled ``p``
   instead of ``beta``), and ``phi/sigma^p`` is plotted (signed, linear y)
   against ``sigma_i`` -- jittered multiplicatively (see
   ``plot_phi_ratio_page``) purely so points sharing a very close
   ``sigma_i`` (e.g. the same mode across nearby steps) don't all stack on
   the same x position; the jitter never touches the fit or the plotted y
   value, only where each point is drawn. A centered, edge-preserving
   rolling median + IQR band (see ``_rolling_median_smooth``) is overlaid.

Each page has its own ``--<name>``/``--no-<name>`` CLI flag (default: on).
A page with no plottable data anywhere for a given matrix (e.g.
``compute_gamma`` was off for the whole run) is skipped rather than written
as an all-empty page.

Example:
    python src/muon_research/scripts/analyze_curv_profile.py \\
        experiments/exp003_curvature/muon_curvature
    python src/muon_research/scripts/analyze_curv_profile.py \\
        experiments/exp003_curvature/muon_curvature --out_dir /tmp/muon_curv
    python src/muon_research/scripts/analyze_curv_profile.py \\
        experiments/exp003_curvature/muon_curvature --no-gamma-heatmap
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from typing import Callable, NamedTuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import CenteredNorm, LinearSegmentedColormap, Normalize

PDF_FILE = "svd_curv_scatter.pdf"

# Sentinel x_key: synthesizes the 1-indexed mode rank (see _pool_fields)
# rather than reading a field out of the saved payload.
INDEX_KEY = "__index__"

# Clip fraction for plot_phi_ratio_page's y -- dividing by sigma_i^p can
# blow up for small sigma_i, so a handful of outliers would otherwise
# dominate the axis scale and swamp the smoothing line.
RATIO_CLIP_QUANTILE = 0.1


class PairSpec(NamedTuple):
    name: str  # CLI flag name (--<name>/--no-<name>) and enabled-dict key
    x_key: str = ""
    y_key: str = ""
    x_label: str = ""
    y_label: str = ""
    title: str = ""
    x_log: bool = True  # False plots x on a linear axis
    y_log: bool = True  # False plots y on a linear axis
    # y can be negative (e.g. gamma_diag): plot |y| (log-safe) but color
    # each point by its own original sign, so negative modes stay visible.
    color_by_sign: bool = False
    # Whether to fit+draw the log-log power-law line (see _log_fit) -- off
    # for sigma_vs_index, whose x isn't itself log-scaled.
    fit_line: bool = True
    # "pair": generic pooled scatter, handled by plot_pooled_pair_page.
    # "gamma_heatmap"/"eta_perp_heatmap"/"phi_ratio"/"phi_perp_ratio":
    # handled by their own dedicated functions instead -- kind just keeps
    # them out of the generic pair loop and drives CLI-flag generation.
    kind: str = "pair"
    # CLI --help text; None auto-generates "Include the {title!r} page."
    help: str | None = None


PAIR_SPECS: tuple[PairSpec, ...] = (
    PairSpec(
        "sigma_vs_index",
        INDEX_KEY,
        "sigma",
        r"$i$",
        r"$\sigma_i$",
        r"$\sigma_i$ vs mode index $i$ (spectral decay)",
        x_log=False,
        fit_line=False,
    ),
    PairSpec(
        "gamma_diag_vs_sigma",
        "sigma",
        "gamma_diag",
        r"$\sigma_i$",
        r"$|\gamma_{ii}|$",
        r"$|\gamma_{ii}|$ vs $\sigma_i$ (HVP curvature along $D_i$)",
        color_by_sign=True,
    ),
    PairSpec(
        "alpha_vs_sigma",
        "sigma",
        "alpha",
        r"$\sigma_i$",
        r"$\alpha_i$",
        r"$\alpha_i$ vs $\sigma_i$ (HVP residual size)",
    ),
    PairSpec(
        "gamma_perp_diag_vs_sigma",
        "sigma",
        "gamma_perp_diag",
        r"$\sigma_i$",
        r"$|\gamma^\perp_{ii}|$",
        r"$|\gamma^\perp_{ii}|$ vs $\sigma_i$ (curvature along $Q_i$)",
        color_by_sign=True,
    ),
    PairSpec(
        "gamma_heatmap",
        kind="gamma_heatmap",
        help=(
            "Include the gamma_ij/sqrt(|gamma_ii*gamma_jj|) heatmap page "
            "(needs profile.compute_gamma=true) -- one PDF page per group, "
            "one subplot per step in that group (see plot_heatmap_pages)."
        ),
    ),
    PairSpec(
        "eta_perp_heatmap",
        kind="eta_perp_heatmap",
        help=(
            "Include the eta_perp_ij = <Q_i, Q_j> residual "
            "cross-correlation heatmap page (needs "
            "profile.compute_gamma=true) -- one PDF page per group, one "
            "subplot per step in that group."
        ),
    ),
    PairSpec(
        "phi_ratio",
        kind="phi_ratio",
        help=(
            "Include the phi_samples/sigma_i^p ratio page (needs "
            "profile.compute_phi=true), p fit per group from "
            "log|phi| ~ p*log(sigma) (see plot_phi_ratio_page)."
        ),
    ),
    PairSpec(
        "phi_perp_ratio",
        kind="phi_perp_ratio",
        help=(
            "Include the phi_perp_samples/sigma_i^p ratio page (needs "
            "profile.compute_phi=true AND profile.compute_gamma=true, "
            "same recipe as --phi-ratio)."
        ),
    ),
)


def _short_matrix_name(name: str) -> str:
    """``blocks.0.mlp.fc`` -> ``0.mlp.fc``."""
    if name.startswith("blocks."):
        return name[len("blocks.") :]
    return name


def _log_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """Least squares ``log10(y) = beta*log10(x) + alpha`` -> ``(beta, alpha)``."""
    if x.size < 2:
        return None
    lx = np.log10(x)
    if np.std(lx) <= 0:
        return None
    beta, alpha = np.polyfit(lx, np.log10(y), 1)
    return float(beta), float(alpha)


def _declutter_log_axis(ax, which: str = "x") -> None:
    """Drop minor-tick labels on a log-scale axis. With a narrow dynamic
    range (a handful of decades, common once a subplot grid shrinks each
    panel), matplotlib's default minor ticks (2x10, 3x10, ...) get labeled
    too and crowd/overlap the major (10^n) labels. Keeps the (unlabeled)
    minor tick marks for visual reference -- only the labels are dropped."""
    axis_obj = ax.xaxis if which == "x" else ax.yaxis
    axis_obj.set_minor_formatter(mticker.NullFormatter())


def _log_limits(v: np.ndarray) -> tuple[float, float]:
    """Data limits with a small multiplicative margin (log-scale axes)."""
    lo, hi = float(v.min()), float(v.max())
    if lo <= 0.0:
        raise ValueError("log limits need positive data")
    if lo == hi:
        return lo / 2.0, hi * 2.0
    pad = (hi / lo) ** 0.05
    return lo / pad, hi * pad


def _linear_limits(v: np.ndarray) -> tuple[float, float]:
    """Data limits with a small additive margin (linear-scale axis)."""
    lo, hi = float(v.min()), float(v.max())
    if lo == hi:
        return lo - 1.0, hi + 1.0
    pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


def _plot_log_fit_line(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    fit: tuple[float, float],
    *,
    x_log: bool = True,
    y_log: bool = True,
) -> None:
    """Faint dashed fit line ``y = 10^alpha * x^beta``, pinned to the data range.

    ``x_log=True`` (log-log axes): the fit is a straight line, so two
    endpoints suffice. ``x_log=False`` (linear x, log y): the same curve is
    no longer straight in this projection, so it's sampled densely instead —
    over the true (unpadded) data range, since the padded linear ``xlim``
    can dip to/below 0 (undefined for ``x**beta`` at non-integer ``beta``),
    whereas ``x`` itself is already guaranteed positive by the caller.
    ``y_log`` only affects the y-axis limits computed here (log vs linear
    margin) -- the fit itself (see ``_log_fit``) is always a log-log power
    law, requiring positive ``x``/``y``, regardless of how either axis is
    displayed.
    """
    beta, alpha = fit
    xlim = _log_limits(x) if x_log else _linear_limits(x)
    ylim = _log_limits(y) if y_log else _linear_limits(y)
    t = (
        np.asarray(xlim, dtype=np.float64)
        if x_log
        else np.linspace(float(x.min()), float(x.max()), 200)
    )
    # A near-zero x-spread fit (e.g. only a handful of similarly-sized
    # sigma_i) can give a huge |beta|, which overflows float64 once
    # exponentiated. Work in log-space first and skip the line (but keep
    # the axes/annotation) if it would.
    log_fit_y = alpha + beta * np.log10(t)
    if np.all(np.isfinite(log_fit_y)) and np.all(np.abs(log_fit_y) < 300.0):
        ax.plot(t, 10.0**log_fit_y, ls="--", lw=1.0, color="0.6", alpha=0.9)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def _rolling_mean_smooth(
    x: np.ndarray,
    y: np.ndarray,
    frac: float = 0.2,
    min_points: int = 5,
    max_evals: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Dependency-free trend line (+ uncertainty band) for a scatter whose
    ``y`` isn't a magnitude (so the log-log power-law fit in ``_log_fit``
    doesn't apply, e.g. a signed ratio): sort by ``x``, then a centered
    rolling window over a window sized ``frac`` of the points (odd, at
    least ``min_points``); within each window, the mean (the trend line)
    and ``mean -+ std`` (a parametric band -- how spread out ``y`` actually
    is near that ``x``, not a standard error of the mean). The window is
    centered and shrinks (rather than dropping the point) near the edges,
    so every point -- including the first/last -- gets its own trend/band
    value. Prefer ``_rolling_median_smooth`` instead when ``y`` has heavy
    tails or outliers a plain mean/std would be sensitive to; this one
    assumes a window's own points are well-summarized by their mean.

    The window is only EVALUATED at up to ``max_evals`` positions (evenly
    spaced by index, always including the first/last point so the line
    still reaches the true edges) rather than at every one of ``n``
    points -- each evaluation still uses the full ``frac``-sized window of
    actual data centered there, so this only caps how many times that
    window is computed, not its size or which points feed it. A smooth
    trend line doesn't need one evaluation per sample, and a group with a
    large pooled count would otherwise make the naive per-point loop the
    bottleneck. A no-op
    (evaluates at every point, exactly as before) whenever ``n <=
    max_evals``.

    Returns ``(xs, mean, mean_minus_std, mean_plus_std)``, or ``None`` if
    there are fewer than ``min_points`` -- too few for a meaningful
    trend."""
    if x.size < min_points:
        return None
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    n = xs.size
    w = max(min_points, int(round(n * frac)))
    w = min(w, n if n % 2 == 1 else n - 1)
    if w % 2 == 0:
        w += 1
    half = w // 2
    eval_idx = (
        np.arange(n)
        if n <= max_evals
        else np.unique(np.linspace(0, n - 1, max_evals).round().astype(np.int64))
    )
    mean = np.empty(eval_idx.size, dtype=np.float64)
    band_lo = np.empty(eval_idx.size, dtype=np.float64)
    band_hi = np.empty(eval_idx.size, dtype=np.float64)
    for out_i, i in enumerate(eval_idx):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window = ys[lo:hi]
        m = np.mean(window)
        s = np.std(window)
        mean[out_i] = m
        band_lo[out_i] = m - s
        band_hi[out_i] = m + s
    return xs[eval_idx], mean, band_lo, band_hi


def _rolling_median_smooth(
    x: np.ndarray,
    y: np.ndarray,
    frac: float = 0.2,
    min_points: int = 5,
    band_quantile: float = 0.25,
    max_evals: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Dependency-free trend line (+ uncertainty band) for a scatter whose
    ``y`` isn't a magnitude (so the log-log power-law fit in ``_log_fit``
    doesn't apply, e.g. a signed ratio): sort by ``x``, then a centered
    rolling window over a window sized ``frac`` of the points (odd, at
    least ``min_points``); within each window, the median (the trend line)
    and the ``[band_quantile, 1 - band_quantile]`` quantiles (an IQR-style
    band -- how spread out ``y`` actually is near that ``x``, not a
    standard error of the median). The window is centered and shrinks
    (rather than dropping the point) near the edges, so every point --
    including the first/last -- gets its own trend/band value.

    The window is only EVALUATED at up to ``max_evals`` positions (evenly
    spaced by index, always including the first/last point so the line
    still reaches the true edges) rather than at every one of ``n``
    points -- each evaluation still uses the full ``frac``-sized window of
    actual data centered there, so this only caps how many times that
    window is computed, not its size or which points feed it. A smooth
    trend line doesn't need one evaluation per sample, and a group with a
    large pooled count would otherwise make the naive per-point loop the
    bottleneck. A no-op
    (evaluates at every point, exactly as before) whenever ``n <=
    max_evals``.

    Returns ``(xs, median, lo, hi)``, or ``None`` if there are fewer than
    ``min_points`` -- too few for a meaningful trend."""
    if x.size < min_points:
        return None
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    n = xs.size
    w = max(min_points, int(round(n * frac)))
    w = min(w, n if n % 2 == 1 else n - 1)
    if w % 2 == 0:
        w += 1
    half = w // 2
    eval_idx = (
        np.arange(n)
        if n <= max_evals
        else np.unique(np.linspace(0, n - 1, max_evals).round().astype(np.int64))
    )
    med = np.empty(eval_idx.size, dtype=np.float64)
    band_lo = np.empty(eval_idx.size, dtype=np.float64)
    band_hi = np.empty(eval_idx.size, dtype=np.float64)
    for out_i, i in enumerate(eval_idx):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window = ys[lo:hi]
        med[out_i] = np.median(window)
        band_lo[out_i] = np.quantile(window, band_quantile)
        band_hi[out_i] = np.quantile(window, 1.0 - band_quantile)
    return xs[eval_idx], med, band_lo, band_hi


DEFAULT_STAT_FONTSIZE = 9


def _annotate_fit_stats(
    ax,
    x_for_fit: np.ndarray,
    y_for_fit: np.ndarray,
    fit: tuple[float, float] | None,
    exponent_symbol: str = "β",
    fontsize: float = DEFAULT_STAT_FONTSIZE,
) -> None:
    """Per-panel annotation box: ``n`` (always -- how many points went into
    ``fit``) and, only when ``fit is not None``, ``R²`` of
    ``log10(x_for_fit)`` vs ``log10(y_for_fit)`` -- exactly the data
    ``fit`` (see ``_log_fit``) was itself computed from, so ``R²`` is that
    fit's own goodness-of-fit, not some other correlation of whatever
    happens to be on-screen -- plus the fit's own exponent (labeled
    ``exponent_symbol``: ``"β"`` for the pair/stability pages, ``"p"`` for
    ``plot_phi_ratio_page``, matching this module's own naming elsewhere)
    and intercept ``α``. First line: ``n``, ``R²``. Second line (only if a
    fit exists): the exponent, ``α``. See ``_fit_stats_caption`` for what
    these mean, explained once per figure rather than repeated in every
    panel."""
    line1 = f"n={x_for_fit.size}"
    lines = None
    if fit is not None:
        beta, alpha = fit
        rho = float(np.corrcoef(np.log10(x_for_fit), np.log10(y_for_fit))[0, 1])
        line1 += f"  R²={rho**2:.2f}"
        lines = [line1, f"{exponent_symbol}={beta:.2f} α={alpha:.2f}"]
    else:
        lines = [line1]
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fontsize,
        color="0.25",
    )


def _fit_stats_caption(exponent_symbol: str = "β", has_fit: bool = True) -> str:
    """Caption text explaining ``_annotate_fit_stats``'s own annotation box
    -- present regardless of caller (notebook or CLI), unlike
    ``plot_pooled_pair_page``'s own ``mname_label``/``group_label``
    overrides, which only apply where a caller opts in. ``has_fit=False``
    (e.g. ``sigma_vs_index``, which has no fit at all) drops every line
    but ``n``, matching what that page's own panels actually show."""
    lines = ["n: number of points."]
    if has_fit:
        lines.append(
            f"R², {exponent_symbol}, α: least-squares fit "
            f"log10(y) = {exponent_symbol}·log10(x) + α "
            f"({exponent_symbol} the fitted power-law exponent, α the intercept)."
        )
    return "\n".join(lines)


def _finalize_figure_with_caption(
    fig, caption: str, fontsize: float = DEFAULT_STAT_FONTSIZE
) -> None:
    """Reserve room at the bottom of ``fig`` for ``caption`` (shared across
    every panel, one line per stat -- see ``_fit_stats_caption``) and lay
    out the rest of the figure normally. Shared by every page-building
    function in this module that shows fit statistics, so the caption
    placement/margin logic lives in exactly one place. Call after
    ``fig.suptitle(...)``, in place of a bare ``fig.tight_layout(...)``."""
    n_lines = caption.count("\n") + 1
    bottom = 0.02 + 0.025 * n_lines
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=fontsize, color="0.3")
    fig.tight_layout(rect=[0, bottom, 1, 0.96])


def load_curv_profile_groups(run_path: str) -> dict:
    """Load every run_curv_profile.py output under ``run_path``: each
    immediate subdirectory (e.g. ``fork_000000``, ``fork_000100``, ...) is
    one GROUP -- see this module's own docstring for why groups (not
    individual profiling events) are the unit every page pools by. Within a
    group, every ``step_*/profiles/step_*/svd_curv.pt`` is loaded and
    sorted by ``iter_num``."""
    run_path = os.path.abspath(run_path)
    pattern = os.path.join(run_path, "*", "step_*", "profiles", "step_*", "svd_curv.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(
            f"no */step_*/profiles/step_*/svd_curv.pt found under {run_path!r}"
        )
    by_group: dict[str, list[dict]] = {}
    for f in paths:
        group_name = os.path.relpath(f, run_path).split(os.sep)[0]
        rec = torch.load(f, map_location="cpu", weights_only=False)
        by_group.setdefault(group_name, []).append(rec)
    groups = []
    for name in sorted(by_group.keys()):
        records = sorted(by_group[name], key=lambda r: int(r["iter_num"]))
        groups.append(
            {
                "name": name,
                "records": records,
                "iters": np.asarray(
                    [int(r["iter_num"]) for r in records], dtype=np.int64
                ),
            }
        )
    matrix_names = list(groups[0]["records"][0]["matrix_names"])
    return {
        "path": run_path,
        "groups": groups,
        "matrix_names": matrix_names,
        "optim": groups[0]["records"][0].get("optim"),
        "profile_source": groups[0]["records"][0].get("profile_source"),
    }


def _pool_fields(
    group: dict, mname: str, x_key: str, y_key: str, *, pool_window: bool = False
) -> tuple[np.ndarray, np.ndarray] | None:
    """Pool aligned ``(x, y)`` pairs for matrix ``mname``, keeping each
    ``x``/``y`` pair aligned to the SAME record/mode -- so pooling several
    steps' worth of otherwise unrelated per-mode bases still gives valid
    ``(x_i, y_i)`` pairs, one per (step, mode). By default
    (``pool_window=False``), only the group's own FIRST record is used --
    its actual checkpoint/fork step, not the extra steps profiled just
    after it for window-stability pooling (see this module's own
    docstring, and ``plot_fork_window_stability``, whose whole point is
    checking whether those extra steps are safe to combine at all).
    ``pool_window=True`` opts into pooling every record in ``group``
    instead, combining that checkpoint step with its own post-checkpoint
    window, on the assumption (checked elsewhere, not enforced here) that
    the underlying geometry barely moves over that window. ``x_key ==
    INDEX_KEY`` synthesizes each record's own 1-indexed mode rank instead
    of reading a field. ``None`` if no record in the group has both fields
    for this matrix (e.g. ``compute_gamma`` was off)."""
    xs, ys = [], []
    records = group["records"] if pool_window else group["records"][:1]
    for rec in records:
        mat = rec["matrices"].get(mname)
        if mat is None:
            continue
        if x_key == INDEX_KEY:
            sigma = mat.get("sigma")
            if sigma is None:
                continue
            x = np.arange(1, sigma.numel() + 1, dtype=np.float64)
        else:
            xv = mat.get(x_key)
            if xv is None:
                continue
            x = xv.numpy().astype(np.float64)
        yv = mat.get(y_key)
        if yv is None:
            continue
        y = yv.numpy().astype(np.float64)
        xs.append(x)
        ys.append(y)
    if not xs:
        return None
    return np.concatenate(xs), np.concatenate(ys)


def _pool_phi_samples(
    group: dict, mname: str, field: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Pool one BATCH-MEAN point per (mode, record) of ``field``
    (``phi_samples``/``phi_perp_samples``, shape ``(r, n_examples)`` per
    record) across every step in ``group``, each paired with its own
    mode's ``sigma``.

    ``phi_samples[i, k]`` (the per-example gradient projection onto mode
    ``i``, for example ``k``) decomposes as ``phi_samples[i, k] = mu_i +
    eps_ik``: a true per-mode signal ``mu_i`` plus per-example noise
    ``eps_ik`` from which example happened to land in that profiling
    batch. What the true/batch gradient's own projection onto mode ``i``
    actually estimates is ``mu_i``, via ``mean_k(phi_samples[i, k])`` (see
    ``sample_gradient_projections`` in ``curv.py``) -- NOT any individual
    ``phi_samples[i, k]``. Averaging over the example axis here, before
    pooling across the group's own records, is what makes the fit this
    feeds (see ``plot_phi_ratio_page``/``plot_phi_ratio_grid``) a fit of
    that signal ``mu_i`` against ``sigma_i``, rather than a fit dominated
    by ``eps_ik``'s own noise scale, which individual per-example points
    would otherwise mix in. This also mirrors every other pooled quantity
    in this module (``gamma_diag``, ``alpha``, ...), which are already
    one value per (mode, record), not one value per (mode, record,
    example).

    ``None`` if no record in the group has both ``field`` and ``sigma``
    for this matrix."""
    xs, ys = [], []
    for rec in group["records"]:
        mat = rec["matrices"].get(mname)
        if mat is None:
            continue
        samples = mat.get(field)
        sigma = mat.get("sigma")
        if samples is None or sigma is None:
            continue
        samples = samples.numpy().astype(np.float64)  # (r, n_examples)
        sigma = sigma.numpy().astype(np.float64)  # (r,)
        phi_bar = samples.mean(axis=1)  # (r,) -- batch-mean projection per mode
        xs.append(sigma)
        ys.append(phi_bar)
    if not xs:
        return None
    return np.concatenate(xs), np.concatenate(ys)


def _plot_pair_panel(
    ax,
    group: dict,
    mname: str,
    spec: PairSpec,
    label: str,
    *,
    pool_window: bool = False,
    stat_fontsize: float = DEFAULT_STAT_FONTSIZE,
) -> bool:
    """Draw one ``(group, mname, spec)`` scatter panel into ``ax`` --
    pooled points, sign-coloring, log-log fit line + annotation box, same
    as one panel of ``plot_pooled_pair_page``. Returns whether anything
    was actually plotted (``False`` for missing/all-non-positive data,
    in which case ``ax`` is left as an explanatory placeholder instead).
    Factored out of ``plot_pooled_pair_page`` so ``plot_pair_grid`` (an
    arbitrary, caller-specified row/col layout of panels, rather than one
    panel per group of a single ``mname``) can share the exact same
    per-panel drawing logic. ``pool_window``: see ``_pool_fields``.
    ``stat_fontsize``: forwarded to ``_annotate_fit_stats``."""
    pooled = _pool_fields(group, mname, spec.x_key, spec.y_key, pool_window=pool_window)
    if pooled is None:
        ax.set_axis_off()
        ax.set_title(f"{label} (missing)", fontsize=9)
        return False
    x, y_raw = pooled
    keep = np.isfinite(x) & np.isfinite(y_raw)
    if spec.x_log:
        keep &= x > 0
    if spec.y_log:
        keep &= (y_raw != 0) if spec.color_by_sign else (y_raw > 0)
    if not keep.any():
        ax.text(0.5, 0.5, "no points>0", ha="center", va="center")
        ax.set_title(label, fontsize=9)
        return False
    x, y_raw = x[keep], y_raw[keep]

    if spec.color_by_sign:
        y = np.abs(y_raw)
        pos = y_raw >= 0
        neg = ~pos
        ax.scatter(
            x[pos], y[pos], s=8, alpha=0.6, edgecolors="none", c="#1f77b4", label="+"
        )
        ax.scatter(
            x[neg], y[neg], s=8, alpha=0.6, edgecolors="none", c="#d62728", label="-"
        )
        if pos.any() and neg.any():
            ax.legend(fontsize=6, loc="lower left", framealpha=0.5)
    else:
        y = y_raw
        ax.scatter(x, y, s=8, alpha=0.6, edgecolors="none", c="#1f77b4")

    ax.set_xscale("log" if spec.x_log else "linear")
    ax.set_yscale("log" if spec.y_log else "linear")
    if spec.x_log:
        _declutter_log_axis(ax, "x")
    if spec.y_log:
        _declutter_log_axis(ax, "y")
    fit = _log_fit(x, y) if (spec.fit_line and spec.x_log and spec.y_log) else None
    if fit is not None:
        _plot_log_fit_line(ax, x, y, fit, x_log=spec.x_log, y_log=spec.y_log)
    _annotate_fit_stats(ax, x, y, fit, fontsize=stat_fontsize)
    ax.set_xlabel(spec.x_label)
    ax.set_ylabel(spec.y_label)
    ax.set_title(label, fontsize=9)
    return True


def plot_pooled_pair_page(
    mname: str,
    groups: list[dict],
    spec: PairSpec,
    *,
    mname_label: str | None = None,
    group_label: Callable[[dict], str] | None = None,
    pool_window: bool = False,
    stat_fontsize: float = DEFAULT_STAT_FONTSIZE,
) -> "plt.Figure | None":
    """Build (but do not save/show/close) one figure for ``(mname, spec)``:
    grid of scatters, one panel PER GROUP -- by default, each panel uses
    only that group's own checkpoint/fork step, not the extra steps
    profiled just after it; pass ``pool_window=True`` to instead pool
    every step in the group (see ``_pool_fields``, and
    ``plot_fork_window_stability`` for the check that justifies doing so).
    Returns the figure, or ``None`` if no group has any plottable point
    (nothing built). What to do with the figure -- save it to a
    ``PdfPages``, ``plt.show()`` it inline, close it -- is entirely the
    caller's call.

    ``mname_label``: text for the figure's own suptitle in place of
    ``_short_matrix_name(mname)`` (the default when ``None``) -- e.g. a
    human-readable name for a notebook, where the raw dotted param name
    isn't the point. ``group_label``: a function computing each panel's
    own title from its ``group`` dict, in place of that group's raw
    ``group["name"]`` (the default when ``None``) -- e.g. a caller wanting
    ``f"step {group['records'][0]['iter_num']}"`` instead of
    ``"fork_000000"``. Both are opt-in overrides so this function's own
    default behavior (used by the CLI/PDF output) is unchanged; a caller
    supplies the actual replacement text/logic rather than this function
    hardcoding any particular naming scheme.

    Every panel also gets a small annotation box (see
    ``_annotate_fit_stats``: ``n``/``ρ``/``R²``/the fit's own ``β``/``α``)
    -- and the figure as a whole gets one shared caption at the bottom
    explaining what those mean, so a reader doesn't have to already know
    this module's own notation (present regardless of caller, unlike
    ``mname_label``/``group_label``: this one caption is the same
    information anywhere this figure is shown). ``stat_fontsize``:
    forwarded to both the per-panel annotation box and the shared caption.
    """
    n_groups = len(groups)
    ncols = int(math.ceil(math.sqrt(n_groups)))
    nrows = int(math.ceil(n_groups / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols + 0.6, 2.8 * nrows + 0.8), squeeze=False
    )
    any_plotted = False
    for idx, group in enumerate(groups):
        r, c = divmod(idx, ncols)
        label = group_label(group) if group_label is not None else group["name"]
        any_plotted |= _plot_pair_panel(
            axes[r][c],
            group,
            mname,
            spec,
            label,
            pool_window=pool_window,
            stat_fontsize=stat_fontsize,
        )
    for idx in range(n_groups, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")
    if not any_plotted:
        plt.close(fig)
        return None

    title = mname_label if mname_label is not None else _short_matrix_name(mname)
    fig.suptitle(f"{title}: {spec.title}", fontsize=11)
    _finalize_figure_with_caption(
        fig, _fit_stats_caption("β", spec.fit_line), fontsize=stat_fontsize
    )
    return fig


def plot_pair_grid(
    rows: list[list[tuple[str, str, dict] | None]],
    spec: PairSpec,
    *,
    suptitle: str | None = None,
    panel_size: tuple[float, float] = (3.2, 2.8),
    pool_window: bool = False,
    stat_fontsize: float = DEFAULT_STAT_FONTSIZE,
) -> "plt.Figure | None":
    """Build (but do not save/show/close) one figure laid out exactly as
    ``rows`` specifies, rather than one-panel-per-group-of-a-single-mname
    like ``plot_pooled_pair_page``: each row is a list of panels, each
    panel either ``None`` (left blank -- rows need not have equal length;
    shorter rows are padded with blank panels out to the widest row) or an
    ``(label, mname, group)`` tuple, drawn via the same ``_plot_pair_panel``
    ``plot_pooled_pair_page`` itself uses. This is for a caller that wants
    to compare panels that don't share one ``mname`` (e.g. different
    matrices, different runs) side by side in a specific, hand-picked
    arrangement -- grouping rows by some caller-meaningful property (e.g.
    matrices of the same aspect ratio) rather than by group/mname alone.
    ``pool_window``: see ``_pool_fields`` -- off by default, so each panel
    uses only its own group's checkpoint/fork step.

    Returns the figure, or ``None`` if no panel had any plottable point.
    """
    nrows = len(rows)
    ncols = max(len(row) for row in rows)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size[0] * ncols + 0.6, panel_size[1] * nrows + 0.8),
        squeeze=False,
    )
    any_plotted = False
    for r, row in enumerate(rows):
        for c in range(ncols):
            ax = axes[r][c]
            cell = row[c] if c < len(row) else None
            if cell is None:
                ax.axis("off")
                continue
            label, mname, group = cell
            any_plotted |= _plot_pair_panel(
                ax,
                group,
                mname,
                spec,
                label,
                pool_window=pool_window,
                stat_fontsize=stat_fontsize,
            )
    if not any_plotted:
        plt.close(fig)
        return None

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=11)
    _finalize_figure_with_caption(
        fig, _fit_stats_caption("β", spec.fit_line), fontsize=stat_fontsize
    )
    return fig


def plot_spec_grid(
    mname: str,
    specs: list[PairSpec],
    groups: list[dict],
    *,
    mname_label: str | None = None,
    group_label: Callable[[dict], str] | None = None,
    pool_window: bool = False,
    stat_fontsize: float = DEFAULT_STAT_FONTSIZE,
) -> "plt.Figure | None":
    """One figure for a SINGLE matrix (``mname``): one row per spec in
    ``specs`` (each its own y-quantity against the same x, e.g. several
    different HVP-derived fields all plotted against ``sigma_i``), one
    column per group in ``groups`` (in the given order). Complements
    ``plot_pooled_pair_page`` (one spec, one panel per every group) and
    ``plot_pair_grid`` (one spec, an arbitrary hand-picked row/col layout
    of panels): this one is for comparing several DIFFERENT specs against
    the same small, hand-picked set of groups, for one matrix at a time --
    e.g. a compact few-checkpoint summary meant to sit inline in a
    notebook, alongside a fuller ``plot_pooled_pair_page`` sweep (every
    group, one spec at a time) saved elsewhere.

    Column headers (top row only) come from ``group_label`` (or each
    group's own raw name); every panel still gets its own row's ``spec.
    y_label``/``x_label`` and fit-stats annotation, same as
    ``plot_pooled_pair_page``. ``pool_window``: see ``_pool_fields`` --
    off by default, so each panel uses only its own group's
    checkpoint/fork step, not the extra steps profiled just after it.
    Returns ``None`` if no panel anywhere has any plottable point.
    """
    n_rows = len(specs)
    n_cols = len(groups)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols + 0.6, 2.6 * n_rows + 0.8),
        squeeze=False,
    )
    any_plotted = False
    for c, group in enumerate(groups):
        col_label = group_label(group) if group_label is not None else group["name"]
        for r, spec in enumerate(specs):
            ax = axes[r][c]
            any_plotted |= _plot_pair_panel(
                ax,
                group,
                mname,
                spec,
                "",
                pool_window=pool_window,
                stat_fontsize=stat_fontsize,
            )
        axes[0][c].set_title(col_label, fontsize=10)
    if not any_plotted:
        plt.close(fig)
        return None

    title = mname_label if mname_label is not None else _short_matrix_name(mname)
    fig.suptitle(title, fontsize=11)
    _finalize_figure_with_caption(
        fig, _fit_stats_caption("β", True), fontsize=stat_fontsize
    )
    return fig


def plot_fork_window_stability(
    mname: str,
    groups: list[dict],
    *,
    y_key: str,
    y_label: str,
    title: str,
    y_log: bool = True,
    mname_label: str | None = None,
    group_label: Callable[[dict], str] | None = None,
    stat_fontsize: float = DEFAULT_STAT_FONTSIZE,
) -> "plt.Figure | None":
    """Build (but do not save/show/close) one figure for ``mname``: grid
    of scatters, one panel per group that has MORE THAN ONE step (a group
    with a single checkpoint has nothing to show stability over), ``|y_key|``
    vs ``sigma`` with EVERY POINT colored by its own step's position within
    the group's own window (0 = that group's first/closest-to-fork step, up
    to its own last step) via a sequential colormap -- unlike every other
    page-building function in this module, which pools a group's steps
    together into one indistinguishable color. The same log-log fit line
    ``plot_pooled_pair_page`` draws (fit on the group's own full pooled
    data) is overlaid too.

    This is the direct visual check for whether pooling a group's several
    post-fork steps together (as every other plot in this module does) is
    actually justified: if the local Hessian geometry barely moves across
    that short window, differently colored points should fall on top of
    each other along the same curve rather than separating into
    same-colored bands. ``y_key`` values are plotted as ``|y_key|`` (sign
    dropped) since color here is spent on step-since-fork, not sign -- see
    ``plot_pooled_pair_page``'s ``color_by_sign`` for the sign story on
    the pooled page itself. Returns ``None`` if fewer than one such
    multi-step group has any plottable point for this matrix.

    ``mname_label``/``group_label``: same opt-in overrides as
    ``plot_pooled_pair_page`` (see that function's own docstring) -- text
    for the suptitle in place of ``_short_matrix_name(mname)``, and a
    function computing each panel's own title (prefixed to its own
    ``(d=0..N)`` suffix here) in place of the group's raw ``group["name"]``.
    ``stat_fontsize``: forwarded to both the per-panel annotation box and
    the shared caption.
    """
    multi_step_groups = [g for g in groups if len(g["records"]) > 1]
    if not multi_step_groups:
        return None
    n = len(multi_step_groups)
    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols + 0.8, 3.0 * nrows + 0.8), squeeze=False
    )
    any_plotted = False
    for idx, group in enumerate(multi_step_groups):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        label = group_label(group) if group_label is not None else group["name"]
        records = group["records"]
        step0 = int(records[0]["iter_num"])
        xs, ys, ds = [], [], []
        for rec in records:
            mat = rec["matrices"].get(mname)
            sigma = None if mat is None else mat.get("sigma")
            v = None if mat is None else mat.get(y_key)
            if sigma is None or v is None:
                continue
            sigma = sigma.numpy().astype(np.float64)
            v = v.numpy().astype(np.float64)
            xs.append(sigma)
            ys.append(v)
            ds.append(
                np.full(sigma.shape, int(rec["iter_num"]) - step0, dtype=np.float64)
            )
        if not xs:
            ax.set_axis_off()
            ax.set_title(f"{label} (missing)", fontsize=9)
            continue
        x, y_raw, d_arr = np.concatenate(xs), np.concatenate(ys), np.concatenate(ds)
        keep = np.isfinite(x) & np.isfinite(y_raw) & (x > 0)
        if y_log:
            keep &= y_raw != 0
        if not keep.any():
            ax.text(0.5, 0.5, "no points>0", ha="center", va="center")
            ax.set_title(label, fontsize=9)
            continue
        x, y_raw, d_arr = x[keep], y_raw[keep], d_arr[keep]
        y = np.abs(y_raw) if y_log else y_raw

        sc = ax.scatter(
            x, y, s=8, alpha=0.7, edgecolors="none", c=d_arr, cmap="viridis"
        )
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="step since fork")

        ax.set_xscale("log")
        ax.set_yscale("log" if y_log else "linear")
        _declutter_log_axis(ax, "x")
        if y_log:
            _declutter_log_axis(ax, "y")
        fit = _log_fit(x, y) if y_log else None
        if fit is not None:
            _plot_log_fit_line(ax, x, y, fit, x_log=True, y_log=True)
        _annotate_fit_stats(ax, x, y, fit, fontsize=stat_fontsize)
        ax.set_xlabel(r"$\sigma_i$")
        ax.set_ylabel(y_label)
        ax.set_title(f"{label} (d=0..{int(d_arr.max())})", fontsize=9)
        any_plotted = True
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")
    if not any_plotted:
        plt.close(fig)
        return None

    title_prefix = mname_label if mname_label is not None else _short_matrix_name(mname)
    fig.suptitle(
        f"{title_prefix}: {title}, colored by step since fork",
        fontsize=11,
    )
    _finalize_figure_with_caption(
        fig, _fit_stats_caption("β", y_log), fontsize=stat_fontsize
    )
    return fig


# field -> (value_tex, use_abs, x_log) for plot_ecdf_grid. value_tex is bare
# (no $ delimiters) LaTeX, already wrapped in |...| where use_abs is True --
# matches the sign/scale conventions PAIR_SPECS and plot_fork_window_stability
# already use for each of these same fields (e.g. gamma_diag/gamma_perp_diag
# are plotted as |.| since they can be negative -- non-convex directions --
# and a log x-axis can't show a non-positive value).
_ECDF_FIELD_SPECS: dict[str, tuple[str, bool, bool]] = {
    "sigma": (r"\sigma_i", False, True),
    "gamma_diag": (r"|\gamma_{ii}|", True, True),
    "alpha": (r"\alpha_i", False, True),
    "gamma_perp_diag": (r"|\gamma^\perp_{ii}|", True, True),
    "phi_samples": (r"|\phi_i|", True, True),
    "phi_perp_samples": (r"|\phi^\perp_i|", True, True),
}


def plot_ecdf_grid(
    mnames: list[str],
    groups: list[dict],
    *,
    field: str,
    linewidth: float = 1.1,
    cmap: str = "viridis",
    cmap_range: tuple[float, float] = (0.0, 0.82),
    mname_labels: dict[str, str] | None = None,
    group_label: Callable[[dict], str] | None = None,
) -> "plt.Figure | None":
    """One figure: one column per matrix in ``mnames`` (in the given
    order), one row per group in ``groups`` (in the given order); each
    panel overlays one ECDF of ``field`` PER RECORD (step) in that group
    for that matrix -- unlike every pooling function elsewhere in this
    module, records are never concatenated together here. ``field`` is any
    key in ``_ECDF_FIELD_SPECS`` (``"sigma"``, ``"gamma_diag"``,
    ``"alpha"``, ``"gamma_perp_diag"``, ``"phi_samples"``,
    ``"phi_perp_samples"`` -- the same per-mode quantities
    ``plot_fork_window_stability``'s own ``y_key`` accepts), which fixes
    the LaTeX label, whether ``|field|`` is taken first (needed for a
    signed quantity like ``gamma_diag`` since a log axis can't show a
    non-positive value), and whether the x-axis is log or linear -- so
    every caller gets the same label/sign/scale convention for a given
    field rather than repeating it at each call site.

    Each record's own ECDF is colored by its step-since-fork (0 = the
    group's first/closest-to-fork step) via ``cmap``, restricted to the
    ``cmap_range`` fraction of it (default trims viridis's own yellow tail;
    pass ``cmap_range=(0, 1)`` for the full colormap, or a different
    ``cmap`` name entirely) -- the same step-since-fork coloring convention
    ``plot_fork_window_stability`` uses for its own scatter -- rather than a
    single flat color, so which step is which stays visually distinguishable
    even where curves overlap almost exactly. ``linewidth`` sets every
    curve's own line width. This is the same kind of stability check
    ``plot_fork_window_stability`` runs for gamma/sigma, but for
    ``field``'s own marginal distribution: if a group's several post-fork
    steps have a near-identical distribution, their ECDFs should sit right
    on top of each other regardless of color; a single group with visibly
    separated ECDF curves means that quantity is still moving within that
    fork window. Each column shares its own x/y limits
    (``sharex="col"``/``sharey="col"``) so ECDF shape is directly
    comparable down a column. Panels with no data at all are left blank
    (axis off). Returns ``None`` if no panel has any plottable data.

    ``mname_labels``: optional ``{mname: label}`` mapping for each column's
    own title, in place of ``_short_matrix_name(mname)``. ``group_label``:
    same opt-in override as elsewhere in this module -- a function
    computing each row's own label in place of the group's raw
    ``group["name"]``.
    """
    value_tex, use_abs, x_log = _ECDF_FIELD_SPECS[field]
    mname_labels = mname_labels or {}
    n_rows = len(groups)
    n_cols = len(mnames)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols + 0.9, 1.8 * n_rows + 0.6),
        squeeze=False,
        sharex="col",
        sharey="col",
    )
    any_plotted = False
    for c, mname in enumerate(mnames):
        col_label = mname_labels.get(mname, _short_matrix_name(mname))
        axes[0][c].set_title(col_label, fontsize=10)
        for r, group in enumerate(groups):
            ax = axes[r][c]
            row_label = group_label(group) if group_label is not None else group["name"]
            plotted, d_max, trunc_cmap, norm = _plot_ecdf_panel(
                ax,
                group,
                mname,
                field,
                cmap=cmap,
                cmap_range=cmap_range,
                linewidth=linewidth,
            )
            if not plotted:
                continue
            if c == 0:
                ax.set_ylabel(row_label, fontsize=8)
            if r == n_rows - 1:
                ax.set_xlabel(rf"${value_tex}$")
            if c == n_cols - 1 and d_max > 0:
                sm = plt.cm.ScalarMappable(norm=norm, cmap=trunc_cmap)
                sm.set_array([])
                fig.colorbar(sm, ax=ax, fraction=0.08, pad=0.04, label="d")
            any_plotted = True
    if not any_plotted:
        plt.close(fig)
        return None
    fig.suptitle(
        rf"ECDF of ${value_tex}$ per step (overlaid within each fork window)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def _plot_ecdf_panel(
    ax,
    group: dict,
    mname: str,
    field: str,
    *,
    cmap: str,
    cmap_range: tuple[float, float],
    linewidth: float,
):
    """Draw one group's per-record ECDF overlay for ``(mname, field)`` into
    ``ax`` -- the body of ``plot_ecdf_grid``'s own double loop, factored out
    so ``plot_ecdf_grid_multi`` (several fields side by side in one figure)
    can share it instead of duplicating the drawing logic. Returns
    ``(plotted, d_max, trunc_cmap, norm)``: ``plotted`` is whether anything
    was drawn (``ax`` is turned off and the rest is meaningless if not);
    ``d_max``/``trunc_cmap``/``norm`` are what a caller needs to add its own
    shared colorbar (the color encodes steps since the group's own first
    record, the same convention throughout this module)."""
    value_tex, use_abs, x_log = _ECDF_FIELD_SPECS[field]
    base_cmap = plt.get_cmap(cmap)
    cmap_lo, cmap_hi = cmap_range
    trunc_cmap = LinearSegmentedColormap.from_list(
        "trunc", base_cmap(np.linspace(cmap_lo, cmap_hi, 256))
    )
    records = group["records"]
    step0 = int(records[0]["iter_num"])
    d_max = max((int(rec["iter_num"]) - step0 for rec in records), default=0)
    norm = Normalize(vmin=0, vmax=max(d_max, 1))
    plotted_any_record = False
    for rec in records:
        mat = rec["matrices"].get(mname)
        vals = None if mat is None else mat.get(field)
        if vals is None:
            continue
        vals = vals.numpy().astype(np.float64)
        if use_abs:
            vals = np.abs(vals)
        keep = np.isfinite(vals)
        if x_log:
            keep &= vals > 0
        vals = vals[keep]
        if vals.size == 0:
            continue
        vals_sorted = np.sort(vals)
        ecdf_y = np.arange(1, vals_sorted.size + 1) / vals_sorted.size
        d = int(rec["iter_num"]) - step0
        ax.plot(
            vals_sorted,
            ecdf_y,
            drawstyle="steps-post",
            lw=linewidth,
            color=trunc_cmap(norm(d)),
        )
        plotted_any_record = True
    if not plotted_any_record:
        ax.set_axis_off()
        return False, d_max, trunc_cmap, norm
    if x_log:
        ax.set_xscale("log")
        _declutter_log_axis(ax, "x")
    ax.set_ylim(0, 1)
    return True, d_max, trunc_cmap, norm


def plot_ecdf_grid_multi(
    mnames: list[str],
    groups: list[dict],
    fields: list[str],
    *,
    linewidth: float = 1.1,
    cmap: str = "viridis",
    cmap_range: tuple[float, float] = (0.0, 0.82),
    mname_labels: dict[str, str] | None = None,
    group_label: Callable[[dict], str] | None = None,
) -> "plt.Figure | None":
    """Like ``plot_ecdf_grid``, but for several ``fields`` at once, laid out
    side by side in ONE figure instead of one figure per field: columns are
    grouped into one block per field (in the given order), each block
    holding one column per matrix in ``mnames`` (in the given order); rows
    are still one per group in ``groups``. Each panel's own title names
    both its matrix and its field, since a bare matrix name is no longer
    enough to tell columns apart once several fields share the figure. The
    step-since-checkpoint colorbar is shown once, on the last column of the
    last field's block, since every panel shares the same color convention.
    Returns ``None`` if no panel anywhere has any plottable data.
    """
    mname_labels = mname_labels or {}
    n_rows = len(groups)
    n_mnames = len(mnames)
    n_cols = n_mnames * len(fields)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols + 0.9, 1.8 * n_rows + 0.6),
        squeeze=False,
    )
    any_plotted = False
    for f_idx, field in enumerate(fields):
        value_tex = _ECDF_FIELD_SPECS[field][0]
        for m_idx, mname in enumerate(mnames):
            c = f_idx * n_mnames + m_idx
            mname_label = mname_labels.get(mname, _short_matrix_name(mname))
            axes[0][c].set_title(rf"{mname_label}: ${value_tex}$", fontsize=10)
            col_axes = axes[:, c]
            for r in range(1, n_rows):
                col_axes[r].sharex(col_axes[0])
                col_axes[r].sharey(col_axes[0])
            for r, group in enumerate(groups):
                ax = axes[r][c]
                row_label = (
                    group_label(group) if group_label is not None else group["name"]
                )
                plotted, d_max, trunc_cmap, norm = _plot_ecdf_panel(
                    ax,
                    group,
                    mname,
                    field,
                    cmap=cmap,
                    cmap_range=cmap_range,
                    linewidth=linewidth,
                )
                if not plotted:
                    continue
                if c == 0:
                    ax.set_ylabel(row_label, fontsize=8)
                if r == n_rows - 1:
                    ax.set_xlabel(rf"${value_tex}$")
                if c == n_cols - 1 and d_max > 0:
                    sm = plt.cm.ScalarMappable(norm=norm, cmap=trunc_cmap)
                    sm.set_array([])
                    fig.colorbar(sm, ax=ax, fraction=0.08, pad=0.04, label="d")
                any_plotted = True
    if not any_plotted:
        plt.close(fig)
        return None
    fig.suptitle(
        "ECDF per step, overlaid within each checkpoint's own window",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def normalize_gamma(gamma: np.ndarray) -> np.ndarray:
    """``gamma_ij / sqrt(|gamma_ii * gamma_jj|)`` -- a covariance-to-
    correlation-style normalization. Uses ``|gamma_ii*gamma_jj|`` under the
    square root (not the raw product) so a negative diagonal entry (a
    non-convex direction -- ``gamma_ii`` isn't sign-constrained the way a
    variance is) still gives a well-defined, non-negative denominator
    instead of ``NaN``; the diagonal of the result is then
    ``sign(gamma_ii)`` (+-1), not always ``+1``. Entries where either
    diagonal is exactly 0 are ``NaN`` (no meaningful normalization)."""
    diag = np.diagonal(gamma)
    denom = np.sqrt(np.abs(np.outer(diag, diag)))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = gamma / denom
    out[denom == 0] = np.nan
    return out


def _draw_heatmap_panel(
    fig,
    ax,
    rec: dict,
    mname: str,
    field: str,
    transform: Callable[[np.ndarray], np.ndarray],
    dynamic_range: bool,
    label: str,
) -> bool:
    """Draw one heatmap panel (or a ``(missing)`` placeholder, if ``rec``
    has no ``field`` for ``mname``) onto ``ax``. Shared by
    ``plot_heatmap_pages`` (one step per panel) and ``plot_heatmap_page``
    (one group's fork step per panel) so the actual drawing logic --
    ``transform``, the 0-centered colormap, labels -- lives in exactly one
    place. Returns whether anything was actually drawn."""
    mat = rec["matrices"].get(mname)
    raw = None if mat is None else mat.get(field)
    if raw is None:
        ax.set_axis_off()
        ax.set_title(f"{label} (missing)", fontsize=9)
        return False
    # CenteredNorm (not a plain vmin/vmax Normalize) explicitly pins 0 to
    # the colormap's exact midpoint -- white, for RdBu_r -- regardless of
    # the display range.
    m = transform(raw.numpy().astype(np.float64))
    halfrange = max(1.0, float(np.nanmax(np.abs(m)))) if dynamic_range else 1.0
    im = ax.imshow(
        m,
        norm=CenteredNorm(vcenter=0.0, halfrange=halfrange),
        cmap="RdBu_r",
        origin="upper",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel(r"mode $j$")
    ax.set_ylabel(r"mode $i$")
    ax.set_title(label, fontsize=9)
    return True


def plot_heatmap_pages(
    mname: str,
    groups: list[dict],
    *,
    field: str,
    title: str,
    transform: Callable[[np.ndarray], np.ndarray],
    dynamic_range: bool,
):
    """Build (but do not save/show/close) one figure PER GROUP for
    ``mname``, one subplot per step actually profiled within that group
    (unlike ``plot_heatmap_page``, which shows only each group's fork step
    on a single combined figure) -- this is the version the CLI/PDF output
    uses, where seeing every step in a group's own post-fork window
    matters. Every page shares the SAME grid size (sized to whichever
    group has the most steps, so PDF pages don't come out a different
    physical size group to group) -- a group with fewer steps just leaves
    the extra panels blank rather than shrinking its own page. See
    ``_draw_heatmap_panel`` for ``transform``/``dynamic_range``. A
    GENERATOR (one ``yield`` per group with any plottable step -- groups
    with none are skipped): the caller decides what to do with each figure
    (save it to a ``PdfPages``, ``plt.show()`` it, close it, ...), e.g.
    ``for fig in plot_heatmap_pages(...): ...``."""
    max_records = max((len(g["records"]) for g in groups), default=0)
    if max_records == 0:
        return
    ncols = int(math.ceil(math.sqrt(max_records)))
    nrows = int(math.ceil(max_records / ncols))
    for group in groups:
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.2 * ncols + 0.6, 2.8 * nrows + 0.8), squeeze=False
        )
        any_plotted = False
        for idx, rec in enumerate(group["records"]):
            r, c = divmod(idx, ncols)
            label = f"iter {int(rec['iter_num'])}"
            if _draw_heatmap_panel(
                fig, axes[r][c], rec, mname, field, transform, dynamic_range, label
            ):
                any_plotted = True
        for idx in range(len(group["records"]), nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r][c].axis("off")
        if not any_plotted:
            plt.close(fig)
            continue

        fig.suptitle(
            f"{_short_matrix_name(mname)} [{group['name']}]: {title}", fontsize=11
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        yield fig


def plot_heatmap_page(
    mname: str,
    groups: list[dict],
    *,
    field: str,
    title: str,
    transform: Callable[[np.ndarray], np.ndarray],
    dynamic_range: bool,
    mname_label: str | None = None,
    group_label: Callable[[dict], str] | None = None,
) -> "plt.Figure | None":
    """Build (but do not save/show/close) one COMBINED figure for
    ``mname``: one panel PER GROUP (same layout as
    ``plot_pooled_pair_page``), each showing only that group's own FORK
    STEP (its first record, ``d=0``) -- not every step in the group's own
    post-fork window (see ``plot_heatmap_pages`` for that, one page per
    group instead, used by the CLI/PDF output). Picking just the fork step
    is what makes a single combined figure workable here: an ``(r, r)``
    matrix lives in its own step's mode basis, so unlike a scalar per-mode
    field (see ``_pool_fields``), different steps can't be pooled onto one
    panel the way ``plot_pooled_pair_page`` pools a whole group -- one
    representative step per group is the compromise. See
    ``_draw_heatmap_panel`` for ``transform``/``dynamic_range``. Returns
    ``None`` if no group has ``field`` for this matrix at its fork step.

    ``mname_label``/``group_label``: same opt-in overrides as
    ``plot_pooled_pair_page`` (see that function's own docstring) -- text
    for the suptitle in place of ``_short_matrix_name(mname)``, and a
    function computing each panel's own title (prefixed to its own
    ``(iter N)`` suffix here) in place of the group's raw ``group["name"]``.
    """
    n_groups = len(groups)
    ncols = int(math.ceil(math.sqrt(n_groups)))
    nrows = int(math.ceil(n_groups / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols + 0.6, 2.8 * nrows + 0.8), squeeze=False
    )
    any_plotted = False
    for idx, group in enumerate(groups):
        r, c = divmod(idx, ncols)
        rec = group["records"][0]  # the group's own fork step (d=0)
        label = group_label(group) if group_label is not None else group["name"]
        panel_label = f"{label} (iter {int(rec['iter_num'])})"
        if _draw_heatmap_panel(
            fig, axes[r][c], rec, mname, field, transform, dynamic_range, panel_label
        ):
            any_plotted = True
    for idx in range(n_groups, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")
    if not any_plotted:
        plt.close(fig)
        return None

    title_prefix = mname_label if mname_label is not None else _short_matrix_name(mname)
    fig.suptitle(f"{title_prefix}: {title}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_heatmap_grid(
    rows: list[tuple[str, str, str, Callable[[np.ndarray], np.ndarray], bool]],
    groups: list[dict],
    *,
    group_label: Callable[[dict], str] | None = None,
) -> "plt.Figure | None":
    """One combined figure: one row per ``(row_label, mname, field,
    transform, dynamic_range)`` entry in ``rows`` (e.g. several different
    matrix/field combinations), one column per group in ``groups`` (in the
    given order), each column using that group's own fork step (``d=0`` --
    a heatmap can't pool several steps onto one panel the way a scalar
    per-mode field can, see ``plot_heatmap_page``, whose own per-panel
    drawing this shares via ``_draw_heatmap_panel``). Column headers (top
    row) come from ``group_label`` (or each group's own raw name); row
    labels (leftmost column only) come from each row's own ``row_label``.
    Complements ``plot_heatmap_page`` (one matrix/field, one panel per
    EVERY group): this is for a compact, hand-picked few-checkpoint
    summary across several matrix/field combinations at once, in a single
    figure, meant to sit inline in a notebook. Returns ``None`` if no panel
    anywhere has any plottable data.
    """
    n_rows = len(rows)
    n_cols = len(groups)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.6 * n_cols + 0.6, 2.6 * n_rows + 0.6),
        squeeze=False,
    )
    any_plotted = False
    for c, group in enumerate(groups):
        col_label = group_label(group) if group_label is not None else group["name"]
        rec = group["records"][0]
        for r, (row_label, mname, field, transform, dynamic_range) in enumerate(rows):
            ax = axes[r][c]
            plotted = _draw_heatmap_panel(
                fig, ax, rec, mname, field, transform, dynamic_range, ""
            )
            any_plotted = any_plotted or plotted
            ax.set_title("")
            if c == 0:
                ax.set_ylabel(row_label, fontsize=9)
        axes[0][c].set_title(col_label, fontsize=10)
    if not any_plotted:
        plt.close(fig)
        return None
    fig.tight_layout()
    return fig


def _plot_phi_ratio_panel(
    ax,
    group: dict,
    mname: str,
    field: str,
    value_tex: str,
    rng: np.random.Generator,
    *,
    frac: float,
    jitter_scale: float,
    scatter_alpha: float,
    scatter_downsample: float,
    label: str,
    stat_fontsize: float = DEFAULT_STAT_FONTSIZE,
) -> bool:
    """Draw one group's ``field/sigma^p`` panel into ``ax`` -- the body of
    ``plot_phi_ratio_page``'s own per-group loop, factored out so
    ``plot_phi_ratio_grid`` (several matrix/field combinations against a
    hand-picked set of groups, in one combined figure) can share it. ``rng``
    is caller-owned and consumed in place (jitter, then the scatter
    downsample draw), so a caller building several panels from one ``rng``
    gets a single reproducible draw across all of them, the same convention
    ``plot_phi_ratio_page`` itself uses across its own panels. Returns
    whether anything was actually plotted."""
    pooled = _pool_phi_samples(group, mname, field)
    if pooled is None:
        ax.set_axis_off()
        ax.set_title(f"{label} (missing)", fontsize=9)
        return False
    sigma, y_signed = pooled
    keep = (sigma > 0) & np.isfinite(y_signed)
    if keep.sum() < 2:
        ax.text(0.5, 0.5, "not enough points", ha="center", va="center")
        ax.set_title(label, fontsize=9)
        return False
    sigma, y_signed = sigma[keep], y_signed[keep]
    fit = _log_fit(sigma, np.abs(y_signed))
    if fit is None:
        ax.text(0.5, 0.5, "not enough points to fit", ha="center", va="center")
        ax.set_title(label, fontsize=9)
        return False
    p, _alpha = fit
    y = y_signed / (sigma**p)
    x = sigma * np.exp(rng.normal(scale=jitter_scale, size=sigma.shape))

    # Same smooth-then-clip order/rationale as the log-log pair pages:
    # smoothing runs on the unclipped y so a few outliers don't bias
    # which window each median falls in, then the same clip bound
    # applies to both the scatter and the smoothed line.
    smoothed = _rolling_mean_smooth(x, y, frac=frac)
    bound = float(np.quantile(np.abs(y), 1.0 - RATIO_CLIP_QUANTILE))
    if bound > 0:
        y = np.clip(y, -bound, bound)
        if smoothed is not None:
            sx, smed, slo, shi = smoothed
            smoothed = (
                sx,
                np.clip(smed, -bound, bound),
                np.clip(slo, -bound, bound),
                np.clip(shi, -bound, bound),
            )

    n_points = x.size
    n_plot = max(1, int(round(n_points * scatter_downsample)))
    if n_plot < n_points:
        plot_idx = rng.choice(n_points, size=n_plot, replace=False)
    else:
        plot_idx = np.arange(n_points)
    ax.scatter(
        x[plot_idx],
        y[plot_idx],
        s=4,
        alpha=scatter_alpha,
        edgecolors="none",
        c="#1f77b4",
    )
    if smoothed is not None:
        sx, smed, slo, shi = smoothed
        ax.fill_between(sx, slo, shi, color="#d62728", alpha=0.15, linewidth=0)
        ax.plot(sx, smed, ls="-", lw=1.2, color="#d62728", alpha=0.9)
    ax.axhline(y=0, color="gray", linestyle="--")
    ax.set_xscale("log")
    _declutter_log_axis(ax, "x")
    ax.set_xlim(*_log_limits(x))
    _annotate_fit_stats(
        ax, sigma, np.abs(y_signed), fit, exponent_symbol="p", fontsize=stat_fontsize
    )
    ax.set_xlabel(r"$\sigma_i$ (jittered)")
    ax.set_ylabel(rf"${value_tex}/\sigma_i^p$ (clipped)")
    ax.set_title(label, fontsize=9)
    return True


def plot_phi_ratio_grid(
    rows: list[tuple[str, str, str, str]],
    groups: list[dict],
    *,
    frac: float = 0.1,
    jitter_scale: float = 0.02,
    rng_seed: int = 0,
    scatter_alpha: float = 0.2,
    scatter_downsample: float = 0.5,
    group_label: Callable[[dict], str] | None = None,
    stat_fontsize: float = DEFAULT_STAT_FONTSIZE,
) -> "plt.Figure | None":
    """Like ``plot_phi_ratio_page``, but for several ``(row_label, mname,
    field, value_tex)`` combinations at once, laid out as one row each in a
    SINGLE combined figure instead of one page per (mname, field): columns
    are ``groups`` (in the given order), shared across every row. One
    ``rng`` (seeded by ``rng_seed``) is consumed across the whole grid, row
    by row, so the figure as a whole is still reproducible even though it
    spans several matrix/field combinations. Row labels (leftmost column
    only) come from each row's own ``row_label``; column headers (top row)
    come from ``group_label`` (or each group's own raw name). Returns
    ``None`` if no panel anywhere has any plottable data.
    """
    rng = np.random.default_rng(rng_seed)
    n_rows = len(rows)
    n_cols = len(groups)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols + 0.6, 2.8 * n_rows + 0.8),
        squeeze=False,
    )
    any_plotted = False
    for c, group in enumerate(groups):
        col_label = group_label(group) if group_label is not None else group["name"]
        for r, (row_label, mname, field, value_tex) in enumerate(rows):
            ax = axes[r][c]
            plotted = _plot_phi_ratio_panel(
                ax,
                group,
                mname,
                field,
                value_tex,
                rng,
                frac=frac,
                jitter_scale=jitter_scale,
                scatter_alpha=scatter_alpha,
                scatter_downsample=scatter_downsample,
                label="",
                stat_fontsize=stat_fontsize,
            )
            any_plotted = any_plotted or plotted
            ax.set_title("")
            if c == 0:
                ax.set_ylabel(row_label, fontsize=9)
        axes[0][c].set_title(col_label, fontsize=10)
    if not any_plotted:
        plt.close(fig)
        return None
    _finalize_figure_with_caption(
        fig, _fit_stats_caption("p", has_fit=True), fontsize=stat_fontsize
    )
    return fig


def plot_phi_ratio_page(
    mname: str,
    groups: list[dict],
    *,
    field: str,
    value_tex: str,
    frac: float = 0.1,
    jitter_scale: float = 0.02,
    rng_seed: int = 0,
    scatter_alpha: float = 0.2,
    scatter_downsample: float = 0.5,
    mname_label: str | None = None,
    group_label: Callable[[dict], str] | None = None,
    stat_fontsize: float = DEFAULT_STAT_FONTSIZE,
) -> "plt.Figure | None":
    """Build (but do not save/show/close) one figure for ``mname``: grid
    of scatters, one panel PER GROUP, pooling one batch-mean point per
    (mode, step) of ``field`` from every step in that group (see
    ``_pool_phi_samples``, and its own docstring for why the per-example
    samples are averaged first rather than pooled raw). A power ``p`` is
    fit PER GROUP via ``_log_fit`` on ``log|field| ~ p*log(sigma)`` (using
    the TRUE, unjittered ``sigma``), then ``field/sigma^p`` (kept signed)
    is plotted against ``sigma`` -- jittered multiplicatively
    (``sigma * exp(N(0, jitter_scale))``, a fixed seed for reproducibility)
    purely so points sharing a very close ``sigma`` (e.g. the same mode
    across nearby steps) don't all stack on the same x position; the
    jitter is applied only to where each point is drawn, never to the fit
    or to the plotted y value itself. A centered, edge-preserving rolling
    median + IQR band (see ``_rolling_median_smooth``, ``frac`` of the
    group's own pooled sample count) is overlaid on the jittered x (so it
    visually aligns with the scatter). ``y`` is clipped to
    ``+-quantile(|y|, 1 - RATIO_CLIP_QUANTILE)`` (smoothing runs on the
    unclipped ``y`` first, then the same bound applies to both the scatter
    and the smoothed line) so a handful of blown-up points (small ``sigma``
    inflating ``field/sigma^p``) don't swamp the axis scale. ``value_tex``
    is the bare (no ``$`` delimiters) LaTeX for ``field``, e.g.
    ``r"\\phi_i"``. Returns ``None`` (nothing built) if no group has
    ``field`` for this matrix -- what to do with a built figure is
    entirely the caller's call.

    ``scatter_alpha``/``scatter_downsample``: since pooling is now one
    point per (mode, step) rather than one point per (mode, step,
    example), a group's own pooled count is the same order of magnitude
    as the other pooled quantities in this module (hundreds to a couple
    thousand, not tens of thousands), so neither a very low alpha nor
    downsampling is needed by default; both are still exposed for a
    caller with an unusually large group. ``scatter_downsample`` draws
    only that fraction of a group's own pooled points in the scatter (via
    ``rng``, the same one used for jitter) -- a PURELY VISUAL thinning for
    render time/file size; the fit (``p``/``_log_fit``), the
    rolling-median smoothing, the clip bound, and the annotated fit stats
    all still use every pooled point, never the downsampled subset.

    ``mname_label``/``group_label``: same opt-in overrides as
    ``plot_pooled_pair_page`` (see that function's own docstring) -- text
    for the suptitle in place of ``_short_matrix_name(mname)``, and a
    function computing each panel's own title in place of ``group["name"]``.
    """
    rng = np.random.default_rng(rng_seed)
    n_groups = len(groups)
    ncols = int(math.ceil(math.sqrt(n_groups)))
    nrows = int(math.ceil(n_groups / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols + 0.6, 2.8 * nrows + 0.8), squeeze=False
    )
    any_plotted = False
    for idx, group in enumerate(groups):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        label = group_label(group) if group_label is not None else group["name"]
        any_plotted |= _plot_phi_ratio_panel(
            ax,
            group,
            mname,
            field,
            value_tex,
            rng,
            frac=frac,
            jitter_scale=jitter_scale,
            scatter_alpha=scatter_alpha,
            scatter_downsample=scatter_downsample,
            label=label,
            stat_fontsize=stat_fontsize,
        )
    for idx in range(n_groups, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")
    if not any_plotted:
        plt.close(fig)
        return None

    title_prefix = mname_label if mname_label is not None else _short_matrix_name(mname)
    fig.suptitle(
        f"{title_prefix}: "
        rf"${value_tex}/\sigma_i^p$ vs $\sigma_i$ "
        rf"($p$ fit per group from $\log|{value_tex}| \sim p\log\sigma_i$, "
        "one batch-mean point per mode per step in the group)",
        fontsize=11,
    )
    _finalize_figure_with_caption(
        fig, _fit_stats_caption("p", has_fit=True), fontsize=stat_fontsize
    )
    return fig


def plot_curv_profile_scatter(
    data: dict, pdf_path: str, *, enabled: dict[str, bool] | None = None
) -> str:
    """One page per (matrix, pair) -- see this module's own docstring for
    the full page list/order. ``enabled`` maps a ``PairSpec.name`` to
    whether that page should be plotted at all; missing names default to
    enabled. Pages with no plottable data anywhere are skipped rather than
    written as an all-empty page, regardless of ``enabled``."""
    enabled = enabled or {}
    os.makedirs(os.path.dirname(os.path.abspath(pdf_path)) or ".", exist_ok=True)
    groups = data["groups"]

    def _save(fig) -> None:
        if fig is not None:
            pdf.savefig(fig)
            plt.close(fig)

    with PdfPages(pdf_path) as pdf:
        for mname in data["matrix_names"]:
            for spec in PAIR_SPECS:
                if spec.kind != "pair":
                    continue
                if not enabled.get(spec.name, True):
                    continue
                _save(plot_pooled_pair_page(mname, groups, spec, pool_window=True))

            if enabled.get("gamma_heatmap", True):
                for fig in plot_heatmap_pages(
                    mname,
                    groups,
                    field="gamma",
                    title=r"$\gamma_{ij}/\sqrt{|\gamma_{ii}\gamma_{jj}|}$",
                    transform=normalize_gamma,
                    dynamic_range=True,
                ):
                    _save(fig)

            if enabled.get("eta_perp_heatmap", True):
                for fig in plot_heatmap_pages(
                    mname,
                    groups,
                    field="eta_perp",
                    title=r"residual cross-correlation $\eta^\perp_{ij}=\langle Q_i,Q_j\rangle$",
                    transform=lambda m: m,
                    dynamic_range=False,
                ):
                    _save(fig)

            if enabled.get("phi_ratio", True):
                _save(
                    plot_phi_ratio_page(
                        mname, groups, field="phi_samples", value_tex=r"\phi_i"
                    )
                )

            if enabled.get("phi_perp_ratio", True):
                _save(
                    plot_phi_ratio_page(
                        mname,
                        groups,
                        field="phi_perp_samples",
                        value_tex=r"\phi^\perp_i",
                    )
                )

    return pdf_path


def analyze_curv_profile(
    run_path: str, *, out_dir: str | None = None, enabled: dict[str, bool] | None = None
) -> str:
    """Load every group under ``run_path`` (see
    ``load_curv_profile_groups``) and write the scatter PDF. ``out_dir``
    defaults to ``run_path`` itself."""
    out_dir = os.path.abspath(out_dir if out_dir is not None else run_path)
    os.makedirs(out_dir, exist_ok=True)
    data = load_curv_profile_groups(run_path)
    pdf_path = os.path.join(out_dir, PDF_FILE)
    return plot_curv_profile_scatter(data, pdf_path, enabled=enabled)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "run_path",
        help="Base directory a run_curv_profile.py config's --run_path "
        "pointed at (its immediate subdirectories are the groups this "
        "module pools by) -- see this module's own docstring.",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Where to write the PDF. Defaults to run_path itself.",
    )
    for spec in PAIR_SPECS:
        p.add_argument(
            f"--{spec.name.replace('_', '-')}",
            dest=spec.name,
            action=argparse.BooleanOptionalAction,
            default=True,
            help=spec.help or f"Include the {spec.title!r} page.",
        )
    args = p.parse_args()
    enabled = {spec.name: getattr(args, spec.name) for spec in PAIR_SPECS}
    pdf_path = analyze_curv_profile(
        args.run_path, out_dir=args.out_dir, enabled=enabled
    )
    print(pdf_path)


if __name__ == "__main__":
    main()
