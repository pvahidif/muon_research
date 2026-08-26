# pylint: disbale=all

from notebooks.utils import *
from notebooks.helpers import *

figures_dir = resolve_repo_path("website/articles/understanding-muon-part-I/figures")
STAT_FONTSIZE = (
    12  # fit-stat annotation/caption text size in analyze_curv_profile plots
)
os.makedirs(figures_dir, exist_ok=True)


def read_df_full(
    path: str, base_branch: str, *, rolling_window: int = 5
) -> pl.DataFrame:
    df_full = load_branch_full_history([resolve_repo_path(path)])
    print("df shape = ", df_full.shape)
    df_full = (
        df_full.group_by(
            "job_name",
            "branch_name",
            "fork_step",
            "job_checkpoint_step",
            "phase",
            "step",
            "job_reset_optimizer_state",
        )
        .agg(
            pl.col("train_loss").drop_nulls().first().alias("train_loss"),
            pl.col("val_loss").drop_nulls().first().alias("val_loss"),
        )
        .sort("job_name", "branch_name", "job_checkpoint_step", "step")
    )
    df_full = (
        df_full.join(
            df_full.filter(pl.col("branch_name").eq(base_branch)).select(
                "job_name",
                "fork_step",
                "job_checkpoint_step",
                "step",
                pl.col("val_loss").alias("base_val_loss"),
            ),
            on=["job_name", "fork_step", "job_checkpoint_step", "step"],
            how="left",
            coalesce=True,
            validate="m:1",
        )
        .with_columns(
            diff=pl.col("val_loss") - pl.col("base_val_loss"),
            step_since_fork=pl.col("step") - pl.col("fork_step"),
        )
        .with_columns(
            pl.col("diff")
            .rolling_mean(rolling_window, min_samples=1)
            .over(
                "job_name",
                "fork_step",
                "job_checkpoint_step",
                "branch_name",
                order_by="step_since_fork",
            ),
        )
    )

    # fork_step: the step, in the baseline Muon run, at which these branches
    # were forked and given their own (KL-matched) update power -- renamed
    # divergence_step. job_checkpoint_step: the step, further into that
    # divergent window, at which pow_continue takes back over and reunites
    # every branch under plain Muon -- renamed continue_step. steps_diverged
    # (= continue_step - divergence_step, one of 1/4/16 here) is how long each
    # branch got to run its own power before rejoining the others.
    df_full = df_full.rename(
        {"fork_step": "divergence_step", "job_checkpoint_step": "continue_step"}
    ).with_columns(
        steps_diverged=pl.col("continue_step") - pl.col("divergence_step"),
        # job_reset_optimizer_state is native only to "shared_continue" rows
        # (run_branch_continue.py's own config field, null on "fork_explore"
        # rows) -- broadcast the one real value across every row of the same
        # job, same convention load_branch_full_history uses for fork_step,
        # so it's never null.
        job_reset_optimizer_state=pl.col("job_reset_optimizer_state")
        .max()
        .over("job_name"),
    )
    return df_full


def branch_power(branch_name: str) -> float:
    """exp002_compare_muon_pow's own branch-naming convention (see that
    experiment's README): "muon" (regular Newton-Schulz Muon) and
    "svdp_z000" (its exact-SVD, power-0 reference) are both power 0;
    "svdp_{m|p}<3 digits>" is +/-<digits>/100 (e.g. "svdp_p025" -> +0.25).
    """
    if branch_name == "muon":
        return 0.0
    sign_char = branch_name.removeprefix("svdp_")[0]
    digits = branch_name.removeprefix("svdp_")[1:]
    sign = {"m": -1.0, "z": 0.0, "p": 1.0}[sign_char]
    return sign * int(digits) / 100


def plot_power_reunification_grid(df_full, base_branch, *, transpose=False):
    """One grid of small multiples: rows are the fork step (divergence_step)
    and columns are the length of the divergent window (steps_diverged), unless
    transpose=True, which swaps that (useful when there's only one divergent
    window length and several fork steps, so panels read left to right instead
    of stacking vertically). Each panel plots, for every non-reference branch,
    that branch's validation loss minus the reference branch's validation loss
    against steps since the fork; the vertical dotted line marks reunification,
    where every branch switches back to the same plain Muon rule."""
    divergence_steps = sorted(df_full["divergence_step"].unique())
    steps_diverged_vals = sorted(df_full["steps_diverged"].unique())
    power_vals = sorted(
        {branch_power(b) for b in df_full["branch_name"].unique() if b != base_branch}
    )
    hue_order = [str(v) for v in power_vals]
    palette = sns.color_palette(n_colors=len(power_vals))

    if transpose:
        row_vals, col_vals = steps_diverged_vals, divergence_steps
    else:
        row_vals, col_vals = divergence_steps, steps_diverged_vals

    fig, axes = plt.subplots(
        len(row_vals),
        len(col_vals),
        figsize=(4.2 * len(col_vals), 2.8 * len(row_vals)),
        squeeze=False,
    )
    for i, row_val in enumerate(row_vals):
        for j, col_val in enumerate(col_vals):
            divergence_step, steps_diverged = (
                (col_val, row_val) if transpose else (row_val, col_val)
            )
            ax = axes[i, j]
            facet_df = df_full.filter(
                pl.col("divergence_step").eq(divergence_step)
                & pl.col("steps_diverged").eq(steps_diverged)
                & pl.col("branch_name").ne(base_branch)
            ).with_columns(
                pl.col("branch_name")
                .map_elements(lambda b: str(branch_power(b)), return_dtype=pl.Utf8)
                .alias("branch_power_label")
            )
            show_legend = i == 0 and j == len(col_vals) - 1
            sns.lineplot(
                facet_df,
                x="step_since_fork",
                y="diff",
                hue="branch_power_label",
                hue_order=hue_order,
                errorbar=("pi", 100),
                palette=palette,
                linewidth=1.0,
                ax=ax,
                legend=show_legend,
            )
            ax.set_xscale("log")
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.axvline(
                steps_diverged + 0.5, color="black", linestyle=":", linewidth=0.8
            )
            if transpose:
                if i == 0:
                    ax.set_title(f"forked at step {divergence_step}")
                ax.set_xlabel("steps since fork" if i == len(row_vals) - 1 else "")
                ax.set_ylabel("diff" if j == 0 else "")
            else:
                if i == 0:
                    ax.set_title(f"divergent window = {steps_diverged} steps")
                ax.set_xlabel("steps since fork" if i == len(row_vals) - 1 else "")
                ax.set_ylabel(
                    f"forked at step {divergence_step}\ndiff" if j == 0 else ""
                )
            if show_legend:
                sns.move_legend(
                    ax,
                    "upper left",
                    bbox_to_anchor=(1, 1),
                    frameon=False,
                    title="power",
                )
    fig.tight_layout()
    return fig
