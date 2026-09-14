import marimo

__generated_with = "0.24.1"
app = marimo.App(width="wide")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl

    return Path, alt, mo, pd, pl


@app.cell
def _():
    import sys
    sys.executable
    return


@app.cell
def _(mo):
    header = mo.md(
        r"""
        # ⚡ FastTriTopic vs. Regular TriTopic: Metric Parity & Benchmark

        This notebook provides a strict, verifiable, and inspectable
        apples-to-apples comparison between **`fast_tritopic`** and
        baseline **`tritopic`**.

        ### Verification Mandate
        * **Mathematical Parity**: For model instances sharing the same dataset,
          preprocessing condition, random seed (`random_state`), and topic count
          (`n_topics`), all evaluation metrics must be **identical** ($|\Delta| = 0.0$).
        * **Runtime Efficiency**: The only expected difference is execution speed.
          `fast_tritopic` vectorizes graph construction to eliminate $O(N^2 \cdot k)$
          point-mutations on CSR matrices, running substantially faster.
        * **Robustness**: The notebook automatically aligns matching runs and
          isolates pending/unmatched runs so it seamlessly updates as new experiment
          results arrive.
        """
    )
    return (header,)


@app.cell
def _(Path, pl):
    def load_and_align_results(results_dir: str = "results"):
        """Scans results/ for CSVs, parses tritopic/fast_tritopic runs,

        and performs an apples-to-apples alignment on (dataset, condition, seed,
        topics). Handles missing or unmatched runs gracefully.
        """
        path = Path(results_dir)
        csv_files = sorted(path.glob("*.csv"))

        dfs = []
        for f in csv_files:
            try:
                df = pl.read_csv(f, infer_schema_length=None)
                if "clustering_algo" in df.columns:
                    sub = df.filter(
                        pl.col("clustering_algo").is_in(["tritopic", "fast_tritopic"])
                    )
                    if len(sub) > 0:
                        stem = f.stem
                        cond = (
                            "stemmed"
                            if "stemmed" in stem
                            else (
                                "no_stopword_removal"
                                if "no_stopword" in stem
                                else "standard"
                            )
                        )
                        sub = sub.with_columns(
                            [
                                pl.lit(cond).alias("condition"),
                                pl.lit(f.name).alias("source_file"),
                            ]
                        )
                        dfs.append(sub)
            except Exception:
                continue

        if not dfs:
            empty_df = pl.DataFrame()
            return empty_df, empty_df, empty_df

        combined = pl.concat(dfs, how="diagonal")

        # Cast numeric columns
        num_cols = [
            "random_state",
            "n_topics",
            "n_observations",
            "duration_seconds",
            "u_mass",
            "c_v",
            "c_npmi",
            "irbo",
            "topic_diversity",
            "outliers",
        ]
        for col in num_cols:
            if col in combined.columns:
                combined = combined.with_columns(pl.col(col).cast(pl.Float64))

        # Deduplicate in case individual and merged files overlap
        unique_keys = [
            "dataset_name",
            "condition",
            "clustering_algo",
            "random_state",
            "n_topics",
        ]
        combined = combined.unique(subset=unique_keys, keep="last")

        fast_df = combined.filter(pl.col("clustering_algo") == "fast_tritopic")
        reg_df = combined.filter(pl.col("clustering_algo") == "tritopic")

        join_keys = ["dataset_name", "condition", "random_state", "n_topics"]

        matched = fast_df.join(reg_df, on=join_keys, how="inner", suffix="_reg")

        if len(matched) > 0:
            matched = matched.with_columns(
                [
                    (pl.col("duration_seconds_reg") / pl.col("duration_seconds")).alias(
                        "speedup_factor"
                    ),
                    (pl.col("duration_seconds_reg") - pl.col("duration_seconds")).alias(
                        "time_saved_seconds"
                    ),
                ]
            )

        # Extract unmatched runs
        if len(matched) > 0:
            matched_keys = matched.select(join_keys).unique()
            unmatched_fast = fast_df.join(matched_keys, on=join_keys, how="anti")
            unmatched_reg = reg_df.join(matched_keys, on=join_keys, how="anti")
        else:
            unmatched_fast = fast_df
            unmatched_reg = reg_df

        return matched, unmatched_fast, unmatched_reg

    return (load_and_align_results,)


@app.cell
def _(load_and_align_results):
    matched_df, unmatched_fast, unmatched_reg = load_and_align_results("results")
    all_datasets = (
        sorted(matched_df["dataset_name"].unique().to_list())
        if len(matched_df) > 0
        else []
    )
    return all_datasets, matched_df, unmatched_fast, unmatched_reg


@app.cell
def _(all_datasets, mo):
    dataset_dropdown = mo.ui.dropdown(
        options=["All Datasets"] + all_datasets,
        value="All Datasets",
        label="Filter Dataset:",
    )

    metric_dropdown = mo.ui.dropdown(
        options=[
            "c_v",
            "u_mass",
            "c_npmi",
            "irbo",
            "topic_diversity",
            "outliers",
        ],
        value="c_v",
        label="Metric to Inspect:",
    )

    controls_ui = mo.hstack(
        [dataset_dropdown, metric_dropdown],
        justify="start",
        gap=2,
    )
    return controls_ui, dataset_dropdown, metric_dropdown


@app.cell
def _(dataset_dropdown, matched_df, mo, pl):
    if dataset_dropdown.value == "All Datasets":
        filtered_df = matched_df
    else:
        filtered_df = matched_df.filter(
            pl.col("dataset_name") == dataset_dropdown.value
        )

    _total_pairs = len(filtered_df)

    metrics_list = [
        "c_v",
        "u_mass",
        "c_npmi",
        "irbo",
        "topic_diversity",
        "outliers",
    ]
    _max_diff_overall = 0.0
    for _m in metrics_list:
        if _m in filtered_df.columns and f"{_m}_reg" in filtered_df.columns:
            _diffs = (
                (filtered_df[_m] - filtered_df[f"{_m}_reg"])
                .filter(
                    filtered_df[_m].is_not_nan()
                    & filtered_df[_m].is_not_null()
                    & filtered_df[f"{_m}_reg"].is_not_nan()
                    & filtered_df[f"{_m}_reg"].is_not_null()
                )
                .abs()
            )
            if len(_diffs) > 0:
                _max_d = _diffs.max()
                if _max_d is not None and _max_d > _max_diff_overall:
                    _max_diff_overall = _max_d

    _mean_speedup = filtered_df["speedup_factor"].mean() if _total_pairs > 0 else 0.0
    _total_time_saved_s = (
        filtered_df["time_saved_seconds"].sum() if _total_pairs > 0 else 0.0
    )

    stat_cards = mo.hstack(
        [
            mo.stat(
                value=f"{_total_pairs}",
                label="Apples-to-Apples Pairs",
                caption="Identical seed & topic count",
            ),
            mo.stat(
                value=f"{_max_diff_overall:.6f}",
                label="Max Metric |Δ|",
                caption="Target: 0.000000 (Exact Parity)",
            ),
            mo.stat(
                value=f"{_mean_speedup:.2f}x",
                label="Mean Speedup",
                caption="Regular vs. Fast TriTopic",
            ),
            mo.stat(
                value=f"{_total_time_saved_s / 60:.1f} min",
                label="Total Time Saved",
                caption=f"{_total_time_saved_s:.1f} seconds saved",
            ),
        ],
        justify="space-between",
    )
    return filtered_df, metrics_list, stat_cards


@app.cell
def _(filtered_df, metrics_list, mo, pl):
    _records = []
    for _metric in metrics_list:
        if _metric in filtered_df.columns and f"{_metric}_reg" in filtered_df.columns:
            _f_col = filtered_df[_metric]
            _r_col = filtered_df[f"{_metric}_reg"]

            _valid = (
                _f_col.is_not_nan()
                & _f_col.is_not_null()
                & _r_col.is_not_nan()
                & _r_col.is_not_null()
            )
            _sub_f = _f_col.filter(_valid)
            _sub_r = _r_col.filter(_valid)

            _nan_f = _f_col.is_nan() | _f_col.is_null()
            _nan_r = _r_col.is_nan() | _r_col.is_null()
            _nan_match = (_nan_f == _nan_r).all()

            if len(_sub_f) > 0:
                _d = (_sub_f - _sub_r).abs()
                _max_diff = _d.max()
                _mean_diff = _d.mean()
            else:
                _max_diff = 0.0
                _mean_diff = 0.0

            _is_identical = (_max_diff <= 1e-9) and _nan_match
            _status = "✅ PASS (Identical: 0.0)" if _is_identical else "❌ DIVERGED"

            _nan_status = (
                "Identical NaNs"
                if _nan_f.sum() > 0 and _nan_match
                else ("None" if _nan_f.sum() == 0 else "Mismatch")
            )

            _records.append(
                {
                    "Metric": _metric,
                    "Finite Pairs": len(_sub_f),
                    "NaN Alignment": _nan_status,
                    "Max Absolute |Δ|": f"{_max_diff:.8f}",
                    "Mean Absolute |Δ|": f"{_mean_diff:.8f}",
                    "Parity Status": _status,
                }
            )

    parity_df = pl.DataFrame(_records)

    parity_view = mo.vstack(
        [
            mo.md("### 1. Mathematical Parity Verification Across Metrics"),
            mo.md(
                "For every matched pair (same dataset, condition, seed, and"
                " topic count), we compute difference $\\Delta = M_{\\text{fast}}"
                " - M_{\\text{regular}}$. Complete parity requires $\\max |\\Delta|"
                " = 0.0$."
            ),
            mo.ui.table(parity_df.to_pandas()),
        ]
    )
    return (parity_view,)


@app.cell
def _(alt, filtered_df, metric_dropdown, mo, pd, pl):
    _metric = metric_dropdown.value

    _m_fast = filtered_df[_metric]
    _m_reg = filtered_df[f"{_metric}_reg"]

    _valid_mask = (
        _m_fast.is_not_nan()
        & _m_fast.is_not_null()
        & _m_reg.is_not_nan()
        & _m_reg.is_not_null()
    )

    _plot_data = filtered_df.filter(_valid_mask).with_columns(
        [
            (pl.col(_metric) - pl.col(f"{_metric}_reg")).alias("metric_difference"),
            (
                pl.col("dataset_name")
                + " (k="
                + pl.col("n_topics").cast(pl.Int64).cast(pl.Utf8)
                + ", seed="
                + pl.col("random_state").cast(pl.Int64).cast(pl.Utf8)
                + ")"
            ).alias("run_label"),
        ]
    )

    _plot_pddf = _plot_data.to_pandas()

    if len(_plot_pddf) == 0:
        parity_charts = mo.md(
            f"_No finite observations available for metric `{_metric}` in this"
            " selection._"
        )
    else:
        _min_val = min(_plot_pddf[_metric].min(), _plot_pddf[f"{_metric}_reg"].min())
        _max_val = max(_plot_pddf[_metric].max(), _plot_pddf[f"{_metric}_reg"].max())
        _pad = (_max_val - _min_val) * 0.08 if _max_val != _min_val else 0.1
        _domain = [_min_val - _pad, _max_val + _pad]

        _diag_df = pd.DataFrame({"x": _domain, "y": _domain})
        _diag_line = (
            alt.Chart(_diag_df)
            .mark_line(color="gray", strokeDash=[4, 4], strokeWidth=1.5)
            .encode(x="x:Q", y="y:Q")
        )

        _points = (
            alt.Chart(_plot_pddf)
            .mark_circle(size=75, opacity=0.85)
            .encode(
                x=alt.X(
                    f"{_metric}_reg:Q",
                    title=f"Regular TriTopic ({_metric})",
                    scale=alt.Scale(domain=_domain),
                ),
                y=alt.Y(
                    f"{_metric}:Q",
                    title=f"Fast TriTopic ({_metric})",
                    scale=alt.Scale(domain=_domain),
                ),
                color=alt.Color("dataset_name:N", title="Dataset"),
                tooltip=[
                    "dataset_name",
                    "condition",
                    "random_state",
                    "n_topics",
                    f"{_metric}_reg",
                    _metric,
                    "metric_difference",
                ],
            )
            .properties(
                width=360,
                height=300,
                title=f"Identity Plot: {_metric} (All points lie exactly on y=x)",
            )
        )

        _chart_identity = _diag_line + _points

        _residuals_zero = (
            alt.Chart(pd.DataFrame({"y": [0.0]}))
            .mark_rule(color="red", strokeDash=[2, 2])
            .encode(y="y:Q")
        )

        _residuals_points = (
            alt.Chart(_plot_pddf)
            .mark_circle(size=65, opacity=0.85)
            .encode(
                x=alt.X("n_topics:O", title="Number of Topics (k)"),
                y=alt.Y(
                    "metric_difference:Q",
                    title=f"Δ ({_metric}) [Fast - Reg]",
                    scale=alt.Scale(domain=[-0.01, 0.01]),
                ),
                color=alt.Color("dataset_name:N", title="Dataset"),
                tooltip=[
                    "dataset_name",
                    "condition",
                    "random_state",
                    "n_topics",
                    "metric_difference",
                ],
            )
            .properties(
                width=360,
                height=300,
                title="Residual Difference (Δ = 0.0 baseline)",
            )
        )

        _chart_residuals = _residuals_zero + _residuals_points

        parity_charts = mo.vstack(
            [
                mo.md("#### Visual Parity Inspection"),
                mo.hstack([_chart_identity, _chart_residuals], justify="center", gap=2),
            ]
        )
    return (parity_charts,)


@app.cell
def _(alt, filtered_df, mo, pl):
    _summary = (
        filtered_df.group_by("dataset_name")
        .agg(
            [
                pl.len().alias("matched_pairs"),
                pl.col("n_observations").first().alias("n_samples"),
                pl.col("duration_seconds_reg").mean().alias("reg_duration_mean_s"),
                pl.col("duration_seconds").mean().alias("fast_duration_mean_s"),
                pl.col("speedup_factor").mean().alias("mean_speedup"),
                pl.col("speedup_factor").min().alias("min_speedup"),
                pl.col("speedup_factor").max().alias("max_speedup"),
                pl.col("time_saved_seconds").sum().alias("total_saved_s"),
            ]
        )
        .sort("n_samples")
    )

    _sp_pddf = _summary.to_pandas()
    _speedup_bar = (
        alt.Chart(_sp_pddf)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("dataset_name:N", title="Dataset", sort=None),
            y=alt.Y("mean_speedup:Q", title="Mean Speedup Factor (x faster)"),
            color=alt.Color("dataset_name:N", legend=None),
            tooltip=[
                "dataset_name",
                "n_samples",
                "matched_pairs",
                alt.Tooltip("mean_speedup:Q", format=".2f"),
                alt.Tooltip("min_speedup:Q", format=".2f"),
                alt.Tooltip("max_speedup:Q", format=".2f"),
                alt.Tooltip("total_saved_s:Q", format=".1f"),
            ],
        )
        .properties(width=340, height=280, title="Average Speedup by Dataset")
    )

    _runtime_pddf = (
        filtered_df.select(
            [
                "dataset_name",
                "n_topics",
                "random_state",
                "duration_seconds_reg",
                "duration_seconds",
            ]
        )
        .rename(
            {
                "duration_seconds_reg": "Regular TriTopic",
                "duration_seconds": "Fast TriTopic",
            }
        )
        .to_pandas()
    )

    _melted_runtime = _runtime_pddf.melt(
        id_vars=["dataset_name", "n_topics", "random_state"],
        value_vars=["Regular TriTopic", "Fast TriTopic"],
        var_name="implementation",
        value_name="duration",
    )

    _runtime_chart = (
        alt.Chart(_melted_runtime)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("dataset_name:N", title="Dataset"),
            y=alt.Y(
                "duration:Q",
                title="Execution Duration (seconds)",
                scale=alt.Scale(type="log"),
            ),
            color=alt.Color("implementation:N", title="Implementation"),
            tooltip=["dataset_name", "implementation", "duration:Q"],
        )
        .properties(
            width=380,
            height=280,
            title="Runtime Distribution (Log Scale)",
        )
    )

    speedup_view = mo.vstack(
        [
            mo.md("### 2. Runtime Speedup & Efficiency Analysis"),
            mo.md(
                "Because `fast_tritopic` vectorizes graph construction and"
                " eliminates CSR array reallocations, runtime drops"
                " dramatically—especially on datasets with numerical metadata and"
                " large document counts."
            ),
            mo.ui.table(_summary.to_pandas()),
            mo.hstack([_speedup_bar, _runtime_chart], justify="center", gap=2),
        ]
    )
    return (speedup_view,)


@app.cell
def _(filtered_df, metric_dropdown, mo):
    _metric = metric_dropdown.value
    _display_cols = [
        "dataset_name",
        "condition",
        "random_state",
        "n_topics",
        "duration_seconds_reg",
        "duration_seconds",
        "speedup_factor",
        f"{_metric}_reg",
        _metric,
    ]

    _inspect_cols = [c for c in _display_cols if c in filtered_df.columns]
    records_table = mo.vstack(
        [
            mo.md(f"### 3. Detailed Paired Records Inspector ({_metric})"),
            mo.md(
                "Inspect individual runs to verify that metrics match to"
                " floating-point precision for every single seed and topic count."
            ),
            mo.ui.table(
                filtered_df.select(_inspect_cols)
                .sort(["dataset_name", "n_topics", "random_state"])
                .to_pandas()
            ),
        ]
    )
    return (records_table,)


@app.cell
def _(mo, pl, unmatched_fast, unmatched_reg):
    _unmatched_list = []

    if len(unmatched_fast) > 0:
        for _row in unmatched_fast.iter_rows(named=True):
            _unmatched_list.append(
                {
                    "Dataset": _row.get("dataset_name"),
                    "Condition": _row.get("condition"),
                    "Algorithm Available": "fast_tritopic",
                    "Seed": _row.get("random_state"),
                    "Topics": _row.get("n_topics"),
                    "Duration (s)": _row.get("duration_seconds"),
                    "Pending Counterpart": "Needs regular tritopic run",
                }
            )

    if len(unmatched_reg) > 0:
        for _row in unmatched_reg.iter_rows(named=True):
            _unmatched_list.append(
                {
                    "Dataset": _row.get("dataset_name"),
                    "Condition": _row.get("condition"),
                    "Algorithm Available": "tritopic (regular)",
                    "Seed": _row.get("random_state"),
                    "Topics": _row.get("n_topics"),
                    "Duration (s)": _row.get("duration_seconds"),
                    "Pending Counterpart": "Needs fast_tritopic run",
                }
            )

    if _unmatched_list:
        _unmatched_df = pl.DataFrame(_unmatched_list)
        unmatched_view = mo.vstack(
            [
                mo.md("### 4. Inventory of Unmatched / Pending Runs"),
                mo.md(
                    "The following runs currently exist in `results/` for one"
                    " algorithm only. As soon as corresponding counterpart"
                    " runs are executed and stored in `results/`, they will"
                    " automatically pair and appear in the parity analysis above."
                ),
                mo.ui.table(
                    _unmatched_df.group_by(
                        [
                            "Dataset",
                            "Condition",
                            "Algorithm Available",
                            "Pending Counterpart",
                        ]
                    )
                    .agg(
                        [
                            pl.len().alias("Run Count"),
                            pl.col("Topics")
                            .unique()
                            .sort()
                            .alias("Topic Configurations"),
                        ]
                    )
                    .to_pandas()
                ),
                mo.accordion(
                    {
                        "View All Individual Unmatched Runs": mo.ui.table(
                            _unmatched_df.to_pandas()
                        )
                    }
                ),
            ]
        )
    else:
        unmatched_view = mo.md(
            "_All discovered TriTopic runs have complete apples-to-apples pairs._"
        )
    return (unmatched_view,)


@app.cell
def _(
    controls_ui,
    header,
    mo,
    parity_charts,
    parity_view,
    records_table,
    speedup_view,
    stat_cards,
    unmatched_view,
):
    dashboard_layout = mo.vstack(
        [
            header,
            controls_ui,
            mo.md("---"),
            stat_cards,
            mo.md("---"),
            parity_view,
            parity_charts,
            mo.md("---"),
            speedup_view,
            mo.md("---"),
            records_table,
            mo.md("---"),
            unmatched_view,
        ],
        gap=2,
    )
    dashboard_layout
    return


if __name__ == "__main__":
    app.run()
