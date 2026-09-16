# Pairwise comparisons: proposed statistical protocol

Part of the [pairwise comparison proposal](pairwise_comparisons.md). Scientific choices here are recommendations pending approval. Primary metrics, hypothesis families, preprocessing scope, and historical-provenance policy remain open.

## Experimental units and aggregation

The five datasets are the independent units for cross-dataset inference. The three fixed seeds and five requested topic counts are repeated experimental conditions within each dataset. They must not be treated as 15 independent datasets or pooled into a 75-observation cross-dataset test.

For each comparison, metric, and dataset:

1. Require the registered 15 matched seed × requested-count cells.
2. Compute improvement-positive paired differences: variant minus baseline for maximized metrics, baseline minus variant for minimized metrics.
3. Average the three seeds within each requested count.
4. Average the five requested counts equally.
5. Preserve all run-level and count-level deltas.
6. Test the resulting five dataset-level differences.

On a complete rectangular grid, this equals the mean of all 15 paired differences. Equal dataset weights prevent larger datasets from dominating the cross-dataset summary. The same grid produces the baseline and variant dataset-level scores.

The estimand is average performance over the prespecified requested-count grid, not performance at the best observed count. Do not choose the aggregation by its p-value.

For HDBSCAN branches, the requested count is BERTopic's `nr_topics` reduction request; K-means/spectral use `n_clusters`. Preserve that mechanism and realized `n_topics` separately. Match requested conditions, not realized topic counts, because realized counts are outcomes.

Primary incomplete/incompatible grids must fail clearly. Do not silently drop seeds, counts, metrics, datasets, or failures. An alternative missingness policy requires an explicit scientific decision fixed before examining performance.

## Why planned pairwise tests

Demšar recommends Wilcoxon signed ranks for comparing two algorithms over multiple datasets. Benavoli, Corani, and Mangili explain that mean-rank post-hoc comparisons can depend on unrelated algorithms in the comparison pool, and recommend pairwise alternatives. These planned contrasts do not require a Friedman/Nemenyi tournament first.

Sources: [Demšar (2006)](https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf) and [Benavoli, Corani, and Mangili (2016)](https://jmlr.org/papers/v17/benavoli16a.html).

The repository's existing all-versus-all analysis can remain available for a different benchmark question. It should not determine which planned edges exist.

## Exact signed-rank computation

Recommend an **exhaustive signed-rank sign-flip calculation**, with a two-sided alternative:

- Rank absolute dataset deltas, using average ranks for tied magnitudes.
- Retain the repository's Pratt zero convention unless another policy is approved: zeros participate in ranking but contribute no signed rank.
- Enumerate all `2^m` sign assignments for the `m` nonzero differences.
- Compute a two-sided tail probability including outcomes equally extreme.
- Return `p=1` when all differences are zero, with a clear all-ties diagnostic.
- Reject missing/nonfinite inputs rather than silently omitting them or converting errors to `p=1`.
- Record the zero count, tied ranks, enumeration size, statistic, and method.

This gives an exact conditional calculation for the observed tied ranks/zeros. With five or six datasets exhaustive enumeration is trivial, so there is no need for Monte Carlo error or an asymptotic fallback.

The current repository helper's call to SciPy `method="exact"` is not sufficient: SciPy documents that ties and zeros change the null distribution and that this option does not by itself give the desired exact treatment. Its broad exception-to-`p=1` fallback must not conceal new-pipeline errors. See [SciPy's Wilcoxon documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).

### Assumptions and alternative

The test assumes independent dataset-level differences and sign symmetry under the null. Exhaustive computation does not eliminate these assumptions. With five selected datasets, conclusions concern this benchmark collection; generalization beyond it is limited.

Ties/zeros require a documented calculation, but do not by themselves force an asymptotic method or a different test. If symmetry is scientifically implausible, a prespecified exact sign test is a defensible alternative. Do not switch tests after comparing their p-values. Do not substitute a paired t-test without a strong justification.

A one-sided alternative requires a genuinely prior directional hypothesis and explicit approval. Improvement-positive coding does not make the test one-sided.

## Precision and ties

Historical CSV metrics are often rounded to three decimals. Preserve this information in the audit, avoid creating artificial rank distinctions from floating-point subtraction, and retain full precision in future results.

Specify deterministic numerical handling before analysis. Numerical equality handling must not silently become a scientific equivalence margin. Ties induced by stored precision should be disclosed; the original unrounded scores cannot be reconstructed from rounded CSVs.

## Multiplicity families

Recommended initial organization:

- One primary **metadata-incorporation family** spanning relevant primary comparisons across the visual trees.
- A separate **geometry-design family** for normalization and affinity interventions.
- Explicit secondary families where the underlying research question differs.

The visual trees must not automatically become correction families. Do not pool unrelated questions merely because they share code, or split a single question merely to obtain significance.

If either of two primary metrics can support the scientific claim, apply Holm across **comparisons × primary metrics** within the registered family. Primary coherence (`c_v` or `c_npmi`) and distinctiveness (IRBO or topic diversity) must be configurable and selected scientifically, not by observed performance. Other metrics can remain descriptively available under a stated secondary policy.

Holm's step-down adjustment is an established multiple-comparison option; see [García and Herrera (2008)](https://jmlr.org/papers/volume9/garcia08a/garcia08a.pdf). Reuse the repository's Holm helper after validating inputs and family construction.

A missing primary comparison should block final family-level inference rather than silently reduce the number of hypotheses. Preserve diagnostics when blocked. Planned comparisons, including future MV-HDBSCAN work, require an explicit protocol/lifecycle decision about when their family is finalized.

## Small-sample interpretation

With five nonzero differences, the smallest attainable exact two-sided p-value is **0.0625**. With six it is **0.03125**. Zeros reduce the effective sign-enumeration count and can make the minimum larger; multiplicity correction can increase reported adjusted p-values further.

Do not organize the report around `p < 0.05`, or relax the threshold just to obtain a positive finding. More seeds stabilize dataset estimates but do not increase the number of independent datasets.

Always show every dataset-level delta, wins/ties/losses, and effect magnitudes. A large consistent effect can be scientifically informative even when the small benchmark cannot produce a low p-value.

## Effect sizes and descriptive variation

Report baseline/variant dataset scores, improvement-positive deltas, mean and median delta across datasets, and wins/ties/losses.

A paired rank-biserial summary is useful alongside raw metric differences, provided its convention is explicit. With the proposed Pratt ranks, report the positive-minus-negative rank sum divided by their sum and label the zero/rank convention. It is undefined when that denominator is zero; do not invent a value. Its rank weighting does not replace the magnitude information in metric-scale deltas.

Preserve topic-count sensitivity and individual seed deltas. Use seed overlays/ranges descriptively, not as confidence intervals for cross-dataset inference. With so few datasets, avoid overstating uncertainty estimates.

Required table/figure outputs and reproducibility metadata are detailed in the [pipeline architecture](pairwise_pipeline_architecture.md).
