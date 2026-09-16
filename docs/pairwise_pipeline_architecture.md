# Pairwise comparisons: pipeline architecture

Part of the [pairwise comparison proposal](pairwise_comparisons.md). No modules described here have been implemented as part of this documentation task.

## Proposed module layout

Use a small `src/comparisons/` package, following the repository's separation of library code from command-line scripts.

| Component | Responsibility |
| --- | --- |
| `src/comparisons/registry.py` | YAML/schema loading, typed records, references, lifecycle, graph validation |
| `src/comparisons/provenance.py` | Resolve configurations, materialize effective defaults, audit declared differences |
| `src/comparisons/results.py` | Discover sources, normalize identifiers, preserve source lineage and numeric precision |
| `src/comparisons/matching.py` | Construct expected grids, resolve duplicates, report missing/incompatible cells |
| `src/comparisons/analysis.py` | Paired differences, dataset aggregation, exact tests, Holm, effect summaries |
| `src/comparisons/reporting.py` | Tables, diagnostics, figures, graph generation |
| `scripts/analysis/compare_models.py` | Thin command-line entry point |

Reuse the current config resolver, optimizer grid semantics, Holm implementation, and plotting/export conventions where appropriate. Do not reuse permissive model grouping, available-row averaging, or default algorithm exclusions.

Formal field validation belongs in JSON Schema; cross-record and scientific semantics belong in explicit Python validation. Typed records can remain lightweight and compatible with existing repository conventions.

## Processing sequence

1. Parse and validate the registry.
2. Resolve concrete configurations and expected result identities.
3. Materialize effective settings and audit the intended differences.
4. Discover relevant result files and preserve file/row provenance.
5. Normalize rows into a tidy representation.
6. Construct expected grids and classify observed cells.
7. Match baseline/variant rows and enforce parity/coverage policies.
8. Compute paired differences and dataset-level summaries.
9. Perform registered inference and family-level correction.
10. Write reproducible diagnostics, tables, and optional graphs/figures.

The initial CLI should stop after the audit stages. Inferential output is a later deliverable.

## Configuration and provenance audit

Compare both explicit YAML fields and **effective behavior**, including defaults contributed by library versions. Resolve from copies so auditing/model construction cannot mutate the registry's configuration objects.

Audit at least:

- Dataset identity, document IDs/content/order, and sample identity.
- Text column, preprocessing regime, embeddings, and their generating settings.
- Metadata variables, encoding/scaling, and the sample used to fit transformations.
- Dimensionality reduction and all material effective parameters.
- Clustering implementation/settings outside the declared intervention.
- Requested topic count and its mechanism.
- Training seed and sampling seed separately.
- Topic representation, stopword policy, evaluation tokenization/settings, and metric definitions.
- Code and dependency versions.

Strict-ablation status must be earned by the audit. Missing historical provenance remains unknown; matching today's YAML and equal `n_observations` are not proof of historical parity.

Configuration labels and nonbehavioral descriptions need not make two models incompatible, but exclusions from comparison must be explicit and limited. A broad permission to change a whole subtree must not conceal an undeclared intervention.

## Result discovery and tidy data

Read existing artifacts without running the merge/archival cleanup utility. Start with registered relevant CSVs; archive discovery can be added read-only if needed.

A normalized run record should retain node/comparison bindings, dataset, condition, source file/row, experiment/run identifiers, training and sampling seeds where available, requested count and mechanism, realized count, sample identity or uncertainty, observation count, metric/value, status/evidence, and precision/provenance metadata.

Preserve separate identities for ordinary Yelp and its subsamples. Do not strip suffixes in a way that collapses different samples. Preserve `_info0` variants as distinct nodes.

The loader must not silently accept unknown models, discard absent metrics, turn failed runs into ordinary missing successes, or infer successful completion merely from an expected grid.

## Matching and duplicate policy

Construct the expected dataset × seed × requested-count grid from the registry. Match concrete nodes on the declared condition and all relevant parity keys.

Classify missing, documented failure, invalid metric, sample mismatch, unresolved provenance, duplicate candidate, and valid matched cell separately. Report both sides of a contrast.

Duplicate candidates must be reported and resolved by a recorded, deterministic policy. Do not average duplicates or choose results by performance. Selecting the latest timestamp is not sufficient if configuration/sample identity differs. Existing merge deduplication lacks complete identity and should not decide this silently.

Primary comparisons require complete compatible grids unless an approved policy explicitly says otherwise. Diagnostic outputs should remain available even when final inference fails. Unknown provenance cannot silently become verified provenance.

Do not match on realized topic counts: they are outcomes. Do not use model-name combination indices as requested counts without an unambiguous configuration mapping.

## Future-run instrumentation

A separate approved change should persist:

- Full-precision metric values.
- Complete resolved configuration, effective settings, and configuration fingerprint.
- Code/dependency versions.
- Document/sample and embedding fingerprints, including ordering.
- Training and sampling seeds.
- Explicit run success/failure records and useful failure details.

These additions cannot retroactively prove old runs' identity. Historical analysis needs its own stated provenance policy.

## Required outputs

For every registered comparison and metric, plan to report:

- Expected, observed, matched, missing, failed, invalid, and provenance-incompatible cells.
- Per-dataset baseline score, variant score, and improvement-positive delta.
- Mean and median dataset delta, and wins/ties/losses.
- Signed-rank statistic, raw exact p-value, and Holm-adjusted p-value within the registered family.
- Effect-size summary with an explicit zero/rank convention.
- Precision/provenance notes, nonzero dataset count, and attainable-p-value limitation.
- Run-level and topic-count-level differences for sensitivity analysis.

Use dedicated subdirectories beneath the established [results/](../results), [tables/](../tables), and [output/](../output) locations. Keep generated analyses distinguishable from training result files so discovery does not ingest its own outputs.

Write a manifest containing registry/code/input hashes, configuration fingerprints, source-selection decisions, protocol settings, and output identities. Outputs should be reproducible from the registered protocol, committed code, and identified inputs.

## Minimal figures

The recommended main set is:

1. A DAG generated from registered comparisons, with status/classification visible.
2. Dataset-level delta plots centered at zero.
3. Topic-count sensitivity plots retaining individual seed deltas.

Paired baseline/variant dumbbell plots are an optional alternative when absolute scores add useful context. Avoid redundant plots.

Seed ranges or overlays describe within-dataset variation; label them accordingly. They are not confidence intervals across independent datasets.

See the [statistical protocol](pairwise_statistical_protocol.md) and [implementation stages](pairwise_implementation_plan.md).
