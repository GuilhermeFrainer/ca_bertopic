# Pairwise comparisons: registry design

Part of the [pairwise comparison proposal](pairwise_comparisons.md). This is a proposed schema, not an implemented registry or a finalized scientific protocol.

## Format and location

Use **YAML plus JSON Schema**, with explicit semantic validation in Python.

- YAML matches the existing experiment convention, supports comments, and is reviewable in Git.
- JSON Schema validates types/fields and rejects misspelled or unknown fields.
- Python validates references, graph cycles, configuration parity, source resolution, and experimental coverage.
- JSON would be less convenient to annotate; TOML would introduce another configuration convention without a clear benefit.
- A database or elaborate framework is unnecessary.

Proposed paths:

```text
comparisons/registry.yaml
comparisons/registry.schema.json
```

Keep the registry outside `experiments/` so existing training-configuration scanners do not mistake it for an experiment.

## Separate concepts

| Concept | Meaning |
| --- | --- |
| Node | One concrete experiment configuration, dataset, and preprocessing condition |
| Contrast | One directed baseline-node → variant-node pair |
| Comparison | Corresponding contrasts across datasets; produces one cross-dataset test per metric |
| Display panel | References comparisons to draw a hierarchy/DAG |
| Hypothesis family | Explicit comparisons and metrics corrected together |
| Protocol | Seeds, requested topic grid, preprocessing condition, aggregation, missingness/provenance rules, and testing conventions |
| Research question | The scientific claim that a comparison is intended to inform |

A concrete node must resolve to an experiment and result identifier; it is not a free-form model-family label. Repeated dataset instances are explicitly bound rather than guessed from naming conventions.

Keep **scientific priority** (`primary`, `secondary`, `exploratory`) separate from **lifecycle** (`planned`, `active`, `excluded`). A primary comparison can be planned or incomplete. Exclusions require explanations and must not depend on favorable performance.

The registry declares intended changes and parity requirements. Computed audits establish whether the claim of strict ablation is justified. Configuration parity and historical-run provenance are separate statuses.

No observed scores, p-values, or result-dependent “best model” designation belongs in the registry.

## Illustrative repository-specific example

The following is a small excerpt with one FED contrast. A five-dataset comparison would explicitly list five concrete contrasts. `c_v` and `irbo` are **illustrative candidates**, not selected primary metrics.

```yaml
schema_version: 1

protocols:
  standard:
    condition: standard
    seeds: [36201624, 62613654, 57116123]
    requested_topics: [10, 20, 30, 40, 50]
    aggregation: equal_weight_complete_grid
    missing_policy: error
    historical_provenance: require_verified
    test: exhaustive_signed_rank
    alternative: two-sided
    zero_method: pratt

metrics:
  c_v: {direction: maximize}
  irbo: {direction: maximize}

research_questions:
  metadata:
    question: Does incorporating document metadata improve topic quality?

nodes:
  fed_pca_text:
    config: experiments/fed/fed_standard_pca_k_means.yaml
    model_id: pca_k_means
    dataset: fed
    protocol: standard
    role: text_only
    requested_topics_field: model.clustering.params.n_clusters

  fed_pca_cast:
    config: experiments/fed/fed_standard_pca_mv_k_means.yaml
    model_id: pca_mv_k_means
    dataset: fed
    protocol: standard
    role: covariate_aware
    requested_topics_field: model.clustering.params.n_clusters

comparisons:
  pca_metadata:
    research_question: metadata
    priority: primary
    lifecycle: planned
    classification: architectural
    intervention: Replace single-view K-means with multi-view K-means
    contrasts:
      - baseline: fed_pca_text
        variant: fed_pca_cast
    declared_changes:
      - model.clustering.type
      - effective.clustering.n_init
    parity:
      config: pending
      historical_runs: unverified

hypothesis_families:
  metadata_primary:
    comparisons: [pca_metadata]
    metrics: [c_v, irbo]
    correction: holm

display:
  panels:
    - id: pca_family
      comparisons: [pca_metadata]
```

Paths in this proposed registry are repository-relative. Documentation links remain relative to their Markdown file.

## Validation requirements

Reject duplicate YAML keys and duplicate semantic IDs, unknown fields/model IDs, nonexistent node references, invalid metric directions, and cycles in displays expected to be acyclic. Do not allow a misspelled field to disappear silently.

Resolve every node's configuration, model ID, dataset, protocol, requested-count mechanism, and expected result identity. Validate edge direction, applicable datasets/metrics, family membership, lifecycle, and parity declarations.

Reject comparisons that silently mix preprocessing regimes, samples, or incompatible grids. A strict-ablation declaration must fail when effective configuration fields differ outside the declared intervention. Audit reports should expose differences rather than hide them behind broad component labels.

Primary incomplete or incompatible comparisons fail clearly unless the registry explicitly defines an approved, scientifically defensible alternative policy. Such policies must be fixed before examining performance.

## Fingerprints and display consistency

Generate configuration fingerprints into the analysis manifest; allow optional registry pins. Include effective settings, not merely raw YAML text.

A hash computed today identifies today's configuration. It cannot establish what a historical run used. Preserve that distinction in validation and reporting.

The graph generator must consume registered comparison references. Display arrows and tested comparisons must not be maintained as unrelated lists. Planned/excluded/architectural/verified-strict statuses should remain visible in generated displays.

See [pipeline architecture](pairwise_pipeline_architecture.md) for execution and [statistical protocol](pairwise_statistical_protocol.md) for family definitions and inference.
