# Pairwise comparison pipeline: proposal index

This proposal organizes model evaluation around explicit research claims and controlled pairwise comparisons. It records the repository inspection and design discussion from **2026-09-16**.

**Status:** documentation of a proposal. The pipeline, registry, experiments, and fixes described here have not been implemented or scientifically approved. Writing these documents does not approve those changes.

## Reading guide

| Document | Read it to understand |
| --- | --- |
| [Repository findings](pairwise_repository_findings.md) | How experiments/results work today, observed coverage, naming mismatches, and correctness/provenance problems |
| [Comparison trees](pairwise_comparison_trees.md) | The proposed branches, what each edge tests, which are architectural comparisons, and missing controls |
| [Registry design](pairwise_registry_design.md) | The proposed YAML format, formal validation, and a concrete example |
| [Pipeline architecture](pairwise_pipeline_architecture.md) | Loading, parity auditing, matching, modules, outputs, and reproducibility |
| [Statistical protocol](pairwise_statistical_protocol.md) | Aggregation, exact inference, multiplicity, effect sizes, and small-sample limitations |
| [Implementation plan and decisions](pairwise_implementation_plan.md) | Small implementation stages, required tests, and questions still needing answers |

For the shortest route, read the comparison trees and then the open decisions. Consult the other documents as needed.

## Objective and scope

The aim is to supplement the existing broad model comparisons with parallel ablation hierarchies or directed comparison graphs. Adjacent models should answer a specific question about adding metadata, changing normalization, modifying modality-specific geometry, or replacing an architecture.

The user's starting branches are:

1. Default BERTopic → MV-HDBSCAN, which the user is implementing separately.
2. BERTopic with PCA + K-means → CAST with PCA + MV K-means.
3. BERTopic with UMAP + spectral clustering → CAST with UMAP + MV spectral clustering.

The user also requested comparisons of AppendUMAP and AlignedUMAP CAST variants against Default BERTopic, and proposals for deeper branches using existing models.

The proposal uses a **directed graph**, rather than requiring every model to have one parent. A model may participate in several scientifically useful contrasts. Display trees, statistical contrasts, and multiplicity families are separate concepts.

An edge is a **strict ablation** only when its declared intervention is the only material difference and parity has been verified. Comparisons changing several components remain useful, but are labeled **architectural comparisons**. An arrow does not claim an improvement.

The registry must not contain observed performance, p-values, or result-dependent “best model” labels. Primary choices must not be selected by inspecting which results are strongest.

## Boundaries

- Existing data and result artifacts are inputs and must remain untouched.
- Current results and configurations are a historical snapshot, not a guarantee about later repository state.
- Proposed fixes, new experiments, and pipeline implementation require the user's subsequent approval.
- The first recommended implementation deliverable is a registry and **audit-only** CLI, before inferential reporting.
- The five datasets are the independent units for cross-dataset inference. Seeds and topic counts are repeated conditions within datasets.

The initial inspection found a clean working tree and made no edits. At the later documentation-writing step, `pyproject.toml` and `uv.lock` had existing user changes; those were left untouched.
