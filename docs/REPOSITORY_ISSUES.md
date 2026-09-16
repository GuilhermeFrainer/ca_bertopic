# Repository findings tracker

Recorded on **2026-09-16**. Detailed evidence: [repository findings](docs/pairwise_repository_findings.md). Future comparison design: [proposal index](docs/pairwise_comparisons.md).

**Implementation handoff:** [experiment integrity repair plan](docs/experiment_integrity_repair_plan.md) covers historical verification, all active UMAP/PCA specifications, BERTopic default alignment, advisor-ready archival, run metadata, full-precision CSVs, and normalization fixes. It records a plan, not completed repairs.

Another agent is integrating MV-HDBSCAN concurrently. Recheck implementation status before acting; this checklist does not claim that fixes have been applied.

## Confirmed issues

- [ ] **Baseline uses 2 UMAP dimensions instead of the intended BERTopic default of 5. Owner: user.** The user explicitly confirmed that the baseline must use BERTopic defaults and will fix this. Runtime construction of the current FED baseline returned `n_components=2`, while plain `BERTopic()` returned 5. The repository supplies a UMAP object without `n_components`, causing UMAP's own default to apply. Although BERTopic technically accepts 2 dimensions, this is a **confirmed configuration bug relative to the intended experiment**, not an unresolved naming preference. Preserve the identity of old 2-dimensional results when introducing corrected runs.
- [ ] **Other baseline defaults need checking against the intended BERTopic defaults.** The same runtime check returned HDBSCAN `min_cluster_size=5` for the FED baseline and 10 for plain `BERTopic()`. The original audit also found dataset-specific HDBSCAN settings. Correcting UMAP dimensionality alone does not establish parity of every default. No baseline code/configuration was changed during this review.
- [x] **Normalization configuration is mutated during model construction.** *(Resolved)* [src/models.py](src/models.py) previously removed `normalize_text_view` and other parameters from the caller's parameter dictionary with `pop`. This has been fixed by introducing `copy.deepcopy` at all factory and runner entry points. Regression test suite added in [tests/test_config_mutation_regression.py](tests/test_config_mutation_regression.py).

## Analysis concerns

- [ ] **Match seed/topic-count grids before averaging.** Existing delta analysis averages available rows before matching; incomplete coverage can therefore compare different conditions. Completed individual runs are not invalid merely because other cells are missing.
- [ ] **Keep sample identities separate.** Inspected Yelp results include 500-document and full-size runs (10,205 observations in inspected full-size rows). Their coexistence is fine; pooling them as the same sample is not. Resolve the intended canonical Yelp input and separate `yelp_s10000` artifacts.
- [ ] **Review exact Wilcoxon handling.** Requesting SciPy `method="exact"` alone does not guarantee the required exact treatment of ties/zeros. The helper's exception-to-`p=1` fallback can conceal errors. This concerns statistical conclusions, not trained-model validity; it does not establish that every existing p-value is wrong. See the [statistical protocol](docs/pairwise_statistical_protocol.md).

## Comparison limitations and interpretation

- [ ] **K-means initialization differs:** single-view uses `n_init="auto"`; MV defaults to 5. Align settings for a controlled intervention or explicitly classify the comparison as architectural.
- [ ] **Spectral affinity bandwidth differs:** single-view uses RBF `gamma=1`; MV estimates bandwidth per view. Metadata is not the only changed component.
- [ ] **PCA and UMAP dimensionality differ in the inspected configurations:** PCA uses 5; UMAP generally uses 2. Recheck after the user's default-setting fix.
- [ ] **`info_view=0` is not metadata-free:** it selects the text-side embedding after multi-view fitting.
- [ ] **Geometry descriptions need precision:** text normalization occurs after reduction; spherical MV K-means normalizes both views; normalized-input Euclidean/RBF methods are not generally identical to the corresponding spherical/cosine methods.
- [ ] **Historical provenance is incomplete:** CSVs lack complete resolved configurations and sample fingerprints. This limits verification; it does not prove runs were wrong. Record provenance in future runs and state uncertainty for historical analyses.
- [ ] **Three-decimal metrics lose precision:** rounding can create ties and affects analysis resolution, not underlying training. Preserve full precision in future results.

## Clarification of “quarantine”

The intended meaning was **mark potentially affected normalized runs as unverified**, and temporarily exclude specific uncertain cells from claims about normalization until their execution history is checked.

Determine whether each run received a fresh configuration. If normalization was lost, correct the bug and rerun affected cells. Preserve original files and document any analytical exclusions. Historical affected files have not yet been identified.

This did not mean deleting/moving results, rejecting every experiment, or stopping MV-HDBSCAN integration. No such action was taken. The 2-dimensional baseline issue is separately confirmed against the user's stated intent and will be fixed by the user.

Unchecked entries indicate pending work or verification; they do not all imply code bugs or mandatory reruns.
