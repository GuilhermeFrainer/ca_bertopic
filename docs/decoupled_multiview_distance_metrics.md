# Decoupled Multi-View Distance Metrics

## 1. Overview & Motivation

Multi-view clustering integrates textual information (dense word/sentence embeddings or dimensionality-reduced representations) with non-textual document metadata (such as numerical ratings, dates, financial indicators, and survey scores).

A fundamental challenge in multi-view clustering is that different data modalities operate under fundamentally different geometric principles:

- **Textual Embeddings (Semantic View):** High-dimensional representations derived from language models (e.g. Sentence-BERT) encode semantic meaning primarily via **angular orientation / direction**. The absolute magnitude ($L_2$ norm) is largely an artifact of token length, word frequency, or pooling dynamics. Consequently, **Cosine distance** is the standard and mathematically appropriate metric.
- **Continuous Metadata (Tabular View):** Non-textual features represent physical or empirical quantities (e.g. star ratings, prices, economic metrics, years). Here, **absolute scale, distance, and magnitude carry essential meaning**. Consequently, **Euclidean distance** on normalized (MinMax or Z-score scaled) features is the appropriate metric.

---

## 2. Problem Statement: Metric Mismatch in Existing Multi-View Models

In the baseline multi-view models (primarily sourced from `mvlearn` and early fusion wrappers), a single distance metric was applied uniformly across all views:

1. **`MultiviewKMeans`:** Computes `scipy.spatial.distance.cdist(X, Y)` using standard **Euclidean distance for all views**. While acceptable on metadata, applying Euclidean distance directly to raw dense embeddings can suffer from norm variance and hubness.
2. **`MultiviewSphericalKMeans`:** L2 row-normalizes every view and computes **Cosine distance for all views**. When applied to continuous metadata, projecting features onto a unit hypersphere severely distorts Cartesian relationships. For example, two metadata vectors $[1.0, 2.0]$ and $[5.0, 10.0]$ have an identical angle ($\cos\theta = 1.0$), resulting in a cosine distance of $0.0$ despite representing completely different empirical quantities.
3. **`MultiviewSpectralClustering`:** Applies an identical affinity metric (e.g. RBF Gaussian kernel on Euclidean distance) to all views simultaneously.
4. **`AppendUMAP`:** Horizontally concatenates $[X_{\text{text}}, X_{\text{meta}}]$ into a single matrix and computes Cosine distance over the mixed features, mixing incomparable scales.

To resolve this limitation, we introduced two complementary solutions that allow decoupling distance metrics across views.

---

## 3. Mathematical Foundations

### 3.1 Monotonic Equivalence of Euclidean and Cosine Distance (Approach 1)

For any two unit-normalized vectors $u, v \in \mathbb{R}^d$ where $\|u\|_2 = \|v\|_2 = 1$:

$$\|u - v\|_2^2 = \|u\|_2^2 + \|v\|_2^2 - 2(u \cdot v) = 1 + 1 - 2\cos(u, v) = 2(1 - \cos(u, v))$$

Therefore:

$$\|u - v\|_2 = \sqrt{2 \cdot D_{\text{cosine}}(u, v)}$$

Because the function $f(z) = \sqrt{2z}$ is strictly monotonically increasing for $z \in [0, 2]$, the nearest-neighbor ranking and relative ordering under Euclidean distance on unit-normalized vectors is identical to that under Cosine distance.

By L2-normalizing **only the textual view** ($X_{\text{text}} \leftarrow X_{\text{text}} / \|X_{\text{text}}\|_2$) while preserving the original Cartesian scaling of the metadata view, standard Euclidean multi-view algorithms (like `MultiviewKMeans` and RBF `MultiviewSpectralClustering`) effectively apply **Cosine distance to the text view** and **Euclidean distance to the metadata view**.

### 3.2 Explicit Decoupled Expectation-Maximization (Approach 2)

In an explicit decoupled formulation, the total assignment objective for sample $i$ to cluster center $c$ is a weighted sum of modality-specific distances:

$$D_{\text{total}}(x_i, c) = w_0 \cdot D_{\text{cosine}}(x_{i, 0}, c_0) + w_1 \cdot D_{\text{euclid}}(x_{i, 1}, c_1)$$

Where:
- For View 0 (Cosine / Directional):
  $$D_{\text{cosine}}(x, c) = 1 - \frac{x \cdot c}{\|x\|_2 \|c\|_2}$$
  Centroids are re-projected onto the unit sphere during each EM step:
  $$c_0 \leftarrow \frac{\sum_{i \in \text{cluster}} x_{i, 0}}{\|\sum_{i \in \text{cluster}} x_{i, 0}\|_2}$$
- For View 1 (Euclidean / Cartesian):
  $$D_{\text{euclid}}(x, c) = \|x_{i, 1} - c_1\|_2$$
  Centroids are standard arithmetic means:
  $$c_1 \leftarrow \frac{1}{|C|} \sum_{i \in \text{cluster}} x_{i, 1}$$

---

## 4. Architectural Comparison: Pros and Cons

| Criteria | Approach 1: L2-Norm Equivalence | Approach 2: Decoupled Multi-View K-Means |
| :--- | :--- | :--- |
| **Component** | `MVCWrapper(normalize_text_view=True)` | `DecoupledMultiviewKMeans` estimator |
| **Mathematical Parity** | Monotonically equivalent to Cosine for text; exact Euclidean for metadata. | Exact Cosine distance with spherical centroid normalization; exact Euclidean for metadata. |
| **Centroid Updates** | Centroids in text view are Euclidean means (slightly shrunk inwards). | Centroids in text view are re-projected onto the unit hypersphere. |
| **View Weighting** | Unweighted ($w_0 = w_1 = 1.0$). | Fully configurable via `view_weights` (e.g. $[1.0, 1.0]$ or $[0.7, 0.3]$). |
| **Maintenance** | Minimal; purely wrapper-level preprocessing. | Requires maintaining dedicated clustering estimator in `src/decoupled_kmeans.py`. |
| **Supported Models** | Compatible with `multi_view_k_means`, `multi_view_spectral_clustering`, and `co_regularized_multi_view_spectral_clustering`. | Standalone estimator (`decoupled_multi_view_k_means`). |

---

## 5. Software Implementation Details

### 5.1 `MVCWrapper` (`src/mvc_wrapper.py`)
The [`MVCWrapper`](../src/mvc_wrapper.py) class now accepts an optional boolean `normalize_text_view` (default: `False`):
- When `normalize_text_view=True`, View 0 ($X$) is row-normalized by its L2 norm before being passed to the multi-view clustering model.
- View 1 (`self.metadata`) remains unaltered.
- When `normalize_text_view=False` (default), the data is passed through with 100% backward compatibility.

### 5.2 `DecoupledMultiviewKMeans` (`src/decoupled_kmeans.py`)
A custom clustering estimator inheriting from `BaseCluster` that:
- Accepts `view_metrics` (e.g. `["cosine", "euclidean"]`) and `view_weights` (e.g. `[1.0, 1.0]`).
- Normalizes centroids for cosine views during EM iterations and final consensus calculation.
- Selects the best initialization across random runs using minimal objective inertia.

### 5.3 `DecoupledMultiviewSpectralClustering` (`src/decoupled_spectral.py`)
A custom spectral clustering estimator inheriting from `MultiviewSpectralClustering` that:
- Accepts `view_affinities` (e.g. `["cosine", "rbf"]` or `["cosine_nn", "rbf"]`).
- Evaluates independent affinity graphs per modality (e.g. non-negative cosine similarity or cosine kNN for text; Gaussian RBF or Euclidean kNN for metadata).
- Solves for multi-view co-training graph Laplacians across the heterogeneous affinity matrices.

### 5.4 Model Factory (`src/models.py`)
The `get_algorithm` function supports:
1. Extracting `normalize_text_view` parameter from `clustering.params` for `multi_view_k_means`, `multi_view_spectral_clustering`, and `co_regularized_multi_view_spectral_clustering`.
2. Registering `type: "decoupled_multi_view_k_means"` (or `"hybrid_multi_view_k_means"`).
3. Registering `type: "decoupled_multi_view_spectral_clustering"` (or `"hybrid_multi_view_spectral_clustering"`).

---

## 6. Experiment Configuration (YAML Spec)

Both options integrate seamlessly into existing experiment configuration files:

### Option A: L2-Normalized Text View (K-Means & Spectral)
```yaml
# K-Means with L2-normalized text view
model:
  id: "mv_k_means_l2_norm"
  description: "UMAP + Multi-View K-Means (L2-Normalized Text View)"
  dimensionality_reduction:
    type: "umap"
    params:
      min_dist: 0.0
      metric: "cosine"
  clustering:
    type: "multi_view_k_means"
    params:
      normalize_text_view: true
      n_clusters: [10, 20, 30, 40, 50]

# Spectral Clustering with L2-normalized text view (exponential cosine kernel)
model:
  id: "mv_spectral_l2_norm"
  description: "UMAP + Multi-View Spectral Clustering (L2-Normalized Text View)"
  dimensionality_reduction:
    type: "umap"
    params:
      min_dist: 0.0
      metric: "cosine"
  clustering:
    type: "multi_view_spectral_clustering"
    params:
      normalize_text_view: true
      n_clusters: [10, 20, 30, 40, 50]
```

### Option B: Decoupled Multi-View Models (Explicit Metrics & Affinities)
```yaml
# Decoupled Multi-View K-Means
model:
  id: "decoupled_mv_k_means"
  description: "UMAP + Decoupled Multi-View K-Means (Cosine Text + Euclidean Metadata)"
  dimensionality_reduction:
    type: "umap"
    params:
      min_dist: 0.0
      metric: "cosine"
  clustering:
    type: "decoupled_multi_view_k_means"
    params:
      view_metrics:
        - "cosine"
        - "euclidean"
      view_weights:
        - 1.0
        - 1.0
      n_clusters: [10, 20, 30, 40, 50]

# Decoupled Multi-View Spectral Clustering
model:
  id: "decoupled_mv_spectral"
  description: "UMAP + Decoupled Multi-View Spectral Clustering (Cosine Text + Euclidean RBF Metadata)"
  dimensionality_reduction:
    type: "umap"
    params:
      min_dist: 0.0
      metric: "cosine"
  clustering:
    type: "decoupled_multi_view_spectral_clustering"
    params:
      view_affinities:
        - "cosine"
        - "rbf"
      n_clusters: [10, 20, 30, 40, 50]
```

---

## 7. Backward Compatibility

All existing YAML experiment configurations without `normalize_text_view` continue to function without any changes:
- `normalize_text_view` defaults to `False`.
- The factory populates standard `mvlearn` models as before when `normalize_text_view` is absent or set to `False`.
- Comprehensive regression and backward-compatibility tests are established in [`tests/test_decoupled_metrics.py`](../tests/test_decoupled_metrics.py).
