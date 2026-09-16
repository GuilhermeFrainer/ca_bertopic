import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger("pipeline")

PROVENANCE_SCHEMA_VERSION = 2
DEFAULT_CAMPAIGN_ID = "bertopic_defaults_v2"

PROVENANCE_COLUMNS = [
    "result_schema_version",
    "campaign_id",
    "run_status",
    "dim_red_n_components",
    "dim_red_output_dim",
    "dim_red_n_neighbors",
    "dim_red_metric",
    "dim_red_min_dist",
    "dim_red_low_memory",
    "cluster_min_cluster_size",
    "cluster_min_samples",
    "cluster_metric",
    "cluster_selection_method",
    "cluster_prediction_data",
    "normalize_text_view",
    "resolved_config_hash",
    "run_manifest_path",
    "code_revision",
    "code_dirty",
    "dependency_lock_hash",
]


def get_git_info() -> Tuple[str, bool]:
    """
    Returns the current git commit hash and dirty status.
    Falls back safely if git is unavailable or repo is detached.
    """
    try:
        rev_output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
    except Exception:
        rev_output = "unknown"

    try:
        status_output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
        is_dirty = len(status_output) > 0
    except Exception:
        is_dirty = False

    return rev_output, is_dirty


def get_dependency_lock_hash() -> str:
    """
    Returns SHA-256 hash of uv.lock if it exists, otherwise 'unknown'.
    """
    lock_file = Path(__file__).resolve().parents[1] / "uv.lock"
    if lock_file.exists():
        try:
            return hashlib.sha256(lock_file.read_bytes()).hexdigest()[:16]
        except Exception:
            return "unknown"
    return "unknown"


def compute_config_hash(config: Dict[str, Any]) -> str:
    """
    Computes a deterministic SHA-256 hash of a configuration dictionary.
    """
    try:
        serialized = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return "hash_error"


def observe_reducer_output_dim(topic_model: Any) -> Optional[int]:
    """
    Observes the actual fitted dimensionality of the text reducer.
    Inspects fitted embeddings or components without re-transforming data.
    """
    if topic_model is None:
        return None

    umap_model = getattr(topic_model, "umap_model", None)
    if umap_model is None:
        return None

    # Plain UMAP or AppendUMAP (stores embedding_ array after fit)
    embedding = getattr(umap_model, "embedding_", None)
    if (
        embedding is not None
        and hasattr(embedding, "shape")
        and len(embedding.shape) > 1
    ):
        return int(embedding.shape[1])

    # AlignedUMAPWrapper (stores training_embeddings array after fit)
    training_embs = getattr(umap_model, "training_embeddings", None)
    if (
        training_embs is not None
        and hasattr(training_embs, "shape")
        and len(training_embs.shape) > 1
    ):
        return int(training_embs.shape[1])

    # PCA (stores n_components_ integer or components_ array after fit)
    n_components_ = getattr(umap_model, "n_components_", None)
    if n_components_ is not None and isinstance(n_components_, (int, float)):
        return int(n_components_)

    components = getattr(umap_model, "components_", None)
    if (
        components is not None
        and hasattr(components, "shape")
        and len(components.shape) > 0
    ):
        return int(components.shape[0])

    # Wrapped models (e.g. wrapper.model)
    inner_model = getattr(umap_model, "model", None)
    if inner_model is not None:
        inner_emb = getattr(inner_model, "embedding_", None)
        if (
            inner_emb is not None
            and hasattr(inner_emb, "shape")
            and len(inner_emb.shape) > 1
        ):
            return int(inner_emb.shape[1])

        inner_embs = getattr(inner_model, "embeddings_", None)
        if isinstance(inner_embs, list) and len(inner_embs) > 0:
            first_slice = inner_embs[0]
            if hasattr(first_slice, "shape") and len(first_slice.shape) > 1:
                return int(first_slice.shape[1])

    return None


def collect_run_provenance(
    topic_model: Any,
    model_config: Dict[str, Any],
    run_id: str,
    campaign_id: str = DEFAULT_CAMPAIGN_ID,
    run_status: str = "success",
    run_manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Collects runtime provenance, effective estimator hyperparameters,
    observed fitted output dimensionality, and execution identity.
    """
    git_rev, git_dirty = get_git_info()
    config_hash = compute_config_hash(model_config)
    dep_lock_hash = get_dependency_lock_hash()

    # 1. Reducer Provenance
    dr_config = model_config.get("dimensionality_reduction") or {}
    dr_params = dr_config.get("params") or {}
    umap_model = getattr(topic_model, "umap_model", None)

    # Resolve effective dim_red_n_components
    dim_red_n_components = (
        getattr(umap_model, "n_components", None)
        or getattr(umap_model, "n_components_", None)
        or dr_params.get("n_components")
    )
    dim_red_output_dim = observe_reducer_output_dim(topic_model)

    dim_red_n_neighbors = getattr(umap_model, "n_neighbors", None) or dr_params.get(
        "n_neighbors"
    )
    dim_red_metric = getattr(umap_model, "metric", None) or dr_params.get("metric")
    dim_red_min_dist = (
        getattr(umap_model, "min_dist", None)
        if getattr(umap_model, "min_dist", None) is not None
        else dr_params.get("min_dist")
    )
    dim_red_low_memory = (
        getattr(umap_model, "low_memory", None)
        if getattr(umap_model, "low_memory", None) is not None
        else dr_params.get("low_memory")
    )

    # 2. Clustering Provenance
    cl_config = model_config.get("clustering") or {}
    cl_params = cl_config.get("params") or {}
    hdbscan_model = getattr(topic_model, "hdbscan_model", None)

    normalize_text_view = getattr(
        hdbscan_model,
        "normalize_text_view",
        None,
    )
    if not isinstance(normalize_text_view, bool):
        normalize_text_view = bool(cl_params.get("normalize_text_view", False))

    inner_clusterer = getattr(hdbscan_model, "model", hdbscan_model)

    cluster_min_cluster_size = getattr(
        inner_clusterer, "min_cluster_size", None
    ) or cl_params.get("min_cluster_size")
    cluster_min_samples = (
        getattr(inner_clusterer, "min_samples", None)
        if getattr(inner_clusterer, "min_samples", None) is not None
        else cl_params.get("min_samples")
    )
    cluster_metric = getattr(inner_clusterer, "metric", None) or cl_params.get("metric")
    cluster_selection_method = getattr(
        inner_clusterer, "cluster_selection_method", None
    ) or cl_params.get("cluster_selection_method")
    cluster_prediction_data = (
        getattr(inner_clusterer, "prediction_data", None)
        if getattr(inner_clusterer, "prediction_data", None) is not None
        else cl_params.get("prediction_data")
    )

    return {
        "result_schema_version": PROVENANCE_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "run_status": run_status,
        "dim_red_n_components": dim_red_n_components,
        "dim_red_output_dim": dim_red_output_dim,
        "dim_red_n_neighbors": dim_red_n_neighbors,
        "dim_red_metric": dim_red_metric,
        "dim_red_min_dist": dim_red_min_dist,
        "dim_red_low_memory": dim_red_low_memory,
        "cluster_min_cluster_size": cluster_min_cluster_size,
        "cluster_min_samples": cluster_min_samples,
        "cluster_metric": cluster_metric,
        "cluster_selection_method": cluster_selection_method,
        "cluster_prediction_data": cluster_prediction_data,
        "normalize_text_view": normalize_text_view,
        "resolved_config_hash": config_hash,
        "run_manifest_path": run_manifest_path,
        "code_revision": git_rev,
        "code_dirty": git_dirty,
        "dependency_lock_hash": dep_lock_hash,
    }


def save_run_manifest(
    output_path: Union[str, Path],
    manifest_data: Optional[Dict[str, Any]] = None,
    *,
    provenance: Optional[Dict[str, Any]] = None,
    resolved_config: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Writes a comprehensive JSON run manifest alongside results.
    Accepts either a pre-constructed manifest_data dictionary or
    individual components (provenance, resolved_config, extra_metadata).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_data is None:
        manifest_data = {
            "provenance": provenance or {},
            "config": resolved_config or {},
            "extra_metadata": extra_metadata or {},
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4, default=str)
    return output_path
