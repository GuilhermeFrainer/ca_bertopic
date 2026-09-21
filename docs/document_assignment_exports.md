# Document assignment exports and focused FED reruns

`scripts/experiments/run_experiment.py` and `run_optimizer.py` export complete final
assignments by default for BERTopic-backed models, TriTopic, and FastTriTopic.
The `queue_exp.py` → `slurm_job.sh` → `run_optimizer.py` workflow uses this export
automatically, including split jobs selected with `--model`. No model settings,
embedding generation, probability computation, or text filtering policy changed.
The loader's existing selected-text filter is recorded explicitly; this work does
not change preprocessing to impose an additional stemmed-text filter.

## Durable artifacts

Each attempted training execution gets a UUID, shared by metrics, topic rows, and
the per-execution manifest. Discover all executions, including superseded and
failed ones, at `output/document_assignments/<dataset>/*/manifest.json`.

```text
output/document_assignments/<dataset>/<run_uid>/
  manifest.json
  assignments.parquet
  topics.json
  representative_documents.json
  metrics.json
```

Assignments include `run_uid`, contiguous `input_position`,
`source_document_key`, physical `source_row_ordinal`, final `topic_id` (including
`-1` noise), and original `index`/`id` columns when available. Keys use the full
source file SHA-256 plus physical row ordinal attached before filtering/sampling.
They distinguish identical texts and repeated parent IDs. The manifest records
whether the source's `index` and `id` columns are actually unique.
The local FED snapshot checked during implementation has 5,446 rows and unique
`index` values, but only 461 distinct `id` values; each rerun verifies its own source.

Input provenance includes the resolved path (including Yelp fallback), checksum,
source/selected row counts, selected columns, sampling size and effective seed,
ordered key checksum, raw covariate types, min/max scaling parameters, encoded
feature order, and the versioned encoding procedure. Sampling still uses the
runner's primary seed. All model seeds in one invocation share that prepared input.
The manifest also records resolved configuration, observed estimator settings,
code revision/dirty flag, lock hash, dependency versions, requested K, actual
non-noise count, noise count, and the `observed` metadata condition.

`manifest.json` is an atomic state journal, with separate fit, evaluation, and
export statuses. A hard interruption leaves a pending/running state, never a
complete success. Each artifact is replaced atomically and then registered with
its checksum and row count. Assignments and topic rows are retained before metric
evaluation starts. Evaluation failures retain these artifacts and report failure;
export failures propagate to runner error logging. An unregistered `.tmp` file
after a hard interruption is not a committed artifact. Representative text is
resolved only within its assigned topic: duplicate matches retain all candidate
keys and an ambiguous status. Exact indices are used when exposed by TriTopic.
No confidence is fabricated; `strength_kind` is null.

Canonical `topics.json` and `metrics.json` copies remain in the execution directory.
Legacy top-level CSV/topic JSON outputs also include the UUID and assignment links.
The existing merger scans only top-level files, so merging/archiving them leaves
all execution directories and their canonical references intact, including older
executions no longer selected in merged metrics. Historical outputs remain readable
but have no complete assignments; representative lists cannot backfill them.

In `run_experiment.py`, `--no-assignment-export` disables assignment/topic export and records
`export_status: disabled` in an execution manifest. Shared loader/training APIs
remain backward compatible. The optimizer CLI passes the source-aligned prepared
inputs to every grid/seed execution. Direct library callers of `Optimizer` must
provide `prepared_data` to enable exports; legacy array-only calls issue a warning
and keep their previous behavior because source identities cannot be reconstructed.
**STM assignment export remains deferred.** STM requires a separate theta
mapping and dominant-topic semantics; its runner is unchanged.

## Stage A: three full-corpus runs

From the repository root, run these separately:

```bash
uv run python scripts/experiments/run_experiment.py --exp fed/fed_qualitative_k50_mv_spectral --seed 36201624
uv run python scripts/experiments/run_experiment.py --exp fed/fed_qualitative_k50_baseline --seed 36201624
uv run python scripts/experiments/run_experiment.py --exp fed/fed_qualitative_k50_umap_spectral --seed 36201624
```

The [CAST2](../experiments/fed/fed_qualitative_k50_mv_spectral.yaml),
[baseline](../experiments/fed/fed_qualitative_k50_baseline.yaml), and
[text-only spectral](../experiments/fed/fed_qualitative_k50_umap_spectral.yaml)
configs preserve their production model parameters and standard `clean_text` /
`clean_text_embedding` inputs, changing only experiment names and requested K.
Representation stopword removal remains enabled by default. No sampling is set.

K is a scalar 50: the direct runner does not expand parameter grids. Human model
IDs are `mv_spectral`, `baseline`, and `umap_spectral`; UUID plus manifest seed
identify executions. Do not expect the optimizer's historical `_5_seed...` names.
Separate executions avoid the combined-run baseline topic-count override.

## Stage B: nine runs total, including Stage A

If Stage A is usable, run **six additional runs** (PowerShell):

```powershell
foreach ($seed in 62613654,57116123) {
    foreach ($model in 'mv_spectral','baseline','umap_spectral') {
        uv run python scripts/experiments/run_experiment.py --exp "fed/fed_qualitative_k50_$model" --seed $seed
    }
}
```

Or Bash:

```bash
for seed in 62613654 57116123; do
  for model in mv_spectral baseline umap_spectral; do
    uv run python scripts/experiments/run_experiment.py --exp "fed/fed_qualitative_k50_$model" --seed "$seed"
  done
done
```

To start Stage B from scratch, omit `--seed` from each of the three Stage A
commands: each config contains exactly the three specified seeds, giving nine
runs. Do not also run Stage A separately unless deliberate repeats are intended.

## Remote SLURM workflow

Your existing `queue_exp.py` commands now export assignments automatically. The
queue dispatcher needs no new flag: its optimizer entry point enables export.
Both ordinary grid jobs and split jobs retain their existing configuration/seed
ordering and human-readable model IDs. Each execution additionally receives a UUID.

For the dedicated K=50 configs, submit the existing worker with an explicit config
path. Stage A uses split index 1 (one K setting × the first of three seeds):

```bash
mkdir -p slurm_log
for model in mv_spectral baseline umap_spectral; do
  sbatch --partition=cidia --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=24:00:00 \
    --job-name="fed_qualitative_$model" \
    --output="slurm_log/%x_%j.out" --error="slurm_log/%x_%j.err" \
    scripts/experiments/slurm_job.sh fed "fed/fed_qualitative_k50_$model" --remove-rep-stopwords 1
done
```

This is Stage A (three jobs). For the additional Stage B runs, repeat with final
argument 2, then 3 (seeds 62613654 and 57116123), giving six additional jobs.
For all nine runs from scratch, omit the final index to run all three seeds per
model. Optimizer IDs for these configs are `<model>_1_seed<seed>`.

The worker copies the complete output subtree back from job scratch to
`~/slurm/output/document_assignments/fed/<run_uid>/`, including assignments from
fits whose subsequent evaluation failed. It now also performs the result copy
when the Python process returns a nonzero status, then returns that status.
Normal recursive result retrieval includes the new subtree. The source parquet
must still be retained separately outside disposable scratch space; use `--source`
when inspecting retrieved manifests. No jobs were submitted by implementation.

## Validate and inspect

Find execution manifests in PowerShell:

```powershell
Get-ChildItem output/document_assignments/fed/*/manifest.json
```

Then validate a run and inspect the historical candidate indices:

```bash
uv run python scripts/analysis/inspect_document_assignments.py output/document_assignments/fed/<run_uid>/manifest.json --source data/processed/fed_embeddings.parquet
```

`--source` allows relocation of an unchanged snapshot. The inspector validates
artifact checksums, ordered keys, and the source checksum/mapping, then reports
new topic assignments for indices 3367 and 5129 and raw numerical ranges/means
and categorical counts over each complete topic. Use `--indices` to inspect
other source indices. Missing requested indices produce no selected row; they
must not be interpreted as noise. Noise is explicitly topic `-1`.

Compare co-membership and substantive topic content across runs, never numeric
topic IDs. A split into noise is not a meaningful two-topic split. If the baseline
already separates this pair, report that and search other pairs from the full
assignment table. This export does not establish a causal metadata effect.

## Validation and return package

Implementation validation on 2026-09-21: **523 tests passed**, including optimizer
grid/split/CLI export and evaluation-failure tests, as well as real
synthetic fits for baseline, CAST2, and text-only spectral; Ruff checks passed for
all changed Python files. The full suite ran using the existing virtualenv with
`.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --tb=short`.
The SLURM worker also passed `bash -n` syntax validation. No cluster job was run.

Focused tests and synthetic smoke fits:

```bash
uv run pytest tests/test_document_assignments.py
uv run pytest
```

The three real smoke fits use 90 synthetic documents and K=3 in temporary
directories, with metrics disabled. They exercise the same model constructors and
export lifecycle but are not scientific reruns. For an optional sampled FED smoke,
append `--sample 200 --single-seed` to a Stage A command (it gets a dry-run name and
unique UUID). Do not substitute this for the full-corpus rerun.

Retain **one unchanged source parquet snapshot** for the entire campaign, outside
disposable scratch space. A checksum detects replacement but cannot reconstruct
an overwritten file. Return all execution directories (assignment Parquets,
manifests with input provenance, topic JSONs, representative identities, and
metrics), top-level metrics/topic outputs and run manifests, and the matching
source parquet or an accessible unchanged copy. Keep all three seeds for Stage B.

Saved candidates under `scratch/poster_candidates/` and both search scripts were
left intact. Historical topic IDs remain references only; recover new IDs from
the new assignments by source identity.
