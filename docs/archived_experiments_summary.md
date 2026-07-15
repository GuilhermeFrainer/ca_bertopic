# Reorganization Summary: Archived Experiments

The `experiments/` directory has been reorganized to isolate the **standard (production) experiments** from exploratory, optimization, and test files.

## Archived Directory Structure

All non-standard configurations have been moved to `experiments/archive/` and grouped by dataset:

```
experiments/
├── archive/                           # Centralized archive folder
│   ├── anes/                          # Archived ANES experiments
│   ├── anes_stemmed/                  # Archived Stemmed ANES experiments
│   ├── fed/                           # Archived FED experiments
│   ├── gadarian/                      # Archived Gadarian experiments
│   ├── trump/                         # Archived Trump experiments
│   └── yelp/                          # Archived Yelp experiments
├── datasets/                          # Dataset base specifications (Active)
│   ├── anes.yaml
│   ├── fed.yaml
│   └── ...
├── anes/                              # Active ANES standard experiments
├── fed/                               # Active FED standard experiments
├── gadarian/                          # Active Gadarian standard experiments
├── trump/                             # Active Trump standard experiments
└── yelp/                              # Active Yelp standard experiments
```

## Migration Stats

*   **Active Subdirectories Kept:** `anes`, `fed`, `gadarian`, `trump`, `yelp`
*   **Active Configurations Kept:** Only files matching `*_standard_*.yaml` remain in the active dataset subdirectories.
*   **Fully Archived Subdirectories:** `anes_stemmed` (contains no standard runs, so all files were moved to `archive/` and the empty active directory was deleted).
*   **Total Files Moved:** 98 files (including optimization runs `*_opt_*.yaml`, exploratory tests `*_test.yaml`, alternative configurations `*_alt_vanilla.yaml`, and baseline runs `*.yaml` that aren't the standard runs).

## Running Archived Experiments

Because the configuration resolver uses the path relative to the `experiments/` directory, you can run any archived configuration without moving it back:

```bash
# Example: Run an archived optimization config
python scripts/experiments/run_experiment.py --exp archive/trump/trump_opt_mv_spectral
```

## Verification

The entire test suite (54/54 tests passing) was run both before and after the reorganization to confirm that the changes did not break the model loading, config inheritance, or test runners.
