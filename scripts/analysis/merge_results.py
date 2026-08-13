import argparse
import os
import pathlib
import re
from typing import Dict, List, Tuple

import polars as pl


def normalize_dataset_name(name: str) -> str:
    """
    Strips sampling suffixes like _s10000 and _stemmed from the dataset name.
    """
    if not name:
        return name
    name = re.sub(r"_s\d+$", "", name)
    name = re.sub(r"_stemmed$", "", name)
    return name


def parse_filename(filename: str) -> Tuple[str, str, str]:
    """
    Parses a filename into experiment_id, timestamp, and random_state.
    Format: experiment_id-YYYYMMDD-HHMMSS-random_state.ext
    """
    # Remove extension
    stem = pathlib.Path(filename).stem
    parts = stem.split("-")

    if len(parts) < 3:
        return None, None, None

    random_state = parts[-1]
    # Timestamp has a dash in it: YYYYMMDD-HHMMSS
    timestamp = f"{parts[-3]}-{parts[-2]}"
    experiment_id = "-".join(parts[:-3])

    return experiment_id, timestamp, random_state


def get_dataset_info(file_path: pathlib.Path) -> Tuple[str | None, str | None]:
    """
    Reads the file and/or inspects filename to extract (dataset_name, dataset_type).
    dataset_type can be 'stemmed', 'no_stopword_removal', or 'standard'.
    """
    dataset_name = None
    df = None

    try:
        if file_path.suffix == ".csv":
            df = pl.read_csv(file_path, n_rows=5, infer_schema_length=None)
        elif file_path.suffix == ".json":
            df = pl.read_json(file_path, infer_schema_length=None)
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")

    if df is not None and "dataset_name" in df.columns and len(df) > 0:
        dataset_name = str(df["dataset_name"][0])

    if not dataset_name:
        exp_id, _, _ = parse_filename(file_path.name)
        if exp_id:
            dataset_name = exp_id.split("_")[0]

    if not dataset_name:
        return None, None

    filename_lower = file_path.name.lower()

    # 1. Stemmed check
    is_stemmed = "stemmed" in filename_lower
    if df is not None:
        if "dataset_name" in df.columns and len(df) > 0:
            if "stemmed" in str(df["dataset_name"][0]).lower():
                is_stemmed = True
        if "text_column" in df.columns and len(df) > 0:
            if "stemmed" in str(df["text_column"][0]).lower():
                is_stemmed = True

    if is_stemmed:
        dataset_type = "stemmed"
    else:
        # 2. No stopword removal check
        is_no_stopword = ("no_stopword" in filename_lower) or (
            "keep_rep_stopwords" in filename_lower
        )
        if df is not None and "stopword_removal" in df.columns and len(df) > 0:
            status = str(df["stopword_removal"][0]).lower()
            if status in ("keep_rep_stopwords", "none", "no_stopword_removal", "false"):
                is_no_stopword = True

        if is_no_stopword:
            dataset_type = "no_stopword_removal"
        else:
            dataset_type = "standard"

    dataset_name = normalize_dataset_name(dataset_name)
    return dataset_name, dataset_type


def get_dataset_name(file_path: pathlib.Path) -> str | None:
    """
    Reads the file to extract the dataset_name.
    """
    d_name, _ = get_dataset_info(file_path)
    return d_name


def group_files(
    directory: pathlib.Path, extension: str, ignore_suffix: str = None
) -> Dict[Tuple[str, str], List[pathlib.Path]]:
    """
    Groups files by (dataset_name, dataset_type) and keeps only the latest run for each experiment.
    Also includes files that don't match the experiment pattern but have a dataset_name.
    """
    latest_runs: Dict[Tuple[str, str, str, str], Tuple[str, pathlib.Path]] = {}
    base_files: Dict[Tuple[str, str], List[pathlib.Path]] = {}

    for file_path in directory.glob(f"*{extension}"):
        # Avoid including the file we might be writing to
        if ignore_suffix and file_path.stem.endswith(ignore_suffix):
            continue

        exp_id, timestamp, random_state = parse_filename(file_path.name)

        dataset_name, dataset_type = get_dataset_info(file_path)
        if not dataset_name or not dataset_type:
            continue

        if exp_id:
            key = (dataset_name, dataset_type, exp_id, random_state)
            if key not in latest_runs or timestamp > latest_runs[key][0]:
                latest_runs[key] = (timestamp, file_path)
        else:
            group_key = (dataset_name, dataset_type)
            if group_key not in base_files:
                base_files[group_key] = []
            base_files[group_key].append(file_path)

    # Regroup by (dataset_name, dataset_type)
    grouped: Dict[Tuple[str, str], List[pathlib.Path]] = {}
    for (dataset_name, dataset_type, _, _), (_, file_path) in latest_runs.items():
        group_key = (dataset_name, dataset_type)
        if group_key not in grouped:
            grouped[group_key] = []
        grouped[group_key].append(file_path)

    for group_key, files in base_files.items():
        if group_key not in grouped:
            grouped[group_key] = []
        # Add base files, but avoid duplicates if they were somehow already in latest_runs
        for bf in files:
            if bf not in grouped[group_key]:
                grouped[group_key].append(bf)

    return grouped


def merge_files(
    files: List[pathlib.Path], output_path: pathlib.Path, dry_run: bool, force: bool
):
    """
    Merges multiple files into one.
    """
    if not files:
        return

    print(f"Merging {len(files)} files into {output_path}...")
    if dry_run:
        for f in files:
            print(f"  - {f.name}")
        return

    if output_path.exists() and not force:
        confirm = input(f"File {output_path} already exists. Overwrite? [y/N]: ")
        if confirm.lower() != "y":
            print(f"Skipping {output_path}")
            return

    try:
        dfs = []
        for f in files:
            if f.suffix == ".csv":
                dfs.append(pl.read_csv(f, infer_schema_length=None))
            elif f.suffix == ".json":
                dfs.append(pl.read_json(f, infer_schema_length=None))

        if not dfs:
            return

        merged_df = pl.concat(dfs, how="diagonal")

        # Deduplicate: only remove rows that are 100% identical across all columns
        if output_path.suffix == ".csv":
            merged_df = merged_df.unique(maintain_order=True)

        if output_path.suffix == ".csv":
            merged_df.write_csv(output_path)
        elif output_path.suffix == ".json":
            # For JSON, we use the same pretty-print logic as in the main scripts
            import json

            json_str = merged_df.write_json()
            parsed_json = json.loads(json_str)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, indent=4)

        print(f"Successfully merged into {output_path}")
    except Exception as e:
        print(f"Error merging files into {output_path}: {e}")


import argparse
import datetime
import os
import pathlib
import re
import zipfile
from typing import Dict, List, Tuple

import polars as pl


def move_files(files: List[pathlib.Path], move_to_dir: pathlib.Path, dry_run: bool):
    """
    Moves files to a specified directory.
    """
    if not move_to_dir.exists() and not dry_run:
        move_to_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        target = move_to_dir / f.name
        print(f"Moving {f.name} to {target}...")
        if not dry_run:
            try:
                os.replace(f, target)
            except Exception as e:
                print(f"Error moving {f.name}: {e}")


def archive_files(
    files: List[pathlib.Path],
    archive_dir: pathlib.Path,
    dataset: str,
    dataset_type: str,
    output_path: pathlib.Path,
    dry_run: bool,
    keep_originals: bool = False,
):
    """
    Packages original raw files into a timestamped ZIP archive containing a README.txt,
    and removes the original files from the source directory unless keep_originals is True.
    """
    if not files:
        return

    now = datetime.datetime.now().astimezone()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    zip_name = f"{dataset}_{dataset_type}_merged_{timestamp_str}.zip"
    zip_path = archive_dir / zip_name

    file_list_str = "\n".join(f"  {idx + 1}. {f.name}" for idx, f in enumerate(files))
    readme_content = (
        "================================================================================\n"
        "CA-BERTopic - Merged Experiment Results Archive\n"
        "================================================================================\n\n"
        "Merge Metadata:\n"
        "---------------\n"
        f"Merge Timestamp:        {now.strftime('%Y-%m-%d %H:%M:%S %z')}\n"
        f"Dataset Name:           {dataset}\n"
        f"Dataset Type:           {dataset_type}\n"
        f"Target Merged File:     {output_path.name}\n"
        f"Total Raw Files Merged: {len(files)}\n\n"
        "Archived Files:\n"
        "---------------\n"
        f"This archive contains the {len(files)} original raw experiment result files that were\n"
        f"consolidated into '{output_path.name}':\n\n"
        f"{file_list_str}\n\n"
        "Notes:\n"
        "------\n"
        "- Only the latest run for each (experiment_id, random_state) pair was kept during merge.\n"
        "- Identical duplicate rows across all columns were deduplicated in the final merged file.\n"
    )

    if dry_run:
        cleanup_msg = (
            "originals will be kept" if keep_originals else "originals will be removed"
        )
        print(
            f"  [Dry Run] Archive ZIP: {zip_path} (contains README.txt + {len(files)} files; {cleanup_msg})"
        )
        return

    print(f"Archiving {len(files)} files into {zip_path}...")
    if not archive_dir.exists():
        archive_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("README.txt", readme_content)
            for f in files:
                zf.write(f, arcname=f.name)

        print(f"Successfully created archive {zip_path}")

        if not keep_originals:
            for f in files:
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Error removing {f.name}: {e}")
    except Exception as e:
        print(f"Error archiving files into {zip_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge experiment results and outputs by dataset and dataset type."
    )
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Directory for CSV results."
    )
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Directory for JSON outputs."
    )
    parser.add_argument(
        "--suffix", type=str, default="_merged", help="Suffix for merged files."
    )
    parser.add_argument(
        "--archive-dir",
        type=str,
        help="Optional custom directory for ZIP archives.",
    )
    parser.add_argument(
        "--move-to",
        type=str,
        help="Optional directory to move original files after merging (when --no-archive is set).",
    )
    parser.add_argument(
        "--keep-originals",
        action="store_true",
        help="Keep original raw files in place after creating the ZIP archive.",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Disable ZIP archive creation.",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be done without doing it.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files without prompting.",
    )

    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    output_dir = pathlib.Path(args.output_dir)

    # Process CSV Results
    print("--- Processing CSV Results ---")
    if results_dir.exists():
        grouped_csv = group_files(results_dir, ".csv", ignore_suffix=args.suffix)
        all_csv_to_move = []
        for (dataset, dataset_type), files in grouped_csv.items():
            output_path = results_dir / f"{dataset}_{dataset_type}{args.suffix}.csv"
            merge_files(files, output_path, args.dry_run, args.force)
            if not args.no_archive:
                csv_archive_dir = (
                    pathlib.Path(args.archive_dir)
                    if args.archive_dir
                    else results_dir / "archive"
                )
                archive_files(
                    files,
                    csv_archive_dir,
                    dataset,
                    dataset_type,
                    output_path,
                    args.dry_run,
                    keep_originals=args.keep_originals,
                )
            all_csv_to_move.extend(files)

        if args.move_to and all_csv_to_move and args.no_archive:
            move_files(all_csv_to_move, pathlib.Path(args.move_to), args.dry_run)
    else:
        print(f"Results directory {results_dir} not found.")

    # Process JSON Outputs
    print("\n--- Processing JSON Outputs ---")
    if output_dir.exists():
        grouped_json = group_files(output_dir, ".json", ignore_suffix=args.suffix)
        all_json_to_move = []
        for (dataset, dataset_type), files in grouped_json.items():
            output_path = output_dir / f"{dataset}_{dataset_type}{args.suffix}.json"
            merge_files(files, output_path, args.dry_run, args.force)
            if not args.no_archive:
                json_archive_dir = (
                    pathlib.Path(args.archive_dir)
                    if args.archive_dir
                    else output_dir / "archive"
                )
                archive_files(
                    files,
                    json_archive_dir,
                    dataset,
                    dataset_type,
                    output_path,
                    args.dry_run,
                    keep_originals=args.keep_originals,
                )
            all_json_to_move.extend(files)

        if args.move_to and all_json_to_move and args.no_archive:
            move_files(all_json_to_move, pathlib.Path(args.move_to), args.dry_run)
    else:
        print(f"Output directory {output_dir} not found.")


if __name__ == "__main__":
    main()
