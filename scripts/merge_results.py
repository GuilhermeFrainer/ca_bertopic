import os
import argparse
import pathlib
import datetime
import polars as pl
from typing import Dict, List, Tuple


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


def get_dataset_name(file_path: pathlib.Path) -> str:
    """
    Reads the file to extract the dataset_name.
    """
    try:
        if file_path.suffix == ".csv":
            # Just read the first few rows to get the dataset_name column
            df = pl.read_csv(file_path, n_rows=5)
            if "dataset_name" in df.columns:
                return df["dataset_name"][0]
        elif file_path.suffix == ".json":
            df = pl.read_json(file_path)
            if "dataset_name" in df.columns:
                return df["dataset_name"][0]
    except Exception as e:
        print(f"Warning: Could not read dataset_name from {file_path}: {e}")
    return None


def group_files(directory: pathlib.Path, extension: str) -> Dict[str, List[pathlib.Path]]:
    """
    Groups files by dataset_name and keeps only the latest run for each experiment.
    """
    latest_runs: Dict[Tuple[str, str, str], Tuple[str, pathlib.Path]] = {}
    
    for file_path in directory.glob(f"*{extension}"):
        exp_id, timestamp, random_state = parse_filename(file_path.name)
        if not exp_id:
            continue
            
        dataset_name = get_dataset_name(file_path)
        if not dataset_name:
            continue
            
        key = (dataset_name, exp_id, random_state)
        
        if key not in latest_runs or timestamp > latest_runs[key][0]:
            latest_runs[key] = (timestamp, file_path)
            
    # Regroup by dataset_name
    grouped: Dict[str, List[pathlib.Path]] = {}
    for (dataset_name, _, _), (_, file_path) in latest_runs.items():
        if dataset_name not in grouped:
            grouped[dataset_name] = []
        grouped[dataset_name].append(file_path)
        
    return grouped


def merge_files(files: List[pathlib.Path], output_path: pathlib.Path, dry_run: bool, force: bool):
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
        if confirm.lower() != 'y':
            print(f"Skipping {output_path}")
            return

    try:
        dfs = []
        for f in files:
            if f.suffix == ".csv":
                dfs.append(pl.read_csv(f))
            elif f.suffix == ".json":
                dfs.append(pl.read_json(f))
        
        if not dfs:
            return

        merged_df = pl.concat(dfs, how="diagonal")
        
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


def main():
    parser = argparse.ArgumentParser(description="Merge experiment results and outputs by dataset.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for CSV results.")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for JSON outputs.")
    parser.add_argument("--suffix", type=str, default="_merged", help="Suffix for merged files.")
    parser.add_argument("--move-to", type=str, help="Optional directory to move original files after merging.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without doing it.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files without prompting.")
    
    args = parser.parse_args()
    
    results_dir = pathlib.Path(args.results_dir)
    output_dir = pathlib.Path(args.output_dir)
    
    # Process CSV Results
    print("--- Processing CSV Results ---")
    if results_dir.exists():
        grouped_csv = group_files(results_dir, ".csv")
        all_csv_to_move = []
        for dataset, files in grouped_csv.items():
            output_path = results_dir / f"{dataset}{args.suffix}.csv"
            merge_files(files, output_path, args.dry_run, args.force)
            all_csv_to_move.extend(files)
            
        if args.move_to and all_csv_to_move:
            move_files(all_csv_to_move, pathlib.Path(args.move_to), args.dry_run)
    else:
        print(f"Results directory {results_dir} not found.")

    # Process JSON Outputs
    print("\n--- Processing JSON Outputs ---")
    if output_dir.exists():
        grouped_json = group_files(output_dir, ".json")
        all_json_to_move = []
        for dataset, files in grouped_json.items():
            output_path = output_dir / f"{dataset}{args.suffix}.json"
            merge_files(files, output_path, args.dry_run, args.force)
            all_json_to_move.extend(files)
            
        if args.move_to and all_json_to_move:
            move_files(all_json_to_move, pathlib.Path(args.move_to), args.dry_run)
    else:
        print(f"Output directory {output_dir} not found.")


if __name__ == "__main__":
    main()
