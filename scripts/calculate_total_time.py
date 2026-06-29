import polars as pl
import glob
import os

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"

def main():
    result_files = glob.glob("results/*.csv")
    
    total_duration = 0
    trump_duration = 0
    
    for file in result_files:
        try:
            df = pl.read_csv(file)
            if "duration_seconds" in df.columns:
                # Sum duration for the whole file
                file_total = df["duration_seconds"].sum()
                total_duration += file_total
                
                # Sum duration for trump dataset
                if "dataset_name" in df.columns:
                    trump_total = df.filter(pl.col("dataset_name") == "trump")["duration_seconds"].sum()
                    trump_duration += trump_total
                else:
                    # If dataset_name is not in columns, check the filename
                    if "trump" in os.path.basename(file).lower():
                        trump_duration += file_total
        except Exception as e:
            print(f"Error reading {file}: {e}")

    print(f"Total time spent on experiments: {format_time(total_duration)} ({total_duration:.2f} seconds)")
    print(f"Time spent on Trump dataset: {format_time(trump_duration)} ({trump_duration:.2f} seconds)")

if __name__ == "__main__":
    main()
