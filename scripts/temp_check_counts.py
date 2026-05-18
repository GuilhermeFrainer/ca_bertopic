import polars as pl
from pathlib import Path

datasets = ["anes", "gadarian", "trump", "yelp"]
for d in datasets:
    try:
        raw_path = f"data/interim/{d}.parquet" if d != "trump" else "data/raw/trump_tweets.csv"
        if not Path(raw_path).exists():
             print(f"{d}: raw file not found at {raw_path}")
             continue
        
        if raw_path.endswith(".parquet"):
            raw_h = pl.scan_parquet(raw_path).select(pl.len()).collect().item()
        else:
            raw_h = pl.scan_csv(raw_path).select(pl.len()).collect().item()
            
        proc_path = f"data/interim/{d}_processed.parquet"
        if Path(proc_path).exists():
            proc_h = pl.scan_parquet(proc_path).select(pl.len()).collect().item()
        else:
            proc_h = "N/A"
            
        print(f"{d}: raw={raw_h}, proc={proc_h}")
    except Exception as e:
        print(f"{d}: error {e}")
