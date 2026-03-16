import pytest
from pathlib import Path
import tempfile

from src.embeddings import get_next_batch_index, sort_batch_files


# --- Test Suite ---

def test_get_next_batch_index_empty_directory():
    """
    Tests that get_next_batch_index returns 0 for an empty or non-existent directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        batch_dir = Path(tmpdir)
        assert get_next_batch_index(batch_dir) == 0
        
        # Test non-existent directory
        non_existent_dir = batch_dir / "non_existent"
        assert get_next_batch_index(non_existent_dir) == 0

def test_get_next_batch_index_with_files():
    """
    Tests that get_next_batch_index correctly finds the next index from existing batch files.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        batch_dir = Path(tmpdir)
        batch_dir.mkdir(exist_ok=True)
        (batch_dir / "batch_0.parquet").touch()
        (batch_dir / "batch_1.parquet").touch()
        (batch_dir / "batch_2.parquet").touch()
        
        assert get_next_batch_index(batch_dir) == 3

def test_get_next_batch_index_with_unordered_files():
    """
    Tests that get_next_batch_index finds the highest index even if files are not ordered.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        batch_dir = Path(tmpdir)
        batch_dir.mkdir(exist_ok=True)
        (batch_dir / "batch_10.parquet").touch()
        (batch_dir / "batch_2.parquet").touch()
        (batch_dir / "batch_0.parquet").touch()
        
        assert get_next_batch_index(batch_dir) == 11
        
def test_get_next_batch_index_with_other_files_present():
    """
    Tests that other non-batch files in the directory are ignored.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        batch_dir = Path(tmpdir)
        batch_dir.mkdir(exist_ok=True)
        (batch_dir / "batch_0.parquet").touch()
        (batch_dir / "batch_1.parquet").touch()
        (batch_dir / "info.txt").touch()
        (batch_dir / "wrong_batch_format_1.parquet").touch()
        
        assert get_next_batch_index(batch_dir) == 2


def test_sort_batch_files():
    """
    Tests that batch files are sorted numerically, not lexicographically.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        files = [
            p / "batch_10.parquet",
            p / "batch_1.parquet",
            p / "batch_2.parquet",
            p / "batch_0.parquet"
        ]
        
        # Create dummy files
        for f in files:
            f.touch()
            
        # Get actual list from disk to mimic the script's behavior
        files_on_disk = list(p.glob("*.parquet"))
        
        sorted_files = sort_batch_files(files_on_disk)
        
        expected_order = [
            p / "batch_0.parquet",
            p / "batch_1.parquet",
            p / "batch_2.parquet",
            p / "batch_10.parquet",
        ]
        
        assert sorted_files == expected_order

def test_sort_batch_files_empty_list():
    """
    Tests that sorting an empty list results in an empty list.
    """
    assert sort_batch_files([]) == []
