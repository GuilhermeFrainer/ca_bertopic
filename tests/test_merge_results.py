import pathlib
import sys

import polars as pl

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.merge_results import (  # noqa: E402
    archive_files,
    get_dataset_info,
    group_files,
    merge_files,
    normalize_dataset_name,
)
import zipfile


def test_normalize_dataset_name():
    assert normalize_dataset_name("fed_s10000") == "fed"
    assert normalize_dataset_name("anes_stemmed") == "anes"
    assert normalize_dataset_name("trump") == "trump"
    assert normalize_dataset_name("") == ""


def test_get_dataset_info_stemmed(tmp_path):
    f_stemmed = tmp_path / "fed_stemmed_standard_baseline-20260805-120000-1234.csv"
    df = pl.DataFrame({"dataset_name": ["fed_stemmed"], "c_v": [0.5]})
    df.write_csv(f_stemmed)

    d_name, d_type = get_dataset_info(f_stemmed)
    assert d_name == "fed"
    assert d_type == "stemmed"


def test_get_dataset_info_no_stopword(tmp_path):
    f_nostop = tmp_path / "fed_no_stopword_baseline-20260805-120000-1234.csv"
    df = pl.DataFrame(
        {
            "dataset_name": ["fed"],
            "stopword_removal": ["keep_rep_stopwords"],
            "c_v": [0.5],
        }
    )
    df.write_csv(f_nostop)

    d_name, d_type = get_dataset_info(f_nostop)
    assert d_name == "fed"
    assert d_type == "no_stopword_removal"


def test_get_dataset_info_standard(tmp_path):
    f_std = tmp_path / "fed_standard_baseline-20260805-120000-1234.csv"
    df = pl.DataFrame(
        {
            "dataset_name": ["fed"],
            "stopword_removal": ["remove_rep_stopwords"],
            "c_v": [0.5],
        }
    )
    df.write_csv(f_std)

    d_name, d_type = get_dataset_info(f_std)
    assert d_name == "fed"
    assert d_type == "standard"


def test_group_files_and_merge(tmp_path):
    # Setup test files
    f_std_1 = tmp_path / "fed_standard_baseline-20260801-100000-1234.csv"
    f_std_2 = tmp_path / "fed_standard_baseline-20260802-100000-1234.csv"  # Latest std
    f_stemmed = tmp_path / "fed_stemmed_baseline-20260801-100000-1234.csv"

    pl.DataFrame({"dataset_name": ["fed"], "val": [1]}).write_csv(f_std_1)
    pl.DataFrame({"dataset_name": ["fed"], "val": [2]}).write_csv(f_std_2)
    pl.DataFrame({"dataset_name": ["fed_stemmed"], "val": [3]}).write_csv(f_stemmed)

    grouped = group_files(tmp_path, ".csv", ignore_suffix="_merged")

    assert ("fed", "standard") in grouped
    assert ("fed", "stemmed") in grouped
    assert len(grouped[("fed", "standard")]) == 1
    assert grouped[("fed", "standard")][0] == f_std_2
    assert len(grouped[("fed", "stemmed")]) == 1
    assert grouped[("fed", "stemmed")][0] == f_stemmed

    # Test merge_files
    out_std = tmp_path / "fed_standard_merged.csv"
    merge_files(grouped[("fed", "standard")], out_std, dry_run=False, force=True)
    assert out_std.exists()

    res_df = pl.read_csv(out_std)
    assert res_df["val"][0] == 2


def test_archive_files(tmp_path):
    f_std_1 = tmp_path / "fed_standard_baseline-20260801-100000-1234.csv"
    f_std_2 = tmp_path / "fed_standard_baseline-20260802-100000-1234.csv"

    pl.DataFrame({"dataset_name": ["fed"], "val": [1]}).write_csv(f_std_1)
    pl.DataFrame({"dataset_name": ["fed"], "val": [2]}).write_csv(f_std_2)

    archive_dir = tmp_path / "archive"
    out_std = tmp_path / "fed_standard_merged.csv"

    archive_files(
        [f_std_1, f_std_2],
        archive_dir,
        "fed",
        "standard",
        out_std,
        dry_run=False,
        keep_originals=False,
    )

    zip_files = list(archive_dir.glob("*.zip"))
    assert len(zip_files) == 1
    zip_path = zip_files[0]
    assert "fed_standard_merged_" in zip_path.name

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        assert "README.txt" in names
        assert f_std_1.name in names
        assert f_std_2.name in names

        readme_text = zf.read("README.txt").decode("utf-8")
        assert "fed" in readme_text
        assert "standard" in readme_text
        assert out_std.name in readme_text
        assert f_std_1.name in readme_text

    # Originals should be cleaned up
    assert not f_std_1.exists()
    assert not f_std_2.exists()
