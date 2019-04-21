import pytest
import fnmatch
import os
from pathlib import Path

from eobox.sampledata import get_dataset
from eobox.raster import extract
from eobox.raster import load_extracted


@pytest.fixture
def extraction_input_1(tmpdir):

    s2l1c = get_dataset("s2l1c")
    src_raster = fnmatch.filter(s2l1c["raster_files"], "*T33UUU_20170216T102101_B11.jp2")
    src_raster += fnmatch.filter(s2l1c["raster_files"], "*T33UUU_20170216T102101_B12.jp2")
    input_kwargs = {
        "s2l1c": s2l1c,
        "src_vector": s2l1c["vector_file"],
        "burn_attribute": "pid",
        "src_raster": src_raster,
        "dst_names": ["B11", "B12"],
        "extracted_dir": tmpdir.mkdir("temp_dst_dir-0"),
    }
    return input_kwargs


@pytest.fixture
def extracted_data_1(extraction_input_1, tmpdir):

    extraction_input_1["extracted_dir"] = tmpdir.mkdir("temp_dst_dir-1")
    exit_code = extract(
        src_vector=extraction_input_1["src_vector"],
        burn_attribute=extraction_input_1["burn_attribute"],
        src_raster=extraction_input_1["src_raster"],
        dst_names=extraction_input_1["dst_names"],
        dst_dir=extraction_input_1["extracted_dir"],
        src_raster_template=None,
    )
    assert exit_code == 0
    expected_basenames = [
        "aux_vector_pid.npy",
        "aux_coord_x.npy",
        "aux_coord_y.npy",
        "B11.npy",
        "B12.npy",
    ]
    extraction_input_1["extracted_basenames"] = expected_basenames
    extraction_input_1["extracted_files"] = [
        (extraction_input_1["extracted_dir"] / bname) for bname in expected_basenames
    ]
    assert all([fname.exists() for fname in extraction_input_1["extracted_files"]])
    return extraction_input_1


def test_extract_and_save_to_npy_successfully(extraction_input_1):

    exit_code = extract(
        src_vector=extraction_input_1["src_vector"],
        burn_attribute=extraction_input_1["burn_attribute"],
        src_raster=extraction_input_1["src_raster"],
        dst_names=extraction_input_1["dst_names"],
        dst_dir=extraction_input_1["extracted_dir"],
        src_raster_template=None,
    )
    assert exit_code == 0
    expected_basenames = [
        "aux_vector_pid.npy",
        "aux_coord_x.npy",
        "aux_coord_y.npy",
        "B11.npy",
        "B12.npy",
    ]
    extraction_input_1["extracted_files"] = [
        (extraction_input_1["extracted_dir"] / bname) for bname in expected_basenames
    ]
    assert all(
        [(extraction_input_1["extracted_dir"] / bname).exists() for bname in expected_basenames]
    )


def test_extract_and_save_to_npy_successfully_parallel(extraction_input_1):

    exit_code = extract(
        src_vector=extraction_input_1["src_vector"],
        burn_attribute=extraction_input_1["burn_attribute"],
        src_raster=extraction_input_1["src_raster"],
        dst_names=extraction_input_1["dst_names"],
        dst_dir=extraction_input_1["extracted_dir"],
        src_raster_template=None,
        n_jobs=2,
    )
    assert exit_code == 0
    expected_basenames = [
        "aux_vector_pid.npy",
        "aux_coord_x.npy",
        "aux_coord_y.npy",
        "B11.npy",
        "B12.npy",
    ]
    assert all(
        [(extraction_input_1["extracted_dir"] / bname).exists() for bname in expected_basenames]
    )


def test_extract_and_save_to_npy_rerun_on_existing_data_successfully(extracted_data_1):

    assert Path(extracted_data_1["extracted_dir"]).exists()
    assert all([fname.exists() for fname in extracted_data_1["extracted_files"]])
    exit_code = extract(
        src_vector=extracted_data_1["src_vector"],
        burn_attribute=extracted_data_1["burn_attribute"],
        src_raster=extracted_data_1["src_raster"],
        dst_names=extracted_data_1["dst_names"],
        dst_dir=extracted_data_1["extracted_dir"],
        src_raster_template=None,
    )
    assert exit_code == 0
    assert all([filename.exists() for filename in extracted_data_1["extracted_files"]])


def test_load_extracted(extracted_data_1):

    df_extracted = load_extracted(extracted_data_1["extracted_dir"])
    assert df_extracted.shape[1] == 5
    assert ~df_extracted["aux_vector_pid"].isin([18]).any()


def test_load_extracted_subset(extracted_data_1):

    df_extracted = load_extracted(extracted_data_1["extracted_dir"])
    index_29 = df_extracted["aux_vector_pid"] == 29
    df_extracted_29_manual = df_extracted[index_29]
    assert df_extracted_29_manual["aux_vector_pid"].nunique() == 1
    assert df_extracted_29_manual["B11"].isin([1664]).any()
    assert df_extracted_29_manual["B12"].isin([1088]).any()

    # load subset directly
    df_extracted_29 = load_extracted(extracted_data_1["extracted_dir"], index=index_29)
    assert df_extracted_29.shape[0] == index_29.sum()
    assert df_extracted_29["aux_vector_pid"].nunique() == 1
    assert df_extracted_29["B11"].isin([1664]).any()
    assert df_extracted_29["B12"].isin([1088]).any()
