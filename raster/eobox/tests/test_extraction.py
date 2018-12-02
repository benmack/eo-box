import pytest
import fnmatch
import os
from pathlib import Path

from eobox.sampledata import get_dataset
from eobox.raster import extract
from eobox.raster import load_extracted

s2l1c = get_dataset("s2l1c")
src_vector = s2l1c["vector_file"]
burn_attribute = "pid"
src_raster = fnmatch.filter(s2l1c["raster_files"], "*T33UUU_20170216T102101_B11.jp2")
src_raster += fnmatch.filter(s2l1c["raster_files"], "*T33UUU_20170216T102101_B12.jp2")
src_raster = [Path(src) for src in src_raster]
dst_names = ["B11", "B12"]

def test_extract_and_save_to_npy_successfully(tmpdir):
    dst_dir = tmpdir.mkdir("temp_dst_dir")
    exit_code = extract(src_vector=src_vector,
                        burn_attribute=burn_attribute,
                        src_raster=src_raster,
                        dst_names=dst_names,
                        dst_dir=dst_dir,
                        src_raster_template=None)
    assert(exit_code == 0)
    expected_basenames = ['aux_vector_pid.npy',
                          'aux_coord_x.npy',
                          'aux_coord_y.npy',
                          'B11.npy',
                          'B12.npy']
    assert(all([(dst_dir / bname).exists() for bname in expected_basenames]))


def test_load_extracted(tmpdir):
    extraction_dir = tmpdir.mkdir("temp_dst_dir")
    exit_code = extract(src_vector=src_vector,
                        burn_attribute=burn_attribute,
                        src_raster=src_raster,
                        dst_names=dst_names,
                        dst_dir=extraction_dir,
                        src_raster_template=None)
    # load all
    df_extracted = load_extracted(extraction_dir)
    assert(df_extracted.shape[1] == 5)
    assert(~df_extracted["aux_vector_pid"].isin([18]).any())
    index_29 = df_extracted["aux_vector_pid"] == 29
    df_extracted_29 = df_extracted[index_29]
    assert(df_extracted_29["B11"].isin([1664]).any())
    assert(df_extracted_29["B12"].isin([1088]).any())

    # load subset
    df_extracted_29 = load_extracted(extraction_dir, index=index_29)
    assert(df_extracted_29.shape[0] == index_29.sum())
