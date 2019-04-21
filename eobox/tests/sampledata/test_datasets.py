import os
import pytest

from eobox.sampledata import get_dataset


def test_dataset_s2l1c():
    ds = get_dataset("s2l1c")
    assert isinstance(ds, dict)
    assert len(ds["raster_files"]) == 13
    assert len(ds["raster_bands"]) == 13
    assert "B02" in ds["raster_bands"]
    assert len(ds["raster_times"]) == 13
    assert "20170216T102101" in ds["raster_times"]
    assert isinstance(ds["vector_file"], str)
    assert all([os.path.exists(path) for path in ds["raster_files"]])
    assert os.path.exists(ds["vector_file"])

def test_dataset_lsts():
    ds = get_dataset("lsts")
    assert isinstance(ds, dict)
    assert len(ds["raster_files"]) == 420
    assert len(ds["raster_bands"]) == 420
    assert len(ds["raster_times"]) == 420
    assert "b3" in ds["raster_bands"]
    assert "b4" in ds["raster_bands"]
    assert "b5" in ds["raster_bands"]
    assert "fmask" in ds["raster_bands"]
    assert "2008118" in ds["raster_times"]
    assert all([os.path.exists(path) for path in ds["raster_files"]])
