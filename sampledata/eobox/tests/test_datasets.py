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
