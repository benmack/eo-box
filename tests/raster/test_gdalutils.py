import pytest

from eobox.raster import gdalutils


def test_proximity_path():
    assert gdalutils.PROXIMITY_PATH is not None

def test_polygonize_path():
    assert gdalutils.POLYGONIZE_PATH is not None
