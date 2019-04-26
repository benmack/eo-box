import pytest
import fnmatch
import os
from pathlib import Path

from eobox.sampledata import get_dataset
from eobox.vector import convert_polygons_to_lines
from eobox.vector import calc_distance_to_border



@pytest.fixture
def testdata_1(tmpdir):

    ds = get_dataset('s2l1c')

    tmp_procdir = tmpdir.mkdir("temp_proc_dir")

    input_kwargs = {
        'src_file': ds['vector_file'],
        'template_file_raster': ds['raster_files'][0],
        'interim_file_lines': tmp_procdir / "_interim_sample_vector_dataset_lines.shp",
        'interim_file_lines_raster': tmp_procdir / "/_interim_sample_vector_dataset_lines_raster.tif",
        'dst_file_proximity': tmp_procdir / "distance_to_polygon_border__vector_dataset.tif"
    }
    return input_kwargs


def test_convert_polygons_to_lines_default(testdata_1):

    exit_code = convert_polygons_to_lines(src_polygons=testdata_1["src_file"],
                                          dst_lines=testdata_1["interim_file_lines"],
                                          crs=None, add_allone_col=True)
    assert (exit_code == 0)

def test_calc_distance_to_border_default(testdata_1):

    exit_code = calc_distance_to_border(polygons=testdata_1["src_file"],
                                        template_raster=testdata_1["template_file_raster"],
                                        dst_raster=testdata_1["dst_file_proximity"],
                                        overwrite=False,
                                        keep_interim_files=False)
    assert (exit_code == 0)

