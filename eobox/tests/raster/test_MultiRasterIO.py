import pytest
import fnmatch
import os
from pathlib import Path
import numpy as np

from eobox.sampledata import get_dataset
from eobox.raster import MultiRasterIO

dataset = get_dataset("s2l1c")
layer_files = fnmatch.filter(dataset["raster_files"], "*B04*")
layer_files += fnmatch.filter(dataset["raster_files"], "*B08*")
layer_files += fnmatch.filter(dataset["raster_files"], "*B8A*")


def process_ndvi(arr, idx_b04, idx_b08, idx_b8a):
    """A custom function for calculating an NDVI based on the two S2 NIR bands."""
    import numpy as np

    def normalized_difference_index(arr, idx_minuend, idx_subtrahend):
        di = (arr[:, :, idx_minuend] - arr[:, :, idx_subtrahend]) / (
            arr[:, :, idx_minuend] + arr[:, :, idx_subtrahend]
        )
        di[di > 1.0] = 1.0001
        di[di < -1.0] = -1.0001
        di = (di * 10000).astype(np.int16)
        return di

    ndvi_b08 = normalized_difference_index(arr, idx_b08, idx_b04)
    ndvi_b8a = normalized_difference_index(arr, idx_b8a, idx_b04)
    return [ndvi_b08, ndvi_b8a]


def test_MultiRasterIO_after_initialization():
    mr = MultiRasterIO(layer_files=layer_files)
    assert len(mr._layer_files) == 3
    assert len(mr._layer_resolution) == 3
    assert len(mr._layer_meta) == 3
    assert len(mr._res_indices) == 2  # the unique resolutions
    assert mr.windows is None  # initialized e.g. with .block_windows
    assert mr._windows_res is mr.dst_res
    assert mr.dst_res == 10.0
    assert mr.upsampler == "nearest"
    assert mr.downsampler == "average"


def test_MultiRasterIO_after__block_windows():
    mr = MultiRasterIO(layer_files=layer_files)
    mr.block_windows()
    assert len(mr.windows) == 72
    assert len(mr.windows_row) == 72
    assert len(mr.windows_col) == 72
    assert mr._windows_res == 20.0
    assert type(mr.windows[0]).__name__ == "Window"


def test_MultiRasterIO_after__ji_windows():
    mr = MultiRasterIO(layer_files=layer_files)
    mr.block_windows()
    ji_res_wins = mr.ji_windows(0)
    assert isinstance(ji_res_wins, dict)
    assert list(ji_res_wins.keys()) == [10.0, 20.0]


def test_MultiRasterIO__get_array__with_window_index__native_resolution():
    mr = MultiRasterIO(layer_files=layer_files)
    mr.block_windows()
    # we usually do not want to set dst_res = None
    # but it is a way to get a list of the window arrays (one element per layer) in their native resolution.
    mr.dst_res = None
    arrs = mr.get_arrays(ji_win=0)
    assert isinstance(arrs, list)
    assert len(arrs) == 3
    assert len(arrs[0].shape) == 2
    assert arrs[0].shape[0] == 128
    assert arrs[0].shape[1] == 128
    assert arrs[1].shape[0] == 128
    assert arrs[1].shape[1] == 128
    assert arrs[2].shape[0] == 64
    assert arrs[2].shape[1] == 64


def test_MultiRasterIO__get_array__with_windows__native_resolution():
    mr = MultiRasterIO(layer_files=layer_files)
    mr.block_windows()
    # we usually do not want to set dst_res = None
    # but it is a way to get a list of the window arrays (one element per layer) in their native resolution.
    mr.dst_res = None
    ji_res_wins = mr.ji_windows(0)
    arrs = mr.get_arrays(
        ji_win=ji_res_wins
    )  # only this line is different compared to the previous test
    assert isinstance(arrs, list)
    assert len(arrs) == 3
    assert len(arrs[0].shape) == 2
    assert arrs[0].shape[0] == 128
    assert arrs[0].shape[1] == 128
    assert arrs[1].shape[0] == 128
    assert arrs[1].shape[1] == 128
    assert arrs[2].shape[0] == 64
    assert arrs[2].shape[1] == 64


def test_MultiRasterIO__get_array__resampled_10():
    mr = MultiRasterIO(layer_files=layer_files)
    mr.block_windows()
    arr = mr.get_arrays(0)
    assert len(arr.shape) == 3
    assert arr.shape[0] == 128
    assert arr.shape[1] == 128
    assert arr.shape[2] == 3


def test_MultiRasterIO__get_array__resampled_20():
    mr = MultiRasterIO(layer_files=layer_files, dst_res=20.0)
    mr.block_windows()
    arr = mr.get_arrays(0)
    assert len(arr.shape) == 3
    assert arr.shape[0] == 64
    assert arr.shape[1] == 64
    assert arr.shape[2] == 3


def test_MultiRasterIO__process_window():
    mr = MultiRasterIO(layer_files=layer_files, dst_res=20)
    mr.block_windows()
    result_arr_window = mr._process_window(0, func=process_ndvi, idx_b04=0, idx_b08=1, idx_b8a=2)
    assert result_arr_window is not None
    assert len(result_arr_window) == 2
    assert len(result_arr_window[0].shape) == 2
    assert result_arr_window[0].shape[0] == 64
    assert result_arr_window[0].shape[1] == 64


def test_MultiRasterIO___get_template_for_given_resolution():
    mr = MultiRasterIO(layer_files=layer_files, dst_res=20)
    path = mr._get_template_for_given_resolution(res=mr.dst_res, return_="path")
    meta = mr._get_template_for_given_resolution(res=mr.dst_res, return_="meta")
    windows = mr._get_template_for_given_resolution(res=mr.dst_res, return_="windows")
    assert os.path.exists(path)
    assert isinstance(meta, dict)
    assert len(windows) == 72


# this test takes too long!
def test_MultiRasterIO__process_windows():
    mr = MultiRasterIO(layer_files=layer_files, dst_res=20)
    mr.block_windows()
    result_arrs_windows = mr._process_windows(func=process_ndvi, idx_b04=0, idx_b08=1, idx_b8a=2)
    assert len(result_arrs_windows) == 72  # number of windows
    assert len(result_arrs_windows[0]) == 2  # number of output layers


# this test takes too long!
def test_MultiRasterIO__process_windows_merge_stack():
    mr = MultiRasterIO(layer_files=layer_files, dst_res=20)
    mr.block_windows()
    result_arr = mr._process_windows_merge_stack(
        func=process_ndvi, idx_b04=0, idx_b08=1, idx_b8a=2
    )
    assert result_arr.shape[0] == 384
    assert result_arr.shape[1] == 768
    assert result_arr.shape[2] == 2


def test_MultiRasterIO__apply_and_safe(tmpdir):
    mr = MultiRasterIO(layer_files=layer_files, dst_res=20)
    mr.block_windows()
    dst_dir = tmpdir.mkdir("temp_dst_dir")
    dst_files = [str(dst_dir / "ndvib08.jp2"), str(dst_dir / "ndvib8a.jp2")]
    assert not os.path.exists(dst_files[0])
    assert not os.path.exists(dst_files[1])
    exit_code = mr.apply_and_save(
        dst_files=dst_files, func=process_ndvi, idx_b04=0, idx_b08=1, idx_b8a=2
    )
    assert exit_code == 0
    assert os.path.exists(dst_files[0])
    assert os.path.exists(dst_files[1])
