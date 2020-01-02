import numpy as np
from pathlib import Path
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling
from rasterio import transform
import shutil
import subprocess
import tempfile

from .gdalutils import PROXIMITY_PATH

class MultiRasterIO():
    def __init__(self, layer_files: list,
                 dst_res: float = None,
                 downsampler: str = 'average',
                 upsampler: str = 'nearest'):
        """Read, process and write multiple single layer raster files.

        Arguments:
            layer_files {list} -- List of paths to single band raster files.
            dst_res {float} -- Destination resolution. If `None` it is set to the highest resolution of all layers.

        Keyword Arguments:
            dst_res {numeric} -- Destination resolution in which to return the array.
                If `None`, each layer is returned in its native resolution,
                else each layer is resampled to the destination resolution `dst_res` (default: {None}).
            downsampler {str} -- The method used for downsampling layers with higher
                raster cell size (default: {'average'}).
            upsampler {str} -- The method used for upsampling layers with lower
                raster cell size  (default: {'nearest'})
        """

        # build the multi-resolution data frame here
        self._layer_files = layer_files
        self._layer_resolution = []  # make these public but with tupels ???
        self._layer_meta = []
        self._res_indices = {}
        self.windows = None
        self.windows_row = None
        self.windows_col = None
        for i, filename in enumerate(self._layer_files):
            with rasterio.open(filename) as src:
                res = src.meta["transform"][0]
                self._layer_resolution.append(res)
                self._layer_meta.append(src.meta)
            if res not in self._res_indices.keys():
                self._res_indices[res] = [i]
            else:
                self._res_indices[res].append(i)
        self.dst_res = self._get_dst_resolution(dst_res)
        self._windows_res = self.dst_res
        self.downsampler = downsampler
        self.upsampler = upsampler

    # def windows(self, windows, resolution): # setter and getter ?
    #     raise NotImplementedError()
        # ...
        # self.windows = ...

    def _get_dst_resolution(self, dst_res=None):
        """Get default resolution, i.e. the highest resolution or smallest cell size."""
        if dst_res is None:
            dst_res = min(self._res_indices.keys())
        return dst_res

    def block_windows(self, res=None):  # setter and getter ?
        """Load windows for chunks-wise processing from raster internal tiling (first raster of given resolution).

        Arguments:
            res {numeric} -- Resolution determining the raster (1st of resolution group) from which to take the tiling.
        """

        if res is None:
            res = max(self._res_indices.keys())
        self._windows_res = res
        a_file_index_given_res = self._res_indices[res][0]
        with rasterio.open(self._layer_files[a_file_index_given_res]) as src:
            wins_of_first_dst_res_layer = tuple(src.block_windows())
        self.windows = np.array([win[1] for win in wins_of_first_dst_res_layer])
        self.windows_row = np.array([win[0][0] for win in wins_of_first_dst_res_layer])
        self.windows_col = np.array([win[0][1] for win in wins_of_first_dst_res_layer])

    def windows_from_blocksize(self, blocksize_xy=512):
        """Create rasterio.windows.Window instances with given size which fully cover the raster.

        Arguments:
            blocksize_xy {int or list of two int} -- Size of the window. If one integer is given it defines
                the width and height of the window. If a list of two integers if given the first defines the
                width and the second the height.

        Returns:
            None -- But the attributes ``windows``, ``windows_row`` and ``windows_col`` are updated.
        """
        meta = self._get_template_for_given_resolution(self.dst_res, "meta")
        width = meta["width"]
        height = meta["height"]
        blocksize_wins = windows_from_blocksize(blocksize_xy, width, height)

        self.windows = np.array([win[1] for win in blocksize_wins])
        self.windows_row = np.array([win[0][0] for win in blocksize_wins])
        self.windows_col = np.array([win[0][1] for win in blocksize_wins])
        return self

    def _get_template_for_given_resolution(self, res, return_):
        """Given specified resolution ('res') return template layer 'path', 'meta' or 'windows'."""
        path = self._layer_files[self._res_indices[res][0]]
        if return_ == "path":
            return_value = path
        else:
            with rasterio.open(str(path)) as src:
                if return_ == "meta":
                    return_value = src.meta
                elif return_ == "windows":
                    return_value = tuple(src.block_windows())
                else:
                    raise ValueError("'return_' must be 'path', meta' or 'windows'.")
        return return_value

    def windows_df(self):
        """Get Windows (W) W-row, W-col and W-index of windows e.g. loaded with :meth:`block_windows` as a dataframe.

        Returns:
            [dataframe] -- A dataframe with the window information and indices (row, col, index).
        """

        import pandas as pd
        if self.windows is None:
            raise Exception("You need to call the block_windows or windows before.")
        df_wins = []
        for row, col, win in zip(self.windows_row, self.windows_col, self.windows):
            df_wins.append(pd.DataFrame({"row":[row], "col":[col], "Window":[win]}))
        df_wins = pd.concat(df_wins).set_index(["row", "col"])
        df_wins["window_index"] = range(df_wins.shape[0])
        df_wins = df_wins.sort_index()
        return df_wins

    def ji_windows(self, ij_win):  # what can be given to ij_win NOT intuitive/right name by now!!!
        """For a given specific window, i.e. an element of :attr:`windows`, get the windows of all resolutions.

        Arguments:
            ij_win {int} -- The index specifying the window for which to return the resolution-windows.
        """
        ji_windows = {}
        transform_src = self._layer_meta[self._res_indices[self._windows_res][0]]["transform"]
        for res in self._res_indices:
            transform_dst = self._layer_meta[self._res_indices[res][0]]["transform"]
            ji_windows[res] = window_from_window(window_src=self.windows[ij_win],
                                                 transform_src=transform_src,
                                                 transform_dst=transform_dst)
        return ji_windows

    def get_arrays(self, ji_win):
        """Get the data of the a window given the ji_windows derived with :method:`ji_windows`.

        Arguments:
            ji_win {[type]} -- The index of the window or the (multi-resolution) windows returned by :meth:`ji_window`.

        Returns:
            (list of) array(s) -- List of 2D arrays in native resolution in case `dst_res` is `None`
                or a 3D array where all layers are resampled to `dst_res` resolution.
        """
        if isinstance(ji_win, dict):
            ji_windows = ji_win
        else:
            ji_windows = self.ji_windows(ji_win)

        arrays = []
        for filename, res in zip(self._layer_files, self._layer_resolution):
            with rasterio.open(filename) as src:
                arr = src.read(1, window=ji_windows[res])
            arrays.append(arr)
        if self.dst_res is not None:
            arrays = self._resample(arrays=arrays, ji_windows=ji_windows)
        return arrays

    def _resample(self, arrays, ji_windows):
        """Resample all arrays with potentially different resolutions to a common resolution."""
        # get a destination array template
        win_dst = ji_windows[self.dst_res]
        aff_dst = self._layer_meta[self._res_indices[self.dst_res][0]]["transform"]
        arrays_dst = list()
        for i, array in enumerate(arrays):
            arr_dst = np.zeros((int(win_dst.height), int(win_dst.width)))
            if self._layer_resolution[i] > self.dst_res:
                resampling = getattr(Resampling, self.upsampler)
            elif self._layer_resolution[i] < self.dst_res:
                resampling = getattr(Resampling, self.downsampler)
            else:
                arrays_dst.append(array.copy())
                continue
            reproject(array, arr_dst,  # arr_dst[0, :, :, i],
                      src_transform=self._layer_meta[i]["transform"],
                      dst_transform=aff_dst,
                      src_crs=self._layer_meta[0]["crs"],
                      dst_crs=self._layer_meta[0]["crs"],
                      resampling=resampling)
            arrays_dst.append(arr_dst.copy())
        arrays_dst = np.stack(arrays_dst, axis=2) # n_images x n x m x 10 would be the synergise format
        return arrays_dst

    def apply_and_save(self, dst_files, func, **kwargs):

        result_0 = self._process_window(0, func, **kwargs)
        if len(dst_files) != len(result_0):
            raise ValueError("The number of file paths in 'dst' need to match the number of output layers.")
        meta = self._get_template_for_given_resolution(res=self.dst_res, return_="meta")

        dtypes = [arr.dtype for arr in result_0]
        dsts = []
        for layer_idx, dst_name_dtype in enumerate(zip(dst_files, dtypes)):
            dst_name, dtype = dst_name_dtype
            Path(dst_name).parent.mkdir(parents=True, exist_ok=True)
            meta.update(dtype=dtype)
            dst = rasterio.open(str(dst_name), "w", **meta)
            ji_windows = self.ji_windows(0)
            dst.write(result_0[layer_idx], 1, window=ji_windows[self.dst_res])
            dsts.append(dst)

        for ji_win in range(1, len(self.windows)):
            result_ji = self._process_window(ji_win, func, **kwargs)
            for layer_idx, dtype in enumerate(dtypes):
                Path(dst_name).parent.mkdir(parents=True, exist_ok=True)
                meta.update(dtype=dtype)
                ji_windows = self.ji_windows(ji_win)
                dsts[layer_idx].write(result_ji[layer_idx], 1, window=ji_windows[self.dst_res])

        for dst in dsts:
            dst.close()

        return 0

    def _process_windows_merge_stack(self, func, **kwargs):
        """Load (resampled) array of all windows, apply custom function on it, merge and stack results to one array."""
        ji_results = self._process_windows(func, **kwargs)
        for idx_layer in range(len(ji_results[0])):  # this is the number of output layers
            for j in np.unique(self.windows_row):
                win_indices_j = np.where(self.windows_row == j)[0]
                layer_merged_j = np.hstack([ji_results[idx][idx_layer] for idx in win_indices_j])
                if j == 0:
                    layer_merged = layer_merged_j
                else:
                    layer_merged = np.vstack([layer_merged, layer_merged_j])
            if idx_layer == 0:
                layers_merged = layer_merged
            else:
                layers_merged = np.stack([layers_merged, layer_merged], axis=2)
        return layers_merged

    def _process_windows(self, func, **kwargs):
        """Load (resampled) array of all windows and apply custom function on it."""
        ji_results = []
        for ji_win in range(len(self.windows)):
            ji_results.append(self._process_window(ji_win, func, **kwargs))
        return ji_results

    def _process_window(self, ji_win, func, **kwargs):
        """Load (resampled) array of window ji_win and apply custom function on it. """
        arr = self.get_arrays(ji_win)
        result = func(arr, **kwargs)
        return result

    def get_window_from_xy(self, xy):
        """Get the window index given a coordinate (raster CRS)."""
        a_transform = self._get_template_for_given_resolution(res=self.dst_res, return_="meta")["transform"]
        row, col = transform.rowcol(a_transform, xy[0], xy[1])
        ij_containing_xy = None
        for ji, win in enumerate(self.windows):
            (row_start, row_end), (col_start, col_end) = rasterio.windows.toranges(win)
            # print(row, col, row_start, row_end, col_start, col_end)
            if ((col >= col_start) & (col < col_end)) & ((row >= row_start) & (row < row_end)):
                ij_containing_xy = ji
                break
        if ij_containing_xy is None:
            raise ValueError("The given 'xy' value is not contained in any window.")
        return ij_containing_xy

def window_from_window(window_src, transform_src, transform_dst):
    # extend that transform can be a filename, rasterio dataset, meta (dict), or Affine
    spatial_bounds = rasterio.windows.bounds(window=window_src, transform=transform_src,
                                             height=0, width=0)  # defaults
    window_dst = rasterio.windows.from_bounds(spatial_bounds[0],
                                              spatial_bounds[1],
                                              spatial_bounds[2],
                                              spatial_bounds[3],
                                              transform=transform_dst,
                                              height=None, width=None, precision=None)  # defaults
    return window_dst

def windows_from_blocksize(blocksize_xy, width, height):
    """Create rasterio.windows.Window instances with given size which fully cover a raster.

    Arguments:
        blocksize_xy {int or list of two int} -- [description]
        width {int} -- With of the raster for which to create the windows.
        height {int} -- Heigth of the raster for which to create the windows.

    Returns:
        list -- List of windows according to the following format
            ``[[<row-index>, <column index>], rasterio.windows.Window(<col_off>, <row_off>, <width>, <height>)]``.
    """

    # checks the blocksize input
    value_error_msg = "'blocksize must be an integer or a list of two integers.'"
    if isinstance(blocksize_xy, int):
        blockxsize, blockysize = (blocksize_xy, blocksize_xy)
    elif isinstance(blocksize_xy, list):
        if len(blocksize_xy) != 2:
            raise ValueError(value_error_msg)
        else:
            if not all([isinstance(blocksize_xy[0], int), isinstance(blocksize_xy[1], int)]):
                raise ValueError(value_error_msg)
            blockxsize, blockysize = blocksize_xy
    else:
        raise ValueError(value_error_msg)

    # create the col_off and row_off elements for all windows
    n_cols = int(np.ceil(width / blockxsize))
    n_rows = int(np.ceil(height / blockysize))
    col = list(range(n_cols)) * n_rows
    col_off = np.array(col) * blockxsize
    row = np.repeat(list(range(n_rows)), n_cols)
    row_off = row * blockysize

    # create the windows
    # if necessary, reduce the width and/or height of the border windows
    blocksize_wins = []
    for ridx, roff, cidx, coff, in zip(row, row_off, col, col_off):
        if coff + blockxsize > width:
            bxsize = width - coff
        else:
            bxsize = blockxsize
        if roff + blockysize > height:
            bysize = height - roff
        else:
            bysize = blockysize
        blocksize_wins.append([[ridx, cidx], rasterio.windows.Window(coff, roff, bxsize, bysize)])
    return blocksize_wins

def create_distance_to_raster_border(src_raster, dst_raster, maxdist=None, overwrite=False):
    """Create a raster with pixels values being the distance to the raster border (pixel distances)."""
    if not dst_raster.exists() or overwrite:

        # from a template raster create a raster where the outer pixel rows and columns are 1
        # and the rest are 0
        with rasterio.open(src_raster) as src:
            arr = (src.read() * 0).astype('uint8')
            arr[0, 0, :] = 1
            arr[0, arr.shape[1]-1, :] = 1
            arr[0, :, 0] = 1
            arr[0, :, arr.shape[2]-1] = 1
            meta = src.meta

            meta.update(dtype='uint8')
            temp_dir = Path(tempfile.mkdtemp(prefix=f"TEMPDIR_{dst_raster.stem}_", dir=dst_raster.parent))
            temp_file = temp_dir / "frame.tif"
            with rasterio.open(temp_file, 'w', **meta) as dst:
                dst.write(arr)
        
        # calculate the proximity to the raster border

        # get the minimum datatype necessary to store the distances
        if maxdist is None:
            maxdist = int(np.ceil(max(arr.shape) / 2))
        dtype = rasterio.dtypes.get_minimum_dtype(maxdist)

        output_format = rasterio.dtypes.typename_fwd[rasterio.dtypes.dtype_rev[dtype]]
        cmd = f"{PROXIMITY_PATH} " \
              f"{str(Path(temp_file).absolute())} " \
              f"{str(Path(dst_raster).absolute())} " \
              f"-co COMPRESS=DEFLATE " \
              f"-ot {output_format} -distunits PIXEL -values 1 -maxdist {maxdist}"
        subprocess.check_call(cmd, shell=True)
        shutil.rmtree(temp_dir)