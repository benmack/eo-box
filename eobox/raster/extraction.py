"""
Module for extracting values of raster sampledata at location given by a vector dataset.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from ..raster.gdalutils import rasterize

def extract(src_vector: str,
            burn_attribute: str,
            src_raster: list,
            dst_names: list,
            dst_dir: str,
            src_raster_template: str = None,
            gdal_dtype: int = 4,
            n_jobs: int = 1):
    """Extract values from list of single band raster for pixels overlapping with a vector data.

    The extracted data will be stored in the ``dst_dir`` by using the ``dst_names`` for the
    filename. If a file with a given name already exists the raster will be skipped.

    Arguments:
        src_vector {str} -- Filename of the vector dataset. Currently it must have the same CRS as
            the raster.
        burn_attribute {str} -- Name of the attribute column in the ``src_vector`` dataset to be
            stored with the extracted data. This should usually be a unique ID for the features
            (points, lines, polygons) in the vector dataset.
        src_raster {list} -- List of filenames of the single band raster files from which to
            extract.
        dst_names {list} -- List corresponding to ``src_raster`` names used to store and later
            identify the extracted to.
        dst_dir {str} -- Directory to store the data to.

    Keyword Arguments:
        src_raster_template {str} -- A template raster to be used for rasterizing the vectorfile.
            Usually the first element of ``src_raster``. (default: {None})
        gdal_dtype {int} -- Numeric GDAL data type, defaults to 4 which is UInt32.
            See https://github.com/mapbox/rasterio/blob/master/rasterio/dtypes.py for useful look-up
            tables.

    Returns:
        [int] -- If successful, 0 is returned as exit code.
    """
    if src_raster_template is None:
        src_raster_template = src_raster[0]
    path_rasterized = os.path.join(dst_dir, f"burn_attribute_rasterized_{burn_attribute}.tif")
    paths_extracted_aux = {ele: os.path.join(dst_dir, f"{ele}.npy") \
                           for ele in [f"aux_vector_{burn_attribute}",
                                       "aux_coord_x",
                                       "aux_coord_y"]}
    paths_extracted_raster = {}
    for path, name in zip(src_raster, dst_names):
        dst = f"{os.path.join(dst_dir, name)}.npy"
        if not os.path.exists(dst):
            paths_extracted_raster[path] = dst

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    # if it does not already exist, here we first create the rasterized data
    if not os.path.exists(path_rasterized):
        if src_raster_template is None:
            src_raster_template = src_raster[0]
        # print("Rasterizing vector attribute.")
        rasterize(src_vector=src_vector,
                  burn_attribute=burn_attribute,
                  src_raster_template=src_raster_template,
                  dst_rasterized=path_rasterized,
                  gdal_dtype=gdal_dtype)

    # if any of the destination files do not exist we need the locations of the pixels to be
    # extracted in form of a numpy array bool (mask_arr) that fits the rasters from which we will
    # extract below
    if not (all([os.path.exists(path) for path in paths_extracted_aux.values()]) and \
            all([os.path.exists(path) for path in paths_extracted_raster.values()])):
        # print("Creating mask array for pixels to be extracted.")
        mask_arr = _get_mask_array(path_rasterized, paths_extracted_aux, burn_attribute)
    else:
        return 0

    # create the pixel coordinates if they do not exist
    if not all([os.path.exists(paths_extracted_aux["aux_coord_x"]),
                os.path.exists(paths_extracted_aux["aux_coord_y"])]):
        _create_and_save_coords(path_rasterized, paths_extracted_aux, mask_arr)

    # lets extract the raster values in case of sequential processing
    # or remove existing raster layers to prepare parallel processing
    if n_jobs == 1:
        for path_src, path_dst in tqdm(paths_extracted_raster.items(),
                                       total=len(paths_extracted_raster)):
            _extract_and_save_one_layer(path_src, path_dst, mask_arr)
    else:
        import multiprocessing as mp
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        pool = mp.Pool(processes=n_jobs)
        _ = [pool.apply_async(_extract_and_save_one_layer,
                              args=(src, dst, mask_arr)) for \
                                  src, dst in paths_extracted_raster.items()]
        pool.close()
        pool.join()
    return 0

def _get_mask_array(path_rasterized, paths_extracted_aux, burn_attribute):
    with rasterio.open(path_rasterized) as src:
        fids_arr = src.read()
        mask_arr = fids_arr > 0
        if not os.path.exists(paths_extracted_aux[f"aux_vector_{burn_attribute}"]):
            fids = fids_arr[mask_arr]
            del fids_arr
            np.save(paths_extracted_aux[f"aux_vector_{burn_attribute}"], fids)
            del fids
    return mask_arr

def _create_and_save_coords(path_rasterized, paths_extracted_aux, mask_arr):
    src = rasterio.open(path_rasterized)
    coords = {"x": rasterio.transform.xy(src.meta["transform"],
                                         rows=[0] * src.meta["width"],
                                         cols=np.arange(src.meta["width"]),
                                         offset='center')[0],
              "y": rasterio.transform.xy(src.meta["transform"],
                                         rows=np.arange(src.meta["height"]),
                                         cols=[0] * src.meta["height"],
                                         offset='center')[1]}
    coords_2d_array_x, coords_2d_array_y = np.meshgrid(coords["x"], coords["y"])
    del coords
    np.save(paths_extracted_aux["aux_coord_x"],
            np.expand_dims(coords_2d_array_x, axis=0)[mask_arr])
    del coords_2d_array_x
    np.save(paths_extracted_aux["aux_coord_y"],
            np.expand_dims(coords_2d_array_y, axis=0)[mask_arr])
    del coords_2d_array_y


def _extract_and_save_one_layer(path_src, path_dst, mask_arr):
    with rasterio.open(path_src) as src:
        raster_vals = src.read()[mask_arr]
        np.save(path_dst, raster_vals)


def load_extracted(src_dir: str,
                   patterns="*.npy",
                   vars_in_cols: bool = True,
                   index: pd.Series = None):
    """Load data extracted and stored by :py:func:`extract`

    Arguments:
        src_dir {str} -- The directory where the data is stored.

    Keyword Arguments:
        patterns {str, or list of str} -- A pattern (str) or list of patterns (list)
            to identify the variables to be loaded.
            The default loads all variables, i.e. all .npy files. (default: {'*.npy'})
        vars_in_cols {bool} -- Return the variables in columns (``True``) or rows ``False``
            (default: {True})
        index {pd.Series} -- A boolean pandas Series which indicates with ``True`` which samples to
            load.

    Returns:
        pandas.DataFrame -- A dataframe with the data.
    """
    def _load(path, index):
        if index is None:
            arr = np.load(str(path))
        else:
            arr = np.load(str(path), mmap_mode="r")[index]
        return arr

    src_dir = Path(src_dir)
    paths = []
    if isinstance(patterns, str):
        patterns = [patterns]
    for pat in patterns:
        paths += src_dir.glob(pat)

    if vars_in_cols:
        df_data = {}
        for path in paths:
            df_data[path.stem] = _load(path, index)
            df_data = pd.DataFrame(df_data)
            if index is not None:
                df_data.index = index.index[index]
    else:
        df_data = []
        for path in paths:
            arr = _load(path, index)
            df_data.append(pd.DataFrame(np.expand_dims(arr, 0), index=[path.stem]))
        df_data = pd.concat(df_data)
        if index is not None:
            df_data.columns = index.index[index]
    return df_data
