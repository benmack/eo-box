"""
Module for extracting values of raster sampledata at location given by a vector dataset.
"""

import numpy as np
import geopandas as gpd
import os
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from ..raster.gdalutils import rasterize
from ..raster.rasterprocessing import create_distance_to_raster_border
from ..vector import calc_distance_to_border

def extract(src_vector: str,
            burn_attribute: str,
            src_raster: list,
            dst_names: list,
            dst_dir: str,
            dist2pb: bool = False,
            dist2rb: bool = False,
            src_raster_template: str = None,
            gdal_dtype: int = 4,
            n_jobs: int = 1) -> int:
    """Extract pixel values of a list of single-band raster files overlaying with a vector dataset.
    
    This function does not return the extracted values but stores them in the ``dst_dir`` directory.
    The extracted values of each raster will be stored as a separate *NumPy* binary file as well as 
    the values of the ``burn_attribute``. 
    Additionally, the folder will contain one or more intermediate GeoTIFF files, e.g, the 
    rasterized ``burn_attribute`` and, if selected, the ``dist2pb`` and/or ``dist2rp`` layer.

    Note that also the pixel coordinates will be extracted and stored as ``aux_coord_y`` and 
    ``aux_coord_x``. Therefore these names should be avoided in ``dst_names``.

    The function ``add_vector_data_attributes_to_extracted`` can be used to add other attributes 
    from ``src_vector`` to the store of extracted values such that they can be loaded easily 
    together with the other data.

    With ``load_extracted`` the data can then be loaded conveniently.

    If a file with a given name already exists the raster will be skipped.

    Arguments:
        src_vector {str} -- Filename of the vector dataset. Currently, it must have the same CRS as the raster.
        burn_attribute {str} -- Name of the attribute column in the ``src_vector`` dataset to be
            stored with the extracted data. This should usually be a unique ID for the features
            (points, lines, polygons) in the vector dataset. Note that this attribute should not contain zeros 
            since this value is internally used for pixels that should not be extracted, or, in other words, 
            that to not overlap with the vector data.
        src_raster {list} -- List of file paths of the single-band raster files from which to extract the pixel 
            values from.
        dst_names {list} -- List corresponding to ``src_raster`` names used to store and later
            identify the extracted to.
        dst_dir {str} -- Directory to store the data to.
        
    Keyword Arguments:
        dist2pb {bool} -- Create an additional auxiliary layer containing the distance to the closest 
            polygon border for each extracted pixels. Defaults to ``False``.
        dist2rb {bool} -- Create an additional auxiliary layer containing the distance to the closest 
            raster border for each extracted pixels. Defaults to ``False``.
        src_raster_template {str} -- A template raster to be used for rasterizing the vectorfile.
            Usually the first element of ``src_raster``. (default: {None})
        gdal_dtype {int} -- Numeric GDAL data type, defaults to 4 which is UInt32.
            See https://github.com/mapbox/rasterio/blob/master/rasterio/dtypes.py for useful look-up
            tables.
        n_jobs {int} -- Number of parallel processors to be use for extraction. -1 uses all processors.
            Defaults to 1. 

    Returns:
        [int] -- If successful the function returns 0 as an exit code and 1 otherwise.
    """
    if src_raster_template is None:
        src_raster_template = src_raster[0]
    path_rasterized = os.path.join(dst_dir, f"burn_attribute_rasterized_{burn_attribute}.tif")
    paths_extracted_aux = {ele: os.path.join(dst_dir, f"{ele}.npy") \
                           for ele in [f"aux_vector_{burn_attribute}",
                                       "aux_coord_x",
                                       "aux_coord_y"]}
    if dist2pb:
        path_dist2pb = os.path.join(dst_dir, f"aux_vector_dist2pb.tif")
        paths_extracted_aux["aux_vector_dist2pb"] = os.path.join(dst_dir, f"aux_vector_dist2pb.npy")
    if dist2rb:
        path_dist2rb = os.path.join(dst_dir, f"aux_raster_dist2rb.tif")
        paths_extracted_aux["aux_raster_dist2rb"] = os.path.join(dst_dir, f"aux_raster_dist2rb.npy")

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

    if dist2pb and not os.path.exists(paths_extracted_aux["aux_vector_dist2pb"]):
        calc_distance_to_border(polygons=src_vector,
                                template_raster=path_rasterized,
                                dst_raster=path_dist2pb,
                                overwrite=True,
                                keep_interim_files=False)
        _extract_and_save_one_layer(path_dist2pb, 
                                    paths_extracted_aux["aux_vector_dist2pb"], 
                                    mask_arr)

    if dist2rb and not os.path.exists(paths_extracted_aux["aux_raster_dist2rb"]):
        create_distance_to_raster_border(src_raster = Path(path_rasterized),
                                        dst_raster = Path(path_dist2rb),
                                        maxdist=None, # None means we calculate distances for all pixels
                                        overwrite=True)
        _extract_and_save_one_layer(path_dist2rb, 
                                    paths_extracted_aux["aux_raster_dist2rb"], 
                                    mask_arr)


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

def get_paths_of_extracted(src_dir: str,
                           patterns="*.npy", 
                           sort=True):
    """Get the paths of extracted features. Used in load_extracted."""
    src_dir = Path(src_dir)
    paths = []
    if isinstance(patterns, str):
        patterns = [patterns]
    for pat in patterns:
        if sort:
            paths_add = sorted(Path(src_dir).glob(pat))
            # paths += list(sorted(src_dir.glob(pat)))
        else:
            paths_add = src_dir.glob(pat)
        if len(paths_add) == 0 :
            raise Exception(f"Could not find any matches for: Path('{str(src_dir)}').glob('{pat}')")
        paths += paths_add

    return paths

def load_extracted(src_dir: str,
                   patterns="*.npy",
                   vars_in_cols: bool = True,
                   index: pd.Series = None,
                   head: bool = False,
                   sort: bool = True):
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
        head {bool} -- Get a dataframe with the first five samples. 

    Returns:
        pandas.DataFrame -- A dataframe with the data.
    """

    paths = get_paths_of_extracted(src_dir, patterns, sort=sort)

    src_dir = Path(src_dir)
    if head: # get a index which returns the first 5 rows
        index = _load(paths[0], index=None)
        index = pd.Series([0] * len(index), dtype=bool)
        index.loc[0:5] = True
        # print(index)


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

def load_extracted_partitions(src_dir: dict,
                              patterns="*.npy",
                              index: dict = None,
                              to_crs: dict = None,
                              verbosity=0):
    """Load multiple row-wise appended partitions (same columns) with :py:func:`load_extracted`.

    Arguments:
        src_dir {dict of str or Path} -- Multiple ``src_dir`` as in :py:func:`load_extracted` 
            wrapped in a dictionary where the keys are the partition identifiers. 
            The key will be written as column in the returning dataframe. 

    Keyword Arguments:
        patterns {str, or list of str} -- See :py:func:`load_extracted`.
        index {dict pd.Series} -- See :py:func:`load_extracted` but as dict as in ``src_dir``.

    Returns:
        pandas.DataFrame -- A dataframe with the data.
    """
    
    sort = True # does not make sense here to set to false - might cause concatenation problems 
    head = False # does not make sense to be used here
    vars_in_cols = True # not supported
    
    dfs = []
    for i, (key, src) in enumerate(src_dir.items()): 
        if verbosity > 0:
            print("*" * 80)
            print(f"Loading partition {key}")
            print("- " * 20)
        df_part = load_extracted(src, patterns, vars_in_cols=True, 
                                  index=index, head=False, sort=True)
        # # categories are nice:
        # part_col = pd.Series(pd.Categorical([key] * df_part.shape[0], 
        #                                     categories=src_dir.keys(),
        #                                     ordered=False)).to_frame().rename({0:'partition'}, axis=1)
        # # but if we save it we will get the message 'type not understood' ...
        # # ... therefore we keep it simple by now
        part_col = pd.Series([key] * df_part.shape[0]).to_frame().rename({0:'partition'}, axis=1)
        part_col.index = df_part.index
        df_part = pd.concat([part_col, df_part], axis=1)
        print(f"Shape of partition dataframe: {df_part.shape}")
        if to_crs is not None:
            print(f"Converting partition to GeoDataFrame.")
            df_part = convert_df_to_geodf(df_part, crs=src)
            if isinstance(to_crs, dict):
                if df_part.crs != to_crs:
                    print(f"Reprojecting GeoDataFrame to {to_crs}.")
                    df_part = df_part.to_crs(to_crs)
            else:
                # we set this here in the first look such that in case the projection changes we 
                # reproject later
                print(f"Setting target CRS to {df_part.crs}.")
                to_crs = df_part.crs
        dfs.append(df_part)
        # print("\n".join([Path(p).stem for p in paths_npy[tile]]))
        # print(df_aux[-1].columns)
    dfs = pd.concat(dfs, axis='rows')
    dfs = dfs.reset_index().rename({'index': 'inner_index'}, axis=1)
    if verbosity > 0:
        print("*" * 80)
        print("*" * 80)
        print(f"Shape of concatenated dataframe: {dfs.shape}")
    return dfs

def add_vector_data_attributes_to_extracted(ref_vector, pid, dir_extracted, overwrite=False):
    """From the vector dataset used for extraction save attributes as npy files corresponding to the extracted pixels values.
    
    Parameters
    ----------
    ref_vector : str or pathlib.Path
        The vector dataset which has been used in :py:func:`extract`. 
    pid : str
        The ``burn_attribute`` that has been used in :py:func:`extract`.
        Note that this only makes sense if the burn attribute is a unique feature (e.g. polygon) identifier.
    dir_extracted : str or pathlib.Path
        The output directory which has been used in :py:func:`extract`. 
    overwrite : bool, optional
        If ``True`` existing data will be overwritten, by default `False`-
    """
    df_pixels = load_extracted(dir_extracted, f'aux_vector_{pid}.npy')
    if ~isinstance(ref_vector, gpd.geodataframe.GeoDataFrame):
        ref_vector = gpd.read_file(ref_vector)
    df_pixels = df_pixels.merge(ref_vector.drop('geometry', axis=1), how='left', left_on=f'aux_vector_{pid}', right_on=pid)
    for col in df_pixels.columns:
        if df_pixels.dtypes[col] == 'object':
            print(f"Skipping column {col} - datatype 'object' not (yet) supported.")
            continue
        if col != 'aux_vector_pid':
            path_dst = Path(dir_extracted) / f'aux_vector_{col}.npy'
            if not path_dst.exists() or overwrite:
                np.save(path_dst, df_pixels[col].values)

def convert_df_to_geodf(df, crs=None):
    """Convert dataframe returned by ``load_extracted`` to a geodataframe.
    
    Parameters
    ----------
    df : dataframe
        Dataframe as returned by ``load_extracted``. It must contain the columns *aux_coord_x* and *aux_coord_y*. 
    crs : None, dict, str or Path, optional
        The crs of the saved coordinates given as a dict, e.g. ``{'init':'epsg:32632'}``, or via the ``dir_extracted``.
        In the latter (str, Path) case, it is assumed that the crs can be derived from any tiff that is located in the folder.
        The defaultdoes not set any crs. By default None.
    """
    if isinstance(crs, str) or isinstance(crs, Path):
        tif = list(crs.glob("*.tif"))[0]
        with rasterio.open(tif) as src:
            crs = src.crs
    from shapely.geometry import Point
    df["geometry"] = [Point(x, y)
                      for x, y in zip(df['aux_coord_x'].values, df['aux_coord_y'].values)]
    df = gpd.GeoDataFrame(df)
    if crs is not None:
        df.crs = crs
    return df

def _load(path, index=None, as_df=False):
    """Load a single feature stored as npy file."""
    if index is None:
        arr = np.load(str(path), allow_pickle=True)
    else:
        arr = np.load(str(path), mmap_mode="r", allow_pickle=True)[index]

    if as_df:
        arr = pd.DataFrame({path.stem: arr})
    return arr

def load_extracted_dask(npy_path_list, index=None):
    """Create a dask dataframe from a list of single features npy paths to be concatenated along the columns."""

    import dask.delayed
    from dask import delayed
    import dask.dataframe as dd
    
    @delayed
    def _load_column(path, index=None):
        """Load a single dataframe column given a numpy file path."""
        if index is None:
            arr = np.load(str(path), allow_pickle=True)
        else:
            arr = np.load(str(path), mmap_mode="r", allow_pickle=True)[index]
        df = pd.DataFrame(arr)
        df.columns = [path.stem]
        return df

    @delayed
    def _concat_columns(column_list):
        """Concatenate single dataframe columns."""
        return pd.concat(column_list, axis=1)
        
    column_list = []
    for npy_path in npy_path_list:
        column_list.append(_load_column(npy_path, index=index))
    df = _concat_columns(column_list)
    df = dd.from_delayed(df)
    
    return df

def load_extracted_partitions_dask(src_dir: dict,
                                   global_index_col: str, # e.g. "aux_index_global",
                                   patterns="*.npy",
                                   verbosity=0):
    """Load multiple row-wise appended partitions (same columns) with :py:func:`load_extracted` as dask dataframe.

    Arguments:
        src_dir {dict of str or Path} -- Multiple ``src_dir`` as in :py:func:`load_extracted` 
            wrapped in a dictionary where the keys are the partition identifiers. 
            The key will be written as column in the returning dataframe. 
    Keyword Arguments:
        global_index_col {str}: -- One of the columns matched by the patterns should be a 
            global index, i.e. a index where each element is unique over all partitions. 
        patterns {str, or list of str} -- See :py:func:`load_extracted`.

    Returns:
        dask.dataframe.core.DataFrame -- A dask dataframe with the data.
    """
    paths_npy = {}
    stems_last = None
    for i, (key, src) in enumerate(src_dir.items()):
        paths_npy[key] = get_paths_of_extracted(src_dir[key], patterns, sort=True)
        stems = [p.stem for p in paths_npy[key]]
        # check if the stems are the same - needed to concatenate the tabels...
        if stems_last is None:
            stems_last = stems.copy()
        else:
            if not all([stem == stem_last for stem, stem_last in zip(stems, stems_last)]):
                raise ValueError(f"Path stems in partition {key} do not match former stems.")
    
    dfs = []
    for i, (key, patterns_part) in enumerate(paths_npy.items()):
        if verbosity > 0:
            print(key)
        dfs.append(load_extracted_dask(patterns_part).set_index(global_index_col))
    dfs = dd.concat(dfs)
    return dfs