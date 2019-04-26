""""
Base module for loading sampledata.
"""

import glob
import os

DIR_DATA = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)), "data")

def get_dataset(dataset="s2l1c"):
    """Get a specific sampledata to play around.

    So far the following sampledata exist:

    * 's2l1c': One Sentinel-2 Level 1C scene with a reference dataset.
    * 'lsts': A time series of 105 Landsat scenes each with the bands b3 (red), b4 (nir), b5 (swir1) and fmask.

    Keyword Arguments:
        dataset {str} -- The name of the dataset (default: {'s2l1c'}).

    Returns:
        [dict] -- A dictionary with paths and information about the sampledata.
    """
    if dataset == "s2l1c":
        search_string = os.path.join(DIR_DATA, dataset, "**", "*_B??.jp2")
        files = glob.glob(search_string, recursive=True)
        if not files:
            raise IOError(f"Could not find raster files of the s2l1c dataset. Search string: {search_string}")
        basename_splitted = [pth.replace(".jp2", "").split("_")[-2:] for pth in files]
        dset = {"raster_files": files,
                "raster_bands": [ele[1] for ele in basename_splitted],
                "raster_times": [ele[0] for ele in basename_splitted],
                "vector_file": os.path.join(DIR_DATA, "s2l1c", "s2l1c_ref.gpkg"),
                "vector_file_osm": os.path.join(DIR_DATA, "s2l1c", "gis_osm_landuse-water_a_free_1_area-10000-to-500000.gpkg")}

    elif dataset == "lsts":
        search_string = os.path.join(DIR_DATA, dataset, "**", "*.tif")
        files = glob.glob(search_string, recursive=True)
        if not files:
            raise IOError(f"Could not find raster files of the lsts dataset. Search string: {search_string}")
        basename_splitted = [os.path.basename(pth).replace(".tif", "").split("_") for pth in files]
        dset = {"raster_files": files,
                "raster_bands": [ele[1] for ele in basename_splitted],
                "raster_times": [ele[0][9:16] for ele in basename_splitted]}

    # If you want to add a new dataset here, do not forget to do all of the following steps:
    # 1) add the dataset in the eo-box/sampledata/eobox/sampledata/data/<name of new dataset>
    # 2) write the code here to get the paths of the data and eventually some additional information
    # 3) write a test to make sure you get the data
    # 4) add the new dataset to package_data in eo-box/sampledata/eobox/setup.py
    # 5) add the new dataset to package_data in eo-box/sampledata/MANIFEST.in
    # 4) change the version number in eo-box/sampledata/eobox/sampledata/__init__.py to '<current>.<current+1>.0'

    return dset
