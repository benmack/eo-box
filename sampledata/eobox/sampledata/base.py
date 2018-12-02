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

    * 's2l1c'

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
                "vector_file": os.path.join(DIR_DATA, "s2l1c", "s2l1c_ref.gpkg")}
    return dset
