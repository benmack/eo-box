"""
eobox.raster subpackage

Subpackage which mainly builds upon rasterio (and thus gdal) functionality.
It provides tools for 

* extracting values of raster sampledata at location given by a vector dataset,
* chunk-wise processing (e.g. classification) of multiple single layer files.


For more information on the package content, visit [readthedocs](https://eo-box.readthedocs.raster/en/latest/eobox.raster.html).

"""

from .extraction import extract
from .extraction import get_paths_of_extracted
from .extraction import load_extracted
from .extraction import add_vector_data_attributes_to_extracted
from .extraction import load_extracted_partitions
from .extraction import convert_df_to_geodf
from .extraction import load_extracted_dask
from .extraction import load_extracted_partitions_dask

from .rasterprocessing import MultiRasterIO
from .rasterprocessing import windows_from_blocksize
from .rasterprocessing import window_from_window
from .rasterprocessing import create_distance_to_raster_border

from .gdalutils import reproject_on_template_raster
from .gdalutils import rasterize

from .cube import EOCube
from .cube import EOCubeSceneCollection
from .cube import create_virtual_time_series
from .cube import create_statistical_metrics

from .utils import dtype_checker_df
from .utils import cleanup_df_values_for_given_dtype
