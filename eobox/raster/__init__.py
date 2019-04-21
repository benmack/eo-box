"""
eobox.raster subpackage

Subpackage which mainly builds upon rasterio (and thus gdal) functionality.
It provides tools for 

* extracting values of raster sampledata at location given by a vector dataset,
* chunk-wise processing (e.g. classification) of multiple single layer files.


For more information on the package content, visit [readthedocs](https://eo-box.readthedocs.raster/en/latest/eobox.raster.html).

"""

from .extraction import extract
from .extraction import load_extracted
from .rasterprocessing import MultiRasterIO
from .rasterprocessing import windows_from_blocksize
