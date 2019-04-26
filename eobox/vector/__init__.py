"""
eobox.vector subpackage

Subpackage which mainly builds upon shapely, fiona and geopandas (and thus ogr) functionality.
It provides tools for 

* clean convertion of polygons to lines.


For more information on the package content, visit [readthedocs](https://eo-box.readthedocs.raster/en/latest/eobox.vector.html).

"""

from . import conversion

from .conversion import calc_distance_to_border
from .conversion import convert_polygons_to_lines
