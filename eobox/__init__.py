__version__ = "0.3.1"

from . import sampledata
from .sampledata import get_dataset

from . import raster
from .raster import extract
from .raster import load_extracted
from .raster import MultiRasterIO

from . import vector
from .vector import convert_polygons_to_lines
from .vector import calc_distance_to_border

__path__ = __import__('pkgutil').extend_path(__path__, __name__)
