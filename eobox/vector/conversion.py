import geopandas as gpd 
from pathlib import Path
from osgeo import gdal
import rasterio
import shutil
import subprocess
import tempfile
from tqdm import tqdm

from ..raster.gdalutils import rasterize
from ..raster.gdalutils import PROXIMITY_PATH


def calc_distance_to_border(polygons, template_raster, dst_raster, overwrite=False,
                            keep_interim_files=False, verbosity=0):
    """Calculate the distance of each raster cell (in and outside the polygons) to the next polygon border.
    
    Arguments:
        polygons {str} -- Filename to a geopandas-readable file with polygon features. 
        template_raster {[type]} -- Filename to a rasterio-readable file.
        dst_raster {[type]} -- Destination filename for the distance to polygon border raster file (tif).
    
    Keyword Arguments:
        overwrite {bool} -- Overwrite files if they exists? (default: {False})
        keep_interim_files {bool} -- Keep the interim line vector and raster files (default: {True})
    
    Returns:
        [type] -- [description]
    """
    if Path(dst_raster).exists() and not overwrite:
        if verbosity > 0:
            print(f"Returning 0 - File exists: {dst_raster}")
        return 0

    with rasterio.open(template_raster) as tmp:
        crs = tmp.crs
        
    dst_raster = Path(dst_raster)
    dst_raster.parent.mkdir(exist_ok=True, parents=True)

    tempdir = Path(tempfile.mkdtemp(prefix=f"TEMPDIR_{dst_raster.stem}_", dir=dst_raster.parent))
    interim_file_lines_vector = tempdir / "interim_sample_vector_dataset_lines.shp"
    interim_file_lines_raster = tempdir / "interim_sample_vector_dataset_lines.tif"

    exit_code = convert_polygons_to_lines(polygons, 
                                          interim_file_lines_vector, 
                                          crs=crs, 
                                          add_allone_col=True)
    
    rasterize(src_vector=str(interim_file_lines_vector),
              burn_attribute="ALLONE",
              src_raster_template=str(template_raster),
              dst_rasterized=str(interim_file_lines_raster),
              gdal_dtype=1)

    cmd = f"{PROXIMITY_PATH} " \
          f"{str(Path(interim_file_lines_raster).absolute())} " \
          f"{str(Path(dst_raster).absolute())} " \
          f"-ot Float32 -distunits PIXEL -values 1 -maxdist 255"
    subprocess.check_call(cmd, shell=True)
    
    if not keep_interim_files:
        shutil.rmtree(tempdir)
    else:
        if verbosity > 0:
            print(f"Interim files are in {tempdir}")
    return 0

def convert_polygons_to_lines(src_polygons, dst_lines, crs=None, add_allone_col=False):
    """Convert polygons to lines.

    Arguments:
        src_polygons {path to geopandas-readable file} -- Filename of the the polygon vector dataset to be 
            converted to lines.
        dst_lines {[type]} -- Filename where to write the line vector dataset to.

    Keyword Arguments:
        crs {dict or str} -- Output projection parameters as string or in dictionary format.
            This will reproject the data when a crs is given (not {None}) (default: {None}).
        add_allone_col {bool} -- Add an additional attribute column with all ones.
            This is useful, e.g. in case you want to use the lines with gdal_proximity afterwards (default: {True}).

    Returns:
        int -- Exit code 0 if successeful.
    """
    gdf = gpd.read_file(src_polygons)
    geom_coords = gdf["geometry"] # featureset.get(5)["geometry"]["coordinates"]
    lines = []
    row_ids = []
    for i_row, pol in tqdm(enumerate(geom_coords), total=len(geom_coords)):
        boundary = pol.boundary
        if boundary.type == 'MultiLineString':
            for line in boundary:
                lines.append(line)
                row_ids.append(i_row)
        else:
            lines.append(boundary)
            row_ids.append(i_row)

    gdf_lines = gdf.drop("geometry", axis=1).iloc[row_ids, :]
    gdf_lines["Coordinates"] = lines
    gdf_lines = gpd.GeoDataFrame(gdf_lines, geometry='Coordinates', crs=gdf.crs)
    if crs is not None:
        gdf_lines = gdf_lines.to_crs(crs)
    if add_allone_col:
        gdf_lines["ALLONE"] = 1
    Path(dst_lines).parent.mkdir(exist_ok=True, parents=True)
    gdf_lines.to_file(dst_lines)
    return 0
