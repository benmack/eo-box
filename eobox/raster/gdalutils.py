from os.path import relpath
from osgeo import gdal, gdalconst, ogr
from pathlib import Path
import warnings

def _find_gdal_py_file(name):
    # find the path to the gdal_proximity.py - I found it in these places so far...
    try:
        PATH = str(list(Path(gdal.__file__).parent.rglob(name))[0])
    except:
        PATH = None
    try:
        PATH = str(list(Path(gdal.__file__).parent.parent.rglob(f"GDAL*/scripts/{name}"))[0])
    except:
        PATH = None

    if PATH is None:
        warnings.warn(f"Could not find the path of {name}: Searched in " + \
                        f"{Path(gdal.__file__).parent}, {str(Path(gdal.__file__).parent.parent)+'/GDAL*/scripts'}.")
    return PATH
PROXIMITY_PATH = _find_gdal_py_file(name="gdal_proximity.py")
POLYGONIZE_PATH = _find_gdal_py_file(name="gdal_polygonize.py")

def buildvrt(input_file_list, output_file,
             relative=True, **kwargs):
    """Build a VRT

    See also: https://www.gdal.org/gdalbuildvrt.html

    You can find the possible BuildVRTOptions (**kwargs**) here:
    https://github.com/nextgis/pygdal/blob/78a793057d2162c292af4f6b240e19da5d5e52e2/2.1.0/osgeo/gdal.py#L1051

    Arguments:
        input_file_list {list of str or Path objects} -- List of input files.
        output_file {str or Path object} -- Output file (VRT).

    Keyword Arguments:
        relative {bool} -- If ``True``, the ``input_file_list`` paths are converted to relative
            paths (relative to the output file) and the VRT works even if the data is moved somewhere else -
            given that the relative location of theVRT and the input files does not chance!
        **kwargs {} -- BuildVRTOptions - see function description for a link to .

    Returns:
        [int] -- If successful, 0 is returned as exit code.
    """

    # create destination directory
    if not Path(output_file).parent.exists():
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # make sure we have absolute paths and strings since BuildVRT does not like something else
    input_file_list = [str(Path(p).absolute()) for p in input_file_list]
    output_file = str(Path(output_file).absolute())

    vrt_options = gdal.BuildVRTOptions(**kwargs)

    vrt = gdal.BuildVRT(output_file,
                        input_file_list,
                        options=vrt_options)
    vrt = None

    # if needed, create the input file paths relative to the output vrt path
    # and replace them in the vrt.
    # if desired, fix the paths and the relativeToVRT tag in the VRT
    if relative:
        input_file_list_relative = [relpath(p, Path(output_file).parent) for p in input_file_list]

        with open(output_file, 'r') as file:
            # read a list of lines into data
            lines = file.readlines()

        new_lines = []
        counter = -1
        for line in lines:
            # sometimes it is relative by default
            # maybe when all files contain the parent directory of the output file (?)
            if "relativeToVRT=\"1\"" in line:
                counter += 1
            elif "relativeToVRT=\"0\"" in line:
                counter += 1
                input_file = str(input_file_list[counter])
                input_file_relative = str(input_file_list_relative[counter])
                if input_file not in line:
                    raise Exception(f"Expect path {input_file} not part of line {line}.")
                line = line.replace(input_file,
                                    input_file_relative)
                line = line.replace("relativeToVRT=\"0\"",
                                    "relativeToVRT=\"1\"")
            else:
                pass
            new_lines.append(line)

        with open(output_file, 'w') as file:
            file.writelines(new_lines)
    return 0

def reproject_on_template_raster(src_file, dst_file, template_file, resampling="near", compress=None, overwrite=False):
    """Reproject a one-band raster to fit the projection, extend, pixel size etc. of a template raster.  
    
    Function based on https://stackoverflow.com/questions/10454316/how-to-project-and-resample-a-grid-to-match-another-grid-with-gdal-python

    Arguments:
        src_file {str} -- Filename of the source one-band raster. 
        dst_file {str} -- Filename of the destination raster. 
        template_file {str} -- Filename of the template raster.
        resampling {str} -- Resampling type:
             'near' (default), 'bilinear', 'cubic', 'cubicspline', 'lanczos', 'average', 'mode', 'max', 'min', 'med', 'q1', 'q3',
            see https://www.gdal.org/gdalwarp.html -r parameter.
        compress {str} -- Compression type: None (default), 'lzw', 'packbits', 'defalte'.
    """
    
    if not overwrite and Path(dst_file).exists():
        print("Processing skipped. Destination file exists.")
        return 0
    
    GDAL_RESAMPLING_ALGORITHMS = {
        "bilinear": "GRA_Bilinear",
        "cubic": "GRA_Cubic",
        "cubicspline": "GRA_CubicSpline",
        "lanczos": "GRA_Lanczos",
        "average": "GRA_Average",
        "mode": "GRA_Mode",
        "max": "GRA_Max",
        "min": "GRA_Min",
        "med": "GRA_Med",
        "near": "GRA_NearestNeighbour",
        "q1": "GRA_Q1",
        "q3": "GRA_Q3"
    }

    compressions = ["lzw", "packbits", "deflate"]

    if resampling not in GDAL_RESAMPLING_ALGORITHMS.keys():
        raise ValueError(f"'resampling must be one of {', '.join(GDAL_RESAMPLING_ALGORITHMS.keys())}")

    if compress is None:
        options = []
    else:
        if compress.lower() not in compressions:
            raise ValueError(f"'compress must be one of {', '.join(compressions)}")
        else:
            options = [f'COMPRESS={compress.upper()}']
    
    # Source
    src = gdal.Open(src_file, gdalconst.GA_ReadOnly)
    src_band = src.GetRasterBand(1)
    src_proj = src.GetProjection()

    # We want a section of source that matches this:
    match_ds = gdal.Open(template_file, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize

    # Output / destination
    Path(dst_file).parent.mkdir(parents=True, exist_ok=True)
    dst = gdal.GetDriverByName('GTiff').Create(dst_file, wide, high, 1, src_band.DataType, options=options)
    dst.SetGeoTransform( match_geotrans )
    dst.SetProjection( match_proj)

    # Do the work
    gdal.ReprojectImage(src, dst, src_proj, match_proj, 
                        getattr(gdalconst, GDAL_RESAMPLING_ALGORITHMS[resampling]))


    del dst # Flush
    return 0

def rasterize(src_vector: str,
              burn_attribute: str,
              src_raster_template: str,
              dst_rasterized: str,
              gdal_dtype: int = 4):
    """Rasterize the values of a spatial vector file.

    Arguments:
        src_vector {str}} -- A OGR vector file (e.g. GeoPackage, ESRI Shapefile) path containing the
            data to be rasterized.
        burn_attribute {str} -- The attribute of the vector data to be burned in the raster.
        src_raster_template {str} -- Path to a GDAL raster file to be used as template for the
            rasterized data.
        dst_rasterized {str} -- Path of the destination file.
        gdal_dtype {int} -- Numeric GDAL data type, defaults to 4 which is UInt32.
            See https://github.com/mapbox/rasterio/blob/master/rasterio/dtypes.py for useful look-up
            tables.
    Returns:
        None
    """

    data = gdal.Open(str(src_raster_template),  # str for the case that a Path instance arrives here
                     gdalconst.GA_ReadOnly)
    geo_transform = data.GetGeoTransform()
    #source_layer = data.GetLayer()
    # x_max = x_min + geo_transform[1] * data.RasterXSize
    # y_min = y_max + geo_transform[5] * data.RasterYSize
    x_res = data.RasterXSize
    y_res = data.RasterYSize
    mb_v = ogr.Open(src_vector)
    mb_l = mb_v.GetLayer()
    target_ds = gdal.GetDriverByName('GTiff').Create(dst_rasterized,
                                                     x_res, y_res, 1,
                                                     gdal_dtype)  # gdal.GDT_Byte
    # import osr
    target_ds.SetGeoTransform((geo_transform[0],  # x_min
                               geo_transform[1],  # pixel_width
                               0,
                               geo_transform[3],  # y_max
                               0,
                               geo_transform[5]  # pixel_height
                               ))
    prj = data.GetProjection()
    # srs = osr.SpatialReference(wkt=prj)  # Where was this needed?
    target_ds.SetProjection(prj)
    band = target_ds.GetRasterBand(1)
    # NoData_value = 0
    # band.SetNoDataValue(NoData_value)
    band.FlushCache()
    gdal.RasterizeLayer(target_ds, [1], mb_l, options=[f"ATTRIBUTE={burn_attribute}"])

    target_ds = None
