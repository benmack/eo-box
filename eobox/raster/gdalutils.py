from os.path import relpath
from osgeo import gdal, gdalconst
from pathlib import Path


def buildvrt(input_file_list, output_file, relative=True, **kwargs):
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

    vrt = gdal.BuildVRT(output_file, input_file_list, options=vrt_options)
    vrt = None

    # if needed, create the input file paths relative to the output vrt path
    # and replace them in the vrt.
    # if desired, fix the paths and the relativeToVRT tag in the VRT
    if relative:
        input_file_list_relative = [relpath(p, Path(output_file).parent) for p in input_file_list]

        with open(output_file, "r") as file:
            # read a list of lines into data
            lines = file.readlines()

        new_lines = []
        counter = -1
        for line in lines:
            # sometimes it is relative by default
            # maybe when all files contain the parent directory of the output file (?)
            if 'relativeToVRT="1"' in line:
                counter += 1
            elif 'relativeToVRT="0"' in line:
                counter += 1
                input_file = str(input_file_list[counter])
                input_file_relative = str(input_file_list_relative[counter])
                if input_file not in line:
                    raise Exception(f"Expect path {input_file} not part of line {line}.")
                line = line.replace(input_file, input_file_relative)
                line = line.replace('relativeToVRT="0"', 'relativeToVRT="1"')
            else:
                pass
            new_lines.append(line)

        with open(output_file, "w") as file:
            file.writelines(new_lines)
    return 0


def reproject_on_template_raster(
    src_file, dst_file, template_file, resampling="near", compress=None, overwrite=False
):
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
        "q3": "GRA_Q3",
    }

    compressions = ["lzw", "packbits", "deflate"]

    if resampling not in GDAL_RESAMPLING_ALGORITHMS.keys():
        raise ValueError(
            f"'resampling must be one of {', '.join(GDAL_RESAMPLING_ALGORITHMS.keys())}"
        )

    if compress is None:
        options = []
    else:
        if compress.lower() not in compressions:
            raise ValueError(f"'compress must be one of {', '.join(compressions)}")
        else:
            options = [f"COMPRESS={compress.upper()}"]

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
    dst = gdal.GetDriverByName("GTiff").Create(
        dst_file, wide, high, 1, src_band.DataType, options=options
    )
    dst.SetGeoTransform(match_geotrans)
    dst.SetProjection(match_proj)

    # Do the work
    gdal.ReprojectImage(
        src, dst, src_proj, match_proj, getattr(gdalconst, GDAL_RESAMPLING_ALGORITHMS[resampling])
    )

    del dst  # Flush
    return 0
