
import glob
import pandas as pd
from pathlib import Path
import shutil
import typer

from ..raster import extraction

app = typer.Typer(help="Extract and load raster values over vector files.")

@app.command(help="Extract raster values ")
def extract(
        src_vector: str = typer.Argument(...,
            help="Vector filpath."),
        burn_attribute: str = typer.Argument(...,
            help="Attribute table field of the src_vector dataset to be stored along with the raster values."),
        src_raster: str = typer.Argument(...,
            help="Raster path matching specification passed to glob.glob (e.g. ./**/*.jp2) or text filepath with paths to single-band raster files from which to extract."),
        dst_dir : str = typer.Argument(...,
            help="Destination directory to store the extracted data to."),
        dst_names: str = typer.Option(None, 
            help="Text filepath with destination names corresponding to src_raster. If not given the stem of the paths in src_raster is used."),
        recursive: bool = typer.Option(True,
            help="Define if you want to search for matching rasters recursively. Only used if src_raster is a raster path matching specification."),
        dist2pb: bool = typer.Option(True,
            help="Generate distance to polygon border for each extracted pixel."),
        dist2rb: bool = typer.Option(True,
            help="Generate distance to tile border for each extracted pixel."),
        dst_parquet: str = typer.Option(None,
            help="Path to store all extracted numpy files additionally as parquet (engine=auto, compression=GZIP)."),
#         delete_npy: str = typer.Option(False,
#             help="Delete numpy files after writing parquet - only considered if dst_parquet is given."),
        delete_dst_dir: bool = typer.Option(False,
            help="Delete the whole destination dir - only if dst_parquet is given."), 
            # not only the npy files as possible with delete_npy 
):
    if Path(src_raster).exists():
        with open(src_raster, "r") as src:
            src_raster_list = src.read().split("\n")
    else:
        src_raster_list = glob.glob(src_raster, recursive=recursive)
        if not len(src_raster_list):
            raise Exception(f"Could not match any files with 'glob.glob(\"{src_raster}\", recursive={recursive})' called from path {str(Path('.').resolve())}.")
    
    if "" in src_raster_list:
        src_raster_list.remove("")
    for rpath in src_raster_list:
        if not Path(rpath).exists():
            raise Exception(f"Raster file does not exist:{rpath}")
    if dst_names is None:
        dst_names_list = [Path(rpath).stem for rpath in src_raster_list]
    else:
        with open(dst_names, "r") as src:
            dst_names_list = src.read().split("\n")
        if "" in dst_names_list:
            dst_names_list.remove("")
        if len(dst_names_list) != len(src_raster_list):
            raise Exception(f"Number of dst_names {len(dst_names_list)} and src_raster {len(src_raster_list)} do not match.")

    extraction.extract(src_vector=src_vector,
                       burn_attribute=burn_attribute,
                       src_raster=src_raster_list,
                       dst_names=dst_names_list,
                       dst_dir=dst_dir,
                       dist2pb=dist2pb,
                       dist2rb=dist2rb,
                      )
    if dst_parquet:
        pths = extraction.get_paths_of_extracted(dst_dir,
                                          patterns=["aux*.npy"] + [f"{dn}.npy" for dn in dst_names_list])
        extraction.load_extracted_dask(pths).to_parquet(dst_parquet, compression="GZIP", engine="pyarrow")
        if delete_dst_dir:
            shutil.rmtree(dst_dir)