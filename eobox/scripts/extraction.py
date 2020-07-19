
import pandas as pd
from pathlib import Path
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
            help="Text filepath with paths to single-band raster files from which to extract."),
        dst_dir : str = typer.Argument(...,
            help="Destination directory to store the extracted data to."),
        dst_names: str = typer.Option(None, 
            help="Text filepath with destination names corresponding to src_raster. If not given the stem of the paths in src_raster is used."),
):
    # typer.echo("Hello from the extract command ....")
    # typer.echo(f"{src_vector}")

    with open(src_raster, "r") as src:
        src_raster_list = src.read().split("\n")
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
                       dst_dir=dst_dir)
