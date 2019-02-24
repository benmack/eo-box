import pytest
import pandas as pd
from pathlib import Path

from eobox import sampledata
from eobox.raster import cube


@pytest.fixture
def eocube_input_1(tmpdir):
    year = 2008
    dataset = sampledata.get_dataset("lsts")
    layers_paths = [Path(p) for p in dataset["raster_files"]]
    layers_df = pd.Series([p.stem for p in layers_paths]).str.split("_", expand=True) \
                .rename({0: "sceneid", 1:"band"}, axis=1)

    layers_df["date"] = pd.to_datetime(layers_df.sceneid.str[9:16], format="%Y%j")
    layers_df["uname"] = layers_df.sceneid.str[:3] + "_" + layers_df.date.dt.strftime("%Y-%m-%d") + \
                         "_" + layers_df.band.str[::] 
    layers_df["path"] = layers_paths

    layers_df = layers_df.sort_values(["date", "band"])
    layers_df = layers_df.reset_index(drop=True)

    layers_df_year = layers_df[(layers_df.date >= str(year)) & (layers_df.date < str(year+1))]
    layers_df_year = layers_df_year.reset_index(drop=True)
    input_kwargs = {
            "df_layers": layers_df_year,
            "tmpdir": tmpdir.mkdir("temp_dst_dir-0")
    }
    return input_kwargs

@pytest.fixture
def eocube_input_onescene(tmpdir):
    dataset = sampledata.get_dataset("lsts")
    layers_paths = [Path(p) for p in dataset["raster_files"][:4]]
    layers_df = pd.Series([p.stem for p in layers_paths]).str.split("_", expand=True) \
    .rename({0: "sceneid", 1:"band"}, axis=1)

    layers_df["date"] = pd.to_datetime(layers_df.sceneid.str[9:16], format="%Y%j")
    layers_df["uname"] = layers_df.sceneid.str[:3] + "_" + layers_df.date.dt.strftime("%Y-%m-%d") + "_" + layers_df.band.str[::] 
    layers_df["path"] = layers_paths

    layers_df = layers_df.sort_values(["date", "band"])
    layers_df = layers_df.reset_index(drop=True)

    a_tmpdir = tmpdir.mkdir("temp_dst_dir-onescene")
    # print(a_tmpdir)
    dst_paths = [Path(a_tmpdir) / (Path(p).stem+".vrt") for p in layers_df["path"]]

    input_kwargs = {
            "layers_df": layers_df,
            "tmpdir": a_tmpdir,
            "dst_paths": dst_paths
    }
    return input_kwargs

def test_eocube_initialization(eocube_input_1):
    df_layers = eocube_input_1["df_layers"]
    eoc = cube.EOCube(df_layers, chunksize=2**5)
    assert eoc.df_layers.shape[0] == 92
    assert eoc.chunksize == 2**5
    assert eoc.n_chunks == 4
    
def test_eocube_chunksize(eocube_input_1):
    df_layers = eocube_input_1["df_layers"]
    eoc = cube.EOCube(df_layers, chunksize=2**6)
    assert eoc.chunksize == 2**6
    assert eoc.n_chunks == 1
    eoc.chunksize = 2**5
    assert eoc.chunksize == 2**5
    assert eoc.n_chunks == 4

def test_eocube_get_chunk(eocube_input_1):
    df_layers = eocube_input_1["df_layers"]
    ji = 0
    
    eoc = cube.EOCube(df_layers, chunksize=2**5)
    eoc_chunk = eoc.get_chunk(ji)

    #assert isinstance(eoc_chunk, eocube.EOCubeChunk)
    assert type(eoc_chunk).__name__ == "EOCubeChunk"
    assert eoc_chunk.ji == 0

# eoc_chunk - EOCubeChunk class
def test_eoc_chunk_initialization(eocube_input_1):
    df_layers = eocube_input_1["df_layers"]
    ji = 0
    eoc_chunk = cube.EOCubeChunk(ji, df_layers, chunksize=2**5)
    assert eoc_chunk.df_layers.shape[0] == 92
    assert eoc_chunk.chunksize == 2**5
    assert eoc_chunk.ji == 0

def test_eoc_chunk_initialization_from_eocube(eocube_input_1):
    df_layers = eocube_input_1["df_layers"]
    ji = 0
    
    eoc = cube.EOCube(df_layers, chunksize=2**5)
    eoc_chunk_from_eocube = cube.EOCubeChunk.from_eocube(eoc, ji)
    eoc_chunk = cube.EOCubeChunk(ji, df_layers, chunksize=2**5)
    
    assert eoc_chunk_from_eocube.chunksize == eoc_chunk.chunksize 
    assert eoc_chunk_from_eocube._mrio.windows[0] == eoc_chunk._mrio.windows[0] 

def test_eoc_chunk_chunksize_setter_raises_error(eocube_input_1):
    df_layers = eocube_input_1["df_layers"]
    eoc_chunk = cube.EOCubeChunk(0, df_layers, chunksize=2**5)
    with pytest.raises(NotImplementedError):
        eoc_chunk.chunksize = 2**4 

def test_eoc_chunk_read_data(eocube_input_1):
    df_layers = eocube_input_1["df_layers"]
    eoc_chunk = cube.EOCubeChunk(0, df_layers, chunksize=2**5)
    eoc_chunk = eoc_chunk.read_data()

    assert len(eoc_chunk.data.shape) == 3
    assert eoc_chunk._data_structure == "ndarray"

def test_eoc_chunk_convert_data_to_dataframe(eocube_input_1):
    df_layers = eocube_input_1["df_layers"]
    eoc_chunk = cube.EOCubeChunk(1, df_layers, chunksize=2**5)
    eoc_chunk = eoc_chunk.read_data()

    eoc_chunk.convert_data_to_dataframe()
    assert eoc_chunk._data_structure == "DataFrame"
    assert (eoc_chunk.data.columns == eoc_chunk.df_layers["uname"]).all()

    with pytest.raises(Exception):
        eoc_chunk.convert_data_to_dataframe()

def test_eoc_chunk_convert_data_to_ndarray(eocube_input_1):
    df_layers = df_layers = eocube_input_1["df_layers"]
    eoc_chunk = cube.EOCubeChunk(1, df_layers, chunksize=2**5)
    eoc_chunk = eoc_chunk.read_data()

    assert eoc_chunk._data_structure == "ndarray"
    with pytest.raises(Exception):
        eoc_chunk.convert_data_to_ndarray()

    # convert to dataframe and back and compare ndarrays
    data_ndarray = eoc_chunk.data.copy()
    eoc_chunk = eoc_chunk.convert_data_to_dataframe()
    assert eoc_chunk._data_structure == "DataFrame"
    eoc_chunk = eoc_chunk.convert_data_to_ndarray()
    assert (data_ndarray == eoc_chunk.data).all()

# back to EOCube tests
def test_eocube_read_and_write(eocube_input_onescene):

    def convert_to_chunk_data(eoc_chunk, dst_paths):
        result = eoc_chunk.read_data().data
        eoc_chunk.write_ndarray(result=result, dst_paths=dst_paths)
        return dst_paths
    df_layers = eocube_input_onescene["layers_df"]
    tmpdir = eocube_input_onescene["tmpdir"]
    dst_paths = eocube_input_onescene["dst_paths"]
    eoc = cube.EOCube(df_layers, chunksize=2**5)
    
    assert dst_paths[0].parent.exists()
    assert len(list(Path(tmpdir).rglob("*.vrt"))) == 0
    assert all([~p.exists() for p in dst_paths])
    eoc.apply_and_write(convert_to_chunk_data, dst_paths=dst_paths)
    assert all([p.exists() for p in dst_paths])

# EOCubeImageCollection
