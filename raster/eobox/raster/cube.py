import pandas as pd
from pathlib import Path
import rasterio
from tqdm import tqdm

from eobox.raster import MultiRasterIO
from eobox.raster.gdalutils import buildvrt

class EOCube():
    def __init__(self, df_layers, chunksize=2**5, wdir=None):
        self._df_layers = df_layers
        self._chunksize = chunksize
        # In this whole class we could directly get the windows from the
        # eobox.raster.windows_from_blocksize(blocksize_xy, width, height)
        # then we do not need the more expensive initialization of MultiRasterIO
        self._mrio = None
        self._initialize_mrio_if_attr_is_none()
        if wdir:
            self._wdir = Path(wdir)
        else:
            self._wdir = wdir

    @property
    def df_layers(self):
        return self._df_layers

    @property
    def chunksize(self):
        return self._chunksize

    @chunksize.setter
    def chunksize(self, value):
        self._mrio = self._mrio.windows_from_blocksize(value)
        self._chunksize = value

    @property
    def n_chunks(self):
        return len(self._mrio.windows)

    @property
    def wdir(self):
        return self._wdir

    @wdir.setter
    def wdir(self, value):
        self._wdir = value

    def _initialize_mrio_if_attr_is_none(self):
        if not self._mrio:
            self._mrio = MultiRasterIO(layer_files=self.df_layers["path"].values)
            self._mrio = self._mrio.windows_from_blocksize(self.chunksize)

    def get_chunk(self, ji):
        """Get a EOCubeChunk"""
        return EOCubeChunk.from_eocube(self, ji)

    def apply_and_write(self, fun, dst_paths, **kwargs):
        ji_process = []
        for ji in range(self.n_chunks):
            dst_paths_ji_exist = [self.get_chunk_path_from_layer_path(
                dst, ji, mkdir=False).exists() for dst in dst_paths]
            if not all(dst_paths_ji_exist):
                ji_process.append(ji)
        if len(ji_process) != self.n_chunks:
            print(f"{self.n_chunks - len(ji_process)} chunks already processed and skipped.")
        results = []
        for ji in tqdm(ji_process, total=len(ji_process)):
            eoc_chunk = self.get_chunk(ji)
            results.append(fun(eoc_chunk, dst_paths, **kwargs))
        for pth in dst_paths:
            if not Path(pth).exists():
                self.create_vrt_from_chunks(pth)

    def get_chunk_path_from_layer_path(self, path_layer, ji, mkdir=True):
        path_layer = Path(path_layer)
        bdir_chunks = path_layer.parent / f"xchunks_cs{self.chunksize}" / path_layer.stem
        if mkdir:
            bdir_chunks.mkdir(exist_ok=True, parents=True)
        ndigits = len(str((self.n_chunks)))
        dst_path_chunk = bdir_chunks / (path_layer.stem + f"_ji-{ji:0{ndigits}d}.tif")
        return dst_path_chunk

    def create_vrt_from_chunks(self, dst_path):
        chunk_paths = []
        for ji in range(self.n_chunks):
            pth = self.get_chunk_path_from_layer_path(dst_path, ji,
                                                      mkdir=True).absolute()
            if not pth.exists():
                raise FileNotFoundError(pth)
            else:
                chunk_paths.append(str(pth))
        buildvrt(chunk_paths, str(dst_path), relative=True)


class EOCubeChunk(EOCube):
    def __init__(self, ji, df_layers, chunksize=2**5, wdir=None):
        super().__init__(df_layers=df_layers, chunksize=chunksize, wdir=wdir)
        self._ji = ji
        self._data = None # set with self.read_data()
        self._data_structure = None # can be ndarray, dataframe
        self._window = self._mrio.windows[self.ji]
        self._width = self._window.width
        self._height = self._window.height
        self._n_layers = self.df_layers.shape[0]
        self.result = None

    @property
    def ji(self):
        return self._ji

    @property
    def data(self):
        return self._data

    @property
    def chunksize(self):
        return self._chunksize

    @chunksize.setter
    def chunksize(self, value):
        raise NotImplementedError("It is not allowed to set the EOCubeChunk chunksize (but of a EOCube object).")

    def read_data(self):
        self._data = self._mrio.get_arrays(self.ji)
        self._update_data_structure()
        return self

    def _update_data_structure(self):
        self._data_structure = self.data.__class__.__name__

    def convert_data_to_dataframe(self):
        if self._data_structure != "ndarray":
            raise Exception(f"Data is not an ndarray but {self._data_structure}.")
        if "uname" not in self.df_layers.columns:
            raise Exception("You need a column named uname with unique names for all layers.")
        if not self.df_layers["uname"].is_unique:
            raise Exception("You need a column named uname with unique names for all layers.")

        self._data = pd.DataFrame(self._reshape_3to2d(self._data))
        self._update_data_structure()
        self._data.columns = self.df_layers["uname"]
        return self

    def convert_data_to_ndarray(self):
        """Converts the data from dataframe to ndarray format. Assumption: df-columns are ndarray-layers (3rd dim.)"""
        if self._data_structure != "DataFrame":
            raise Exception(f"Data is not a DataFrame but {self._data_structure}.")
        self._data = self._convert_to_ndarray(self._data)
        self._update_data_structure()
        return self

    def _convert_to_ndarray(self, data):
        """Converts data from dataframe to ndarray format. Assumption: df-columns are ndarray-layers (3rd dim.)"""
        if data.__class__.__name__ != "DataFrame":
            raise Exception(f"data is not a DataFrame but {data.__class__.__name__}.")
        shape_ndarray = (self._height, self._width, data.shape[1])
        data_ndarray = data.values.reshape(shape_ndarray)
        return data_ndarray

    def write_dataframe(self, result, dst_paths, nodata=None, compress='lzw'):
        """Write results (dataframe) to disc."""
        result = self._convert_to_ndarray(result)
        self.write_ndarray(result, dst_paths, nodata=nodata, compress=compress)

    def write_ndarray(self, result, dst_paths, nodata=None, compress='lzw'):
        """Write results (ndarray) to disc."""

        assert len(dst_paths) == result.shape[2]
        assert result.shape[0] == self._height
        assert result.shape[1] == self._width
        assert result.shape[2] == len(dst_paths)
        with rasterio.open(self._mrio._get_template_for_given_resolution(self._mrio.dst_res, "path")) as src_layer:
            pass # later we need src_layer for src_layer.window_transform(win)
        for i, pth in enumerate(dst_paths):
            dst_path_chunk = self.get_chunk_path_from_layer_path(pth, self.ji)

            result_layer_i = result[:, :, [i]]
            assert result_layer_i.shape[2] == 1
            kwargs = self._mrio._get_template_for_given_resolution(
                res=self._mrio.dst_res, return_="meta").copy()
            kwargs.update({"driver": "GTiff",
                           "compress": compress,
                           "nodata": nodata,
                           "height": self._height,
                           "width": self._width,
                           "dtype": result_layer_i.dtype,
                           "transform": src_layer.window_transform(self._window)})
            with rasterio.open(dst_path_chunk, "w", **kwargs) as dst:
                dst.write(result_layer_i[:, :, 0], 1)

    @staticmethod
    def _reshape_3to2d(arr_3d):
        new_shape = (arr_3d.shape[0] * arr_3d.shape[1], arr_3d.shape[2],)
        return arr_3d.reshape(new_shape)

    @staticmethod
    def from_eocube(eocube, ji):
        """Create a EOCubeChunk object from an EOCube object."""
        eocubewin = EOCubeChunk(ji, eocube.df_layers, eocube.chunksize, eocube.wdir)
        return eocubewin
