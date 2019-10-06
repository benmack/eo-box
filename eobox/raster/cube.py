import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
import time
from tqdm import tqdm

from .rasterprocessing import MultiRasterIO
from .gdalutils import buildvrt

from .utils import dtype_checker_df
from .utils import cleanup_df_values_for_given_dtype


class EOCubeAbstract():
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

    def get_chunk_path_from_layer_path(self, path_layer, ji, mkdir=True):
        path_layer = Path(path_layer)
        bdir_chunks = path_layer.parent / f"xchunks_cs{self.chunksize}" / path_layer.stem
        if mkdir:
            bdir_chunks.mkdir(exist_ok=True, parents=True)
        ndigits = len(str((self.n_chunks)))
        dst_path_chunk = bdir_chunks / (path_layer.stem + f"_ji-{ji:0{ndigits}d}.tif")
        return dst_path_chunk

    def get_df_ilocs(self, band, date):
        """Get positions of rows matching specific band(s) and date(s).

        The method supports three typical queries:

        * one band and one date (both given as strings)

        * one band and of several dates (band given as strings, date as list of strings)

        * several band and of one date (date given as strings, band as list of strings)

        Parameters
        ----------
        band : str or list
            Band(s) for which to derive the iloc index.
        date : str or list
            Date(s) for which to derive the iloc index.
        
        Returns
        -------
        int or list 
            Integer (if band and date are str) or list of iloc indices.
        """

        df = self.df_layers.copy()
        df["index"] = range(df.shape[0])

        idx_layers = []
        if isinstance(band, str) and isinstance(date, str):
            idx_layers = df[(df["date"] == date) &  (df["band"] == band)]["index"].values[0]
        if isinstance(band, list) and isinstance(date, str):
            for b in band:
                idx = df[(df["date"] == date) &  (df["band"] == b)]["index"].values[0]
                idx_layers.append(idx)
        elif isinstance(band, str) and isinstance(date, list):
            for d in date:
                idx = df[(df["band"] == band) &  (df["date"] == d)]["index"].values[0]
                idx_layers.append(idx)
        return idx_layers

class EOCube(EOCubeAbstract):
    def __init__(self, df_layers, chunksize=2**5, wdir=None):
        super().__init__(df_layers=df_layers, chunksize=chunksize, wdir=wdir)

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


class EOCubeChunk(EOCubeAbstract):
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
        self._spatial_bounds = self._get_spatial_bounds()
        
    @property
    def ji(self):
        return self._ji

    @property
    def data(self):
        return self._data

    @property
    def chunksize(self):
        return self._chunksize

    @property
    def spatial_bounds(self):
        return self._spatial_bounds

    @chunksize.setter
    def chunksize(self, value):
        raise NotImplementedError("It is not allowed to set the EOCubeChunk chunksize (but of a EOCube object).")

    def _get_spatial_bounds(self):
        """Get the spatial bounds of the chunk.""" 
        # This should be a MultiRasterIO method
        with rasterio.open(self._mrio._get_template_for_given_resolution(self._mrio.dst_res, "path")) as src_layer:
            pass # later we need src_layer for src_layer.window_transform(win)
        win_transform = src_layer.window_transform(self._window)
        bounds = rasterio.windows.bounds(window=self._window,
                                         transform=win_transform,
                                         height=0, width=0)
        return bounds

    def read_data(self):
        self._data = self._mrio.get_arrays(self.ji)
        if self.data.shape[0] * self.data.shape[1] != self._width * self._height:
            raise Exception(f"X/Y dimension size of extracted window (={'/'.join(self.data.shape)}) different from expected shape (={self._width}/{self._height}).")
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
    def robust_data_range(arr, robust=False, vmin=None, vmax=None):
        """Get a robust data range, i.e. 2nd and 98th percentile for vmin, vmax parameters."""
        # from the seaborn code 
        # https://github.com/mwaskom/seaborn/blob/3a3ec75befab52c02650c62772a90f8c23046038/seaborn/matrix.py#L201

        def _get_vmin_vmax(arr2d, vmin=None, vmax=None):
            if vmin is None:
                vmin = np.percentile(arr2d, 2) if robust else arr2d.min()
            if vmax is None:
                vmax = np.percentile(arr2d, 98) if robust else arr2d.max()
            return vmin, vmax

        if len(arr.shape) == 3 and vmin is None and vmax is None:
            vmin = []
            vmax = []
            for i in range(arr.shape[2]):
                arr_i = arr[:, :, i]
                vmin_i, vmax_i = _get_vmin_vmax(arr_i, vmin=None, vmax=None)
                vmin.append(vmin_i)
                vmax.append(vmax_i)
        else:
            vmin, vmax = _get_vmin_vmax(arr, vmin=vmin, vmax=vmax)
        return vmin, vmax

    def plot_raster(self,
                    idx_layer,
                    robust=False, vmin=None, vmax=None,
                    spatial_bounds=False,
                    figsize=None, ax=None):

        import matplotlib.pyplot as plt

        if self.data is None:
            self.read_data()
        arr = self.data[:, :, idx_layer]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        vmin, vmax = self.robust_data_range(arr,
                                            robust=robust,
                                            vmin=vmin, vmax=vmax)
        ax = ax.imshow(arr,
                       extent=self.spatial_bounds if spatial_bounds else None,
                       vmin=vmin, vmax=vmax)
        return ax

    def plot_raster_rgb(self,
                        idx_layers,
                        robust=False, vmin=None, vmax=None,
                        spatial_bounds=False,
                        figsize=None, ax=None):

        import matplotlib.pyplot as plt

        def _select_vmxx(vmxx, i):
            """If vmxx is a list, return i-th value of the list, else return vmxx."""
            if isinstance(vmxx, list):
                return vmxx[i]
            else:
                return vmxx

        if self.data is None:
            self.read_data()
        arr = self.data[:, :, idx_layers].astype(float)

        for i in range(arr.shape[2]):
            arr_i = arr[:, :, i]

            vmin_i, vmax_i = self.robust_data_range(arr_i, 
                                                    robust=robust, 
                                                    vmin=_select_vmxx(vmin, i),
                                                    vmax=_select_vmxx(vmax, i))
            arr_i[arr_i < vmin_i] = vmin_i
            arr_i[arr_i > vmax_i] = vmax_i
            arr[:, :, i] = (arr_i - vmin_i) / (vmax_i - vmin_i)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax = ax.imshow(arr, 
                       extent=self.spatial_bounds if spatial_bounds else None)

        return ax

    

    @staticmethod
    def _reshape_3to2d(arr_3d):
        new_shape = (arr_3d.shape[0] * arr_3d.shape[1], arr_3d.shape[2],)
        return arr_3d.reshape(new_shape)

    @staticmethod
    def from_eocube(eocube, ji):
        """Create a EOCubeChunk object from an EOCube object."""
        eocubewin = EOCubeChunk(ji, eocube.df_layers, eocube.chunksize, eocube.wdir)
        return eocubewin


class EOCubeSceneCollectionAbstract(EOCubeAbstract):
    def __init__(self,
                 df_layers,
                 chunksize,
                 variables,
                 qa,
                 qa_valid,
                 # timeless=None,
                 wdir=None):
        """Handling scene collections, i.e. a set of scenes each with the same layers.

        The class enables to perform chunkwise processing over a set of scenes having
        the same variables / bands.
        Therefore the ``df_layers`` dataframe requires information to be stored in the following columns:

        * *sceneid* (unique identifier of the scene),
        
        * *date* (the aquisition date of the scene as datetime type),
        
        * *band* (the layers / bands that exist for all scenes),
        
        * *uname* (unique identifier for all layers, i.e. scene + variable/qu-layer),
        
        * *path* (the path where the raster for that layer is located).

        Parameters
        ----------
        df_layers : dataframe
            A dataframe, see description above. 
        chunksize : int
            Size of the spatial window used as processing unit.
        variables : list of str
            Those values in ``df_layers['band']`` that are treated as variables.
        qa : str
            The value in ``df_layers['band']`` which is treated as quality assessment layer.
        qa_valid : list of int
            The values in the qualitiy assessment layer that identify pixels
            to be considered as valid in the variable rasters., by default None
        wdir : str, optional
            Working directory
            
        Raises
        ------
        ValueError
            [description]
        """
        # validation and formatting
        n_per_sceneid = len(variables) + 1 # i.e. qa layer
        scenes_complete = df_layers.groupby("sceneid").apply(
            lambda x: x["band"].isin(variables + [qa]).sum() == n_per_sceneid)
        if not all(scenes_complete):
            scenes_incomplete = scenes_complete.index[~scenes_complete].values
            raise ValueError(f"Variable or qa layers missing in the following scenes: {scenes_incomplete}")
        df_layers = df_layers.sort_values(["date", "sceneid", "band"])

        EOCubeAbstract.__init__(self, df_layers=df_layers, chunksize=chunksize, wdir=wdir)
        self._variables = variables
        self._qa = qa
        self._qa_valid = qa_valid
        # self._timeless = timeless

    @property
    def variables(self):
        return self._variables

    @property
    def qa(self):
        return self._qa

    @property
    def qa_valid(self):
        return self._qa_valid

"""     @property
    def timeless(self):
        return self._timeless
 """

class EOCubeSceneCollection(EOCubeSceneCollectionAbstract, EOCube):

    def get_chunk(self, ji):
        """Get a EOCubeSceneCollectionChunk"""
        return EOCubeSceneCollectionChunk(ji=ji,
                                          df_layers=self.df_layers,
                                          chunksize=self.chunksize,
                                          variables=self.variables,
                                          qa=self.qa,
                                          qa_valid=self.qa_valid,
                                          wdir=self.wdir)

    def apply_and_write_by_variable(self,
                                    fun,
                                    dst_paths,
                                    dtypes,
                                    compress,
                                    nodata,
                                    **kwargs):

        def check_variable_specific_args(arg, variables):
            if isinstance(arg, dict):
                if not set(arg.keys()) == set(self.variables):
                    raise ValueError("'dtypes'-, 'nodata'-dicts keys must match (self.)variables.")
                else:
                    return arg # TODO: better validate the args before
            else:
                # TODO: validate the args before
                return {var: arg for var in variables}

        dtypes = check_variable_specific_args(dtypes, self.variables)
        nodata = check_variable_specific_args(nodata, self.variables)
        compress = check_variable_specific_args(compress, self.variables)

        ji_process = {}
        for var in self.variables:
            ji_process[var] = []
            counter = 0
            for ji in range(self.n_chunks):
                dst_paths_ji_exist = [self.get_chunk_path_from_layer_path(
                    dst, ji, mkdir=False).exists() for dst in dst_paths[var]]
                if all(dst_paths_ji_exist):
                    counter += 1
                else:
                    ji_process[var].append(ji)
            if len(ji_process[var]) != self.n_chunks:
                print(f"{var}: {counter} / {self.n_chunks} chunks already processed and skipped.")

        # a sorted list of all chunks that we need to run somewhere
        ji_process_over_all_variables = []
        for key, process in ji_process.items():
            ji_process_over_all_variables += process
        ji_process_over_all_variables = np.unique(ji_process_over_all_variables)

        for ji in tqdm(ji_process_over_all_variables, total=len(ji_process)):
            eoc_chunk = self.get_chunk(ji)
            eoc_chunk = eoc_chunk.read_data_by_variable(mask=True)
            results = {}
            for var in self.variables:
                if ji in ji_process[var]:
                    results[var] = fun(eoc_chunk.data[var], **kwargs)
                    results[var] = cleanup_df_values_for_given_dtype(results[var],
                                                                     dtype=dtypes[var],
                                                                     lower_as=None,
                                                                     higher_as=None,
                                                                     nan_as=None)
                    eoc_chunk.write_dataframe(results[var],
                                              dst_paths[var],
                                              nodata[var],
                                              compress[var])
        for var in self.variables:
            for pth in dst_paths[var]:
                if not Path(pth).exists():
                    self.create_vrt_from_chunks(pth)

    def create_virtual_time_series(self,
                                   idx_virtual,
                                   dst_pattern, #"./xxx_uncontrolled/ls2008_vts4w/ls2008_vts4w_{date}_{var}.vrt"
                                   dtypes,
                                   compress="lzw",
                                   nodata=None,
                                   num_workers=1):
        dst_paths = {}
        for var in self.variables:
            dst_paths[var] = []
            for date in idx_virtual:
                dst_paths[var].append(dst_pattern.format(**{"var": var, "date": date.strftime("%Y-%m-%d")}))
        assert (len(idx_virtual) * len(self.variables)) == sum([len(dst_paths[var]) for var in self.variables])

        self.apply_and_write_by_variable(fun=create_virtual_time_series,
                                         dst_paths=dst_paths,
                                         dtypes=dtypes,
                                         compress=compress,
                                         nodata=nodata,
                                         idx_virtual=idx_virtual,
                                         num_workers=num_workers,
                                         verbosity=0)

    def create_statistical_metrics(self,
                                   percentiles,
                                   iqr,
                                   diffs,
                                   dst_pattern, #"./xxx_uncontrolled/ls2008_vts4w/ls2008_vts4w_{date}_{var}.vrt"
                                   dtypes,
                                   compress="lzw",
                                   nodata=None,
                                   num_workers=1):

        metrics = ['mean', 'std', 'min']
        metrics += [f'p{int(p*100):02d}' for p in percentiles]
        metrics += ['max']
        if iqr:
            metrics += ['p75-p25']
        if diffs:
            metrics += ['min-max']
            if 0.05 in percentiles and 0.95 in percentiles:
                metrics += ['p95-p05']
            if 0.10 in percentiles and 0.90 in percentiles:
                metrics += ['p90-p10']

        dst_paths = {}
        for var in self.variables:
            dst_paths[var] = []
            for metric in metrics:
                dst_paths[var].append(dst_pattern.format(**{"var": var, "metric": metric}))
        assert (len(metrics) * len(self.variables)) == sum([len(dst_paths[var]) for var in self.variables])

        self.apply_and_write_by_variable(fun=create_statistical_metrics,
                                         dst_paths=dst_paths,
                                         dtypes=dtypes,
                                         compress=compress,
                                         nodata=nodata,
                                         percentiles=percentiles,
                                         iqr=iqr,
                                         diffs=diffs,
                                         num_workers=num_workers)


class EOCubeSceneCollectionChunk(EOCubeSceneCollectionAbstract, EOCubeChunk):
    def __init__(self,
                 ji,
                 df_layers,
                 chunksize,
                 variables,
                 qa,
                 qa_valid,
                 # timeless=None,
                 wdir=None):
        EOCubeSceneCollectionAbstract.__init__(self,
                                               df_layers=df_layers,
                                               chunksize=chunksize,
                                               variables=variables,
                                               qa=qa,
                                               qa_valid=qa_valid,
                                               wdir=wdir)
        EOCubeChunk.__init__(self,
                             ji=ji,
                             df_layers=df_layers,
                             chunksize=chunksize,
                             wdir=wdir)

    def read_data_by_variable(self, mask=True):
        """Reads and masks (if desired) the data and converts it in one dataframe per variable."""
        def print_elapsed_time(start, last_stopped, prefix):
            # print(f"{prefix} - Elapsed time [s] since start / last stopped: \
            #     {(int(time.time() - start_time))} / {(int(time.time() - last_stopped))}")
            return time.time()
        start_time = time.time()
        last_stopped = time.time()
        last_stopped = print_elapsed_time(start_time, last_stopped, "Starting chunk function")

        verbose = False

        self.read_data()
        last_stopped = print_elapsed_time(start_time, last_stopped, "Data read")

        # 2.
        sc_chunk = self.convert_data_to_dataframe()
        last_stopped = print_elapsed_time(start_time, last_stopped, "Data converted to df")


        # 3.B.
        if mask:
            # 3.A.
            ilocs_qa = np.where((self.df_layers["band"] == self.qa).values)[0]
            df_qa = self.data.iloc[:, ilocs_qa]
            df_qa.columns = self.df_layers["date"].iloc[ilocs_qa]
            df_clearsky = df_qa.isin(self.qa_valid)
            last_stopped = print_elapsed_time(start_time, last_stopped, "Clearsky df created")

            return_bands = self.variables
        else:
            return_bands = self.variables + [self.qa]

        dfs_variables = {}
        for var in return_bands:
            if verbose:
                print("VARIABLE:", var)
            ilocs_var = np.where((self.df_layers["band"] == var).values)[0]
            df_var = self.data.iloc[:, ilocs_var]
            df_var.columns = self.df_layers["date"].iloc[ilocs_var]
            if mask:
                df_var = df_var.where(df_clearsky, other=np.nan)
            dfs_variables[var] = df_var
        last_stopped = print_elapsed_time(start_time, last_stopped, "Clearsky df created")
        self._data = dfs_variables
        return self


def create_virtual_time_series(df_var, idx_virtual, num_workers=1, verbosity=0):
    """Create a virtual time series from a dataframe with pd.DateTimeIndex and a instance (e.g. pixels) dimension. 
    
    Parameters
    ----------
    df_var : [type]
        [description]
    idx_virtual : [type]
        [description]
    num_workers : int, optional
        [description], by default 1
    verbosity : int, optional
        [description], by default 0
    
    Returns
    -------
    [type]
        [description]
    """
    def _create_virtual_time_series_core(df, idx_virtual, verbosity=0):

        if verbosity:
            print("Shape of input dataframe:", df.shape)

        transpose_before_return = False
        if isinstance(df.columns, pd.DatetimeIndex):
            transpose_before_return = True
            df = df.transpose()

        # define the virtual points in the time series index
        # idx_virtual = df.resample(rule).asfreq().index
        if not df.index.is_unique:
            if verbosity:
                print(f"Aggregating (max) data with > observations per day: {df.index[df.index.duplicated()]}")
            df = df.groupby(level=0).max()
            assert df.index.is_unique

        if verbosity:
            print("Length of virtual time series:", len(idx_virtual))
        # add the existing time series points to the virtual time series index
        idx_virtual_and_data = idx_virtual.append(df.index).unique()
        idx_virtual_and_data = idx_virtual_and_data.sort_values()
        if verbosity:
            print("Length of virtual and data time series:", len(idx_virtual_and_data))
        # extend the time series data such that it contains all existing and virtual time series points
        df = df.reindex(index=idx_virtual_and_data)
        # interpolate between dates and forward/backward fill edges with closest values
        df = df.interpolate(method='time')
        df = df.bfill()
        df = df.ffill()
        df = df.loc[idx_virtual]

        if transpose_before_return:
            df = df.transpose()
        if verbosity:
            print("Shape of output dataframe:", df.shape)
        return df

    if (num_workers > 1) or (num_workers == -1):
        import dask.dataframe as dd
        df_var = dd.from_pandas(df_var, npartitions=num_workers)
        df_result = df_var.map_partitions(_create_virtual_time_series_core,
                                          idx_virtual=idx_virtual,
                                          verbosity=verbosity)
        df_result = df_result.compute(scheduler='processes', num_workers=num_workers)
    else:
        df_result = _create_virtual_time_series_core(df_var,
                                                     idx_virtual,
                                                     verbosity=verbosity)
    return df_result

def create_statistical_metrics(df_var, percentiles=None, iqr=True, diffs=True, num_workers=1, verbosity=0):
    """Calculate statistial metrics from a dataframe with pd.DateTimeIndex and a instance (e.g. pixels) dimension. 
    
    Parameters
    ----------
    df_var : [type]
        [description]
    percentiles : [type]
        [description]
    iqr : [type]
        [description]
    num_workers : int, optional
        [description], by default 1
    verbosity : int, optional
        [description], by default 0
    
    Returns
    -------
    [type]
        [description]
    """

    def _calc_statistical_metrics(df, percentiles=None, iqr=True, diffs=True):
        """Calculate statistical metrics and the count of valid observations."""
        metrics_df = df.transpose().describe(percentiles=percentiles).transpose()
        if iqr and all(np.isin(["25%", "75%"], metrics_df.columns)):
            metrics_df["p75-p25"] = metrics_df["75%"] - metrics_df["25%"]
        # other differences         
        if diffs and all(np.isin(["min", "max"], metrics_df.columns)):
            metrics_df["max-min"] = metrics_df["max"] - metrics_df["min"]
        if diffs and all(np.isin(["5%", "95%"], metrics_df.columns)):
            metrics_df["p95-p05"] = metrics_df["95%"] - metrics_df["5%"]
        if diffs and all(np.isin(["10%", "90%"], metrics_df.columns)):
            metrics_df["p90-p10"] = metrics_df["90%"] - metrics_df["10%"]
        metrics_df = metrics_df.drop(labels=["count"], axis=1)
        return metrics_df

    if percentiles is None:
        percentiles = [.1, .25, .50, .75, .9]

    if (num_workers > 1) or (num_workers == -1):
        import dask.dataframe as dd
        df_var = dd.from_pandas(df_var, npartitions=num_workers)
        df_result = df_var.map_partitions(_calc_statistical_metrics,
                                          percentiles=percentiles,
                                          iqr=iqr,
                                          diffs=diffs)
        df_result = df_result.compute(scheduler='processes', num_workers=num_workers)
    else:
        df_result = _calc_statistical_metrics(df_var,
                                              percentiles,
                                              iqr,
                                              diffs=diffs)
    return df_result

