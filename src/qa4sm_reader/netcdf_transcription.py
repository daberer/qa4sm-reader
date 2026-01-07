import xarray as xr
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import os
import calendar
import time
import shutil
import tempfile
import sys
from pathlib import Path
import subprocess, zarr, json
import psutil
from qa4sm_reader.intra_annual_temp_windows import TemporalSubWindowsCreator, InvalidTemporalSubWindowError
from qa4sm_reader.globals import    METRICS, TC_METRICS, STABILITY_METRICS, NON_METRICS, METADATA_TEMPLATE, \
                                    IMPLEMENTED_COMPRESSIONS, ALLOWED_COMPRESSION_LEVELS, \
                                    INTRA_ANNUAL_METRIC_TEMPLATE, INTRA_ANNUAL_TCOL_METRIC_TEMPLATE, \
                                    TEMPORAL_SUB_WINDOW_SEPARATOR, DEFAULT_TSW, TEMPORAL_SUB_WINDOW_NC_COORD_NAME, \
                                    MAX_NUM_DS_PER_VAL_RUN, DATASETS, OLD_NCFILE_SUFFIX, status_replace

class TemporalSubWindowMismatchError(Exception):
    '''Exception raised when the temporal sub-windows provided do not match the ones present in the provided netCDF file.'''

    def __init__(self, provided, expected):
        super().__init__(
            f'The temporal sub-windows provided ({provided}) do not match the ones present in the provided netCDF file ({expected}).'
        )


class Pytesmo2Qa4smResultsTranscriber:
    """
    Transcribes (=converts) the pytesmo results netCDF4 file format to a more user friendly format, that
    is used by QA4SM.

    Parameters
    ----------
    pytesmo_results : str
        Path to results netCDF4 written by `qa4sm.validation.validation.check_and_store_results`, which is in the old `pytesmo` format.
    intra_annual_slices : Union[None, TemporalSubWindowsCreator]
        The temporal sub-windows for the results. Default is None, which means that no temporal sub-windows are
        used, but only the 'bulk'. If an instance of `valdiator.validation.TemporalSubWindowsCreator` is provided,
        the temporal sub-windows are used as provided by the TemporalSubWindowsCreator instance.
    keep_pytesmo_ncfile : Optional[bool]
        Whether to keep the original pytesmo results netCDF file. Default is False. \
            If True, the original file is kept and indicated by the suffix `qa4sm_reader.globals.OLD_NCFILE_SUFFIX`.
    """

    def __init__(self,
                 pytesmo_results: str,
                 intra_annual_slices: Union[None,
                                            TemporalSubWindowsCreator] = None,
                 keep_pytesmo_ncfile: Optional[bool] = False):

        self.original_pytesmo_ncfile = str(pytesmo_results)

        # windows workaround
        # windows keeps a file lock on the original file, which prevents it from being renamed or deleted
        # to circumvent this, the file is copied to a temporary directory and the copy is used instead

        if sys.platform.startswith("win"):
            if not isinstance(pytesmo_results, Path):
                pytesmo_results = Path(pytesmo_results)

            _tmp_dir = Path(tempfile.mkdtemp())
            tmp_dir = _tmp_dir / pytesmo_results.parent.name

            if not tmp_dir.exists():
                tmp_dir.mkdir()

            new_pytesmo_results = tmp_dir / pytesmo_results.name
            shutil.copy(pytesmo_results, new_pytesmo_results)
            pytesmo_results = str(new_pytesmo_results)

        self.pytesmo_ncfile = f'{pytesmo_results}'
        if not Path(pytesmo_results).is_file():
            self.exists = False
            raise FileNotFoundError(
                f'\n\nFile {pytesmo_results} not found. Please provide a valid path to a pytesmo results netCDF file.'
            )
            return None
        else:
            self.exists = True

        # make sure the intra-annual slices from the argument are the same as the ones present in the pytesmo results
        pytesmo_results_tsws = Pytesmo2Qa4smResultsTranscriber.get_tsws_from_ncfile(
            pytesmo_results)
        if isinstance(intra_annual_slices, TemporalSubWindowsCreator):
            self.provided_tsws = intra_annual_slices.names
        elif intra_annual_slices is None:
            self.provided_tsws = intra_annual_slices
        else:
            raise InvalidTemporalSubWindowError(intra_annual_slices,
                                                ['months', 'seasons'])

        if self.provided_tsws != pytesmo_results_tsws:
            print(
                f'The temporal sub-windows provided ({self.provided_tsws}) do not match the ones present in the provided netCDF file ({pytesmo_results_tsws}).'
            )
            raise TemporalSubWindowMismatchError(self.provided_tsws,
                                                 pytesmo_results_tsws)

        self.intra_annual_slices: Union[
            None, TemporalSubWindowsCreator] = intra_annual_slices
        self._temporal_sub_windows: Union[
            None, TemporalSubWindowsCreator] = intra_annual_slices

        self._default_non_metrics: List[str] = NON_METRICS

        self.METADATA_TEMPLATE: Dict[str, Union[None, Dict[str, Union[
            np.ndarray, np.float32, np.array]]]] = METADATA_TEMPLATE

        self.temporal_sub_windows_checker_called: bool = False
        self.only_default_case: bool = True

        with xr.open_dataset(pytesmo_results) as pr:
            self.pytesmo_results: xr.Dataset = pr

        self.keep_pytesmo_ncfile = keep_pytesmo_ncfile

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(pytesmo_results="{self.pytesmo_ncfile}", intra_annual_slices={self.intra_annual_slices.__repr__()})'

    def __str__(self) -> str:
        return f'{self.__class__.__name__}("{Path(self.pytesmo_ncfile).name}", {self.intra_annual_slices})'

    def temporal_sub_windows_checker(
            self) -> Tuple[bool, Union[List[str], None]]:
        """
        Checks the temporal sub-windows and returns which case of temporal sub-window is used, as well as a list of the
        temporal sub-windows. Keeps track of whether the method has been called before.

        Returns
        -------
        Tuple[bool, Union[List[str], None]]
            A tuple indicating the temporal sub-window type and the list of temporal sub-windows.
            bulk case: (True, [`globals.DEFAULT_TSW`]),
            intra-annual windows case: (False, list of temporal sub-windows)
        """

        self.temporal_sub_windows_checker_called = True
        if self.intra_annual_slices is None:
            return True, [DEFAULT_TSW]
        elif isinstance(self.intra_annual_slices, TemporalSubWindowsCreator):
            return False, self.provided_tsws
        else:
            raise InvalidTemporalSubWindowError(self.intra_annual_slices)

    @property
    def non_metrics_list(self) -> List[str]:
        """
            Get the non-metrics from the pytesmo results.

            Returns
            -------
            List[str]
                A list of non-metric names.

            Raises
            ------
            None
            """

        non_metrics_lst = []
        for non_metric in self._default_non_metrics:
            if non_metric in self.pytesmo_results:
                non_metrics_lst.append(non_metric)
            # else:
            # print(
            #     f'Non-metric \'{non_metric}\' not contained in pytesmo results. Skipping...'
            # )
            # continue
        return non_metrics_lst

    def is_valid_metric_name(self, metric_name):
        """
        Checks if a given metric name is valid, based on the defined `globals.INTRA_ANNUAL_METRIC_TEMPLATE`.

        Parameters:
        metric_name (str): The metric name to be checked.

        Returns:
        bool: True if the metric name is valid, False otherwise.
        """
        valid_prefixes = [
            "".join(
                template.format(tsw=tsw, metric=metric)
                for template in INTRA_ANNUAL_METRIC_TEMPLATE)
            for tsw in self.provided_tsws for metric in METRICS
        ]
        return any(metric_name.startswith(prefix) for prefix in valid_prefixes)

    def is_valid_tcol_metric_name(self, tcol_metric_name):
        """
        Checks if a given metric name is a valid TCOL metric name, based on the defined `globals.INTRA_ANNUAL_TCOL_METRIC_TEMPLATE`.

        Parameters:
        tcol_metric_name (str): The metric name to be checked.

        Returns:
        bool: True if the metric name is valid, False otherwise.
        """
        valid_prefixes = [
            "".join(
                template.format(
                    tsw=tsw, metric=metric, number=number, dataset=dataset)
                for template in INTRA_ANNUAL_TCOL_METRIC_TEMPLATE)
            for tsw in self.provided_tsws for metric in TC_METRICS
            for number in range(MAX_NUM_DS_PER_VAL_RUN) for dataset in DATASETS
        ]
        return any(
            tcol_metric_name.startswith(prefix) for prefix in valid_prefixes)

    def is_valid_stability_metric_name(self, metric_name):
        """
        Checks if a given stability metric name is valid, based on the defined `globals.INTRA_ANNUAL_METRIC_TEMPLATE`.

        Parameters:
        metric_name (str): The stability metric name to be checked.

        Returns:
        bool: True if the stability metric name is valid, False otherwise.
        """
        valid_prefixes = [
            "".join(
                template.format(tsw=tsw, metric=metric)
                for template in INTRA_ANNUAL_METRIC_TEMPLATE)
            for tsw in self.provided_tsws for metric in STABILITY_METRICS
        ]
        return any(metric_name.startswith(prefix) for prefix in valid_prefixes)

    @property
    def metrics_list(self) -> List[str]:
        """Get the metrics dictionary. Whole procedure based on the premise, that metric names of valdiations of intra-annual
        temporal sub-windows are of the form: `metric_long_name = 'intra_annual_window{validator.validation.globals.TEMPORAL_SUB_WINDOW_SEPARATOR}metric_short_name'`. If this is not the
        case, it is assumed the 'bulk' case is present and the metric names are assumed to be the same as the metric
        short names.

        Returns
        -------
        Dict[str, str]
            The metrics dictionary.
        """

        # check if the metric names are of the form: `metric_long_name = 'intra_annual_window{TEMPORAL_SUB_WINDOW_SEPARATOR}metric_short_name'` and if not, assume the 'bulk' case

        _metrics = [
            metric for metric in self.pytesmo_results
            if self.is_valid_metric_name(metric)
            or self.is_valid_tcol_metric_name(metric)
            or self.is_valid_stability_metric_name(metric)
        ]

        if len(_metrics) != 0:  # intra-annual case
            return list(set(_metrics))
        else:  # bulk case
            return [
                long for long in self.pytesmo_results
                if long not in self.non_metrics_list
            ]

    def get_pytesmo_attrs(self) -> None:
        """
        Get the attributes of the pytesmo results and add them to the transcribed dataset.
        """
        for attr in self.pytesmo_results.attrs:
            self.transcribed_dataset.attrs[attr] = self.pytesmo_results.attrs[
                attr]

    def handle_n_obs(self) -> None:
        """
        Each data variable of the flavor 'n_obs_between_*' contains the same data. Thus, only one is kept and renamned\
            to plain 'n_obs'.
        """

        _n_obs_vars = sorted(
            [var for var in self.transcribed_dataset if 'n_obs' in var])

        if _n_obs_vars[0] != 'n_obs':
            self.transcribed_dataset = self.transcribed_dataset.drop_vars(
                _n_obs_vars[1:])
            self.transcribed_dataset = self.transcribed_dataset.rename(
                {_n_obs_vars[0]: 'n_obs'})

    def drop_obs_dim(self) -> None:
        """
        Drops the 'obs' dimension from the transcribed dataset, if it exists.
        """
        if 'obs' in self.transcribed_dataset.dims:
            self.transcribed_dataset = self.transcribed_dataset.drop_dims(
                'obs')

    def mask_redundant_tsw_values(self) -> None:
        """
        For all variables starting with 'slope', replace all tsw values ('2010', '2011', etc.) with NaN 
        except for the default tsw.
        """
        slope_vars = [
            var for var in self.transcribed_dataset if var.startswith("slope")
        ]

        for var in slope_vars:
            if TEMPORAL_SUB_WINDOW_NC_COORD_NAME in self.transcribed_dataset[
                    var].dims:
                mask = self.transcribed_dataset[var][
                    TEMPORAL_SUB_WINDOW_NC_COORD_NAME] == DEFAULT_TSW
                self.transcribed_dataset[var] = self.transcribed_dataset[
                    var].where(mask, other=np.nan)

    @staticmethod
    def update_dataset_var(ds: xr.Dataset, var: str, coord_key: str,
                           coord_val: str, data_vals: List) -> xr.Dataset:
        """
        Update a variable of given coordinate in the dataset.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to be updated.
        var : str
            The variable to be updated.
        coord_key : str
            The name of the coordinate of the variable to be updated.
        coord_val : str
            The value of the coordinate of the variable to be updated.
        data_vals : List
            The data to be updated.

        Returns
        -------
        xr.Dataset
            The updated dataset.
        """

        ds[var] = ds[var].copy(
        )  # ugly, but necessary, as xr.Dataset objects are immutable
        ds[var].loc[{coord_key: coord_val}] = data_vals

        return ds

    def get_transcribed_dataset(self) -> xr.Dataset:
        """
        Get the transcribed dataset, containing all metric and non-metric data provided by the pytesmo results. 


        Returns
        -------
        xr.Dataset
            The transcribed dataset.
        """
        self.only_default_case, self.provided_tsws = self.temporal_sub_windows_checker(
        )

        self.pytesmo_results[
            TEMPORAL_SUB_WINDOW_NC_COORD_NAME] = self.provided_tsws

        metric_vars = self.metrics_list
        self.transcribed_dataset = xr.Dataset()

        for var_name in metric_vars:
            new_name = var_name
            if not self.only_default_case:
                _tsw, new_name = new_name.split(TEMPORAL_SUB_WINDOW_SEPARATOR)

            if new_name not in self.transcribed_dataset:
                # takes the data associated with the metric new_name and adds it as a new variable
                # more precisely, it assigns copies of this data to each temporal sub-window, which is the new dimension
                self.transcribed_dataset[new_name] = self.pytesmo_results[
                    var_name].expand_dims(
                        {
                            TEMPORAL_SUB_WINDOW_NC_COORD_NAME:
                            self.provided_tsws
                        },
                        axis=-1)
            else:
                # the variable already exists, but we need to update it with the real data (as opposed to the data of the first temporal sub-window, which is the default behaviour of expand_dims())
                self.transcribed_dataset = Pytesmo2Qa4smResultsTranscriber.update_dataset_var(
                    ds=self.transcribed_dataset,
                    var=new_name,
                    coord_key=TEMPORAL_SUB_WINDOW_NC_COORD_NAME,
                    coord_val=_tsw,
                    data_vals=self.pytesmo_results[var_name].data)

            # Copy attributes from the original variable to the new variable
            self.transcribed_dataset[new_name].attrs = self.pytesmo_results[
                var_name].attrs

        # Add non-metric variables directly
        self.transcribed_dataset = self.transcribed_dataset.merge(
            self.pytesmo_results[self.non_metrics_list])

        self.get_pytesmo_attrs()
        self.handle_n_obs()
        self.drop_obs_dim()
        self.mask_redundant_tsw_values()

        self.transcribed_dataset[
            TEMPORAL_SUB_WINDOW_NC_COORD_NAME].attrs = dict(
                long_name="temporal sub-window",
                standard_name="temporal sub-window",
                units="1",
                valid_range=[0, len(self.provided_tsws)],
                axis="T",
                description="temporal sub-window name for the dataset",
                temporal_sub_window_type="No temporal sub-windows used"
                if self.only_default_case is True else self.
                _temporal_sub_windows.metadata['Temporal sub-window type'],
                overlap="No temporal sub-windows used"
                if self.only_default_case is True else
                self._temporal_sub_windows.metadata['Overlap'],
                intra_annual_window_definition="No temporal sub-windows used"
                if self.only_default_case is True else
                self._temporal_sub_windows.metadata['Pretty Names [MM-DD]'],
            )

        try:
            _dict = {
                'attr_name': DEFAULT_TSW,
                'attr_value': self._temporal_sub_windows.metadata[DEFAULT_TSW]
            }
            self.transcribed_dataset[
                TEMPORAL_SUB_WINDOW_NC_COORD_NAME].attrs.update(
                    {_dict['attr_name']: _dict['attr_value']})
        except AttributeError:
            pass

        self.pytesmo_results.close()

        return self.transcribed_dataset

    def build_outname(self, root: str, keys: List[Tuple[str]]) -> str:
        """
        Build the output name for the NetCDF file. Slight alteration of the original function from pytesmo
        `pytesmo.validation_framework.results_manager.build_filename`.

        Parameters
        ----------
        root : str
            The root path, where the file is to be written to.
        keys : List[Tuple[str]]
            The keys of the pytesmo results.

        Returns
        -------
        str
            The output name for the NetCDF file.

        """

        ds_names = []
        for key in keys:
            for ds in key:
                if isinstance(ds, tuple):
                    ds_names.append(".".join(list(ds)))
                else:
                    ds_names.append(ds)

        fname = "_with_".join(ds_names)
        ext = "nc"
        if len(str(Path(root) / f"{fname}.{ext}")) > 255:
            ds_names = [str(ds[0]) for ds in key]
            fname = "_with_".join(ds_names)

            if len(str(Path(root) / f"{fname}.{ext}")) > 255:
                fname = "validation"
        self.outname = Path(root) / f"{fname}.{ext}"
        return self.outname

    def write_to_netcdf(self,
                        path: str,
                        mode: Optional[str] = 'w',
                        format: Optional[str] = 'NETCDF4',
                        engine: Optional[str] = 'netcdf4',
                        encoding: Optional[dict] = None,
                        compute: Optional[bool] = True,
                        **kwargs) -> str:
        """
        Write the transcribed dataset to a NetCDF file, based on `xarray.Dataset.to_netcdf`

        Parameters
        ----------
        path : str
            The path to write the NetCDF file
        mode : Optional[str], optional
            The mode to open the NetCDF file, by default 'w'
        format : Optional[str], optional
            The format of the NetCDF file, by default 'NETCDF4'
        engine : Optional[str], optional
            The engine to use, by default 'netcdf4'
        encoding : Optional[dict], optional
            The encoding to use, by default {"zlib": True, "complevel": 5}
        compute : Optional[bool], optional
            Whether to compute the dataset, by default True
        **kwargs : dict
            Keyword arguments passed to `xarray.Dataset.to_netcdf`.

        Returns
        -------
        str
            The path to the NetCDF file.
        """
        # Default encoding applied to all variables
        if encoding is None:
            encoding = {}
            for var in self.transcribed_dataset.variables:
                if np.issubdtype(self.transcribed_dataset[var].dtype,
                                 np.number):
                    encoding[str(var)] = {'zlib': True, 'complevel': 6}
                else:
                    encoding[str(var)] = {'zlib': False}

        try:
            self.pytesmo_results.close()
            Path(self.original_pytesmo_ncfile).rename(
                self.original_pytesmo_ncfile + OLD_NCFILE_SUFFIX)
        except PermissionError as e:
            shutil.copy(self.original_pytesmo_ncfile,
                        self.original_pytesmo_ncfile + OLD_NCFILE_SUFFIX)

        if not self.keep_pytesmo_ncfile:
            retry_count = 5
            for i in range(retry_count):
                try:
                    self.pytesmo_results.close()
                    Path(self.original_pytesmo_ncfile +
                         OLD_NCFILE_SUFFIX).unlink()
                    break
                except PermissionError:
                    if i < retry_count - 1:
                        time.sleep(1)

        # for var in self.transcribed_dataset.data_vars:
        #     # Check if the data type is Unicode (string type)
        #     if self.transcribed_dataset[var].dtype.kind == 'U':
        #         # Find the maximum string length in this variable
        #         max_len = self.transcribed_dataset[var].str.len().max().item()
        #
        #         # Create a character array of shape (n, max_len), where n is the number of strings
        #         char_array = np.array([
        #             list(s.ljust(max_len))
        #             for s in self.transcribed_dataset[var].values
        #         ],
        #                               dtype=f'S1')
        #
        #         # Create a new DataArray for the character array with an extra character dimension
        #         self.transcribed_dataset[var] = xr.DataArray(
        #             char_array,
        #             dims=(self.transcribed_dataset[var].dims[0],
        #                   f"{var}_char"),
        #             coords={
        #                 self.transcribed_dataset[var].dims[0]:
        #                 self.transcribed_dataset[var].coords[
        #                     self.transcribed_dataset[var].dims[0]]
        #             },
        #             attrs=self.transcribed_dataset[var].
        #             attrs  # Preserve original attributes if needed
        #         )

        # Ensure the dataset is closed
        if isinstance(self.transcribed_dataset, xr.Dataset):
            self.transcribed_dataset.close()

        # Write the transcribed dataset to a new NetCDF file
        self.transcribed_dataset.to_netcdf(
            path=path,
            mode=mode,
            encoding=encoding,
        )

        return path

    def compress(self,
                 path: str,
                 compression: str = 'zlib',
                 complevel: int = 5) -> None:
        """
        Opens the generated results netCDF file and writes to a new netCDF file with new compression parameters. The smaller of both files is then deleted and the remainign one named according to the original file.

        Parameters
        ----------
        fpath: str
            Path to the netCDF file to be re-compressed.
        compression: str
            Compression algorithm to be used. Currently only 'zlib' is implemented.
        complevel: int
            Compression level to be used. The higher the level, the better the compression, but the longer it takes.

        Returns
        -------
        None
        """

        if compression in IMPLEMENTED_COMPRESSIONS and complevel in ALLOWED_COMPRESSION_LEVELS:

            def encoding_params(ds: xr.Dataset, compression: str,
                                complevel: int) -> dict:
                return {
                    str(var): {
                        compression: True,
                        'complevel': complevel
                    }
                    for var in ds.variables
                    if not np.issubdtype(ds[var].dtype, np.object_)
                    and ds[var].dtype.kind in {'i', 'u', 'f'}
                }

            try:
                with xr.open_dataset(path) as ds:
                    parent_dir = Path(path).parent
                    file = Path(path).name
                    re_name = parent_dir / f're_{file}'
                    ds.to_netcdf(re_name,
                                 mode='w',
                                 format='NETCDF4',
                                 encoding=encoding_params(
                                     ds, compression, complevel))
                    print(f'\n\nRe-compression finished\n\n')

                # for small initial files, the re-compressed file might be larger than the original
                # delete the larger file and keep the smaller; rename the smaller file to the original name if necessary
                fpath_size = os.path.getsize(path)
                re_name_size = os.path.getsize(re_name)

                if fpath_size < re_name_size:
                    Path(re_name).unlink()
                else:
                    Path(path).unlink()
                    Path(re_name).rename(path)

                return True

            except Exception as e:
                print(
                    f'\n\nRe-compression failed. {e}\nContinue without re-compression.\n\n'
                )
                return False

        else:
            raise NotImplementedError(
                f'\n\nRe-compression failed. Compression has to be {IMPLEMENTED_COMPRESSIONS} and compression levels other than {ALLOWED_COMPRESSION_LEVELS} are not supported. Continue without re-compression.\n\n'
            )

    @staticmethod
    def get_tsws_from_qa4sm_ncfile(ncfile: str) -> Union[List[str], None]:
        """
        Get the temporal sub-windows from a proper QA4SM NetCDF file.

        Parameters
        ----------
        ncfile : str
            The path to the NetCDF file.

        Returns
        -------
        List[str]
            The temporal sub-windows.
        """

        with xr.open_dataset(ncfile) as ds:
            try:
                return ds[TEMPORAL_SUB_WINDOW_NC_COORD_NAME].values.tolist()
            except KeyError:
                return None

    @staticmethod
    def get_tsws_from_pytesmo_ncfile(ncfile: str) -> Union[List[str], None]:
        """
        Get the temporal sub-windows from a pytesmo NetCDF file.

        **ATTENTION**: Only retrieves the temporal sub-windows if they are explicitly stated in the data variable names \
            present in the file. An implicit presence of the bulk case in pytesmo files is not detected.

        Parameters
        ----------
        ncfile : str
            The path to the NetCDF file.

        Returns
        -------
        List[str]
            The temporal sub-windows.
        """

        with xr.open_dataset(ncfile) as ds:
            try:
                out = list({
                    data_var.split(TEMPORAL_SUB_WINDOW_SEPARATOR)[0]
                    for data_var in list(ds.data_vars)
                    if TEMPORAL_SUB_WINDOW_SEPARATOR in data_var
                    and any([metric in data_var for metric in METRICS])
                })
                if not out:
                    return None
                return out

            except KeyError:
                return None

    @staticmethod
    def get_tsws_from_ncfile(ncfile: str) -> Union[List[str], None]:
        """
        Get the temporal sub-windows from a QA4SM or pytesmo NetCDF file.

        **ATTENTION**: Only retrieves the temporal sub-windows if they are explicitly stated in the data variable names \
            present in the file. An implicit presence of the bulk case is not detected.

        Parameters
        ----------
        ncfile : str
            The path to the NetCDF file.

        Returns
        -------
        Union[List[str], None]
            A list of temporal sub-windows or None if the file does not contain any.
        """

        def sort_tsws(tsw_list: List[str]) -> List[str]:
            '''Sort the temporal sub-windows in the order of the calendar months, the seasons, \
                and the custom temporal sub-windows. Only sorts if temporal sub-windows of only one \
                    kind are present;

                Parameters
                ----------
                tsw_list : List[str]
                    The list of temporal sub-windows.

                Returns
                -------
                List[str]
                    The sorted list of temporal sub-windows.
            '''
            if not tsw_list:
                return tsw_list

            bulk_present = DEFAULT_TSW in tsw_list
            if bulk_present:
                tsw_list.remove(DEFAULT_TSW)

            month_order = {
                month: index
                for index, month in enumerate(calendar.month_abbr) if month
            }
            seasons_1_order = {f'S{i}': i - 1 for i in range(1, 5)}
            seasons_2_order = {
                season: index
                for index, season in enumerate(['DJF', 'MAM', 'JJA', 'SON'])
            }

            def get_custom_tsws(tsw_list):
                customs = [
                    tsw for tsw in tsw_list
                    if tsw not in month_order and tsw not in seasons_1_order
                    and tsw not in seasons_2_order
                ]
                return customs, list(set(tsw_list) - set(customs))

            custom_tsws, tsw_list = get_custom_tsws(tsw_list)

            if all(tsw.isdigit() for tsw in custom_tsws):
                custom_tsws = sorted(custom_tsws, key=int)

            lens = {len(tsw) for tsw in tsw_list}

            if lens == {2} and all(
                    tsw.startswith('S')
                    for tsw in tsw_list):  # seasons like S1, S2, S3, S4
                _presorted = sorted(tsw_list, key=seasons_1_order.get)

            elif lens == {3} and all(
                    tsw in seasons_2_order
                    for tsw in tsw_list):  # seasons like DJF, MAM, JJA, SON
                _presorted = sorted(tsw_list, key=seasons_2_order.get)

            elif lens == {3} and all(tsw.isalpha()
                                     for tsw in tsw_list) and all(
                                         tsw in month_order for tsw in tsw_list
                                     ):  # months like Jan, Feb, Mar, Apr, ...
                _presorted = sorted(tsw_list, key=month_order.get)

            else:
                _presorted = tsw_list

            return ([DEFAULT_TSW]
                    if bulk_present else []) + _presorted + custom_tsws

        tsws = Pytesmo2Qa4smResultsTranscriber.get_tsws_from_qa4sm_ncfile(
            ncfile)
        if not tsws:
            tsws = Pytesmo2Qa4smResultsTranscriber.get_tsws_from_pytesmo_ncfile(
                ncfile)
        return sort_tsws(tsws)



process = psutil.Process()

def log_resources(tag):
    """Write CPU and memory usage to logfile."""
    cpu = psutil.cpu_percent(interval=None)
    virt = psutil.virtual_memory()
    rss = process.memory_info().rss / 1024**2
    print(
        f"[RES] {tag} CPU {cpu} percent RAM used {virt.percent} percent "
        f"Avail {virt.available / 1024**2:.1f} MB ProcRSS {rss:.1f} MB"
    )

# def replace_status_values(arr, replace_map):
#     """Replace values in an integer array according to a mapping dict."""
#     result = arr.copy()
#     for old_val, new_val in replace_map.items():
#         result[arr == old_val] = new_val
#     return result

class ZarrTimeoutError(Exception):
    """Custom timeout exception to avoid shadowing built-in TimeoutError."""
    pass

class NetCDFNotFoundError(Exception):
    """Raised when the NetCDF file does not exist."""
    pass



def timeout_handler(signum, frame):
    raise ZarrTimeoutError("Operation timed out")


# ============================================================================
# SUBPROCESS WORKER SCRIPT (embedded as string)
# ============================================================================

WORKER_SCRIPT = '''
import sys
import json
import numpy as np
import xarray as xr
import dask

def replace_status_values(data_array, replace_map):
    """Replace values in a DataArray according to a mapping dict.

    Works with the underlying numpy array to avoid xarray indexing limitations.
    """
    if not replace_map:
        return data_array

    # Extract values as numpy array
    arr = data_array.values.copy()

    # Perform replacement on numpy array (supports n-dimensional boolean indexing)
    for old_val, new_val in replace_map.items():
        arr[arr == old_val] = new_val

    # Return a new DataArray with the modified values, preserving dims/coords/attrs
    return xr.DataArray(
        arr,
        dims=data_array.dims,
        coords=data_array.coords,
        attrs=data_array.attrs,
        name=data_array.name
    )

def process_variable(config):
    nc_path = config["nc_path"]
    zarr_path = config["zarr_path"]
    var_name = config["var_name"]
    mode = config["mode"]
    include_coords = config["include_coords"]
    chunk_size = tuple(config["chunk_size"]) if config["chunk_size"] else None
    is_status_var = config["is_status_var"]
    status_replace = config.get("status_replace", {})

    # Open dataset with chunking for lazy loading
    chunks = "auto"
    ds = xr.open_dataset(nc_path, chunks=chunks)

    if 'tsw' in ds.dims:
        ds = ds.sel(tsw='bulk')

    var_data = ds[var_name]

    # Compute to load into memory (with synchronous scheduler)
    with dask.config.set(scheduler="synchronous"):
        var_computed = var_data.compute()

    # Apply status value replacement if needed (AFTER compute)
    if is_status_var and status_replace:
        # Convert status_replace keys to int (JSON serializes dict keys as strings)
        status_map = {int(k): v for k, v in status_replace.items()}
        var_computed = replace_status_values(var_computed, status_map)

    # Build dataset for this variable
    if include_coords:
        ds_single = xr.Dataset(
            {var_name: var_computed},
            coords=ds.coords
        )
    else:
        # For append mode, only include the variable (not coords again)
        ds_single = xr.Dataset({var_name: var_computed})

    # Prepare encoding
    encoding = {}
    if chunk_size:
        encoding[var_name] = {"chunks": chunk_size}

    # Write to zarr
    ds_single.to_zarr(
        zarr_path,
        mode=mode,
        zarr_format=3,
        consolidated=False,
        safe_chunks=False,
        encoding=encoding if encoding else None
    )

    ds.close()
    print(f"SUCCESS: {var_name}")

if __name__ == "__main__":
    config = json.loads(sys.argv[1])
    process_variable(config)
'''


class Qa4smResults2ZarrTranscriber:
    """Transcriber for converting QA4SM results to Zarr v3 format using subprocess isolation.

    IMPORTANT: Call save_zarr() only AFTER the NetCDF file has been fully created.
    This class does NOT wait for file creation - it fails immediately if the file is missing.
    """

    def __init__(self, nc_filepath, out_dir):
        """
        Initialize transcriber.

        Args:
            nc_filepath: Path to the NetCDF file (must exist when save_zarr() is called)
            out_dir: Directory where the Zarr store will be created
        """
        print("[ZARR-INIT] Creating transcriber")
        log_resources("init")

        self.nc_filepath = Path(nc_filepath)
        self.out_dir = Path(out_dir)
        self.zarr_filepath = self.out_dir / f"{self.nc_filepath.stem}.zarr"
        self.ds = None

        print(f"[ZARR-INIT] Source: {self.nc_filepath}")
        print(f"[ZARR-INIT] Output: {self.zarr_filepath}")
        log_resources("after_paths")

    def _validate_netcdf_exists(self):
        """Check that the NetCDF file exists and is readable. Fail fast if not."""
        if not self.nc_filepath.exists():
            raise NetCDFNotFoundError(
                f"NetCDF file does not exist: {self.nc_filepath}. "
                f"Ensure the validation process has completed before calling save_zarr()."
            )

        # Quick validation that file is readable
        try:
            with xr.open_dataset(self.nc_filepath) as test_ds:
                var_count = len(test_ds.data_vars)
            print(f"[ZARR] NetCDF validated: {var_count} variables found")
        except Exception as e:
            raise NetCDFNotFoundError(
                f"NetCDF file exists but cannot be opened: {self.nc_filepath}. Error: {e}"
            )

    def _load_dataset(self):
        """Load the dataset."""
        if self.ds is None:
            print("[ZARR] Loading dataset...")
            self.ds = xr.open_dataset(self.nc_filepath, chunks="auto")
            
            # Select tsw dimension if it exists
            if 'tsw' in self.ds.dims:
                self.ds = self.ds.sel(tsw=DEFAULT_TSW)
            
            print(f"[ZARR] Dataset loaded: {len(self.ds.data_vars)} variables")
        return self.ds

    def _run_variable_subprocess(self, config):
        """Run a subprocess to process a single variable."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(WORKER_SCRIPT)
            script_path = f.name

        try:
            config_json = json.dumps(config)

            result = subprocess.run(
                [sys.executable, script_path, config_json],
                capture_output=True,
                text=True,
                timeout=3600
            )

            if result.returncode != 0:
                print(f"[ZARR] Subprocess failed for {config['var_name']}")
                print(f"[ZARR] STDERR: {result.stderr}")
                raise RuntimeError(f"Failed to process variable {config['var_name']}: {result.stderr}")

            print(f"[ZARR] Subprocess output: {result.stdout.strip()}")
            return True

        except subprocess.TimeoutExpired:
            print(f"[ZARR] Timeout processing variable {config['var_name']}")
            raise
        finally:
            os.unlink(script_path)

    def save_zarr(self):
        """
        Convert NetCDF to Zarr format.

        IMPORTANT: The NetCDF file MUST exist before calling this method.
        This method does NOT wait for file creation.

        Returns:
            Path to the created Zarr store.

        Raises:
            NetCDFNotFoundError: If the NetCDF file doesn't exist or is unreadable.
        """
        print("=" * 60)
        print(f"[ZARR] Starting save_zarr at {time.strftime('%H:%M:%S')}")
        print("=" * 60)

        start_time = time.time()
        log_resources("start")

        # ========== STEP 1: Validate NetCDF exists (fail fast) ==========
        self._validate_netcdf_exists()
        log_resources("after_validation")

        # ========== STEP 2: Load dataset ==========
        self._load_dataset()
        log_resources("after_load_dataset")

        # ========== STEP 3: Prepare output ==========
        self.out_dir.mkdir(parents=True, exist_ok=True)
        log_resources("after_mkdir")

        if self.zarr_filepath.exists():
            print("[ZARR] Removing existing Zarr...")
            shutil.rmtree(self.zarr_filepath)
            log_resources("after_rm_old")

        # ========== STEP 4: Filter variables ==========
        variables_to_export = [
            x for x in self.ds.data_vars
            if x not in ['_row_size', 'gpi']
        ]
        print(f"[ZARR] Exporting {len(variables_to_export)} variables via subprocesses")
        log_resources("after_var_filter")

        chunk_sizes = {var: list(self.ds[var].shape) for var in variables_to_export}

        # ========== STEP 5: Process variables in subprocesses ==========
        failed_vars = []
        for i, var_name in enumerate(variables_to_export, 1):
            var_start = time.time()
            print(f"[ZARR] [{i}/{len(variables_to_export)}] Processing {var_name} in subprocess...")
            log_resources(f"before_subprocess_{var_name}")

            mode = 'w' if i == 1 else 'a'
            include_coords = (i == 1)
            is_status_var = var_name.startswith('status_')

            config = {
                "nc_path": str(self.nc_filepath),
                "zarr_path": str(self.zarr_filepath),
                "var_name": var_name,
                "mode": mode,
                "include_coords": include_coords,
                "chunk_size": chunk_sizes[var_name],
                "is_status_var": is_status_var,
                "status_replace": {str(k): v for k, v in status_replace.items()},
            }

            try:
                self._run_variable_subprocess(config)
                log_resources(f"after_subprocess_{var_name}")

                var_elapsed = time.time() - var_start
                print(f"[ZARR] {var_name} done in {var_elapsed:.2f}s")

            except Exception as e:
                print(f"[ZARR] Failed to process {var_name}: {e}")
                failed_vars.append(var_name)

                if i == 1:
                    raise RuntimeError(f"First variable failed, cannot continue: {e}")
                continue

        if failed_vars:
            print(f"[ZARR] Failed variables: {failed_vars}")

        # ========== STEP 6: Consolidate metadata ==========
        print("[ZARR] Consolidating metadata")
        self._consolidate_metadata_v3()
        log_resources("after_consolidation")

        # ========== STEP 7: Report results ==========
        total_size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fs in os.walk(self.zarr_filepath)
            for f in fs
        )
        log_resources("after_size_calc")

        elapsed = time.time() - start_time
        print("=" * 60)
        print("[ZARR] Complete")
        print(f"[ZARR] Size {total_size / 1024**2:.2f} MB")
        print(f"[ZARR] Time {elapsed:.2f}s")
        print(f"[ZARR] Variables: {len(variables_to_export) - len(failed_vars)}/{len(variables_to_export)}")
        print(f"[ZARR] Location {self.zarr_filepath}")
        print("=" * 60)

        log_resources("done")

        if self.ds is not None:
            self.ds.close()
            self.ds = None

        return self.zarr_filepath

    def _consolidate_metadata_v3(self):
        """Consolidate metadata using zarr v3 API."""
        try:
            store = zarr.storage.LocalStore(self.zarr_filepath)
            zarr.consolidate_metadata(store)
            print("[ZARR] Metadata consolidation OK")
        except Exception as e:
            print(f"[ZARR] Metadata consolidation skipped: {e}")


if __name__ == '__main__':
    pth = '/tmp/qa4sm/basic/0-ISMN.soil moisture_with_1-C3S.sm.nc'

    transcriber = Pytesmo2Qa4smResultsTranscriber(pytesmo_results=pth,
                                                  intra_annual_slices=None,
                                                  keep_pytesmo_ncfile=True)
    ds = transcriber.get_transcribed_dataset()
    print('writing to netcdf')
    transcriber.write_to_netcdf(
        path='/tmp/qa4sm/basic/0-ISMN.soil moisture_with_1-C3S.sm.nc.new')
