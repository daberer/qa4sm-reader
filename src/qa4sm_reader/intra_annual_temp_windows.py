from qa4sm_reader.globals import TEMPORAL_SUB_WINDOWS, DEFAULT_TSW

from pytesmo.validation_framework.metric_calculators_adapters import TsDistributor
from pytesmo.time_series.grouping import YearlessDatetime

from typing import Optional, List, Tuple, Dict, Union
from pathlib import Path
from datetime import datetime
import json
from abc import ABC, abstractmethod
import numpy as np


class InvalidTemporalSubWindowError(Exception):
    '''Exception raised when an invalid temporal sub-window is provided.'''

    def __init__(self, tsw, valid_tsw):
        super().__init__(
            f'The provided temporal sub-window ({tsw}) is invalid. Please provide one of these valid temporal sub-windows: {valid_tsw}.'
        )


class TemporalSubWindowsDefault(ABC):
    '''
    Class to load default temporal sub-window definitions from the `validator.validation.globals` file.
    Alternatively, the user can provide a custom JSON file containing the definitions.
    temporal sub-window definitions are stored in dictionaries with the following structure:

    {"seasons":
        {"S1": [[12, 1], [2, 28]]      # December 1st to February 28th
        "S2": [[3, 1], [5, 31]]       # March 1st to May 31st
        "S3": [[6, 1], [8, 31]]       # June 1st to August 31st
        "S4": [[9, 1], [11, 30]] }}     # September 1st to November 30th

    These dictionaries will be loaded as properties of the class, so that they can be accessed later on and be treated as the default.

    Parameters
    ----------

    custom_file : str, optional
        JSON File containing the temporal sub-window definitions in the same format as in `validator.validation.globals`, by default None. If None, the default as defined in `validator.validation.globals` will be used.

    '''

    def __init__(self, custom_file: Optional[str] = None) -> None:
        self.custom_file = custom_file

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(custom_file={self.custom_file})'

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({Path(self.custom_file).name})'

    def _load_json_data(self, json_path: str) -> dict:
        '''Reads and loads the JSON file into a dictionary.

        Parameters
        ----------
        json_path : str
            Path to the JSON file containing the temporal sub-window definitions.

        Returns
        -------
        dict
            Dictionary containing the default temporal sub-window definitions.
        '''

        with open(json_path, 'r') as f:
            return json.load(f)

    @abstractmethod
    def _get_available_temp_sub_wndws(self):
        pass


class NewSubWindow:
    """
    Class to store the name and the begin and end date of a new temporal sub-window.

    Parameters
    ----------
    name : str
        Name of the new temporal sub-window.
    begin_date : datetime or YearlessDatetime
        Begin date of the new temporal sub-window.
    end_date : datetime or YearlessDatetime
        End date of the new temporal sub-window.

    """

    def __init__(self, name: str, begin_date: Union[datetime,
                                                    YearlessDatetime],
                 end_date: Union[datetime, YearlessDatetime]) -> None:
        self.name = str(name)

        if type(begin_date) != type(end_date):
            raise TypeError(
                f"`begin_date` and `end_date` must be of the same type, not '{type(begin_date).__name__}' and '{type(end_date).__name__}'"
            )

        if not isinstance(begin_date, (datetime, YearlessDatetime)):
            raise TypeError(
                f"`begin_date` must be of type 'datetime' or 'YearlessDatetime', not '{type(begin_date).__name__}'"
            )
        else:
            self.begin_date = begin_date
        if not isinstance(end_date, (datetime, YearlessDatetime)):
            raise TypeError(
                f"`end_date` must be of type 'datetime' or 'YearlessDatetime', not '{type(end_date).__name__}'"
            )
        else:
            self.end_date = end_date

        if isinstance(begin_date, datetime) and isinstance(
                end_date, datetime) and self.begin_date > self.end_date:
            raise ValueError(
                f"begin_date ({self.begin_date}) must be before end_date ({self.end_date})"
            )

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.name}, {self.begin_date}, {self.end_date})'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name}, begin_date={self.begin_date}, end_date={self.end_date})'

    @property
    def begin_date_pretty(self) -> str:
        """
        Returns the begin date in a pretty format.

        Returns
        -------
        str
            Pretty formatted begin date.
        """
        return self.begin_date.strftime('%Y-%m-%d')

    @property
    def end_date_pretty(self) -> str:
        """
        Returns the end date in a pretty format.

        Returns
        -------
        str
            Pretty formatted end date.
        """
        return self.end_date.strftime('%Y-%m-%d')


class TemporalSubWindowsFactory:
    @staticmethod
    def create(temporal_sub_window_type: str, overlap: int, period: Optional[List[datetime]], custom_subwindows: Optional[dict] = None):
        """
        Factory method to instantiate the appropriate TemporalSubWindowsCreator
        based on the type of metrics passed from the validation run.
        """
        # Handle intra-annual or stability based on the type passed
        if temporal_sub_window_type in ['seasons', 'months']:
            return TemporalSubWindowsFactory._create_intra_annual(temporal_sub_window_type, overlap, period)

        elif temporal_sub_window_type == 'stability':
            return TemporalSubWindowsFactory._create_stability(overlap, period, custom_subwindows)

        return None

    @staticmethod
    def _create_intra_annual(temporal_sub_window_type: str, overlap: int, period: Optional[List[datetime]]):
        """
        Create a TemporalSubWindowsCreator for intra-annual metrics.
        """
        temp_sub_wdw_instance = TemporalSubWindowsCreator(
            temporal_sub_window_type=temporal_sub_window_type,
            overlap=overlap,
            custom_file=None  # Default to no custom file for intra-annual metrics
        )

        # Set default sub-windows
        if not period:
            period = [datetime(year=1978, month=1, day=1), datetime.now()]

        default_temp_sub_wndw = NewSubWindow(DEFAULT_TSW, period[0], period[1])
        temp_sub_wdw_instance.add_temp_sub_wndw(
            new_temp_sub_wndw=default_temp_sub_wndw,
            insert_as_first_wndw=True
        )

        return temp_sub_wdw_instance

    @staticmethod
    def _create_stability(overlap: int, period: Optional[List[datetime]], custom_subwindows: Optional[dict] = None):
        """
        Create a TemporalSubWindowsCreator for stability metrics.
        """
        temp_sub_wdw_instance = TemporalSubWindowsCreator(
            temporal_sub_window_type="stability",
            overlap=overlap,
            custom_file=None
        )

        # Remove existing sub-windows
        temp_sub_wdw_instance.remove_temp_sub_wndws()

        # Set default sub-windows
        if not period:
            period = [datetime(year=1978, month=1, day=1), datetime.now()]

        default_temp_sub_wndw = NewSubWindow(DEFAULT_TSW, period[0], period[1])
        temp_sub_wdw_instance.add_temp_sub_wndw(default_temp_sub_wndw)

        # Add annual sub-windows based on the years in the period
        start_year, end_year = map(lambda dt: dt.year, period)
        years = list(range(start_year, end_year + 1))
        add_annual_subwindows(temp_sub_wdw_instance, years)

        # Add custom sub-windows if provided
        if custom_subwindows:
            for key, value in custom_subwindows.items():
                new_subwindow = NewSubWindow(
                    name=key,
                    begin_date=datetime(value[0][0], value[0][1], value[0][2]),
                    end_date=datetime(value[1][0], value[1][1], value[1][2]),
                )
                temp_sub_wdw_instance.add_temp_sub_wndw(new_subwindow)

        return temp_sub_wdw_instance


# Helper function to add annual sub-windows
def add_annual_subwindows(temp_sub_wdw_instance: 'TemporalSubWindowsCreator', years: List[int]):
    """
    Add annual sub-windows based on the list of years.

    Parameters
    ----------
    temp_sub_wdw_instance: TemporalSubWindowsCreator
        The instance to which the sub-windows will be added.
    years: List[int]
        List of years to generate sub-windows for.
    """
    for year in years:
        temp_sub_wdw_instance.add_temp_sub_wndw(
            NewSubWindow(f"{year}", begin_date=datetime(year, 1, 1), end_date=datetime(year, 12, 31))
        )




class TemporalSubWindowsCreator(TemporalSubWindowsDefault):
    '''Class to create custom temporal sub-windows, based on the default definitions.

    Parameters
    ----------
    temporal_sub_window_type : str
        Type of temporal sub-window to be created. Must be one of the available default types. Officially, "months" and "seasons" are implemented. The user can implement their own defaults, though (see `TemporalSubWindowsDefault`). Default is "months".
    overlap : int, optional
        Number of days to be added/subtracted to the beginning/end of the temporal sub-window. Default is 0.
    custom_file : str, optional
        Path to the JSON file containing the temporal sub-window definitions, by default None (meaning the defaults as defined in `validator.validation.globals` will be used)
    '''

    def __init__(self,
                 temporal_sub_window_type: Optional[str] = 'months',
                 overlap: Optional[int] = 0,
                 custom_file: Optional[str] = None):
        self.overlap = int(np.round(overlap))
        self.temporal_sub_window_type = temporal_sub_window_type
        super().__init__(custom_file=custom_file)

        self.available_temp_sub_wndws = self._get_available_temp_sub_wndws()
        if not self.available_temp_sub_wndws:
            raise FileNotFoundError(
                f'Invalid custom file path. Please provide a valid JSON file containing the temporal sub-window definitions.'
            )
        elif self.temporal_sub_window_type not in self.available_temp_sub_wndws:
            raise InvalidTemporalSubWindowError(self.temporal_sub_window_type,
                                                self.available_temp_sub_wndws)
            raise KeyError(
                f'Invalid temporal sub-window type. Available types are: {self.available_temp_sub_wndws}'
            )

        self.custom_temporal_sub_windows = self._custom_temporal_sub_windows()
        self.additional_temp_sub_wndws_container = {}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(temporal_sub_window_type={self.temporal_sub_window_type}, overlap={self.overlap})'

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.temporal_sub_window_type}, {self.overlap})'

    def __date_to_doy(self, date_tuple: Tuple[int, int]) -> int:
        '''Converts a date list [month, day] to a day of the year (doy). Leap years are neglected.

        Parameters
        ----------
        date_tuple : List[int]
            List containing the month and day of the date to be converted. The year is not required, as it is not used in the conversion.

        Returns
        -------
        int
            Day of the year (doy) corresponding to the date provided.
        '''

        _doy = YearlessDatetime(date_tuple[0], date_tuple[1]).doy
        if _doy > 60:  # assume NO leap year
            _doy -= 1

        return _doy

    def __doy_to_date(self, doy: int) -> Tuple[int, int]:
        '''Converts a day of the year (doy) to a date tuple (month, day). Leap years are neglected.

        Parameters
        ----------
        doy : int
            Day of the year (doy) to be converted.

        Returns
        -------
        Tuple[int]
            Tuple containing the month and day corresponding to the doy provided.
        '''

        date = datetime.strptime(str(doy), '%j')
        month = date.strftime('%m')
        day = date.strftime('%d')

        return int(month), int(day)

    def __update_date(self, date: Tuple[int, int],
                      overlap_direction: float) -> Tuple[int, int]:
        '''Updates a date tuple (month, day) by adding/subtracting the overlap value to/from it.

        Parameters
        ----------
        date : Tuple[int]
            Date to be updated.
        overlap_direction : float
            Direction of the overlap. Must be either -1 or +1. -1: subtract overlap value from date; +1: add overlap value to date.

        Returns
        -------
        Tuple[int]
            Updated date tuple.
        '''

        overlap_direction = overlap_direction / abs(
            overlap_direction)  # making sure it's either -1 or +1
        _doy = self.__date_to_doy(date)
        _doy += int(overlap_direction * self.overlap)

        if _doy < 1:
            _doy = 365 - abs(_doy)
        elif _doy > 365:
            _doy = _doy - 365

        return self.__doy_to_date(_doy)

    def _custom_temporal_sub_window(self):
        return {
            key: (self.__update_date(val[0], overlap_direction=-1),
                  self.__update_date(val[1], overlap_direction=+1))
            for key, val in self.temporal_sub_windows_dict[
                self.temporal_sub_window_type].items()
        }

    def _get_available_temp_sub_wndws(self) -> Union[List[str], None]:
        if not self.custom_file:
            self.temporal_sub_windows_dict = TEMPORAL_SUB_WINDOWS
            return list(self.temporal_sub_windows_dict.keys())
        elif Path(self.custom_file).is_file():
            self.temporal_sub_windows_dict = self._load_json_data(
                self.custom_file)
            return list(self.temporal_sub_windows_dict.keys())
        else:
            return None

    def _custom_temporal_sub_windows(self) -> Dict[str, TsDistributor]:
        '''Custom temporal sub-window based, on the default definitions and the overlap value.

        Parameters
        ----------
        None

        Returns
        -------
        dict[str, TsDistributor]
            Dictionary containing the custom temporal sub-window definitions.
        '''

        def tsdistributor(_begin_date: Tuple[int],
                          _end_date: Tuple[int]) -> TsDistributor:
            return TsDistributor(yearless_date_ranges=[(YearlessDatetime(
                *_begin_date), YearlessDatetime(*_end_date))])

        return {
            key: tsdistributor(val[0], val[1])
            for key, val in self._custom_temporal_sub_window().items()
        }

    def add_temp_sub_wndw(
        self,
        new_temp_sub_wndw: NewSubWindow,
        insert_as_first_wndw: Optional[bool] = False
    ) -> Union[None, Dict[str, TsDistributor]]:
        '''Adds a new custom temporal sub-window to the existing ones.

        Parameters
        ----------
        new_temp_sub_wndw : NewSubWindow
            Dataclass containing the name, begin date, and end date of the new temporal sub-window.
        insert_as_first_wndw : bool, optional
            If True, the new temporal sub-window will be inserted as new first element in `TemporalSubWindowsCreator.custom_temporal_sub_windows`. Default is False.

        Returns
        -------
        Union[None, Dict[str, TsDistributor]]
            None if the new temp_sub_wndw already exists. Otherwise, the dictionary containing the custom temporal sub-window definitions.

        '''

        self.additional_temp_sub_wndws_container[
            new_temp_sub_wndw.name] = new_temp_sub_wndw
        try:
            if new_temp_sub_wndw.name in self.custom_temporal_sub_windows:
                print(
                    f'temporal sub-window "{new_temp_sub_wndw.name}" already exists. Overwriting not possible.\
                         Please choose a different name. If you want to overwrite the existing temporal sub-window, \
                             use the `overwrite_temp_sub_wndw` method instead.'
                )
                return None
            elif insert_as_first_wndw:
                _new_first_element = {
                    new_temp_sub_wndw.name:
                    TsDistributor(date_ranges=[(new_temp_sub_wndw.begin_date,
                                                new_temp_sub_wndw.end_date)])
                }
                self.custom_temporal_sub_windows = {
                    **_new_first_element,
                    **self.custom_temporal_sub_windows
                }
                return self.custom_temporal_sub_windows
            else:
                self.custom_temporal_sub_windows[
                    new_temp_sub_wndw.name] = TsDistributor(
                        date_ranges=[(new_temp_sub_wndw.begin_date,
                                      new_temp_sub_wndw.end_date)])
                return self.custom_temporal_sub_windows
        except Exception as e:
            print(f'Error: {e}')
            return None

    def overwrite_temp_sub_wndw(
        self, new_temp_sub_wndw: NewSubWindow
    ) -> Union[None, Dict[str, TsDistributor]]:
        '''Overwrites an existing temporal sub-window with a new definition.

        Parameters
        ----------
        new_temp_sub_wndw : NewSubWindow
            Dataclass containing the name, begin date, and end date of the new temporal sub-window.

        Returns
        -------
        Union[None, Dict[str, TsDistributor]]
            None if the new temp_sub_wndw does not exist. Otherwise, the dictionary containing the custom temporal sub-window definitions.

        '''

        self.additional_temp_sub_wndws_container[
            new_temp_sub_wndw.name] = new_temp_sub_wndw
        try:
            if new_temp_sub_wndw.name not in self.custom_temporal_sub_windows:
                print(
                    f'temporal sub-window "{new_temp_sub_wndw.name}" does not exist. Overwriting not possible.\
                         Please choose a different name. If you want to add a new temporal sub-window, \
                             use the `add_temp_sub_wndw` method instead.')
                return None
            elif isinstance(
                    new_temp_sub_wndw.begin_date, datetime
            ) and isinstance(
                    new_temp_sub_wndw.end_date, datetime
            ) and new_temp_sub_wndw.begin_date < new_temp_sub_wndw.end_date:
                self.custom_temporal_sub_windows[
                    new_temp_sub_wndw.name] = TsDistributor(
                        date_ranges=[(new_temp_sub_wndw.begin_date,
                                      new_temp_sub_wndw.end_date)])
                return self.custom_temporal_sub_windows
            elif isinstance(new_temp_sub_wndw.begin_date,
                            YearlessDatetime) and isinstance(
                                new_temp_sub_wndw.end_date, YearlessDatetime):
                self.custom_temporal_sub_windows[
                    new_temp_sub_wndw.name] = TsDistributor(
                        yearless_date_ranges=[(new_temp_sub_wndw.begin_date,
                                               new_temp_sub_wndw.end_date)])
                return self.custom_temporal_sub_windows

        except Exception as e:
            print(f'Error: {e}')
            return None
        
    def remove_temp_sub_wndws(self) -> None:
        '''Removes all existing custom temporal sub-windows.

        This method clears both the `custom_temporal_sub_windows` and the `additional_temp_sub_wndws_container`.
        '''
        self.custom_temporal_sub_windows.clear()
        self.additional_temp_sub_wndws_container.clear()
        print("All custom temporal sub-windows have been removed.")        

    @property
    def names(self) -> List[str]:
        '''Returns the names of the temporal sub-windows.

        Parameters
        ----------
        None

        Returns
        -------
        List[str]
            List containing the names of the temporal sub-windows.
        '''

        return list(self.custom_temporal_sub_windows.keys())

    @property
    def metadata(self) -> Dict:
        '''Returns the metadata of the temporal sub-windows.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, Union[str, List[str]]]
            Dictionary containing the metadata of the temporal sub-windows.
        '''

        def _date_formatter(_date: Tuple[int, int]) -> str:
            return f'{_date[0]:02d}-{_date[1]:02d}'

        metadata_dict = {
            'Temporal sub-window type':
            self.temporal_sub_window_type,
            'Overlap':
            f'{self.overlap} days',
            'Pretty Names [MM-DD]': (', ').join([
                f'{key}: {_date_formatter(val[0])} to {_date_formatter(val[1])}'
                for key, val in self._custom_temporal_sub_window().items()
            ])
        }

        if self._custom_temporal_sub_window().items() != self.names:
            unique_tsws = list(
                set(self.names) -
                set(self._custom_temporal_sub_window().keys()))
            if DEFAULT_TSW in unique_tsws:
                metadata_dict[
                    DEFAULT_TSW] = f'This is the default case "{DEFAULT_TSW}". The user specified it to range from {self.additional_temp_sub_wndws_container[DEFAULT_TSW].begin_date_pretty} to {self.additional_temp_sub_wndws_container[DEFAULT_TSW].end_date_pretty}. Note: This specified time interval might differ from the actual time interval in which all datasets are available. Refer to the datasets section (https://qa4sm.eu/ui/datasets) for more information.'

        return metadata_dict
