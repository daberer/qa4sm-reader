import os
from pathlib import Path
import pytest
import json
from copy import deepcopy
from datetime import datetime
from pytesmo.validation_framework.metric_calculators_adapters import TsDistributor
from pytesmo.time_series.grouping import YearlessDatetime

from qa4sm_reader.intra_annual_temp_windows import TemporalSubWindowsDefault, TemporalSubWindowsCreator, NewSubWindow, InvalidTemporalSubWindowError


@pytest.fixture
def default_monthly_sub_windows_no_overlap():
    # default 'months' defined in globals.py
    return TemporalSubWindowsCreator(temporal_sub_window_type="months",
                                     overlap=0,
                                     custom_file=None)


@pytest.fixture
def default_seasonal_sub_windows_no_overlap():
    # default 'seasons' defined in globals.py
    return TemporalSubWindowsCreator(temporal_sub_window_type="seasons",
                                     overlap=0,
                                     custom_file=None)


@pytest.fixture
def seasonal_sub_windows_positive_overlap():
    # the ovelap is in units of days and can be positive or negative and is applied to both ends of the temporal sub-windows
    # a positive overlap will result in temporal sub-windows that overlap with each other
    return TemporalSubWindowsCreator(temporal_sub_window_type="seasons",
                                     overlap=5,
                                     custom_file=None)


@pytest.fixture
def seasonal_sub_windows_negative_overlap():
    # a negative overlap will result in temporal sub-windows that have gaps between them
    return TemporalSubWindowsCreator(temporal_sub_window_type="seasons",
                                     overlap=-5,
                                     custom_file=None)


@pytest.fixture
def temporal_sub_windows_custom():
    # load custom temporal sub-windows from json file
    return TemporalSubWindowsCreator(
        temporal_sub_window_type='custom',
        overlap=0,
        custom_file=Path(__file__).resolve().parent.parent / 'tests' /
        'test_data' / 'intra_annual' / 'custom_intra_annual_windows.json')


@pytest.fixture
def additional_temp_sub_window():
    # create a new temporal sub-window, to be used in addition to the default ones
    return NewSubWindow(name="Feynman",
                        begin_date=datetime(1918, 5, 11),
                        end_date=datetime(1988, 2, 15))


#------------------------- Tests for TemporalSubwindowsDefault class -----------------------------------------------------------------------


class TemporalSubWindowsConcrete(TemporalSubWindowsDefault):
    # used to test the abstract class TemporalSubWindowsDefault
    def _get_available_temp_sub_wndws(self):
        return {"seasons": {"S1": [[12, 1], [2, 28]], "S2": [[3, 1], [5, 31]]}}


def test_initialization():
    temp_sub_windows = TemporalSubWindowsConcrete(custom_file='test.json')
    assert temp_sub_windows.custom_file == 'test.json'


def test_load_json_data(tmp_path):
    test_data = {
        "seasons": {
            "S1": [[12, 1], [2, 28]],
            "S2": [[3, 1], [5, 31]]
        }
    }
    test_file = tmp_path / "test.json"
    with open(test_file, 'w') as f:
        json.dump(test_data, f)

    temp_sub_windows = TemporalSubWindowsConcrete()
    loaded_data = temp_sub_windows._load_json_data(test_file)
    assert loaded_data == test_data


def test_get_available_temp_sub_wndws():
    temp_sub_windows = TemporalSubWindowsConcrete()
    available_windows = temp_sub_windows._get_available_temp_sub_wndws()
    assert available_windows == {
        "seasons": {
            "S1": [[12, 1], [2, 28]],
            "S2": [[3, 1], [5, 31]]
        }
    }


#------------------------- Tests for NewSubWindow class -----------------------------------------------------------------------


def test_new_sub_window_attributes(additional_temp_sub_window):
    # used to generate proper description of temporal sub-window dimenison in the netCDF file
    assert additional_temp_sub_window.begin_date_pretty == '1918-05-11'
    assert additional_temp_sub_window.end_date_pretty == '1988-02-15'


def test_faulty_new_sub_window():
    # begin_date and end_date must be datetime objects
    with pytest.raises((TypeError, AttributeError)):
        NewSubWindow(name="Test Window",
                     begin_date="2023-01-01",
                     end_date=datetime.now())

    with pytest.raises((TypeError, AttributeError)):
        NewSubWindow(name="Test Window",
                     begin_date=datetime.now(),
                     end_date="2023-12-31")

    # begin_date must be before end_date, bc date is NOT a yearless date
    with pytest.raises(ValueError):
        NewSubWindow(name="Test Window",
                     begin_date=datetime(5000, 1, 1),
                     end_date=datetime(1000, 1, 1))

    # both begin_date and end_date must be instances of the same class
    with pytest.raises(TypeError):
        NewSubWindow(name="Test Window",
                     begin_date=datetime(5000, 1, 1),
                     end_date=YearlessDatetime(1, 1))


#------------------------- Tests for TemporalSubWindowsCreator class ----------------------------------------------------------


def test_default_monthly_sub_windows_attributes(
        default_monthly_sub_windows_no_overlap,
        default_seasonal_sub_windows_no_overlap):
    assert default_monthly_sub_windows_no_overlap.temporal_sub_window_type == "months"

    assert default_seasonal_sub_windows_no_overlap.temporal_sub_window_type == "seasons"

    assert default_monthly_sub_windows_no_overlap.overlap == default_seasonal_sub_windows_no_overlap.overlap == 0

    assert default_monthly_sub_windows_no_overlap.custom_file == default_seasonal_sub_windows_no_overlap.custom_file == None

    assert default_monthly_sub_windows_no_overlap.available_temp_sub_wndws == default_seasonal_sub_windows_no_overlap.available_temp_sub_wndws == [
        'seasons', 'months', 'stability'
    ]

    assert default_monthly_sub_windows_no_overlap.names == [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
        'Nov', 'Dec'
    ]

    # included so that if definition of months changes in the globals.py file, the test will fail
    assert default_monthly_sub_windows_no_overlap.custom_temporal_sub_windows[
        'Jan'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(1, 1),
            YearlessDatetime(1, 31))]).yearless_date_ranges
    assert default_monthly_sub_windows_no_overlap.custom_temporal_sub_windows[
        'Feb'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(2, 1),
            YearlessDatetime(2, 28))]).yearless_date_ranges
    assert default_monthly_sub_windows_no_overlap.custom_temporal_sub_windows[
        'Mar'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(3, 1),
            YearlessDatetime(3, 31))]).yearless_date_ranges
    assert default_monthly_sub_windows_no_overlap.custom_temporal_sub_windows[
        'Apr'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(4, 1),
            YearlessDatetime(4, 30))]).yearless_date_ranges
    assert default_monthly_sub_windows_no_overlap.custom_temporal_sub_windows[
        'May'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(5, 1),
            YearlessDatetime(5, 31))]).yearless_date_ranges
    assert default_monthly_sub_windows_no_overlap.custom_temporal_sub_windows[
        'Jun'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(6, 1),
            YearlessDatetime(6, 30))]).yearless_date_ranges
    assert default_monthly_sub_windows_no_overlap.custom_temporal_sub_windows[
        'Jul'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(7, 1),
            YearlessDatetime(7, 31))]).yearless_date_ranges
    assert default_monthly_sub_windows_no_overlap.custom_temporal_sub_windows[
        'Aug'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(8, 1),
            YearlessDatetime(8, 31))]).yearless_date_ranges
    assert default_monthly_sub_windows_no_overlap.custom_temporal_sub_windows[
        'Sep'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(9, 1),
            YearlessDatetime(9, 30))]).yearless_date_ranges
    assert default_monthly_sub_windows_no_overlap.custom_temporal_sub_windows[
        'Oct'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(10, 1),
            YearlessDatetime(10, 31))]).yearless_date_ranges
    assert default_monthly_sub_windows_no_overlap.custom_temporal_sub_windows[
        'Nov'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(11, 1),
            YearlessDatetime(11, 30))]).yearless_date_ranges
    assert default_monthly_sub_windows_no_overlap.custom_temporal_sub_windows[
        'Dec'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(12, 1),
            YearlessDatetime(12, 31))]).yearless_date_ranges

    assert default_monthly_sub_windows_no_overlap.additional_temp_sub_wndws_container == {}

    # used to generate proper description of temporal sub-window dimenison in the netCDF file
    assert default_monthly_sub_windows_no_overlap.metadata == {
        'Temporal sub-window type':
        'months',
        'Overlap':
        '0 days',
        'Pretty Names [MM-DD]':
        'Jan: 01-01 to 01-31, Feb: 02-01 to 02-28, Mar: 03-01 to 03-31, Apr: 04-01 to 04-30, May: 05-01 to 05-31, Jun: 06-01 to 06-30, Jul: 07-01 to 07-31, Aug: 08-01 to 08-31, Sep: 09-01 to 09-30, Oct: 10-01 to 10-31, Nov: 11-01 to 11-30, Dec: 12-01 to 12-31'
    }


def test_default_seasonal_sub_windows_attributes(
        default_seasonal_sub_windows_no_overlap):
    assert default_seasonal_sub_windows_no_overlap.temporal_sub_window_type == "seasons"

    assert default_seasonal_sub_windows_no_overlap.overlap == 0

    assert default_seasonal_sub_windows_no_overlap.custom_file == None

    assert default_seasonal_sub_windows_no_overlap.available_temp_sub_wndws == [
        'seasons', 'months', 'stability'
    ]

    assert default_seasonal_sub_windows_no_overlap.names == [
        'S1', 'S2', 'S3', 'S4'
    ]

    # included so that if definition of months changes in the globals.py file, the test will fail
    assert default_seasonal_sub_windows_no_overlap.custom_temporal_sub_windows[
        'S1'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(12, 1),
            YearlessDatetime(2, 28))]).yearless_date_ranges
    assert default_seasonal_sub_windows_no_overlap.custom_temporal_sub_windows[
        'S2'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(3, 1),
            YearlessDatetime(5, 31))]).yearless_date_ranges
    assert default_seasonal_sub_windows_no_overlap.custom_temporal_sub_windows[
        'S3'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(6, 1),
            YearlessDatetime(8, 31))]).yearless_date_ranges
    assert default_seasonal_sub_windows_no_overlap.custom_temporal_sub_windows[
        'S4'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(9, 1),
            YearlessDatetime(11, 30))]).yearless_date_ranges

    assert default_seasonal_sub_windows_no_overlap.additional_temp_sub_wndws_container == {}

    # used to generate proper description of temporal sub-window dimenison in the netCDF file
    assert default_seasonal_sub_windows_no_overlap.metadata == {
        'Temporal sub-window type':
        'seasons',
        'Overlap':
        '0 days',
        'Pretty Names [MM-DD]':
        'S1: 12-01 to 02-28, S2: 03-01 to 05-31, S3: 06-01 to 08-31, S4: 09-01 to 11-30'
    }


def test_faulty_temporal_sub_windows_creator():
    # temporal_sub_window_type must be either 'months' or 'seasons'
    with pytest.raises(InvalidTemporalSubWindowError):
        TemporalSubWindowsCreator(
            temporal_sub_window_type="not-a-default-value",
            overlap=0,
            custom_file=None)


def test_load_custom_temporal_sub_windows(temporal_sub_windows_custom):
    # 'temporal_sub_window_type' corresponds to the defined temporal sub-windows in the provided json file
    # the file may contain any number of temporal sub-windows, but one is selected via a keyword argument 'temporal_sub_window_type' for each TemporalSubWindowsCreator instance

    assert temporal_sub_windows_custom.custom_file == Path(__file__).resolve(
    ).parent.parent / 'tests' / 'test_data' / 'intra_annual' / 'custom_intra_annual_windows.json'

    assert temporal_sub_windows_custom.temporal_sub_window_type == 'custom'

    assert temporal_sub_windows_custom.overlap == 0

    assert temporal_sub_windows_custom.available_temp_sub_wndws == [
        'seasons', 'months', 'custom'
    ]

    assert temporal_sub_windows_custom.names == [
        'star wars month', 'halloween season', 'advent', 'movember',
        'christmas'
    ]

    assert temporal_sub_windows_custom.custom_temporal_sub_windows[
        'star wars month'].yearless_date_ranges == TsDistributor(
            yearless_date_ranges=[(
                YearlessDatetime(5, 1),
                YearlessDatetime(5, 31))]).yearless_date_ranges
    assert temporal_sub_windows_custom.custom_temporal_sub_windows[
        'halloween season'].yearless_date_ranges == TsDistributor(
            yearless_date_ranges=[(
                YearlessDatetime(10, 1),
                YearlessDatetime(10, 31))]).yearless_date_ranges
    assert temporal_sub_windows_custom.custom_temporal_sub_windows[
        'advent'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(12, 1),
            YearlessDatetime(12, 24))]).yearless_date_ranges
    assert temporal_sub_windows_custom.custom_temporal_sub_windows[
        'movember'].yearless_date_ranges == TsDistributor(
            yearless_date_ranges=[(
                YearlessDatetime(11, 1),
                YearlessDatetime(11, 30))]).yearless_date_ranges
    assert temporal_sub_windows_custom.custom_temporal_sub_windows[
        'christmas'].yearless_date_ranges == TsDistributor(
            yearless_date_ranges=[(
                YearlessDatetime(12, 24),
                YearlessDatetime(12, 26))]).yearless_date_ranges


def test_load_nonexistent_custom_temporal_sub_windows():
    with pytest.raises(FileNotFoundError):
        TemporalSubWindowsCreator(temporal_sub_window_type='whatever',
                                  overlap=0,
                                  custom_file='i_dont_exist.json')


def test_overlap_parameter(seasonal_sub_windows_positive_overlap,
                           seasonal_sub_windows_negative_overlap):
    # overlap is added to both ends of the temporal sub-windows
    assert seasonal_sub_windows_positive_overlap.overlap == 5
    assert seasonal_sub_windows_positive_overlap.custom_temporal_sub_windows[
        'S1'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(11, 26),
            YearlessDatetime(3, 5))]).yearless_date_ranges
    assert seasonal_sub_windows_positive_overlap.custom_temporal_sub_windows[
        'S2'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(2, 24),
            YearlessDatetime(6, 5))]).yearless_date_ranges
    assert seasonal_sub_windows_positive_overlap.custom_temporal_sub_windows[
        'S3'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(5, 27),
            YearlessDatetime(9, 5))]).yearless_date_ranges
    assert seasonal_sub_windows_positive_overlap.custom_temporal_sub_windows[
        'S4'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(8, 27),
            YearlessDatetime(12, 5))]).yearless_date_ranges
    assert seasonal_sub_windows_positive_overlap.metadata == {
        'Temporal sub-window type':
        'seasons',
        'Overlap':
        '5 days',
        'Pretty Names [MM-DD]':
        'S1: 11-26 to 03-05, S2: 02-24 to 06-05, S3: 05-27 to 09-05, S4: 08-27 to 12-05'
    }

    # overlap is subtracted from both ends of the temporal sub-windows
    assert seasonal_sub_windows_negative_overlap.overlap == -5
    assert seasonal_sub_windows_negative_overlap.custom_temporal_sub_windows[
        'S1'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(12, 6),
            YearlessDatetime(2, 23))]).yearless_date_ranges
    assert seasonal_sub_windows_negative_overlap.custom_temporal_sub_windows[
        'S2'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(3, 6),
            YearlessDatetime(5, 26))]).yearless_date_ranges
    assert seasonal_sub_windows_negative_overlap.custom_temporal_sub_windows[
        'S3'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(6, 6),
            YearlessDatetime(8, 26))]).yearless_date_ranges
    assert seasonal_sub_windows_negative_overlap.custom_temporal_sub_windows[
        'S4'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(9, 6),
            YearlessDatetime(11, 25))]).yearless_date_ranges
    assert seasonal_sub_windows_negative_overlap.metadata == {
        'Temporal sub-window type':
        'seasons',
        'Overlap':
        '-5 days',
        'Pretty Names [MM-DD]':
        'S1: 12-06 to 02-23, S2: 03-06 to 05-26, S3: 06-06 to 08-26, S4: 09-06 to 11-25'
    }

    # overlap is rounded to the nearest integer
    float_overlap = TemporalSubWindowsCreator(
        temporal_sub_window_type="seasons", overlap=5.2, custom_file=None)

    assert float_overlap.overlap == 5

    assert [
        x.yearless_date_ranges
        for x in float_overlap.custom_temporal_sub_windows.values()
    ] == [
        x.yearless_date_ranges for x in seasonal_sub_windows_positive_overlap.
        custom_temporal_sub_windows.values()
    ]

    assert float_overlap.metadata == seasonal_sub_windows_positive_overlap.metadata

    # make sure cyclic boundaries are handled correctly
    # no overlap and +/-365 days overlap should be the same
    aa = TemporalSubWindowsCreator(temporal_sub_window_type="seasons",
                                   overlap=0,
                                   custom_file=None)
    aa_plus = TemporalSubWindowsCreator(temporal_sub_window_type="seasons",
                                        overlap=365,
                                        custom_file=None)
    aa_minus = TemporalSubWindowsCreator(temporal_sub_window_type="seasons",
                                         overlap=-365,
                                         custom_file=None)
    # +376 days overlap should be the same as +11 days overlap and -354 days overlap
    bb = TemporalSubWindowsCreator(temporal_sub_window_type="seasons",
                                   overlap=376,
                                   custom_file=None)
    bb_plus = TemporalSubWindowsCreator(temporal_sub_window_type="seasons",
                                        overlap=11,
                                        custom_file=None)
    bb_minus = TemporalSubWindowsCreator(temporal_sub_window_type="seasons",
                                         overlap=-354,
                                         custom_file=None)

    assert [
        x.yearless_date_ranges
        for x in aa.custom_temporal_sub_windows.values()
    ] == [
        x.yearless_date_ranges
        for x in aa_plus.custom_temporal_sub_windows.values()
    ] == [
        x.yearless_date_ranges
        for x in aa_minus.custom_temporal_sub_windows.values()
    ]
    assert [
        x.yearless_date_ranges
        for x in bb.custom_temporal_sub_windows.values()
    ] == [
        x.yearless_date_ranges
        for x in bb_plus.custom_temporal_sub_windows.values()
    ] == [
        x.yearless_date_ranges
        for x in bb_minus.custom_temporal_sub_windows.values()
    ]


def test_add_temporal_sub_window(seasonal_sub_windows_positive_overlap,
                                 additional_temp_sub_window):
    seasonal_sub_windows_positive_overlap.add_temp_sub_wndw(
        additional_temp_sub_window)

    assert seasonal_sub_windows_positive_overlap.names == [
        'S1', 'S2', 'S3', 'S4', 'Feynman'
    ]

    assert seasonal_sub_windows_positive_overlap.custom_temporal_sub_windows[
        'Feynman'].date_ranges == [(datetime(1918, 5,
                                             11), datetime(1988, 2, 15))]

    # if a new window is to be added, it should not have a name that already exists. In this case, this new window should not be added
    name_exists = NewSubWindow(name="S1",
                               begin_date=YearlessDatetime(5, 11),
                               end_date=YearlessDatetime(2, 15))

    seasonal_sub_windows_positive_overlap_copy = deepcopy(
        seasonal_sub_windows_positive_overlap)

    seasonal_sub_windows_positive_overlap_copy.add_temp_sub_wndw(name_exists)

    assert seasonal_sub_windows_positive_overlap_copy.names == seasonal_sub_windows_positive_overlap.names

    assert [
        x.yearless_date_ranges
        for x in seasonal_sub_windows_positive_overlap_copy.
        custom_temporal_sub_windows.values()
    ] == [
        x.yearless_date_ranges for x in seasonal_sub_windows_positive_overlap.
        custom_temporal_sub_windows.values()
    ]

    # if a new window is added and specified to become the first window, it should be added at the beginning
    seasonal_sub_windows_positive_overlap_copy.add_temp_sub_wndw(
        NewSubWindow('I am first', YearlessDatetime(1, 1),
                     YearlessDatetime(2, 2)),
        insert_as_first_wndw=True)
    assert seasonal_sub_windows_positive_overlap_copy.names[0] == 'I am first'

    # if an existing window is to be overwritten, it should exist.
    seasonal_sub_windows_positive_overlap_copy = deepcopy(
        seasonal_sub_windows_positive_overlap)
    seasonal_sub_windows_positive_overlap_copy.overwrite_temp_sub_wndw(
        name_exists)

    assert seasonal_sub_windows_positive_overlap_copy.names == seasonal_sub_windows_positive_overlap.names

    assert seasonal_sub_windows_positive_overlap_copy.custom_temporal_sub_windows[
        'S1'].yearless_date_ranges == TsDistributor(yearless_date_ranges=[(
            YearlessDatetime(5, 11),
            YearlessDatetime(2, 15))]).yearless_date_ranges

    # when overwriting an existing window, it should be possible to use a new datatype for the dates (but always either datetime or YearlessDatetime)
    seasonal_sub_windows_positive_overlap_copy.overwrite_temp_sub_wndw(
        NewSubWindow('S1', datetime(2023, 1, 1), datetime(2023, 12, 31)))

    assert seasonal_sub_windows_positive_overlap_copy.custom_temporal_sub_windows[
        'S1'].yearless_date_ranges == None
    assert seasonal_sub_windows_positive_overlap_copy.custom_temporal_sub_windows[
        'S1'].date_ranges == TsDistributor(
            date_ranges=[(datetime(2023, 1, 1),
                          datetime(2023, 12, 31))]).date_ranges
