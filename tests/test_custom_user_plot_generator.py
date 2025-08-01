import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import xarray as xr


from qa4sm_reader.custom_user_plot_generator import (
    combined_boxplot,
    validate_and_subset_data,
    calculate_padded_extent,
    plot_static_map,
    map_plot
)

@pytest.fixture
def plotdir():
    plotdir = tempfile.mkdtemp()

    return plotdir



@pytest.fixture
def sample_dataframe():
    """Test combined_boxplot with valid input."""
    file_name = '3-ERA5_LAND.swvl1_with_1-C3S.sm_with_2-ASCAT.sm.nc'
    file_path = os.path.join(os.path.dirname(__file__), '..', 'tests',
                                   'test_data', 'tc', file_name)
    ds = xr.open_dataset(file_path)
    df = ds.to_dataframe()
    return df


from pathlib import Path


def test_combined_boxplot_valid_input(plotdir):
    """Test combined_boxplot with valid input."""
    testfile_1 = '3-ERA5_LAND.swvl1_with_1-C3S.sm_with_2-ASCAT.sm.nc'
    testfile_path_1 = os.path.join(os.path.dirname(__file__), '..', 'tests',
                                   'test_data', 'tc', testfile_1)
    ds1 = xr.open_dataset(testfile_path_1)
    df1 = ds1.to_dataframe()
    testfile_2 = '3-GLDAS.SoilMoi0_10cm_inst_with_1-C3S.sm_with_2-SMOS.Soil_Moisture.nc'
    testfile_path_2 = os.path.join(os.path.dirname(__file__), '..', 'tests',
                                   'test_data', 'tc', testfile_2)
    ds2 = xr.open_dataset(testfile_path_2)
    df2 = ds2.to_dataframe()

    # Convert output_dir to a Path object
    output_dir = Path(plotdir)

    # Call combined_boxplot
    combined_boxplot([df1, df2], [["R_between_3-ERA5_LAND_and_1-C3S"],
                                  ["R_between_3-GLDAS_and_1-C3S"]],
                     output_dir=str(output_dir), title="Test Plot")

    # Check if the file exists using Path
    assert (
            output_dir / "combined_boxplot.png").exists(), "Boxplot file was not created."


def test_combined_boxplot_column_mismatch(tmp_path):
    """Test combined_boxplot with mismatched columns and DataFrames."""
    df1 = pd.DataFrame({"col1": [1, 2, 3]})
    df2 = pd.DataFrame({"col3": [7, 8]})
    output_dir = tmp_path

    with pytest.raises(ValueError,
                       match="The number of DataFrames must match the number of column name lists."):
        combined_boxplot([df1, df2], [["col1"], ["col3"], ["col4"]],
                         output_dir=str(output_dir))


# Test: validate_and_subset_data
def test_validate_and_subset_data_valid(sample_dataframe):
    """Test validate_and_subset_data with valid input."""
    df = sample_dataframe
    column_name = "R_between_3-ERA5_LAND_and_1-C3S"

    result, label = validate_and_subset_data(df, column_name)
    assert not result.empty, "No data was returned."
    assert label == column_name, "Label does not match column name."


def test_validate_and_subset_data_missing_column(sample_dataframe):
    """Test validate_and_subset_data with a missing column."""
    df = sample_dataframe

    with pytest.raises(ValueError,
                       match="Column 'missing_column' not found in the DataFrame."):
        validate_and_subset_data(df, "missing_column")


# Test: calculate_padded_extent
def test_calculate_padded_extent_valid(sample_dataframe):
    """Test calculate_padded_extent with valid input."""
    df = sample_dataframe

    extent = calculate_padded_extent(df, padding_fraction=0.1)
    assert len(extent) == 4, "Extent must return 4 values: min_lon, max_lon, min_lat, max_lat."





# Test: plot_static_map
def test_plot_static_map_valid(sample_dataframe, tmp_path):
    """Test plot_static_map with valid input."""
    df = sample_dataframe
    column_name = "R_between_3-ERA5_LAND_and_1-C3S"
    colormap = "coolwarm"
    plotsize = (10, 6)
    extent = (-180, 180, -90, 90)
    output_dir = str(tmp_path)

    plot_static_map(df, column_name, colormap, output_dir, plotsize, extent)
    assert (
            tmp_path / f"{column_name}_static_map.png").exists(), "Static map image file was not created."



# Test: map_plot
def test_map_plot_valid(sample_dataframe, tmp_path):
    """Test map_plot high-level function with a valid DataFrame."""
    df = sample_dataframe
    column_name = "R_between_3-ERA5_LAND_and_1-C3S"
    output_dir = str(tmp_path)

    map_plot(df, column_name, output_dir=output_dir, colormap="viridis")
    expected_output = tmp_path / f"{column_name}_static_map.png"
    assert expected_output.exists(), "Map plot file was not created."