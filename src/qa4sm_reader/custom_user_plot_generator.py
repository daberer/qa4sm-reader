from typing import Optional, Tuple
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature

status = {
    -1: 'Other error',
    0: 'Success',
    1: 'Not enough data',
    2: 'Metric calculation failed',
    3: 'Temporal matching failed',
    4: 'No overlap for temporal match',
    5: 'Scaling failed',
    6: 'Unexpected validation error',
    7: 'Missing GPI data',
    8: 'Data reading failed'
}
metric_value_ranges = {  # from /qa4sm/validator/validation/graphics.py
    'R': [-1, 1],
    'p_R': [0, 1],  # probability that observed correlation is statistical fluctuation
    'rho': [-1, 1],
    'p_rho': [0, 1],
    'tau': [-1, 1],
    'p_tau': [0, 1],
    'RMSD': [0, None],
    'BIAS': [None, None],
    'n_obs': [0, None],
    'urmsd': [0, None],
    'RSS': [0, None],
    'mse': [0, None],
    'mse_corr': [0, None],
    'mse_bias': [0, None],
    'mse_var': [0, None],
    'snr': [None, None],
    'err_std': [None, None],
    'beta': [None, None],
    'status': [-1, len(status)-2],
    'slopeR': [None, None],
    'slopeURMSD': [None, None],
    'slopeBIAS': [None, None],
}



def combined_boxplot(dataframes: list, columnnames: list[list[str]],
                     output_dir: str,
                     title: Optional[str] = None):
    """
    Combines multiple DataFrames and creates a boxplot for specified columns.

    Parameters:
    - dataframes: list of pd.DataFrame, the DataFrames to include in the combined plot.
    - columnnames: list of list of str, where each sublist contains column names
                   to extract and include in the combined boxplot.
    - output_dir: str, path to save the plot.
    - title: Optional[str], title for the boxplot (default: None).
    """


    if len(dataframes) != len(columnnames):
        raise ValueError(
            "The number of DataFrames must match the number of column name lists.")

    # Step 1: Combine all data based on specified columns
    combined_data = []
    for df, cols in zip(dataframes, columnnames):
        for col in cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' is not in the DataFrame.")

            # Add the data along with a label indicating the source DataFrame/column
            combined_data.append(pd.DataFrame({
                "value": df[col].dropna(),
                # Drop NaN values for correct plotting
                "source": col  # Source label for grouping
            }))

    combined_df = pd.concat(combined_data, ignore_index=True)

    # Step 2: Create the boxplot
    plt.figure(figsize=(12, 8))
    combined_df.boxplot(by="source", column=["value"], grid=False,
                        return_type="axes")
    plt.suptitle("")  # Remove the default title added by pandas
    if title:
        plt.title(title)
    plt.xlabel("Source (Columns)")
    plt.ylabel("Values")
    plt.grid(visible=True, linestyle="--", alpha=0.7)

    # Save the plot
    output_path = f"{output_dir}/combined_boxplot.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Boxplot saved to {output_path}")


def validate_and_subset_data(df: pd.DataFrame, column_name: str) -> Tuple[
    pd.DataFrame, str]:
    """
    - column_name: str, column name to filter and subset data

    Parameters:
    - df: pd.DataFrame containing the data
    - metric: str, metric to filter and subset data

    Returns:
    - Tuple with filtered DataFrame and the column name to use as a label
    """
    required_columns = {'lat', 'lon'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    metric_data = None
    label = None

    # Filter DataFrame for relevant metric column
    # Subset DataFrame for the specified column
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
    # metric_data = df[["idx", "lat", "lon", column_name]].dropna()
    metric_data = df[["lat", "lon", column_name]].dropna()
    label = column_name

    if metric_data is None or metric_data.empty:
        raise ValueError(
            f"No data found for column '{column_name}' in the DataFrame.")

    # Define color for the metric values
    metric_data['color'] = metric_data[column_name].apply(
        lambda x: 'negative' if x < 0 else 'positive')
    return metric_data, label


def calculate_padded_extent(df: pd.DataFrame,
                            padding_fraction: float = 0.05) -> Tuple[
    float, float, float, float]:
    """
    Calculate the min/max latitude and longitude boundaries with padding.

    Parameters:
    - df: pd.DataFrame containing 'lat' and 'lon' columns.
    - padding_fraction: float, the fraction of range to use for padding.

    Returns:
    - Tuple: (min_lon_padded, max_lon_padded, min_lat_padded, max_lat_padded)
    """
    min_lon, max_lon = df["lon"].min(), df["lon"].max()
    min_lat, max_lat = df["lat"].min(), df["lat"].max()

    # Calculate padding
    padding_lat = (max_lat - min_lat) * padding_fraction
    padding_lon = (max_lon - min_lon) * padding_fraction

    # Apply padding
    return (min_lon - padding_lon, max_lon + padding_lon,
            min_lat - padding_lat, max_lat + padding_lat)


def plot_static_map(df: pd.DataFrame, column_name: str, colormap: str, output_dir: str,
                    plotsize: Tuple[float, float],
                    extent: Tuple[float, float, float, float],
                    value_range: Optional[Tuple[float, float]] = None,
                    metric: Optional[str] = None):
    """
    Plot the static map using Matplotlib and Cartopy.

    Parameters:
    - df: pd.DataFrame containing the metric data with 'lat' and 'lon'.
    - column_name: str, Column name of the metric to display.
    - colormap: str, Name of the colormap.
    - plotsize: tuple (width, height) for the figure size.
    - extent: tuple of padded lat/lon boundaries.
    """
    # Create a figure and an axis with a specific map projection
    fig, ax = plt.subplots(figsize=plotsize,
                           subplot_kw={'projection': ccrs.PlateCarree()})

    # Add geographical features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="black")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.gridlines(draw_labels=True)

    metric = next(
    (key for key in metric_value_ranges.keys() if column_name.startswith(key)),
    None)
    if value_range == None:
        v_min = metric_value_ranges[metric][0]
        v_max = metric_value_ranges[metric][1]
    else:
        v_min = value_range[0]
        v_max = value_range[1]
    # Plot data
    scatter = ax.scatter(
        df['lon'], df['lat'],
        c=df[column_name],  # Color based on column values
        cmap=colormap if colormap else 'coolwarm',  # Default colormap
        s=40,  # Marker size
        edgecolor='black',
        transform=ccrs.PlateCarree(),  # Ensures correct projection
        vmin=v_min,
        vmax=v_max,
    )

    # Set map extent
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Add colorbar
    cbar = plt.colorbar(scatter, orientation='horizontal', pad=0.05,
                        shrink=0.8, aspect=30)
    cbar.ax.tick_params(labelsize=12)  # Font size for tick labels
    cbar.set_label(column_name)

    # Title and Layout
    ax.set_title(f"Static Map Plot of {column_name}", fontsize=16)
    plt.tight_layout()

    # Save and display
    plt.savefig(output_dir + f"/{column_name}_static_map.png", dpi=300)
    plt.show()


def map_plot(df, column_name: str, output_dir: str,
             colormap: Optional[str] = None,
             value_range: Optional[Tuple[float, float]] = None,
             plotsize: Optional[Tuple[float, float]] = (16, 10)):
    """
    High-level function to plot a static map with data from a specific column
    and Cartopy features.

    Parameters:
    - df: pd.DataFrame, the DataFrame containing the data.
    - column_name: str, the name of the column to process and plot.
    - output_dir: str, the directory to save the output.
    - colormap: Optional[str], the colormap to use for the plot.
    - value_range: Optional[Tuple[float, float]], range of values for color mapping.
    - plotsize: Optional[Tuple[float, float]], size of the plot (default: (16, 10)).
    """
    # Step 1: Validate and Prepare Data
    data, label = validate_and_subset_data(df, column_name)

    # Step 2: Calculate Map Extent with Padding
    extent = calculate_padded_extent(data)

    # Step 3: Plot the Static Map
    plot_static_map(data, label, colormap, output_dir, plotsize, extent=extent,
                    value_range=value_range, metric=column_name)









