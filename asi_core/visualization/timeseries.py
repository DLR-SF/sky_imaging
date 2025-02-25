# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This module provides functions to visualize timeseries data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fastcore.basics import ifnone


def heatmap_missing_data(missing_timestamps, start_date=None, end_date=None, hour_min=6, hour_max=20, y_dim='hour',
                         x_dim='date', title=None, interactive=False):
    """
    Generate a heatmap displaying the distribution of missing data over time.

    :param missing_timestamps: A pandas Series of timestamps representing missing data occurrences.
    :type missing_timestamps: pandas.Series
    :param start_date: The start of the time range for the heatmap. Defaults to the earliest timestamp in `missing_timestamps`.
    :type start_date: str or datetime, optional
    :param end_date: The end of the time range for the heatmap. Defaults to the latest timestamp in `missing_timestamps`.
    :type end_date: str or datetime, optional
    :param hour_min: The minimum hour of the day to include in the heatmap (default is 6).
    :type hour_min: int, optional
    :param hour_max: The maximum hour of the day to include in the heatmap (default is 20).
    :type hour_max: int, optional
    :param y_dim: The variable to use for the y-axis (default is 'hour').
    :type y_dim: str, optional
    :param x_dim: The variable to use for the x-axis (default is 'date').
    :type x_dim: str, optional
    :param title: The title of the heatmap. Defaults to 'Heatmap of Missing Data'.
    :type title: str, optional
    :param interactive: If True, generates an interactive heatmap using `hvplot`; otherwise, generates a static heatmap using `seaborn`.
    :type interactive: bool, optional
    :return: A heatmap figure displaying missing data patterns over time.
    :rtype: matplotlib.figure.Figure or holoviews.element.HeatMap
    """
    # Convert missing_data_dict to a single DataFrame
    start_date = ifnone(start_date, missing_timestamps.date.min())
    end_date = ifnone(end_date, missing_timestamps.date.max())
    missing_df = pd.DataFrame({'timestamp': missing_timestamps})
    missing_df['date'] = missing_df['timestamp'].dt.date
    missing_df['hour'] = missing_df['timestamp'].dt.hour
    missing_df['minute'] = missing_df['timestamp'].dt.minute
    tz = missing_df['timestamp'].dt.tz

    # Aggregate missing counts by the desired interval
    missing_df['time_bin'] = missing_df['timestamp'].dt.floor('1h')
    aggregated = missing_df.groupby('time_bin').size().reset_index(name='missing_count')

    # Create a full time range to fill gaps
    time_range = pd.date_range(start=start_date, end=end_date, freq='1h', tz=tz)
    aggregated = aggregated.set_index('time_bin').reindex(time_range, fill_value=0)
    aggregated.reset_index(inplace=True)
    aggregated.rename(columns={'index': 'time_bin'}, inplace=True)

    # Add day and hour for heatmap indexing
    aggregated['date'] = aggregated['time_bin'].dt.date
    aggregated['hour'] = aggregated['time_bin'].dt.hour

    # Filter for hours within the specified range again (to handle reindexing gaps)
    aggregated = aggregated[(aggregated['hour'] >= hour_min) & (aggregated['hour'] <= hour_max)]

    # Create pivot table for heatmap
    heatmap_data = aggregated.pivot_table(index=y_dim, columns=x_dim, values='missing_count', fill_value=0)

    title = ifnone(title, 'Heatmap of Missing Data')

    if interactive:
        # Interactive visualization with hvplot
        heatmap_fig = heatmap_data.hvplot.heatmap(
            x=x_dim,
            y=y_dim,
            C='value',
            cmap='YlOrRd',
            colorbar=True,
            title=title
        )
    else:
        # Static visualization with matplotlib and seaborn
        heatmap_fig, ax = plt.subplots(1, figsize=(15, 8))
        sns.heatmap(heatmap_data, cmap='YlOrRd', cbar_kws={'label': 'Missing Entries'}, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Date')
        ax.invert_yaxis()
        heatmap_fig.tight_layout()
    return heatmap_fig


def plot_data_distributions(df, columns, n_rows, df_ref=None, figsize=(20, 12), title=None):
    """
    Plot histogram of data in subplots.

    :param df: dataframe containing timeseries data.
    :param columns: list of column names.
    :param n_rows: number of rows in figure.
    :param df_ref: reference dataframe (requires the same columns).
    :param figsize: figure size.
    :param title: title of figure.
    :return: figure object.
    """
    if type(columns) is str:
        columns = [columns]
    n_plots = len(columns)
    n_cols = int(np.ceil(n_plots / n_rows))
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i, col in enumerate(columns):
        df_plot = df[[col]].copy()
        if n_rows == 1 and n_cols == 1:
            ax_i = ax
        elif n_rows > 1 and n_cols > 1:
            ax_i = ax[int(np.floor(i / n_cols)), i % n_cols]
        else:
            ax_i = ax[i]
        if df_ref is not None:
            df_plot['data'] = 'original'
            df_plot = pd.concat([df_plot, pd.DataFrame({col: df_ref[col], 'data': 'reference'})], axis=0,
                                ignore_index=True)
            sns.histplot(df_plot, y=col, ax=ax_i, hue='data')
        else:
            sns.histplot(df, y=col, ax=ax_i)
    if title is not None:
        fig.suptitle(title)
    return fig
