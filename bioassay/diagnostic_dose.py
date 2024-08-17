import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def analyze_diagnostic_dose(df, groupby_columns, total_column='total', dead_column='dead'):
    """
    Analyze diagnostic-dose testing data.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the diagnostic dose data
    groupby_columns (list): List of column names to group by (e.g., ['location', 'strain'])
    total_column (str): Name of the column containing total number of insects
    dead_column (str): Name of the column containing number of dead insects

    Returns:
    pandas.DataFrame: Results of the analysis
    """
    def calculate_ci(row):
        n = row[total_column]
        p = row[dead_column] / n
        se = np.sqrt(p * (1 - p) / n)
        ci = stats.t.interval(0.95, n-1, loc=p, scale=se)
        return pd.Series({'mean_mortality': p * 100, 'ci_lower': ci[0] * 100, 'ci_upper': ci[1] * 100})

    # Group the data and calculate mean mortality and confidence intervals
    results = df.groupby(groupby_columns).apply(calculate_ci).reset_index()
    
    return results

def plot_diagnostic_dose(results, x_column, color_column=None, title="Diagnostic Dose Mortality"):
    """
    Plot the results of diagnostic-dose testing.

    Parameters:
    results (pandas.DataFrame): Results from analyze_diagnostic_dose function
    x_column (str): Name of the column to use for x-axis categories
    color_column (str, optional): Name of the column to use for color-coding bars
    title (str): Title of the plot

    Returns:
    plotly.graph_objs._figure.Figure: Plotly figure object
    """
    if color_column is None:
        color_column = x_column

    # Create figure
    fig = go.Figure()

    # Add bars for each group
    for group in results[color_column].unique():
        group_data = results[results[color_column] == group]
        fig.add_trace(go.Bar(
            x=group_data[x_column],
            y=group_data['mean_mortality'],
            name=group,
            error_y=dict(
                type='data',
                symmetric=False,
                array=group_data['ci_upper'] - group_data['mean_mortality'],
                arrayminus=group_data['mean_mortality'] - group_data['ci_lower']
            )
        ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title="Mortality (%)",
        barmode='group',
        legend_title=color_column
    )

    return fig
