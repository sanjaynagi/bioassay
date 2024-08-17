import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Binomial

def calculate_lc_probit(df, lc_value=50, by_strain=True):
    """
    Calculate LC value using probit analysis for each strain or the entire dataset.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing columns: strain, concentration, total, dead
    lc_value (float): LC value to calculate (default 50 for LC50)
    by_strain (bool): Whether to calculate LC values for each strain separately (default True)
    
    Returns:
    dict: Dictionary containing LC values and confidence intervals for each strain (or overall if by_strain=False)
    """
    
    def calculate_lc(data):
        # Calculate proportion dead
        data['proportion'] = data['dead'] / data['total']
        
        # Fit probit model
        model = glm("proportion ~ concentration", data=data, family=Binomial(link=stats.probit))
        result = model.fit()
        
        # Calculate LC value
        lc = np.exp((stats.norm.ppf(lc_value/100) - result.params[0]) / result.params[1])
        
        # Calculate confidence interval
        se = np.sqrt(np.diag(result.cov_params()))
        ci_lower = np.exp((stats.norm.ppf(lc_value/100) - result.params[0] - 1.96*se[0]) / (result.params[1] + 1.96*se[1]))
        ci_upper = np.exp((stats.norm.ppf(lc_value/100) - result.params[0] + 1.96*se[0]) / (result.params[1] - 1.96*se[1]))
        
        return {
            f"LC{lc_value}": lc,
            "CI_lower": ci_lower,
            "CI_upper": ci_upper
        }
    
    if by_strain:
        results = {}
        for strain in df['strain'].unique():
            strain_data = df[df['strain'] == strain]
            results[strain] = calculate_lc(strain_data)
    else:
        results = calculate_lc(df)
    
    return results




import plotly.express as px

def plot_lc_probit(df, lc_results, lc_value=50):
    """
    Plot probit analysis results.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing columns: strain, concentration, total, dead
    lc_results (dict): Dictionary of LC results from calculate_lc_probit function
    lc_value (float): LC value that was calculated (default 50 for LC50)
    
    Returns:
    plotly.graph_objs._figure.Figure: Plotly figure object
    """
    
    # Calculate proportion dead
    df['proportion'] = df['dead'] / df['total']
    
    # Create the plot
    fig = px.scatter(df, x='concentration', y='proportion', color='strain',
                     labels={'concentration': 'Concentration', 'proportion': 'Proportion Dead'},
                     title=f'Probit Analysis - LC{lc_value}')
    
    # Add fitted lines for each strain
    for strain in df['strain'].unique():
        strain_data = df[df['strain'] == strain]
        
        # Fit probit model
        model = glm("proportion ~ concentration", data=strain_data, family=Binomial(link=stats.probit))
        result = model.fit()
        
        # Generate points for the fitted line
        x_range = np.linspace(strain_data['concentration'].min(), strain_data['concentration'].max(), 100)
        y_fitted = result.predict(pd.DataFrame({'concentration': x_range}))
        
        # Add the fitted line
        fig.add_scatter(x=x_range, y=y_fitted, mode='lines', name=f'{strain} (fitted)',
                        line=dict(dash='dash'))
    
    # Add LC value annotations
    for strain, results in lc_results.items():
        lc = results[f"LC{lc_value}"]
        ci_lower = results["CI_lower"]
        ci_upper = results["CI_upper"]
        
        fig.add_annotation(x=lc, y=lc_value/100,
                           text=f"{strain}<br>LC{lc_value} = {lc:.2f}<br>95% CI: ({ci_lower:.2f}, {ci_upper:.2f})",
                           showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#636363",
                           ax=20, ay=-30, bordercolor="#c7c7c7", borderwidth=2, borderpad=4,
                           bgcolor="#ff7f0e", opacity=0.8)
    
    return fig
