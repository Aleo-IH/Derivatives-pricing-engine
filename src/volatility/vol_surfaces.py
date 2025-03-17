import QuantLib as ql
import numpy as np
import plotly.graph_objects as go
import pandas as pd

def build_black_variance_surface(df : pd.DataFrame, expiration_col : str ='exercise_date', strike_col : str ='k',
                                 vol_col : str ='sigma', interpolation_method : str ='bicubic'):
    """
    Constructs a QuantLib BlackVarianceSurface from a DataFrame with option market data.
    
    The DataFrame must have columns for expiration dates (as Python date objects),
    strike prices, and implied volatilities.
    
    Parameters:
        df (pd.DataFrame): Option market data.
        expiration_col (str): Name of the expiration date column.
        strike_col (str): Name of the strike price column.
        vol_col (str): Name of the implied volatility column.
        interpolation_method (str): Interpolation method (e.g., 'bicubic').
    
    Returns:
        ql.BlackVarianceSurface: The constructed volatility surface.
    """
    # Create a pivot table with strikes as rows, expiration dates as columns,
    # and volatilities as values, aggregating by the mean of squared values
    pivot = df.pivot_table(index=strike_col, columns=expiration_col, values=vol_col, 
                          aggfunc=lambda x: np.mean(np.square(x)))
    # Get unique expiration dates and convert to QuantLib Dates.
    expirations = sorted(df[expiration_col].unique())

    ql_dates = [ql.Date(exp.day, exp.month, exp.year) for exp in expirations]
    # Get unique strikes.
    strikes = sorted(df[strike_col].unique())
    pivot = pivot.reindex(index=strikes, columns=expirations)
    pivot = pivot.interpolate(method='linear', axis=0).interpolate(method='linear', axis=1)
    pivot = pivot.ffill().bfill()
    

    n_strike, n_exp = len(strikes), len(ql_dates)
    # Build a volatility matrix.
    vol_matrix = ql.Matrix(n_strike, n_exp)
    for i in range(n_strike):
        for j in range(n_exp):
            vol_matrix[i][j] = np.sqrt(pivot.iloc[i,j])
    
    vol_surface = ql.BlackVarianceSurface(
        ql.Settings.instance().evaluationDate, ql.TARGET(), ql_dates, strikes, vol_matrix, ql.Actual365Fixed()
    )
    vol_surface.setInterpolation(interpolation_method)
    return vol_surface


def plot_vol_surface(vol_surface, strike_min  : float = 0, strike_max : float = float('inf'), num_strikes : int =100, num_ttm : int =100):
    """
    Generates a 3D surface plot of a QuantLib BlackVarianceSurface using Plotly.
    
    Parameters:
        vol_surface (ql.BlackVarianceSurface): The volatility surface.
        strike_min (float): Minimum strike price for the plot.
        strike_max (float): Maximum strike price for the plot.
        num_strikes (int): Number of strike points.
        num_ttm (int): Number of time-to-maturity points.
    
    Returns:
        plotly.graph_objects.Figure: The 3D surface plot.
    """

    # Use the maximum of the input min_strike and the surface's min_strike
    effective_min_strike = max(strike_min, vol_surface.minStrike())
    # Use the minimum of the input max_strike and the surface's max_strike
    effective_max_strike = min(strike_max, vol_surface.maxStrike())
    strikes = np.linspace(effective_min_strike, effective_max_strike, num_strikes)
    
    today = ql.Settings.instance().evaluationDate

    max_ttm = ql.Actual365Fixed().yearFraction(today, vol_surface.maxDate())
    ttm_range = np.linspace(0.01, max_ttm, num_ttm)
    
    # Build a grid of volatilities using meshgrid
    strike_mesh, ttm_mesh = np.meshgrid(strikes, ttm_range, indexing='ij')
    vol_grid = np.zeros_like(strike_mesh)
    
    # Vectorized approach to calculate volatilities
    for i in range(strike_mesh.shape[0]):
        for j in range(strike_mesh.shape[1]):
            vol_val = vol_surface.blackVol(ttm_mesh[i, j], strike_mesh[i, j])
            vol_grid[i, j] = vol_val
    
    # Create the 3D surface plot
    fig = go.Figure(data=[go.Surface(
        x=ttm_mesh, 
        y=strike_mesh, 
        z=vol_grid,
        colorscale='viridis',
        colorbar=dict(title='Implied Volatility (%)')
    )])

    # Set labels and title
    fig.update_layout(
        title='Implied Volatility Surface',
        scene=dict(
            xaxis_title='Time to Maturity (Years)',
            yaxis_title='Strike Price',
            zaxis_title='Implied Volatility (%)',
        ),
        width=900,
        height=700,
        title_font_size=16,
        title_x=0.5,  # Center the title
        template="plotly_white",  # Clean white background with minimal grid lines
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial, sans-serif"
        )
    )
    
    # Convert volatility values to percentages for display
    vol_grid_percent = vol_grid * 100
    
    # Update the z-values to show as percentages
    fig.update_traces(
        z=vol_grid_percent,
        hovertemplate="<b>Time to Maturity</b>: %{x:.2f} years<br>" +
                      "<b>Strike</b>: %{y:.2f}<br>" +
                      "<b>Implied Vol</b>: %{z:.2f}%<br>"+"<extra></extra>"
    )
    fig.show()
    # Show the plot
    return vol_grid, strike_mesh, ttm_mesh