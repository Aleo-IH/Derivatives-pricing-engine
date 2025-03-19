import QuantLib as ql
import numpy as np
import plotly.graph_objects as go
import pandas as pd


class Black_Variance_Surface:
    """
    A class for creating and visualizing Black-Scholes volatility surfaces from option market data.

    This class provides functionality to construct a QuantLib BlackVarianceSurface from
    option market data and visualize it as a 3D surface plot.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        expiration_col: str = "exercise_date",
        strike_col: str = "k",
        vol_col: str = "sigma",
    ) -> None:
        """
        Initialize the Black_Variance_Surface with option market data.

        Parameters:
            df (pd.DataFrame): DataFrame containing option market data.
            expiration_col (str): Name of the column containing expiration dates (as Python date objects).
            strike_col (str): Name of the column containing strike prices.
            vol_col (str): Name of the column containing implied volatilities.
        """
        self.df = df
        self.expiration_col = expiration_col
        self.strike_col = strike_col
        self.vol_col = vol_col

    def build_black_variance_surface(
        self,
        interpolation_method: str = "bicubic",
        pivot_agg_func=lambda x: np.mean(np.square(x)),
    ):
        """
        Constructs a QuantLib BlackVarianceSurface from the provided option market data.

        This method creates a volatility surface by organizing the data into a grid of
        strikes and expiration dates, filling in any missing values through interpolation,
        and constructing a QuantLib BlackVarianceSurface object.

        Parameters:
            interpolation_method (str): Interpolation method for the volatility surface (e.g., 'bicubic', 'bilinear').
            pivot_agg_func (callable): Function to aggregate volatility values when multiple values exist for the same strike/expiration.
                                      Default aggregates by mean of squared values.

        Returns:
            ql.BlackVarianceSurface: The constructed volatility surface object.
        """

        self.interpolation_method = interpolation_method
        self.pivot_agg_func = pivot_agg_func

        # Create a pivot table with strikes as rows, expiration dates as columns,
        # and volatilities as values, aggregating by the mean of squared values
        pivot = self.df.pivot_table(
            index=self.strike_col,
            columns=self.expiration_col,
            values=self.vol_col,
            aggfunc=self.pivot_agg_func,
        )
        # Get unique expiration dates and convert to QuantLib Dates.
        self.expirations = sorted(self.df[self.expiration_col].unique())

        self.ql_dates = [
            ql.Date(exp.day, exp.month, exp.year) for exp in self.expirations
        ]
        self.minDate, self.maxDate = self.ql_dates[0], self.ql_dates[-1]

        # Get unique strikes.
        self.strikes = sorted(self.df[self.strike_col].unique())

        self.minStrike, self.maxStrike = self.strikes[0], self.strikes[-1]

        pivot = pivot.reindex(index=self.strikes, columns=self.expirations)
        pivot = pivot.interpolate(method="linear", axis=0).interpolate(
            method="linear", axis=1
        )
        pivot = pivot.ffill().bfill()

        n_strike, n_exp = len(self.strikes), len(self.ql_dates)
        # Build a volatility matrix.
        vol_matrix = ql.Matrix(n_strike, n_exp)
        for i in range(n_strike):
            for j in range(n_exp):
                vol_matrix[i][j] = np.sqrt(pivot.iloc[i, j])

        self.vol_surface = ql.BlackVarianceSurface(
            ql.Settings.instance().evaluationDate,
            ql.TARGET(),
            self.ql_dates,
            self.strikes,
            vol_matrix,
            ql.Actual365Fixed(),
        )
        self.vol_surface.setInterpolation(interpolation_method)
        return self.vol_surface

    def plot_vol_surface(
        self,
        strike_min: float = 0,
        strike_max: float = float("inf"),
        num_strikes: int = 100,
        num_ttm: int = 100,
        date_axis: bool = False,
    ) -> None:
        """
        Generates a 3D surface plot of the volatility surface using Plotly.

        This method creates an interactive 3D visualization of the volatility surface,
        allowing for customization of the strike range, time-to-maturity range, and
        axis display options.

        Parameters:
            strike_min (float): Minimum strike price to display on the plot. Defaults to 0.
            strike_max (float): Maximum strike price to display on the plot. Defaults to infinity.
            num_strikes (int): Number of strike points to sample for the plot. Defaults to 100.
            num_ttm (int): Number of time-to-maturity points to sample for the plot. Defaults to 100.
            date_axis (bool): If True, displays actual calendar dates on the x-axis instead of time to maturity.
                             Defaults to False.

        Returns:
            None: Displays an interactive 3D surface plot in the current notebook or environment.
        """
        if not hasattr(self, "vol_surface"):
            _ = self.build_black_variance_surface()

        # Use the maximum of the input min_strike and the surface's min_strike
        effective_min_strike = max(strike_min, self.minStrike)
        # Use the minimum of the input max_strike and the surface's max_strike
        effective_max_strike = min(strike_max, self.maxStrike)
        strikes = np.linspace(effective_min_strike, effective_max_strike, num_strikes)

        today = ql.Settings.instance().evaluationDate

        max_ttm = ql.Actual365Fixed().yearFraction(today, self.maxDate)
        min_ttm = ql.Actual365Fixed().yearFraction(today, self.minDate)
        ttm_range = np.linspace(min_ttm, max_ttm, num_ttm)

        # Build a grid of volatilities using meshgrid
        self.strike_mesh, self.ttm_mesh = np.meshgrid(strikes, ttm_range, indexing="ij")
        vol_grid = np.zeros_like(self.strike_mesh)

        # Vectorized approach to calculate volatilities
        for i in range(self.strike_mesh.shape[0]):
            for j in range(self.strike_mesh.shape[1]):
                vol_val = self.vol_surface.blackVol(
                    self.ttm_mesh[i, j], self.strike_mesh[i, j]
                )
                vol_grid[i, j] = vol_val

        # If date_axis is True, convert ttm to actual dates
        if date_axis:
            date_mesh = np.zeros_like(self.ttm_mesh, dtype="datetime64[D]")
            base_date = np.datetime64(
                f"{self.minDate.year()}-{self.minDate.month():02d}-{self.minDate.dayOfMonth():02d}"
            )
            for i in range(self.ttm_mesh.shape[0]):
                for j in range(self.ttm_mesh.shape[1]):
                    days_to_add = int(self.ttm_mesh[i, j] * 365)
                    date_mesh[i, j] = base_date + np.timedelta64(days_to_add, "D")
            x_values = date_mesh
            x_axis_title = "Expiry Date"
        else:
            x_values = self.ttm_mesh
            x_axis_title = "Time to Maturity (Years)"

        # Create the 3D surface plot
        fig = go.Figure(
            data=[
                go.Surface(
                    x=x_values,
                    y=self.strike_mesh,
                    z=vol_grid,
                    colorscale="viridis",
                    colorbar=dict(title="Implied Volatility (%)"),
                )
            ]
        )

        # Set labels and title
        fig.update_layout(
            title="Implied Volatility Surface",
            scene=dict(
                xaxis_title=x_axis_title,
                yaxis_title="Strike Price",
                zaxis_title="Implied Volatility (%)",
            ),
            width=900,
            height=700,
            title_font_size=16,
            title_x=0.5,  # Center the title
            template="plotly_white",  # Clean white background with minimal grid lines
            hoverlabel=dict(
                bgcolor="white", font_size=12, font_family="Arial, sans-serif"
            ),
        )

        # Convert volatility values to percentages for display
        vol_grid_percent = vol_grid * 100

        # Update the z-values to show as percentages with appropriate hover template
        if date_axis:
            fig.update_traces(
                z=vol_grid_percent,
                hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br>"
                + "<b>Strike</b>: %{y:.2f}<br>"
                + "<b>Implied Vol</b>: %{z:.2f}%<br>"
                + "<extra></extra>",
            )
        else:
            fig.update_traces(
                z=vol_grid_percent,
                hovertemplate="<b>Time to Maturity</b>: %{x:.2f} years<br>"
                + "<b>Strike</b>: %{y:.2f}<br>"
                + "<b>Implied Vol</b>: %{z:.2f}%<br>"
                + "<extra></extra>",
            )

        fig.show()
        return None
