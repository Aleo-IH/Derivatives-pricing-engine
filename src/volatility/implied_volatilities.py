from pandas import DataFrame
from ..pricing_models.tree_model import Tree_USVanilla


def get_USImpliedVol(
    df: DataFrame, end_col: str = "calculated_col", verbose: bool = True
):
    """
    Calculate implied volatility for US vanilla options and store results in the dataframe.

    This function iterates through each row of the input dataframe, creates a Tree_USVanilla
    option model for each option, and calculates its implied volatility based on the market price.
    Missing values are filled using forward and backward fill methods.

    Parameters:
        df (DataFrame): DataFrame containing option data with required fields for Tree_USVanilla model
        end_col (str): Column name where implied volatility values will be stored. Defaults to 'calculated_col'
        verbose (bool): If True, prints information about the number of failed calculations. Defaults to True

    Returns:
        DataFrame: The input dataframe with the added implied volatility column
    """
    df[end_col] = 0.0
    counter = 0
    for i, row in df.iterrows():
        option, bsm_process = Tree_USVanilla(row.to_dict())
        try:
            implied_vol = option.impliedVolatility(row["lastPrice"], bsm_process)
        except Exception:
            implied_vol = float("nan")
            counter += 1
        df.loc[i, end_col] = implied_vol

    df.loc[:, end_col] = df[end_col].ffill().bfill()
    if verbose:
        print(
            f"Implied volatility calculation failed for {counter/i:.2%} of values and was filled with interpolation"
        )
    return df
