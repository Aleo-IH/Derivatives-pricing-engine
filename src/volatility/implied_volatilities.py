from pandas import DataFrame
from ..pricing_models.tree_model import Tree_USVanilla


def get_USImpliedVol(df: DataFrame):
    df["calculated_vol"] = 0.0
    counter = 0
    for i, row in df.iterrows():
        option, bsm_process = Tree_USVanilla(row.to_dict())
        try:
            implied_vol = option.impliedVolatility(row["lastPrice"], bsm_process)
        except Exception:
            implied_vol = float("nan")
            counter += 1
        df.loc[i, "calculated_vol"] = implied_vol

    df.loc[:, "calculated_vol"] = df["calculated_vol"].ffill().bfill()
    print(f"df filed nans for {counter/i : .2%} of values")
    return df
