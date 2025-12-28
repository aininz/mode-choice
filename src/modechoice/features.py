import numpy as np
import pandas as pd

def add_relative_features_long(
    df_long: pd.DataFrame,
    w_ovt=2.0,
    freq_period_minutes=1440.0,  # 1440 if freq per day; 60 if per hour
) -> pd.DataFrame:
    df = df_long.copy()

    df["cost_log"] = np.log(df["cost"].clip(lower=1e-6))

    df["wait_time"] = freq_period_minutes / (2.0 * df["freq"].replace(0, np.nan))
    df["wait_time"] = df["wait_time"].fillna(freq_period_minutes)

    df["gen_time"] = df["ivt"] + w_ovt * df["ovt"] + df["wait_time"]

    g = df.groupby("case", sort=False)
    df["min_cost_case"] = g["cost"].transform("min")
    df["min_gen_time_case"] = g["gen_time"].transform("min")

    df["rel_cost"] = df["cost"] - df["min_cost_case"]
    df["rel_gen_time"] = df["gen_time"] - df["min_gen_time_case"]

    # optional interactions (safe)
    if "income" in df.columns:
        df["income_x_rel_cost"] = df["income"] * df["rel_cost"]
    if "urban" in df.columns:
        df["urban_x_ovt"] = df["urban"] * df["ovt"]

    return df