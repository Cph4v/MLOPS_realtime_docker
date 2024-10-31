import polars as pl
import pandas as pd
from configs.history_data_crawlers_config import symbols_dict

def add_candle_features(
    df, symbol: str, tf_list=[5, 15, 60, 120, 240, 1440], fe_prefix="fe_cndl"
):
    org_columns = set(df.columns)
    org_columns.remove("_time")
    for tf in tf_list:
        df = df.with_columns(
            [
                (
                    (pl.col(f"M{tf}_CLOSE") - pl.col(f"M{tf}_OPEN"))
                    / symbols_dict[symbol]["pip_size"]
                ).alias(f"{fe_prefix}_M{tf}_CLOSE_to_OPEN"),
                (
                    (pl.col(f"M{tf}_HIGH") - pl.col(f"M{tf}_LOW"))
                    / symbols_dict[symbol]["pip_size"]
                ).alias(f"{fe_prefix}_M{tf}_HIGH_to_LOW"),
                (
                    (pl.col(f"M{tf}_HIGH") - pl.col(f"M{tf}_CLOSE"))
                    / symbols_dict[symbol]["pip_size"]
                ).alias(f"{fe_prefix}_M{tf}_HIGH_to_CLOSE"),
                (
                    (pl.col(f"M{tf}_HIGH") - pl.col(f"M{tf}_OPEN"))
                    / symbols_dict[symbol]["pip_size"]
                ).alias(f"{fe_prefix}_M{tf}_HIGH_to_OPEN"),
                (
                    (pl.col(f"M{tf}_OPEN") - pl.col(f"M{tf}_LOW"))
                    / symbols_dict[symbol]["pip_size"]
                ).alias(f"{fe_prefix}_M{tf}_OPEN_to_LOW"),
                (
                    (pl.col(f"M{tf}_CLOSE") - pl.col(f"M{tf}_LOW"))
                    / symbols_dict[symbol]["pip_size"]
                ).alias(f"{fe_prefix}_M{tf}_CLOSE_to_LOW"),
            ]
        )

    df = df.drop(org_columns)

    return df

def add_shifted_candle_features(
    df, tf, shift_sizes=[1], fe_prefix="fe_cndl_shift"
):
    org_columns = set(df.columns)
    org_columns.remove("_time")
    for sh in shift_sizes:
        df = df.with_columns(
            [
                (pl.col(f"M{tf}_CLOSE_-{sh}") - pl.col(f"M{tf}_OPEN_-{sh}")).alias(
                    f"{fe_prefix}_M{tf}_CLOSE_to_OPEN_-{sh}"
                ),
                (pl.col(f"M{tf}_HIGH_-{sh}") - pl.col(f"M{tf}_LOW_-{sh}")).alias(
                    f"{fe_prefix}_M{tf}_HIGH_to_LOW_-{sh}"
                ),
                (pl.col(f"M{tf}_HIGH_-{sh}") - pl.col(f"M{tf}_CLOSE_-{sh}")).alias(
                    f"{fe_prefix}_M{tf}_HIGH_to_CLOSE_-{sh}"
                ),
                (pl.col(f"M{tf}_HIGH_-{sh}") - pl.col(f"M{tf}_OPEN_-{sh}")).alias(
                    f"{fe_prefix}_M{tf}_HIGH_to_OPEN_-{sh}"
                ),
                (pl.col(f"M{tf}_OPEN_-{sh}") - pl.col(f"M{tf}_LOW_-{sh}")).alias(
                    f"{fe_prefix}_M{tf}_OPEN_to_LOW_-{sh}"
                ),
                (pl.col(f"M{tf}_CLOSE_-{sh}") - pl.col(f"M{tf}_LOW_-{sh}")).alias(
                    f"{fe_prefix}_M{tf}_CLOSE_to_LOW_-{sh}"
                ),
            ]
        )

    df = df.drop(*org_columns)

    return df

