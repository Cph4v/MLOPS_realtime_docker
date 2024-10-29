
import polars as pl
include_target = False
import polars as pl
from collections import defaultdict


# Define the prefixes - !!! order maters.
prefixes = [
    "fe_cndl_shift",
    "fe_cndl",
    "fe_WIN_argmin",
    "fe_WIN_argmax",
    "fe_WIN_max",
    "fe_WIN_min",
    "fe_WIN",
    "fe_ratio_RSI",
    "fe_ratio_EMA",
    "fe_ratio_RSTD",
    "fe_ratio_SMA",
    "fe_ratio_ATR",
    "fe_ratio",
    "fe_RSTD",
    "fe_RSI",
    "fe_EMA",
    "fe_SMA",
    "fe_ATR",
    "fe_market_close",
]


# Function to categorize keys
def categorize_keys(data, prefixes=prefixes):
    categorized = defaultdict(list)
    uncategorized = []

    for key in data.keys():
        categorized_flag = False
        for prefix in prefixes:
            if key.startswith(prefix):
                categorized[prefix].append(key)
                categorized_flag = True
                break
        if not categorized_flag:
            uncategorized.append(key)

    return categorized, uncategorized

def add_symbol(col_name, symbol, prefixes):
    for prefix in prefixes:
        if col_name.startswith(prefix):
            new_col_name = col_name.replace(prefix, f"{prefix}_{symbol}")
            return new_col_name
    return col_name

def add_symbol_to_prefixes(df_cols, symbol, prefixes=prefixes):
    """
    Adds a given symbol after specific prefixes in the column names of a Polars DataFrame.

    Parameters:
    df (pl.DataFrame): The Polars DataFrame.
    symbol (str): The symbol to add after the prefixes.
    prefixes (list): List of known prefixes to search for in the column names.

    Returns:
    pl.DataFrame: The DataFrame with updated column names.
    """

    new_column_names = {col: add_symbol(col, symbol, prefixes) for col in df_cols}

    return new_column_names

def group_by_symbol_and_rename(df, symbol_col="symbol"):
    """
    Groups the DataFrame by the symbol column, removes the symbol column, and appends the symbol value to each feature column name.

    Parameters:
    df (pl.DataFrame): The Polars DataFrame.
    symbol_col (str): The name of the column containing the symbols.

    Returns:
    pl.DataFrame: The DataFrame with updated column names and without the symbol column.
    """
    partitions = df.partition_by(symbol_col)

    # Initialize an empty list to store the renamed partitions
    renamed_partitions = []

    # Process each partition
    for partition in partitions:
        # Get the symbol value from the first row (all rows have the same symbol in the partition)
        symbol = partition[symbol_col][0]
        renamed_partition = partition.rename(
            add_symbol_to_prefixes(partition.columns, symbol)
        )
        renamed_partition = renamed_partition.drop(symbol_col)

        # Add the renamed partition to the list
        renamed_partitions.append(renamed_partition)

    # Combine all the renamed partitions on the _time column
    result_df = renamed_partitions[0]
    for renamed_partition in renamed_partitions[1:]:
        result_df = result_df.join(
            renamed_partition, on="_time", how="full", coalesce=True
        )

    return result_df
