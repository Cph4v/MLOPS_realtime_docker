import polars as pl

#!!: add truncate first rows
def shift_by_params(periods, time_frame, col_name=None):
    """
    this function returns shifted feature
    when col_name == none it returns shift indices
    """
    min_valid_index = periods * (time_frame // 5)
    df_truncated = pl.col("index") >= min_valid_index

    idx = df_truncated - (
        ((pl.col("minutesPassed") % time_frame) / 5)
        + (periods - 1) * (time_frame / 5)
        + 1
    )

    validated_idx = pl.when(idx > 0).then(idx).otherwise(0).cast(pl.Int32, strict=False)

    if not (col_name):
        return validated_idx
    target_col = (pl.col(col_name).gather(validated_idx)).alias(
        f"{col_name}_-{periods}"
    )
    return target_col


def create_shifted_col(
    df, pair_name, periods, time_frame, columns=["OPEN", "HIGH", "LOW", "CLOSE"]
):  
    
    # print(f"-----> df.col: {df.columns}")
    return df.with_columns(
        [
            shift_by_params(
                periods=periods,
                time_frame=time_frame,
                col_name=f"M{time_frame}_{col}",
            )
            for col in columns
        ]
    )


def drop_first_day_polars(df):
    # Extract the first date from the "_time" column
    first_date = df.select(pl.col("_time").dt.date()).row(0)[0]

    # Filter the DataFrame to drop rows with the first date
    df_filtered = df.filter(pl.col("_time").dt.date() != first_date)

    return df_filtered

