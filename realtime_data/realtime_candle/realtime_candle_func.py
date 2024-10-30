import polars as pl

def make_realtime_candle(df, tf_list, symbol):
    # opt
    for tf_int in tf_list:
        tf_str = str(tf_int)
        df = (
            df.set_sorted("_time")
            .rolling(index_column="_time", period=tf_str + "m")
            .agg(
                [
                    pl.all().exclude("_time").last(),
                    pl.col(f"M5_OPEN")
                    .slice(
                        pl.arg_where(
                            (pl.col("minutesPassed") % tf_int == 0)
                            | (pl.col("isFirst") == 1)
                        ).last()
                    )
                    .first()
                    .alias("M" + tf_str + "_OPEN"),
                    pl.col(f"M5_HIGH")
                    .slice(
                        pl.arg_where(
                            (pl.col("minutesPassed") % tf_int == 0)
                            | (pl.col("isFirst") == 1)
                        ).last()
                    )
                    .max()
                    .alias("M" + tf_str + "_HIGH"),
                    pl.col(f"M5_LOW")
                    .slice(
                        pl.arg_where(
                            (pl.col("minutesPassed") % tf_int == 0)
                            | (pl.col("isFirst") == 1)
                        ).last()
                    )
                    .min()
                    .alias("M" + tf_str + "_LOW"),
                    pl.col(f"M5_CLOSE")
                    .slice(
                        pl.arg_where(
                            (pl.col("minutesPassed") % tf_int == 0)
                            | (pl.col("isFirst") == 1)
                        ).last()
                    )
                    .last()
                    .alias("M" + tf_str + "_CLOSE"),
                    pl.col(f"M5_VOLUME")
                    .slice(
                        pl.arg_where(
                            (pl.col("minutesPassed") % tf_int == 0)
                            | (pl.col("isFirst") == 1)
                        ).last()
                    )
                    .sum()
                    .alias("M" + tf_str + "_VOLUME"),
                ]
            )
        )
    return df
