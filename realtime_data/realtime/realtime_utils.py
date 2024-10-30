import numpy as np
import pandas as pd
import polars as pl
import glob
import shutil
from pathlib import Path
import time as tt
import pytz
from polars import DataFrame as pl_DataFrame
from create_dataset.columns_merge_func import add_symbol_to_prefixes
from configs.history_data_crawlers_config import root_path, symbols_dict
from utils.clean_data import remove_weekends
from create_features.indicator_func import (
    cal_RSI_base_func,
    cal_EMA_base_func,
    cal_SMA_base_func,
    cal_ATR_func,
    cal_RSTD_func,
    add_candle_base_indicators_polars,
    add_all_ratio_by_config,
)
from create_features.create_basic_features_func import (
    add_candle_features,
    add_shifted_candle_features,
)
from utils.df_utils import (
    generate_true_time_df_pandas,
)
from realtime_candle.realtime_candle_func import (
    make_realtime_candle,
)

from data_crawlers.metatrader_func import (
    crawl_OHLCV_data_metatrader_one_symbol,
)

from create_features.realtime_shift_func import create_shifted_col
from create_features.window_agg_features_func import add_win_fe_base_func
from feature_engine.datetime import DatetimeFeatures
from datetime import datetime, time, timedelta


def get_vantage_time_now():
    return (datetime.now(tz=pytz.timezone("US/Eastern")) + timedelta(hours=7)).replace(
        tzinfo=None
    )


def is_forex_market_open():
    now = get_vantage_time_now()  # Get the current broker time
    weekday = now.weekday()  # Monday is 0 and Sunday is 6

    if weekday == 5 or weekday == 6:
        return False  # Saturday or Sunday
    else:
        return True  # Market is open


def sleep_until_next_run(run_every=5, offset_seconds=1, reporter=None):
    sleep_until_market_opens(reporter)
    now = get_vantage_time_now()
    next_run = now + (datetime.min - now) % timedelta(minutes=run_every)
    wait_seconds = (next_run - now).total_seconds() + offset_seconds
    reporter.send_message(
        f"--> sleep for {wait_seconds} seconds for next candle"
    )
    now = get_vantage_time_now()
    next_run = now + (datetime.min - now) % timedelta(minutes=run_every)
    wait_seconds = (next_run - now).total_seconds() + offset_seconds
    tt.sleep(wait_seconds)

    now = get_vantage_time_now()
    assert (
        now.minute % run_every == 0 and now.second == offset_seconds
    ), "!!! Wrong timing."
    reporter.send_message(f"Function executed at {now}")


def sleep_until_market_opens(reporter):
    if is_forex_market_open():
        reporter.send_message("Forex market is already open.")
        return

    now = get_vantage_time_now()
    weekday = now.weekday()

    if weekday == 5:  # Saturday
        # Sleep until 00:00 AM on Sunday
        open_time = (now + timedelta(days=2)).replace(
            hour=0, minute=5, second=0, microsecond=0
        )
    elif weekday == 6:  # Sunday before market opens
        # Sleep until 00:00 AM on Sunday
        open_time = (now + timedelta(days=1)).replace(
            hour=0, minute=5, second=0, microsecond=0
        )
    else:
        # The market is open on weekdays, no need to sleep
        return

    # Calculate the time to sleep
    time_to_sleep = (open_time - now).total_seconds() + 60 * 5
    reporter.send_message(
        f"Sleeping for {round(time_to_sleep/60,2)} minuts until the forex market opens."
    )
    reporter.send_message(
        f"market open_time will be: {open_time}."
    )
    tt.sleep(time_to_sleep)
    reporter.send_message("Forex market is now open.")


def merge_mean_dfs(df1, df2, mean_cols):
    merged_df = df1.merge(df2, on=["_time"], how="outer")
    for col in mean_cols:
        merged_df[col] = merged_df[[col + "_x", col + "_y"]].mean(axis=1)

    merged_df = merged_df.drop(
        columns=[col + "_x" for col in mean_cols] + [col + "_y" for col in mean_cols]
    )
    return merged_df


def realtime_stage_one(df):
    # ? remove weekends:
    df = remove_weekends(df, weekends_day=["Saturday", "Sunday"], convert_tz=False)

    # print(df.isnull().sum())
    # ? check for null and inf:
    assert df.isnull().sum().sum() == 0, "DataFrame contains null values"
    assert np.isfinite(df).all().all(), "DataFrame contains infinite values"

    # ? Check order of OHLC makes sense
    assert (df["Open"] <= df["High"]).all(), "Open higher than high"
    assert (df["Open"] >= df["Low"]).all(), "Open lower than low"
    assert (df["High"] >= df["Low"]).all(), "High lower than low"

    # ? Check for outliers in returns
    returns = df["Close"].pct_change().iloc[1:]
    assert returns.between(-0.2, 0.2).all(), "pct_change outlier returns detected"

    # ? check for big time gap
    time_diffs = df["_time"].diff().iloc[1:]
    assert time_diffs.between(
        pd.Timedelta("1min"),
        pd.Timedelta(days=10),
    ).all(), "Gaps detected in timestamps"

    # ? dtypes
    # assert df["_time"].dtypes == "datetime64[ns, Europe/Istanbul]", "Time column is not datetime"
    df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].astype(
        float
    )
    return df


def drop_first_day_pandas(df):
    # ? drop first day
    firt_date = df["_time"].dt.date[0]
    df["_date"] = df["_time"].dt.date
    df = df.loc[df["_date"] != firt_date]
    return df


def add_symbol_to_column_name_pandas(df, symbol,mode="no_symbol_name"):
    
    if mode=="no_symbol_name":
        df.rename(
            columns={
                "Open": "M5_OPEN",
                "Close": "M5_CLOSE",
                "Low": "M5_LOW",
                "High": "M5_HIGH",
                "Volume": "M5_VOLUME",
            },
            inplace=True,
        )

    elif mode=="with_symbol":

        df.rename(
            columns={
                "Open": symbol + "_M5_OPEN",
                "Close": symbol + "_M5_CLOSE",
                "Low": symbol + "_M5_LOW",
                "High": symbol + "_M5_HIGH",
                "Volume": f"{symbol}_M5_VOLUME",
            },
            inplace=True,
        )
    else:
        raise ValueError("!!!")
    
    return df


def generate_realtime_candle_realtime(df, symbol, tf_list=[15, 60, 240, 1440]):
    df = add_symbol_to_column_name_pandas(df, symbol)
    df_pl = pl.from_pandas(df)

    first_row_time = df_pl.row(0, named=True)["_time"]
    if time(0, 0) < first_row_time.time():
        df_pl = df_pl.filter(
            pl.col("_time").dt.date() > first_row_time.date()
        )  # Delete the first day that started from the middle of the day

    df_pl = df_pl.with_columns((pl.col("_time").dt.date()).alias("_date"))
    df_pl = df_pl.with_columns(
        (
            pl.col("_time").dt.minute().cast(pl.Int32, strict=False)
            + (pl.col("_time").dt.hour().cast(pl.Int32, strict=False)) * 60
        ).alias("minutesPassed")
    )


    df_pl = df_pl.with_columns(
        pl.when(pl.col("_time").dt.time().is_in(pl.time(0, 0, 0)))
        .then(1)
        .otherwise(0)
        .alias("isFirst")
    )


    df_pl = df_pl.with_row_index().with_columns(
        pl.col("index").cast(pl.Int32, strict=False).alias("index")
    )  
    df_pl = make_realtime_candle(df_pl, tf_list=tf_list, symbol=symbol)

    return df_pl


##? indicators functions: ------------------------------------------------------------------------
def add_RSI_to_realtime_dataset(dataset, feature_config):
    t0 = tt.time()
    modes = {
        "fe_RSI": {"func": cal_RSI_base_func},
        "fe_EMA": {"func": cal_EMA_base_func},
        "fe_SMA": {"func": cal_SMA_base_func},
        "fe_ATR": {"func": cal_ATR_func},
        "fe_RSTD": {"func": cal_RSTD_func},
    }

    for symbol in feature_config:

        symbol_ratio_dfs = []
        for fe_prefix in modes:
            if fe_prefix not in list(dataset.keys()):
                dataset[fe_prefix] = {}

            if fe_prefix not in list(feature_config[symbol].keys()):
                continue
            features_folder_path = f"{root_path}/data/realtime_cache/{fe_prefix}/"
            shutil.rmtree(features_folder_path, ignore_errors=True)
            Path(features_folder_path).mkdir(parents=True, exist_ok=True)

            try:
                base_cols = feature_config[symbol][fe_prefix]["base_columns"]
            except Exception as e:
                print(e)
                print(symbol)
                print(fe_prefix)
                raise ValueError("!!!")
            opts = {
                "symbol": symbol,
                "candle_timeframe": feature_config[symbol][fe_prefix]["timeframe"],
                "window_size": feature_config[symbol][fe_prefix]["window_size"],
                "features_folder_path": features_folder_path,
            }

            base_features = [
                f"M{tf}_{col}"
                for col in base_cols
                for tf in opts["candle_timeframe"]
            ]
            opts["base_feature"] = base_features
            needed_columns = ["_time", "minutesPassed"] + base_features
            df = dataset["candles"][symbol][needed_columns]

            add_candle_base_indicators_polars(
                df_base=df,
                prefix=fe_prefix,
                base_func=modes[fe_prefix]["func"],
                opts=opts,
            )

            # ? merge
            df = df[["_time"]]
            pathes = glob.glob(
                f"{features_folder_path}/unmerged/{fe_prefix}_**_{symbol}_*.parquet"
            )

            for df_path in pathes:
                df_loaded = pl.read_parquet(df_path)
                df = df.join(df_loaded, on="_time", how="left")

            max_candle_timeframe = max(opts["candle_timeframe"])
            max_window_size = max(opts["window_size"])
            drop_rows = (max_window_size + 1) * (max_candle_timeframe / 5) - 1

            df = df.with_row_count()

            df = (
                df.filter(pl.col("row_nr") >= drop_rows)
                .fill_null(strategy="forward")
                .drop(["row_nr"])
            )

            dataset[fe_prefix][symbol] = df

            # ?? add ratio:
            ratio_prefix = "fe_ratio"
            if ratio_prefix not in list(dataset.keys()):
                dataset[ratio_prefix] = {}
            if ratio_prefix not in list(feature_config[symbol].keys()):
                continue
            if fe_prefix.replace("fe_", "") in list(
                feature_config[symbol][ratio_prefix].keys()
            ):
                ratio_config = feature_config[symbol][ratio_prefix][
                    fe_prefix.replace("fe_", "")
                ]

            symbol_ratio_dfs.append(
                add_all_ratio_by_config(
                    df,
                    symbol,
                    fe_name=fe_prefix.replace("fe_", ""),
                    ratio_config=ratio_config,
                    fe_prefix="fe_ratio",
                )
            )

        # ? merge ratio for one symbol:
        if len(symbol_ratio_dfs) == 0:
            # print("!!! no ratio feature.")
            continue
        elif len(symbol_ratio_dfs) == 1:
            df = symbol_ratio_dfs[0]
        else:
            df = symbol_ratio_dfs[0]
            for i in range(1, len(symbol_ratio_dfs)):
                df = df.join(symbol_ratio_dfs[i], on="_time")

        dataset[ratio_prefix][symbol] = df
        # print(f'--> {fe_prefix}_{symbol} saved.')

    print(f"--> add_RSI_to_realtime_dataset done. time: {(tt.time() - t0):.2f}")

    return dataset


##? fe_shift: ------------------------------------------------------------------------
def add_fe_cndl_shift_fe_realtime_run(dataset, feature_config):
    t0 = tt.time()
    fe_prefix = "fe_cndl_shift"
    dataset[fe_prefix] = {}
    for symbol in feature_config:
        if fe_prefix not in list(feature_config[symbol].keys()):
            continue
        shift_columns = feature_config[symbol][fe_prefix]["columns"]
        shift_configs = feature_config[symbol][fe_prefix]["shift_configs"]

        sh_dfs = []
        df = dataset["candles"][symbol]

        for shift_config in shift_configs:
            timeframe = shift_config["timeframe"]
            shift_sizes = shift_config["shift_sizes"]

            for shift_size in shift_sizes:
                # print(f"symbol:{symbol} , timeframe: {timeframe} , shift_size: {shift_size} :")
                sh_df = create_shifted_col(
                    df, pair_name=symbol, periods=shift_size, time_frame=timeframe
                )
                new_cols = [
                    f"M{timeframe}_{col}_-{shift_size}"
                    for col in shift_columns
                ]
                sh_dfs.append(sh_df[["_time"] + new_cols])

        if len(sh_dfs) == 0:
            raise ValueError("!!! nothing to save.")

        shift_df = sh_dfs[0]
        if len(sh_dfs) > 1:
            for sh_df in sh_dfs[1:]:
                shift_df = shift_df.join(sh_df, on="_time", how="inner")

        dataset[fe_prefix][symbol] = shift_df

    return dataset


##? fe_cndl: ------------------------------------------------------------------------


def add_candle_fe(dataset, feature_config):
    t0 = tt.time()

    # ? -------------------
    fe_prefix = "fe_cndl"
    dataset[fe_prefix] = {}
    for symbol in feature_config:
        tf_list = feature_config[symbol]["fe_cndl"]
        dataset[fe_prefix][symbol] = add_candle_features(
            dataset["candles"][symbol], symbol, tf_list=tf_list, fe_prefix=fe_prefix
        )
        # print(f"--> fe_cndl {symbol} saved | category: {cat}")

    # ? -------------------
    fe_prefix = "fe_cndl_shift"
    # dataset[fe_prefix] = {}
    for symbol in feature_config:
        if fe_prefix not in list(feature_config[symbol].keys()):
            continue
        shift_configs = feature_config[symbol][fe_prefix]["shift_configs"]
        df_sh = dataset[fe_prefix][symbol]
        df_all = df_sh[["_time"]]

        for shift_config in shift_configs:
            timeframe = shift_config["timeframe"]
            shift_sizes = shift_config["shift_sizes"]

            df = add_shifted_candle_features(
                df_sh,
                tf=timeframe,
                shift_sizes=shift_sizes,
                fe_prefix=fe_prefix,
            )
            df_all = df_all.join(df, on="_time", how="left")

        dataset[fe_prefix][symbol] = df_all

    print(f"--> add_candle_fe done. time: {(tt.time() - t0):.2f}")
    return dataset


##? fe_WIN: ------------------------------------------------------------------------
def add_fe_win_realtime_run(dataset, feature_config, round_to=3, fe_prefix="fe_WIN"):
    t0 = tt.time()

    dataset[fe_prefix] = {}

    for symbol in feature_config:
        base_cols = feature_config[symbol][fe_prefix]["base_columns"]
        raw_features = [f"M5_{base_col}" for base_col in base_cols]
        needed_columns = ["_time", "minutesPassed"] + raw_features

        df = dataset["candles"][symbol][needed_columns].to_pandas()
        df.sort_values("_time", inplace=True)

        df = add_win_fe_base_func(
            df,
            symbol,
            raw_features=raw_features,
            timeframes=feature_config[symbol][fe_prefix]["timeframe"],
            window_sizes=feature_config[symbol][fe_prefix]["window_size"],
            round_to=round_to,
            fe_prefix=fe_prefix,
        )

        df.drop(columns=raw_features + ["minutesPassed"], inplace=True)

        dataset[fe_prefix][symbol] = pl.from_pandas(df)

    print(f"--> add_fe_win_realtime_run done. time: {(tt.time() - t0):.2f}")
    return dataset


##? real time candles: ------------------------------------------------------

def add_real_time_candles(dataset, feature_config):
    t0 = tt.time()

    fe_prefix = "candles"
    dataset[fe_prefix] = {}
    for symbol in feature_config:
        tf_list = feature_config[symbol]["base_candle_timeframe"]
        dataset[fe_prefix][symbol] = generate_realtime_candle_realtime(
            dataset["st_one"][symbol].copy(), symbol, tf_list=tf_list
        )

    t0 = tt.time()

    print(f"--> add_real_time_candles done. time: {(tt.time() - t0):.2f}")

    return dataset

def merge_realtime_dataset(dataset, dataset_config): 
    df_list = []
    for feature in dataset_config["features"]:
        match dataset[feature]:
            case pd.DataFrame():
                # print(f"flag 1: {feature}")
                df = dataset[feature]
                df = df.sort_values("_time").drop("symbol",errors='ignore')
                # df = df.rename(add_symbol_to_prefixes(df.columns, symbol))
                df_list.append(df)
            case pl.DataFrame():
                # print(f"flag 2: {feature}")
                df = dataset[feature].to_pandas()
                df = df.sort_values("_time").drop("symbol",errors='ignore')
                # df = df.rename(add_symbol_to_prefixes(df.columns, symbol))
                df_list.append(df)
            case dict():
                # print(f"flag 3: {feature}")
                for symbol in dataset[feature]:
                    # Check if it is a polars dataframe
                    if isinstance(dataset[feature][symbol], pl.DataFrame):
                        # print(feature)
                        df = dataset[feature][symbol].to_pandas()
                        df = df.sort_values("_time").drop("symbol",errors='ignore')
                        df.rename(columns=add_symbol_to_prefixes(df.columns, symbol),inplace=True)
                        # print(f"{feature} | {symbol}")
                        # print(f"df.columns: {df.columns}")
                        df_list.append(df)
                    # Check if it is a pandas dataframe
                    elif isinstance(dataset[feature][symbol], pd.DataFrame):
                        df = dataset[feature][symbol]
                        df = df.sort_values("_time").drop("symbol",errors='ignore')
                        df = df.rename(add_symbol_to_prefixes(df.columns, symbol))
                        df_list.append(df)
                    else:
                        raise ValueError(
                            f"Unsupported data type for {feature} -> {symbol}"
                        )
            case _:
                raise ValueError(f"Unsupported data type for {feature}")

    final_df = df_list[0]
    for df in df_list[1:]:
        final_df = final_df.merge(df, on="_time", how="inner")

    return final_df


def crawl_realtime_data_metatrader(
    mt5, dataset, feature_config, mode="init", forward_fill=True, data_size_in_days=12
):
    prefix = "st_one"
    if mode == "init":
        dataset[prefix] = {}
        for symbol in feature_config:
            symbol_m = symbols_dict[symbol]["metatrader_id"]

            df = crawl_OHLCV_data_metatrader_one_symbol(
                mt5, symbol_m, data_size_in_days, forward_fill=forward_fill
            )
            df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "tick_volume": "Volume",
                },
                inplace=True,
            )

            df = realtime_stage_one(df)
            # df = drop_first_day_pandas(df)

            true_time_df = generate_true_time_df_pandas(df)
            df = true_time_df.merge(df, on=["_time"], how="left")
            # print(f"--> {symbol} nulls:",((df.isnull()["Open"].sum()/df.shape[0])*100))
            df.sort_values("_time", inplace=True)
            df.drop(
                columns=["real_volume", "spread", "_date"],
                inplace=True,
                errors="ignore",
            )

            dataset[prefix][symbol] = df

    elif mode == "update":
        for symbol in feature_config:
            symbol_m = symbols_dict[symbol]["metatrader_id"]

            df = crawl_OHLCV_data_metatrader_one_symbol(
                mt5, symbol_m, data_size_in_days, forward_fill=forward_fill
            )
            df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "tick_volume": "Volume",
                },
                inplace=True,
            )

            df = pd.concat([dataset["st_one"][symbol], df])
            df.drop_duplicates("_time", inplace=True, keep="last")
            df.sort_values("_time", inplace=True)

            
            df.drop(
                columns=["real_volume", "spread", "_date"],
                inplace=True,
                errors="ignore",
            )

            dataset["st_one"][symbol] = df

    return dataset


## fe_time features -----------------------------------------------------------
def add_fe_time_realtime_run(dataset, feature_config):
    fe_prefix = "fe_time"
    dataset[fe_prefix] = {}

    df = (
        dataset["st_one"][list(feature_config.keys())[0]][["_time"]]
        .sort_values("_time")
        .reset_index(drop=True)
    )
    df["_time"] = df["_time"] - timedelta(hours=7)

    dtf = DatetimeFeatures(
        features_to_extract=[
            "month",
            "quarter",
            "semester",
            "week",
            "day_of_week",
            "day_of_month",
            "day_of_year",
            "month_start",
            "month_end",
            "quarter_start",
            "quarter_end",
            "year_start",
            "year_end",
            "hour",
            "minute",
        ],
        drop_original=False,
    )

    dtf.fit(df.rename(columns={"_time": fe_prefix}))
    fe_time = dtf.transform(df.rename(columns={"_time": fe_prefix})).rename(
        columns={fe_prefix: "_time"}
    )

    # section 2 -----------------
    markets_trade_times = {
        "New_York": (8, 17),
        "Tokyo": (19, 4),
        "Sydney": (15, 0),
        "London": (3, 11),
    }

    for market_name in markets_trade_times:
        start_time = markets_trade_times[market_name][0]
        stop_time = markets_trade_times[market_name][1]

        col_name = f"{fe_prefix}_isin_{market_name}"
        fe_time[col_name] = 0

        if stop_time > start_time:
            fe_time.loc[
                (fe_time["_time"].dt.hour >= start_time)
                & (fe_time["_time"].dt.hour < stop_time),
                col_name,
            ] = 1

        else:
            fe_time.loc[
                (fe_time["_time"].dt.hour >= start_time)
                | (fe_time["_time"].dt.hour < stop_time),
                col_name,
            ] = 1

    fe_time["_time"] = fe_time["_time"] + timedelta(hours=7)
    dataset[fe_prefix] = fe_time
    return dataset


def add_fe_market_close_realtime_run(dataset, feature_config):
    fe_prefix = "fe_market_close"
    dataset[fe_prefix] = {}
    ##? in EST time zone
    markets_trade_times = {
        "New_York": {
            "hour": 16,
            "minute": 55,
        },
        "Tokyo": {
            "hour": 3,
            "minute": 55,
        },
        "Sydney": {
            "hour": 23,
            "minute": 55,
        },
        "London": {
            "hour": 10,
            "minute": 55,
        },
    }

    for symbol in feature_config:
        df = (
            dataset["st_one"][symbol][["_time", "Close"]]
            .sort_values("_time")
            .reset_index(drop=True)
        )
        df["_time"] = df["_time"] - timedelta(hours=7)

        for market in markets_trade_times:
            fiter = (df["_time"].dt.hour == markets_trade_times[market]["hour"]) & (
                df["_time"].dt.minute == markets_trade_times[market]["minute"]
            )

            df["last_close_price"] = None
            df.loc[fiter, "last_close_price"] = df.loc[fiter, "Close"]

            df["last_close_time"] = None
            df.loc[fiter, "last_close_time"] = df.loc[fiter, "_time"]
            with pd.option_context("future.no_silent_downcasting", True):
                df = df.ffill(inplace=False).infer_objects(copy=False)
                # df.ffill(inplace=True)

            df[f"{fe_prefix}_{symbol}_{market}"] = (
                df["Close"] - df["last_close_price"]
            ) / symbols_dict[symbol]["pip_size"]
            df[f"{fe_prefix}_{symbol}_{market}_time"] = (
                df["_time"] - df["last_close_time"]
            ).dt.seconds // 60

        ##? parquet save:
        df.drop(columns=["last_close_price", "Close", "last_close_time"], inplace=True)
        df["_time"] = df["_time"] + timedelta(hours=7)
        df.dropna(inplace=True)
        df.reset_index(drop=True).dropna(inplace=True)

        dataset[fe_prefix][symbol] = df

    return dataset


# prediction func --------------------
def predict_realtime(
    models_list, predictions, preds_df, crawl_time, final_df, reporter=None
):
    reporter.send_message("run predict_realtime:")
    predictions[crawl_time] = {}

    for model_dict in models_list:
        reporter.send_message("-" * 15)
        reporter.send_message(f"model name:{model_dict['model_name']}")
        model_feature_names_in_ = model_dict["model_object"].model.feature_names_in_
        model_dtypes = model_dict["model_object"].input_cols
        x_p = final_df[model_feature_names_in_].iloc[-2:].astype(model_dtypes)
        x_p = final_df[model_feature_names_in_].iloc[-2:]
        # x_p = final_df.iloc[-2:]

        predictions[crawl_time][model_dict["model_name"]] = {
            "model_id_name": model_dict["model_name"],
            # "target_name": models_dict[model_id_name]["target"],
            "_time": x_p.index[0],
            "crawl_time": crawl_time,
            "predict_time": get_vantage_time_now(),
            "model_prediction": model_dict["model_object"].model.predict(x_p)[-1],
            "strategy_trade_mode": model_dict["strategy_details"]["trade_mode"],
            "strategy_target_symbol": model_dict["strategy_details"]["target_symbol"],
            "strategy_look_ahead": model_dict["strategy_details"]["look_ahead"],
            "strategy_take_profit": model_dict["strategy_details"]["take_profit"],
            "strategy_stop_loss": model_dict["strategy_details"]["stop_loss"],
            "strategy_volume": model_dict["strategy_details"]["volume"],
        }

        reporter.send_message(
            f"model prediction: {predictions[crawl_time][model_dict['model_name']]['model_prediction']} , candle_time: {predictions[crawl_time][model_dict['model_name']]['_time']}"
        )
        if predictions[crawl_time][model_dict["model_name"]]["model_prediction"] == 1:
            reporter.send_message("*** SIGNAL")

    # print(predictions[dataset_time][models_dict[model_id_name]["target"]])
    preds_df = pd.concat(
        [preds_df, pd.DataFrame(predictions[crawl_time]).T]
    ).reset_index(drop=True)
    return predictions, preds_df
