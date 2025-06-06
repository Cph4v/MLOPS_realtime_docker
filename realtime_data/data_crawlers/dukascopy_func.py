"""

get data from www.dukascopy.com:
    https://www.dukascopy.com/plugins/fxMarketWatch/?historical_data
    https://github.com/Leo4815162342/dukascopy-node/tree/e5b361bae5d9a7b4c899949a7c50f5ae0ad0ba1f
    https://ticks.alpari.org/
    https://twelvedata.com/account/
    https://github.com/RomelTorres/alpha_vantage
    https://github.com/cuemacro/findatapy

"""

import lzma
import struct
import pandas as pd
import aiohttp
import asyncio
import os
from pathlib import Path
import os.path
import time
import polars as pl
from configs.history_data_crawlers_config import start_date, stop_date
from datetime import datetime, timezone
from configs.history_data_pipeline_configs import databese_tables
from database.timescaldb_utils import *
from utils.logging_tools import default_logger
from configs.history_data_crawlers_config import (
    data_folder,
    symbols_dict,
)

async def download_file(url, file_path, try_again=6, logger =default_logger):
    for retry in range(try_again):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.read()
                    with open(file_path, "wb") as f:
                        f.write(data)
                    return True
                await asyncio.sleep(0.2)  # pause before retrying

    return False

async def get_dukascopy_data_01(
    symbol,
    dates_list,
    folder_path,
    decimal_divide=1e5,
    concurrency_limit=5,
    fmp=">5I1f",
    logger =default_logger,
):
    sem = asyncio.Semaphore(concurrency_limit)

    save_folder_orginal_files = f"{folder_path}/{symbol}/"
    Path(save_folder_orginal_files).mkdir(parents=True, exist_ok=True)

    faild_dates = []
    data = []

    chunk_size = struct.calcsize(fmp)

    async def process_date(date_item):
        async with sem:
            file_path = f"{save_folder_orginal_files}/{symbol}_{date_item.strftime('%Y')}-{date_item.strftime('%m')}-{date_item.strftime('%d')}_BID_candles_min_1.bi5"

            if os.path.isfile(file_path):
                with lzma.open(file_path) as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if chunk:
                            data.append((date_item,) + struct.unpack(fmp, chunk))
                        else:
                            break

            else:
                url = f"https://datafeed.dukascopy.com/datafeed/{symbol}/{date_item.strftime('%Y')}/{'{0:0=2d}'.format(int(date_item.strftime('%m'))-1)}/{date_item.strftime('%d')}/BID_candles_min_1.bi5"

                success = await download_file(url, file_path)
                if not success:
                    faild_dates.append(date_item)
                    logger.info(f"--> faild_date: {url}")
                    return

                with lzma.open(file_path) as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if chunk:
                            data.append((date_item,) + struct.unpack(fmp, chunk))
                        else:
                            break

    await asyncio.gather(*[process_date(date) for date in dates_list])

    df = pd.DataFrame(
        data,
        columns=["_time", "_second", "open", "close", "low", "high", "tick_volume"],
    )
    df["open"] = df["open"] / decimal_divide
    df["high"] = df["high"] / decimal_divide
    df["low"] = df["low"] / decimal_divide
    df["close"] = df["close"] / decimal_divide

    df["_second"] = df["_second"].astype("float64")
    df["_time"] = df["_time"] + pd.to_timedelta(df["_second"], unit="s")
    df.drop(columns=["_second"], inplace=True)

    return df, faild_dates

def make_M5_from_M1(df: pl.DataFrame, logger =default_logger):
    """
    Transforms a DataFrame of 1-minute OHLCV data into 5-minute OHLCV data.

    Parameters:
    - df (polars DataFrame): Input DataFrame containing 1-minute OHLCV data.
    - logger (Logger, optional): Logger object for logging messages (default is default_logger).

    Returns:
    - DataFrame: Transformed Polars DataFrame containing 5-minute OHLCV data with columns 'open', 'high', 'low', 'close', 'tick_volume'.

    Example:
        make_M5_from_M1(df=my_1min_data)
    """
    df = df.sort("_time")
    df = df.group_by_dynamic(index_column="_time", every="5m").agg(
        [
            pl.first("open").alias("open"),
            pl.max("high").alias("high"),
            pl.min("low").alias("low"),
            pl.last("close").alias("close"),
            pl.sum("tick_volume").alias("tick_volume"),
        ]
    )
    return df

async def crawl_OHLCV_data_dukascopy(
    feature_config: dict,
    logger = default_logger,
    start_date=start_date,
    stop_date=stop_date,
    data_folder: str=data_folder,
 
):  
    
    logger.info(f"- " * 25)
    logger.info(f"--> start crawl_OHLCV_data_dukascopy fumc:")

    key_name = "rawdata_dukascopy"
    table_name = databese_tables[key_name]["table_name"]
    dukascopy_crawl_batch_size = databese_tables[key_name]["batch_size"]

    ensure_table_exists_and_create_hypertable(
        table_name, ts_config
    )
    
    data_source = "dukascopy"
    dates_list = pd.date_range(start_date, stop_date, freq="d")
    dates_list = set(
        [
            pd.to_datetime(date).to_pydatetime().replace(tzinfo=timezone.utc)
            for date in dates_list
        ]
    )
    logger.info(f"--> number of days ALL in time range: {len(dates_list)}")
    logger.info(f"--> start_date: {start_date} , stop_date: {stop_date}")

    faild_dates_dict = {}
    for symbol in feature_config:
        t0 = time.time()
        logger.info("=" * 30)
        logger.info(f"--> {symbol}")
        symbol = symbols_dict[symbol]["dukascopy_id"]
        faild_dates_dict.setdefault(symbol, [])

        cols, existing_dataes = query_timescaledb(
            table_name,
            ts_config,
            requested_columns=["_time"],
            start_time=None,
            end_time=None,
            symbol=symbol,
        )
        flat_list2 = [item[0] for item in existing_dataes]
        filtered_dates = {
            dt.replace(hour=0, minute=0, second=0, microsecond=0)
            for dt in flat_list2
            if dt.hour == 0 and dt.minute == 0 and dt.second == 0
        }

        missing_dates = sorted(dates_list - set(filtered_dates))
        logger.info(f"--> number of missing days: {len(missing_dates)}")

        for i in range(0, len(missing_dates), dukascopy_crawl_batch_size):
            start_idx = i
            end_idx = min(i + dukascopy_crawl_batch_size, len(missing_dates) - 1)
            logger.info(
                f"--> getting data from: {missing_dates[start_idx]} to {missing_dates[end_idx]}"
            )

            df, faild_dates = await get_dukascopy_data_01(
                symbol,
                missing_dates[start_idx:end_idx],
                data_folder,
                decimal_divide=symbols_dict[symbol]["decimal_divide"],
                concurrency_limit=5,
                fmp=">5I1f",
            )
            faild_dates_dict[symbol] += faild_dates

            if df.shape[0] <= 0:
                logger.warning(f"--> f{key_name}: no data, df.shape: {df.shape}")
                continue

            # main M5 data from M1
            df_pl = pl.from_pandas(df)
            df_pl = make_M5_from_M1(df_pl)
            df_pl = df_pl.with_columns(pl.lit(data_source).alias("data_source"))
            df_pl = df_pl.with_columns(pl.lit(symbol).alias("symbol"))
            check_insert_dataframe_to_timescaledb(df=df_pl, table_name=table_name,ts_config =ts_config, drop_na= True) 

        logger.info(f"--> faild_dates: {faild_dates}")
        logger.info(f"--> f{key_name} run-time: {(time.time()-t0):.2f} seconds")

    logger.info(f"--> crawl_OHLCV_data_dukascopy successfully.")
    return faild_dates_dict

if __name__ == "__main__":
    from utils.config_utils import read_feature_config
    from configs.feature_configs_general import generate_general_config
    key_name = "rawdata_dukascopy"
    table_name = databese_tables[key_name]["table_name"]
    delete_table(table_name=table_name, ts_config=ts_config)
    config_general = generate_general_config()
    asyncio.run(crawl_OHLCV_data_dukascopy(config_general))
    default_logger.info(f"--> crawl_OHLCV_data_dukascopy DONE.")
