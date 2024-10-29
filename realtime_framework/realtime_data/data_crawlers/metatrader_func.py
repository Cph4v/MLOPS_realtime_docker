
import pandas as pd
import pytz
from mt5linux import MetaTrader5
from utils.df_utils import (
    ffill_df_to_true_time_steps,
)

from utils.logging_tools import default_logger

import os
from dotenv import load_dotenv
load_dotenv()
timezone = pytz.timezone("Etc/UTC")

def initialize_login_metatrader(logger=default_logger):
    mt5 = MetaTrader5(
        host=os.getenv('METATRADER_HOST'),
        port=os.getenv('METATRADER_PORT'),
    )
    assert mt5.initialize(), "!!! mt5 is not configed."
    logger.info(f"--> metatrader connected.")
    return mt5


def crawl_data_from_metatrader(
    mt5, symbol, timeframe, number_of_days, forward_fill=False,logger=default_logger
):
    """
    crawl data to pandas dataframe.
    min date is = 2023-01-01
    """

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 288 * number_of_days)
    rates_df = pd.DataFrame(rates)
    if rates_df.shape[0] == 0:
        logger.warning(f"!!! metatrader: {symbol} no new data to crawl.")
        rates_df["_time"] = None
        return rates_df

    rates_df.rename(columns={"time": "_time"}, inplace=True)
    rates_df["_time"] = pd.to_datetime(rates_df["_time"], unit="s")

    if forward_fill and rates_df.shape[0] > 0:
        rates_df = ffill_df_to_true_time_steps(rates_df)

    return rates_df


def get_symbols_info(mt5,logger=default_logger):
    """
    Get all financial instruments info from the MetaTrader 5 terminal.

    """
    # ? list if vilable symbols:
    symbols_dict = mt5.symbols_get()
    symbols_dict[0]

    symbols_info = {}
    for item in symbols_dict:
        temp_dict = {
            "description": item.description,
            "name": item.name,
            "path": item.path,
            "currency_base": item.currency_base,
            "currency_profit": item.currency_profit,
            "currency_margin": item.currency_margin,
            "digits": item.digits,
        }
        symbols_info[item.name] = temp_dict

    logger.info(f"--> number of all symbols: {len(symbols_info)}")
    return symbols_info

def crawl_OHLCV_data_metatrader_one_symbol(
    mt5, symbol, number_of_days, forward_fill=False
):
    """
    get data for realtime loop.

    """
    timeframe = mt5.TIMEFRAME_M5

    df = (
        crawl_data_from_metatrader(
            mt5, symbol, timeframe, number_of_days, forward_fill=forward_fill
        )
        .sort_values("_time")
        .reset_index(drop=True)
    )
    return df


