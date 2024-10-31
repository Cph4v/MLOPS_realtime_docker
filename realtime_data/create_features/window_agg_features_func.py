
from configs.history_data_crawlers_config import symbols_dict
import numpy as np


def cal_window_max(array, window_size):
    res = np.zeros([array.shape[0], 4])
    res[:window_size, :] = np.nan
    for i in range(window_size, array.shape[0]):
        selected_slice = array[i - window_size + 1 : i + 1]

        res[i, :] = [
            np.min(selected_slice),
            1 - (np.argmin(selected_slice) / (window_size - 1)),
            np.max(selected_slice),
            1 - (np.argmax(selected_slice) / (window_size - 1)),
        ]
    return res


def add_win_fe_base_func(
    df, symbol, raw_features, timeframes, window_sizes, round_to=3, fe_prefix="fe_WIN"
):
    for tf in timeframes:
        for w_size in window_sizes:
            assert (
                tf == 5
            ), "!!! for now, this code only work with 5M timeframe, tf must be 5."
            # if int(tf) != 5:
            #     df = df[df.minutesPassed == tf -5]

            col_min = f"{fe_prefix}_min_W{w_size}_M{tf}"
            col_argmin = f"{fe_prefix}_argmin_W{w_size}_M{tf}"
            col_max = f"{fe_prefix}_max_W{w_size}_M{tf}"
            col_argmax = f"{fe_prefix}_argmax_W{w_size}_M{tf}"

            array = df[raw_features].to_numpy()
            # logger.info(col_max, col_max)
            res = cal_window_max(array, w_size)
            df[col_min] = (df[f"M5_CLOSE"] - res[:, 0]) / symbols_dict[symbol][
                "pip_size"
            ]
            df[col_argmin] = np.round(res[:, 1], round_to)
            df[col_max] = (res[:, 2] - df[f"M5_CLOSE"]) / symbols_dict[symbol][
                "pip_size"
            ]
            df[col_argmax] = np.round(res[:, 3], round_to)

    return df


