
from realtime.realtime_utils import (

    merge_realtime_dataset,
)
import pytz
from realtime.realtime_utils import crawl_realtime_data_metatrader
import time as tt
from datetime import datetime, timedelta

def dataset_gen_realtime_loop(
    mt5,
    fe_functions,
    feature_config,
    dataset_config,
    dataset={},
    data_size_in_days=25,
):
    print("-" * 38)
    t0 = tt.time()
    if dataset == {}:
        print("--> init. dataset.")
        # ? get data:
        # mt5 = initialize_login_metatrader()
        dataset = crawl_realtime_data_metatrader(
            mt5,
            dataset,
            feature_config,
            mode="init",
            forward_fill=True,
            data_size_in_days=data_size_in_days,
        )
        crawl_time = datetime.now(tz=pytz.timezone("US/Eastern")) + timedelta(hours=7)

        print(f"--> crawl time: {(tt.time() - t0):.2f}")

        # return dataset,0,0
        # ? create features:
        for func in fe_functions:
            dataset = func(dataset, feature_config)

    else:
        weekday_number_now = (
            datetime.now(tz=pytz.timezone("US/Eastern")) + timedelta(hours=7)
        ).weekday()
        is_weekend = weekday_number_now in [5, 6]

        if not is_weekend:
            print("--> update dataset.")
            # ? update data:
            dataset = crawl_realtime_data_metatrader(
                mt5, dataset,feature_config,  mode="update", forward_fill=True
            )
            crawl_time = datetime.now(tz=pytz.timezone("US/Eastern")) + timedelta(
                hours=7
            )
            print(f"--> crawl time: {(tt.time() - t0):.2f}")

            # ? create features:
            for func in fe_functions:
                dataset = func(dataset, feature_config)
        else:
            crawl_time = datetime.now(tz=pytz.timezone("US/Eastern")) + timedelta(
                hours=7
            )
            print("!! weekend time. no update")

    t1 = tt.time()
    final_df = merge_realtime_dataset(dataset, dataset_config)
    final_df.set_index("_time", inplace=True)
    final_df.sort_index(inplace=True)

    print(f"--> merge_columns time: {(tt.time() - t1):.2f}")
    print(f"dataset time: {(tt.time() - t0):.2f}")
    print("-" * 38)

    return dataset, final_df, crawl_time
