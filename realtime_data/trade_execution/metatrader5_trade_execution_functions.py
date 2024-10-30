import time
from mt5linux import MetaTrader5
import random
from realtime.realtime_utils import (
    get_vantage_time_now,
)

from datetime import datetime, timedelta, timezone


"""
https://quantra.quantinsti.com/glossary/Automated-Trading-using-MT5-and-Python


##? codes to see:
https://github.com/Ichinga-Samuel/aiomql
https://github.com/boxxello/MetaTrader5-Listener-Telegram/blob/master/trader.py
https://github.com/Joaopeuko/metatrader5EasyT/blob/master/metatrader5EasyT/trade.py
https://github.com/jimtin/how_to_build_a_metatrader5_trading_bot_expert_advisor
https://github.com/Zsunflower/Monn/tree/main

##? gym-mtsim
https://github.com/AminHP/gym-mtsim
https://github.com/isaiahbjork/Auto-GPT-MetaTrader-Plugin

"""


# Function to initialize a symbol on MT5
def initialize_symbols(mt5, symbol_array):
    # Get a list of all symbols supported in MT5
    all_symbols = mt5.symbols_get()
    # Create an array to store all the symbols
    symbol_names = []
    # Add the retrieved symbols to the array
    for symbol in all_symbols:
        symbol_names.append(symbol.name)

    # Check each symbol in symbol_array to ensure it exists
    for provided_symbol in symbol_array:
        if provided_symbol in symbol_names:
            # If it exists, enable
            if mt5.symbol_select(provided_symbol, True):
                print(f"Sybmol {provided_symbol} enabled")
            else:
                return ValueError
        else:
            return SyntaxError
    # Return true when all symbols enabled
    return True


def get_account_info(mt5, assert_in_demo: bool = True, print_info: bool = False):
    """
    assert_in_demo : assert we are in demo account.

    """
    account_info_dict = mt5.account_info()._asdict()

    if print_info:
        print("-" * 45)
        print(f"  account owner: {account_info_dict['name']}")
        print(f"  server: {account_info_dict['server']}")
        print(
            f"  balance: {account_info_dict['balance']} | equity: {account_info_dict['equity']}"
        )
        print(f"  currency: {account_info_dict['currency']}")
        print("-" * 45)

    if assert_in_demo:
        assert (
            "Demo" in account_info_dict["server"]
        ), "!!! you are in live mode. NOT DEMO."

    return account_info_dict


def place_order_instant(
    mt5,
    trade_side: str,
    symbol: str,
    lot: float,
    deviation_points: int,
    tp_points: int,
    sl_points: int,
    last_candle_close_price:float,
    base_price_mode:str ="tick_price",
    magic_number: int = random.randint(2000, 100000),
    
):
    """ 
    
    base_price_mode : "tick_price" , "candle_close"
    """
    point = mt5.symbol_info(symbol).point

    trade_inputs = {
        "trade_side": trade_side,
        "symbol": symbol,
        "lot": lot,
        "deviation_points": deviation_points,
        "tp_points": tp_points,
        "sl_points": sl_points,
        "magic_number": magic_number,
        "point": point,
    }

    if trade_side == "long":
        price = mt5.symbol_info_tick(symbol).ask
        t_type = mt5.ORDER_TYPE_BUY
        if base_price_mode=="tick_price":
            tp_price = price + tp_points * point
            sl_price = price - sl_points * point
        elif base_price_mode=="candle_close":
            tp_price = last_candle_close_price + tp_points * point
            sl_price = last_candle_close_price - sl_points * point
        else:
            raise ValueError("!!! wrong base_price_mode mode.")

    elif trade_side == "short":
        price = mt5.symbol_info_tick(symbol).bid
        t_type = mt5.ORDER_TYPE_SELL
        if base_price_mode=="tick_price":
            tp_price = price - tp_points * point
            sl_price = price + sl_points * point
        elif base_price_mode=="candle_close":
            tp_price = last_candle_close_price - tp_points * point
            sl_price = last_candle_close_price + sl_points * point
        else:
            raise ValueError("!!! wrong base_price_mode mode.")
            
    else:
        raise ValueError("!!! wrong trade_side.")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": t_type,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": deviation_points,
        "magic": magic_number,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        # "type_time": mt5.ORDER_TIME_SPECIFIED,
        # "expiration": expiration,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        position_placed_successfuly = False
        print(
            f"!!! order filed. retcode = {result.retcode} | comment = {result.comment}"
        )
        print(result)
    else:
        position_placed_successfuly = True
        print("--> position successfuly opened.")

    return {
        "request": request,
        "trade_inputs": trade_inputs,
        "result": result,
        "position_placed_successfuly": position_placed_successfuly,
    }


# Function to cancel an order
def cancel_order(order_number):
    # Create the request
    request = {
        "action": mt5.TRADE_ACTION_REMOVE,
        "order": order_number,
        "comment": "Order Removed",
    }
    # Send order to MT5
    order_result = mt5.order_send(request)
    return order_result


# Function to modify an open position
def modify_position(order_number, symbol, new_stop_loss, new_take_profit):
    # Create the request
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "sl": new_stop_loss,
        "tp": new_take_profit,
        "position": order_number,
    }
    # Send order to MT5
    order_result = mt5.order_send(request)
    if order_result[0] == 10009:
        print("done")
        return order_result
    else:
        print("!!! failed")
        return order_result


# Function to retrieve all open orders from MT5
def get_open_orders(mt5):
    orders = mt5.orders_get()
    order_array = []
    for order in orders:
        order_array.append(order[0])
    return order_array


def get_history_orders(mt5):
    orders = mt5.history_orders_get()
    order_array = []
    for order in orders:
        order_array.append(order[0])
    return order_array


# Function to retrieve all open positions
def get_open_positions(mt5):
    # Get position objects
    positions = mt5.positions_get()
    # Return position objects
    return positions


def close_positions_by_id(mt5, ticket_id_rmv, comment=None):
    open_positions = mt5.positions_get()
    if len(open_positions) > 0:
        try:
            chosen_position = next(
                i for i in open_positions if i.ticket == ticket_id_rmv
            )
        except StopIteration as e:
            print(
                f"No positions to remove were found for account: {mt5.account_info().login}"
            )
            return f"No positions to remove were found for account: {mt5.account_info().login}"

        order_type = chosen_position.type
        ticket = chosen_position.ticket
        symbol = chosen_position.symbol
        volume = chosen_position.volume

        if comment is None:
            comment = "Close trade id"
        if order_type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        elif order_type == mt5.ORDER_TYPE_SELL:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        else:
            raise ValueError("!!!!")

        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "position": ticket,
            "price": price,
            "magic": random.randint(2000, 100000),
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(close_request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(
                "Position to close order: Ticket ID: {}, Encountered Error {} , Return code {}".format(
                    ticket, mt5.last_error(), result.retcode
                )
            )
            order_closed_successfuly = False
        else:
            print(f"Position successfully closed! Ticket ID: {ticket}")
            order_closed_successfuly = True

        return {
            "close_request": close_request,
            "chosen_position": chosen_position,
            "ticket_id_rmv": ticket_id_rmv,
            "result": result,
            "order_closed_successfuly": order_closed_successfuly,
        }
    else:
        print("!!! no active position to remove.")
        return "!!! no active position to remove."


def close_positions_all_of_them(mt5, symbol: str = None, comment=None) -> list:
    if symbol is None:
        open_positions = mt5.positions_get()
    else:
        open_positions = mt5.positions_get(symbol=symbol)

    fn_output = []

    if len(open_positions) > 0:
        for i in range(len(open_positions)):
            chosen_position = open_positions[i]
            close_positions_by_id(mt5, chosen_position.ticket, comment=None)

    else:
        if symbol is None:
            print(
                f"No positions to remove were found for account: {mt5.account_info().login}"
            )
        else:
            print(
                f"No positions to remove with symbol {symbol} were found for account: {mt5.account_info().login}"
            )

    return fn_output


def check_active_positions_for_time_force_close(mt5, all_trades):
    mt5_open_positions = get_open_positions(mt5)
    print(f"---> len mt5_open_positions: {len(mt5_open_positions)}")
    for mt5_position in mt5_open_positions:
        print("-" * 15)
        now = get_vantage_time_now()
        order_id = int(mt5_position.ticket)
        print(f"order_id: {order_id}")
        if order_id not in list(all_trades.keys()):
            print(f"!!! unclosed position from before runs. order_id: {order_id}")
            continue
        force_close_position_time = all_trades[order_id]["force_close_position_time"]
        if now >= force_close_position_time:
            print(
                f"time to close | Now time: {now} , force_sel_time: {force_close_position_time}"
            )
            closed_position_info = close_positions_by_id(mt5, order_id)
            all_trades[order_id]["position_status_active"] = False
            all_trades[order_id]["position_close_time"] = get_vantage_time_now()

    return all_trades

def cal_candle_time_now(time_frame:int=5):
  now = get_vantage_time_now()
  return now + (datetime.min - now) % timedelta(minutes=int(time_frame)) - timedelta(minutes=int(2*time_frame))
