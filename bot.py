#!/usr/bin/python3

import sys
import numpy as np
import time
from pandas import Timestamp
from queue import Queue
import logging

plot_queue = Queue()
class Bot:
    def __init__(self, simulation_mode=False, historical_data=None, pair="BTC_USDT", init_stack=1):
        self.botState = BotState(pair, init_stack)
        self.simulation_mode = simulation_mode
        self.historical_data = historical_data
        self.decisions = []
        self.candle_data = []
        self.usdt_values = []
        self.btc_values = []
        self.pair = pair
        self.strategy = None
        self.stats = []
        self.keep_running = True
    def run(self, data_queue):
        if self.simulation_mode and self.historical_data:
            self.initial_analysis(self.historical_data[:100])
            for data in self.historical_data[:100]:
                self.parse(data)
            for data in self.historical_data[100:]:
                self.parse(data)
        else:
            data = data_queue.get()
            self.initial_analysis(data)
            for datas in data:
                new_candle = Candle(self.botState.candleFormat, datas)
                self.botState.update_chart(datas)
                self.candle_data.append(datas)
            while self.keep_running:
                logging.info("Bot running")
                if not data_queue.empty():
                    logging.info("Data in queue")
                    data = data_queue.get()
                    self.parse(data)
                else:
                    print("No data in queue")
                    time.sleep(1)
        self.evaluate_performance()
        plot_data = {
            'candle_data': self.candle_data,
            'usdt_values': self.usdt_values,
            'btc_values': self.btc_values,
            'stats': self.stats
        }
        plot_queue.put(plot_data)


    def initial_analysis(self, initial_data):
        closing_prices = [candle['close'] for candle in initial_data]
        if self.is_market_volatile(closing_prices):
            self.strategy = self.long_term_strategy
        else:
            self.strategy = self.short_term_strategy
        self.strategy = self.short_term_strategy

    def signal_shutdown(self):
        self.keep_running = False
    def is_market_volatile(self, closing_prices):
        std_dev = np.std(closing_prices)
        volatility_threshold = 50
        return std_dev > volatility_threshold

    def long_term_strategy(self):
        pass

    def short_term_strategy(self):
        if self.calculate_strategy_score_0() >= self.calculate_strategy_score_1():
            self.short_term_strategy_0()
        else:
            self.short_term_strategy_1()


    def short_term_strategy_2(self):
        macd = self.botState.charts[self.pair].indicators['macd']
        macd_signal = self.botState.charts[self.pair].indicators['macd_signal']
        rsi = self.botState.charts[self.pair].indicators['rsi']
        usdt = self.botState.stacks[self.pair.split("_")[1]]
        btc = self.botState.stacks[self.pair.split("_")[0]]
        current_price = self.botState.charts[self.pair].closes[-1]
        current_time = self.botState.charts[self.pair].dates[-1]

        if len(macd) > 0 and len(rsi) > 0:
            if macd[-1] > macd_signal[-1] and rsi[-1] > 30:
                self.execute_buy_logic(usdt, btc, current_price, current_time)
            elif macd[-1] < macd_signal[-1] and rsi[-1] < 70:
                self.execute_sell_logic(usdt, btc, current_price, current_time)

    def calculate_strategy_score_2(self):
        score = 0
        macd = self.botState.charts[self.pair].indicators['macd']
        macd_signal = self.botState.charts[self.pair].indicators['macd_signal']
        rsi = self.botState.charts[self.pair].indicators['rsi']

        if len(macd) > 0 and len(rsi) > 0:
            recent_crossover = any(macd[i] > macd_signal[i] and macd[i-1] <= macd_signal[i-1] for i in range(1, min(5, len(macd))))
            score += 50 if recent_crossover else 0

            optimal_rsi = 30 < rsi[-1] < 70
            score += 50 if optimal_rsi else 0

        return score
    def short_term_strategy_1(self):
        short_term_sma = self.botState.charts[self.pair].calculate_sma(20)
        long_term_sma = self.botState.charts[self.pair].calculate_sma(100)
        usdt = self.botState.stacks[self.pair.split("_")[1]]
        btc = self.botState.stacks[self.pair.split("_")[0]]

        current_closing_price = self.botState.charts[self.pair].closes[-1]
        current_time = self.botState.charts[self.pair].dates[-1]

        print(f"Current {self.pair.split('_')[0]} Price: {current_closing_price}")
        print(f"Current Stacks - USDT: {usdt}, {self.pair.split('_')[0]}: {btc}")

        if short_term_sma is not None and long_term_sma is not None:
            if short_term_sma > long_term_sma:
                self.execute_buy_logic(usdt, btc, current_closing_price, current_time)
            elif short_term_sma < long_term_sma:
                self.execute_sell_logic(usdt, btc, current_closing_price, current_time)
        self.usdt_values.append({
            'time': current_time,
            'value': self.botState.stacks[self.pair.split("_")[1]]
        })
        self.btc_values.append({
            'time': current_time,
            'value': self.botState.stacks[self.pair.split("_")[0]] * current_closing_price
        })

    def execute_buy_logic(self, usdt, btc, current_price, current_time):
        affordable = usdt / current_price
        if affordable >= 0.001:
            trade_amount = min(affordable, 0.5 * affordable)
            self.botState.stacks[self.pair.split("_")[1]] -= trade_amount * current_price
            self.botState.stacks[self.pair.split("_")[0]] += trade_amount
            self.decisions.append({
                "time": current_time,
                "action": "buy",
                "amount": trade_amount,
                "price": current_price
            })
            print(f"Decision: Buy {trade_amount} {self.pair.split('_')[0]}")
        else:
            print("Decision: No action (insufficient funds)")

    def execute_sell_logic(self, usdt, btc, current_price, current_time):
        affordable = usdt / current_price
        if btc >= 0.001:
            trade_amount = min(affordable, 0.5 * affordable)
            usdt_spent = trade_amount * current_price
            self.botState.stacks[self.pair.split("_")[1]] -= usdt_spent
            self.botState.stacks[self.pair.split("_")[0]] += trade_amount
            self.decisions.append({
                "time": current_time,
                "action": "buy",
                "amount": trade_amount,
                "price": current_price
            })
            print(f"Decision: Sell {btc} {self.pair.split('_')[0]}")
        else:
            print("Decision: No action (insufficient {self.pair.split('_')[0]})")

    def calculate_strategy_score_1(self):
        closing_prices = [candle['close'] for candle in self.candle_data]
        short_term_ma = self.calculate_moving_average(closing_prices, 10)
        long_term_ma = self.calculate_moving_average(closing_prices, 50)

        min_length = min(len(short_term_ma), len(long_term_ma))
        if min_length < 1:
            return 0

        crossover_index = None
        for i in range(1, min_length):
            if short_term_ma[-i] is not None and long_term_ma[-i] is not None:
                if short_term_ma[-i] > long_term_ma[-i] and short_term_ma[-i - 1] <= long_term_ma[-i - 1]:
                    crossover_index = i
                    break

        if crossover_index is not None:
            score = max(0, 100 - crossover_index)
        else:
            score = 0

        return score

    def evaluate_performance(self):
        total_profit = self.calculate_total_profit()
        max_drawdown = self.calculate_max_drawdown()
        consistency_score = self.calculate_consistency()
        strategy_adherence = self.calculate_strategy_adherence()

        grade = self.assign_grade(total_profit, max_drawdown, consistency_score, strategy_adherence)
        print(f"Total Profit: {total_profit}")
        print(f"Max Drawdown: {max_drawdown}")
        print(f"Consistency Score: {consistency_score}")
        print(f"Strategy Adherence: {strategy_adherence}")
        print(f"Trading Performance Grade: {grade}")
        self.stats = {
            "Total Profit": total_profit,
            "Max Drawdown": max_drawdown,
            "Consistency Score": consistency_score,
            "Strategy Adherence": strategy_adherence,
            "Trading Performance Grade": grade
        }

    def calculate_total_profit(self):
        final_usdt = self.botState.stacks[self.pair.split("_")[1]]
        final_btc = self.botState.stacks[self.pair.split("_")[0]]
        final_btc_value_in_usdt = final_btc * self.botState.charts[self.pair].closes[-1]
        total_profit = (final_usdt + final_btc_value_in_usdt) - (self.botState.initialStack * self.botState.charts[self.pair].closes[0])
        return total_profit

    def calculate_moving_average(self, data, window):
        if len(data) < window:
            return [None] * len(data)
        return np.convolve(data, np.ones(window) / window, 'valid').tolist()

    def calculate_max_drawdown(self):
        values = [val['value'] for val in self.usdt_values + self.btc_values]

        if not values:
            return 0

        peak = values[0]
        max_drawdown = 0
        for value in values:
            if value > peak:
                peak = value
            drawdown = peak - value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return max_drawdown

    def calculate_consistency(self):
        returns = [self.candle_data[i]['close'] / self.candle_data[i - 1]['close'] - 1 for i in
                   range(1, len(self.candle_data))]
        consistency = np.std(returns)
        return consistency

    def calculate_strategy_adherence(self):
        adherent_decisions = 0
        for decision in self.decisions:
            if self.check_strategy_adherence(decision):
                adherent_decisions += 1
        if len(self.decisions) != 0:
            adherence_rate = adherent_decisions / len(self.decisions)
            return adherence_rate
        else:
            return 0
        return 0

    def check_strategy_adherence(self, decision):
        short_term_sma = self.botState.charts[self.pair].calculate_sma(20)
        long_term_sma = self.botState.charts[self.pair].calculate_sma(100)
        macd = self.botState.charts[self.pair].indicators['macd']
        rsi = self.botState.charts[self.pair].indicators['rsi']

        if not (short_term_sma and long_term_sma and macd and rsi):
            return False

        if decision['action'] == 'buy':
            return short_term_sma > long_term_sma and macd[-1] > 0 and rsi[-1] > 20
        elif decision['action'] == 'sell':
            return short_term_sma < long_term_sma and macd[-1] < 0 and rsi[-1] < 80
        else:
            return False

    def assign_grade(self, profit, drawdown, consistency, adherence):
        if profit > 10000 and drawdown < 2000 and consistency < 0.05 and adherence > 0.8:
            return 'A'
        elif profit > 5000 and drawdown < 4000 and consistency < 0.1 and adherence > 0.6:
            return 'B'
        elif profit > 0:
            return 'C'
        else:
            return 'D'

    def parse(self, candle_data):
        if isinstance(candle_data, dict) and 'open' in candle_data and 'close' in candle_data:
            new_candle = Candle(self.botState.candleFormat, candle_data)
            self.botState.update_chart(candle_data)
            self.candle_data.append(candle_data)

            self.make_decision()
        else:
            print("Invalid candle data format", file=sys.stderr)

    def make_decision(self):
        if self.strategy:
            self.strategy()
        else:
            print("Strategy not set")

    def calculate_strategy_score_0(self):
        score = 0
        short_term_sma = self.botState.charts[self.pair].calculate_sma(20)
        long_term_sma = self.botState.charts[self.pair].calculate_sma(100)
        macd = self.botState.charts[self.pair].indicators['macd']
        rsi = self.botState.charts[self.pair].indicators['rsi']

        if short_term_sma is not None and long_term_sma is not None and len(macd) > 0 and len(rsi) > 0:
            if short_term_sma > long_term_sma:
                score += 30

            if macd[-1] > 0:
                score += 30

            if 20 < rsi[-1] < 80:
                score += 40

        return score
    def short_term_strategy_0(self):
        short_term_sma = self.botState.charts[self.pair].calculate_sma(20)
        long_term_sma = self.botState.charts[self.pair].calculate_sma(100)
        usdt = self.botState.stacks[self.pair.split("_")[1]]
        btc = self.botState.stacks[self.pair.split("_")[0]]

        current_closing_price = self.botState.charts[self.pair].closes[-1]
        current_time = self.botState.charts[self.pair].dates[-1]

        print(f"Current {self.pair.split('_')[0]} Price: {current_closing_price}")
        print(f"Current Stacks - USDT: {usdt}, {self.pair.split('_')[0]}: {btc}")

        if short_term_sma is not None and long_term_sma is not None:
            macd = self.botState.charts[self.pair].indicators['macd']
            rsi = self.botState.charts[self.pair].indicators['rsi']

            if len(macd) > 0 and len(rsi) > 0:
                if short_term_sma > long_term_sma and macd[-1] > 0 and rsi[-1] > 20:
                    affordable = usdt / current_closing_price
                    if affordable >= 0.001:
                        trade_amount = min(affordable, 0.5 * affordable)
                        self.botState.stacks[self.pair.split("_")[1]] -= trade_amount * current_closing_price
                        self.botState.stacks[self.pair.split("_")[0]] += trade_amount
                        self.decisions.append({
                            "time": current_time,
                            "action": "buy",
                            "amount": trade_amount,
                            "price": current_closing_price
                        })
                        print(f"Decision: Buy {trade_amount} {self.pair.split('_')[0]}")
                    else:
                        print("Decision: No action (insufficient funds)")
                elif short_term_sma < long_term_sma and macd[-1] < 0 and rsi[-1] < 80:
                    if btc >= 0.001:
                        self.botState.stacks[self.pair.split("_")[1]] += btc * current_closing_price
                        self.botState.stacks[self.pair.split("_")[0]] = 0
                        self.decisions.append({
                            "time": current_time,
                            "action": "sell",
                            "amount": btc,
                            "price": current_closing_price
                        })
                        print(f"Decision: Sell {btc} {self.pair.split('_')[0]}")
                    else:
                        print("Decision: No action (insufficient {self.pair.split('_')[0]})")
                else:
                    print("Decision: No action")
            else:
                print("Decision: No action (indicators not available)")
        else:
            print("Decision: No action (SMA not available)")

        self.usdt_values.append({
            'time': current_time,
            'value': self.botState.stacks[self.pair.split("_")[1]]
        })
        self.btc_values.append({
            'time': current_time,
            'value': self.botState.stacks[self.pair.split("_")[0]] * current_closing_price
        })

class Candle:
    def __init__(self, format, candle_data):
        if isinstance(candle_data, dict):
            self.pair = candle_data.get('pair', '')

            date = candle_data.get('date', None)
            if isinstance(date, Timestamp):
                self.date = int(date.timestamp())
            elif isinstance(date, (int, float, str)):
                self.date = int(date)
            else:
                self.date = 0

            self.high = float(candle_data.get('high', 0.0))
            self.low = float(candle_data.get('low', 0.0))
            self.open = float(candle_data.get('open', 0.0))
            self.close = float(candle_data.get('close', 0.0))
            self.volume = float(candle_data.get('volume', 0.0))
            self.timestamp = candle_data.get('timestamp')
        else:
            print("Candle data is not in the expected format", file=sys.stderr)


    def __repr__(self):
        return str(self.pair) + str(self.date) + str(self.close) + str(self.volume)

class Chart:
    def __init__(self):
        self.dates = []
        self.opens = []
        self.highs = []
        self.lows = []
        self.closes = []
        self.volumes = []
        self.indicators = {'macd': [], 'macd_signal': [], 'rsi': []}

    def add_candle(self, candle: Candle):
        self.dates.append(candle.date)
        self.opens.append(candle.open)
        self.highs.append(candle.high)
        self.lows.append(candle.low)
        self.closes.append(candle.close)
        self.volumes.append(candle.volume)

        close_array = np.array(self.closes)

        macd, macd_signal = self.calculate_macd(close_array, 12, 26)
        self.indicators['macd'].append(macd[-1] if len(macd) else None)
        self.indicators['macd_signal'].append(macd_signal[-1] if len(macd_signal) else None)

        rsi = self.calculate_rsi(close_array, 14)
        self.indicators['rsi'].append(rsi if rsi is not None else None)

    def calculate_sma(self, period):
        if len(self.closes) >= period:
            sma = np.mean(self.closes[-period:])
            return sma
        else:
            return None

    def calculate_ema(self, data, window):
        if len(data) < window:
            return np.array([None] * len(data))

        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        a = np.convolve(data, weights, mode='full')[:len(data)]
        a[:window] = a[-1]
        return a


    def calculate_macd(self, data, short_window, long_window):
        if len(data) < max(short_window, long_window):
            return np.array([None] * len(data)), np.array([None] * len(data))

        short_ema = self.calculate_ema(data, short_window)
        long_ema = self.calculate_ema(data, long_window)
        macd_line = short_ema - long_ema
        signal_line = self.calculate_ema(macd_line, 9)
        return macd_line, signal_line

    def calculate_rsi(self, data, window):
        if len(data) <= window:
            return None

        delta = np.diff(data)
        up = delta.clip(min=0)
        down = -1 * delta.clip(max=0)
        avg_gain = np.mean(up[-window:])
        avg_loss = np.mean(down[-window:])
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi


class BotState:
    def __init__(self, pair, init_stack=1):
        self.pair = pair
        self.timeBank = 0
        self.maxTimeBank = 0
        self.timePerMove = 1
        self.candleInterval = 1
        self.candleFormat = []
        self.candlesTotal = 0
        self.candlesGiven = 0
        self.initialStack = init_stack
        self.transactionFee = 0.1
        self.date = 0
        self.stacks = dict()
        self.charts = dict()
        self.initialize_stacks(init_stack)
        self.charts[self.pair] = Chart()

    def initialize_stacks(self, init_stack):
        currencies = self.pair.split("_")
        self.stacks[currencies[0]] = init_stack
        self.stacks[currencies[1]] = 0
    def update_chart(self, new_candle_str: str):
        if self.pair not in self.charts:
            self.charts[self.pair] = Chart()

        new_candle_obj = Candle(self.candleFormat, new_candle_str)
        self.charts[self.pair].add_candle(new_candle_obj)

    def update_stack(self, key: str, value: float):
        self.stacks[key] = value

    def update_settings(self, key: str, value: str):
        if key == "timebank":
            self.maxTimeBank = int(value)
            self.timeBank = int(value)
        if key == "time_per_move":
            self.timePerMove = int(value)
        if key == "candle_interval":
            self.candleInterval = int(value)
        if key == "candle_format":
            self.candleFormat = value.split(",")
        if key == "candles_total":
            self.candlesTotal = int(value)
        if key == "candles_given":
            self.candlesGiven = int(value)
        if key == "initial_stack":
            self.initialStack = int(value)
        if key == "transaction_fee_percent":
            self.transactionFee = float(value)

    def update_game(self, key: str, value: str):
        if key == "next_candles":
            new_candles = value.split(";")
            pair = new_candles[0].split(",")[0]
            self.date = int(new_candles[0].split(",")[1])
            for candle_str in new_candles:
                candle_infos = candle_str.strip().split(",")
                self.update_chart(candle_infos[0], candle_str)
        if key == "stacks":
            new_stacks = value.split(",")
            for stack_str in new_stacks:
                stack_infos = stack_str.strip().split(":")
                self.update_stack(stack_infos[0], float(stack_infos[1]))

if __name__ == "__main__":
    mybot = Bot()
    mybot.run()
