import threading
import queue
import time
from get_data import fetch_binance_data_formatted
from bot import Bot, plot_queue
from queue import Queue
from graphing import generate_graph
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk


mybot = None
bot_thread = None
def fetch_historical_data(symbol):
    return fetch_binance_data_formatted(symbol=symbol, interval='1h', limit=2000)  # Adjust the limit as needed


def start_bot(input_currency, initial_stack=1, mode='real_time'):
    data_queue = queue.Queue()
    bot_thread = None
    symbol = input_currency + '_USDT'

    if mode == 'real_time':
        bot_instance = Bot(simulation_mode=False, init_stack=initial_stack, pair=symbol)
        def data_fetching_thread():
            data = fetch_binance_data_formatted(symbol=symbol, interval='1m', limit=100)
            if data:
                data_queue.put(data)
            while True:
                data = fetch_binance_data_formatted(symbol=symbol, interval='1m', limit=1)
                if data:
                    data_queue.put(data[0])
                time.sleep(60)

        fetch_thread = threading.Thread(target=data_fetching_thread)
        fetch_thread.start()

        bot_instance = Bot(simulation_mode=False, init_stack=initial_stack, pair=symbol)
        bot_thread = threading.Thread(target=bot_instance.run, args=(data_queue,))
    elif mode == 'simulated_real_time':
        historical_data = fetch_historical_data(symbol=symbol)
        bot_instance = Bot(simulation_mode=True, historical_data=historical_data, pair=symbol, init_stack=initial_stack)
        bot_thread = threading.Thread(target=bot_instance.run, args=(None,))
    bot_thread.start()
    return bot_instance, bot_thread

def stop_bot():
    global mybot, bot_thread
    if mybot:
        mybot.signal_shutdown()
def display_results(symbol, stats):
    result_window = tk.Toplevel()
    result_window.title(f"Results for {symbol}")

    img = Image.open("output.png")
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(result_window, image=img)
    panel.image = img 
    panel.pack()

    stats_frame = tk.Frame(result_window)
    stats_frame.pack(side=tk.BOTTOM)

    for stat_name, stat_value in stats.items():
        stat_label = tk.Label(stats_frame, text=f"{stat_name}: {stat_value}")
        stat_label.pack()
def process_bot(callback):
    global mybot
    plot_data = plot_queue.get()
    symbol = entry.get().upper() + '_USDT'
    generate_graph(plot_data=plot_data, symbol=symbol)
    callback()

def on_processing_complete():
    plot_data = plot_queue.get()
    symbol = entry.get().upper() + '_USDT'
    generate_graph(plot_data=plot_data, symbol=symbol)
    display_results(symbol=symbol, stats=plot_data['stats'])
def run_bot():
    global mybot, bot_thread
    symbol = entry.get().upper()
    initial_stack_value = float(initial_stack_entry.get())
    mode = mode_var.get()
    mybot, bot_thread = start_bot(symbol, initial_stack=initial_stack_value, mode=mode)
    if mode == 'simulated_real_time':
        plot_data = plot_queue.get()
        symbol = entry.get().upper() + '_USDT'
        generate_graph(plot_data=plot_data, symbol=symbol)
        display_results(symbol=symbol, stats=plot_data['stats'])
    # Start the processing thread with the callback
    processing_thread = threading.Thread(target=process_bot, args=(on_processing_complete,))
    processing_thread.start()

root = tk.Tk()
root.title("Trading Bot GUI")

label = tk.Label(root, text="Enter the symbol of the currency you want to trade:")
label.pack()

entry = tk.Entry(root)
entry.pack()

# Initial stack value input
initial_stack_label = tk.Label(root, text="Enter initial stack value:")
initial_stack_label.pack()

initial_stack_entry = tk.Entry(root)
initial_stack_entry.pack()

# Mode selection
mode_var = tk.StringVar(value='real_time')  # Default mode
modes = [("Real Time", "real_time"), ("Simulated Real Time", "simulated_real_time")]
for text, mode in modes:
    radio = tk.Radiobutton(root, text=text, variable=mode_var, value=mode)
    radio.pack()

run_button = tk.Button(root, text="Run Bot", command=run_bot)
run_button.pack()

stop_button = tk.Button(root, text="Stop Bot", command=stop_bot)
stop_button.pack()

root.mainloop()
