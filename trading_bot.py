import argparse
import sys
import pandas as pd
import numpy as np
import ccxt
from typing import Dict, List, Any, Optional


def calculate_sma(df: pd.DataFrame, window: int, column: str = 'close') -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    """
    return df[column].rolling(window=window).mean()


def calculate_rsi(df: pd.DataFrame, window: int = 14, column: str = 'close') -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    """
    delta = df[column].diff()
    gain_ema = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    loss_ema = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()

    # Avoid division by zero
    rs = gain_ema / loss_ema.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def add_technical_indicators(df: pd.DataFrame, short_window: int = 10, long_window: int = 30, rsi_window: int = 14) -> pd.DataFrame:
    """
    Add SMA and RSI technical indicators to the DataFrame.
    """
    df = df.copy()
    df['sma_short'] = calculate_sma(df, window=short_window)
    df['sma_long'] = calculate_sma(df, window=long_window)
    df['rsi'] = calculate_rsi(df, window=rsi_window)
    return df


class TradingStrategy:
    """
    Trading Strategy based on SMA Crossover and RSI thresholds.
    """
    def __init__(self, short_window: int = 10, long_window: int = 30, rsi_window: int = 14,
                 rsi_oversold: float = 30.0, rsi_overbought: float = 70.0):
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals: 1 (BUY), -1 (SELL), 0 (HOLD)
        """
        data = add_technical_indicators(df, self.short_window, self.long_window, self.rsi_window)
        data['signal'] = 0

        # Golden Cross (Short SMA crosses above Long SMA) & RSI not overbought
        buy_condition = (
            (data['sma_short'] > data['sma_long']) &
            (data['sma_short'].shift(1) <= data['sma_long'].shift(1)) &
            (data['rsi'] < self.rsi_overbought)
        )

        # Death Cross (Short SMA crosses below Long SMA) or RSI overbought
        sell_condition = (
            (
                (data['sma_short'] < data['sma_long']) &
                (data['sma_short'].shift(1) >= data['sma_long'].shift(1))
            ) | (data['rsi'] > self.rsi_overbought)
        )

        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1

        return data


class Backtester:
    """
    Backtesting engine for simulating strategy performance over historical OHLCV data.
    """
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001,
                 stop_loss_pct: Optional[float] = 0.03, take_profit_pct: Optional[float] = 0.05):
        self.initial_capital = initial_capital
        self.commission = commission
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def run(self, df: pd.DataFrame, strategy: TradingStrategy) -> Dict[str, Any]:
        """
        Execute backtest on DataFrame using provided strategy.
        """
        data = strategy.generate_signals(df)

        capital = self.initial_capital
        position = 0.0  # units of asset
        entry_price = 0.0
        trades: List[Dict[str, Any]] = []
        equity_curve: List[float] = []

        for idx, row in data.iterrows():
            current_price = row['close']
            signal = row.get('signal', 0)
            timestamp = row.get('timestamp', idx)

            # Check Stop Loss / Take Profit if currently in a position
            if position > 0 and entry_price > 0:
                price_change = (current_price - entry_price) / entry_price

                # Check Stop Loss
                if self.stop_loss_pct and price_change <= -self.stop_loss_pct:
                    sell_val = position * current_price * (1 - self.commission)
                    capital += sell_val
                    trades.append({
                        'timestamp': timestamp,
                        'type': 'SELL (Stop Loss)',
                        'price': current_price,
                        'amount': position,
                        'capital': capital,
                        'pnl': sell_val - (position * entry_price)
                    })
                    position = 0.0
                    entry_price = 0.0

                # Check Take Profit
                elif self.take_profit_pct and price_change >= self.take_profit_pct:
                    sell_val = position * current_price * (1 - self.commission)
                    capital += sell_val
                    trades.append({
                        'timestamp': timestamp,
                        'type': 'SELL (Take Profit)',
                        'price': current_price,
                        'amount': position,
                        'capital': capital,
                        'pnl': sell_val - (position * entry_price)
                    })
                    position = 0.0
                    entry_price = 0.0

            # Signal evaluation
            if signal == 1 and position == 0:  # BUY
                amount_to_buy = capital / current_price
                cost = amount_to_buy * current_price * (1 + self.commission)
                if cost <= capital:
                    position = amount_to_buy
                    entry_price = current_price
                    capital -= cost
                    trades.append({
                        'timestamp': timestamp,
                        'type': 'BUY',
                        'price': current_price,
                        'amount': position,
                        'capital': capital,
                        'pnl': 0.0
                    })

            elif signal == -1 and position > 0:  # SELL
                sell_val = position * current_price * (1 - self.commission)
                capital += sell_val
                trades.append({
                    'timestamp': timestamp,
                    'type': 'SELL (Signal)',
                    'price': current_price,
                    'amount': position,
                    'capital': capital,
                    'pnl': sell_val - (position * entry_price)
                })
                position = 0.0
                entry_price = 0.0

            # Calculate total portfolio value
            current_portfolio_val = capital + (position * current_price)
            equity_curve.append(current_portfolio_val)

        final_val = capital + (position * data.iloc[-1]['close'] if len(data) > 0 else 0)
        total_return_pct = ((final_val - self.initial_capital) / self.initial_capital) * 100

        # Calculate metrics
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        closed_trades = [t for t in trades if 'SELL' in t['type']]
        win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0.0

        return {
            'initial_capital': self.initial_capital,
            'final_value': final_val,
            'total_return_pct': round(total_return_pct, 2),
            'total_trades': len(trades),
            'closed_trades': len(closed_trades),
            'win_rate_pct': round(win_rate, 2),
            'trades': trades,
            'equity_curve': equity_curve
        }


class CryptoBot:
    """
    Main Crypto Bot client interacting with CCXT exchanges.
    """
    def __init__(self, exchange_id: str = 'binance', api_key: str = '', api_secret: str = '', sandbox: bool = True):
        self.exchange_id = exchange_id
        exchange_class = getattr(ccxt, exchange_id, None)
        if not exchange_class:
            raise ValueError(f"Exchange '{exchange_id}' is not supported by CCXT.")

        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })

        if sandbox and hasattr(self.exchange, 'set_sandbox_mode'):
            self.exchange.set_sandbox_mode(True)

    def fetch_ohlcv(self, symbol: str = 'BTC/USDT', timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """
        Fetch OHLCV historical candles from the exchange.
        """
        raw_candles = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def evaluate_live_signal(self, symbol: str = 'BTC/USDT', timeframe: str = '1h', strategy: Optional[TradingStrategy] = None) -> Dict[str, Any]:
        """
        Fetch recent market data and generate current trading signal.
        """
        if strategy is None:
            strategy = TradingStrategy()

        df = self.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=100)
        df_signals = strategy.generate_signals(df)
        latest_row = df_signals.iloc[-1]

        signal_map = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}
        signal_str = signal_map.get(latest_row['signal'], 'HOLD')

        return {
            'symbol': symbol,
            'timestamp': str(latest_row['timestamp']),
            'close': float(latest_row['close']),
            'sma_short': float(latest_row['sma_short']) if pd.notna(latest_row['sma_short']) else None,
            'sma_long': float(latest_row['sma_long']) if pd.notna(latest_row['sma_long']) else None,
            'rsi': float(latest_row['rsi']) if pd.notna(latest_row['rsi']) else None,
            'signal': signal_str
        }


def generate_simulated_ohlcv(num_candles: int = 200, base_price: float = 50000.0) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for testing and offline backtesting.
    """
    np.random.seed(42)
    returns = np.random.normal(loc=0.0002, scale=0.015, size=num_candles)
    price_path = base_price * np.cumprod(1 + returns)

    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=num_candles, freq='h')

    highs = price_path * (1 + np.abs(np.random.normal(0.001, 0.005, size=num_candles)))
    lows = price_path * (1 - np.abs(np.random.normal(0.001, 0.005, size=num_candles)))
    opens = price_path * (1 + np.random.normal(0, 0.002, size=num_candles))
    volumes = np.random.uniform(100, 1000, size=num_candles)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': price_path,
        'volume': volumes
    })
    return df


def main():
    parser = argparse.ArgumentParser(description="Cryptocurrency Trading Bot & Backtester")
    parser.add_argument('--mode', choices=['backtest', 'signal', 'simulated-backtest'], default='simulated-backtest',
                        help="Operation mode: 'backtest', 'signal', or 'simulated-backtest'")
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help="Trading pair symbol")
    parser.add_argument('--exchange', type=str, default='binance', help="Exchange ID (e.g. binance, kraken)")
    parser.add_argument('--capital', type=float, default=10000.0, help="Initial capital for backtesting")

    args = parser.parse_args()

    strategy = TradingStrategy()

    if args.mode == 'simulated-backtest':
        print(f"--- Running Backtest on Simulated Market Data ({args.symbol}) ---")
        df = generate_simulated_ohlcv(num_candles=200)
        backtester = Backtester(initial_capital=args.capital)
        results = backtester.run(df, strategy)

        print(f"Initial Capital: ${results['initial_capital']:.2f}")
        print(f"Final Value:     ${results['final_value']:.2f}")
        print(f"Total Return:    {results['total_return_pct']}%")
        print(f"Total Trades:    {results['total_trades']}")
        print(f"Win Rate:        {results['win_rate_pct']}%")

    elif args.mode == 'backtest':
        print(f"--- Fetching historical data from {args.exchange} for {args.symbol} ---")
        try:
            bot = CryptoBot(exchange_id=args.exchange)
            df = bot.fetch_ohlcv(symbol=args.symbol, timeframe='1h', limit=200)
            backtester = Backtester(initial_capital=args.capital)
            results = backtester.run(df, strategy)

            print(f"Initial Capital: ${results['initial_capital']:.2f}")
            print(f"Final Value:     ${results['final_value']:.2f}")
            print(f"Total Return:    {results['total_return_pct']}%")
            print(f"Total Trades:    {results['total_trades']}")
            print(f"Win Rate:        {results['win_rate_pct']}%")
        except Exception as e:
            print(f"Error fetching data from exchange: {e}")
            sys.exit(1)

    elif args.mode == 'signal':
        print(f"--- Fetching real-time market signal from {args.exchange} for {args.symbol} ---")
        try:
            bot = CryptoBot(exchange_id=args.exchange)
            signal_data = bot.evaluate_live_signal(symbol=args.symbol, strategy=strategy)
            print(f"Symbol:     {signal_data['symbol']}")
            print(f"Timestamp:  {signal_data['timestamp']}")
            print(f"Last Price: ${signal_data['close']:.2f}")
            print(f"SMA Short:  {signal_data['sma_short']}")
            print(f"SMA Long:   {signal_data['sma_long']}")
            print(f"RSI:        {signal_data['rsi']}")
            print(f"SIGNAL:     {signal_data['signal']}")
        except Exception as e:
            print(f"Error evaluating live signal: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()
