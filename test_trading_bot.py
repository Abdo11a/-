import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from trading_bot import (
    calculate_sma,
    calculate_rsi,
    add_technical_indicators,
    TradingStrategy,
    Backtester,
    CryptoBot,
    generate_simulated_ohlcv
)


class TestTradingBot(unittest.TestCase):

    def setUp(self):
        # Create a sample dataframe with 50 rows of deterministic price data
        dates = pd.date_range(start='2025-01-01', periods=50, freq='h')
        prices = [100.0 + i * (1.5 if i % 2 == 0 else -0.5) for i in range(50)]
        self.df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 1.0 for p in prices],
            'low': [p - 1.0 for p in prices],
            'close': prices,
            'volume': [1000.0] * 50
        })

    def test_calculate_sma(self):
        sma_10 = calculate_sma(self.df, window=10)
        self.assertEqual(len(sma_10), 50)
        self.assertTrue(pd.isna(sma_10.iloc[8]))
        self.assertFalse(pd.isna(sma_10.iloc[9]))
        expected_sma = sum(self.df['close'].iloc[:10]) / 10.0
        self.assertAlmostEqual(sma_10.iloc[9], expected_sma)

    def test_calculate_rsi(self):
        rsi = calculate_rsi(self.df, window=14)
        self.assertEqual(len(rsi), 50)
        self.assertTrue((rsi >= 0).all() and (rsi <= 100).all())

    def test_add_technical_indicators(self):
        df_ind = add_technical_indicators(self.df, short_window=5, long_window=15, rsi_window=14)
        self.assertIn('sma_short', df_ind.columns)
        self.assertIn('sma_long', df_ind.columns)
        self.assertIn('rsi', df_ind.columns)

    def test_strategy_signals(self):
        strategy = TradingStrategy(short_window=5, long_window=15, rsi_window=14)
        df_signals = strategy.generate_signals(self.df)
        self.assertIn('signal', df_signals.columns)
        self.assertTrue(set(df_signals['signal'].unique()).issubset({-1, 0, 1}))

    def test_backtester_run(self):
        strategy = TradingStrategy(short_window=5, long_window=15)
        backtester = Backtester(initial_capital=10000.0, stop_loss_pct=0.05, take_profit_pct=0.10)
        results = backtester.run(self.df, strategy)

        self.assertIn('initial_capital', results)
        self.assertIn('final_value', results)
        self.assertIn('total_return_pct', results)
        self.assertIn('total_trades', results)
        self.assertIn('win_rate_pct', results)
        self.assertEqual(results['initial_capital'], 10000.0)

    def test_generate_simulated_ohlcv(self):
        sim_df = generate_simulated_ohlcv(num_candles=100)
        self.assertEqual(len(sim_df), 100)
        required_cols = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        self.assertTrue(required_cols.issubset(set(sim_df.columns)))

    @patch('ccxt.binance')
    def test_crypto_bot_mock(self, mock_exchange_class):
        mock_instance = MagicMock()
        mock_exchange_class.return_value = mock_instance

        # Mock fetch_ohlcv return
        mock_instance.fetch_ohlcv.return_value = [
            [1609459200000, 100, 105, 95, 102, 500] for _ in range(50)
        ]

        bot = CryptoBot(exchange_id='binance', sandbox=False)
        df = bot.fetch_ohlcv('BTC/USDT')

        self.assertEqual(len(df), 50)
        self.assertIn('close', df.columns)


if __name__ == '__main__':
    unittest.main()
