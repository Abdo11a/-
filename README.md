# Cryptocurrency Trading Bot & Backtester (بوت تداول العملات الرقمية)

A Python trading bot supporting technical indicators (SMA, RSI), strategy backtesting, stop-loss / take-profit risk management, and live exchange integration using CCXT.

بوت تداول عملات رقمية بلغة بايثون يدعم المؤشرات الفنية (المتوسطات المتحركة SMA ومؤشر القوة النسبية RSI)، واختبار الاستراتيجيات (Backtesting)، وإدارة المخاطر (جني الأرباح ووقف الخسارة)، والتكامل مع منصات التداول عبر مكتبة CCXT.

---

## Features / المميزات

1. **Technical Indicators / المؤشرات الفنية**:
   - Simple Moving Average (SMA - Short & Long) / المتوسط المتحرك البسيط.
   - Relative Strength Index (RSI) / مؤشر القوة النسبية.

2. **Strategy & Risk Management / الاستراتيجية وإدارة المخاطر**:
   - SMA Crossover signals (Golden Cross / Death Cross).
   - RSI overbought/oversold filtering.
   - Configurable Stop Loss & Take Profit logic.

3. **Backtesting & Simulation / المحاكاة والاختبار**:
   - Backtest on historical real market data or simulated price data.
   - Performance metrics: Total Return, Win Rate, Closed Trades, Equity Curve.

4. **Exchange Integration / التكامل مع المنصات**:
   - CCXT support for multi-exchange live market data (e.g., Binance, Kraken, KuCoin).

---

## Installation / التثبيت

Install dependencies using `pip`:

```bash
pip install pandas numpy ccxt
```

---

## Usage / كيفية الاستخدام

### 1. Simulated Backtest (اختبار باك تست افتراضي)
Run backtest using generated synthetic market data:
```bash
python3 trading_bot.py --mode simulated-backtest --capital 10000
```

### 2. Real Market Backtest (اختبار باك تست ببيانات سوق حقيقية)
Fetch historical candle data from Binance and run backtest:
```bash
python3 trading_bot.py --mode backtest --exchange binance --symbol BTC/USDT --capital 10000
```

### 3. Fetch Live Signal (جلب إشارة التداول الحالية)
Get current real-time market signal (BUY/SELL/HOLD):
```bash
python3 trading_bot.py --mode signal --exchange binance --symbol BTC/USDT
```

---

## Running Tests / تشغيل الاختبارات

Run unit test suite:
```bash
python3 -m unittest test_trading_bot.py
```
