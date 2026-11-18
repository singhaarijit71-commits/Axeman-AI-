
"""
AXEMAN AI - Indian Stock Market, Mutual Fund, ETF & Crypto Quantitative Analyst
-------------------------------------------------------------------------------
Author: Gemini AI
Description: A standalone Python web application powered by Streamlit, yfinance, 
and Plotly that performs technical & quantitative analysis on Indian Equities, 
ETFs, Mutual Funds, and Cryptocurrencies to generate actionable Buy/Sell/Hold 
and Long/Short signals.

Installation Requirements:
    pip install streamlit yfinance pandas numpy plotly

Usage:
    streamlit run axeman_ai.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ==============================================================================
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ==============================================================================
st.set_page_config(
    page_title="AXEMAN AI | Market & Crypto Intelligence",
    page_icon="🪓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a sleek, modern dark/adaptive financial terminal aesthetic
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF914D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888888;
        margin-bottom: 20px;
    }
    .signal-box-buy {
        padding: 20px;
        border-radius: 12px;
        background-color: rgba(0, 200, 83, 0.15);
        border: 1px solid #00C853;
        color: #00E676;
    }
    .signal-box-sell {
        padding: 20px;
        border-radius: 12px;
        background-color: rgba(255, 23, 68, 0.15);
        border: 1px solid #FF1744;
        color: #FF5252;
    }
    .signal-box-hold {
        padding: 20px;
        border-radius: 12px;
        background-color: rgba(255, 196, 0, 0.15);
        border: 1px solid #FFC400;
        color: #FFD740;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #777777;
        border-top: 1px solid #333333;
        padding-top: 10px;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. ASSET CATALOG (CURATED INDIAN ASSETS & CRYPTO)
# ==============================================================================
INDIAN_ASSETS = {
    "Indian Stocks (NSE)": {
        "Reliance Industries": "RELIANCE.NS",
        "Tata Consultancy Services (TCS)": "TCS.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "Infosys": "INFY.NS",
        "State Bank of India (SBI)": "SBIN.NS",
        "Bharti Airtel": "BHARTIARTL.NS",
        "ITC Limited": "ITC.NS",
        "Larsen & Toubro": "LT.NS",
        "Tata Motors": "TATAMOTORS.NS"
    },
    "Indian ETFs (NSE)": {
        "Nippon India ETF Nifty BeES": "NIFTYBEES.NS",
        "Nippon India ETF Bank BeES": "BANKBEES.NS",
        "Nippon India ETF Gold BeES": "GOLDBEES.NS",
        "Motilal Oswal Nasdaq 100 ETF": "MON100.NS",
        "Nippon India ETF Junior BeES": "JUNIORBEES.NS",
        "SBI Nifty Next 50 ETF": "SETFNN50.NS"
    },
    "Indian Mutual Funds (Proxies/AMFI)": {
        "SBI Bluechip Fund (Direct Growth)": "0P0000XVAA.BO",
        "Parag Parikh Flexi Cap Fund": "0P0000YWL1.BO",
        "HDFC Mid-Cap Opportunities Fund": "0P0000XW8F.BO",
        "Nippon India Small Cap Fund": "0P0000XVFY.BO",
        "ICICI Prudential Technology Fund": "0P0000XUZ6.BO"
    }
}

CRYPTO_ASSETS = {
    "Bitcoin (BTC/USD)": "BTC-USD",
    "Ethereum (ETH/USD)": "ETH-USD",
    "Solana (SOL/USD)": "SOL-USD",
    "Ripple (XRP/USD)": "XRP-USD",
    "Binance Coin (BNB/USD)": "BNB-USD",
    "Cardano (ADA/USD)": "ADA-USD",
    "Dogecoin (DOGE/USD)": "DOGE-USD",
    "Avalanche (AVAX/USD)": "AVAX-USD"
}

# ==============================================================================
# 3. AXEMAN AI CORE QUANTITATIVE ENGINE
# ==============================================================================
class AxemanAIEngine:
    """
    Core analytical engine for Axeman AI.
    Fetches historical OHLCV data, computes technical indicators, 
    calculates risk metrics, and generates weighted AI trading decisions.
    """

    @staticmethod
    @st.cache_data(ttl=300)  # Cache data for 5 minutes to prevent rate limiting
    def fetch_market_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Fetches historical market data from Yahoo Finance with error handling."""
        try:
            asset = yf.Ticker(ticker)
            df = asset.history(period=period, interval=interval)
            if df.empty:
                return pd.DataFrame()
            df.dropna(inplace=True)
            return df
        except Exception as e:
            st.error(f"Axeman Data Stream Error for {ticker}: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Computes RSI, MACD, Bollinger Bands, EMAs, and ATR."""
        data = df.copy()

        # 1. Exponential Moving Averages (Trend)
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
        data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()

        # 2. Relative Strength Index (Momentum - 14 period)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # 3. MACD (12, 26, 9)
        ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = ema_12 - ema_26
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

        # 4. Bollinger Bands (20-day SMA +/- 2 Std Dev)
        data['BB_Mid'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Mid'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Mid'] - (bb_std * 2)

        # 5. Average True Range (ATR - 14 period for Stop-Loss/Take-Profit)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        data['ATR'] = true_range.rolling(14).mean()

        return data

    @staticmethod
    def analyze_indian_asset(df: pd.DataFrame, asset_type: str) -> dict:
        """
        Generates Invest / Hold / Withdraw decisions for Indian Stocks, ETFs, and Mutual Funds.
        Uses a multi-factor scoring model (-100 to +100).
        """
        if len(df) < 50:
            return {"error": "Insufficient historical data for deep AI analysis."}

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        score = 0
        reasons = []

        # Factor 1: Long-Term Trend (EMA 50 vs EMA 200)
        if latest['Close'] > latest['EMA_200']:
            score += 25
            reasons.append("Price is trading above the 200-day EMA (Bullish primary trend).")
        else:
            score -= 25
            reasons.append("Price is below the 200-day EMA (Bearish primary trend).")

        if latest['EMA_20'] > latest['EMA_50']:
            score += 15
            reasons.append("Short-term 20 EMA is above 50 EMA (Positive momentum).")
        else:
            score -= 15
            reasons.append("Short-term 20 EMA has crossed below 50 EMA (Weakening momentum).")

        # Factor 2: RSI Valuation / Exhaustion
        rsi = latest['RSI']
        if rsi < 35:
            score += 25
            reasons.append(f"RSI is {rsi:.1f} (Oversold territory - attractive entry/accumulation zone).")
        elif rsi > 72:
            score -= 30
            reasons.append(f"RSI is {rsi:.1f} (Overbought territory - high risk of pullback, consider booking profits).")
        elif 45 <= rsi <= 65:
            score += 10
            reasons.append(f"RSI is healthy at {rsi:.1f} (Steady upward strength).")

        # Factor 3: MACD Momentum
        if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            score += 20
            reasons.append("Fresh Bullish MACD Crossover detected.")
        elif latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
            score -= 20
            reasons.append("Bearish MACD Crossover detected (Momentum shifting downward).")
        elif latest['MACD'] > latest['MACD_Signal']:
            score += 10
            reasons.append("MACD remains above signal line.")
        else:
            score -= 10
            reasons.append("MACD remains below signal line.")

        # Factor 4: Bollinger Band Extremes
        if latest['Close'] <= latest['BB_Lower']:
            score += 15
            reasons.append("Price touched lower Bollinger Band (Statistical support).")
        elif latest['Close'] >= latest['BB_Upper']:
            score -= 15
            reasons.append("Price pierced upper Bollinger Band (Stretched valuation).")

        # Final Decision Classification
        if score >= 30:
            action = "INVEST / ACCUMULATE"
            status = "buy"
            summary = f"Axeman AI detects strong upward momentum and favorable risk-reward for this {asset_type}."
        elif score <= -25:
            action = "WITHDRAW / BOOK PROFITS"
            status = "sell"
            summary = f"Axeman AI detects deteriorating trend structure or overbought exhaustion. Consider exiting or trimming exposure."
        else:
            action = "HOLD / SIP MODE"
            status = "hold"
            summary = f"Market signals are mixed/consolidating. Maintain existing positions or continue staggered SIPs."

        confidence = min(98, max(52, int(50 + abs(score) * 0.55)))

        return {
            "action": action,
            "status": status,
            "score": score,
            "confidence": confidence,
            "summary": summary,
            "reasons": reasons,
            "price": latest['Close'],
            "rsi": rsi,
            "support": latest['BB_Lower'],
            "resistance": latest['BB_Upper']
        }

    @staticmethod
    def analyze_crypto_chart(df: pd.DataFrame) -> dict:
        """
        Generates Long (Buy) / Short (Sell) / Neutral signals with ATR-based 
        Entry, Stop-Loss, and Take-Profit zones for Crypto trading.
        """
        if len(df) < 50:
            return {"error": "Insufficient chart candles for Axeman Crypto Engine."}

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        score = 0
        reasons = []

        # 1. Trend Alignment (EMA 20 vs 50)
        if latest['Close'] > latest['EMA_20'] > latest['EMA_50']:
            score += 30
            reasons.append("Strong Bullish Alignment: Price > 20 EMA > 50 EMA.")
        elif latest['Close'] < latest['EMA_20'] < latest['EMA_50']:
            score -= 30
            reasons.append("Strong Bearish Alignment: Price < 20 EMA < 50 EMA.")

        # 2. Momentum Shift (MACD Histogram)
        if latest['MACD_Hist'] > 0 and latest['MACD_Hist'] > prev['MACD_Hist']:
            score += 20
            reasons.append("Expanding bullish MACD histogram momentum.")
        elif latest['MACD_Hist'] < 0 and latest['MACD_Hist'] < prev['MACD_Hist']:
            score -= 20
            reasons.append("Expanding bearish MACD histogram pressure.")

        # 3. RSI Reversal / Continuation
        rsi = latest['RSI']
        if rsi < 30:
            score += 25
            reasons.append(f"Extreme Oversold RSI ({rsi:.1f}) - High probability bounce zone.")
        elif rsi > 70:
            score -= 25
            reasons.append(f"Extreme Overbought RSI ({rsi:.1f}) - Prime short/rejection zone.")
        elif 50 < rsi <= 68:
            score += 10
            reasons.append(f"Bullish RSI regime ({rsi:.1f}).")
        elif 32 <= rsi < 50:
            score -= 10
            reasons.append(f"Bearish RSI regime ({rsi:.1f}).")

        # 4. Volatility Breakout Check
        if latest['Close'] > latest['BB_Upper']:
            score += 15
            reasons.append("Upper Bollinger Band breakout detected.")
        elif latest['Close'] < latest['BB_Lower']:
            score -= 15
            reasons.append("Lower Bollinger Band breakdown detected.")

        entry_price = latest['Close']
        atr = latest['ATR'] if not np.isnan(latest['ATR']) else entry_price * 0.03

        if score >= 25:
            direction = "LONG (BUY)"
            status = "buy"
            stop_loss = entry_price - (1.5 * atr)
            take_profit_1 = entry_price + (2.0 * atr)
            take_profit_2 = entry_price + (3.5 * atr)
            summary = "Bullish confluence detected. Look for Long entries with strict ATR risk management."
        elif score <= -25:
            direction = "SHORT (SELL)"
            status = "sell"
            stop_loss = entry_price + (1.5 * atr)
            take_profit_1 = entry_price - (2.0 * atr)
            take_profit_2 = entry_price - (3.5 * atr)
            summary = "Bearish breakdown detected. Look for Short setups on minor pullbacks."
        else:
            direction = "NO TRADE / WAIT FOR BREAKOUT"
            status = "hold"
            stop_loss = entry_price - (1.5 * atr)
            take_profit_1 = entry_price + (1.5 * atr)
            take_profit_2 = entry_price + (2.5 * atr)
            summary = "Market is chopping inside a range. Wait for a clear breakout before deploying leverage."

        confidence = min(96, max(50, int(50 + abs(score) * 0.5)))

        return {
            "direction": direction,
            "status": status,
            "score": score,
            "confidence": confidence,
            "summary": summary,
            "reasons": reasons,
            "entry": entry_price,
            "stop_loss": stop_loss,
            "tp1": take_profit_1,
            "tp2": take_profit_2,
            "rsi": rsi,
            "atr": atr
        }

# ==============================================================================
# 4. INTERACTIVE CHARTING ENGINE (PLOTLY)
# ==============================================================================
def render_financial_chart(df: pd.DataFrame, title: str, currency_symbol: str, trade_levels: dict = None):
    """Renders an interactive Candlestick + EMA + Bollinger Bands + RSI + MACD chart."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{title} Price Action & AI Levels", "RSI (14)", "MACD Momentum")
    )

    # Row 1: Candlestick & Overlays
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Price"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='#00E5FF', width=1.3), name="20 EMA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='#FF9100', width=1.5), name="50 EMA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(255,255,255,0.2)', dash='dot'), name="BB Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(255,255,255,0.2)', dash='dot'), name="BB Lower"), row=1, col=1)

    # Add Crypto Long/Short Levels if provided
    if trade_levels and trade_levels.get("status") in ["buy", "sell"]:
        fig.add_hline(y=trade_levels['entry'], line_dash="solid", line_color="#FFFFFF", annotation_text=f"Entry: {currency_symbol}{trade_levels['entry']:,.2f}", row=1, col=1)
        fig.add_hline(y=trade_levels['stop_loss'], line_dash="dash", line_color="#FF1744", annotation_text=f"Stop Loss: {currency_symbol}{trade_levels['stop_loss']:,.2f}", row=1, col=1)
        fig.add_hline(y=trade_levels['tp1'], line_dash="dash", line_color="#00E676", annotation_text=f"Target 1: {currency_symbol}{trade_levels['tp1']:,.2f}", row=1, col=1)

    # Row 2: RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#B388FF', width=1.5), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#FF5252", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00E676", row=2, col=1)

    # Row 3: MACD
    colors = ['#00E676' if val >= 0 else '#FF5252' for val in df['MACD_Hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=colors, name="MACD Hist"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#40C4FF', width=1.2), name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='#FFAB40', width=1.2), name="Signal"), row=3, col=1)

    fig.update_layout(
        height=720,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 5. STREAMLIT USER INTERFACE & DASHBOARD ROUTING
# ==============================================================================
def main():
    # Header
    st.markdown('<div class="main-header">🪓 AXEMAN AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Autonomous Indian Market Wealth Advisor & Crypto Chart Intelligence</div>', unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.title("🧭 Axeman Command Center")
    module = st.sidebar.radio(
        "Select Intelligence Module:",
        ["🇮🇳 Indian Market Advisor (Stocks, ETFs & MF)", "₿ Crypto Chart Trader (Long / Short)", "🔍 Multi-Asset Portfolio Scanner"]
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**How Axeman Works:**\n"
        "Axeman fuses multi-timeframe trend tracking (20/50/200 EMAs), momentum oscillators (RSI & MACD), "
        "and volatility bands (ATR & Bollinger Bands) to evaluate optimal capital allocation."
    )

    # --------------------------------------------------------------------------
    # MODULE 1: INDIAN STOCK MARKET, ETF & MUTUAL FUND ADVISOR
    # --------------------------------------------------------------------------
    if module == "🇮🇳 Indian Market Advisor (Stocks, ETFs & MF)":
        st.subheader("🇮🇳 Indian Equity, ETF & Mutual Fund Allocation Engine")

        col1, col2, col3 = st.columns([1.2, 1.5, 1])
        with col1:
            asset_category = st.selectbox("Asset Class", list(INDIAN_ASSETS.keys()))
        with col2:
            selection_mode = st.radio("Ticker Selection", ["Curated List", "Custom NSE/BSE Ticker"], horizontal=True)
            if selection_mode == "Curated List":
                selected_name = st.selectbox("Select Instrument", list(INDIAN_ASSETS[asset_category].keys()))
                ticker = INDIAN_ASSETS[asset_category][selected_name]
            else:
                ticker = st.text_input("Enter Yahoo Finance India Ticker (e.g., ZOMATO.NS, INFY.NS)", value="RELIANCE.NS").upper()
                selected_name = ticker
        with col3:
            timeframe = st.selectbox("Analysis Horizon", ["6mo", "1y", "2y", "5y"], index=1)

        if st.button("🚀 Run Axeman Wealth Analysis", type="primary"):
            with st.spinner(f"Axeman AI is scanning {selected_name} ({ticker})..."):
                raw_df = AxemanAIEngine.fetch_market_data(ticker, period=timeframe, interval="1d")

                if raw_df.empty:
                    st.warning(f"No data returned for `{ticker}`. Ensure NSE tickers end with `.NS` and BSE tickers end with `.BO`.")
                else:
                    df = AxemanAIEngine.compute_indicators(raw_df)
                    result = AxemanAIEngine.analyze_indian_asset(df, asset_category)

                    if "error" in result:
                        st.error(result["error"])
                    else:
                        # Top Metrics Row
                        m1, m2, m3, m4 = st.columns(4)
                        prev_close = df['Close'].iloc[-2]
                        pct_change = ((result['price'] - prev_close) / prev_close) * 100

                        m1.metric("Current NAV / Price", f"₹{result['price']:,.2f}", f"{pct_change:+.2f}%")
                        m2.metric("Axeman Decision", result['action'])
                        m3.metric("AI Confidence", f"{result['confidence']}%", f"Score: {result['score']}/100")
                        m4.metric("14-Day RSI", f"{result['rsi']:.1f}", "Oversold <30 | Overbought >70")

                        # Decision Card
                        box_class = f"signal-box-{result['status']}"
                        st.markdown(f"""
                        <div class="{box_class}">
                            <h3>🪓 Axeman Verdict: {result['action']}</h3>
                            <p><strong>Executive Summary:</strong> {result['summary']}</p>
                            <p><strong>Key Support Zone:</strong> ₹{result['support']:,.2f} &nbsp;|&nbsp; <strong>Key Resistance Zone:</strong> ₹{result['resistance']:,.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("#### 🧠 Quantitative Reasoning Breakdown")
                        for reason in result['reasons']:
                            st.write(f"• {reason}")

                        # Interactive Chart
                        render_financial_chart(df.tail(180), selected_name, "₹")

    # --------------------------------------------------------------------------
    # MODULE 2: CRYPTO CHART TRADER (LONG / SHORT ENGINE)
    # --------------------------------------------------------------------------
    elif module == "₿ Crypto Chart Trader (Long / Short)":
        st.subheader("₿ Crypto Chart Intelligence — Long & Short Signal Generator")

        c1, c2, c3 = st.columns([1.5, 1, 1])
        with c1:
            crypto_mode = st.radio("Pair Selection", ["Top Cryptocurrencies", "Custom Crypto Ticker"], horizontal=True)
            if crypto_mode == "Top Cryptocurrencies":
                crypto_name = st.selectbox("Select Crypto Pair", list(CRYPTO_ASSETS.keys()))
                crypto_ticker = CRYPTO_ASSETS[crypto_name]
            else:
                crypto_ticker = st.text_input("Enter Crypto Ticker (e.g., PEPE-USD, LINK-USD)", value="BTC-USD").upper()
                crypto_name = crypto_ticker
        with c2:
            interval = st.selectbox("Chart Timeframe", ["15m", "1h", "4h", "1d"], index=1)
        with c3:
            # Map appropriate lookback periods for intraday vs daily intervals
            period_map = {"15m": "5d", "1h": "1mo", "4h": "3mo", "1d": "1y"}
            lookback = period_map[interval]
            st.text_input("Lookback Window", value=lookback, disabled=True)

        if st.button("⚡ Analyze Crypto Chart", type="primary"):
            with st.spinner(f"Axeman AI is reading candlestick structure for {crypto_name} ({interval})..."):
                raw_df = AxemanAIEngine.fetch_market_data(crypto_ticker, period=lookback, interval=interval)

                if raw_df.empty:
                    st.warning(f"Unable to fetch chart data for `{crypto_ticker}`. Use format `SYMBOL-USD`.")
                else:
                    df = AxemanAIEngine.compute_indicators(raw_df)
                    trade = AxemanAIEngine.analyze_crypto_chart(df)

                    if "error" in trade:
                        st.error(trade["error"])
                    else:
                        # Metrics
                        cm1, cm2, cm3, cm4 = st.columns(4)
                        cm1.metric("Live Mark Price", f"${trade['entry']:,.4f}" if trade['entry'] < 10 else f"${trade['entry']:,.2f}")
                        cm2.metric("Trade Signal", trade['direction'])
                        cm3.metric("Signal Confidence", f"{trade['confidence']}%")
                        cm4.metric("Volatility (ATR)", f"${trade['atr']:,.4f}" if trade['atr'] < 10 else f"${trade['atr']:,.2f}")

                        # Trade Setup Box
                        box_class = f"signal-box-{trade['status']}"
                        fmt = ",.4f" if trade['entry'] < 10 else ",.2f"
                        st.markdown(f"""
                        <div class="{box_class}">
                            <h3>🪓 Axeman Signal: {trade['direction']}</h3>
                            <p>{trade['summary']}</p>
                            <hr style="border-color: rgba(255,255,255,0.2);">
                            <p>
                                <strong>🎯 Optimal Entry Zone:</strong> ${trade['entry']:{fmt}} &nbsp;&nbsp;|&nbsp;&nbsp;
                                <strong>🛑 Stop-Loss (1.5x ATR):</strong> ${trade['stop_loss']:{fmt}} <br>
                                <strong>💰 Take-Profit 1 (Conservative):</strong> ${trade['tp1']:{fmt}} &nbsp;&nbsp;|&nbsp;&nbsp;
                                <strong>🚀 Take-Profit 2 (Runner):</strong> ${trade['tp2']:{fmt}}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("#### 📊 Technical Confluence Factors")
                        for r in trade['reasons']:
                            st.write(f"• {r}")

                        render_financial_chart(df.tail(150), f"{crypto_name} ({interval})", "$", trade_levels=trade)

    # --------------------------------------------------------------------------
    # MODULE 3: MULTI-ASSET SCREENER
    # --------------------------------------------------------------------------
    elif module == "🔍 Multi-Asset Portfolio Scanner":
        st.subheader("🔍 Axeman Batch Screener — Indian Market & Crypto")
        st.write("Scan all curated Indian Stocks, ETFs, Mutual Funds, and Cryptocurrencies simultaneously to find top Invest/Long and Withdraw/Short opportunities.")

        if st.button("🛰️ Scan All Markets Now", type="primary"):
            results_list = []
            progress = st.progress(0)
            all_items = []

            for cat, items in INDIAN_ASSETS.items():
                for name, tkr in items.items():
                    all_items.append((cat, name, tkr, "INR"))
            for name, tkr in CRYPTO_ASSETS.items():
                all_items.append(("Crypto (1D)", name, tkr, "USD"))

            total = len(all_items)
            for idx, (cat, name, tkr, curr) in enumerate(all_items):
                df_raw = AxemanAIEngine.fetch_market_data(tkr, period="1y", interval="1d")
                if not df_raw.empty and len(df_raw) >= 50:
                    df_ind = AxemanAIEngine.compute_indicators(df_raw)
                    if curr == "INR":
                        res = AxemanAIEngine.analyze_indian_asset(df_ind, cat)
                        action = res["action"]
                        price_str = f"₹{res['price']:,.2f}"
                    else:
                        res = AxemanAIEngine.analyze_crypto_chart(df_ind)
                        action = res["direction"]
                        price_str = f"${res['entry']:,.2f}"

                    results_list.append({
                        "Category": cat,
                        "Asset Name": name,
                        "Ticker": tkr,
                        "Last Price": price_str,
                        "RSI (14)": round(res["rsi"], 1),
                        "AI Score": res["score"],
                        "Confidence": f"{res['confidence']}%",
                        "Axeman Recommendation": action
                    })
                progress.progress((idx + 1) / total)

            if results_list:
                scan_df = pd.DataFrame(results_list).sort_values(by="AI Score", ascending=False)
                st.dataframe(scan_df, use_container_width=True, hide_index=True)

    # Footer Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Regulatory & Risk Disclaimer:</strong> Axeman AI is an algorithmic quantitative analysis tool built for educational and research purposes. 
        It is not a SEBI-registered investment advisor. Stock, Mutual Fund, and Cryptocurrency markets carry high financial risk. 
        Always verify signals and consult a certified financial planner before committing capital.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
