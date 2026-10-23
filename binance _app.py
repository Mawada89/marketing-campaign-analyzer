
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

from datetime import datetime, timezone

# ---------------- إعدادات التطبيق ----------------

st.set_page_config(
    page_title="محلل Binance Spot",
    page_icon="📈",
    layout="wide"
)

st.title("📈 محلل Binance Spot")
st.caption(
    "دعم ومقاومة | انعكاس | اختراق | إطار 15 دقيقة"
)

BASE_URL = "https://data-api.binance.vision"
INTERVAL = "15m"

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT",
    "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "LINKUSDT", "LTCUSDT",
    "TRXUSDT", "AVAXUSDT", "DOTUSDT"
]

# ---------------- جلب بيانات السوق ----------------

@st.cache_data(ttl=60)
def get_candles(symbol):
    response = requests.get(
        f"{BASE_URL}/api/v3/klines",
        params={
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": 250
        },
        timeout=20
    )
    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list) or len(data) < 60:
        raise ValueError("بيانات السوق غير كافية.")

    columns = [
        "time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume",
        "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ]

    df = pd.DataFrame(data, columns=columns)

    df["time"] = pd.to_datetime(
        df["time"], unit="ms", utc=True
    )

    for col in [
        "open", "high", "low", "close", "volume"
    ]:
        df[col] = pd.to_numeric(df[col])

    # استبعاد الشمعة التي لم يحن وقت إغلاقها
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["time"] + pd.Timedelta(minutes=15) <= now]

    if len(df) < 60:
        raise ValueError("لا توجد شموع مكتملة كافية.")

    return df.reset_index(drop=True)


# ---------------- المؤشرات الفنية ----------------

def calculate_indicators(df):
    df = df.copy()

    df["EMA20"] = df["close"].ewm(
        span=20, adjust=False
    ).mean()

    df["EMA50"] = df["close"].ewm(
        span=50, adjust=False
    ).mean()

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(
        span=12, adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26, adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_signal"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    # ATR لقياس التذبذب
    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = tr.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    df["Volume_MA20"] = df["volume"].rolling(20).mean()

    return df


# ---------------- اكتشاف الدعم والمقاومة ----------------

def find_levels(df):
    # لا نستخدم أحدث ثلاث شموع في تحديد القمم والقيعان
    # لتقليل اعتماد المستويات على شموع لم تستقر بعد.
    history = df.iloc[:-3].tail(120).reset_index(drop=True)

    supports = []
    resistances = []

    # قمم وقيعان محلية محاطة بشمعتين من كل جانب
    for i in range(2, len(history) - 2):
        low = history.loc[i, "low"]
        high = history.loc[i, "high"]

        surrounding_lows = history.loc[
            i - 2:i + 2, "low"
        ].drop(index=i)

        surrounding_highs = history.loc[
            i - 2:i + 2, "high"
        ].drop(index=i)

        if low < surrounding_lows.min():
            supports.append(float(low))

        if high > surrounding_highs.max():
            resistances.append(float(high))

    return supports, resistances


def nearest_levels(price, supports, resistances, atr):
    # تجميع المستويات المتقاربة في منطقة واحدة
    tolerance = max(price * 0.0015, atr * 0.35)

    def cluster(values):
        if not values:
            return []

        groups = []

        for value in sorted(values):
            if not groups or value - groups[-1][-1] > tolerance:
                groups.append([value])
            else:
                groups[-1].append(value)

        return [float(np.mean(g)) for g in groups]

    support_zones = cluster(supports)
    resistance_zones = cluster(resistances)

    below = [x for x in support_zones if x < price]
    above = [x for x in resistance_zones if x > price]

    support = max(below) if below else None
    resistance = min(above) if above else None

    return support, resistance


# ---------------- تحليل الإشارات ----------------

def analyze_signal(df, support, resistance):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(last["close"])
    atr = float(last["ATR"])

    result = {
        "signal": "WAIT",
        "reason": "شروط الدخول غير مكتملة",
        "entry": None,
        "stop": None,
        "target1": None,
        "target2": None
    }

    if not np.isfinite(atr) or atr <= 0:
        result["reason"] = "بيانات التذبذب غير كافية"
        return result

    if not np.isfinite(last["RSI"]):
        result["reason"] = "بيانات المؤشرات غير كافية"
        return result

    # هامش تأكيد الاختراق
    buffer = max(price * 0.0005, atr * 0.10)

    bullish_candle = last["close"] > last["open"]
    bearish_candle = last["close"] < last["open"]

    macd_bullish = last["MACD"] > last["MACD_signal"]
    macd_bearish = last["MACD"] < last["MACD_signal"]

    volume_ok = (
        pd.notna(last["Volume_MA20"])
        and last["volume"] >= last["Volume_MA20"]
    )

    # 1. اختراق مقاومة بإغلاق مكتمل
    breakout = (
        resistance is not None
        and prev["close"] <= resistance + buffer
        and price > resistance + buffer
        and bullish_candle
        and macd_bullish
        and 50 <= last["RSI"] <= 70
        and volume_ok
    )

    # 2. ارتداد صاعد من الدعم
    bounce = (
        support is not None
        and last["low"] <= support + buffer
        and price > support + buffer
        and bullish_candle
        and macd_bullish
        and 35 <= last["RSI"] <= 60
    )

    # 3. رفض المقاومة: إشارة ضعف أو خروج محتمل
    rejection = (
        resistance is not None
        and last["high"] >= resistance - buffer
        and price < resistance
        and bearish_candle
        and macd_bearish
        and last["RSI"] >= 55
    )

    # 4. كسر الدعم: تحذير من استمرار الهبوط
    breakdown = (
        support is not None
        and prev["close"] >= support - buffer
        and price < support - buffer
        and bearish_candle
        and macd_bearish
        and last["RSI"] < 45
        and volume_ok
    )

    if breakout:
        result["signal"] = "BUY"
        result["reason"] = (
            "اختراق مقاومة بإغلاق مكتمل، "
            "مع تأكيد MACD وRSI وحجم التداول"
        )

    elif bounce:
        result["signal"] = "BUY"
        result["reason"] = (
            "ارتداد محتمل من الدعم مع شمعة صاعدة "
            "وتأكيد المؤشرات"
        )

    elif breakdown:
        result["signal"] = "SELL"
        result["reason"] = (
            "كسر الدعم مع تأكيد المؤشرات؛ "
            "إشارة لتقييم الخروج من العملة المملوكة"
        )

    elif rejection:
        result["signal"] = "SELL"
        result["reason"] = (
            "رفض سعري عند المقاومة مع علامات ضعف؛ "
            "إشارة لتقييم الخروج من العملة المملوكة"
        )

    if result["signal"] == "BUY":
        entry = price

        # وقف الخسارة يعتمد على ATR ومستوى الدعم
        stop = entry - 1.5 * atr

        if support is not None and support < entry:
            stop = min(stop, support - 0.15 * atr)

        risk = entry - stop

        if risk <= 0 or risk > 4 * atr:
            result["signal"] = "WAIT"
            result["reason"] = (
                "مخاطرة الدخول مرتفعة أو مستوى الوقف غير مناسب"
            )
            return result

        result.update({
            "entry": entry,
            "stop": stop,
            "target1": entry + 1.5 * risk,
            "target2": entry + 2.5 * risk
        })

    return result


# ---------------- واجهة المستخدم ----------------

symbol = st.selectbox("اختاري العملة", SYMBOLS)

if st.button("🔄 تحديث بيانات السوق"):
    st.cache_data.clear()

try:
    df = get_candles(symbol)
    df = calculate_indicators(df)

    supports, resistances = find_levels(df)

    last = df.iloc[-1]
    price = float(last["close"])
    atr = float(last["ATR"])

    support, resistance = nearest_levels(
        price, supports, resistances, atr
    )

    result = analyze_signal(df, support, resistance)

    st.caption(
        "آخر شمعة مكتملة: "
        + last["time"].strftime("%Y-%m-%d %H:%M UTC")
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("السعر المرجعي", f"{price:.8g}")
    c2.metric("RSI", f"{last['RSI']:.2f}")
    c3.metric("ATR", f"{atr:.8g}")

    st.subheader("الدعم والمقاومة")

    a, b = st.columns(2)

    a.metric(
        "الدعم الأقرب",
        f"{support:.8g}" if support is not None else "غير محدد"
    )

    b.metric(
        "المقاومة الأقرب",
        f"{resistance:.8g}" if resistance is not None else "غير محدد"
    )

    st.subheader("📌 إشارة التحليل")

    if result["signal"] == "BUY":
        st.success("🟢 شراء محتمل — لا تنفذ الصفقة تلقائيًا")

    elif result["signal"] == "SELL":
        st.error("🔴 إشارة ضعف أو خروج محتمل — Spot فقط")

    else:
        st.warning("🟡 انتظار — لا توجد إشارة مكتملة")

    st.write(result["reason"])

    if result["entry"] is not None:
        a, b, c, d = st.columns(4)

        a.metric("الدخول المرجعي", f"{result['entry']:.8g}")
        b.metric("وقف الخسارة", f"{result['stop']:.8g}")
        c.metric("الهدف الأول", f"{result['target1']:.8g}")
        d.metric("الهدف الثاني", f"{result['target2']:.8g}")

        st.caption(
            "المستويات تقديرية وليست أوامر فعلية. "
            "قد يتغير السعر قبل التنفيذ."
        )

    # ---------------- الرسم البياني ----------------

    st.subheader("📊 الشموع والدعم والمقاومة")

    chart = df.tail(100)

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=chart["time"],
        open=chart["open"],
        high=chart["high"],
        low=chart["low"],
        close=chart["close"],
        name="15m"
    ))

    fig.add_trace(go.Scatter(
        x=chart["time"],
        y=chart["EMA20"],
        name="EMA20"
    ))

    fig.add_trace(go.Scatter(
        x=chart["time"],
        y=chart["EMA50"],
        name="EMA50"
    ))

    if support is not None:
        fig.add_hline(
            y=support,
            line_dash="dash",
            annotation_text="Support"
        )

    if resistance is not None:
        fig.add_hline(
            y=resistance,
            line_dash="dash",
            annotation_text="Resistance"
        )

    fig.update_layout(
        height=550,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("مؤشرات إضافية")

    st.line_chart(
        df.tail(100).set_index("time")[["RSI"]]
    )

    st.line_chart(
        df.tail(100).set_index("time")[
            ["MACD", "MACD_signal"]
        ]
    )

    st.subheader("آخر الشموع المكتملة")

    st.dataframe(
        df.tail(15)[[
            "time", "open", "high", "low", "close",
            "volume", "RSI", "MACD"
        ]],
        use_container_width=True
    )

    report = df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "📥 تنزيل تقرير CSV",
        data=report,
        file_name=f"{symbol}_15m_analysis.csv",
        mime="text/csv"
    )

    st.caption(
        "تنبيه: الإشارات احتمالية وليست مضمونة. "
        "الكود لا يرسل أوامر تداول ولا يتصل بحسابك الخاص. "
        "لا يحدد الحكم الشرعي للعملات تلقائيًا."
    )

except requests.RequestException:
    st.error(
        "تعذر الاتصال ببيانات Binance. "
        "تحققي من الإنترنت وحاولي التحديث لاحقًا."
    )

except Exception as error:
    st.error("تعذر إكمال التحليل.")
    st.caption(f"تفاصيل الخطأ: {error}")
