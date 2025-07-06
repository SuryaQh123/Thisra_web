import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

st.set_page_config(page_title="Stock Preprocessing App", layout="wide")
st.title("📈 Stock Analysis and Preprocessing App")
#L5X1DGIJ0OVOELK1

# Now continue analysis only if data is fetched
if "stock_data" not in st.session_state:
    st.session_state.stock_data = None
if "data_fetched" not in st.session_state:
    st.session_state.data_fetched = False

# Stock list
stock_list = {
    "Apple": "AAPL", "Google": "GOOGL", "Microsoft": "MSFT", "Tesla": "TSLA", "Amazon": "AMZN",
    "NVIDIA": "NVDA", "Meta (Facebook)": "META", "Netflix": "NFLX", "Intel": "INTC", "AMD": "AMD",
    "Reliance": "RELIANCE.NS", "Infosys": "INFY.NS", "TCS": "TCS.NS", "HDFC Bank": "HDFCBANK.NS",
    "SBI": "SBIN.NS", "ICICI Bank": "ICICIBANK.NS", "Axis Bank": "AXISBANK.NS", "Wipro": "WIPRO.NS",
    "Bharti Airtel": "BHARTIARTL.NS", "Adani Enterprises": "ADANIENT.NS", "ITC": "ITC.NS",
    "Hindustan Unilever": "HINDUNILVR.NS", "Maruti Suzuki": "MARUTI.NS",
    "Bajaj Finance": "BAJFINANCE.NS", "HCL Tech": "HCLTECH.NS",
    # Added US
    "PayPal": "PYPL", "Salesforce": "CRM", "Adobe": "ADBE", "Qualcomm": "QCOM",
    "PepsiCo": "PEP", "Coca‑Cola": "KO", "McDonald’s": "MCD", "Boeing": "BA",
    "Ford": "F", "Walmart": "WMT", "Visa": "V", "Mastercard": "MA",
    "JPMorgan": "JPM", "Goldman Sachs": "GS", "ExxonMobil": "XOM",
    # Added India
    "LTIMindtree": "LTIM.NS", "Tata Motors": "TATAMOTORS.NS", "JSW Steel": "JSWSTEEL.NS",
    "UltraTech Cement": "ULTRACEMCO.NS", "Dr. Reddy’s": "DRREDDY.NS", "Sun Pharma": "SUNPHARMA.NS",
    "Tech Mahindra": "TECHM.NS", "ONGC": "ONGC.NS", "Coal India": "COALINDIA.NS",
    "Havells": "HAVELLS.NS", "Zomato": "ZOMATO.NS", "Nykaa": "NYKAA.NS",
    "DMart": "DMART.NS", "Bharat Electronics": "BEL.NS", "IndusInd Bank": "INDUSINDBK.NS"
}

company = st.selectbox("Select a Company:", list(stock_list.keys()))
ticker = stock_list[company]
min_date=datetime.date(2000,1,1)
max_date=datetime.date.today()
default_start=datetime.date(2010,1,1)
default_end=datetime.date(2020,1,1)

start_date = st.date_input("Start date",value=default_start,min_value=min_date,max_value=max_date)
end_date = st.date_input("End date",value=default_end,min_value=min_date,max_value=max_date)
#start_date = st.date_input("Start Date", value=datetime.date(2010, 1, 1), min_value=datetime.date(1990, 1, 1))
#end_date = st.date_input("End Date", value=datetime.date.today())

if st.button("Fetch Stock Data"):
    if start_date >= end_date:
        st.error("End date must be after start date.")
    else:
        with st.spinner("Fetching data..."):
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                st.warning("No data found for the selected range.")
                st.session_state.stock_data = None
                st.session_state.data_fetched = False
            else:
                st.session_state.stock_data = data
                st.session_state.data_fetched = True
                st.success("Data fetched successfully!")

# Now continue analysis only if data is fetched
if st.session_state.data_fetched:
    data = st.session_state.stock_data.copy()

    st.subheader("📊 First 5 Rows of the Dataset")
    st.dataframe(data.head())

    # Null check and handling
    st.subheader("🧹 Null Value Check and Handling")
    st.write("Null values before fillna:", data.isnull().sum())
    data.fillna(method='ffill', inplace=True)
    st.write("Null values after fillna:", data.isnull().sum())

    # ADF test
    st.subheader("🧪 ADF Test for Stationarity")
    result = adfuller(data['Close'].values)
    st.write("ADF Statistic:", result[0])
    st.write("p-value:", result[1])
    if result[1] < 0.05:
        st.success("✅ Reject Null Hypothesis — Data is Stationary.")
    else:
        st.warning("⚠ Fail to Reject Null Hypothesis — Data is Non-Stationary.")

    # Seasonality and Trend
    st.subheader("📉 Seasonality, Trend and Residual Plot")
    data_monthly = data.resample('M').sum()
    decomp = seasonal_decompose(data_monthly['Close'], model='additive')

    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    decomp.observed.plot(ax=axs[0], title='Observed')
    decomp.trend.plot(ax=axs[1], title='Trend')
    decomp.seasonal.plot(ax=axs[2], title='Seasonality')
    decomp.resid.plot(ax=axs[3], title='Residual')
    plt.tight_layout()
    st.pyplot(fig)

    # Volatility plot
    st.subheader("⚡ Volatility Check: Daily Returns")
    data['Returns'] = data['Close'].pct_change()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(data['Returns'], color='orange')
    ax2.set_title('Daily Returns')
    st.pyplot(fig2)




    # ----------------- 🔍 Optional View: Fundamental Analysis -----------------
    st.subheader("📌 Want to view fundamental metrics?")
    fundamental_choice = st.selectbox("Select 1 to show the fundamental analysis table:", options=[0, 1],
                                      key="fundamental_choice")
    if fundamental_choice == 1:
        st.subheader("📊 Fundamental Analysis Metrics")

        # Get the fundamental data using yfinance.Ticker
        stock_obj = yf.Ticker(ticker)
        info = stock_obj.info

        # Create the fundamental metrics dictionary
        fundamentals = {
            "Metric": [
                "P/E Ratio", "P/B Ratio", "EPS", "ROE", "Profit Margin",
                "Debt to Equity", "Market Cap", "Dividend Yield",
                "Book Value", "Total Revenue"
            ],
            "Value": [
                info.get("trailingPE", "N/A"),
                info.get("priceToBook", "N/A"),
                info.get("trailingEps", "N/A"),
                info.get("returnOnEquity", "N/A"),
                info.get("profitMargins", "N/A"),
                info.get("debtToEquity", "N/A"),
                info.get("marketCap", "N/A"),
                info.get("dividendYield", "N/A"),
                info.get("bookValue", "N/A"),
                info.get("totalRevenue", "N/A")
            ]
        }

        descriptions = {
            "P/E Ratio": "Price-to-Earnings (Valuation)",
            "P/B Ratio": "Price-to-Book (Asset Valuation)",
            "EPS": "Earnings Per Share",
            "ROE": "Return on Equity (%)",
            "Profit Margin": "Net Profit % of Revenue",
            "Debt to Equity": "Financial Leverage",
            "Market Cap": "Total Market Value of the Company",
            "Dividend Yield": "Return via Dividends (%)",
            "Book Value": "Net Asset Value per Share",
            "Total Revenue": "Gross Earnings"
        }

        fundamentals_df = pd.DataFrame(fundamentals)
        fundamentals_df["Description"] = fundamentals_df["Metric"].map(descriptions)

        st.dataframe(fundamentals_df)
        with st.expander("📘 Click to View Interpretation of Each Metric"):
            st.markdown("""
            ### 🧾 Fundamental Metrics Explained:

            - *📊 P/E Ratio (Price-to-Earnings)*  
              - 🟢 < 10: Possibly undervalued  
              - 🔵 10–25: Fairly valued  
              - 🔴 > 30: Possibly overvalued  

            - *📘 P/B Ratio (Price-to-Book)*  
              - 🟢 < 1: Undervalued (trading below book value)  
              - 🔵 1–3: Reasonable  
              - 🔴 > 3: May be overvalued  

            - *💰 EPS (Earnings Per Share)*  
              - 🔴 < 0: Negative earnings  
              - 🔵 0–5: Low to moderate earnings  
              - 🟢 > 5: Strong profitability  

            - *📈 ROE (Return on Equity)*  
              - 🔴 < 5%: Weak  
              - 🔵 5–15%: Average  
              - 🟢 > 15%: Strong  

            - *💵 Profit Margin*  
              - 🔴 < 5%: Low profitability  
              - 🔵 5–20%: Acceptable  
              - 🟢 > 20%: Excellent  

            - *⚖ Debt to Equity Ratio*  
              - 🟢 < 0.5: Low risk  
              - 🔵 0.5–1: Manageable  
              - 🔴 > 1: High leverage (riskier)  

            - *🏢 Market Capitalization*  
              - 🟢 > ₹50,000 Cr / $10B: Large-cap (Stable)  
              - 🔵 ₹10,000–50,000 Cr / $2B–10B: Mid-cap  
              - 🔴 < ₹10,000 Cr / $2B: Small-cap (Volatile)  

            - *💸 Dividend Yield*  
              - 🔴 < 1%: Low (growth-focused)  
              - 🔵 1–4%: Balanced  
              - 🟢 > 4%: Attractive for income investors  

            - *📚 Book Value*  
              - ℹ Indicates value of company’s assets per share.  
              - 🟢 Higher = stronger fundamentals  

            - *💼 Total Revenue*  
              - ℹ Reflects scale and business size.  
              - 🟢 Rising trend = business expansion  
            """)
    else:
        st.info("Fundamental metrics table is hidden. Select 1 to view.")
    st.subheader("📌 Want to view model recommendations?")
    choice = st.selectbox("Select 1 to show the model recommendation table:", options=[0, 1])
    if choice == 1:
        st.subheader("📊 Model Recommendation Table Based on Data Characteristics")
        st.markdown("""
        | Data Characteristic                | Recommended Models                                                                 |
        |-----------------------------------|-------------------------------------------------------------------------------------|
        | 📉 *Stationary*                 | ARIMA, SARIMA, Holt-Winters                                                        |
        | 🔄 *Seasonal Patterns*          | SARIMA, Prophet, Holt-Winters,Garch(1,1) Model                                                    |
        | 📈 *Trend Present*              | Holt’s Linear Trend, Prophet, SARIMA,Garch(1,1) Model                                              |
        | ⚡ *Volatility / Variance Shift*| GARCH, XGBoost, LSTM                                                               |
        | 🧠 *Non-linear / Complex Patterns* | LSTM, GRU, XGBoost, DeepAR,RandomForest                                                        |
        | 🔁 *Multiple Variables Involved*| VAR, SARIMAX, XGBoost                                                              |
        """)
    else:
        st.info("Table hidden. Select 1 to view the recommendation table.")
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from prophet import Prophet
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense,LSTM
    from sklearn.neural_network import MLPRegressor
    # Dataset already available
    df = data.copy()
    df['ds'] = df.index
    df['y'] = df['Close']


    # Forecast helper function
    def plot_forecast(original_df, forecast_df, title):
        plt.figure(figsize=(14, 6))
        plt.plot(original_df.index, original_df['Close'], label='Historical')
        plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='red')
        plt.axvline(original_df.index[-1], color='gray', linestyle='--', alpha=0.7)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(plt)


    # Feature engineering for ML models
    def create_features(df):
        df_feat = df.copy()
        df_feat['lag1'] = df_feat['Close'].shift(1)
        df_feat['lag2'] = df_feat['Close'].shift(2)
        df_feat['lag3'] = df_feat['Close'].shift(3)
        df_feat['dayofweek'] = df_feat.index.dayofweek
        df_feat['month'] = df_feat.index.month
        return df_feat.dropna()


    # Model selector
    model = st.selectbox("Select Forecasting Model",
                         ["ARIMA", "SARIMAX", "Prophet", "Holt-Winters", "Random Forest", "HistGradientBoosting",
                          "SVR","LSTM"])

    if st.button("Generate Forecast"):
        future_days = 252
        future_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='B')

        if model == "ARIMA":
            arima_model = ARIMA(data['Close'], order=(5, 1, 0))
            arima_fit = arima_model.fit()
            forecast = arima_fit.forecast(steps=future_days)
            forecast_df = pd.DataFrame({'Forecast': forecast.values}, index=future_index)
            plot_forecast(data, forecast_df, "ARIMA Forecast")

        elif model == "SARIMAX":
            sarimax_model = SARIMAX(data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            sarimax_fit = sarimax_model.fit(disp=False)
            forecast = sarimax_fit.forecast(steps=future_days)
            forecast_df = pd.DataFrame({'Forecast': forecast.values}, index=future_index)
            plot_forecast(data, forecast_df, "SARIMAX Forecast")

        elif model == "Prophet":
            m = Prophet()
            m.fit(df[['ds', 'y']])
            future = m.make_future_dataframe(periods=future_days)
            forecast = m.predict(future)
            forecast_df = forecast[['ds', 'yhat']].set_index('ds').rename(columns={'yhat': 'Forecast'})
            forecast_df = forecast_df[forecast_df.index > df.index[-1]]
            plot_forecast(df.set_index('ds'), forecast_df, "Prophet Forecast")

        elif model == "Holt-Winters":
            hw_model = ExponentialSmoothing(data['Close'], trend='add', seasonal='add', seasonal_periods=12)
            hw_fit = hw_model.fit()
            forecast = hw_fit.forecast(steps=future_days)
            forecast_df = pd.DataFrame({'Forecast': forecast.values}, index=future_index)
            plot_forecast(data, forecast_df, "Holt-Winters Forecast")

        elif model == "Random Forest":
            df_feat = create_features(data)
            X = df_feat[['lag1', 'lag2', 'lag3', 'dayofweek', 'month']]
            y = df_feat['Close']
            rf_model = RandomForestRegressor()
            rf_model.fit(X, y)

            last_known = df_feat.iloc[-3:].copy()
            preds = []

            for date in future_index:
                features = {
                    'lag1': last_known.iloc[-1]['Close'],
                    'lag2': last_known.iloc[-2]['Close'],
                    'lag3': last_known.iloc[-3]['Close'],
                    'dayofweek': date.dayofweek,
                    'month': date.month
                }
                X_pred = pd.DataFrame([features])
                pred = rf_model.predict(X_pred)[0]
                preds.append(pred)

                # Update last_known to maintain lag logic
                new_row = pd.DataFrame({'Close': [pred]}, index=[date])
                last_known = pd.concat([last_known, new_row])
                last_known = last_known.iloc[-3:]

            forecast_df = pd.DataFrame({'Forecast': preds}, index=future_index)
            plot_forecast(data, forecast_df, "Random Forest Forecast")

        elif model == "HistGradientBoosting":
            df_feat = create_features(data)
            X = df_feat[['lag1', 'lag2', 'lag3', 'dayofweek', 'month']]
            y = df_feat['Close']
            hgb_model = HistGradientBoostingRegressor()
            hgb_model.fit(X, y)

            last_known = df_feat.iloc[-3:].copy()
            preds = []

            for date in future_index:
                features = {
                    'lag1': last_known.iloc[-1]['Close'],
                    'lag2': last_known.iloc[-2]['Close'],
                    'lag3': last_known.iloc[-3]['Close'],
                    'dayofweek': date.dayofweek,
                    'month': date.month
                }
                X_pred = pd.DataFrame([features])
                pred = hgb_model.predict(X_pred)[0]
                preds.append(pred)

                new_row = pd.DataFrame({'Close': [pred]}, index=[date])
                last_known = pd.concat([last_known, new_row])
                last_known = last_known.iloc[-3:]

            forecast_df = pd.DataFrame({'Forecast': preds}, index=future_index)
            plot_forecast(data, forecast_df, "HistGradientBoosting Forecast")

        elif model == "SVR":
            df_feat = create_features(data)
            X = df_feat[['lag1', 'lag2', 'lag3', 'dayofweek', 'month']]
            y = df_feat['Close']

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            svr_model = SVR(kernel='rbf')
            svr_model.fit(X_scaled, y)

            last_known = df_feat[['Close']].copy()
            lag_values = list(last_known['Close'].iloc[-3:].values)

            preds = []

            for date in future_index:
                features = {
                    'lag1': lag_values[-1],
                    'lag2': lag_values[-2],
                    'lag3': lag_values[-3],
                    'dayofweek': date.dayofweek,
                    'month': date.month
                }
                X_pred = pd.DataFrame([features])
                X_pred_scaled = scaler.transform(X_pred)
                pred = svr_model.predict(X_pred_scaled)[0]
                preds.append(pred)

                lag_values.append(pred)
                lag_values = lag_values[-3:]

            forecast_df = pd.DataFrame({'Forecast': preds}, index=future_index)
            plot_forecast(data, forecast_df, "SVR Forecast")
        #elif model == "ANN":

        elif model == "LSTM":
            df_lstm = data.copy()
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df_lstm[['Close']])

            sequence_length = 60
            X, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i - sequence_length:i, 0])
                y.append(scaled_data[i, 0])

            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            lstm_model = Sequential()
            lstm_model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
            lstm_model.add(Dense(1))
            lstm_model.compile(optimizer='adam', loss='mse')
            lstm_model.fit(X, y, epochs=10, batch_size=32, verbose=0)

            last_seq = scaled_data[-sequence_length:]
            preds = []
            for _ in range(future_days):
                input_seq = last_seq[-sequence_length:].reshape(1, sequence_length, 1)
                pred = lstm_model.predict(input_seq, verbose=0)[0][0]
                preds.append(pred)
                last_seq = np.append(last_seq, [[pred]], axis=0)

            preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
            forecast_df = pd.DataFrame({'Forecast': preds}, index=future_index)
            plot_forecast(data, forecast_df, "LSTM Forecast")
        st.markdown("""
            ## 🙏 Thank You!
            We're grateful you used our app. Hope the forecasts help you make smart decisions.

            **Made with ❤️ using Streamlit.**
           """)
