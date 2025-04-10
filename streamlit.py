import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import numpy as np
import joblib
from PIL import Image
import base64



# Streamlit page config
st.set_page_config(page_title="Healthcare Demand & Shortage Prediction", layout="wide")
#st.title("🏥 Healthcare Demand & Shortage Prediction")

# Function to convert image to base64
def get_image_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load your logo and convert to base64
logo_base64 = get_image_base64("hos.png")

# Display logo + title
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{logo_base64}" width="200" style="margin-right: 15px;">
        <h1 style="margin: 0;">Healthcare Demand & Shortage Prediction</h1>
    </div>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_model1_and_scaler():
    model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model1, scaler = load_model1_and_scaler()

@st.cache_resource
def load_model2():
    model_sh = joblib.load("rf_model_shortage.pkl")
    return model_sh

model_sh = load_model2()

@st.cache_resource
def load_model3():
    model_avg = joblib.load("AVG_xgb.pkl")
    return model_avg

model_avg = load_model3()

# Apply Dark Theme using custom CSS
dark_theme_css = """
<style>
    /* Main background */
    body, .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }

    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #161a24;
    }

    /* Tabs and widgets */
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #ffffff;
        border: none !important;
        box-shadow: none !important;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #ffffff;
    }

    /* Input and select boxes */
    .stTextInput > div > div > input,
    .stNumberInput input,
    .stDateInput input,
    .stSelectbox > div > div,
    .stButton button {
        background-color: #21262d;
        color: #ffffff;
        border: 1px solid #30363d;
    }

    /* Buttons */
    .stButton button {
        background-color: #238636;
        color: white;
        border-radius: 0.4rem;
    }

    .stButton button:hover {
        background-color: #2ea043;
        color: white;
    }

    /* Markdown links */
    a {
        color: #58a6ff;
    }

    .stSuccess {
        background-color: #1d2b1f;
        color: #a3f7bf;
    }
</style>
"""

# Inject CSS
st.markdown(dark_theme_css, unsafe_allow_html=True)

# function needed for first model(demand forcasting regressor):

def predict_regressor(date):

    # extract year, month, day from the date
    year = date.year
    month = date.month
    day = date.day

    # seasonality calculation
    def compute_raw_seasonality_factor(month):
        a = -0.5119
        b = 0.5208 * np.pi / 12
        c = 2.8289
        d = 0.5004
        return a * np.sin(b * month + c) + d 
    
    min_original = 0.9004022675566267
    max_original = 1.0991446802709368
    min_raw = compute_raw_seasonality_factor(1)
    max_raw = compute_raw_seasonality_factor(12)

    def rescale_seasonality_factor(raw_value):
        return min_original + (raw_value - min_raw) * (max_original - min_original) / (max_raw - min_raw)

    def compute_seasonality_factor(month):
        raw_value = compute_raw_seasonality_factor(month)
        return rescale_seasonality_factor(raw_value)

    seasonality = compute_seasonality_factor(month)

    # Create feature vector and scale
    features = np.array([[year, month, day, seasonality]])
    scaled_features = scaler.transform(features)

    # Predict using the loaded model
    prediction = model1.predict(scaled_features)[0]
    return prediction

#########################################################################
# function needed to the second model (demand forecasting sarima model):

def predict_sarima(n_months):
    # Load saved SARIMA model
    model = joblib.load("sarima_model.pkl")
    
    # Load last 12 observed values (should be a pandas Series with datetime index)
    last_series = joblib.load("last_values.pkl")

    # Forecast next n_months
    forecast = model.forecast(steps=n_months)
    
    # Reverse seasonal differencing using last 12 logged values
    forecast = forecast.copy()
    for i in range(len(forecast)):
        forecast.iloc[i] += last_series[-12 + i]  # Assuming seasonal period = 12
    
    
    # Reverse log transformation
    forecast = np.expm1(forecast)

   
    # Generate future dates starting from the last available one
    last_date = last_series.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=n_months, freq='MS')
    forecast.index = forecast_dates
    
    return forecast

#########################################################################
# function needed for the third model (shortage classification model):

def predict_shortage(current_stock, min_required, avg_usage_per_day, restock_lead_time):
    
    # calculate reorder_point
    Reorder_Point = (avg_usage_per_day * restock_lead_time) + (min_required * 0.5)

    # Create feature vector and scale
    feature_df = pd.DataFrame([{
                "Current_Stock": current_stock,
                "Min_Required": min_required,
                "Avg_Usage_Per_Day": avg_usage_per_day,
                "Restock_Lead_Time": restock_lead_time,
                "Reorder_Point": (avg_usage_per_day * restock_lead_time) + (min_required * 0.5)
    }])

    # Predict using the loaded model
    prediction = model_sh.predict(feature_df)[0]
    return prediction

#########################################################################
# function needed for the fourth model (daily usage forecasting model):

def create_features(data):
                data = data.copy()
                data["month"] = data["Date"].dt.month
                data["day"] = data["Date"].dt.day
                data["weekday"] = data["Date"].dt.weekday
                data["quarter"] = data["Date"].dt.quarter

                data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
                data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
                data["day_sin"] = np.sin(2 * np.pi * data["day"] / 31)
                data["day_cos"] = np.cos(2 * np.pi * data["day"] / 31)
                data["weekday_sin"] = np.sin(2 * np.pi * data["weekday"] / 7)
                data["weekday_cos"] = np.cos(2 * np.pi * data["weekday"] / 7)
                data["quarter_cos"] = np.cos(2 * np.pi * data["quarter"] / 4)

                data = data.sort_values("Date")
                data["lag_1"] = data["Avg_Usage_Per_Day"].shift(1)
                data["lag_2"] = data["Avg_Usage_Per_Day"].shift(2)
                data["rolling_mean_3"] = data["Avg_Usage_Per_Day"].rolling(3).mean()

                return data
############################################################################################################################################

# Tabs for Demand Forecasting, Shortage Prediction, and Daily Usage Prediction

tab1, tab2, tab3 = st.tabs(["📈 Demand Forecasting", "🚨 Shortage Prediction", "📅 Daily Usage Forecasting"])

# Demand Forecasting Tab
with tab1:
    st.subheader("Monthly Demand Forecasting")
    col1, col2 = st.columns([2, 1])
    
    # Left: Demand visualization
    with col1:
        st.markdown("### Demand Over Time")
        df = pd.read_csv('deployment_version.csv')
        fig = px.line(df, x="Order_Date", y="Monthly_Demand", markers=True, title="Monthly Demand Trend")
        fig.update_layout(
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font_color='#ffffff'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Right: Prediction Models
    with col2:
        st.markdown("### Regressor Model")
        date_input_reg = st.date_input("Select Date (Regressor)", datetime.date.today(), key="regressor_date")
        if st.button("Predict (Regressor)"):
            prediction = predict_regressor(date_input_reg)
            st.success(f"Predicted Demand: {prediction}")
        
        st.markdown("### SARIMA Model")

        n_months = st.slider("Select Number of Months to Forecast (SARIMA)", 1, 12, 3)

        if st.button("Predict (SARIMA)"):
              forecast = predict_sarima(n_months)
              st.write("📅 Forecasted Demand:")
              st.dataframe(forecast.rename("Predicted Demand"))

    # Models Details and GitHub Section
    st.markdown("## 🔍 Curious About the Models?")
    st.write("If you're interested in how these models were built, check out the full technical details on GitHub.")
    
    # Define GitHub URLs
    github_links = {
           "Demand Forecasting Models": "https://github.com/dalya-gaber/healthcare-project/tree/Demand-forecasting-regressor-and-SARIMA-models",
           "Shortage Classifier & Daily Forecasting": "https://github.com/dalya-gaber/healthcare-project/tree/Shortage-classifier-and-daily-usage-forecasting"
    }

    # Create two columns for side-by-side buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📈 Demand Forecasting Models"):
            st.write(f"[View on GitHub]({github_links['Demand Forecasting Models']})")

    with col2:
        if st.button("⚠️ Shortage Classifier & Daily Forecasting"):
            st.write(f"[View on GitHub]({github_links['Shortage Classifier & Daily Forecasting']})")



# Shortage Risk Tab
with tab2:
    st.subheader("Shortage Risk Prediction")
    item_name = st.selectbox("Select Item", ["Ventilator", "Surgical Mask", "IV Drip", "Gloves", "X-ray Machine"])
    current_stock = st.number_input("Current Stock ( 60 : 500 )", min_value=0, step=1)
    min_required = st.number_input("Minimum Required ( 10 : 1000 )", min_value=0, step=1)
    avg_usage_per_day = st.number_input("Average Usage Per Day ( 2 : 500 )", min_value=0, step=1)
    restock_lead_time = st.number_input("Restock Lead Time ( days 1:30 )", min_value=0, step=1)
    
    def suggest_earlier_days(current_stock, avg_usage_per_day, min_required, restock_lead_time):
        days_until_shortage = (current_stock - min_required) / avg_usage_per_day
        suggested_lead_time = max(1, int(days_until_shortage - 1))  # buffer of 1 day
        earlier_days = max(0, restock_lead_time - suggested_lead_time)
        return earlier_days


    if st.button("Predict Shortage Risk"):
        risk = predict_shortage(current_stock, min_required, avg_usage_per_day, restock_lead_time)
        if risk == 1:
           earlier_days = suggest_earlier_days(current_stock, avg_usage_per_day, min_required, restock_lead_time)
           st.warning(
               f"⚠️ There is a **risk of facing a shortage** with the current data.\n\n"
               f"💡 To prevent it, consider **rescheduling the next order** so that restocking happens **{earlier_days} days earlier**."
           )
        else:
           st.success("✅ No shortage risk predicted. Stock level and restocking plan look safe!")
    
    

        
# Daily Usage Prediction Tab
with tab3:
    st.subheader("📅 Daily Usage Forecasting")

    # Load internal historical data (e.g., from CSV)
    df = pd.read_csv("avg_daily_usage.csv", parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    col1, col2 = st.columns([1, 2])

    # Left: Input & Forecast
    with col1:
        usage_date = st.date_input("Select Date for Forecasting (after Feb 12, 2026)", datetime.date.today(), key="usage_date")
        device_type = st.selectbox("Select Device Type", ["Ventilator", "Surgical Mask", "IV Drip", "Gloves", "X-ray Machine"])

        if st.button("Forecasting Daily Usage"):
           
           df = create_features(df)
           last_date = df["Date"].max()
           target_date = pd.to_datetime(usage_date)

           if (target_date - last_date).days == 1:
                # Scenario 1: Predict next day
                last_row = df.iloc[-1]
                input_features = last_row[[
                    "month_sin", "month_cos", "day_sin", "day_cos",
                    "weekday_sin", "weekday_cos", "quarter_cos",
                    "lag_1", "lag_2", "rolling_mean_3"
                ]].values.reshape(1, -1)

                prediction = model_avg.predict(input_features)[0]
                st.success(f"📈 Predicted Usage for {device_type} on {target_date.date()}: **{round(prediction, 2)} units**")

           elif (target_date - last_date).days > 1:
                # Scenario 2: Successive forecasts
                future_df = df.copy()
                future_df["Date"] = pd.to_datetime(future_df["Date"])
                future_df = future_df.sort_values("Date")

                while last_date < target_date:
                    next_date = last_date + pd.Timedelta(days=1)

                    # Manually create time-based features
                    next_features = {
                        "Date": next_date,
                        "month_sin": np.sin(2 * np.pi * next_date.month / 12),
                        "month_cos": np.cos(2 * np.pi * next_date.month / 12),
                        "day_sin": np.sin(2 * np.pi * next_date.day / 31),
                        "day_cos": np.cos(2 * np.pi * next_date.day / 31),
                        "weekday_sin": np.sin(2 * np.pi * next_date.weekday() / 7),
                        "weekday_cos": np.cos(2 * np.pi * next_date.weekday() / 7),
                        "quarter_cos": np.cos(2 * np.pi * next_date.quarter / 4),
                    }

                    # Get last known usage values for lag features
                    last_usages = future_df["Avg_Usage_Per_Day"].values
                    if len(last_usages) < 3:
                        st.error("❗ Not enough data to generate lag and rolling features.")
                        break

                    next_features.update({
                        "lag_1": last_usages[-1],
                        "lag_2": last_usages[-2],
                        "rolling_mean_3": np.mean(last_usages[-3:])
                    })


                    # Prepare input vector and predict
                    input_vector = pd.DataFrame([next_features]).drop(columns=["Date"])
                    pred_value = model_avg.predict(input_vector)[0]
                
                    # Append prediction
                    new_row = pd.DataFrame({
                        "Date": [next_date],
                        "Avg_Usage_Per_Day": [pred_value]
                    })
                    future_df = pd.concat([future_df, new_row], ignore_index=True)

                    last_date = next_date


                # Final result lookup
                future_df["Date"] = pd.to_datetime(future_df["Date"])
                matched_row = future_df.loc[future_df["Date"] == target_date]
               
                if not matched_row.empty:
                    forecasted_value = matched_row["Avg_Usage_Per_Day"].iloc[0]
                    st.success(f"🔁 Predicted Usage for {device_type} on {target_date.date()} after successive forecasting: **{round(forecasted_value, 2)} units**")

                    st.markdown("### 🔍 Forecasted Trend")
                    forecasted_data = future_df[future_df["Date"] > df["Date"].max()]
                    st.line_chart(forecasted_data.set_index("Date")["Avg_Usage_Per_Day"])
                else:
                    st.error("❗ Could not find the forecasted date in the generated data. Something went wrong.")
                    st.write("📅 Final date available in forecasted data:", future_df["Date"].max())
                    st.write("🎯 You asked for:", target_date)


            
        

    # Right: Past Daily Usage Visualization
    with col2:
        # Plot the historical usage
        st.markdown("### 📊 Past Daily Usage")
        fig = px.line(df, x="Date", y="Avg_Usage_Per_Day", markers=True)
        fig.update_layout(
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#ffffff"
        )
        st.plotly_chart(fig, use_container_width=True)

    