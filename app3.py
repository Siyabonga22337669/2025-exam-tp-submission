# =====================================================
# 🧠 Crime Data Streamlit Dashboard
# =====================================================
# This app shows EDA, Classification, and Forecasting results
# for your crime data analysis project.
# =====================================================

# -----------------------------
# 1️⃣ Import libraries
# -----------------------------
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from statsmodels.tsa.statespace.sarimax import SARIMAX

# -----------------------------
# 2️⃣ Load data with uploader
# -----------------------------
def load_data():
    uploaded_file = st.sidebar.file_uploader("Upload your crime CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Automatically detect the date column
            date_col = None
            for col in df.columns:
                if "date" in col.lower():
                    date_col = col
                    break
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
            else:
                st.warning("No date column detected. Some features may not work properly.")

            st.success("File uploaded successfully!")
            st.write("Columns detected:", df.columns.tolist())
            return df, date_col

        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            return None, None
    else:
        st.warning("Please upload a CSV file to continue.")
        st.stop()  # Stop the app until a file is uploaded

# -----------------------------
# 3️⃣ Load the data
# -----------------------------
df, date_col = load_data()

# -----------------------------
# 4️⃣ Sidebar Filters
# -----------------------------
st.sidebar.title("Crime Dashboard Controls")
st.sidebar.header("Filters")

crime_categories = df["crime_category"].unique().tolist()
selected_categories = st.sidebar.multiselect("Crime Category", crime_categories, default=crime_categories)

locations = df["location"].unique().tolist()
selected_locations = st.sidebar.multiselect("Location", locations, default=locations)

# Date range filter
if date_col:
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    date_range = st.sidebar.date_input("Date range", [min_date, max_date])
else:
    date_range = [None, None]

# Apply filters
filtered = df[
    (df["crime_category"].isin(selected_categories)) &
    (df["location"].isin(selected_locations))
]

if date_col and None not in date_range:
    filtered = filtered[
        filtered[date_col].between(date_range[0], date_range[1])
    ]

# -----------------------------
# 5️⃣ EDA Section
# -----------------------------
st.header("🔍 Exploratory Data Analysis")
st.write("Sample of filtered data:")
st.dataframe(filtered.head())

# Crime count by category
fig1 = px.histogram(
    filtered,
    x="crime_category",
    color="location",
    title="Crime Count by Category and Location"
)
st.plotly_chart(fig1)

# Crimes over time
if date_col:
    fig2 = px.line(
        filtered.groupby(date_col).size().reset_index(name="counts"),
        x=date_col,
        y="counts",
        title="Crimes Over Time"
    )
    st.plotly_chart(fig2)

# -----------------------------
# 6️⃣ Classification Results
# -----------------------------
st.header("🤖 Classification Model Results")

# Example placeholders (replace with your model outputs)
y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 0, 0, 1])

cm = confusion_matrix(y_true, y_pred)
st.subheader("Confusion Matrix")
fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig_cm)

st.subheader("Classification Report")
report = classification_report(y_true, y_pred, output_dict=True)
st.json(report)

# -----------------------------
# 7️⃣ Time Series Forecasting
# -----------------------------
st.header("📈 Time Series Forecasting")

if date_col:
    ts = filtered.set_index(date_col).resample("M").size()
    if len(ts) > 3:
        model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=False)

        forecast_steps = 12
        forecast_res = model_fit.get_forecast(steps=forecast_steps)
        forecast = forecast_res.predicted_mean
        ci = forecast_res.conf_int()

        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ts.plot(ax=ax3, label="Historical")
        forecast.plot(ax=ax3, label="Forecast", color="orange")
        ax3.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="orange", alpha=0.3)
        ax3.legend()
        ax3.set_title("Crime Count Forecast with Confidence Intervals")
        st.pyplot(fig3)
    else:
        st.warning("Not enough data for time series forecasting. Please use a larger dataset.")

# -----------------------------
# 8️⃣ Summary & Insights
# -----------------------------
st.header("🧭 Summary & Insights")

st.markdown("""
### Technical Summary
- Crime data can be filtered by category, location, and date range.
- EDA visualizations reveal temporal and spatial crime patterns.
- Classification evaluates prediction accuracy using confusion matrix and metrics.
- Time series forecasting predicts future crime trends with confidence intervals.

### Non-Technical (Layperson) Summary
This dashboard helps users explore how crime happens over time and across locations.
It shows where and when crimes are most frequent, predicts future patterns,
and gives insights for better decision-making and resource planning.
""")

# =====================================================
# End of App
# =====================================================
