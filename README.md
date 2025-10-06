# 2025-exam-tp-submission

# Crime Data Analysis & Drone Simulation Dashboard

##  Overview

This project presents a **Crime Data Analysis Dashboard** built with **Streamlit**, integrated with a conceptual **Drone Simulation** for hotspot monitoring.
The aim is to analyze, predict, and visualize crime patterns, while exploring how drone technology can be used for crime prevention and urban monitoring.

---

##  Features

###  1. Streamlit Dashboard

An interactive web-based dashboard that provides:

* **Data Filtering** — Filter crimes by category, location, and time period.
* **Exploratory Data Analysis (EDA)** — Interactive charts and visual summaries of crime trends.
* **Classification Results** — Confusion matrix, accuracy, precision, recall, and F1-score metrics.
* **Time Series Forecasting** — ARIMA-based forecast of future crime trends with confidence intervals.
* **Summaries for All Users** — Clear explanations tailored for both technical and non-technical audiences.

###  2. Drone Simulation (Conceptual)

A simulated model of how drones can support **crime prevention** by:

* Mapping identified **crime hotspots** as Points of Interest (POIs).
* Visualizing a **3D grid area** (e.g., 1 km × 1 km × altitude) for patrol routes.
* Demonstrating how drones could navigate efficiently between POIs for monitoring and rapid response.

---

## Model Monitoring & Feedback

* Integrates potential **live updates** from SAPS or other public data APIs.
* Includes mechanisms to monitor **model drift** as new data becomes available.
* Recommends **periodic model retraining** to maintain prediction accuracy.

---

## Technologies Used

* **Python 3.10+**
* **Streamlit** for dashboard interface
* **Pandas, NumPy** for data processing
* **Matplotlib, Seaborn, Plotly** for data visualization
* **scikit-learn** for classification
* **Statsmodels (ARIMA)** for forecasting

---

## How to Run the App

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/crime-data-dashboard.git
   cd crime-data-dashboard
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```
4. The dashboard will open in your browser at:
   `http://localhost:8501`

---

##  Future Improvements

* Integration with **live SAPS data streams**.
* Enhanced drone path optimization algorithms.
* Deployment on cloud platforms (Streamlit Cloud / Heroku).
* Addition of machine learning models (e.g., Random Forest, XGBoost) for improved crime prediction.

---

 👤 Author

**[Siyabonga Mthembu]**
Bachelor of Computer and Information Technology
2025 TP Exam Project Submission
