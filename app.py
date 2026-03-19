import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
from sklearn.tree import plot_tree

st.set_page_config(layout="wide", page_title="Air Quality Visual Analytics Dashboard")

st.title("Air Quality Visual Analytics Dashboard")

# -----------------------------
# 1. LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/yeswanth/Desktop/VA/Dataset/Visual_Analytics/cleaned_air_quality_merged.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").dropna().reset_index(drop=True)
    # Use last 5000 rows for interactive performance
    if len(df) > 5000:
        df = df.tail(5000).reset_index(drop=True)
    return df

df = load_data()
pollutants = [col for col in df.columns if col != "datetime"]

# -----------------------------
# 5. MODEL IMPROVEMENT PANEL (SIDEBAR)
# -----------------------------
st.sidebar.header("Model Improvement Panel")

target = st.sidebar.selectbox("Target Pollutant to Forecast", pollutants, index=pollutants.index("NO2") if "NO2" in pollutants else 0)

available_features = [p for p in pollutants if p != target]
selected_features = st.sidebar.multiselect("Input Predictors (Features)", available_features, default=available_features[:4])

n_trees = st.sidebar.slider("Number of Trees", min_value=1, max_value=200, value=50, step=10)
max_depth = st.sidebar.slider("Tree Maximum Depth", min_value=1, max_value=20, value=6, step=1)

# Default if no features selected
if not selected_features:
    st.warning("Please select at least one predictor feature in the sidebar.")
    st.stop()

# -----------------------------
# TRAIN MODEL
# -----------------------------
features_df = df[selected_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(features_df, y, test_size=0.2, random_state=42, shuffle=False)
test_dates = df.loc[X_test.index, "datetime"]

model = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

st.subheader("Model Performance Metrics")
m1, m2, m3 = st.columns(3)
m1.metric("R² Score", f"{r2:.3f}")
m2.metric("Mean Absolute Error", f"{mae:.3f}")
m3.metric("Mean Squared Error", f"{mse:.3f}")
st.markdown("---")

# Layout columns for main content
c1, c2 = st.columns(2)

# -----------------------------
# 1. TIME SERIES FORECAST VISUALIZATION
# -----------------------------
with c1:
    st.header("Time Series Forecast Visualization")
    
    ts_df = pd.DataFrame({"datetime": test_dates, "Actual": y_test, "Predicted": pred})
    
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=ts_df["datetime"], y=ts_df["Actual"], mode='lines', name='Actual', line=dict(color='#1f77b4', width=2)))
    fig_ts.add_trace(go.Scatter(x=ts_df["datetime"], y=ts_df["Predicted"], mode='lines', name='Predicted', line=dict(color='#ff7f0e', width=2, dash='dot')))
    fig_ts.update_layout(
        title="Actual vs Predicted Pollutant Levels", 
        xaxis_title="Time", 
        yaxis_title=target, 
        height=400,
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_ts, use_container_width=True)

# -----------------------------
# 2. PREDICTION ERROR ANALYSIS
# -----------------------------
with c2:
    st.header("Prediction Error Analysis")
    
    error_df = pd.DataFrame({"Actual": y_test, "Predicted": pred, "Error": y_test - pred, "AbsError": np.abs(y_test - pred)})
    
    fig_err = px.histogram(error_df, x="Error", nbins=40, title="Error Distribution (Residuals)", 
                           color_discrete_sequence=['#ef553b'], template="plotly_white")
    fig_err.update_layout(
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fig_err.update_traces(marker_line_width=1, marker_line_color="black")
    st.plotly_chart(fig_err, use_container_width=True)

# -----------------------------
# 3. FEATURE IMPORTANCE VISUALIZATION (SHAP)
# -----------------------------
st.header("Feature Importance Visualization")

# SHAP values can be slow to compute, so we compute them on the test set
# Also wrap in st.spinner in case it takes a moment
with st.spinner("Calculating SHAP values..."):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

c3, c4 = st.columns(2)
with c3:
    st.subheader("SHAP Summary Plot")
    fig_shap, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
    fig_shap.patch.set_facecolor('none')
    ax.set_facecolor('none')
    plt.tight_layout()
    st.pyplot(fig_shap, clear_figure=True)

with c4:
    st.subheader("Feature Importance Bar Chart")
    fig_shap_bar, ax_bar = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    fig_shap_bar.patch.set_facecolor('none')
    ax_bar.set_facecolor('none')
    plt.tight_layout()
    st.pyplot(fig_shap_bar, clear_figure=True)

# -----------------------------
# 4. RANDOM FOREST MODEL STRUCTURE
# -----------------------------
st.header("Random Forest Model Structure")

tree_id = st.slider("Select an Individual Tree to Explore", 0, n_trees - 1, 0)
st.write(f"Visualizing Tree #{tree_id} (max visual depth limited to 3 for readability)")

fig_tree, ax_tree = plt.subplots(figsize=(20, 8))
plot_tree(model.estimators_[tree_id], feature_names=selected_features, filled=True, max_depth=3, fontsize=10, ax=ax_tree, rounded=True, proportion=True)
fig_tree.patch.set_facecolor('none')
ax_tree.set_facecolor('none')
plt.tight_layout()
st.pyplot(fig_tree, clear_figure=True)

#streamlit run app.py