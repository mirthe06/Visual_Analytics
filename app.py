import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# -----------------------------
# LOAD DATA
# -----------------------------
DATA_PATH = "/Users/yeswanth/Desktop/VA/Dataset/cleaned_air_quality_2023_2024.csv"

df = pd.read_csv(DATA_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime")
df = df.dropna()

pollutants = df.columns.drop("datetime")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Model Controls")

target = st.sidebar.selectbox(
    "Pollutant to Forecast",
    pollutants
)

n_trees = st.sidebar.slider(
    "Number of Trees",
    10, 200, 50
)

max_depth = st.sidebar.slider(
    "Tree Depth",
    2, 20, 6
)

view_mode = st.sidebar.radio(
    "View Mode",
    ["Single Pollutant", "All Pollutants"]
)

# -----------------------------
# MODEL PREPARATION
# -----------------------------
features = df.drop(columns=["datetime", target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=n_trees,
    max_depth=max_depth,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

# difference between predicted and actual
difference = np.abs(pred - y_test)

# -----------------------------
# TITLE
# -----------------------------
st.title("Urban Air Quality Forecast Explorer")

st.write(
"""
This tool helps explore how a Random Forest model predicts air pollution.
Instead of focusing only on accuracy, the goal is to understand **how the model behaves**
and when it makes mistakes.
"""
)

# -----------------------------
# USER DESCRIPTION
# -----------------------------
st.header("Intended Users")

st.write("""
This application can be used by:

• **Environmental analysts** who want to understand pollution patterns in cities  
• **Public health researchers** studying links between pollution and diseases  
• **Policy makers** evaluating environmental risks  
• **Citizens or asthma patients** who want a clearer explanation of pollution conditions
""")

# -----------------------------
# PREDICTION DIFFERENCE PLOT
# -----------------------------
st.header("Prediction Error Visualization")

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": pred,
    "Error": difference
})

fig = px.scatter(
    results,
    x="Actual",
    y="Predicted",
    color="Error",
    color_continuous_scale="RdYlGn_r",
    title="Actual vs Predicted (Color shows prediction error)"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TIME SERIES VIEW
# -----------------------------
st.header("Forecast vs Measurement Over Time")

time_df = pd.DataFrame({
    "datetime": df.loc[y_test.index, "datetime"],
    "Actual": y_test,
    "Predicted": pred,
    "Error": difference
}).sort_values("datetime")

fig2 = px.scatter(
    time_df,
    x="datetime",
    y="Actual",
    color="Error",
    title="Prediction Error Over Time",
    color_continuous_scale="RdYlGn_r"
)

st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# ALL POLLUTANTS VIEW
# -----------------------------
if view_mode == "All Pollutants":

    st.header("All Pollutants Overview")

    melted = df.melt(
        id_vars="datetime",
        value_vars=pollutants,
        var_name="Pollutant",
        value_name="Value"
    )

    fig3 = px.line(
        melted,
        x="datetime",
        y="Value",
        color="Pollutant",
        title="All Pollutants Over Time"
    )

    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.header("Which Pollutants Influence the Prediction")

importance = pd.DataFrame({
    "feature": features.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

fig4 = px.bar(
    importance,
    x="importance",
    y="feature",
    orientation="h",
    title="Feature Importance"
)

st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# TREE EXPLORATION
# -----------------------------
st.header("Explore One Decision Tree")

tree_id = st.slider("Select Tree", 0, n_trees-1, 0)

fig5 = plt.figure(figsize=(20,10))

plot_tree(
    model.estimators_[tree_id],
    feature_names=features.columns,
    filled=True,
    max_depth=3
)

st.pyplot(fig5)

#streamlit run app.py