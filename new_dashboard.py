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

# -----------------------------
# 1. LOAD DATA + TRANSLATE
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/yeswanth/Desktop/VA/Dataset/Visual_Analytics/city_pollutant_health_merged_v2.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").dropna(subset=['datetime']).reset_index(drop=True)
    # Use last 5000 rows for interactive performance on forecasting
    if len(df) > 5000:
        df = df.tail(5000).reset_index(drop=True)
    return df

df = load_data()

DISEASE_NAME_MAPPING = {
    'TotaalNieuwvormingen_8': 'Total Neoplasms (Cancer)',
    'TotaalEndocrieneVoedingsStofwZ_32': 'Endocrine & Metabolic Diseases',
    'TotaalPsychischeStoornissen_35': 'Mental Disorders',
    'TotaalZiektenVanHartEnVaatstelsel_43': 'Cardiovascular Diseases (Total)',
    'TotaalZiektenVanDeKransvaten_44': 'Coronary Heart Diseases (Total)',
    'k_711AcuutHartinfarct_45': 'Acute Heart Infarction',
    'k_712OverigeZiektenVanDeKransvaten_46': 'Other Coronary Heart Diseases',
    'k_72OverigeHartziekten_47': 'Other Heart Diseases',
    'TotaalZiektenVanDeAdemhalingsorganen_50': 'Respiratory System Diseases (Total)',
    'k_81Griep_51': 'Influenza (Flu)',
    'k_82Longontsteking_52': 'Pneumonia',
    'TotaalChronischeAandOndersteLucht_53': 'Chronic Lower Respiratory Diseases',
    'k_831Astma_54': 'Asthma',
    'k_832OvChronAandOndersteLuchtw_55': 'Other Chronic Lower Respiratory',
    'k_84OverigeZiektenAdemhalingsorganen_56': 'Other Respiratory Diseases',
    'TotaalZiektenSpierenBeendBindwfsl_64': 'Musculoskeletal & Connective Tissue',
    'k_111ReumatoideArtritisEnArtrose_65': 'Rheumatoid Arthritis & Osteoarthritis',
    'k_112OvZktnSpierenBeendBindwfsl_66': 'Other Musculoskeletal'
}

# Rename the columns in the dataframe immediately to English
df.rename(columns=DISEASE_NAME_MAPPING, inplace=True)

non_pollutant_cols = ['datetime', 'City', 'Year']
disease_columns = list(DISEASE_NAME_MAPPING.values())
pollutants = [col for col in df.columns if col not in non_pollutant_cols and col not in disease_columns]

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("App Navigation")
page = st.sidebar.radio("Go to:", ["1. Forecasting & Tuning", "2. Health & Pollutant Interactions"])

# ==========================================================
# PAGE 1: FORECASTING
# ==========================================================
if page == "1. Forecasting & Tuning":
    st.title("Air Quality Forecast Simulator")

    # Sidebar inputs
    st.sidebar.markdown("---")
    st.sidebar.header("Model Parameters")

    target = st.sidebar.selectbox("Target Pollutant to Forecast", pollutants, index=pollutants.index("NO2") if "NO2" in pollutants else 0)

    available_features = [p for p in pollutants if p != target]
    selected_features = st.sidebar.multiselect("Input Predictors (Features)", available_features, default=available_features[:4])

    n_trees = st.sidebar.slider("Number of Trees", min_value=1, max_value=200, value=50, step=10)
    max_depth = st.sidebar.slider("Tree Maximum Depth", min_value=1, max_value=20, value=6, step=1)

    # Default if no features selected
    if not selected_features:
        st.warning("Please select at least one predictor feature in the sidebar.")
        st.stop()

    # TRAIN MODEL
    rf_df = df.dropna(subset=selected_features + [target])

    if len(rf_df) < 50:
        st.error("Not enough data rows available after dropping missing values.")
        st.stop()

    features_df = rf_df[selected_features]
    y = rf_df[target]

    X_train, X_test, y_train, y_test = train_test_split(features_df, y, test_size=0.2, random_state=42, shuffle=False)
    test_dates = rf_df.loc[X_test.index, "datetime"]

    model = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)

    mcol1, mcol2 = st.columns(2)
    mcol1.metric("Overall R² Score (Accuracy fit)", f"{r2:.3f}")
    mcol2.metric("Mean Absolute Error (Prediction drift)", f"{mae:.3f}")

    # Layout columns for main content
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Time Series Prediction")
        ts_df = pd.DataFrame({"datetime": test_dates, "Actual": y_test, "Predicted": pred})
        
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=ts_df["datetime"], y=ts_df["Actual"], 
            mode='lines+markers', name='Actual', 
            line=dict(color='deepskyblue', width=1.5), marker=dict(size=4, opacity=0.6)
        ))
        
        fig_ts.add_trace(go.Scatter(
            x=ts_df["datetime"], y=ts_df["Predicted"], 
            mode='lines', name='Forecast', 
            line=dict(color='darkorange', width=2.5)
        ))

        error_values = np.abs(ts_df["Actual"].values - ts_df["Predicted"].values)
        if len(error_values) > 0:
            max_err_idx = np.argmax(error_values)
            fig_ts.add_annotation(
                x=ts_df["datetime"].iloc[max_err_idx], y=ts_df["Actual"].iloc[max_err_idx],
                text=f"Max Error: {error_values[max_err_idx]:.2f}",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="red"
            )
        
        fig_ts.update_layout(xaxis_title="Time", yaxis_title=target, height=450, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_ts, use_container_width=True)

    with c2:
        st.subheader("Residual / Error Spread")
        error_df = pd.DataFrame({"Error": y_test - pred})
        fig_err = px.histogram(error_df, x="Error", nbins=40, color_discrete_sequence=['red'])
        fig_err.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_err, use_container_width=True)

    with st.expander("Expand to view Component Importances (SHAP)"):
        with st.spinner("Calculating SHAP values (may take a moment)..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

        c3, c4 = st.columns(2)
        with c3:
            fig_shap, ax = plt.subplots(figsize=(6, 4))
            shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
            st.pyplot(fig_shap, clear_figure=True)

        with c4:
            fig_shap_bar, ax_bar = plt.subplots(figsize=(6, 4))
            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
            st.pyplot(fig_shap_bar, clear_figure=True)

    st.subheader("Random Forest Logic Tree Search")
    tree_id = st.slider("Select an Individual Tree branch to explore its logic splits:", 0, n_trees - 1, 0)
    
    st.markdown("### Dynamic Tree Insight")
    
    # Extract the logic mathematics of the specific selected tree decision path
    tree_estimator = model.estimators_[tree_id]
    tree_data = tree_estimator.tree_
    
    if tree_data.node_count > 1:
        root_feature_idx = tree_data.feature[0]
        root_feature_name = selected_features[root_feature_idx]
        root_threshold = tree_data.threshold[0]
        
        st.info(f"**Based on the specific tree #{tree_id} selected above:**\n\n"
                f"The most dominant indicator driving the prediction in this tree is **{root_feature_name}**. "
                f"The algorithm first splits the data by asking: *Is {root_feature_name} less than or equal to {root_threshold:.2f}?* \n\n"
                f"If True, it moves down the left path. If False, it moves right. This indicates that at this "
                f"exact step, {root_feature_name} had the strongest mathematical variance on {target}.")
    else:
        st.info("This tree is completely flat (1 logic node) and predicts a single value. Try increasing the 'Tree Maximum Depth' in the sidebar.")

    fig_tree = plt.figure(figsize=(20, 7))
    visual_depth = min(max_depth, 3) 
    plot_tree(tree_estimator, feature_names=selected_features, filled=True, max_depth=visual_depth, fontsize=10, rounded=True)
    st.pyplot(fig_tree, clear_figure=True)


# ==========================================================
# PAGE 2: HEALTH CORRELATION
# ==========================================================
elif page == "2. Health & Pollutant Interactions":
    st.title("Interactive Health Outcome Analytics")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Health Analytics Controls")
    
    # Allow the user to pick multple pollutants to correlate
    default_polls = [pollutants[0], pollutants[1]] if len(pollutants) >= 2 else pollutants
    selected_pollutants = st.sidebar.multiselect(
        "Select Multiple Pollutants for Combination Impact Analysis:", 
        pollutants, 
        default=default_polls
    )
    
    if not selected_pollutants:
        st.warning("Select at least one pollutant from the sidebar menu.")
        st.stop()
        
    # Correlation Math
    analysis_df = df[selected_pollutants + disease_columns].dropna()
    corr_df = analysis_df.corr()
    corr_subset = corr_df.loc[selected_pollutants, disease_columns]
    
    # Convert data into long format for Plotly grouped bar chart
    melt_corr = corr_subset.reset_index().melt(id_vars='index', var_name='Disease', value_name='Correlation')
    melt_corr.rename(columns={'index': 'Pollutant'}, inplace=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Simultaneous Pollutant Correlation")
        fig_bar = px.bar(
            melt_corr, 
            x="Correlation", 
            y="Disease", 
            color="Pollutant", 
            barmode='group',
            orientation='h',
            title="Impact of Selected Pollutants on Health Conditions"
        )
        # Adapt height based on how many diseases we are looking at
        fig_bar.update_layout(height=max(500, len(disease_columns)*30), margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("Dynamic Correlation Insight")
        
        # Auto-Generate Insights
        top_correlations = []
        positive_impacts = {}
        negative_impacts = {}
        
        for pol in selected_pollutants:
            series = corr_subset.loc[pol].dropna()
            
            if not series.empty:
                highest_disease = series.idxmax()
                highest_val = series[highest_disease]
                
                if highest_val > 0.05:
                    top_correlations.append(f"**{pol}**: Highly connected to *{highest_disease}* ({highest_val:.2f})")
                    positive_impacts[pol] = highest_disease
                elif highest_val < -0.05:
                    # If everything was negative, pick min
                    lowest_disease = series.idxmin()
                    lowest_val = series[lowest_disease]
                    top_correlations.append(f"**{pol}**: Negatively connected to *{lowest_disease}* ({lowest_val:.2f})")
                    negative_impacts[pol] = lowest_disease
                
        if len(melt_corr.dropna()) == 0:
            st.error("There is no overlapping temporal data for these specific pollutants and the health datasets to compute a mathmatical correlation.")
        elif top_correlations:
            # Build dynamic insight summary
            insight_text = "Based on your feature combinations selected in the sidebar:\n\n* " + "\n* ".join(top_correlations)
            insight_text += "\n\n**Dynamic AI Insight:**\n"
            
            if positive_impacts:
                pol_list = ", ".join(positive_impacts.keys())
                dis_list = ", ".join(set(positive_impacts.values()))
                insight_text += f" When **{pol_list}** emissions aggregate in surveyed municipal areas, historical medical data mathematically traces a proportional rise in incidents connected to **{dis_list}**.\n\n"
                
            if negative_impacts:
                pol_list = ", ".join(negative_impacts.keys())
                dis_list = ", ".join(set(negative_impacts.values()))
                insight_text += f" Conversely, increased measurements of **{pol_list}** negatively correlate with **{dis_list}**, indicating potential environmental trade-offs or seasonal displacements in health patterns.\n\n"
                
            st.info(insight_text.strip())
        else:
            st.info("The selected pollutants do not exhibit strong positive or negative correlations with any specific disease. Try adding/removing pollutants to discover patterns!")


    with st.expander("Expand to view Dense Correlation Heatmap (All Features)"):
        corr_all = df[pollutants + disease_columns].corr()
        corr_all_subset = corr_all.loc[disease_columns, pollutants] 
        fig_heatmap = px.imshow(
            corr_all_subset, 
            color_continuous_scale="RdBu_r", 
            zmin=-1, zmax=1, 
            aspect="auto",
        )
        fig_heatmap.update_layout(height=700)
        st.plotly_chart(fig_heatmap, use_container_width=True)
