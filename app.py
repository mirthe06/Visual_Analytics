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

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  body { font-family: 'Inter', sans-serif; }
  .spike-banner {
    background: linear-gradient(135deg, #1e1e2e, #23233a);
    border-left: 4px solid #f97316;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
    color: #fff;
  }
  .spike-banner h3 { margin: 0 0 4px 0; color: #f97316; }
  .heatmap-caption { font-size: 0.82rem; color: #94a3b8; margin-top: 4px; }
  .tree-badge {
    display: inline-block;
    background: #334155;
    border-radius: 5px;
    padding: 2px 7px;
    font-size: 0.78rem;
    color: #e2e8f0;
    margin: 2px;
  }
</style>
""", unsafe_allow_html=True)

st.title("🌿 Air Quality Visual Analytics Dashboard")

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/yeswanth/Desktop/VA/Dataset/Visual_Analytics/cleaned_air_quality_merged.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").dropna().reset_index(drop=True)
    if len(df) > 5000:
        df = df.tail(5000).reset_index(drop=True)
    return df

df = load_data()
pollutants = [col for col in df.select_dtypes(include=[np.number]).columns if col != "datetime"]

# ── APP NAVIGATION ────────────────────────────────────────────────────────────
view = st.sidebar.radio("App View", ["Forecasting & Tuning", "Correlation Explorer"], index=0)
if view == "Correlation Explorer":
    try:
        from correlation_view import render_relation
        render_relation()
    except Exception as e:
        st.error(f"Unable to load correlation explorer: {e}")
    st.stop()

# ── SIDEBAR: MODEL PANEL ──────────────────────────────────────────────────────
st.sidebar.header("Model Improvement Panel")

default_target = "n02_palmes"
target = st.sidebar.selectbox(
    "Target Pollutant to Forecast", pollutants,
    index=pollutants.index(default_target) if default_target in pollutants else 0
)

available_features = [p for p in pollutants if p != target]
selected_features = st.sidebar.multiselect(
    "Input Predictors (Features)", available_features, default=available_features[:4]
)

if not selected_features:
    st.warning("Please select at least one predictor feature in the sidebar.")
    st.stop()

features_df = df[selected_features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    features_df, y, test_size=0.2, random_state=42, shuffle=False
)
test_dates = df.loc[X_test.index, "datetime"]

tuning_obj = st.sidebar.radio("Tuning Objective", ["R2 (maximize)", "MSE (minimize)"], index=0)
mode = st.sidebar.selectbox("Mode", ["Manual", "Finding the Best Prediction"], index=0)

if "n_trees" not in st.session_state:
    st.session_state["n_trees"] = 50
if "max_depth" not in st.session_state:
    st.session_state["max_depth"] = 6

# ── AUTO-TUNE ─────────────────────────────────────────────────────────────────
if mode == "Finding the Best Prediction":
    st.sidebar.info("Running grid search — this may take a little while...")
    n_list = [10, 50, 100, 200]
    depth_list = [3, 6, 10, 15]
    results = []
    best_model = None
    best_pred = None
    best_params = (st.session_state["n_trees"], st.session_state["max_depth"])
    maximize_r2 = tuning_obj.startswith("R2")
    best_score = -np.inf if maximize_r2 else np.inf

    with st.spinner("Searching for best hyperparameters..."):
        for nt in n_list:
            for md in depth_list:
                clf = RandomForestRegressor(n_estimators=nt, max_depth=md, random_state=42, n_jobs=-1)
                clf.fit(X_train, y_train)
                yhat = clf.predict(X_test)
                mae_tmp = mean_absolute_error(y_test, yhat)
                r2_tmp = r2_score(y_test, yhat)
                mse_tmp = mean_squared_error(y_test, yhat)
                results.append({"n_estimators": nt, "max_depth": md, "MAE": mae_tmp, "MSE": mse_tmp, "R2": r2_tmp})
                if maximize_r2:
                    if r2_tmp > best_score:
                        best_score, best_model, best_pred, best_params = r2_tmp, clf, yhat, (nt, md)
                else:
                    if mse_tmp < best_score:
                        best_score, best_model, best_pred, best_params = mse_tmp, clf, yhat, (nt, md)

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("R2" if maximize_r2 else "MSE", ascending=not maximize_r2).reset_index(drop=True)
    best_n, best_md = best_params
    st.session_state["n_trees"] = int(best_n)
    st.session_state["max_depth"] = int(best_md)
    mse = mean_squared_error(y_test, best_pred)
    mae = mean_absolute_error(y_test, best_pred)
    r2 = r2_score(y_test, best_pred)
    st.session_state["best_summary"] = {"mse": float(mse), "mae": float(mae), "r2": float(r2), "maximize_r2": bool(maximize_r2)}
    st.session_state["best_res_df"] = res_df.head(10).to_dict()
    model = best_model
    pred = best_pred

n_trees = st.sidebar.slider("Number of Trees", min_value=11, max_value=200, value=st.session_state["n_trees"], step=10, key="n_trees")
max_depth = st.sidebar.slider("Tree Maximum Depth", min_value=1, max_value=20, value=st.session_state["max_depth"], step=1, key="max_depth")

if mode == "Manual":
    model = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

# ── METRICS ───────────────────────────────────────────────────────────────────
st.subheader("Model Performance Metrics")
m1, m2, m3 = st.columns(3)
m1.metric("R² Score", f"{r2:.3f}")
m2.metric("Mean Absolute Error", f"{mae:.3f}")
m3.metric("Mean Squared Error", f"{mse:.3f}")
st.markdown("---")

if mode == "Finding the Best Prediction" and "best_summary" in st.session_state:
    best_summary = st.session_state["best_summary"]
    best_df = pd.DataFrame(st.session_state["best_res_df"])
    st.info("Auto-tune applied. Below are metrics from the best found model:")
    b1, b2, b3 = st.columns(3)
    b1.metric("Chosen n_estimators", str(st.session_state.get("n_trees")))
    b2.metric("Chosen max_depth", str(st.session_state.get("max_depth")))
    if best_summary.get("maximize_r2"):
        b3.metric("Best R²", f"{best_summary['r2']:.3f}")
    else:
        b3.metric("Best MSE", f"{best_summary['mse']:.3f}")
    st.markdown("Top grid search results:")
    st.dataframe(pd.DataFrame(best_df))

# ══════════════════════════════════════════════════════════════════════════════
#  TIME SERIES FORECAST WITH SPIKE SELECTOR
# ══════════════════════════════════════════════════════════════════════════════
st.header("📈 Time Series Forecast Visualization")
st.caption("Select a spike region below to inspect tree decisions and correlation heatmap.")

# Build base dataframe
ts_df = pd.DataFrame({
    "datetime": test_dates.values,
    "Actual":    y_test.values,
    "Predicted": pred,
    "AbsError":  np.abs(y_test.values - pred),
}).reset_index(drop=True)

# ── Per-tree predictions (needed for voting control) ─────────────────────────
@st.cache_data(show_spinner=False)
def get_per_tree_predictions(_model, X):
    """Return array (n_trees, n_samples) of individual tree predictions."""
    return np.array([t.predict(X) for t in _model.estimators_])

X_test_reset = X_test.reset_index(drop=True)
tree_preds = get_per_tree_predictions(model, X_test_reset)   # (n_trees, n_test)

# ── Session state for tree voting ────────────────────────────────────────────
if "disabled_trees" not in st.session_state:
    st.session_state["disabled_trees"] = set()

# ── Modified prediction (after toggling trees) ────────────────────────────────
disabled_trees = st.session_state["disabled_trees"]
active_mask = np.ones(len(model.estimators_), dtype=bool)
for i in disabled_trees:
    if i < len(model.estimators_):
        active_mask[i] = False

active_tree_preds = tree_preds[active_mask]
if active_tree_preds.shape[0] > 0:
    modified_pred = active_tree_preds.mean(axis=0)
else:
    modified_pred = pred  # fallback

n_disabled = (~active_mask).sum()

# ── Build 3-line figure ───────────────────────────────────────────────────────
fig_ts = go.Figure()

# Actual
fig_ts.add_trace(go.Scatter(
    x=ts_df["datetime"], y=ts_df["Actual"],
    mode="lines", name="Actual",
    line=dict(color="#38bdf8", width=2)
))

# Original prediction (baseline)
fig_ts.add_trace(go.Scatter(
    x=ts_df["datetime"], y=ts_df["Predicted"],
    mode="lines", name="Original Prediction",
    line=dict(color="#fb923c", width=2, dash="dot"),
    opacity=0.75
))

# Modified prediction (after toggling trees)
if n_disabled > 0:
    fig_ts.add_trace(go.Scatter(
        x=ts_df["datetime"], y=modified_pred,
        mode="lines", name=f"Modified Prediction ({n_disabled} trees disabled)",
        line=dict(color="#a78bfa", width=2.5)
    ))

# Error spikes — dots sit ON the predicted line so the gap to actual is obvious
error_threshold = ts_df["AbsError"].quantile(0.90)
spike_mask = ts_df["AbsError"] >= error_threshold

# Build hover text for spike dots
spike_hover = [
    f"<b>High-Error Spike</b><br>Time: {d}<br>Predicted: {p:.3f}<br>Actual: {a:.3f}<br>Abs Error: {e:.3f}"
    for d, p, a, e in zip(
        ts_df.loc[spike_mask, "datetime"],
        ts_df.loc[spike_mask, "Predicted"],
        ts_df.loc[spike_mask, "Actual"],
        ts_df.loc[spike_mask, "AbsError"],
    )
]
fig_ts.add_trace(go.Scatter(
    x=ts_df.loc[spike_mask, "datetime"],
    y=ts_df.loc[spike_mask, "Predicted"],   # <── sits on the predicted line
    mode="markers", name="High-Error Spike",
    marker=dict(color="#f43f5e", size=9, symbol="circle",
                opacity=1.0, line=dict(color="#fff", width=1)),
    text=spike_hover, hoverinfo="text"
))

# ── determine a sensible default x-window (latest ~60 points or full range) ──
window_pts = min(60, len(ts_df))
range_start = str(ts_df["datetime"].iloc[-window_pts])[:10]
range_end   = str(ts_df["datetime"].iloc[-1])[:10]

fig_ts.update_layout(
    title="Actual vs Predicted Pollutant Levels — drag the rangeslider to pan/zoom",
    xaxis_title="Time", yaxis_title=target,
    height=520, template="plotly_dark",
    margin=dict(l=40, r=40, t=50, b=80),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
    # ── rangeslider (flow window) ──────────────────────────────────────────────
    xaxis=dict(
        rangeslider=dict(visible=True, thickness=0.06, bgcolor="#1e293b"),
        rangeselector=dict(
            buttons=[
                dict(count=7,  label="1W",  step="day",  stepmode="backward"),
                dict(count=14, label="2W",  step="day",  stepmode="backward"),
                dict(count=1,  label="1M",  step="month",stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="#1e293b", activecolor="#7c3aed",
            font=dict(color="#e2e8f0", size=11),
            x=0, y=1.08
        ),
        range=[range_start, range_end],   # default view = last ~60 pts
        type="date"
    ),
)
st.plotly_chart(fig_ts, use_container_width=True)

# ── Spike index selector ─────────────────────────────────────────────────────
spike_indices = ts_df.index[spike_mask].tolist()
if spike_indices:
    col_sp1, col_sp2 = st.columns([3, 1])
    with col_sp1:
        sel_spike_pos = st.select_slider(
            "🎯 Select a Spike Point to Inspect",
            options=list(range(len(spike_indices))),
            value=0,
            format_func=lambda i: str(ts_df.loc[spike_indices[i], "datetime"])[:16]
        )
    with col_sp2:
        st.metric("Abs Error at Spike", f"{ts_df.loc[spike_indices[sel_spike_pos], 'AbsError']:.3f}")

    sel_idx = spike_indices[sel_spike_pos]
    sel_date = ts_df.loc[sel_idx, "datetime"]
    sel_actual = ts_df.loc[sel_idx, "Actual"]
    sel_pred_orig = ts_df.loc[sel_idx, "Predicted"]
    sel_pred_mod  = modified_pred[sel_idx]

    st.markdown(f"""
    <div class="spike-banner">
      <h3>⚡ Spike at {sel_date}</h3>
      <b>Actual:</b> {sel_actual:.3f} &nbsp;|&nbsp;
      <b>Original Prediction:</b> {sel_pred_orig:.3f} &nbsp;|&nbsp;
      <b>Modified Prediction:</b> {sel_pred_mod:.3f} &nbsp;|&nbsp;
      <b>Error:</b> {sel_actual - sel_pred_orig:.3f}
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  TREE DECISIONS + CORRELATION HEATMAP + VOTING CONTROL
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    col_heat, col_trees = st.columns([1, 1])

    # ── Per-tree predictions at spike ────────────────────────────────────────
    tree_preds_at_spike = tree_preds[:, sel_idx]   # shape (n_trees,)
    spike_features = X_test_reset.iloc[sel_idx]     # single-row Series

    # ── Interactive Plotly Heatmap ────────────────────────────────────────────
    with col_heat:
        st.subheader("🌡️ Tree Vote × Feature Correlation Heatmap")
        st.caption(
            "**Click any cell** to inspect that tree's decision at the spike — "
            "hover shows correlation strength. 🔴 = disabled tree column."
        )

        window_start = max(0, sel_idx - 50)
        window_end   = min(len(X_test_reset), sel_idx + 51)
        X_window  = X_test_reset.iloc[window_start:window_end]
        tp_window = tree_preds[:, window_start:window_end]

        max_trees_heatmap = min(30, len(model.estimators_))
        corr_matrix = np.zeros((len(selected_features), max_trees_heatmap))
        for fi, feat in enumerate(selected_features):
            fv = X_window[feat].values
            for ti in range(max_trees_heatmap):
                tv = tp_window[ti]
                if np.std(fv) > 1e-9 and np.std(tv) > 1e-9:
                    corr_matrix[fi, ti] = np.corrcoef(fv, tv)[0, 1]

        # Build rich hover text: corr value + tree vote at spike + status
        hover_text = []
        for fi, feat in enumerate(selected_features):
            row_txt = []
            for ti in range(max_trees_heatmap):
                corr_val = corr_matrix[fi, ti]
                vote_val = tree_preds_at_spike[ti]
                status   = "⛔ DISABLED" if ti in disabled_trees else "✅ Active"
                strength = ("strong +" if corr_val > 0.5 else
                            "moderate +" if corr_val > 0.2 else
                            "strong −" if corr_val < -0.5 else
                            "moderate −" if corr_val < -0.2 else "weak")
                # Key insight line
                if abs(corr_val) > 0.4:
                    insight = "➜ High coupling: disabling may shift prediction"
                elif ti in disabled_trees:
                    insight = "➜ Already disabled"
                else:
                    insight = "➜ Low coupling: minor impact if disabled"

                row_txt.append(
                    f"<b>Tree T{ti} × {feat}</b><br>"
                    f"Pearson r: <b>{corr_val:.3f}</b> ({strength})<br>"
                    f"Tree vote at spike: <b>{vote_val:.3f}</b> (Actual: {sel_actual:.3f})<br>"
                    f"Error vs actual: <b>{vote_val - sel_actual:+.3f}</b><br>"
                    f"Status: {status}<br>{insight}"
                )
            hover_text.append(row_txt)

        # Colour disabled-tree columns — overlay a semi-transparent red mask
        # by making a separate z-layer of 1s only on disabled columns
        disabled_overlay = np.zeros((len(selected_features), max_trees_heatmap))
        for ti in disabled_trees:
            if ti < max_trees_heatmap:
                disabled_overlay[:, ti] = 1.0

        tree_labels = []
        for ti in range(max_trees_heatmap):
            tag = "🔴" if ti in disabled_trees else f"T{ti}"
            tree_labels.append(tag)

        fig_heat_px = go.Figure()

        # Main correlation heatmap
        fig_heat_px.add_trace(go.Heatmap(
            z=corr_matrix,
            x=tree_labels,
            y=selected_features,
            colorscale="RdYlGn",
            zmid=0, zmin=-1, zmax=1,
            text=hover_text,
            hoverinfo="text",
            colorbar=dict(
                title=dict(text="Pearson r", side="right",
                           font=dict(color="#94a3b8")),
                thickness=14, len=0.8,
                tickfont=dict(color="#cbd5e1"),
            ),
            showscale=True,
        ))

        # Disabled-tree red overlay (separate heatmap, single colour)
        if disabled_trees:
            fig_heat_px.add_trace(go.Heatmap(
                z=disabled_overlay,
                x=tree_labels,
                y=selected_features,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(244,63,94,0.22)"]],
                showscale=False,
                hoverinfo="skip",
            ))

        fig_heat_px.update_layout(
            title=dict(
                text="Feature — Tree Vote Correlation (±50-pt window around spike)",
                font=dict(color="#e2e8f0", size=12)
            ),
            height=max(280, len(selected_features) * 55 + 100),
            template="plotly_dark",
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            margin=dict(l=20, r=20, t=60, b=40),
            xaxis=dict(title="Tree", tickfont=dict(color="#94a3b8", size=10)),
            yaxis=dict(title="Feature", tickfont=dict(color="#94a3b8", size=10)),
            font=dict(color="#e2e8f0"),
        )

        # Capture click event via plotly_events or st.session_state trick
        # We use streamlit-plotly-events if available, else show instructions
        heatmap_click = None
        try:
            from streamlit_plotly_events import plotly_events
            click_data = plotly_events(fig_heat_px, click_event=True,
                                       override_height=max(280, len(selected_features)*55+100),
                                       key="heatmap_click")
            if click_data:
                heatmap_click = click_data[0]
        except ImportError:
            st.plotly_chart(fig_heat_px, use_container_width=True)
            st.caption("💡 Install `streamlit-plotly-events` for click-to-inspect functionality.")

        # ── Tree detail panel (shown when a cell is clicked or from selection) ─
        st.markdown("**🔎 Tree Detail Inspector**")
        col_ti_sel, col_ti_btn = st.columns([2, 1])
        with col_ti_sel:
            inspect_tree_idx = st.selectbox(
                "Inspect tree",
                options=list(range(max_trees_heatmap)),
                format_func=lambda i: f"T{i}{'  ⛔' if i in disabled_trees else ''}",
                index=heatmap_click["pointNumber"] if heatmap_click else 0,
                key="heatmap_inspect_tree"
            )
        with col_ti_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if inspect_tree_idx in disabled_trees:
                if st.button("✅ Re-enable this tree", key="re_enable_heatmap_tree"):
                    st.session_state["disabled_trees"].discard(inspect_tree_idx)
                    st.rerun()
            else:
                if st.button("⛔ Disable this tree", key="disable_heatmap_tree"):
                    st.session_state["disabled_trees"].add(inspect_tree_idx)
                    st.rerun()

        # Detailed stats for the chosen tree
        chosen_vote  = tree_preds_at_spike[inspect_tree_idx]
        chosen_err   = chosen_vote - sel_actual
        worst_abs    = float(np.abs(tree_preds_at_spike - sel_actual).max())
        err_rank     = int(np.sum(np.abs(tree_preds_at_spike - sel_actual) >=
                                  np.abs(chosen_err)))
        corr_for_tree = corr_matrix[:, inspect_tree_idx]   # per-feature
        top_feat_idx  = int(np.argmax(np.abs(corr_for_tree)))
        top_feat_name = selected_features[top_feat_idx]
        top_feat_corr = corr_for_tree[top_feat_idx]

        status_badge = "🔴 DISABLED" if inspect_tree_idx in disabled_trees else "🟢 Active"
        recommendation = (
            "⚠️ Consider disabling — large error & strong feature coupling."
            if abs(chosen_err) > 0.3 * worst_abs and abs(top_feat_corr) > 0.4
            else "✔️ Reasonable vote — low impact if disabled."
        )

        st.markdown(f"""
        <div style='background:#1e293b;border-radius:8px;padding:12px 16px;margin-top:6px;font-size:0.88rem'>
          <b style='color:#a78bfa'>Tree T{inspect_tree_idx}</b> &nbsp; {status_badge}<br><br>
          🎯 <b>Vote at spike:</b> <span style='color:#fb923c'>{chosen_vote:.4f}</span>
          &nbsp;|&nbsp;
          <b>Actual:</b> <span style='color:#38bdf8'>{sel_actual:.4f}</span>
          &nbsp;|&nbsp;
          <b>Error:</b> <span style='color:{"#f43f5e" if chosen_err > 0 else "#34d399"}'>{chosen_err:+.4f}</span><br>
          📊 <b>Error rank among trees:</b> #{err_rank} worst out of {max_trees_heatmap}<br>
          🔗 <b>Strongest feature coupling:</b> <code>{top_feat_name}</code>
          (r = {top_feat_corr:+.3f})<br>
          💡 {recommendation}
        </div>
        """, unsafe_allow_html=True)

    # ── Tree decisions at spike point ────────────────────────────────────────
    with col_trees:
        st.subheader("🌲 Tree Decisions at Spike")
        st.caption(
            "Shows each tree's predicted value at the selected spike point. "
            "The ensemble prediction is the mean of *active* (non-disabled) trees."
        )

        n_show = min(len(model.estimators_), max_trees_heatmap)
        tree_vote_df = pd.DataFrame({
            "Tree": [f"T{i}" for i in range(n_show)],
            "Prediction": tree_preds_at_spike[:n_show],
            "Active": ["✅ Active" if i not in disabled_trees else "⛔ Disabled" for i in range(n_show)],
            "Error_vs_Actual": tree_preds_at_spike[:n_show] - sel_actual
        })

        # Color by active status
        fig_votes = go.Figure()
        colors = ["#f43f5e" if i in disabled_trees else "#34d399" for i in range(n_show)]
        fig_votes.add_trace(go.Bar(
            x=tree_vote_df["Tree"],
            y=tree_vote_df["Prediction"],
            marker_color=colors,
            customdata=tree_vote_df[["Active", "Error_vs_Actual"]].values,
            hovertemplate="<b>%{x}</b><br>Prediction: %{y:.3f}<br>Status: %{customdata[0]}<br>Error vs Actual: %{customdata[1]:.3f}<extra></extra>"
        ))
        fig_votes.add_hline(y=sel_actual, line_dash="dash", line_color="#38bdf8", annotation_text="Actual", annotation_font_color="#38bdf8")
        fig_votes.add_hline(y=sel_pred_orig, line_dash="dot", line_color="#fb923c", annotation_text="Orig Pred", annotation_font_color="#fb923c")
        fig_votes.update_layout(
            title=f"Per-Tree Votes at Spike ({sel_date})"[:55],
            xaxis_title="Tree", yaxis_title="Predicted Value",
            height=380, template="plotly_dark",
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            showlegend=False,
            margin=dict(l=30, r=30, t=50, b=40)
        )
        st.plotly_chart(fig_votes, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  TREE VOTING CONTROL PANEL
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🗳️ Tree Voting Control — Toggle Trees On / Off")
    st.caption(
        "Disable individual trees to see how the modified prediction line changes. "
        "The purple line in the chart above updates to reflect your choices."
    )

    n_total = len(model.estimators_)
    n_cols_ctrl = 10
    tree_chunks = [list(range(i, min(i + n_cols_ctrl, n_total))) for i in range(0, n_total, n_cols_ctrl)]

    # Quick actions
    qa1, qa2, qa3 = st.columns(3)
    with qa1:
        if st.button("✅ Enable All Trees"):
            st.session_state["disabled_trees"] = set()
            st.rerun()
    with qa2:
        # Disable the top-N trees that are furthest from actual at spike
        if st.button("⛔ Disable Top-5 Worst Trees at Spike"):
            worst5 = np.argsort(np.abs(tree_preds_at_spike - sel_actual))[-5:]
            st.session_state["disabled_trees"] = set(int(i) for i in worst5)
            st.rerun()
    with qa3:
        if st.button("🔁 Reset to Baseline"):
            st.session_state["disabled_trees"] = set()
            st.rerun()

    st.markdown("")
    disabled_trees_local = st.session_state["disabled_trees"].copy()

    for chunk in tree_chunks:
        cols = st.columns(len(chunk))
        for j, ti in enumerate(chunk):
            is_active = ti not in disabled_trees_local
            with cols[j]:
                label = f"T{ti}"
                vote = tree_preds_at_spike[ti]
                delta = vote - sel_actual
                delta_str = f"{delta:+.2f}"
                checked = st.checkbox(label, value=is_active, key=f"tree_toggle_{ti}",
                                      help=f"Tree {ti} vote: {vote:.3f} (Δ {delta_str} vs actual)")
                if not checked:
                    disabled_trees_local.add(ti)
                elif ti in disabled_trees_local:
                    disabled_trees_local.discard(ti)

    if disabled_trees_local != st.session_state["disabled_trees"]:
        st.session_state["disabled_trees"] = disabled_trees_local
        st.rerun()

    # ── Summary of modified prediction impact ─────────────────────────────────
    n_active_now = n_total - len(disabled_trees_local)
    st.markdown(f"""
    <div style='background:#1e293b;border-radius:8px;padding:12px 16px;margin-top:10px'>
      <b style='color:#a78bfa'>Modified Prediction at spike:</b>
      <span style='color:#e2e8f0;font-size:1.1rem'>&nbsp;{sel_pred_mod:.4f}</span>
      &nbsp;&nbsp;|&nbsp;&nbsp;
      <b style='color:#fb923c'>Original Prediction:</b>
      <span style='color:#e2e8f0'>&nbsp;{sel_pred_orig:.4f}</span>
      &nbsp;&nbsp;|&nbsp;&nbsp;
      <b style='color:#38bdf8'>Actual:</b>
      <span style='color:#e2e8f0'>&nbsp;{sel_actual:.4f}</span>
      &nbsp;&nbsp;|&nbsp;&nbsp;
      <b style='color:#94a3b8'>Active trees:</b>
      <span style='color:#e2e8f0'>&nbsp;{n_active_now}/{n_total}</span>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("No spike points detected — all errors are below the 90th-percentile threshold. Try a different target or feature set.")

# ══════════════════════════════════════════════════════════════════════════════
#  ERROR ANALYSIS (side panels)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
c1, c2 = st.columns(2)
with c1:
    st.header("📊 Prediction Error Analysis")
    error_df_g = pd.DataFrame({"Actual": y_test.values, "Predicted": pred,
                                "Error": y_test.values - pred, "AbsError": np.abs(y_test.values - pred)})
    fig_err = px.histogram(error_df_g, x="Error", nbins=40, title="Error Distribution (Residuals)",
                           color_discrete_sequence=["#ef553b"], template="plotly_dark")
    fig_err.update_layout(height=380, margin=dict(l=30, r=30, t=40, b=30),
                          paper_bgcolor="#0f172a", plot_bgcolor="#0f172a")
    st.plotly_chart(fig_err, use_container_width=True)

with c2:
    st.header("🔝 Top 5 Absolute Errors")
    top_err_df = error_df_g.sort_values("AbsError", ascending=False).head(5)
    st.table(top_err_df[["Actual", "Predicted", "AbsError"]].reset_index(drop=True))

# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE (SHAP)
# ══════════════════════════════════════════════════════════════════════════════
st.header("🔍 Feature Importance Visualization")
with st.spinner("Calculating SHAP values..."):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

col_shap1, col_shap2 = st.columns(2)
with col_shap1:
    st.subheader("SHAP Summary Plot")
    fig_shap, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
    fig_shap.patch.set_facecolor("none")
    ax.set_facecolor("none")
    plt.tight_layout()
    st.pyplot(fig_shap, clear_figure=True)

with col_shap2:
    st.subheader("Feature Importance Bar Chart")
    fig_shap_bar, ax_bar = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    fig_shap_bar.patch.set_facecolor("none")
    ax_bar.set_facecolor("none")
    plt.tight_layout()
    st.pyplot(fig_shap_bar, clear_figure=True)

# ══════════════════════════════════════════════════════════════════════════════
#  RANDOM FOREST TREE EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
st.header("🌳 Random Forest Model Structure")
tree_id = st.slider("Select an Individual Tree to Explore", 0, n_trees - 1, 0)
st.write(f"Visualizing Tree #{tree_id} (max visual depth limited to 3 for readability)")
fig_tree, ax_tree = plt.subplots(figsize=(20, 8))
plot_tree(model.estimators_[tree_id], feature_names=selected_features, filled=True,
          max_depth=3, fontsize=10, ax=ax_tree, rounded=True, proportion=True)
fig_tree.patch.set_facecolor("none")
ax_tree.set_facecolor("none")
plt.tight_layout()
st.pyplot(fig_tree, clear_figure=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL UNDERSTANDING & IMPROVEMENT PROTOCOL
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("🔬 Model Understanding & Improvement Protocol", expanded=False):
    st.markdown("""
    This panel runs a structured analysis to understand what the model is really learning, expose weaknesses,
    and produce concrete improvement suggestions.
    """)
    if st.button("Run Model Analysis"):
        try:
            residuals = (y_test.values - pred)
            y_test_arr = y_test.values
            X_test_local = X_test.reset_index(drop=True)

            st.subheader("Feature importance")
            fi = getattr(model, "feature_importances_", None)
            if fi is not None:
                fi_df = pd.DataFrame({"feature": X_test_local.columns, "importance": fi})
                fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
                fig_fi = px.bar(fi_df, x="importance", y="feature", orientation="h",
                                title="Feature importances", template="plotly_dark")
                st.plotly_chart(fig_fi, use_container_width=True)

            st.subheader("Error distribution")
            fig_err_l = px.histogram(pd.DataFrame({"residuals": residuals}), x="residuals", nbins=50,
                                     title="Residual distribution", template="plotly_dark")
            st.plotly_chart(fig_err_l, use_container_width=True)
            st.write(f"95th pct abs error: {np.percentile(np.abs(residuals), 95):.3f}")

            st.subheader("Residuals vs Predicted")
            fig_rvp = px.scatter(x=pred, y=residuals, labels={"x": "Predicted", "y": "Residual"},
                                 trendline="lowess", title="Residuals vs Predicted", template="plotly_dark")
            fig_rvp.add_hline(y=0, line_dash="dash", line_color="white")
            st.plotly_chart(fig_rvp, use_container_width=True)

            st.subheader("Residual correlation with features")
            corr_list = []
            for col in X_test_local.columns:
                try:
                    c = np.corrcoef(X_test_local[col].astype(float), residuals.astype(float))[0, 1]
                except Exception:
                    c = np.nan
                corr_list.append((col, c))
            corr_df = pd.DataFrame(corr_list, columns=["feature", "residual_corr"]).dropna()
            corr_df = corr_df.sort_values("residual_corr", key=lambda s: s.abs(), ascending=False)
            st.dataframe(corr_df.head(10))

            st.subheader("Residuals over time")
            rtime_df = pd.DataFrame({"datetime": test_dates.values, "residuals": residuals})
            fig_rt = px.line(rtime_df.sort_values("datetime"), x="datetime", y="residuals",
                             title="Residuals over time", template="plotly_dark")
            fig_rt.add_hline(y=0, line_dash="dash", line_color="white")
            st.plotly_chart(fig_rt, use_container_width=True)

            st.subheader("Predicted vs Actual")
            pav = pd.DataFrame({"Actual": y_test_arr, "Predicted": pred})
            fig_pa = px.scatter(pav, x="Actual", y="Predicted", trendline="ols",
                                title="Predicted vs Actual", template="plotly_dark")
            fig_pa.add_shape(type="line", x0=pav["Actual"].min(), y0=pav["Actual"].min(),
                             x1=pav["Actual"].max(), y1=pav["Actual"].max(), line_dash="dash", line_color="white")
            st.plotly_chart(fig_pa, use_container_width=True)

        except Exception as e:
            st.error(f"Analysis failed: {e}")