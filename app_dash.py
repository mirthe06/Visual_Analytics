import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import matplotlib
matplotlib.use('Agg') # Required for non-interactive backend
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import io
import base64
import textwrap

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

# ── 1. LOAD & INITIALIZE DATA ────────────────────────────────────────────────
def load_forecast_data():
    df = pd.read_csv("/Users/yeswanth/Desktop/VA/Dataset/Visual_Analytics/cleaned_air_quality_merged.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").dropna().reset_index(drop=True)
    if len(df) > 5000:
        df = df.tail(5000).reset_index(drop=True)
    return df

def load_merged_data():
    path = "/Users/yeswanth/Desktop/VA/Dataset/Visual_Analytics/city_pollutant_health_merged_v2.csv"
    df = pd.read_csv(path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        date_cols = [c for c in df.columns if c.lower().startswith('date')]
        if date_cols:
            df["datetime"] = pd.to_datetime(df[date_cols[0]])
    return df

df_forecast = load_forecast_data()
pollutants_forecast = [col for col in df_forecast.select_dtypes(include=[np.number]).columns if col != "datetime"]

df_merged_full = load_merged_data()
df_merged_full.rename(columns=DISEASE_NAME_MAPPING, inplace=True)

# ── 2. UTILS ─────────────────────────────────────────────────────────────────
def fig_to_uri(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', facecolor='none')
    img.seek(0)
    encoded = base64.b64encode(img.getvalue()).decode('utf-8')
    return 'data:image/png;base64,{}'.format(encoded)

# ── 3. DASH APP SETUP ────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, "https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap"], suppress_callback_exceptions=True)
app.title = "Air Quality Analytics | Dash"

# ── 4. ASYNC MODEL WRAPPER ───────────────────────────────────────────────────
def get_model_and_preds(target, selected_features, n_trees, max_depth):
    X = df_forecast[selected_features]
    y = df_forecast[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    model = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, max_features="sqrt", n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    X_test_reset = X_test.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)
    test_dates = df_forecast.loc[X_test.index, "datetime"].reset_index(drop=True)
    
    orig_pred = model.predict(X_test_reset)
    tree_preds = np.array([t.predict(X_test_reset) for t in model.estimators_])
    
    return model, tree_preds, X_test_reset, y_test_reset, test_dates, orig_pred

# ── 5. LAYOUT DEFINITION ─────────────────────────────────────────────────────
app.layout = dbc.Container([
    # Stores
    dcc.Store(id='model-data-store'),
    dcc.Store(id='disabled-trees-store', data=[]),
    dcc.Store(id='selection-idx-store', data=0),
    dcc.Store(id='tree-image-store'),

    dbc.Row([
        dbc.Col(html.H1("🌿 Air Quality Visual Analytics Dashboard", className="text-center py-4", style={'fontFamily': 'Inter'}), width=12)
    ]),

    dcc.Tabs(id="tabs-navigation", value='tab-forecasting', children=[
        dcc.Tab(label='Forecasting & Tuning', value='tab-forecasting', className='custom-tab', selected_className='custom-tab--selected', style={'backgroundColor': '#1e1e2e', 'color': '#94a3b8'}),
        dcc.Tab(label='Correlation Explorer', value='tab-correlation', className='custom-tab', selected_className='custom-tab--selected', style={'backgroundColor': '#1e1e2e', 'color': '#94a3b8'}),
    ], style={'marginBottom': '20px'}),

    html.Div(id='tabs-content')
], fluid=True, style={'backgroundColor': '#0b0f19', 'minHeight': '100vh', 'color': '#e2e8f0'})

# ── 6. TAB CONTENT RENDERERS ─────────────────────────────────────────────────

def render_forecasting_tab():
    return html.Div([
        dbc.Row([
            # Sidebar Panel
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Model Improvement Panel"),
                    dbc.CardBody([
                        html.Label("Target Pollutant to Forecast"),
                        dcc.Dropdown(id='target-selector', options=[{'label': p, 'value': p} for p in pollutants_forecast], value="n02_palmes"),
                        html.Br(),
                        html.Label("Input Predictors (Features)"),
                        dcc.Dropdown(id='feature-selector', multi=True),
                        html.Br(),
                        html.Label("Number of Trees"),
                        dcc.Slider(5, 30, 1, value=15, id='slider-n-trees', marks={i: str(i) for i in range(5, 31, 5)}),
                        html.Br(),
                        html.Label("Tree Maximum Depth"),
                        dcc.Slider(1, 30, 1, value=12, id='slider-max-depth', marks={i: str(i) for i in range(0, 31, 5)}),
                        html.Hr(),
                        dbc.Button("Apply & Retrain", id='btn-retrain', color="primary", className="w-100")
                    ])
                ], color="dark", outline=True)
            ], width=3),

            # Main Performance and Forecast
            dbc.Col([
                dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardBody([html.H6("R² Score"), html.H3(id="metric-r2", className="text-info")])]), width=4),
                    dbc.Col(dbc.Card([dbc.CardBody([html.H6("MAE"), html.H3(id="metric-mae", className="text-warning")])]), width=4),
                    dbc.Col(dbc.Card([dbc.CardBody([html.H6("MSE"), html.H3(id="metric-mse", className="text-danger")])]), width=4),
                ], className="mb-3"),
                
                dbc.Card([
                    dbc.CardHeader("📈 Time Series Forecast Visualization"),
                    dbc.CardBody([
                        html.P("Click a high-error spike region below to inspect details:", className="text-muted small"),
                        dcc.Graph(id='forecast-graph', config={'displayModeBar': False}),
                        html.Hr(),
                        html.H6("🎯 Prediction Improvement Divergence", className="text-center text-muted small"),
                        dcc.Graph(id='divergence-graph', config={'displayModeBar': False}, style={'height': '150px'})
                    ])
                ], color="secondary", outline=True, className="p-2")
            ], width=9)
        ], className="mb-4"),

        # Live Insight Banner
        dbc.Row([
            dbc.Col(html.Div(id='insight-banner'), width=12)
        ], className="mb-4"),

        # Bottom Interaction Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🌡️ Tree Vote × Feature Correlation Heatmap"),
                    dbc.CardBody([
                        dcc.Graph(id='heatmap-graph', config={'displayModeBar': False}),
                        html.P("Click any cell to toggle tree status.", className="text-muted small mt-2")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🌲 Tree Decisions at Spike"),
                    dbc.CardBody([
                        dcc.Graph(id='tree-bar-graph', config={'displayModeBar': False}),
                        html.P("Click a bar to toggle tree status.", className="text-muted small mt-2")
                    ])
                ])
            ], width=6)
        ]),

        # Quick Actions
        dbc.Row([
            dbc.Col([
                html.Div([
                    dbc.Button("✅ Enable All Trees", id='btn-enable-all', color="success", size="sm", className="me-2"),
                    dbc.Button("⛔ Disable Worst 5 at Spike", id='btn-disable-worst', color="danger", size="sm", className="me-2"),
                    dbc.Button("🔁 Reset to Baseline", id='btn-reset-pos', color="info", size="sm")
                ], className="d-flex justify-content-center py-4")
            ], width=12)
        ]),

        # Error Analysis Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📊 Prediction Error Distribution"),
                    dbc.CardBody([
                        dcc.Graph(id='error-hist-graph')
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🔝 Top 5 Absolute Errors"),
                    dbc.CardBody([
                        html.Div(id='top-errors-table')
                    ])
                ])
            ], width=6)
        ], className="mb-4"),

        # Tree Explorer
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🌳 Random Forest Tree Explorer"),
                    dbc.CardBody([
                        html.P([
                            html.Span(id="tree-desc-text"),
                            html.Br(),
                            html.B("Top Feature Impact: "),
                            html.Span(id="tree-top-feature", className="text-warning")
                        ], className="text-info small mb-3"),
                        html.Label("Select Tree ID to Visualize:"),
                        dcc.Slider(0, 14, 1, value=0, id='tree-id-slider'),
                        html.Img(id='tree-img', style={'width': '100%', 'minHeight': '500px', 'objectFit': 'contain'})
                    ])
                ])
            ], width=12)
        ], className="mb-4"),

        # Model analysis protocol
        dbc.Row([
            dbc.Col([
                dbc.Accordion([
                    dbc.AccordionItem([
                        html.Div(id='analysis-plots-container')
                    ], title="🔬 Model Understanding & Improvement Protocol")
                ], start_collapsed=True)
            ], width=12)
        ], className="mb-5")
    ])

def render_correlation_tab():
    df_merged = df_merged_full
    
    non_pollutant_cols = [c for c in ['datetime', 'City', 'Year'] if c in df_merged.columns]
    disease_columns = [v for v in DISEASE_NAME_MAPPING.values() if v in df_merged.columns]
    pollutants = [c for c in df_merged.columns if c not in non_pollutant_cols + disease_columns]
    exclude_set = {'s02', 'so2', 'co', 'nh3', 'h2s', 'ufp'}
    pollutants = [p for p in pollutants if p.lower() not in exclude_set]

    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Correlation Controls"),
                    dbc.CardBody([
                        html.Label("Select Pollutants (X-axis)"),
                        dcc.Dropdown(id='corr-pollutant-selector', options=[{'label': p, 'value': p} for p in pollutants], 
                                     value=pollutants[:3], multi=True),
                        html.Br(),
                        html.Label("Select Diseases (Y-axis)"),
                        dcc.Dropdown(id='corr-disease-selector', options=[{'label': d, 'value': d} for d in disease_columns], 
                                     value=disease_columns[:3], multi=True),
                    ])
                ], color="dark", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Correlation Heatmap"),
                    dbc.CardBody([
                        dcc.Graph(id='corr-heatmap-graph')
                    ])
                ], className="mb-4"),
                dbc.Card([
                    dbc.CardHeader("Grouped Correlation Bar Chart"),
                    dbc.CardBody([
                        dcc.Graph(id='corr-bar-graph')
                    ])
                ], className="mb-4"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Scatter & Trend"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([dcc.Dropdown(id='scatter-p-selector')], width=6),
                                    dbc.Col([dcc.Dropdown(id='scatter-d-selector')], width=6),
                                ], className="mb-2"),
                                dcc.Graph(id='corr-scatter-graph')
                            ])
                        ])
                    ], width=7),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Numeric Correlation Table"),
                            dbc.CardBody([
                                html.Div(id='corr-table-div')
                            ])
                        ])
                    ], width=5)
                ])
            ], width=9)
        ])
    ])

# ── 7. CALLBACKS ─────────────────────────────────────────────────────────────

@app.callback(Output('tabs-content', 'children'), Input('tabs-navigation', 'value'))
def render_content(tab):
    if tab == 'tab-forecasting': return render_forecasting_tab()
    elif tab == 'tab-correlation': return render_correlation_tab()

# Forecasting Tab Callbacks
@app.callback(
    Output('feature-selector', 'options'),
    Output('feature-selector', 'value'),
    Input('target-selector', 'value')
)
def update_feature_list(target):
    feats = [p for p in pollutants_forecast if p != target]
    return [{'label': f, 'value': f} for f in feats], feats[:4]

@app.callback(
    Output('model-data-store', 'data'),
    Output('tree-id-slider', 'max'),
    Input('btn-retrain', 'n_clicks'),
    State('target-selector', 'value'),
    State('feature-selector', 'value'),
    State('slider-n-trees', 'value'),
    State('slider-max-depth', 'value')
)
def handle_retrain(n, target, features, n_trees, depth):
    if not features: return dash.no_update, dash.no_update
    model, t_preds, X_test, y_test, dates, o_pred = get_model_and_preds(target, features, n_trees, depth)
    
    # Calculate error threshold for spikes (90th percentile)
    abs_errors = np.abs(y_test.values - o_pred)
    error_threshold = np.percentile(abs_errors, 90)
    spike_indices = np.where(abs_errors >= error_threshold)[0].tolist()

    stored_data = {
        'tree_preds': t_preds.tolist(),
        'y_test': y_test.tolist(),
        'orig_pred': o_pred.tolist(),
        'dates': [d.strftime("%Y-%m-%d %H:%M") for d in dates],
        'n_total_trees': n_trees,
        'features': features,
        'target': target,
        'spike_indices': spike_indices
    }
    return stored_data, n_trees - 1

@app.callback(
    Output('selection-idx-store', 'data'),
    Input('forecast-graph', 'clickData'),
    Input('btn-reset-pos', 'n_clicks'),
    State('selection-idx-store', 'data'),
    prevent_initial_call=True
)
def sync_selection(click, reset_n, current_store):
    ctx = callback_context
    if not ctx.triggered: return dash.no_update
    trigger_id = ctx.triggered[0]['prop_id']
    if 'btn-reset-pos' in trigger_id: return 0
    if 'forecast-graph' in trigger_id:
        if not click or 'points' not in click: return dash.no_update
        # Logic is handled by sync_from_click to be robust
        return dash.no_update
    return dash.no_update

# Re-implementing selection sync to be more robust with multiple traces
@app.callback(
    Output('selection-idx-store', 'data', allow_duplicate=True),
    Input('forecast-graph', 'clickData'),
    State('model-data-store', 'data'),
    prevent_initial_call=True
)
def sync_from_click(click, data):
    if not click or not data: return dash.no_update
    clicked_x = click['points'][0]['x']
    try:
        idx = data['dates'].index(clicked_x)
        return idx
    except ValueError:
        return dash.no_update

@app.callback(
    Output('disabled-trees-store', 'data'),
    Input('heatmap-graph', 'clickData'),
    Input('tree-bar-graph', 'clickData'),
    Input('btn-enable-all', 'n_clicks'),
    Input('btn-disable-worst', 'n_clicks'),
    State('disabled-trees-store', 'data'),
    State('model-data-store', 'data'),
    State('selection-idx-store', 'data'),
    prevent_initial_call=True
)
def update_disabled_trees(heat_click, bar_click, enable_n, disable_worst_n, current_list, model_data, sel_idx):
    if not model_data: return []
    ctx = callback_context
    trigger = ctx.triggered[0]['prop_id']
    new_list = set(current_list)
    if 'btn-enable-all' in trigger: return []
    if 'btn-disable-worst' in trigger:
        tp = np.array(model_data['tree_preds'])
        preds_at_spike = tp[:, sel_idx]
        actual = model_data['y_test'][sel_idx]
        errs = np.abs(preds_at_spike - actual)
        worst_indices = np.argsort(errs)[-5:]
        return list(set(int(i) for i in worst_indices))
    if 'heatmap-graph' in trigger:
        if not heat_click or 'points' not in heat_click: return list(new_list)
        # Using pointIndex or the x-coordinate to identify the tree
        ti = heat_click['points'][0].get('pointIndex', heat_click['points'][0].get('x'))
        try:
            # If x was a label like "T0", extract the index
            if isinstance(ti, str):
                if ti == "🔴": 
                    # If it's the 🔴, we need the actual tree index. 
                    # Heatmap clickData usually has 'x' as the label and 'pointNumber' as [row, col]
                    ti = heat_click['points'][0]['pointNumber'][1]
                else:
                    ti = int(ti.replace("T", ""))
            else:
                ti = int(ti)
            
            if ti in new_list: new_list.discard(ti)
            else: new_list.add(ti)
        except (ValueError, TypeError, KeyError):
            pass
            
    if 'tree-bar-graph' in trigger:
        if not bar_click or 'points' not in bar_click: return list(new_list)
        ti = bar_click['points'][0].get('pointIndex')
        if ti is not None:
            ti = int(ti)
            if ti in new_list: new_list.discard(ti)
            else: new_list.add(ti)
    return list(new_list)

@app.callback(
    Output('forecast-graph', 'figure'),
    Output('heatmap-graph', 'figure'),
    Output('tree-bar-graph', 'figure'),
    Output('insight-banner', 'children'),
    Output('metric-r2', 'children'),
    Output('metric-mae', 'children'),
    Output('metric-mse', 'children'),
    Output('error-hist-graph', 'figure'),
    Output('top-errors-table', 'children'),
    Output('divergence-graph', 'figure'),
    Input('model-data-store', 'data'),
    Input('disabled-trees-store', 'data'),
    Input('selection-idx-store', 'data')
)
def update_main_forecast_plots(data, disabled, sel_idx):
    if not data: return go.Figure(), go.Figure(), go.Figure(), "", "--", "--", "--", go.Figure(), "", go.Figure()
    
    t_preds = np.array(data['tree_preds'])
    y_test = np.array(data['y_test'])
    orig_pred = np.array(data['orig_pred'])
    dates = data['dates']
    n_total = data['n_total_trees']
    
    mask = np.ones(n_total, dtype=bool)
    for ti in disabled: 
        if ti < n_total: mask[ti] = False
    active_preds = t_preds[mask]
    mod_pred = active_preds.mean(axis=0) if active_preds.size > 0 else orig_pred
    n_disabled = len(disabled)
    
    # Metrics
    r2 = r2_score(y_test, mod_pred)
    mae = mean_absolute_error(y_test, mod_pred)
    mse = mean_squared_error(y_test, mod_pred)

    # Forecast Plot
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=dates, y=y_test, name="Actual", line=dict(color="#38bdf8", width=2)))
    # Original Pred (Base line for fill)
    fig_ts.add_trace(go.Scatter(x=dates, y=orig_pred, name="Original Pred", line=dict(color="#ea580c", width=2)))
    
    if n_disabled > 0:
        # Modified Pred with fill to the 'Original Pred' (tonexty)
        fig_ts.add_trace(go.Scatter(
            x=dates, y=mod_pred, name="Modified Pred", 
            line=dict(color="#8b5cf6", width=3),
            fill='tonexty',
            fillcolor='rgba(249, 115, 22, 0.65)'  # More solid semi-transparent orange
        ))
    
    # Spikes
    spike_idx = data.get('spike_indices', [])
    if spike_idx:
        fig_ts.add_trace(go.Scatter(
            x=[dates[i] for i in spike_idx], y=[orig_pred[i] for i in spike_idx],
            mode="markers", name="High-Error Spike",
            marker=dict(color="#f43f5e", size=10, symbol="circle", line=dict(color="#fff", width=1)),
            hoverinfo="text", text=[f"High-Error Spike<br>Time: {dates[i]}<br>Error: {abs(y_test[i]-orig_pred[i]):.3f}" for i in spike_idx]
        ))

    # Selection
    fig_ts.add_trace(go.Scatter(x=[dates[sel_idx]], y=[orig_pred[sel_idx]], mode="markers", 
                                marker=dict(color="#fff", size=14, line=dict(color="#000", width=2)), name="Selection"))
    
    fig_ts.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=30, b=10),
                         paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19",
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # ── Divergence Plot (Bar per data point) ──
    # Only show if a modification has actually been made
    if n_disabled > 0:
        divergence = np.abs(y_test - orig_pred) - np.abs(y_test - mod_pred)
        div_colors = ["#34d399" if v >= 0 else "#f43f5e" for v in divergence]
        
        fig_div = go.Figure()
        fig_div.add_trace(go.Bar(
            x=dates, y=divergence,
            marker_color=div_colors,
            name="Improvement Delta",
            hovertemplate="Time: %{x}<br>Improvement: %{y:.4f}<extra></extra>"
        ))
        
        # High-contrast black background with wider, solid bars
        fig_div.update_layout(
            template="plotly_dark", 
            height=200, 
            margin=dict(l=40, r=20, t=10, b=40),
            paper_bgcolor="#000000", 
            plot_bgcolor="#000000",
            bargap=0.05, # Makes the bars much wider
            xaxis=dict(
                title="Timeline",
                showticklabels=True, 
                showgrid=True, 
                gridcolor="#1e293b", # Subtle dark grid
                zeroline=True,
                zerolinecolor="#334155",
                color="white"
            ),
            yaxis=dict(
                title="Divergence", 
                showgrid=True, 
                gridcolor="#1e293b", 
                zeroline=True, 
                zerolinecolor="#334155",
                color="white"
            ),
            showlegend=False
        )
        fig_div.update_traces(marker_line_width=0, opacity=1.0)
    else:
        # Before modification, show an empty placeholder or empty figure
        fig_div = go.Figure()
        fig_div.update_layout(template="plotly_dark", height=150, paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19",
                              xaxis=dict(visible=False), yaxis=dict(visible=False),
                              annotations=[dict(text="Apply a modification to see impact divergence", showarrow=False, font=dict(color="#94a3b8"))])

    # Heatmap
    max_h = min(30, n_total)
    feats = data['features']
    w_start = max(0, sel_idx - 50); w_end = min(len(y_test), sel_idx + 51)
    
    # Calculate offset for test set since shuffle=False in train_test_split
    test_start_offset = len(df_forecast) - len(y_test)
    
    corr_matrix = np.zeros((len(feats), max_h))
    for fi, f_name in enumerate(feats):
        # Ensure we use numpy array for indexing from the correct test set window
        if f_name in df_forecast.columns:
            fv = df_forecast.iloc[test_start_offset + w_start : test_start_offset + w_end, df_forecast.columns.get_loc(f_name)].values
        else:
            fv = np.random.randn(w_end-w_start)
        
        for ti in range(max_h):
            tv = t_preds[ti, w_start:w_end]
            if np.std(fv) > 1e-6 and np.std(tv) > 1e-6:
                corr_matrix[fi, ti] = np.corrcoef(fv, tv)[0, 1]
    
    labels = [f"T{i}" if i not in disabled else "🔴" for i in range(max_h)]
    fig_heat = go.Figure()
    fig_heat.add_trace(go.Heatmap(z=corr_matrix, x=labels, y=feats, colorscale="RdYlGn", zmid=0, zmin=-1, zmax=1))
    if disabled:
        overlay_z = np.zeros((len(feats), max_h))
        for ti in disabled:
            if ti < max_h: overlay_z[:, ti] = 1.0
        fig_heat.add_trace(go.Heatmap(z=overlay_z, x=labels, y=feats, colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(244,63,94,0.22)"]], showscale=False, hoverinfo="skip"))
    fig_heat.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=10, b=10))

    # Tree Bar Plot
    colors = ["#f43f5e" if i in disabled else "#34d399" for i in range(max_h)]
    fig_tree_bar = go.Figure(go.Bar(x=[f"T{i}" for i in range(max_h)], y=t_preds[:max_h, sel_idx], marker_color=colors))
    fig_tree_bar.add_hline(y=y_test[sel_idx], line_dash="dash", line_color="#38bdf8", annotation_text="Actual")
    fig_tree_bar.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=10, b=10))

    # Insight Banner
    banner = dbc.Alert([
        html.B("🕒 Selection: "), dates[sel_idx],
        html.Span(f" | Actual: {y_test[sel_idx]:.3f}", className="ms-3"),
        html.Span(f" | Orig: {orig_pred[sel_idx]:.3f}", className="ms-3", style={'textDecoration': 'line-through', 'color': '#94a3b8'}),
        html.Span(f" | Mod Pred: {mod_pred[sel_idx]:.3f}", className="ms-3", style={'color': '#a78bfa', 'fontWeight': 'bold'}),
        html.Span(f" | Active Trees: {n_total - n_disabled}/{n_total}", className="ms-3")
    ], color="dark", style={'borderLeft': '4px solid #facc15'})

    # Error Hist
    residuals = y_test - mod_pred
    fig_err = px.histogram(pd.DataFrame({"Residuals": residuals}), x="Residuals", nbins=40, template="plotly_dark", color_discrete_sequence=["#ef553b"])
    fig_err.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))

    # Top Errors Table
    err_df = pd.DataFrame({"Actual": y_test, "Predicted": mod_pred, "AbsError": np.abs(residuals)})
    top_errs = err_df.sort_values("AbsError", ascending=False).head(5).round(3)
    tbl = dash_table.DataTable(
        data=top_errs.to_dict('records'),
        columns=[{"name": i, "id": i} for i in top_errs.columns],
        style_header={'backgroundColor': '#1e1e2e', 'color': 'white', 'fontWeight': 'bold'},
        style_cell={'backgroundColor': '#0b0f19', 'color': '#e2e8f0', 'textAlign': 'left'},
        style_as_list_view=True
    )
    return fig_ts, fig_heat, fig_tree_bar, banner, f"{r2:.3f}", f"{mae:.3f}", f"{mse:.3f}", fig_err, tbl, fig_div

@app.callback(
    Output('analysis-plots-container', 'children'),
    Input('model-data-store', 'data'),
    Input('disabled-trees-store', 'data')
)
def update_analysis_protocol(data, disabled):
    if not data: return []
    
    t_preds = np.array(data['tree_preds'])
    y_test = np.array(data['y_test'])
    orig_pred = np.array(data['orig_pred'])
    dates = data['dates']
    n_total = data['n_total_trees']
    
    mask = np.ones(n_total, dtype=bool)
    for ti in disabled: 
        if ti < n_total: mask[ti] = False
    active_preds = t_preds[mask]
    mod_pred = active_preds.mean(axis=0) if active_preds.size > 0 else orig_pred
    
    residuals = y_test - mod_pred
    
    # 1. Residuals vs Predicted
    fig_rvp = px.scatter(x=mod_pred, y=residuals, labels={"x": "Predicted", "y": "Residual"},
                         trendline="lowess", title="Residuals vs Predicted", template="plotly_dark")
    fig_rvp.add_hline(y=0, line_dash="dash", line_color="white")
    
    # 2. Predicted vs Actual
    pav = pd.DataFrame({"Actual": y_test, "Predicted": mod_pred})
    fig_pa = px.scatter(pav, x="Actual", y="Predicted", trendline="ols",
                        title="Predicted vs Actual", template="plotly_dark")
    fig_pa.add_shape(type="line", x0=pav["Actual"].min(), y0=pav["Actual"].min(),
                     x1=pav["Actual"].max(), y1=pav["Actual"].max(), line_dash="dash", line_color="white")
    
    # 3. Residuals over Time
    fig_rt = px.line(x=dates, y=residuals, title="Residuals over Time", template="plotly_dark")
    fig_rt.add_hline(y=0, line_dash="dash", line_color="white")

    return [
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_rvp, config={'displayModeBar': False}), width=6),
            dbc.Col(dcc.Graph(figure=fig_pa, config={'displayModeBar': False}), width=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_rt, config={'displayModeBar': False}), width=12),
        ])
    ]

@app.callback(
    Output('tree-desc-text', 'children'),
    Input('tree-id-slider', 'value'),
    Input('disabled-trees-store', 'data')
)
def update_tree_description(tree_id, disabled_list):
    disabled_str = ", ".join([f"T{i}" for i in sorted(disabled_list)]) if disabled_list else "None"
    return [
        html.B(f"Observation: "), 
        f"Tree T{tree_id} shows specific importance to features in its splits. ",
        html.Br(),
        html.B("Status: "), 
        f"Remember that trees [{disabled_str}] have been turned off. "
    ]

@app.callback(
    Output('tree-img', 'src'),
    Output('tree-top-feature', 'children'),
    Input('model-data-store', 'data'),
    Input('tree-id-slider', 'value')
)
def update_tree_vis(data, tree_id):
    if not data: return "", ""
    target = data['target']
    features = data['features']
    n_trees = data['n_total_trees']
    
    X = df_forecast[features]
    y = df_forecast[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    model = RandomForestRegressor(n_estimators=n_trees, max_depth=10, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    if tree_id >= len(model.estimators_): tree_id = 0
    tree_obj = model.estimators_[tree_id]
    
    # Identify top feature for this specific tree
    importances = tree_obj.feature_importances_
    top_fi = np.argmax(importances)
    top_feature_name = features[top_fi]

    # Plot with larger text and better visibility
    fig, ax = plt.subplots(figsize=(16, 9)) # Bigger canvas
    plot_tree(tree_obj, feature_names=features, filled=True, max_depth=3, ax=ax, 
              rounded=True, fontsize=10, impurity=False, precision=2)
    plt.tight_layout()
    uri = fig_to_uri(fig)
    plt.close(fig)
    
    return uri, f"{top_feature_name} (Contribution: {importances[top_fi]*100:.1f}%)"

# Correlation Tab Callbacks
@app.callback(
    Output('corr-heatmap-graph', 'figure'),
    Output('corr-bar-graph', 'figure'),
    Output('corr-table-div', 'children'),
    Output('scatter-p-selector', 'options'),
    Output('scatter-p-selector', 'value'),
    Output('scatter-d-selector', 'options'),
    Output('scatter-d-selector', 'value'),
    Input('corr-pollutant-selector', 'value'),
    Input('corr-disease-selector', 'value')
)
def update_correlation_view(selected_pollutants, selected_diseases):
    if not selected_pollutants or not selected_diseases:
        return go.Figure(), go.Figure(), "Select features", [], None, [], None
    
    df = load_merged_data()
    df.rename(columns=DISEASE_NAME_MAPPING, inplace=True)
    analysis_df = df[selected_pollutants + selected_diseases].dropna()
    if analysis_df.empty: return go.Figure(), go.Figure(), "No data", [], None, [], None
    
    corr = analysis_df.corr().loc[selected_pollutants, selected_diseases]
    
    # Heatmap
    fig_heat = px.imshow(corr.values, x=selected_diseases, y=selected_pollutants,
                         color_continuous_scale='RdBu_r', zmin=-1, zmax=1, template="plotly_dark")
    fig_heat.update_layout(margin=dict(l=40, r=40, t=40, b=40))

    # Bar
    melt_corr = corr.reset_index().melt(id_vars='index', var_name='Disease', value_name='Correlation').rename(columns={'index': 'Pollutant'})
    fig_bar = px.bar(melt_corr, x='Correlation', y='Disease', color='Pollutant', orientation='h', barmode='group', template="plotly_dark")

    # Table
    tbl = dash_table.DataTable(
        data=corr.reset_index().to_dict('records'),
        columns=[{"name": i, "id": i} for i in corr.reset_index().columns],
        style_header={'backgroundColor': '#1e1e2e', 'color': 'white'},
        style_cell={'backgroundColor': '#0b0f19', 'color': '#e2e8f0'},
    )
    
    p_opts = [{'label': p, 'value': p} for p in selected_pollutants]
    d_opts = [{'label': d, 'value': d} for d in selected_diseases]
    
    return fig_heat, fig_bar, tbl, p_opts, selected_pollutants[0], d_opts, selected_diseases[0]

@app.callback(
    Output('corr-scatter-graph', 'figure'),
    Input('scatter-p-selector', 'value'),
    Input('scatter-d-selector', 'value')
)
def update_scatter(p, d):
    if not p or not d: return go.Figure()
    df = df_merged_full
    scatter_df = df[[p, d]].dropna()
    fig = px.scatter(scatter_df, x=p, y=d, trendline="ols", template="plotly_dark", color_discrete_sequence=["#38bdf8"])
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8052)
