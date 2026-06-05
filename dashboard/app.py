"""
Customer Churn Intelligence — Interactive Dashboard
=====================================================
Multi-page Streamlit application with dark theme and polished UI.

Pages:
  1. Executive Summary
  2. Customer Insights
  3. Prediction Center
  4. Business Recommendations
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib

# ── Path Setup ─────────────────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

sys.path.insert(0, BASE_DIR)

# ── Streamlit Config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color Palette ──────────────────────────────────────────────────────────────

COLORS = {
    "primary": "#6C63FF",
    "secondary": "#00D4AA",
    "danger": "#FF6B6B",
    "warning": "#FFD93D",
    "info": "#4ECDC4",
    "bg": "#0E1117",
    "card": "#1A1A2E",
    "text": "#E0E0E0",
    "muted": "#888899",
}

CURRENCY_RATES = {
    "$": 1.0,
    "€": 0.92,
    "₹": 83.5,
    "£": 0.78,
    "¥": 156.0
}

def get_currency_rate():
    return CURRENCY_RATES.get(st.session_state.get("currency_symbol", "$"), 1.0)

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": COLORS["bg"],
        "plot_bgcolor": COLORS["card"],
        "font": {"color": COLORS["text"], "family": "Inter, sans-serif"},
        "colorway": [COLORS["primary"], COLORS["danger"], COLORS["secondary"],
                     COLORS["warning"], COLORS["info"], "#C9B1FF"],
    }
}


# ── Custom CSS ─────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main > div { padding-top: 1rem; }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
        border: 1px solid #333355;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(108, 99, 255, 0.2);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6C63FF, #00D4AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.3rem 0;
    }
    .kpi-label {
        color: #888899;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .kpi-delta {
        font-size: 0.8rem;
        margin-top: 0.3rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #E0E0E0;
        border-left: 4px solid #6C63FF;
        padding-left: 1rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .risk-high { background: rgba(255, 107, 107, 0.2); color: #FF6B6B; border: 1px solid #FF6B6B; }
    .risk-medium { background: rgba(255, 217, 61, 0.2); color: #FFD93D; border: 1px solid #FFD93D; }
    .risk-low { background: rgba(0, 212, 170, 0.2); color: #00D4AA; border: 1px solid #00D4AA; }

    /* Recommendation cards */
    .rec-card {
        background: linear-gradient(135deg, #1A1A2E, #16213E);
        border: 1px solid #333355;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #6C63FF;
    }
    .rec-title { font-weight: 600; font-size: 1.05rem; color: #E0E0E0; }
    .rec-detail { color: #AAAACC; font-size: 0.9rem; margin-top: 0.5rem; }
    .rec-impact { color: #00D4AA; font-size: 0.85rem; font-style: italic; margin-top: 0.4rem; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1A1A2E 100%);
    }
    [data-testid="stSidebar"] .element-container { padding: 0 0.5rem; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { background: transparent; }

    /* Style the sidebar collapse button to be visible and premium */
    [data-testid="stSidebarCollapse"] {
        background-color: #1A1A2E !important;
        border: 1px solid #333355 !important;
        border-radius: 8px !important;
        color: #6C63FF !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stSidebarCollapse"]:hover {
        background-color: #6C63FF !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 10px rgba(108, 99, 255, 0.5) !important;
    }

    /* Gauge */
    .gauge-container {
        text-align: center;
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Data Loading (Cached) ─────────────────────────────────────────────────────

@st.cache_data
def load_cleaned_data():
    path = os.path.join(DATA_DIR, "cleaned_churn.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_segmented_data():
    path = os.path.join(DATA_DIR, "segmented_churn.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_model_metrics():
    path = os.path.join(MODELS_DIR, "model_comparison.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_insights():
    path = os.path.join(REPORTS_DIR, "insights.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_segment_profiles():
    path = os.path.join(REPORTS_DIR, "segment_profiles.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def load_model():
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    fn_path = os.path.join(MODELS_DIR, "feature_names.json")
    if not all(os.path.exists(p) for p in [model_path, scaler_path, fn_path]):
        return None, None, None
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(fn_path) as f:
        feature_names = json.load(f)
    return model, scaler, feature_names


# ── Helper Functions ───────────────────────────────────────────────────────────

def kpi_card(label: str, value: str, delta: str = None, delta_color: str = "normal"):
    delta_html = ""
    if delta:
        color = COLORS["secondary"] if delta_color == "good" else COLORS["danger"]
        delta_html = f'<div class="kpi-delta" style="color: {color};">{delta}</div>'

    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section_header(text: str):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def plotly_layout(fig, height=400):
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        height=height,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor="#333355", gridwidth=0.5)
    fig.update_yaxes(gridcolor="#333355", gridwidth=0.5)
    return fig


# ── Page: Executive Summary ───────────────────────────────────────────────────

def page_executive_summary():
    df = load_cleaned_data()
    if df is None:
        st.error("No data found. Please run the data pipeline first.")
        return

    if "Churn_Binary" not in df.columns:
        df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)

    st.markdown("## 📊 Executive Summary")
    st.markdown("---")

    # KPIs
    total = len(df)
    churn_rate = df["Churn_Binary"].mean()
    revenue_risk = df[df["Churn"] == "Yes"]["TotalCharges"].sum()
    avg_tenure = df["tenure"].mean()
    avg_monthly = df["MonthlyCharges"].mean()
    retained_revenue = df[df["Churn"] == "No"]["TotalCharges"].sum()

    sym = st.session_state.get("currency_symbol", "$")
    rate = get_currency_rate()
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        kpi_card("Total Customers", f"{total:,}")
    with col2:
        kpi_card("Churn Rate", f"{churn_rate:.1%}", "↑ Action needed", "bad")
    with col3:
        kpi_card("Revenue at Risk", f"{sym}{revenue_risk * rate:,.0f}")
    with col4:
        kpi_card("Avg Tenure", f"{avg_tenure:.1f} mo")
    with col5:
        kpi_card("Avg Monthly", f"{sym}{avg_monthly * rate:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    col_left, col_right = st.columns(2)

    with col_left:
        section_header("Churn Distribution")
        fig = go.Figure(data=[go.Pie(
            labels=["Retained", "Churned"],
            values=[total - df["Churn_Binary"].sum(), df["Churn_Binary"].sum()],
            hole=0.55,
            marker=dict(colors=[COLORS["secondary"], COLORS["danger"]],
                        line=dict(color=COLORS["bg"], width=3)),
            textinfo="label+percent",
            textfont=dict(size=14),
        )])
        fig.update_layout(
            showlegend=False,
            annotations=[dict(text=f"{churn_rate:.0%}", x=0.5, y=0.5,
                              font_size=28, font_color=COLORS["danger"],
                              showarrow=False)],
        )
        fig = plotly_layout(fig, 350)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        section_header("Revenue Breakdown")
        sym = st.session_state.get("currency_symbol", "$")
        rate = get_currency_rate()
        fig = go.Figure(data=[go.Bar(
            x=["Retained Revenue", "Revenue at Risk"],
            y=[retained_revenue * rate, revenue_risk * rate],
            marker_color=[COLORS["secondary"], COLORS["danger"]],
            text=[f"{sym}{retained_revenue * rate:,.0f}", f"{sym}{revenue_risk * rate:,.0f}"],
            textposition="outside",
            textfont=dict(size=13, color=COLORS["text"]),
        )])
        fig.update_layout(
            yaxis_title=f"Revenue ({sym})",
            showlegend=False,
        )
        fig = plotly_layout(fig, 350)
        st.plotly_chart(fig, use_container_width=True)

    # Segment distribution
    seg_df = load_segmented_data()
    insights = load_insights()

    col_left2, col_right2 = st.columns(2)

    with col_left2:
        if seg_df is not None and "Segment" in seg_df.columns:
            section_header("Customer Segments")
            seg_counts = seg_df["Segment"].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=seg_counts.index.tolist(),
                values=seg_counts.values.tolist(),
                hole=0.45,
                marker=dict(
                    colors=[COLORS["primary"], COLORS["secondary"],
                            COLORS["danger"], COLORS["warning"]],
                    line=dict(color=COLORS["bg"], width=2),
                ),
                textinfo="label+percent",
                textfont=dict(size=12),
            )])
            fig = plotly_layout(fig, 350)
            st.plotly_chart(fig, use_container_width=True)

    with col_right2:
        if insights:
            section_header("Top Churn Drivers")
            for ins in insights[:4]:
                icon = {"high": "🔴", "medium": "🟡", "info": "🔵"}.get(
                    ins.get("priority", ""), "⚪")
                st.markdown(f"""
                <div style="background: {COLORS['card']}; border-radius: 8px;
                     padding: 0.8rem 1rem; margin-bottom: 0.5rem;
                     border-left: 3px solid {COLORS['primary']};">
                    <span style="font-size: 0.85rem; color: {COLORS['text']};">
                        {icon} {ins['insight']}
                    </span>
                </div>
                """, unsafe_allow_html=True)


# ── Page: Customer Insights ───────────────────────────────────────────────────

def page_customer_insights():
    df = load_cleaned_data()
    if df is None:
        st.error("No data found.")
        return

    if "Churn_Binary" not in df.columns:
        df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)

    st.markdown("## 🔎 Customer Insights")
    st.markdown("---")

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        gender_filter = st.multiselect("Gender", df["gender"].unique().tolist(),
                                       default=df["gender"].unique().tolist())
    with col_f2:
        contract_filter = st.multiselect("Contract", df["Contract"].unique().tolist(),
                                         default=df["Contract"].unique().tolist())
    with col_f3:
        internet_filter = st.multiselect("Internet Service",
                                         df["InternetService"].unique().tolist(),
                                         default=df["InternetService"].unique().tolist())

    filtered = df[
        (df["gender"].isin(gender_filter)) &
        (df["Contract"].isin(contract_filter)) &
        (df["InternetService"].isin(internet_filter))
    ]

    st.markdown(f"*Showing {len(filtered):,} of {len(df):,} customers*")

    # Demographics
    col1, col2 = st.columns(2)

    with col1:
        section_header("Demographics Breakdown")
        demo_col = st.selectbox("Select demographic:",
                                ["gender", "SeniorCitizen", "Partner", "Dependents"])
        ct = pd.crosstab(filtered[demo_col], filtered["Churn"], normalize="index") * 100
        fig = go.Figure()
        for status, color in zip(["No", "Yes"], [COLORS["secondary"], COLORS["danger"]]):
            if status in ct.columns:
                fig.add_trace(go.Bar(
                    name=f"Churn={status}", x=ct.index.astype(str), y=ct[status],
                    marker_color=color,
                ))
        fig.update_layout(barmode="group", yaxis_title="Percentage (%)",
                          xaxis_title=demo_col, legend_title="Churn")
        fig = plotly_layout(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_header("Contract Type Analysis")
        ct2 = pd.crosstab(filtered["Contract"], filtered["Churn"])
        fig = go.Figure()
        for status, color in zip(["No", "Yes"], [COLORS["secondary"], COLORS["danger"]]):
            if status in ct2.columns:
                fig.add_trace(go.Bar(
                    name=f"Churn={status}", x=ct2.index, y=ct2[status],
                    marker_color=color,
                ))
        fig.update_layout(barmode="stack", yaxis_title="Customers",
                          xaxis_title="Contract Type", legend_title="Churn")
        fig = plotly_layout(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    # Second row
    col3, col4 = st.columns(2)

    with col3:
        section_header("Monthly Charges Distribution")
        sym = st.session_state.get("currency_symbol", "$")
        rate = get_currency_rate()
        fig = go.Figure()
        for status, color, name in zip(
            ["No", "Yes"], [COLORS["secondary"], COLORS["danger"]],
            ["Retained", "Churned"]
        ):
            subset = filtered[filtered["Churn"] == status]["MonthlyCharges"] * rate
            fig.add_trace(go.Histogram(
                x=subset, name=name, marker_color=color, opacity=0.7, nbinsx=30,
            ))
        fig.update_layout(barmode="overlay", xaxis_title=f"Monthly Charges ({sym})",
                          yaxis_title="Count")
        fig = plotly_layout(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        section_header("Tenure vs Monthly Charges")
        sym = st.session_state.get("currency_symbol", "$")
        rate = get_currency_rate()
        sample = filtered.sample(min(1000, len(filtered)), random_state=42).copy()
        sample["MonthlyCharges_Converted"] = sample["MonthlyCharges"] * rate
        fig = px.scatter(
            sample, x="tenure", y="MonthlyCharges_Converted", color="Churn",
            color_discrete_map={"No": COLORS["secondary"], "Yes": COLORS["danger"]},
            opacity=0.5, hover_data=["Contract", "PaymentMethod"],
        )
        fig.update_layout(xaxis_title="Tenure (months)", yaxis_title=f"Monthly Charges ({sym})")
        fig = plotly_layout(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    # Service heatmap
    section_header("Service Usage & Churn Correlation")
    service_cols = [
        "PhoneService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    svc_df = pd.DataFrame()
    for col in service_cols:
        svc_df[col] = (filtered[col] == "Yes").astype(int)
    svc_df["Churn"] = filtered["Churn_Binary"].values

    corr = svc_df.corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
    )
    fig.update_layout(title="")
    fig = plotly_layout(fig, 450)
    st.plotly_chart(fig, use_container_width=True)

    # Segment explorer
    seg_df = load_segmented_data()
    if seg_df is not None and "Segment" in seg_df.columns:
        section_header("Segment Explorer")
        segment_choice = st.selectbox("Select segment:", seg_df["Segment"].unique().tolist())
        seg_subset = seg_df[seg_df["Segment"] == segment_choice]

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            kpi_card("Customers", f"{len(seg_subset):,}")
        with sc2:
            kpi_card("Avg Tenure", f"{seg_subset['tenure'].mean():.1f} mo")
        with sc3:
            sym = st.session_state.get("currency_symbol", "$")
            rate = get_currency_rate()
            kpi_card("Avg Monthly", f"{sym}{seg_subset['MonthlyCharges'].mean() * rate:.2f}")
        with sc4:
            seg_churn = (seg_subset["Churn"] == "Yes").mean()
            kpi_card("Churn Rate", f"{seg_churn:.1%}")

        st.markdown("<br>", unsafe_allow_html=True)
        seg_display = seg_subset.copy()
        rate = get_currency_rate()
        seg_display["MonthlyCharges"] = np.round(seg_display["MonthlyCharges"] * rate, 2)
        seg_display["TotalCharges"] = np.round(seg_display["TotalCharges"] * rate, 2)
        st.dataframe(
            seg_display[["customerID", "gender", "tenure", "Contract",
                         "MonthlyCharges", "TotalCharges", "Churn"]].head(20),
            use_container_width=True,
        )


# ── Page: Prediction Center ──────────────────────────────────────────────────

def page_prediction_center():
    st.markdown("## 🎯 Prediction Center")
    st.markdown("---")

    model, scaler, feature_names = load_model()
    if model is None:
        st.error("No trained model found. Please run the training pipeline first.")
        return

    tab1, tab2 = st.tabs(["🧑 Single Customer Prediction", "📁 Batch Upload"])

    with tab1:
        _single_prediction(model, scaler, feature_names)

    with tab2:
        _batch_prediction(model, scaler, feature_names)


def _single_prediction(model, scaler, feature_names):
    """Manual single-customer prediction form."""
    section_header("Customer Details")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with col2:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ])

        with col3:
            sym = st.session_state.get("currency_symbol", "$")
            rate = get_currency_rate()
            if rate < 10:
                min_val = float(np.round(18.0 * rate, 2))
                max_val = float(np.round(120.0 * rate, 2))
                default_val = float(np.round(65.0 * rate, 2))
                step_val = max(0.01, float(np.round(0.5 * rate, 2)))
            else:
                min_val = float(np.round(18.0 * rate))
                max_val = float(np.round(120.0 * rate))
                default_val = float(np.round(65.0 * rate))
                step_val = 1.0
            monthly = st.slider(f"Monthly Charges ({sym})", min_val, max_val, default_val, step_val)
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    if submitted:
        # Build feature dict
        from src.prediction import prepare_customer_for_prediction

        rate = get_currency_rate()
        monthly_usd = monthly / rate
        total_usd = monthly_usd * max(tenure, 1)

        raw_input = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "Contract": contract,
            "InternetService": internet, "PhoneService": phone,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly_usd, "TotalCharges": total_usd,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "TechSupport": tech_support, "DeviceProtection": device_protection,
            "MultipleLines": "No" if phone == "No" else "Yes",
            "StreamingTV": "No", "StreamingMovies": "No",
            "num_services": sum(1 for s in [phone, online_security, online_backup,
                                            tech_support, device_protection] if s == "Yes"),
            "has_premium_support": 1 if all(s == "Yes" for s in [
                online_security, tech_support, device_protection]) else 0,
            "avg_monthly_charge": monthly_usd,
            "charge_per_service": monthly_usd,
            "is_new_month_to_month": 1 if contract == "Month-to-month" and tenure <= 12 else 0,
            "is_long_term_customer": 1 if contract in ["One year", "Two year"] and tenure > 36 else 0,
        }

        customer = prepare_customer_for_prediction(raw_input, feature_names)
        X = np.array([[customer.get(f, 0) for f in feature_names]])
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0][1]

        st.markdown("<br>", unsafe_allow_html=True)

        # Results
        col_r1, col_r2, col_r3 = st.columns([1, 2, 1])

        with col_r1:
            risk_level = "High" if prob >= 0.6 else ("Medium" if prob >= 0.3 else "Low")
            risk_class = f"risk-{risk_level.lower()}"
            kpi_card("Churn Probability", f"{prob:.1%}")
            st.markdown(f"""
            <div style="text-align: center; margin-top: 1rem;">
                <span class="risk-badge {risk_class}">{risk_level} Risk</span>
            </div>
            """, unsafe_allow_html=True)

        with col_r2:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 36, "color": COLORS["text"]}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": COLORS["muted"]},
                    "bar": {"color": COLORS["danger"] if prob > 0.6 else
                            (COLORS["warning"] if prob > 0.3 else COLORS["secondary"])},
                    "bgcolor": COLORS["card"],
                    "steps": [
                        {"range": [0, 30], "color": "rgba(0,212,170,0.15)"},
                        {"range": [30, 60], "color": "rgba(255,217,61,0.15)"},
                        {"range": [60, 100], "color": "rgba(255,107,107,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": COLORS["text"], "width": 2},
                        "thickness": 0.8,
                        "value": prob * 100,
                    },
                },
            ))
            fig = plotly_layout(fig, 300)
            fig.update_layout(margin=dict(l=30, r=30, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_r3:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if risk_level == "High":
                st.warning("⚠️ **High churn risk!** Consider immediate retention action.")
            elif risk_level == "Medium":
                st.info("📋 **Moderate risk.** Monitor and engage proactively.")
            else:
                st.success("✅ **Low risk.** Customer is likely to stay.")


def _batch_prediction(model, scaler, feature_names):
    """Batch CSV upload prediction."""
    section_header("Upload Customer Data")

    st.info("""
    Upload a CSV file with customer data. The file should have the same columns as the training data.
    The system will predict churn probability for each customer.
    """)

    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            st.markdown(f"*Loaded {len(batch_df):,} customers*")

            # Feature engineering on uploaded data
            from src.feature_engineering import (
                add_tenure_group, add_avg_monthly_charge, add_num_services,
                add_premium_support, add_charge_per_service,
                add_contract_tenure_interaction, encode_categoricals,
            )

            processed = batch_df.copy()
            processed = add_tenure_group(processed)
            processed = add_avg_monthly_charge(processed)
            processed = add_num_services(processed)
            processed = add_premium_support(processed)
            processed = add_charge_per_service(processed)
            processed = add_contract_tenure_interaction(processed)
            processed = encode_categoricals(processed)

            # Ensure columns match
            for col in feature_names:
                if col not in processed.columns:
                    processed[col] = 0

            X = processed[feature_names].values
            X_scaled = scaler.transform(X)

            probs = model.predict_proba(X_scaled)[:, 1]
            batch_df["Churn_Probability"] = np.round(probs, 4)
            batch_df["Risk_Level"] = pd.cut(
                probs, bins=[0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"]
            )

            # Summary
            col1, col2, col3 = st.columns(3)
            with col1:
                kpi_card("Total Uploaded", f"{len(batch_df):,}")
            with col2:
                high_risk = (batch_df["Risk_Level"] == "High").sum()
                kpi_card("High Risk", f"{high_risk:,}")
            with col3:
                avg_prob = probs.mean()
                kpi_card("Avg Churn Prob", f"{avg_prob:.1%}")

            st.markdown("<br>", unsafe_allow_html=True)

            # Risk distribution
            fig = px.histogram(
                batch_df, x="Churn_Probability", nbins=30,
                color_discrete_sequence=[COLORS["primary"]],
                labels={"Churn_Probability": "Churn Probability"},
            )
            fig.update_layout(yaxis_title="Count")
            fig = plotly_layout(fig, 300)
            st.plotly_chart(fig, use_container_width=True)

            # Show results
            rate = get_currency_rate()
            df_display = batch_df.copy()
            df_display["MonthlyCharges"] = np.round(df_display["MonthlyCharges"] * rate, 2)
            st.dataframe(
                df_display[["customerID", "tenure", "MonthlyCharges", "Contract",
                            "Churn_Probability", "Risk_Level"]].sort_values(
                    "Churn_Probability", ascending=False
                ),
                use_container_width=True,
            )

            # Download button
            csv = batch_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")


# ── Page: Business Recommendations ───────────────────────────────────────────

def page_recommendations():
    st.markdown("## 💡 Business Recommendations")
    st.markdown("---")

    df = load_cleaned_data()
    insights = load_insights()

    if df is not None:
        if "Churn_Binary" not in df.columns:
            df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)

    # Insights summary
    if insights:
        section_header("Key Insights from Analysis")
        for ins in insights:
            icon = {"high": "🔴", "medium": "🟡", "info": "🔵"}.get(
                ins.get("priority", ""), "⚪")
            st.markdown(f"""
            <div style="background: {COLORS['card']}; border-radius: 10px;
                 padding: 1rem 1.2rem; margin-bottom: 0.6rem;
                 border-left: 3px solid {COLORS['primary']};">
                <span style="color: {COLORS['text']}; font-size: 0.95rem;">
                    {icon} <b>[{ins['category']}]</b> {ins['insight']}
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # Recommendations
    section_header("Actionable Retention Strategies")

    sym = st.session_state.get("currency_symbol", "$")
    rate = get_currency_rate()

    recommendations = [
        {
            "title": "🎁 Introduce Loyalty Discounts",
            "detail": f"Offer loyalty discounts for customers with tenure under 12 months "
                      f"and monthly charges above {sym}{70 * rate:.0f}. This targets the highest-risk segment.",
            "impact": "Could reduce churn by 15-20% in the at-risk segment.",
            "priority": "High",
            "color": COLORS["danger"],
        },
        {
            "title": "📋 Promote Long-Term Contracts",
            "detail": "Incentivize month-to-month customers to switch to annual or biennial "
                      "contracts with a 10-15% discount.",
            "impact": "Potential to stabilize 40%+ of at-risk customers.",
            "priority": "High",
            "color": COLORS["danger"],
        },
        {
            "title": "🌐 Improve Fiber Optic Experience",
            "detail": "Fiber optic customers show higher churn rates despite paying more. "
                      "Investigate service quality and consider value-add bundles.",
            "impact": "Address quality-price perception gap for premium customers.",
            "priority": "Medium",
            "color": COLORS["warning"],
        },
        {
            "title": "💳 Migrate from Electronic Check",
            "detail": f"Customers paying by electronic check churn significantly more. "
                      f"Offer auto-payment setup incentives (e.g., {sym}{5 * rate:.0f}/month credit).",
            "impact": "Reduce friction-based churn in the payment process.",
            "priority": "Medium",
            "color": COLORS["warning"],
        },
        {
            "title": "🛡️ Bundle Security & Support",
            "detail": "Create discounted bundles combining Online Security, Tech Support, "
                      "and Device Protection to improve engagement.",
            "impact": "Increase service stickiness and customer lifetime value.",
            "priority": "Medium",
            "color": COLORS["warning"],
        },
        {
            "title": "👴 Senior Citizen Program",
            "detail": "Launch a dedicated senior-friendly support program with simplified "
                      "billing and priority assistance.",
            "impact": "Improve retention in the senior citizen demographic.",
            "priority": "Low",
            "color": COLORS["secondary"],
        },
    ]

    for rec in recommendations:
        st.markdown(f"""
        <div class="rec-card" style="border-left-color: {rec['color']};">
            <div class="rec-title">{rec['title']}
                <span class="risk-badge risk-{rec['priority'].lower()}"
                      style="float: right; font-size: 0.75rem;">
                    {rec['priority']} Priority
                </span>
            </div>
            <div class="rec-detail">{rec['detail']}</div>
            <div class="rec-impact">📈 Expected Impact: {rec['impact']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Cost-Benefit Estimates
    if df is not None:
        section_header("Cost-Benefit Estimation")

        total_risk = df[df["Churn"] == "Yes"]["TotalCharges"].sum()
        avg_clv = df[df["Churn"] == "No"]["TotalCharges"].mean()

        sym = st.session_state.get("currency_symbol", "$")
        rate = get_currency_rate()
        col1, col2, col3 = st.columns(3)
        with col1:
            kpi_card("Revenue at Risk", f"{sym}{total_risk * rate:,.0f}")
        with col2:
            saved_10 = total_risk * 0.10
            kpi_card("Save 10% Churn", f"{sym}{saved_10 * rate:,.0f}", "Conservative", "good")
        with col3:
            saved_25 = total_risk * 0.25
            kpi_card("Save 25% Churn", f"{sym}{saved_25 * rate:,.0f}", "Optimistic", "good")

    # Generate Report button
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Generate Business Report")

    if st.button("📄 Generate PDF Report", use_container_width=True):
        with st.spinner("Generating report..."):
            try:
                import importlib
                import src.report_generator
                importlib.reload(src.report_generator)
                from src.report_generator import generate_report
                sym = st.session_state.get("currency_symbol", "$")
                rate = get_currency_rate()
                report_path = generate_report(currency_symbol=sym, currency_rate=rate)
                st.success(f"✅ Report generated: {report_path}")

                with open(report_path, "rb") as f:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=f,
                        file_name="churn_intelligence_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Error generating report: {e}")


# ── Navigation & Page Routing ──────────────────────────────────────────────────

PAGES = ["📊 Executive Summary", "🔎 Customer Insights", "🎯 Prediction Center", "💡 Recommendations"]

def main():
    inject_css()

    # Initialize active page state
    if "active_page" not in st.session_state:
        st.session_state.active_page = PAGES[0]
    if "currency_symbol" not in st.session_state:
        st.session_state.currency_symbol = "$"

    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="background: linear-gradient(135deg, #6C63FF, #00D4AA);
                       -webkit-background-clip: text;
                       -webkit-text-fill-color: transparent;
                       font-size: 1.5rem; margin-bottom: 0.3rem;">
                🔍 Churn Intelligence
            </h2>
            <p style="color: #888899; font-size: 0.8rem;">
                Customer Analytics Platform
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        selected_page = st.radio(
            "Navigate",
            PAGES,
            index=PAGES.index(st.session_state.active_page),
            label_visibility="collapsed",
        )
        
        if selected_page != st.session_state.active_page:
            st.session_state.active_page = selected_page
            st.rerun()

        st.markdown("---")

        # Currency Selector
        curr_options = ["USD ($)", "EUR (€)", "INR (₹)", "GBP (£)", "JPY (¥)"]
        curr_symbols = ["$", "€", "₹", "£", "¥"]
        default_idx = curr_symbols.index(st.session_state.currency_symbol) if st.session_state.currency_symbol in curr_symbols else 0
        selected_curr = st.selectbox(
            "Select Currency",
            curr_options,
            index=default_idx,
        )
        new_symbol = curr_symbols[curr_options.index(selected_curr)]
        if new_symbol != st.session_state.currency_symbol:
            st.session_state.currency_symbol = new_symbol
            st.rerun()

        st.markdown("---")

        # Model info
        metrics = load_model_metrics()
        if metrics:
            best = max(metrics, key=lambda x: x["f1_score"])
            st.markdown(f"""
            <div style="background: {COLORS['card']}; border-radius: 8px;
                 padding: 0.8rem; font-size: 0.8rem;">
                <div style="color: {COLORS['muted']}; margin-bottom: 0.3rem;">
                    ACTIVE MODEL
                </div>
                <div style="color: {COLORS['primary']}; font-weight: 600;">
                    {best['model']}
                </div>
                <div style="color: {COLORS['text']}; margin-top: 0.3rem;">
                    F1: {best['f1_score']:.3f} | AUC: {best['roc_auc']:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="color: {COLORS['muted']}; font-size: 0.7rem; text-align: center;">
            Customer Churn Intelligence v1.0<br>
            Built with Streamlit & Scikit-Learn
        </div>
        """, unsafe_allow_html=True)

    # ── Top Navigation Bar (Horizontal Tabs) ──
    cols = st.columns(len(PAGES))
    for i, p in enumerate(PAGES):
        with cols[i]:
            is_active = st.session_state.active_page == p
            # Use primary type for active, secondary for inactive
            if st.button(p, key=f"nav_btn_{i}", use_container_width=True, 
                         type="primary" if is_active else "secondary"):
                st.session_state.active_page = p
                st.rerun()

    st.markdown("<hr style='margin: 0.5rem 0 1.5rem 0; opacity: 0.1;'>", unsafe_allow_html=True)

    # Route pages based on active_page
    current_page = st.session_state.active_page
    if "Executive Summary" in current_page:
        page_executive_summary()
    elif "Customer Insights" in current_page:
        page_customer_insights()
    elif "Prediction Center" in current_page:
        page_prediction_center()
    elif "Recommendations" in current_page:
        page_recommendations()


if __name__ == "__main__":
    main()
