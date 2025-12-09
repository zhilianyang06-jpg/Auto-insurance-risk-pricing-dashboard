import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

import streamlit as st
import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

import plotly.express as px
import plotly.graph_objects as go

# =============================
#  CONFIG & DATA PREP
# =============================

TRAIN_DATA_PATH = "train_data.csv"

FEATURES = [
    "observation_hour", "speed", "rpm", "acceleration", "throttle_position",
    "engine_temperature", "engine_load_value", "heart_rate", "current_weather",
    "visibility", "precipitation", "accidents_onsite", "design_speed", "accidents_time",
]

TARGET_COL = "risk_level"
RISK_ORDER = ["low", "medium", "high", "very high"]

DEFAULT_PRICING_PROFILES = {
    "aggressive": {"low": -0.15, "medium": 0.10, "high": 0.25, "very high": 0.45},
    "standard": {"low": -0.10, "medium": 0.00, "high": 0.10, "very high": 0.20},
    "conservative": {"low": -0.20, "medium": -0.10, "high": 0.20, "very high": 0.25},
}

# =============================
#  PROFESSIONAL CSS
# =============================
def local_css():
    st.markdown("""
    <style>
    /* 1. Global Settings */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1e293b; /* Slate-800 */
    }
    .stApp {
        background-color: #f8f9fa; /* Very light grey background */
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        max-width: 95% !important;
    }
    
    /* Headings */
    h1 { color: #0f172a; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.5rem; }
    h2 { color: #1e293b; border-bottom: 1px solid #e2e8f0; padding-bottom: 10px; margin-top: 40px !important; font-weight: 600; font-size: 1.5rem; }
    h3 { color: #334155; font-weight: 700; font-size: 1.2rem; margin-top: 1.5rem; margin-bottom: 0.5rem; }
    h4 { color: #475569; font-weight: 600; font-size: 1.1rem; margin-top: 1rem; }

    /* 2. Brighter Professional Blue Box */
    .pro-blue-box {
        background-color: #f0f9ff; /* Sky-50 */
        border-left: 5px solid #0ea5e9; /* Sky-500 */
        color: #0f172a; 
        padding: 18px;
        border-radius: 6px;
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 25px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #e0f2fe;
    }
    .pro-blue-box strong {
        color: #0284c7;
        font-weight: 700;
    }

    /* 3. Professional Strategy Cards */
    .strategy-panel {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        height: 100%;
        transition: border-color 0.2s;
    }
    .strategy-panel:hover { border-color: #cbd5e1; }
    
    /* Strategy Header */
    .strategy-header {
        background-color: #f8fafc;
        padding: 15px 20px;
        border-bottom: 1px solid #e2e8f0;
        border-radius: 8px 8px 0 0;
        margin-bottom: 20px;
        min-height: 110px; 
    }
    .strategy-header h4 {
        margin: 0 0 8px 0;
        font-size: 1.1rem;
        color: #0f172a;
    }
    .strategy-header p {
        margin: 0;
        font-size: 0.85rem;
        color: #475569; 
        line-height: 1.4;
        font-weight: 500;
    }

    /* 4. Input Grid Alignment */
    .risk-label-text {
        font-weight: 600;
        font-size: 0.8rem;
        color: #475569;
        text-transform: uppercase;
        padding-top: 12px;
    }
    .price-text {
        font-weight: 700;
        font-size: 1rem;
        color: #0284c7;
        text-align: left;
        padding-top: 8px;
        display: block;
    }
    .col-header {
        font-size: 0.7rem;
        color: #94a3b8;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 8px;
        display: block;
    }

    /* 5. Simple Metrics */
    .metric-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 15px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 2px;
    }
    .metric-sub { font-size: 0.85rem; color: #0ea5e9; font-weight: 600; }
    .metric-sub-neg { font-size: 0.85rem; color: #ef4444; font-weight: 600; }

    /* 6. Compact Grid */
    .compact-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 15px; }
    .grid-item {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px; text-align: center;
    }
    .grid-label { font-size: 0.75rem; color: #64748b; font-weight: 600; text-transform: uppercase; }
    .grid-val { font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-top: 2px; }

    /* Streamlit Inputs Clean up */
    div[data-testid="stNumberInput"] input {
        font-size: 0.9rem; padding: 0px 8px; height: 34px; border-radius: 4px; border: 1px solid #cbd5e1;
    }
                
    /* Make sidebar uploader text smaller (Drag & drop + limit text) */
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] section span {
        font-size: 0.75rem !important;
        line-height: 1.2 !important;
    }

    </style>
    """, unsafe_allow_html=True)

# =============================
#  MODEL & LOGIC
# =============================

@st.cache_resource
def train_gbm_model():
    try:
        df = pd.read_csv(TRAIN_DATA_PATH)
    except FileNotFoundError:
        return None, None, FEATURES, "File not found."
    
    missing = [c for c in FEATURES if c not in df.columns]
    if missing: return None, None, FEATURES, f"Missing: {missing}"

    X = df[FEATURES].copy()
    le = None
    if X["current_weather"].dtype == "object":
        le = LabelEncoder()
        X["current_weather"] = le.fit_transform(X["current_weather"])

    y = df[TARGET_COL].astype(int)
    
    # --- ORIGINAL MODEL PERFORMANCE LOGIC ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    return model, le, FEATURES, "OK"

def predict_risk(df_in, model, le, feats):
    df = df_in.copy()
    X = df[feats].copy()
    if le and X["current_weather"].dtype == "object":
        X["current_weather"] = le.transform(X["current_weather"])
    
    y_pred = model.predict(X)
    df["risk_level_num"] = y_pred.astype(int)
    mapping = {1: "low", 2: "medium", 3: "high", 4: "very high"}
    df["risk"] = df["risk_text"] = df["risk_level_num"].map(mapping)
    return df

def build_methodology_pdf():
    """
    Create a professional Methodology & References PDF fully in memory.
    """
    buffer = io.BytesIO()

    # --- Basic Styles ---
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontSize=22,
        leading=28,
        alignment=1,  # center
        spaceAfter=20,
    )
    h2 = ParagraphStyle(
        "Heading2",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
    )

    story = []

    # ----------------------------------------------------
    # COVER PAGE
    # ----------------------------------------------------
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph("Behavior-Based Auto Insurance Pricing", title_style))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Methodology & Technical References", styles["Heading3"]))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("Generated from Interactive Pricing Dashboard", body))
    story.append(PageBreak())

    # ----------------------------------------------------
    # 5.1 Workflow Architecture
    # ----------------------------------------------------
    story.append(Paragraph("5.1 Dashboard Workflow Architecture", h2))
    story.append(Paragraph(
        "This dashboard follows a linear actuarial simulation pipeline connecting "
        "machine learning–based risk segmentation with commercial pricing strategy. "
        "The workflow includes: Identification → Calibration → Static Pricing → Dynamic Acquisition.",
        body,
    ))
    story.append(Spacer(1, 0.2 * inch))

    # ----------------------------------------------------
    # 5.2 Scientific Basis
    # ----------------------------------------------------
    story.append(Paragraph("5.2 Scientific Basis: From Data to Risk Identification", h2))
    story.append(Paragraph(
        "Using the POLIDriving dataset framework (Marcillo et al., 2024), driver risk is inferred from "
        "telemetry, environmental, and physiological features through a Gradient Boosting Machine (GBM).",
        body,
    ))
    story.append(PageBreak())

    # ----------------------------------------------------
    # 5.3 Pricing Logic (with formula)
    # ----------------------------------------------------
    story.append(Paragraph("5.3 Actuarial Algorithms & Pricing Logic", h2))
    story.append(Paragraph("<b>A. Baseline Volume</b><br/>N_baseline = floor(N_pool × InitialRate)", body))
    story.append(Paragraph("<b>B. Static Pricing</b><br/>P_final = P_base × (1 + Loading%)", body))
    story.append(Paragraph("Rev_static = Σ(N_baseline × P_final)", body))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("<b>C. Dynamic Acquisition</b>", body))
    story.append(Paragraph("Rate_new = Rate_initial × (1 + E × Loading%)", body))
    story.append(Paragraph("N_acquired = floor(N_pool × Clamp(Rate_new, 0%, 100%))", body))
    story.append(PageBreak())

    # ----------------------------------------------------
    # 5.4 Assumptions
    # ----------------------------------------------------
    story.append(Paragraph("5.4 Key Modeling Assumptions", h2))
    story.append(Paragraph(
        "• The dataset is treated as a potential lead pool, not an in-force policy book.<br/>"
        "• Each risk segment uses shared base premiums and elasticity.<br/>"
        "• Rate_new is clamped between 0–100%.<br/>"
        "• Low-risk drivers are more price elastic (E ≈ -1.5), high-risk drivers less (E ≈ -0.2).",
        body,
    ))
    story.append(PageBreak())

    # ----------------------------------------------------
    # 5.5 References
    # ----------------------------------------------------
    story.append(Paragraph("5.5 Academic References", h2))
    story.append(Paragraph(
        "Marcillo et al. (2024). POLIDriving Dataset. Applied Sciences.<br/>"
        "Boucher & Inoussa (2014). Telematics and Price Elasticity. JRI.<br/>"
        "Einav, Finkelstein & Cullen (2010). Welfare in Insurance Markets. QJE.<br/>"
        "Barseghyan et al. (2013). Risk Preferences. RAND Journal.",
        body,
    ))

    # Page numbering
    def add_page_number(canvas, doc):
        page_num = canvas.getPageNumber()
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(letter[0] - 40, 20, f"Page {page_num}")

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)

    buffer.seek(0)
    return buffer



# =============================
#  MAIN APP
# =============================

st.set_page_config(page_title="Auto Insurance Pricing", layout="wide")
local_css()

st.title("🚗 Auto Insurance Pricing Dashboard")

# --- HEADER WORKFLOW (FIXED DISPLAY) ---
st.markdown("""
<div style="background-color:#fff; padding:20px; border-radius:8px; border:1px solid #e2e8f0; margin-bottom:20px;">
    <strong style="color:#1e293b; font-size:1rem;">Workflow Overview:</strong>
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px; margin-top:10px; font-size:0.9rem; color:#475569;">
        <div>1. <b>Pool Identification:</b> Upload data & classify potential leads (N_pool).</div>
        <div>2. <b>Baseline (Status Quo):</b> Apply initial acceptance rate to get status quo (N_baseline).</div>
        <div>3. <b>Strategy & Elasticity:</b> Price changes adjust acceptance (Elasticity) &rarr; Final Volume (N_acquired).</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.header("Configuration")

# 1. Train/load GBM model FIRST (must be before any sidebar-dependent logic)
model_res = train_gbm_model()

if not model_res[0]:
    st.sidebar.error("Model Error")
    st.stop()
else:
    model, le, feats, _ = model_res
    st.sidebar.success("GBM Model Active")


# -------------------------------
# 1. Upload Data
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("1. Upload Data")
up_file = st.sidebar.file_uploader("Upload Data", type=["csv", "xlsx"])


# -------------------------------
# 2. Risk Base Premiums
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("2. Risk Base Premiums ($)")

base_premium = {}
defs = {"low": 700.0, "medium": 850.0, "high": 950.0, "very high": 1150.0}
for r in RISK_ORDER:
    base_premium[r] = st.sidebar.number_input(f"{r.title()}", value=defs[r], step=10.0)


# -------------------------------
# 3. Baseline Acceptance
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("3. Baseline Acceptance")
st.sidebar.caption("Initial % of the pool accepting the Base Price.")

init_accept = {}
for r in RISK_ORDER:
    init_accept[r] = st.sidebar.slider(f"{r.title()} %", 0.0, 100.0, 50.0, 5.0) / 100.0


# -------------------------------
# 4. Strategy
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("4. Strategy")

main_strat = st.sidebar.selectbox(
    "Select Pricing Strategy",
    ["aggressive", "standard", "conservative"],
    index=1
)


if up_file is None:
    st.info("👈 Please upload a file to start.")
    st.stop()

# --- LOAD DATA ---
if up_file.name.endswith(".csv"): df_raw = pd.read_csv(up_file)
else: df_raw = pd.read_excel(up_file)

try:
    df_norm = predict_risk(df_raw, model, le, feats)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# --- GLOBAL CALCULATIONS (INT ROUNDING DOWN) ---
pool_counts = df_norm["risk"].value_counts().reindex(RISK_ORDER).fillna(0).astype(int)
total_pool = int(len(df_norm))

baseline_metrics = {}
total_base_cust = 0
total_base_rev = 0

for r in RISK_ORDER:
    n_p = int(pool_counts[r])
    rate = init_accept[r]
    n_b = int(n_p * rate) 
    rev = n_b * base_premium[r]
    
    total_base_cust += n_b
    total_base_rev += rev
    baseline_metrics[r] = {"pool": n_p, "base": n_b, "price": base_premium[r]}

# ==========================================
# SECTION 1: POOL VIEW
# ==========================================
st.markdown("## 1.Baseline: Traditional Pricing")

st.markdown("""
<span style="color:#475569; font-size:0.9rem;">
<em>
This section provides an overview of the risk groups identified through telematics-based driving-behavior analysis.
<br>We assume the dataset represents potential new customers, priced using default demographic-based premiums and the baseline acceptance rates from the sidebar.
<br>It presents the insurer’s traditional pricing baseline: customer volume and revenue calculated without any risk-based adjustments or elasticity effects.  

</em>
</span>
<br><br>
""", unsafe_allow_html=True)



with st.container():
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Pie Chart
        rd = pool_counts.reset_index()
        rd.columns = ["risk", "count"]
        # Fresh Professional Palette
        colors = {'low':'#34d399', 'medium':'#fbbf24', 'high':'#fb923c', 'very high':'#f87171'}
        
        fig = px.pie(rd, names="risk", values="count", title="Pool Risk Distribution", hole=0.6,
                     color="risk", color_discrete_map=colors)
        fig.update_layout(margin=dict(t=30,b=0,l=0,r=0), height=250, template="plotly_white")
        fig.update_traces(textposition="inside", textinfo="percent")
        st.plotly_chart(fig, use_container_width=True)
        
        # Compact Grid
        grid_html = '<div class="compact-grid">'
        for r in RISK_ORDER:
            grid_html += f'<div class="grid-item"><div class="grid-label">{r}</div><div class="grid-val">{pool_counts[r]:,}</div></div>'
        grid_html += '</div>'
        st.markdown(grid_html, unsafe_allow_html=True)

    with col2:
        # Metrics Row
        max_rev = sum([pool_counts[r] * base_premium[r] for r in RISK_ORDER])
        
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Total Pool Size</div>
                <div class="metric-value">{total_pool:,}</div>
                <div class="metric-sub" style="color:#64748b;">Potential Leads</div>
            </div>
            """, unsafe_allow_html=True)
        with c_m2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Baseline Revenue</div>
                <div class="metric-value" style="color:#0284c7;">${total_base_rev:,.0f}</div>
                <div class="metric-sub" style="color:#64748b;">Based on Base Premium × Baseline Acceptance</div>
            </div>
            """, unsafe_allow_html=True)

            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Data Table
        st.markdown("##### Uploaded Data Preview")
        cols = ["risk", "risk_level_num"] + [c for c in df_norm.columns if c not in ["risk", "risk_level_num", "risk_text"]]
        st.dataframe(df_norm[cols].head(6), use_container_width=True, hide_index=True)

# ==========================================
# SECTION 2: STRATEGY-BASED RISK PRICING + ELASTICITY 
# ==========================================

st.markdown("## 2.Strategy-Based Risk Pricing with Elasticity")

st.markdown("""
<span style="color:#475569; font-size:0.9rem;">
<em>
In this section, insurers can design more granular risk pricing and incorporate 
<strong>price elasticity (E)</strong> to estimate customer volume and revenue with 
real market responses.
<br>
-  insurers can <strong>adjust discounts or surcharges</strong> for each risk group under the three available pricing strategies.  
<br>
- insurers can <strong>modify the price elasticity</strong> for each risk level using the sliders.  
<br>
- By combining strategy adjustments with elasticity, nsurers can simulate the actual revenue under realistic market conditions. 
- The purpose of the simulation is to show how strategic choices interact with telematics risk segmentation to drive volume and revenue.
</em>
</span>
<br><br>
""", unsafe_allow_html=True)

# -------------------------------
# 2.1 Strategy Pricing Panels
# -------------------------------
st.markdown("### Risk-based Pricing Strategy")

st.markdown("""
<span style="color:#475569; font-size:0.85rem;">
<em>
Under each pricing strategy, insurers can customize discounts or surcharges</strong> for each risk group. 
<br>Insurers differ in their risk appetite, regulatory constraints, target market, and pricing strategy. For these reasons, the dashboard treats these inputs as insurer choices rather than fixed backend calculations.
</em>
</span>
<br><br>
""", unsafe_allow_html=True)

strategies = ["aggressive", "standard", "conservative"]
adj_profiles = {s: {} for s in strategies}

s_desc = {
    "aggressive": "Focuses on maximizing profit per customer. <br>Applies steep surcharges to higher-risk customers while offering limited discounts to low-risk segments.",
    "standard":   "Aims to balance profitability and growth. <br>Uses moderate, symmetric adjustments around the base price, with mild discounts for low risk and controlled surcharges for higher risk.",
    "conservative": "Prioritizes customer acquisition and market share.<br> Applies deeper discounts to low- and medium-risk customers and caps price increases for higher-risk segments."
}

cols = st.columns(3)
for i, strat in enumerate(strategies):
    with cols[i]:
        st.markdown(f"""
        <div class="strategy-panel">
            <div class="strategy-header">
                <h4>{strat.title()}</h4>
                <p>{s_desc[strat]}</p>
            </div>
            <div style="padding: 5px 15px 15px 15px;">
        """, unsafe_allow_html=True)
        
        h1, h2, h3 = st.columns([0.8, 1.1, 1.1])
        h1.markdown('<span class="col-header">RISK</span>', unsafe_allow_html=True)
        h2.markdown('<span class="col-header">adj%-percentage change applied to the base premium</span>', unsafe_allow_html=True)
        h3.markdown('<span class="col-header">PRICE</span>', unsafe_allow_html=True)
        
        for r in RISK_ORDER:
            bp = base_premium[r]
            d_pct = DEFAULT_PRICING_PROFILES[strat][r] * 100
            
            c1, c2, c3 = st.columns([0.8, 1.1, 1.1])
            with c1:
                st.markdown(f'<div class="risk-label-text">{r}</div>', unsafe_allow_html=True)
            with c2:
                v = st.number_input(
                    f"{r}{strat}", -90.0, 200.0, float(d_pct), 5.0,
                    key=f"{strat}_{r}", label_visibility="collapsed"
                )
            with c3:
                load = v / 100.0
                adj_profiles[strat][r] = load
                final = bp * (1 + load)
                st.markdown(f'<span class="price-text">${final:,.0f}</span>', unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)



# ----------------------------------------------------------
# 2.2 Elasticity Sensitivity Input
# ----------------------------------------------------------

st.markdown("### Elasticity(E)")
st.markdown("""
<span style="color:#475569; font-size:0.85rem;">
<em>
Elasticity reflects how sensitive customers are to price changes.  
<br>For example, <strong>E (Low Risk) = -1.8</strong> means that for low-risk customers, 
a 1% increase in price is expected to reduce acceptance by approximately <strong>1.8%</strong>.
<br>Customers leave when price rises beyond their willingness to pay, and new customers join when lower prices make the policy more attractive.
<br>Low-risk drivers are generally more price-sensitive (high elasticity), while high-risk drivers tend to be less sensitive (low elasticity). 
<br>These behavioral shifts are reflected through changes in each segment’s acceptance rate.
<br><br>
</em>
</span>
""", unsafe_allow_html=True)

elast = {}
defs_e = {"low": -1.8, "medium": -0.8, "high": -0.3, "very high": -0.1}

ec = st.columns(4)
for i, r in enumerate(RISK_ORDER):
    with ec[i]:
        elast[r] = st.slider(f"{r.title()}", -3.0, 0.0, defs_e[r], 0.1)



# ----------------------------------------------------------
# 2.3 Forecast for Main Strategy
# ----------------------------------------------------------
st.markdown("""
<span style="color:#475569; font-size:0.85rem;">
<em>
On left side you can change the strategy.
</em>
</span>
<br><br>
""", unsafe_allow_html=True)

# Calculate dynamic acquisition results
acq_data = []
prof_m = adj_profiles[main_strat]

total_dyn_rev = 0
total_dyn_cust = 0

for r in RISK_ORDER:
    pool_n = baseline_metrics[r]["pool"]
    base_price = baseline_metrics[r]["price"]
    init_rate = init_accept[r]
    load = prof_m[r]
    E = elast[r]

    new_price = base_price * (1 + load)
    new_rate = max(0.0, min(init_rate * (1 + E * load), 1.0))
    
    new_n = int(pool_n * new_rate)
    new_rev = int(new_n * new_price)
    
    total_dyn_cust += new_n
    total_dyn_rev += new_rev

# Compare ONLY with Baseline
rev_uplift = total_dyn_rev - total_base_rev
cust_uplift = total_dyn_cust - total_base_cust

st.markdown(f"### Prediction for {main_strat.title()} Strategy")

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Revenue</div>
        <div class="metric-value">${total_dyn_rev:,.0f}</div>
        <div class="metric-sub">Final Revenue</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Revenue Change (vs Baseline)</div>
        <div class="metric-value">{rev_uplift:+,.0f}</div>
        <div class="metric-sub">${total_base_rev:,.0f} → ${total_dyn_rev:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Customer Change (vs Baseline)</div>
        <div class="metric-value">{cust_uplift:+,.0f}</div>
        <div class="metric-sub">{total_base_cust:,} → {total_dyn_cust:,}</div>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------
# 2.4 Volume Shift (before vs after)
# ----------------------------------------------------------

# === Combined Layout: Customer Change Pie (left) + Baseline vs New Bar (right) ===
st.markdown(f"#### Impact of Pricing on Customer Volume Under {main_strat.title()} Strategy (Baseline vs New) ")

left_col, right_col = st.columns([1, 1])

# -----------------------
# LEFT: Customer Change Pie Chart
# -----------------------
with left_col:
    st.markdown("###### Customer Change Distribution")

    prof_m = adj_profiles[main_strat]
    changes = {"increased": 0, "decreased": 0, "unchanged": 0}

    for r in RISK_ORDER:
        pool = baseline_metrics[r]["pool"]
        init_r = init_accept[r]
        load = prof_m[r]
        E = elast[r]

        new_rate = max(0.0, min(init_r * (1 + E * load), 1.0))
        new_n = int(pool * new_rate)
        base_n = baseline_metrics[r]["base"]

        if new_n > base_n:
            changes["increased"] += new_n - base_n
        elif new_n < base_n:
            changes["decreased"] += base_n - new_n
        else:
            changes["unchanged"] += base_n

    change_df = pd.DataFrame({
        "type": list(changes.keys()),
        "count": list(changes.values())
    })

    change_colors = {
        "increased": "#34d399",
        "decreased": "#f87171",
        "unchanged": "#fbbf24"
    }

    fig_change = px.pie(
        change_df,
        names="type",
        values="count",
        hole=0.6,
        title="Customer Change Distribution",
        color="type",
        color_discrete_map=change_colors
    )

    fig_change.update_layout(
        margin=dict(t=40, b=0, l=0, r=0),
        height=260,
        template="plotly_white"
    )
    fig_change.update_traces(textposition="inside", textinfo="percent")
    st.plotly_chart(fig_change, use_container_width=True)


# -----------------------
# RIGHT: Baseline vs New Customer Count (bar)
# -----------------------
with right_col:
    st.markdown("###### Customer Volume Shift: Baseline vs New After Pricing Change")

    hist_d = []
    prof_m = adj_profiles[main_strat]

    for r in RISK_ORDER:
        pool = baseline_metrics[r]["pool"]
        init_r = init_accept[r]
        load = prof_m[r]
        E = elast[r]
        new_rate = max(0.0, min(init_r * (1 + E * load), 1.0))
        new_n = int(pool * new_rate)
        
        hist_d.append({"Risk": r, "Type": "Baseline", "Count": baseline_metrics[r]["base"]})
        hist_d.append({"Risk": r, "Type": "New", "Count": new_n})

    fig_h = px.bar(
        pd.DataFrame(hist_d),
        x="Risk",
        y="Count",
        color="Type",
        barmode="group",
        color_discrete_map={"Baseline": "#d1d5db", "New": "#0ea5e9"},
        title="Baseline vs New"
    )
    fig_h.update_layout(
        bargap=0.35,
        template="plotly_white",
        margin=dict(t=50, l=0, r=0, b=0),
        height=240
    )
    fig_h.update_traces(texttemplate='%{y:,.0f}', textposition='inside')
    st.plotly_chart(fig_h, use_container_width=True)



# ----------------------------------------------------------
# 2.5 Revenue & Customer Impact (vs Baseline) - PRO VIEW
# ----------------------------------------------------------

st.markdown("### Revenue and Customer Uplift Relative to Baseline")

# Calculate %
rev_pct = (total_dyn_rev - total_base_rev) / total_base_rev if total_base_rev else 0
cust_pct = (total_dyn_cust - total_base_cust) / total_base_cust if total_base_cust else 0

rev_pct_text = f"{rev_pct*100:+.1f}%"
cust_pct_text = f"{cust_pct*100:+.1f}%"


# ==========================================
# Two Cards (Revenue + Customer)
# ==========================================

c1, c2 = st.columns(2)

# ------------------------------------------
# Revenue Card (Title + Chart together)
# ------------------------------------------
with c1:

    # Revenue Uplift Title (full width, no border)
    st.markdown(f"""
    <div style="
        width:100%;
        text-align:center;
        padding:4px 0 8px 0;
        margin-bottom:6px;
    ">
        <span style="font-size:1.20rem; font-weight:700; color:#0f172a;">
            Revenue Uplift
        </span>
        <span style="font-size:1.35rem; font-weight:800; color:#0284c7; margin-left:6px;">
            {rev_pct_text}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Revenue Chart
    max_rev = max(total_base_rev, total_dyn_rev)

    fig_rev = go.Figure()
    fig_rev.add_bar(
        x=["Baseline", "New"],
        y=[total_base_rev, total_dyn_rev],
        marker_color=["#cbd5e1", "#0284c7"],
        width=0.35,     # narrower columns
        text=[total_base_rev, total_dyn_rev],
        textposition="outside",
        texttemplate="%{text:,}"
    )

    fig_rev.update_layout(
        template="plotly_white",
        height=220,
        margin=dict(t=10, b=20, l=0, r=0),
        yaxis=dict(
            range=[0, max_rev * 1.35],   # prevent clipping
            showgrid=True,
            gridcolor="#f1f5f9"
        ),
        xaxis=dict(title=None),
    )

    st.plotly_chart(fig_rev, use_container_width=True)


# ---------------
# Customer Card 
# ---------------
with c2:

    # Customer Uplift Title (full width, no border)
    st.markdown(f"""
    <div style="
        width:100%;
        text-align:center;
        padding:4px 0 8px 0;
        margin-bottom:6px;
    ">
        <span style="font-size:1.20rem; font-weight:700; color:#0f172a;">
            Customer Uplift
        </span>
        <span style="font-size:1.35rem; font-weight:800; color:#0284c7; margin-left:6px;">
            {cust_pct_text}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Customer Chart
    max_cust = max(total_base_cust, total_dyn_cust)

    fig_c = go.Figure()
    fig_c.add_bar(
        x=["Baseline", "New"],
        y=[total_base_cust, total_dyn_cust],
        marker_color=["#cbd5e1", "#0284c7"],
        width=0.35,     # narrower columns
        text=[total_base_cust, total_dyn_cust],
        textposition="outside",
        texttemplate="%{text:,}"
    )

    fig_c.update_layout(
        template="plotly_white",
        height=220,
        margin=dict(t=10, b=20, l=0, r=0),
        yaxis=dict(
            range=[0, max_cust * 1.35],
            showgrid=True,
            gridcolor="#f1f5f9"
        ),
        xaxis=dict(title=None),
    )

    st.plotly_chart(fig_c, use_container_width=True)


# ==========================================
# SECTION 3: EXPORT
# ==========================================
st.markdown("## 3. Export Results")
st.caption("Download includes: Original Data features, Predicted Risk Level, Base Premium, and Final Strategy Premium.")

df_out = df_norm.copy()
df_out['base_premium'] = df_out['risk'].map(base_premium)
prof_out = adj_profiles[main_strat]
df_out['final_premium'] = df_out.apply(lambda x: x['base_premium'] * (1 + prof_out.get(x['risk'], 0)), axis=1)

st.download_button("Download Full Dataset (CSV)", df_out.to_csv(index=False).encode('utf-8'), "pricing_export.csv", "text/csv")

# ==========================================
# SECTION 4: NOTES (METHODOLOGY)
# ==========================================
st.markdown("---")
st.markdown("## 5. Methodology & References")
st.markdown(
    "<span style='color:#475569; font-size:0.9rem;'>Download the full technical documentation including workflow, algorithms, formulas, and academic references.</span>",
    unsafe_allow_html=True
)
st.markdown("##### 📄 Download Full Methodology & References (PDF)")

try:
    with open("Methodology_References.pdf", "rb") as pdf_file:
        st.download_button(
            label="⬇️ Download Methodology PDF",
            data=pdf_file,
            file_name="Methodology_References.pdf",
            mime="application/pdf"
        )
except FileNotFoundError:
    st.error("Methodology_References.pdf not found. Please upload it to the GitHub repository root directory.")
