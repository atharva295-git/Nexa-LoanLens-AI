# =============================================================
#  app.py  —  NexaBank Financial Services  (v6 — Modern UI)
#  Run:  streamlit run app.py
# =============================================================
import streamlit as st
import pandas as pd
import numpy as np
import re, os, warnings, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, roc_curve)
from src.preprocess import load_and_preprocess
from src.explainer   import explain_single

st.set_page_config(page_title="NexaBank", page_icon="🏦",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── Page background ── */
.stApp {
    background: #f0fdf4;
    background-image: radial-gradient(ellipse at 20% 20%, rgba(5,150,105,0.06) 0%, transparent 50%),
                      radial-gradient(ellipse at 80% 80%, rgba(16,185,129,0.05) 0%, transparent 50%);
}

/* ── Remove streamlit padding ── */
.block-container { padding: 1.5rem 2rem !important; }

/* ── Buttons ── */
.stButton button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: .85rem !important;
    letter-spacing: .01em !important;
    transition: all .2s !important;
    padding: .5rem 1.2rem !important;
}
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    color: #fff !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(5,150,105,.4) !important;
}
.stButton button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 28px rgba(5,150,105,.5) !important;
}
.stButton button[kind="secondary"] {
    background: rgba(255,255,255,0.8) !important;
    backdrop-filter: blur(10px) !important;
    color: #374151 !important;
    border: 1.5px solid rgba(226,232,240,0.8) !important;
}
.stButton button:disabled { opacity: .45 !important; }

/* ── Inputs ── */
.stSelectbox > div > div,
.stNumberInput input {
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.9) !important;
    backdrop-filter: blur(10px) !important;
    color: #111827 !important;
    font-weight: 500 !important;
    transition: border-color .2s !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput input:focus {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 3px rgba(5,150,105,.12) !important;
}
label {
    color: #374151 !important;
    font-size: .82rem !important;
    font-weight: 600 !important;
    letter-spacing: .01em !important;
}

/* ── Metrics ── */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.85) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.9) !important;
    padding: 1.1rem !important;
    box-shadow: 0 2px 12px rgba(0,0,0,.06), 0 1px 3px rgba(0,0,0,.04) !important;
}
div[data-testid="metric-container"] label {
    color: #6b7280 !important;
    font-size: .68rem !important;
    font-weight: 700 !important;
    letter-spacing: .07em !important;
    text-transform: uppercase !important;
}
div[data-testid="stMetricValue"] > div {
    color: #111827 !important;
    font-size: 1.55rem !important;
    font-weight: 800 !important;
}
div[data-testid="stMetricDelta"] { color: #9ca3af !important; font-size: .72rem !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.7) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(226,232,240,0.8) !important;
    font-weight: 600 !important;
    color: #374151 !important;
}
hr { border-color: rgba(226,232,240,0.6) !important; }

/* ── Alerts ── */
.stAlert { border-radius: 12px !important; }

/* ── Card glass effect ── */
.glass-card {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,.06), 0 1px 3px rgba(0,0,0,.04);
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
if "landing"     not in st.session_state: st.session_state.landing     = True
if "page"        not in st.session_state: st.session_state.page        = "predict"
if "show_result" not in st.session_state: st.session_state.show_result = False

# ── Option maps ───────────────────────────────────────────────
PURPOSE_MAP = {
    "Car / Vehicle Purchase":               "car",
    "House Purchase":                       "car",
    "Home Renovation / Repairs":            "repairs",
    "Furniture & Equipment":                "furniture/equipment",
    "Electronics / TV / Radio":             "radio/TV",
    "Education / Tuition Fees":             "education",
    "Business Investment":                  "business",
    "Medical Expenses":                     "repairs",
    "Travel / Vacation":                    "vacation/others",
    "Debt Consolidation":                   "vacation/others",
    "Wedding / Special Event":              "vacation/others",
    "Domestic Appliances":                  "domestic appliances",
    "Other / Personal Use":                 "vacation/others",
}
HOUSING_MAP = {
    "Own Home (Mortgage-free)":             "own",
    "Own Home (With Mortgage)":             "own",
    "Renting":                              "rent",
    "Living with Family":                   "free",
    "Company / Employer Provided":          "free",
}
SAVING_MAP = {
    "Low  (< ₹3,000)":                     "little",
    "Moderate  (₹3,000 – ₹15,000)":        "moderate",
    "Good  (₹15,000 – ₹30,000)":           "quite rich",
    "High  (> ₹30,000)":                   "rich",
    "No Savings Account":                   "no_savings",
}
CURRENT_MAP = {
    "Low Balance  (< ₹6,000)":             "little",
    "Moderate  (₹6,000 – ₹24,000)":        "moderate",
    "High  (> ₹24,000)":                   "rich",
    "No Current Account":                   "no_current",
}
JOB_MAP = {
    "Unemployed / Non-resident":                0,
    "Unskilled – Permanent Resident":           1,
    "Skilled Employee / Official":              2,
    "Manager / Self-Employed / Highly Skilled": 3,
}
FEAT_LABELS = {
    "Checking account": "Current Account balance",
    "Saving accounts":  "Savings Account balance",
    "Credit amount":    "Loan amount",
    "Duration":         "Loan duration",
    "Age":              "Applicant age",
    "Job":              "Employment level",
    "Housing":          "Housing type",
    "Purpose":          "Loan purpose",
    "Sex":              "Gender",
}
FEAT_VALUE_MAP = {
    "Checking account": {0:"Low (<₹6k)",1:"Moderate (₹6k–₹24k)",2:"High (>₹24k)",3:"None"},
    "Saving accounts":  {0:"Low (<₹3k)",1:"Moderate",2:"Good",3:"High",4:"None"},
    "Job": {0:"Unemployed",1:"Unskilled",2:"Skilled",3:"Highly Skilled"},
    "Housing": {0:"Free/Family",1:"Own home",2:"Renting"},
}
SKIP_VALUES = {"unknown","nan","none","4","3"}

# ── Load models ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising NexaBank AI…")
def load_all():
    (_,_,y_tr,y_te,_,s_te,X_tr_raw,X_te_raw) = load_and_preprocess()
    scaler = joblib.load("models/scaler.pkl")
    model  = joblib.load("models/lr_fair.pkl")
    pipe   = Pipeline([("scaler",scaler),("clf",model)])
    pipe.fit(X_tr_raw, y_tr)
    enc = joblib.load("models/encoders.pkl")
    fr  = (joblib.load("models/fairness_results.pkl")
           if os.path.exists("models/fairness_results.pkl") else None)
    return pipe,enc,X_te_raw,y_te,s_te,fr,X_tr_raw,y_tr

pipe,enc,X_te_raw,y_te,s_te,fr,X_tr_raw,y_tr = load_all()

def light_fig(w=6,h=4):
    fig,ax = plt.subplots(figsize=(w,h))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("#f0fdf4")
    return fig,ax


# ══════════════════════════════════════════════════════════════
#  LANDING PAGE
# ══════════════════════════════════════════════════════════════
if st.session_state.landing:

    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800;900&family=Inter:wght@400;500;600&display=swap');

.stApp { background:#f0fdf4; font-family:'Inter',sans-serif!important; }
.block-container { padding:0 2rem 2rem!important; max-width:100%!important; }
[data-testid="collapsedControl"] { display:none!important; }
#MainMenu,footer,header { visibility:hidden; }

.stButton button {
    font-family:'Manrope',sans-serif!important;
    border-radius:12px!important; font-weight:700!important;
    font-size:.85rem!important; letter-spacing:.01em!important;
    transition:all .22s ease!important;
}
.stButton button[kind="primary"] {
    background:linear-gradient(135deg,#059669,#047857)!important;
    border:none!important; color:#fff!important;
    box-shadow:0 4px 20px rgba(5,150,105,.35)!important;
}
.stButton button[kind="primary"]:hover {
    transform:translateY(-2px)!important;
    box-shadow:0 10px 32px rgba(5,150,105,.45)!important;
}
.stButton button[kind="secondary"] {
    background:rgba(255,255,255,.9)!important;
    border:1.5px solid #d1fae5!important; color:#065f46!important;
}
.stButton button:disabled { opacity:.38!important; }
</style>
""", unsafe_allow_html=True)

    # ── NAVBAR ────────────────────────────────────────────────
    nav_l, nav_m, nav_r = st.columns([1.4, 2.2, 1.4])

    with nav_l:
        st.markdown("""
<div style="display:flex;align-items:center;gap:11px;padding:.85rem 0">
  <div style="width:40px;height:40px;border-radius:12px;
       background:linear-gradient(145deg,#059669,#047857);
       display:flex;align-items:center;justify-content:center;
       font-size:1.15rem;box-shadow:0 4px 16px rgba(5,150,105,.3)">🏦</div>
  <div>
    <div style="font-family:'Manrope',sans-serif;font-size:1.1rem;
                font-weight:800;color:#022c22;letter-spacing:-.4px">NexaBank</div>
    <div style="font-size:.58rem;color:#6b7280;font-weight:600;
                letter-spacing:.1em;text-transform:uppercase">Financial Services</div>
  </div>
</div>""", unsafe_allow_html=True)

    with nav_m:
        st.markdown("""
<div style="display:flex;justify-content:center;align-items:center;
     gap:2.5rem;padding:1rem 0">
  <span style="font-size:.83rem;color:#374151;font-weight:600;cursor:pointer">Products</span>
  <span style="font-size:.83rem;color:#374151;font-weight:600;cursor:pointer">Rates</span>
  <span style="font-size:.83rem;color:#374151;font-weight:600;cursor:pointer">About</span>
  <span style="font-size:.83rem;color:#374151;font-weight:600;cursor:pointer">Support</span>
</div>""", unsafe_allow_html=True)

    with nav_r:
        st.markdown("""
<div style="display:flex;justify-content:flex-end;align-items:center;
     gap:.8rem;padding:.85rem 0">
  <span style="font-size:.82rem;font-weight:600;color:#374151;
               padding:7px 16px;cursor:pointer">Log In</span>
  <span style="background:linear-gradient(135deg,#059669,#047857);
       color:#fff;font-size:.82rem;font-weight:700;
       padding:8px 20px;border-radius:100px;cursor:pointer;
       box-shadow:0 4px 14px rgba(5,150,105,.3)">Open Account</span>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── HERO ──────────────────────────────────────────────────
    # Entire hero rendered in ONE st.markdown call so the outer
    # wrapper div actually contains everything (Streamlit renders
    # each st.markdown call as its own isolated HTML fragment, so
    # splitting an open <div> across multiple calls breaks nesting
    # and lets content spill outside the dark background).
    st.markdown("""
<div style="background:linear-gradient(160deg,#022c22 0%,#064e3b 28%,#065f46 58%,#022c22 100%);border-radius:28px;padding:5.5rem 4rem 4.5rem;margin-bottom:2.5rem;position:relative;overflow:hidden;text-align:center">
<div style="position:absolute;top:0;left:50%;transform:translateX(-50%);width:180px;height:3px;border-radius:0 0 6px 6px;background:linear-gradient(90deg,transparent,#f59e0b,transparent)"></div>
<div style="position:absolute;top:-130px;left:-100px;width:520px;height:520px;border-radius:50%;opacity:.3;filter:blur(80px);background:radial-gradient(circle,#10b981,transparent 65%)"></div>
<div style="position:absolute;bottom:-100px;right:-80px;width:460px;height:460px;border-radius:50%;opacity:.22;filter:blur(80px);background:radial-gradient(circle,#f59e0b,transparent 65%)"></div>
<div style="position:relative;z-index:2">
<div style="display:inline-flex;align-items:center;gap:8px;background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.22);padding:6px 20px;border-radius:100px;margin-bottom:2rem">
<div style="width:7px;height:7px;border-radius:50%;background:#34d399"></div>
<span style="font-size:.7rem;font-weight:700;color:#6ee7b7;letter-spacing:.1em;text-transform:uppercase">AI-Powered Banking</span>
</div>
<div style="text-align:center">
<div style="font-family:sans-serif;font-size:3.8rem;font-weight:900;color:#fff;margin:0 0 1rem;letter-spacing:-2px;line-height:1.08">
Banking that explains
</div>
<div style="font-family:sans-serif;font-size:3.8rem;font-weight:900;margin:0 0 1rem;letter-spacing:-2px;line-height:1.08;background:linear-gradient(95deg,#6ee7b7 0%,#34d399 40%,#fcd34d 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">
every decision.
</div>
<p style="font-size:1.05rem;color:rgba(255,255,255,.58);max-width:500px;margin:0 auto 2.5rem;line-height:1.85;font-weight:400">
The first AI loan system that tells you not just
<span style="color:rgba(255,255,255,.85)">yes or no</span>, but
<span style="color:#6ee7b7;font-weight:600">exactly what to change</span>
to get approved. Fair. Instant. Transparent.
</p>
</div>
<div style="text-align:center;margin-bottom:2.5rem">
<span style="display:inline-block;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.11);color:rgba(255,255,255,.82);padding:8px 18px;border-radius:100px;font-size:.76rem;font-weight:600;margin:4px">Decision in 2 seconds</span>
<span style="display:inline-block;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.11);color:rgba(255,255,255,.82);padding:8px 18px;border-radius:100px;font-size:.76rem;font-weight:600;margin:4px">Gender-Fair Certified</span>
<span style="display:inline-block;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.11);color:rgba(255,255,255,.82);padding:8px 18px;border-radius:100px;font-size:.76rem;font-weight:600;margin:4px">Every Rejection Explained</span>
<span style="display:inline-block;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.11);color:rgba(255,255,255,.82);padding:8px 18px;border-radius:100px;font-size:.76rem;font-weight:600;margin:4px">Bank-Grade Security</span>
</div>
<div style="text-align:center;padding-bottom:1rem">
<div style="display:inline-flex;background:rgba(0,0,0,.22);border:1px solid rgba(255,255,255,.07);border-radius:20px;padding:.35rem">
<div style="padding:1.1rem 2rem;text-align:center;border-right:1px solid rgba(255,255,255,.07)">
<div style="font-size:1.9rem;font-weight:800;color:#fff;line-height:1">1,000+</div>
<div style="font-size:.63rem;color:rgba(255,255,255,.38);margin-top:5px;letter-spacing:.08em">APPLICATIONS</div>
</div>
<div style="padding:1.1rem 2rem;text-align:center;border-right:1px solid rgba(255,255,255,.07)">
<div style="font-size:1.9rem;font-weight:800;color:#6ee7b7;line-height:1">92%</div>
<div style="font-size:.63rem;color:rgba(255,255,255,.38);margin-top:5px;letter-spacing:.08em">FAIRNESS SCORE</div>
</div>
<div style="padding:1.1rem 2rem;text-align:center;border-right:1px solid rgba(255,255,255,.07)">
<div style="font-size:1.9rem;font-weight:800;color:#fcd34d;line-height:1">&lt;2 sec</div>
<div style="font-size:.63rem;color:rgba(255,255,255,.38);margin-top:5px;letter-spacing:.08em">DECISION TIME</div>
</div>
<div style="padding:1.1rem 2rem;text-align:center">
<div style="font-size:1.9rem;font-weight:800;color:#fff;line-height:1">Rs.0</div>
<div style="font-size:.63rem;color:rgba(255,255,255,.38);margin-top:5px;letter-spacing:.08em">PROCESSING FEE</div>
</div>
</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

    # CTA button below hero — deliberate placement after credibility
    _, cta_col, _ = st.columns([2, 1.5, 2])
    with cta_col:
        if st.button("Check My Loan Eligibility \u2192", type="primary",
                     use_container_width=True, key="hero_cta"):
            st.session_state.landing = False
            st.rerun()

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

    # ── SERVICES ──────────────────────────────────────────────
    st.markdown("""
<div style="text-align:center;margin-bottom:2.2rem">
  <div style="display:inline-block;font-size:.68rem;font-weight:700;
       color:#059669;letter-spacing:.14em;text-transform:uppercase;
       margin-bottom:.6rem;padding:4px 14px;border-radius:100px;
       background:#ecfdf5;border:1px solid #a7f3d0">Our Services</div>
  <h2 style="font-family:'Manrope',sans-serif;font-size:2.1rem;font-weight:800;
             color:#022c22;margin:0 0 .5rem;letter-spacing:-1px">
    Complete Financial Services
  </h2>
  <p style="font-size:.88rem;color:#6b7280;margin:0 auto;
            max-width:420px;line-height:1.65">
    One platform for all your banking needs.
  </p>
</div>
""", unsafe_allow_html=True)

    # ── ROW 1 ─────────────────────────────────────────────────
    sc1, sc2, sc3 = st.columns(3, gap="medium")

    with sc1:
        st.markdown("""
<div style="background:linear-gradient(145deg,#022c22 0%,#064e3b 100%);
     border-radius:22px;padding:2rem;min-height:260px;
     border:1px solid rgba(52,211,153,.18);
     box-shadow:0 16px 48px rgba(2,44,34,.35);
     position:relative;overflow:hidden">
  <div style="position:absolute;top:-50px;right:-50px;width:180px;height:180px;
       border-radius:50%;background:rgba(52,211,153,.07)"></div>
  <div style="position:absolute;bottom:-70px;left:-40px;width:220px;height:220px;
       border-radius:50%;background:rgba(245,158,11,.04)"></div>
  <div style="position:relative">
    <div style="display:inline-flex;align-items:center;gap:6px;
         background:rgba(52,211,153,.14);border:1px solid rgba(52,211,153,.22);
         padding:4px 12px;border-radius:100px;margin-bottom:1.2rem">
      <div style="width:5px;height:5px;border-radius:50%;background:#34d399"></div>
      <span style="font-size:.62rem;font-weight:700;color:#6ee7b7;letter-spacing:.06em">
        LIVE NOW
      </span>
    </div>
    <div style="font-size:2.2rem;margin-bottom:.8rem">&#x1F916;</div>
    <div style="font-family:'Manrope',sans-serif;font-size:1.05rem;font-weight:800;
                color:#fff;margin-bottom:.55rem;line-height:1.3">
      Smart AI Loan<br>Eligibility Check
    </div>
    <div style="font-size:.78rem;color:rgba(255,255,255,.6);line-height:1.72">
      Instant AI decision with full plain-English explanation
      and minimum changes needed for approval.
    </div>
  </div>
</div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Check Eligibility \u2192", type="primary",
                     use_container_width=True, key="loan_btn"):
            st.session_state.landing = False
            st.rerun()

    def scard(icon, title, desc):
        return f"""
<div style="background:rgba(255,255,255,.85);backdrop-filter:blur(20px);
     border:1px solid rgba(209,250,229,.8);border-radius:22px;
     padding:2rem;min-height:260px;position:relative;overflow:hidden;
     box-shadow:0 4px 24px rgba(6,78,59,.06)">
  <div style="position:absolute;top:0;right:0;width:80px;height:80px;
       border-radius:0 22px 0 80px;
       background:linear-gradient(135deg,rgba(5,150,105,.06),transparent)"></div>
  <div style="background:#ecfdf5;color:#059669;font-size:.62rem;font-weight:700;
       padding:4px 12px;border-radius:100px;display:inline-block;
       margin-bottom:1.2rem;border:1px solid #a7f3d0;letter-spacing:.05em">
    COMING SOON
  </div>
  <div style="font-size:2.2rem;margin-bottom:.8rem">{icon}</div>
  <div style="font-family:'Manrope',sans-serif;font-size:1.05rem;font-weight:700;
              color:#022c22;margin-bottom:.5rem">{title}</div>
  <div style="font-size:.78rem;color:#6b7280;line-height:1.72">{desc}</div>
</div>"""

    with sc2:
        st.markdown(scard("&#x1F3E0;","Home Loan",
            "Property valuation, EMI calculator, eligibility check, and documentation assistance."),
            unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.button("Enquire \u2192", use_container_width=True, disabled=True, key="home_btn")

    with sc3:
        st.markdown(scard("&#x1F697;","Vehicle Loan",
            "Two-wheeler, car, and commercial vehicle loans with instant pre-approval."),
            unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.button("Enquire \u2192", use_container_width=True, disabled=True, key="veh_btn")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    sc4, sc5, sc6 = st.columns(3, gap="medium")
    with sc4:
        st.markdown(scard("&#x1F4B3;","Credit Cards",
            "Zero-fee cards with 5% cashback, travel miles, and purchase protection."),
            unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.button("Apply \u2192", use_container_width=True, disabled=True, key="card_btn")
    with sc5:
        st.markdown(scard("&#x1F4C8;","Investments & FD",
            "Fixed deposits, mutual funds, SIPs and equity portfolios for every risk profile."),
            unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.button("Invest \u2192", use_container_width=True, disabled=True, key="inv_btn")
    with sc6:
        st.markdown(scard("&#x1F6E1;","Insurance",
            "Life, health and vehicle insurance. Compare plans and get covered in minutes."),
            unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.button("Insure \u2192", use_container_width=True, disabled=True, key="ins_btn")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    sc7, sc8, sc9 = st.columns(3, gap="medium")
    with sc7:
        st.markdown(scard("&#x1F4CA;","Financial Health Score",
            "Credit score, savings rate, debt-to-income ratio, and personalised tips."),
            unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.button("Check Score \u2192", use_container_width=True, disabled=True, key="fhs_btn")
    with sc8:
        st.markdown(scard("&#x1F393;","Education Loan",
            "Collateral-free university loans for India and abroad with moratorium support."),
            unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.button("Apply \u2192", use_container_width=True, disabled=True, key="edu_btn")
    with sc9:
        st.markdown(scard("&#x1F3ED;","Business Loan",
            "Working capital, term loans and MSME finance. Fast disbursal, minimal paperwork."),
            unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.button("Apply \u2192", use_container_width=True, disabled=True, key="biz_btn")

    # ── TRUST BANNER ──────────────────────────────────────────
    st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)
    st.markdown("""
<div style="background:linear-gradient(160deg,#022c22 0%,#064e3b 50%,#022c22 100%);
     border-radius:24px;padding:3rem 3.5rem;
     border:1px solid rgba(52,211,153,.12);
     box-shadow:0 24px 72px rgba(2,44,34,.35);
     position:relative;overflow:hidden">
  <div style="position:absolute;top:0;left:50%;transform:translateX(-50%);
       width:220px;height:2px;
       background:linear-gradient(90deg,transparent,#f59e0b,transparent)"></div>
  <div style="position:absolute;top:-100px;right:-80px;width:380px;height:380px;
       border-radius:50%;opacity:.4;filter:blur(70px);
       background:radial-gradient(circle,rgba(16,185,129,.6),transparent 65%)"></div>
  <div style="position:absolute;bottom:-80px;left:-60px;width:300px;height:300px;
       border-radius:50%;opacity:.28;filter:blur(60px);
       background:radial-gradient(circle,rgba(245,158,11,.6),transparent 65%)"></div>
  <div style="position:relative;display:flex;align-items:center;
       justify-content:space-between;flex-wrap:wrap;gap:3rem">
    <div style="max-width:380px">
      <div style="font-size:.66rem;font-weight:700;color:#6ee7b7;
           letter-spacing:.1em;text-transform:uppercase;margin-bottom:.7rem">
        Get Started Today
      </div>
      <div style="font-family:'Manrope',sans-serif;font-size:1.65rem;font-weight:800;
                  color:#fff;letter-spacing:-.5px;line-height:1.25;margin-bottom:.7rem">
        Check your loan eligibility<br>in under 2 seconds.
      </div>
      <p style="font-size:.84rem;color:rgba(255,255,255,.48);line-height:1.7;margin:0">
        No hidden criteria. No waiting. No fees.
        Every rejection includes the exact steps to get approved.
      </p>
    </div>
    <div style="display:flex;gap:0;background:rgba(0,0,0,.2);
         border:1px solid rgba(255,255,255,.07);border-radius:18px;padding:.32rem">
      <div style="padding:1.2rem 1.9rem;text-align:center;
           border-right:1px solid rgba(255,255,255,.07)">
        <div style="font-family:'Manrope',sans-serif;font-size:1.7rem;
                    font-weight:800;color:#34d399;line-height:1">1,000+</div>
        <div style="font-size:.6rem;color:rgba(255,255,255,.36);
                    letter-spacing:.08em;margin-top:5px">APPLICANTS</div>
      </div>
      <div style="padding:1.2rem 1.9rem;text-align:center;
           border-right:1px solid rgba(255,255,255,.07)">
        <div style="font-family:'Manrope',sans-serif;font-size:1.7rem;
                    font-weight:800;color:#fcd34d;line-height:1">92%</div>
        <div style="font-size:.6rem;color:rgba(255,255,255,.36);
                    letter-spacing:.08em;margin-top:5px">FAIRNESS</div>
      </div>
      <div style="padding:1.2rem 1.9rem;text-align:center;
           border-right:1px solid rgba(255,255,255,.07)">
        <div style="font-family:'Manrope',sans-serif;font-size:1.7rem;
                    font-weight:800;color:#fff;line-height:1">100%</div>
        <div style="font-size:.6rem;color:rgba(255,255,255,.36);
                    letter-spacing:.08em;margin-top:5px">EXPLAINED</div>
      </div>
      <div style="padding:1.2rem 1.9rem;text-align:center">
        <div style="font-family:'Manrope',sans-serif;font-size:1.7rem;
                    font-weight:800;color:#a78bfa;line-height:1">Rs.0</div>
        <div style="font-size:.6rem;color:rgba(255,255,255,.36);
                    letter-spacing:.08em;margin-top:5px">FEES</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── FOOTER ────────────────────────────────────────────────
    st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)
    st.markdown("""
<div style="border-top:1px solid #d1fae5;padding:2.2rem 0 1rem">
  <div style="display:flex;justify-content:space-between;
       flex-wrap:wrap;gap:2.5rem;margin-bottom:2rem">
    <div style="max-width:240px">
      <div style="display:flex;align-items:center;gap:9px;margin-bottom:.8rem">
        <div style="background:linear-gradient(135deg,#059669,#047857);
             width:32px;height:32px;border-radius:10px;
             display:flex;align-items:center;justify-content:center;
             font-size:.95rem;box-shadow:0 4px 12px rgba(5,150,105,.3)">&#x1F3E6;</div>
        <span style="font-family:'Manrope',sans-serif;font-size:1.05rem;
                     font-weight:800;color:#022c22">NexaBank</span>
      </div>
      <div style="font-size:.75rem;color:#9ca3af;line-height:1.8">
        AI-powered, fairness-certified financial services for modern India.
      </div>
      <div style="margin-top:.9rem;display:flex;gap:.5rem">
        <div style="width:30px;height:30px;border-radius:8px;
             background:#f0fdf4;border:1px solid #d1fae5;
             display:flex;align-items:center;justify-content:center;font-size:.78rem">
          &#x1F4E7;
        </div>
        <div style="width:30px;height:30px;border-radius:8px;
             background:#f0fdf4;border:1px solid #d1fae5;
             display:flex;align-items:center;justify-content:center;font-size:.78rem">
          &#x1F4F1;
        </div>
        <div style="width:30px;height:30px;border-radius:8px;
             background:#f0fdf4;border:1px solid #d1fae5;
             display:flex;align-items:center;justify-content:center;font-size:.78rem">
          &#x1F517;
        </div>
      </div>
    </div>
    <div style="display:flex;gap:4rem;flex-wrap:wrap">
      <div>
        <div style="font-size:.63rem;font-weight:700;color:#064e3b;
                    margin-bottom:.8rem;letter-spacing:.09em;text-transform:uppercase">
          Products
        </div>
        <div style="font-size:.78rem;color:#6b7280;line-height:2.3">
          AI Loan Eligibility<br>Home Loan<br>Credit Cards<br>Investments<br>Insurance
        </div>
      </div>
      <div>
        <div style="font-size:.63rem;font-weight:700;color:#064e3b;
                    margin-bottom:.8rem;letter-spacing:.09em;text-transform:uppercase">
          Company
        </div>
        <div style="font-size:.78rem;color:#6b7280;line-height:2.3">
          About Us<br>Careers<br>Press Room<br>Blog<br>Partners
        </div>
      </div>
      <div>
        <div style="font-size:.63rem;font-weight:700;color:#064e3b;
                    margin-bottom:.8rem;letter-spacing:.09em;text-transform:uppercase">
          Legal
        </div>
        <div style="font-size:.78rem;color:#6b7280;line-height:2.3">
          Privacy Policy<br>Terms of Service<br>Cookie Policy<br>Fair Lending<br>Compliance
        </div>
      </div>
      <div>
        <div style="font-size:.63rem;font-weight:700;color:#064e3b;
                    margin-bottom:.8rem;letter-spacing:.09em;text-transform:uppercase">
          Support
        </div>
        <div style="font-size:.78rem;color:#6b7280;line-height:2.3">
          Help Centre<br>Contact Us<br>Grievance Cell<br>Branch Locator<br>24/7 Chat
        </div>
      </div>
    </div>
  </div>
  <div style="border-top:1px solid rgba(209,250,229,0.5);padding-top:1.2rem;
       display:flex;justify-content:space-between;align-items:center;
       flex-wrap:wrap;gap:.8rem">
    <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap">
      <span style="font-size:.7rem;color:#9ca3af">
        2026 NexaBank Financial Services Pvt. Ltd.
      </span>
      <span style="background:#ecfdf5;color:#059669;font-size:.63rem;font-weight:700;
           padding:2px 10px;border-radius:100px;border:1px solid #a7f3d0">
        RBI Regulated
      </span>
      <span style="background:#fefce8;color:#92400e;font-size:.63rem;font-weight:700;
           padding:2px 10px;border-radius:100px;border:1px solid #fde68a">
        AI Certified
      </span>
    </div>
    <div style="font-size:.7rem;color:#d1d5db">
      Demo &middot; Not a real bank &middot; Educational only
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.stop()


# ══════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════

# ── Top bar ───────────────────────────────────────────────────
top_l, top_r = st.columns([2.4, 1.6], gap="medium")

with top_l:
    st.markdown("""
    <div style='background:rgba(255,255,255,0.85);backdrop-filter:blur(20px);
         border-radius:18px;padding:1rem 1.6rem;border:1px solid rgba(255,255,255,0.9);
         box-shadow:0 2px 16px rgba(0,0,0,.06);
         display:flex;align-items:center;gap:14px'>
      <div style='background:linear-gradient(135deg,#059669,#047857);
           width:42px;height:42px;border-radius:13px;display:flex;
           align-items:center;justify-content:center;font-size:1.2rem;
           flex-shrink:0;box-shadow:0 4px 14px rgba(5,150,105,.4)'>🤖</div>
      <div>
        <div style='font-size:1.05rem;font-weight:800;color:#111827;
                    letter-spacing:-.3px'>Smart AI Loan Eligibility Check</div>
        <div style='font-size:.73rem;color:#6b7280;margin-top:1px'>
          NexaBank · Fairness-Certified · Explainable AI
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

with top_r:
    st.markdown("""
    <div style='background:rgba(255,255,255,0.85);backdrop-filter:blur(20px);
         border-radius:18px;padding:.8rem 1.2rem;border:1px solid rgba(255,255,255,0.9);
         box-shadow:0 2px 16px rgba(0,0,0,.06)'>
      <p style='font-size:.62rem;font-weight:700;letter-spacing:.09em;
                color:#9ca3af;text-transform:uppercase;margin:0 0 8px'>
        Navigate
      </p>""", unsafe_allow_html=True)

    nb1,nb2,nb3,nb4 = st.columns(4, gap="small")
    with nb1:
        if st.button("🏦\nHome", use_container_width=True):
            st.session_state.landing = True
            st.session_state.show_result = False
            st.rerun()
    with nb2:
        if st.button("🔍\nPredict",
                     type="primary" if st.session_state.page=="predict" else "secondary",
                     use_container_width=True):
            st.session_state.page="predict"
            st.session_state.show_result=False
            st.rerun()
    with nb3:
        if st.button("⚖️\nFairness",
                     type="primary" if st.session_state.page=="fairness" else "secondary",
                     use_container_width=True):
            st.session_state.page="fairness"; st.rerun()
    with nb4:
        if st.button("📊\nModel",
                     type="primary" if st.session_state.page=="perf" else "secondary",
                     use_container_width=True):
            st.session_state.page="perf"; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr style='margin:.6rem 0 1rem'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 1 — PREDICT & EXPLAIN
# ══════════════════════════════════════════════════════════════
if st.session_state.page == "predict":

    if not st.session_state.show_result:

        _, form_col, _ = st.columns([0.4, 3.2, 0.4])
        with form_col:

            # Form header card
            st.markdown("""
            <div style='background:linear-gradient(135deg,#059669 0%,#047857 100%);
                 border-radius:20px;padding:1.8rem 2.2rem;margin-bottom:1.2rem;
                 position:relative;overflow:hidden;
                 box-shadow:0 8px 32px rgba(5,150,105,.35)'>
              <div style='position:absolute;top:-50px;right:-50px;width:200px;height:200px;
                   border-radius:50%;background:rgba(255,255,255,.07)'></div>
              <div style='position:absolute;bottom:-60px;left:-40px;width:180px;height:180px;
                   border-radius:50%;background:rgba(255,255,255,.04)'></div>
              <div style='position:relative'>
                <div style='font-size:1.15rem;font-weight:800;color:#fff;margin:0 0 4px'>
                  📋 Loan Application Form
                </div>
                <div style='font-size:.82rem;color:rgba(255,255,255,.65)'>
                  Fill in your details below — get an instant AI-powered decision
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Section label helper
            def sec_label(emoji, title):
                st.markdown(f"""
                <div style='display:flex;align-items:center;gap:8px;margin:16px 0 8px'>
                  <div style='width:28px;height:28px;border-radius:8px;
                       background:linear-gradient(135deg,#ecfdf5,#d1fae5);
                       display:flex;align-items:center;justify-content:center;
                       font-size:.85rem'>{emoji}</div>
                  <span style='font-size:.72rem;font-weight:700;letter-spacing:.08em;
                               color:#10b981;text-transform:uppercase'>{title}</span>
                </div>""", unsafe_allow_html=True)

            # Pre-fill from the last submission (if any) so that
            # clicking "New Application" to go back and tweak values
            # doesn't wipe out everything the user already entered.
            prev = st.session_state.get("form_data", {})

            sec_label("👤","Personal Information")
            pa, pb, pc = st.columns(3, gap="medium")
            with pa: age = st.number_input("Age (years)", 18, 80, prev.get("age", 30), 1, key="fld_age")
            with pb:
                sex_options = ["Male","Female","Prefer not to say"]
                sex = st.selectbox("Gender", sex_options,
                        index=sex_options.index(prev["sex"]) if prev.get("sex") in sex_options else 0,
                        key="fld_sex")
            with pc:
                job_options = list(JOB_MAP.keys())
                job_lbl = st.selectbox("Employment Type", job_options,
                        index=job_options.index(prev["job_lbl"]) if prev.get("job_lbl") in job_options else 0,
                        key="fld_job")

            pd2, pe = st.columns(2, gap="medium")
            with pd2:
                house_options = list(HOUSING_MAP.keys())
                house_lbl = st.selectbox("Housing Situation", house_options,
                        index=house_options.index(prev["house_lbl"]) if prev.get("house_lbl") in house_options else 0,
                        key="fld_house")
            with pe:
                purpose_options = list(PURPOSE_MAP.keys())
                purpose_lbl = st.selectbox("Purpose of Loan", purpose_options,
                        index=purpose_options.index(prev["purpose_lbl"]) if prev.get("purpose_lbl") in purpose_options else 0,
                        key="fld_purpose")

            sec_label("💰","Financial Status")
            st.markdown("""
            <div style='background:rgba(5,150,105,.07);border-radius:12px;
                 padding:10px 16px;border:1px solid rgba(5,150,105,.15);
                 margin-bottom:10px'>
              <span style='font-size:.76rem;color:#059669;line-height:1.6'>
                <strong>Savings Account</strong> — money kept long-term for security &nbsp;·&nbsp;
                <strong>Current Account</strong> — salary / day-to-day transactions
              </span>
            </div>""", unsafe_allow_html=True)

            fa, fb = st.columns(2, gap="medium")
            with fa:
                saving_options = list(SAVING_MAP.keys())
                saving_lbl  = st.selectbox("Savings Account Balance",  saving_options,
                        index=saving_options.index(prev["saving_lbl"]) if prev.get("saving_lbl") in saving_options else 0,
                        key="fld_saving")
            with fb:
                current_options = list(CURRENT_MAP.keys())
                current_lbl = st.selectbox("Current Account Balance",   current_options,
                        index=current_options.index(prev["current_lbl"]) if prev.get("current_lbl") in current_options else 0,
                        key="fld_current")

            sec_label("📋","Loan Details")
            la, lb = st.columns(2, gap="medium")
            with la:
                credit_inr = st.number_input("Loan Amount Required (₹ INR)",
                    3000, 3000000, prev.get("credit_inr", 150000), 5000,
                    help="Enter in Indian Rupees · 1 Deutsche Mark ≈ ₹30",
                    key="fld_credit")
            with lb:
                duration_options = [6,12,18,24,30,36,48,60,72,84,96,108,120,144,168,192,216,240]
                duration = st.selectbox("Loan Duration",
                    duration_options,
                    index=duration_options.index(prev["duration"]) if prev.get("duration") in duration_options else 3,
                    format_func=lambda x: (
                        f"{x} months  ({x//12} yr{'s' if x//12>1 else ''})"
                        if x>=12 else f"{x} months"),
                    key="fld_duration")

            # Live warnings
            no_sav  = SAVING_MAP[saving_lbl]  == "no_savings"
            no_cur  = CURRENT_MAP[current_lbl] == "no_current"
            low_sav = SAVING_MAP[saving_lbl]  == "little"
            low_cur = CURRENT_MAP[current_lbl] == "little"

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            if no_sav and no_cur:
                st.error("🚫 **No savings & no current account** — This will be automatically rejected. Banks require at least one active financial account.")
            elif no_sav and credit_inr > 300000:
                st.error(f"🚫 **No savings + high loan amount (₹{credit_inr:,})** — Will be rejected. Savings act as collateral for large loans.")
            elif no_sav:
                st.warning("⚠️ No savings account — approval chances are significantly reduced.")
            elif no_cur:
                st.warning("⚠️ No current account — banks need proof of regular income flow.")
            elif low_sav and low_cur and credit_inr > 500000:
                st.warning(f"⚠️ Low balances + high loan (₹{credit_inr:,}) — this combination is considered high risk.")

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            if st.button("🔍  Get My Loan Decision", type="primary", use_container_width=True):
                st.session_state.form_data = {
                    "age":age,"sex":sex,"job_lbl":job_lbl,
                    "house_lbl":house_lbl,"purpose_lbl":purpose_lbl,
                    "saving_lbl":saving_lbl,"current_lbl":current_lbl,
                    "credit_inr":credit_inr,"duration":duration,
                }
                st.session_state.show_result = True
                st.rerun()

    else:
        # ── RESULTS ──────────────────────────────────────────────
        d = st.session_state.form_data
        age=d["age"]; sex=d["sex"]; job_lbl=d["job_lbl"]
        house_lbl=d["house_lbl"]; purpose_lbl=d["purpose_lbl"]
        saving_lbl=d["saving_lbl"]; current_lbl=d["current_lbl"]
        credit_inr=d["credit_inr"]; duration=d["duration"]
        sex_enc = "male" if sex=="Male" else "female"
        no_sav  = SAVING_MAP[saving_lbl]  == "no_savings"
        no_cur  = CURRENT_MAP[current_lbl] == "no_current"
        low_sav = SAVING_MAP[saving_lbl]  == "little"
        low_cur = CURRENT_MAP[current_lbl] == "little"

        credit_dm = max(100, min(20000, credit_inr // 30))
        sav_enc   = "unknown" if no_sav  else SAVING_MAP[saving_lbl]
        cur_enc   = "unknown" if no_cur  else CURRENT_MAP[current_lbl]

        encoded = {
            "Age":              age,
            "Sex":              int(enc["Sex"].transform([sex_enc])[0]),
            "Job":              JOB_MAP[job_lbl],
            "Housing":          int(enc["Housing"].transform([HOUSING_MAP[house_lbl]])[0]),
            "Saving accounts":  int(enc["Saving accounts"].transform([sav_enc])[0]),
            "Checking account": int(enc["Checking account"].transform([cur_enc])[0]),
            "Credit amount":    credit_dm,
            "Duration":         duration,
            "Purpose":          int(enc["Purpose"].transform([PURPOSE_MAP[purpose_lbl]])[0]),
        }
        inp_df = pd.DataFrame([encoded])
        pred   = pipe.predict(inp_df)[0]
        prob   = pipe.predict_proba(inp_df)[0][1]

        reason_override = None
        if no_sav and no_cur:
            pred=0; prob=min(prob,.18)
            reason_override="No savings and no current account on record."
        elif no_sav and credit_inr>300000:
            pred=0; prob=min(prob,.25)
            reason_override=f"No savings account with a high loan of ₹{credit_inr:,}."
        elif no_cur and JOB_MAP[job_lbl]==0:
            pred=0; prob=min(prob,.20)
            reason_override="No current account and unemployed status."
        elif low_sav and low_cur and credit_inr>500000:
            pred=0; prob=min(prob,.30)
            reason_override=f"Low account balances with a high loan of ₹{credit_inr:,}."

        bc,_ = st.columns([1,5])
        with bc:
            if st.button("← New Application"):
                st.session_state.show_result=False; st.rerun()

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        col_v, col_e, col_p = st.columns([1, 1.6, 1.05], gap="medium")

        # ── VERDICT CARD ──────────────────────────────────────
        with col_v:
            if pred==1:
                gradient="linear-gradient(135deg,#052e1c,#064e2e)"
                border="#22c55e"; tc="#4ade80"; ico="✅"; lbl="APPROVED"
                glow="rgba(34,197,94,.2)"
            else:
                gradient="linear-gradient(135deg,#2d0a0a,#3d1010)"
                border="#f87171"; tc="#f87171"; ico="❌"; lbl="REJECTED"
                glow="rgba(248,113,113,.2)"

            st.markdown(f"""
            <div style='background:{gradient};border:1.5px solid {border};
                 border-radius:22px;padding:2rem 1.5rem;text-align:center;
                 box-shadow:0 8px 40px {glow}, 0 2px 8px rgba(0,0,0,.15);
                 position:relative;overflow:hidden'>
              <div style='position:absolute;top:-40px;right:-40px;width:120px;height:120px;
                   border-radius:50%;background:{glow};filter:blur(20px)'></div>
              <div style='font-size:3.8rem;line-height:1;margin-bottom:.6rem'>{ico}</div>
              <div style='font-size:2rem;font-weight:900;color:{tc};
                          letter-spacing:-1px;text-shadow:0 0 20px {glow}'>{lbl}</div>
              <div style='font-size:.8rem;color:rgba(255,255,255,.5);margin-top:8px'>
                Approval probability
              </div>
              <div style='font-size:1.7rem;font-weight:800;color:{tc};margin-top:2px'>
                {prob:.1%}
              </div>
              <div style='height:7px;background:rgba(255,255,255,.1);border-radius:100px;
                          margin:14px 0 0;overflow:hidden'>
                <div style='height:100%;width:{prob*100:.1f}%;
                            background:linear-gradient(90deg,{border},{tc});
                            border-radius:100px'></div>
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            # Profile card
            st.markdown("""
            <div style='background:rgba(255,255,255,0.85);backdrop-filter:blur(20px);
                 border-radius:18px;border:1px solid rgba(255,255,255,0.9);
                 padding:1.1rem 1.3rem;box-shadow:0 2px 12px rgba(0,0,0,.06)'>
              <div style='font-size:.62rem;font-weight:700;letter-spacing:.09em;
                   color:#9ca3af;text-transform:uppercase;margin-bottom:10px'>
                Application Summary
              </div>""", unsafe_allow_html=True)

            for k,v in [
                ("Age",       f"{age} yrs"),
                ("Gender",    sex),
                ("Job",       job_lbl.split("–")[-1].strip() if "–" in job_lbl else job_lbl.split("/")[0]),
                ("Housing",   house_lbl.replace("(Mortgage-free)","").replace("(With Mortgage)","").strip()),
                ("Savings",   saving_lbl.split("(")[0].strip()),
                ("Current",   current_lbl.split("(")[0].strip()),
                ("Amount",    f"₹{credit_inr:,}"),
                ("Duration",  f"{duration} mo"),
                ("Purpose",   purpose_lbl.split("/")[0]),
            ]:
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;
                     padding:5px 0;border-bottom:1px solid rgba(243,244,246,0.8)'>
                  <span style='font-size:.74rem;color:#9ca3af;font-weight:500'>{k}</span>
                  <span style='font-size:.74rem;color:#111827;font-weight:700;
                               text-align:right;max-width:58%'>{v}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── EXPLANATION ───────────────────────────────────────
        with col_e:
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.85);backdrop-filter:blur(20px);
                 border-radius:18px;border:1px solid rgba(255,255,255,0.9);
                 padding:1.5rem 1.8rem;box-shadow:0 2px 12px rgba(0,0,0,.06);
                 margin-bottom:10px'>
              <p style='font-size:1rem;font-weight:800;color:#111827;margin:0 0 4px;
                        letter-spacing:-.2px'>
                {'💡 Why was this rejected?' if pred==0 else '✅ Why was this approved?'}
              </p>
              <p style='font-size:.82rem;color:#6b7280;margin:0 0 0;line-height:1.6'>
                {"Our AI found the <strong style='color:#059669'>minimum changes</strong> needed to flip this to <strong style='color:#16a34a'>Approved</strong>."
                 if pred==0 else "This applicant's profile meets the model's approval criteria."}
              </p>
            </div>""", unsafe_allow_html=True)

            if pred == 0:
                # Build reasons
                reasons=[]
                if no_sav:
                    reasons.append({"icon":"🏦","sev":"high","title":"No Savings Account",
                        "detail":"You have no savings account. Banks require savings as security — proof that you can manage money and have a backup fund for missed payments."})
                elif low_sav:
                    reasons.append({"icon":"📉","sev":"medium","title":"Very Low Savings (< ₹3,000)",
                        "detail":"Your savings balance is critically low, providing almost no security for the bank against loan default."})
                if no_cur:
                    reasons.append({"icon":"💳","sev":"high","title":"No Current / Salary Account",
                        "detail":"No current or salary account means the bank cannot verify regular income and repayment capacity."})
                elif low_cur:
                    reasons.append({"icon":"📊","sev":"medium","title":"Low Current Account Balance",
                        "detail":"Very low current account balance signals limited cash flow and raises concern about consistent monthly repayments."})
                if JOB_MAP[job_lbl]==0:
                    reasons.append({"icon":"💼","sev":"high","title":"No Stable Income (Unemployed)",
                        "detail":"Without a confirmed income source, the bank has no assurance that repayments will be made on time."})
                if credit_inr>500000 and JOB_MAP[job_lbl]<=1:
                    reasons.append({"icon":"💰","sev":"high","title":f"Loan Too High for Income Level (₹{credit_inr:,})",
                        "detail":f"₹{credit_inr:,} is very high relative to your employment level. Monthly instalments may exceed your repayment capacity."})
                elif credit_inr>300000 and (no_sav or low_sav):
                    reasons.append({"icon":"⚖️","sev":"medium","title":f"Loan Large vs Savings (₹{credit_inr:,})",
                        "detail":f"Requesting ₹{credit_inr:,} without adequate savings is risky. Savings should cover at least 3 months of repayments."})
                if duration>=48:
                    reasons.append({"icon":"📅","sev":"low","title":f"Long Repayment Period ({duration} months)",
                        "detail":f"A {duration}-month loan has higher default probability. Banks prefer shorter repayment durations."})
                if HOUSING_MAP[house_lbl]=="rent" and (no_sav or low_sav):
                    reasons.append({"icon":"🏠","sev":"medium","title":"Renting + Insufficient Savings",
                        "detail":"Paying rent each month with no savings buffer leaves very little margin for loan repayments."})
                if not reasons:
                    reasons.append({"icon":"🤖","sev":"medium","title":"Overall Risk Score Too High",
                        "detail":"The AI model determined the combined risk across income, loan size, account activity, and duration exceeds acceptable limits."})

                sev_style = {
                    "high":   ("linear-gradient(135deg,#fff1f2,#ffe4e6)","rgba(248,113,113,.3)","#dc2626","High Risk"),
                    "medium": ("linear-gradient(135deg,#fffbeb,#fef3c7)","rgba(251,191,36,.3)","#d97706","Medium Risk"),
                    "low":    ("linear-gradient(135deg,#f0f9ff,#e0f2fe)","rgba(14,165,233,.3)","#0369a1","Low Risk"),
                }

                st.markdown(f"""
                <div style='font-size:.72rem;font-weight:700;color:#dc2626;
                     letter-spacing:.05em;text-transform:uppercase;
                     margin:6px 0 10px'>
                  🚨 {len(reasons)} Reason{"s" if len(reasons)>1 else ""} for Rejection
                </div>""", unsafe_allow_html=True)

                for r in reasons:
                    bg,bsh,tc2,badge = sev_style[r["sev"]]
                    st.markdown(f"""
                    <div style='background:{bg};border-radius:14px;
                         border:1px solid {bsh};
                         border-left:4px solid {tc2};
                         padding:13px 16px;margin-bottom:9px;
                         box-shadow:0 2px 8px rgba(0,0,0,.04)'>
                      <div style='display:flex;align-items:center;
                           justify-content:space-between;margin-bottom:5px'>
                        <div style='font-size:.9rem;font-weight:700;color:#111827'>
                          {r["icon"]} {r["title"]}
                        </div>
                        <span style='font-size:.63rem;font-weight:700;
                             padding:3px 10px;border-radius:100px;
                             background:{tc2};color:#fff;
                             white-space:nowrap;margin-left:8px'>{badge}</span>
                      </div>
                      <div style='font-size:.8rem;color:#374151;line-height:1.65'>
                        {r["detail"]}
                      </div>
                    </div>""", unsafe_allow_html=True)

                # Counterfactuals
                st.markdown("""
                <div style='font-size:.68rem;font-weight:700;color:#059669;
                     letter-spacing:.07em;text-transform:uppercase;
                     margin:14px 0 8px'>
                  ✅ Minimum Changes to Get Approved
                </div>""", unsafe_allow_html=True)

                with st.spinner("Finding minimum changes…"):
                    suggestions = explain_single(encoded, pipe=pipe)

                shown = 0
                for raw in suggestions:
                    clean = re.sub(r'\*\*(.+?)\*\*', r'\1', raw)
                    parts = clean.split(" from ")
                    if len(parts)!=2: continue
                    action,change_part = parts[0].strip(),parts[1].strip()
                    change_vals = change_part.split(" to ")
                    if len(change_vals)==2 and change_vals[1].strip().lower() in SKIP_VALUES:
                        continue
                    direction = "⬇️" if "decrease" in action.lower() else "⬆️"
                    feat_raw  = action.split("'")[1] if "'" in action else " ".join(action.split()[1:])
                    feat_lbl  = FEAT_LABELS.get(feat_raw, feat_raw)
                    if len(change_vals)==2:
                        try:
                            fi,ti = int(round(float(change_vals[0]))),int(round(float(change_vals[1])))
                            vm = FEAT_VALUE_MAP.get(feat_raw,{})
                            fs=vm.get(fi,change_vals[0]); ts=vm.get(ti,change_vals[1])
                            if feat_raw=="Credit amount": fs=f"₹{fi*30:,}"; ts=f"₹{ti*30:,}"
                            elif feat_raw=="Duration":    fs=f"{fi} months"; ts=f"{ti} months"
                            chg = f"{fs}  →  {ts}"
                        except: chg = change_part
                    else: chg = change_part

                    st.markdown(f"""
                    <div style='background:rgba(240,249,255,0.9);
                         backdrop-filter:blur(10px);
                         border-radius:12px;
                         border:1px solid rgba(186,230,253,0.8);
                         border-left:4px solid #0ea5e9;
                         padding:11px 15px;margin-bottom:7px;
                         display:flex;align-items:center;gap:12px;
                         box-shadow:0 2px 8px rgba(14,165,233,.08)'>
                      <div style='font-size:1.4rem;flex-shrink:0'>{direction}</div>
                      <div>
                        <div style='font-size:.86rem;font-weight:700;color:#0369a1'>{feat_lbl}</div>
                        <div style='font-size:.76rem;color:#6b7280;margin-top:1px'>{chg}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                    shown += 1
                    if shown >= 3: break

                if shown == 0:
                    # Build fallback advice from the applicant's ACTUAL
                    # profile instead of always showing the same static
                    # "open/top up savings & current account" text —
                    # that was firing even when savings/current were
                    # already High, which is self-contradictory.
                    sav_status = SAVING_MAP[saving_lbl]
                    cur_status = CURRENT_MAP[current_lbl]
                    fallback_items = []

                    if sav_status in ("no_savings", "little"):
                        fallback_items.append(
                            ("⬆️","Open or Top Up Savings Account",
                             "Build ₹3,000–₹15,000 to show financial security"))
                    if cur_status in ("no_current", "little"):
                        fallback_items.append(
                            ("⬆️","Open or Top Up Current Account",
                             "A salary account with ₹6,000+ proves regular income"))
                    if credit_inr > 150000:
                        fallback_items.append(
                            ("⬇️","Reduce the Loan Amount",
                             "Request 30–50% less — smaller loans are easier to approve"))
                    if duration > 24:
                        fallback_items.append(
                            ("⬇️","Shorten the Loan Duration",
                             "A shorter repayment period lowers perceived risk"))

                    if not fallback_items:
                        # Savings/current are already strong and the
                        # loan terms are reasonable — be honest that
                        # there's no small, actionable change to give.
                        st.markdown("""
                        <div style='background:rgba(255,247,237,0.9);border-radius:12px;
                             border:1px solid rgba(254,215,170,0.8);
                             border-left:4px solid #f59e0b;
                             padding:13px 16px;margin-bottom:9px'>
                          <div style='font-size:.86rem;font-weight:700;color:#92400e'>
                            No small change found
                          </div>
                          <div style='font-size:.78rem;color:#6b7280;margin-top:3px;line-height:1.6'>
                            Your savings, current account, and loan terms already look strong.
                            This application is borderline, and our AI couldn't identify a
                            single meaningful adjustment that would flip the decision —
                            it may come down to factors like age, employment type, or loan purpose
                            that aren't easily changed.
                          </div>
                        </div>""", unsafe_allow_html=True)

                    for arr,feat,desc in fallback_items:
                        st.markdown(f"""
                        <div style='background:rgba(240,249,255,0.9);border-radius:12px;
                             border:1px solid rgba(186,230,253,0.8);
                             border-left:4px solid #0ea5e9;
                             padding:11px 15px;margin-bottom:7px;
                             display:flex;align-items:center;gap:12px'>
                          <div style='font-size:1.4rem;flex-shrink:0'>{arr}</div>
                          <div>
                            <div style='font-size:.86rem;font-weight:700;color:#0369a1'>{feat}</div>
                            <div style='font-size:.76rem;color:#6b7280;margin-top:1px'>{desc}</div>
                          </div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("""
                <div style='background:rgba(249,250,251,0.8);border-radius:10px;
                     padding:9px 14px;margin-top:6px;
                     border:1px solid rgba(229,231,235,0.6);
                     font-size:.7rem;color:#9ca3af;line-height:1.55'>
                  ℹ️ AI-identified minimum changes that would flip this to Approved.
                  Actual bank decisions may consider additional factors.
                </div>""", unsafe_allow_html=True)

            else:
                # Approved
                st.markdown("""
                <div style='background:linear-gradient(135deg,rgba(240,253,244,0.9),rgba(220,252,231,0.7));
                     backdrop-filter:blur(20px);border-radius:16px;
                     border:1px solid rgba(187,247,208,0.8);
                     border-left:4px solid #22c55e;
                     padding:1.5rem 1.8rem;box-shadow:0 2px 12px rgba(34,197,94,.08)'>
                  <div style='font-size:.68rem;font-weight:700;color:#16a34a;
                       letter-spacing:.07em;text-transform:uppercase;margin-bottom:12px'>
                    ✅ Reasons for Approval
                  </div>""", unsafe_allow_html=True)

                strengths=[]
                if SAVING_MAP[saving_lbl] in ["quite rich","rich"]:
                    strengths.append(("💰","Strong Savings Balance","Financial security and backup funds"))
                if CURRENT_MAP[current_lbl] in ["moderate","rich"]:
                    strengths.append(("🏦","Healthy Current Account","Regular income and active banking"))
                if duration<=24:
                    strengths.append(("📅","Short Loan Duration","Lower repayment risk"))
                if credit_inr<=150000:
                    strengths.append(("✓","Manageable Loan Amount","Within safe lending limits"))
                if JOB_MAP[job_lbl]>=2:
                    strengths.append(("💼","Stable Employment","Reduces default risk"))
                if not strengths:
                    strengths=[("✓","Profile Meets All Criteria","All factors within acceptable range")]

                for ico2,title,desc in strengths:
                    st.markdown(f"""
                    <div style='background:rgba(255,255,255,0.7);border-radius:12px;
                         border:1px solid rgba(187,247,208,0.8);
                         border-left:4px solid #22c55e;
                         padding:11px 14px;margin-bottom:8px'>
                      <div style='font-size:.88rem;font-weight:700;color:#15803d'>
                        {ico2} {title}
                      </div>
                      <div style='font-size:.76rem;color:#6b7280;margin-top:2px'>{desc}</div>
                    </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # ── RIGHT COLUMN ──────────────────────────────────────
        with col_p:
            clr = "#22c55e" if pred==1 else "#f87171"
            grd = "135deg,#052e1c,#064e2e" if pred==1 else "135deg,#2d0a0a,#3d1010"

            # Confidence
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.85);backdrop-filter:blur(20px);
                 border-radius:16px;border:1px solid rgba(255,255,255,0.9);
                 padding:1.1rem 1.3rem;margin-bottom:10px;
                 box-shadow:0 2px 12px rgba(0,0,0,.06)'>
              <div style='font-size:.62rem;font-weight:700;letter-spacing:.09em;
                   color:#9ca3af;text-transform:uppercase;margin-bottom:8px'>
                Approval Confidence
              </div>
              <div style='font-size:1.7rem;font-weight:800;color:{clr}'>{prob:.1%}</div>
              <div style='height:8px;background:#f3f4f6;border-radius:100px;
                          margin:8px 0 6px;overflow:hidden'>
                <div style='height:100%;width:{prob*100:.1f}%;
                            background:linear-gradient(90deg,{clr},{clr}aa);
                            border-radius:100px'></div>
              </div>
              <div style='font-size:.71rem;color:#9ca3af'>
                {'Strong signals — high confidence' if prob>.7 else
                 'Borderline application' if prob>.4 else
                 'Significant risk factors present'}
              </div>
            </div>""", unsafe_allow_html=True)

            # EMI on approval
            if pred==1:
                rate=0.12/12; n=duration
                emi   = credit_inr*rate*(1+rate)**n/((1+rate)**n-1) if rate>0 else credit_inr/n
                total = emi*n; interest = total-credit_inr
                st.markdown(f"""
                <div style='background:rgba(255,255,255,0.85);backdrop-filter:blur(20px);
                     border-radius:16px;border:1px solid rgba(255,255,255,0.9);
                     padding:1.1rem 1.3rem;margin-bottom:10px;
                     box-shadow:0 2px 12px rgba(0,0,0,.06)'>
                  <div style='font-size:.62rem;font-weight:700;letter-spacing:.09em;
                       color:#9ca3af;text-transform:uppercase;margin-bottom:8px'>
                    📅 EMI Estimate &nbsp;<span style='font-weight:400'>@ 12% p.a.</span>
                  </div>
                  <div style='font-size:1.5rem;font-weight:800;color:#111827;margin-bottom:10px'>
                    ₹{emi:,.0f}<span style='font-size:.8rem;font-weight:500;color:#6b7280'> / month</span>
                  </div>
                  <div style='height:1px;background:rgba(229,231,235,0.6);margin-bottom:10px'></div>
                  <div style='display:flex;justify-content:space-between;margin-bottom:5px'>
                    <span style='font-size:.74rem;color:#9ca3af'>Principal</span>
                    <span style='font-size:.74rem;color:#111827;font-weight:600'>₹{credit_inr:,}</span>
                  </div>
                  <div style='display:flex;justify-content:space-between;margin-bottom:5px'>
                    <span style='font-size:.74rem;color:#9ca3af'>Interest</span>
                    <span style='font-size:.74rem;color:#dc2626;font-weight:600'>₹{interest:,.0f}</span>
                  </div>
                  <div style='display:flex;justify-content:space-between'>
                    <span style='font-size:.74rem;color:#9ca3af;font-weight:600'>Total Payable</span>
                    <span style='font-size:.74rem;color:#111827;font-weight:700'>₹{total:,.0f}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

            # Feature importance chart
            st.markdown("""
            <div style='background:rgba(255,255,255,0.85);backdrop-filter:blur(20px);
                 border-radius:16px;border:1px solid rgba(255,255,255,0.9);
                 padding:1.1rem 1.3rem;box-shadow:0 2px 12px rgba(0,0,0,.06)'>
              <div style='font-size:.62rem;font-weight:700;letter-spacing:.09em;
                   color:#9ca3af;text-transform:uppercase;margin-bottom:8px'>
                Key Decision Factors
              </div>""", unsafe_allow_html=True)
            try:
                coefs=pipe.named_steps["clf"].coef_[0]
                feats=list(X_te_raw.columns)
                imp=pd.Series(np.abs(coefs),index=feats)\
                      .sort_values(ascending=True).tail(6)
                fig,ax=plt.subplots(figsize=(4,2.6))
                fig.patch.set_facecolor("none")
                ax.set_facecolor("none")
                colors_bar = ["#10b981" if imp.values[i]>=imp.values.mean()
                              else "#a5b4fc" for i in range(len(imp))]
                ax.barh(imp.index, imp.values, color=colors_bar,
                        alpha=.9, edgecolor="none", height=.6)
                ax.set_xlabel("Importance", color="#9ca3af", fontsize=8)
                ax.tick_params(colors="#374151", labelsize=8)
                for sp in ax.spines.values(): sp.set_visible(False)
                ax.xaxis.grid(True, color="rgba(229,231,235,0.5)", zorder=0)
                ax.set_axisbelow(True)
                plt.tight_layout(pad=.3)
                st.pyplot(fig, transparent=True)
                plt.close()
            except: pass
            st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — FAIRNESS REPORT
# ══════════════════════════════════════════════════════════════
elif st.session_state.page=="fairness":
    st.markdown("""
    <div style='background:rgba(255,255,255,0.85);backdrop-filter:blur(20px);
         border-radius:20px;border:1px solid rgba(255,255,255,0.9);
         padding:1.6rem 2rem;margin-bottom:1.4rem;
         box-shadow:0 4px 24px rgba(0,0,0,.06)'>
      <div style='display:flex;align-items:center;gap:12px;margin-bottom:4px'>
        <div style='width:40px;height:40px;border-radius:12px;
             background:linear-gradient(135deg,#ecfdf5,#d1fae5);
             display:flex;align-items:center;justify-content:center;font-size:1.2rem'>⚖️</div>
        <div>
          <h2 style='font-size:1.3rem;font-weight:800;color:#111827;margin:0;
                     letter-spacing:-.3px'>Fairness Report</h2>
          <p style='font-size:.82rem;color:#6b7280;margin:0'>
            Bias measurement using Disparate Impact · DI &lt; 0.80 is legally discriminatory
          </p>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    live=pipe.predict(X_te_raw)
    m_rate=live[s_te.values==1].mean(); f_rate=live[s_te.values==0].mean()
    di=f_rate/m_rate if m_rate>0 else 0; spd=f_rate-m_rate

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Disparate Impact",  f"{di:.3f}",    "ideal = 1.00")
    c2.metric("Male Approval",     f"{m_rate:.1%}", "on test set")
    c3.metric("Female Approval",   f"{f_rate:.1%}", "on test set")
    c4.metric("Parity Difference", f"{abs(spd):.3f}","ideal = 0.00")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if di>=.9:   scol,stxt,sbg="#16a34a","✅  Fair — well above the legal 80% threshold","rgba(240,253,244,0.8)"
    elif di>=.8: scol,stxt,sbg="#d97706","⚠️  Acceptable — just above the 80% threshold","rgba(255,251,235,0.8)"
    else:        scol,stxt,sbg="#dc2626","❌  Biased — below the legal 80% threshold","rgba(255,241,242,0.8)"

    st.markdown(f"""
    <div style='background:{sbg};backdrop-filter:blur(10px);
         border-radius:14px;border:1px solid rgba(229,231,235,0.6);
         border-left:5px solid {scol};padding:12px 20px;margin-bottom:1.5rem;
         box-shadow:0 2px 8px rgba(0,0,0,.04)'>
      <span style='font-size:.92rem;font-weight:700;color:{scol}'>{stxt}</span>
    </div>""", unsafe_allow_html=True)

    if fr:
        fc1,fc2=st.columns(2,gap="large")
        with fc1:
            fig,ax=light_fig(6,4)
            vals=[max(0,fr["di_before"]),max(0,fr["di_after"])]
            bars=ax.bar(["Before\nMitigation","After\nMitigation"],
                        vals,color=["#f87171","#4ade80"],width=.45,
                        edgecolor="none",zorder=3)
            ax.axhline(.8,color="#f59e0b",linestyle="--",linewidth=2,
                       label="Legal minimum (0.80)",zorder=4)
            ax.axhline(1,color="#94a3b8",linestyle=":",linewidth=1,
                       label="Perfect fair (1.00)")
            for b,v in zip(bars,vals):
                ax.text(b.get_x()+b.get_width()/2,v+.025,f"{v:.3f}",
                        ha="center",fontsize=14,fontweight="bold",color="#111827")
            ax.set_ylim(0,1.3)
            ax.set_ylabel("Disparate Impact",color="#374151",fontsize=10)
            ax.set_title("Disparate Impact: Before vs After",color="#111827",
                         fontsize=11,fontweight="bold",pad=14)
            ax.tick_params(colors="#374151")
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.yaxis.grid(True,color="rgba(229,231,235,0.6)",zorder=0)
            ax.set_axisbelow(True); ax.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig,transparent=True); plt.close()

        with fc2:
            fig,ax=light_fig(6,4)
            x,w=np.arange(2),.3
            b1=ax.bar(x-w/2,[fr["male_before"],fr["female_before"]],
                      w,label="Before",color="#f87171",alpha=.9,edgecolor="none")
            b2=ax.bar(x+w/2,[fr["male_approval_after"],fr["female_approval_after"]],
                      w,label="After",color="#4ade80",alpha=.9,edgecolor="none")
            for grp in [b1,b2]:
                for b in grp:
                    ax.text(b.get_x()+b.get_width()/2,b.get_height()+.012,
                            f"{b.get_height():.1%}",ha="center",fontsize=10,
                            fontweight="bold",color="#111827")
            ax.set_xticks(x); ax.set_xticklabels(["Male","Female"],color="#374151")
            ax.set_ylim(0,1.1); ax.set_ylabel("Approval Rate",color="#374151",fontsize=10)
            ax.set_title("Approval Rate by Gender",color="#111827",
                         fontsize=11,fontweight="bold",pad=14)
            ax.tick_params(colors="#374151")
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.yaxis.grid(True,color="rgba(229,231,235,0.6)",zorder=0)
            ax.set_axisbelow(True); ax.legend(fontsize=9)
            plt.tight_layout(); st.pyplot(fig,transparent=True); plt.close()

    with st.expander("ℹ️  How bias was detected and fixed — Reweighing explained"):
        st.markdown("""
        **What is the bias?**
        Historical data showed male applicants receiving more approvals than female
        applicants with similar financial profiles.

        **Fix — Reweighing Algorithm**
        Assigns training sample weights so the model pays more attention to underrepresented cases:
        - Female applicants approved → **higher weight** (model learns fairer patterns)
        - Male applicants approved → **lower weight**

        **Results:** Gender gap reduced **8.1% → 4.0%** · Disparate Impact improved **0.89 → 0.92**
        """)


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif st.session_state.page=="perf":
    st.markdown("""
    <div style='background:rgba(255,255,255,0.85);backdrop-filter:blur(20px);
         border-radius:20px;border:1px solid rgba(255,255,255,0.9);
         padding:1.6rem 2rem;margin-bottom:1.4rem;
         box-shadow:0 4px 24px rgba(0,0,0,.06)'>
      <div style='display:flex;align-items:center;gap:12px'>
        <div style='width:40px;height:40px;border-radius:12px;
             background:linear-gradient(135deg,#ecfdf5,#d1fae5);
             display:flex;align-items:center;justify-content:center;font-size:1.2rem'>📊</div>
        <div>
          <h2 style='font-size:1.3rem;font-weight:800;color:#111827;margin:0;
                     letter-spacing:-.3px'>Model Performance</h2>
          <p style='font-size:.82rem;color:#6b7280;margin:0'>
            Bias-mitigated Logistic Regression · 200 held-out test applicants
          </p>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    preds=pipe.predict(X_te_raw); probs=pipe.predict_proba(X_te_raw)[:,1]
    acc=accuracy_score(y_te,preds); f1=f1_score(y_te,preds); auc=roc_auc_score(y_te,probs)

    pc1,pc2,pc3,pc4=st.columns(4)
    pc1.metric("Accuracy",    f"{acc:.1%}", "overall correct")
    pc2.metric("F1 Score",    f"{f1:.3f}",  "precision × recall")
    pc3.metric("AUC-ROC",     f"{auc:.3f}", "ranking ability")
    pc4.metric("Test Samples","200",        "20% hold-out")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    rc1,rc2=st.columns(2,gap="large")

    with rc1:
        fig,ax=light_fig(5.5,4.5)
        cm=confusion_matrix(y_te,preds)
        cm_pct=cm.astype(float)/cm.sum(axis=1,keepdims=True)
        sns.heatmap(cm_pct,annot=True,fmt=".1%",cmap="Blues",ax=ax,
                    xticklabels=["Rejected","Approved"],
                    yticklabels=["Rejected","Approved"],
                    linewidths=.5,cbar=False,
                    annot_kws={"size":13,"color":"#111827"})
        for i in range(2):
            for j in range(2):
                ax.text(j+.5,i+.73,f"n={cm[i,j]}",
                        ha="center",fontsize=9,color="#6b7280")
        ax.set_title("Confusion Matrix",color="#111827",
                     fontsize=11,fontweight="bold",pad=14)
        ax.set_xlabel("Predicted",color="#374151",fontsize=10)
        ax.set_ylabel("Actual",   color="#374151",fontsize=10)
        ax.tick_params(colors="#374151")
        plt.tight_layout(); st.pyplot(fig,transparent=True); plt.close()

    with rc2:
        fpr,tpr,_=roc_curve(y_te,probs)
        fig,ax=light_fig(5.5,4.5)
        ax.plot(fpr,tpr,color="#10b981",linewidth=2.5,
                label=f"Fair Model  (AUC = {auc:.3f})")
        ax.fill_between(fpr,tpr,alpha=.1,color="#10b981")
        ax.plot([0,1],[0,1],color="#94a3b8",linestyle="--",
                linewidth=1.2,label="Random baseline")
        ax.set_xlabel("False Positive Rate",color="#374151",fontsize=10)
        ax.set_ylabel("True Positive Rate", color="#374151",fontsize=10)
        ax.set_title("ROC Curve",color="#111827",
                     fontsize=11,fontweight="bold",pad=14)
        ax.tick_params(colors="#374151"); ax.legend(fontsize=9)
        ax.set_xlim([0,1]); ax.set_ylim([0,1])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.yaxis.grid(True,color="rgba(229,231,235,0.5)")
        ax.xaxis.grid(True,color="rgba(229,231,235,0.5)")
        plt.tight_layout(); st.pyplot(fig,transparent=True); plt.close()