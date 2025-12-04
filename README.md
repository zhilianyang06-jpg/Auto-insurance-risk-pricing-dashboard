# 📘 Behavior-Based Auto Insurance Pricing Dashboard  
### Machine Learning × Telematics × Elasticity Modeling × Actuarial Pricing

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-URL.streamlit.app)

This project builds an interactive **auto insurance pricing simulation dashboard** that integrates telematics-based risk segmentation, actuarial pricing logic, and price elasticity modeling. It allows insurers, analysts, or researchers to explore how risk-based pricing strategies influence revenue, customer volume, and portfolio composition.

👉 **Live Demo**  
*https://auto-insurance-risk-pricing-dashboard-u6fsmyvdks2ni2kb9mkgyq.streamlit.app/*


## 🔍 Overview

This Streamlit dashboard enables users to:

- Upload telematics-derived driver data  
- View ML-based risk group segmentation (Low / Medium / High / Very High)  
- Set base premiums for each risk tier  
- Configure three pricing strategies (Aggressive / Standard / Conservative)  
- Adjust price elasticity to simulate real-world customer behavior  
- Compare revenue uplift & customer volume changes vs. baseline  
- Export a full **Methodology & References PDF**

The project demonstrates the full pipeline of **behavior-based pricing**, combining ML predictions, actuarial methods, and behavioral economics.

---

## 🧠 System Architecture

```
                                ┌───────────────────────────┐
                                │     Uploaded Dataset      │
                                │ (Potential Customer Pool) │
                                └──────────────┬────────────┘
                                               │
                                               ▼
                          ┌────────────────────────────────────────┐
                          │   Telematics-Based Risk Segmentation   │
                          │  (ML model → Low / Med / High / VHigh) │
                          └────────────────────┬───────────────────┘
                                               │
                                               ▼
          ┌────────────────────────────────────────────────────────────────────────┐
          │                         Baseline Construction                          │
          │  Base Premiums + Initial Acceptance Rate → Baseline Customers/Revenue  │
          └────────────────────────────────────┬───────────────────────────────────┘
                                               │
                                               ▼
    ┌───────────────────────────────────────────────────────────────────────────────┐
    │        Strategic Pricing Adjustments (Aggressive / Standard / Conservative)   │
    │        Risk-based Loadings → Adjusted Premiums per Risk Tier                  │
    └──────────────────────────────────────────┬────────────────────────────────────┘
                                               │
                                               ▼
             ┌────────────────────────────────────────────────────────────────────┐
             │                        Elasticity Simulation                       │
             │   Acceptance Rate Adjustment → Dynamic Volume → Dynamic Revenue    │
             └────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
       ┌───────────────────────────────────────────────────────────────────────┐
       │                              Dashboard Outputs                        │
       │           Revenue Uplift | Customer Volume Change | Risk Mix Shift    │
       │           Downloadable Methodology PDF | Exportable Results           │
       └───────────────────────────────────────────────────────────────────────┘
```


## 🧠 Core Methodology

Risk segmentation follows the framework introduced by:

**Marcillo et al. (2024). POLIDriving Dataset.**  
Telematics features feed into a conceptual **Gradient Boosting Machine (GBM)** model to estimate crash-risk probability. Drivers are grouped into four risk tiers used for pricing and elasticity simulation.

Pricing logic includes:

- Baseline revenue under traditional demographic pricing  
- Strategy-adjusted premiums  
- Elasticity-adjusted acceptance rates  
- Dynamic acquisition volume & revenue  
- Revenue uplift and portfolio mix impact  

Full methodology is available as a downloadable PDF within the dashboard.


## 📦 Project Structure

```
insurance-pricing-dashboard/
│── app2.py
│── requirements.txt
│── README.md
```


## ▶️ Running Locally

```bash
pip install -r requirements.txt
streamlit run app2.py
```

## ☁️ Deployment (Streamlit Cloud)

1. Push this repo to GitHub  
2. Visit https://share.streamlit.io  
3. Click **“New app”**  
4. Select your repo  
5. Set the main file to:

```
app2.py
```

6. Click **Deploy**  

The application will build automatically using `requirements.txt`.

## 📄 Methodology PDF

The dashboard automatically generates a full professional PDF including:

- Pipeline architecture  
- Risk segmentation logic  
- Pricing formulas  
- Elasticity behavior modeling  
- Key assumptions  
- Academic references  


## 👩‍💻 Author

**Zhilian (Lillian) Yang**  
www.linkedin.com/in/zhilianyang


