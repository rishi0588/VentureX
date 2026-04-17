# VentureX — Intelligent VC Decision Agent

> "Instead of guessing which startup will succeed, VentureX helps us decide intelligently."

**Team:** Rishi Ponda (S046) · Anmol Singh (S007) · Priyanshu Padhi (S040)

---

## 🚀 Quick Start (3 commands)

```bash
# 1. Install everything + generate data + train models
python setup.py

# 2. Launch the Streamlit app
streamlit run app.py
```

That's it. The browser will open at `http://localhost:8501`

---

## 📁 Project Structure

```
VentureX/
├── app.py                    ← Streamlit UI (5 tabs)
├── setup.py                  ← One-click setup script
├── requirements.txt
│
├── data/
│   ├── generate_data.py      ← Generates 600 synthetic startups
│   └── startups.csv          ← Auto-generated
│
├── scoring/
│   └── engine.py             ← financial / market / founder / risk scores
│
├── models/
│   ├── train.py              ← Trains 3 ML models + plots
│   ├── best_model.pkl        ← Auto-saved
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── gradient_boosting.pkl
│
├── agents/
│   └── agents.py             ← Random / Rule-Based / ML / RL agents
│
├── simulation/
│   └── simulate.py           ← Portfolio simulation engine
│
└── plots/                    ← Auto-generated evaluation charts
```

---

## 🧩 System Pipeline

```
Startup Features (11 inputs)
        ↓
Preprocessing + Normalization
        ↓
Scoring Engine
  ├── financial_score()   — revenue, growth, burn
  ├── market_score()      — market size, competition
  ├── founder_score()     — experience, exits, team
  └── risk_score()        — risk level, burn ratio
        ↓
Composite Score = 0.30×fin + 0.25×mkt + 0.20×fnd − 0.25×risk
        ↓
ML Prediction (Random Forest)  +  RL Agent (Q-Learning)
        ↓
Decision: ✅ INVEST / ⏳ WATCHLIST / ❌ REJECT
```

---

## 📊 Scoring Thresholds

| Score     | Decision   |
|-----------|------------|
| ≥ 0.60    | ✅ INVEST   |
| 0.40–0.60 | ⏳ WATCHLIST|
| < 0.40    | ❌ REJECT   |

---

## 🤖 Agent Types

| Agent        | Strategy                          |
|--------------|-----------------------------------|
| Random       | Random decisions (baseline)       |
| Rule-Based   | Fixed financial thresholds        |
| ML Agent     | Random Forest success probability |
| RL Agent     | Q-Learning, improves over time    |

---

## 🖥 App Tabs

1. **Evaluate** — Live scoring of any startup (sidebar sliders)
2. **EDA** — Dataset exploration, correlations, charts
3. **Simulate** — Portfolio simulation across 50–500 startups
4. **RL Agent** — Q-table visualization + reward curve
5. **About** — Team, architecture, tech stack
