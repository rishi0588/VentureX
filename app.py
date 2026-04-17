"""
VentureX — Streamlit Application
Run: streamlit run app.py
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from scoring.engine    import evaluate_startup, financial_score, market_score, founder_score, risk_score
from agents.agents     import RandomAgent, RuleBasedAgent, MLAgent, RLAgent
from simulation.simulate import simulate, summary_stats


# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="VentureX — VC Decision Agent",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main { background-color: #FAFAFA; }
    .block-container { padding-top: 2rem; }
    .verdict-invest    { background:#E8F5E9; border-left:5px solid #4CAF50; padding:12px 18px; border-radius:6px; color: black;}
    .verdict-watchlist { background:#FFF8E1; border-left:5px solid #FFC107; padding:12px 18px; border-radius:6px; color: black;}
    .verdict-reject    { background:#FFEBEE; border-left:5px solid #F44336; padding:12px 18px; border-radius:6px; color: black;}
    .score-card { background:white; border-radius:10px; padding:16px; text-align:center;
                  box-shadow:0 2px 8px rgba(0,0,0,0.07); }
    .metric-title { font-size:13px; color:#777; margin-bottom:4px; }
    .metric-val   { font-size:28px; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Cached resources
# ──────────────────────────────────────────────
@st.cache_resource
def load_ml_agent():
    return MLAgent()

@st.cache_data
def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "data", "startups.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_resource
def get_rl_agent():
    rl = RLAgent()
    df = load_dataset()
    if df is not None:
        startups = df.head(200).to_dict("records")
        rl.train(startups, episodes=1500)
    return rl


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def score_color(val: float):
    if val >= 0.60: return "#4CAF50"
    if val >= 0.40: return "#FFC107"
    return "#F44336"

def verdict_html(result: dict) -> str:
    v = result["verdict"]
    css = {"INVEST": "verdict-invest", "WATCHLIST": "verdict-watchlist", "REJECT": "verdict-reject"}[v]
    icon = {"INVEST": "✅", "WATCHLIST": "⏳", "REJECT": "❌"}[v]
    return f"""
    <div class='{css}'>
      <h2 style='margin:0'>{icon} {v}</h2>
      <p style='margin:6px 0 0'>Composite Score: <b>{result['composite_score']:.2%}</b></p>
    </div>"""

def gauge_chart(score: float, label: str = "Composite Score"):
    fig, ax = plt.subplots(figsize=(3.5, 2.2), subplot_kw=dict(polar=True))
    theta = np.linspace(0, np.pi, 200)
    # Background arc
    ax.plot(theta, [1]*200, color="#E0E0E0", linewidth=12, solid_capstyle="round")
    # Value arc
    fill_theta = np.linspace(0, np.pi * score, 200)
    c = score_color(score)
    ax.plot(fill_theta, [1]*200, color=c, linewidth=12, solid_capstyle="round")
    ax.set_ylim(0, 1.5)
    ax.set_theta_offset(np.pi); ax.set_theta_direction(-1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.text(0, 0, f"{score:.0%}", ha="center", va="center",
            fontsize=22, fontweight="bold", color=c, transform=ax.transData)
    ax.text(0, -0.35, label, ha="center", va="center",
            fontsize=9, color="#666", transform=ax.transData)
    plt.tight_layout(pad=0)
    return fig

def score_breakdown_chart(result: dict):
    labels = ["Financial", "Market", "Founder", "Risk Penalty"]
    vals   = [result["financial_score"], result["market_score"],
              result["founder_score"],   result["risk_penalty"]]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
    fig, ax = plt.subplots(figsize=(5, 2.8))
    bars = ax.barh(labels, vals, color=colors, alpha=0.85, height=0.55)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=10)
    ax.set_xlim(0, 1.1); ax.set_xlabel("Score (0–1)")
    ax.set_title("Score Breakdown", fontsize=11, fontweight="bold")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────
# Sidebar — Startup Inputs
# ──────────────────────────────────────────────

def sidebar_inputs():
    st.sidebar.image("https://img.icons8.com/fluency/96/rocket.png", width=60)
    st.sidebar.title("🚀 VentureX")
    st.sidebar.caption("Intelligent VC Decision Agent")
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Startup Profile")

    startup = {}

    st.sidebar.markdown("**Financial**")
    startup["revenue"]        = st.sidebar.slider("Revenue (₹ Thousands)",  0.0, 1000.0, 150.0, 10.0) * 1000
    startup["growth_rate"]    = st.sidebar.slider("Growth Rate (%)",     0.0, 150.0,  45.0,  1.0)
    startup["burn_rate"]      = st.sidebar.slider("Burn Rate (₹ Thousands)", 1.0, 500.0,  30.0,  5.0) * 1000

    st.sidebar.markdown("**Market**")
    startup["market_size"]      = st.sidebar.slider("Market Size (₹ Lakhs)", 10.0, 50000.0, 2500.0, 100.0) * 100
    startup["competition_score"]= st.sidebar.slider("Competition (1=low, 10=high)", 1.0, 10.0, 4.0, 0.5)
  
    st.sidebar.markdown("**Founder & Team**")
    startup["founder_experience"] = st.sidebar.slider("Founder Experience (yrs)", 0.0, 25.0, 8.0, 0.5)
    startup["prev_exits"]         = st.sidebar.slider("Previous Exits",           0,   5,    1)
    startup["team_size"]          = st.sidebar.slider("Team Size",                 2,  100,  12)
    startup["patents"]            = st.sidebar.slider("Patents Filed",             0,   10,   2)

    st.sidebar.markdown("**Risk & Stage**")
    startup["risk_level"]     = st.sidebar.select_slider("Risk Level", [0,1,2], value=1,
                                    format_func=lambda x: ["Low","Medium","High"][x])
    startup["funding_stage"]  = st.sidebar.selectbox("Funding Stage",
                                    ["Pre-Seed","Seed","Series A","Series B","Series C"], index=2)
    startup["sector"]         = st.sidebar.selectbox("Sector",
                                    ["FinTech","HealthTech","EdTech","SaaS",
                                     "E-Commerce","DeepTech","CleanTech","LogisTech"])
    return startup


# ──────────────────────────────────────────────
# Tab 1 — Evaluate
# ──────────────────────────────────────────────

def tab_evaluate(startup, ml_agent, rl_agent):
    st.title("💡 VentureX — Startup Evaluation")
    result = evaluate_startup(startup)

    col_gauge, col_verdict, col_agents = st.columns([1.4, 1.6, 2])

    with col_gauge:
        st.pyplot(gauge_chart(result["composite_score"]), width="stretch")

    with col_verdict:
        st.markdown(verdict_html(result), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        ml_prob = ml_agent.predict_probability(startup) if ml_agent.ready else 0.5
        st.metric("ML Success Probability", f"{ml_prob:.1%}")

        # Show stage & sector multipliers so users see they affect composite score
        sm  = result.get("stage_multiplier",  1.0)
        sec = result.get("sector_multiplier", 1.0)
        delta_pct = round((sm * sec - 1.0) * 100, 1)
        st.markdown(
            f"📌 **Stage** `{startup.get('funding_stage','?')}` → `{sm:.2f}x`  \n"
            f"📌 **Sector** `{startup.get('sector','?')}` → `{sec:.2f}x`  \n"
            f"📌 **Combined boost:** `{delta_pct:+.1f}%` on positive scores"
        )

    with col_agents:
        st.subheader("🤖 All Agent Decisions")
        agents = {
            "Random":    RandomAgent(),
            "Rule-Based":RuleBasedAgent(),
            "ML Agent":  ml_agent,
            "RL Agent":  rl_agent,
        }
        for name, agent in agents.items():
            dec   = agent.decide(startup)
            icon  = {"INVEST":"✅","WATCHLIST":"⏳","REJECT":"❌"}.get(dec,"⬜")
            color = {"INVEST":"green","WATCHLIST":"orange","REJECT":"red"}.get(dec,"gray")
            st.markdown(f"**{name}** → :{color}[{icon} {dec}]")

    st.markdown("---")

    col_break, col_raw = st.columns([1.5, 1])
    with col_break:
        st.pyplot(score_breakdown_chart(result), width="stretch")
    with col_raw:
        st.subheader("📊 Score Details")
        score_fields = [
            ("Financial Score",   result["financial_score"]),
            ("Market Score",      result["market_score"]),
            ("Founder Score",     result["founder_score"]),
            ("Risk Penalty",      result["risk_penalty"]),
            ("Composite Score",   result["composite_score"]),
        ]
        for label, val in score_fields:
            st.write(f"**{label}**: `{val:.4f}`")
            st.progress(float(np.clip(val, 0, 1)))
        st.write(f"**Stage Multiplier**: `{result.get('stage_multiplier', 1.0):.2f}x`")
        st.write(f"**Sector Multiplier**: `{result.get('sector_multiplier', 1.0):.2f}x`")


# ──────────────────────────────────────────────
# Tab 2 — EDA
# ──────────────────────────────────────────────

def tab_eda():
    st.title("📊 Dataset Exploration")
    df = load_dataset()
    if df is None:
        st.warning("Run `python data/generate_data.py` first.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Startups", len(df))
    c2.metric("Success Rate",   f"{df['success'].mean()*100:.1f}%")
    c3.metric("Avg Growth",     f"{df['growth_rate'].mean():.1f}%")
    c4.metric("Avg Founder Exp",f"{df['founder_experience'].mean():.1f} yrs")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Success Rate by Sector")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        rate = df.groupby("sector")["success"].mean().sort_values() * 100
        ax.barh(rate.index, rate.values, color="#4CAF50", alpha=0.8)
        ax.set_xlabel("Success %"); ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)

    with col2:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        numeric_cols = ["revenue","growth_rate","burn_rate","market_size",
                        "founder_experience","risk_level","success"]
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".1f",
                    cmap="RdYlGn", ax=ax, linewidths=0.5, annot_kws={"size":8})
        plt.tight_layout()
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Growth Rate vs Success")
        fig, ax = plt.subplots(figsize=(5, 3))
        for lbl, grp in df.groupby("success"):
            ax.hist(grp["growth_rate"], bins=25, alpha=0.6,
                    label=["Failed","Succeeded"][lbl], density=True)
        ax.legend(); ax.set_xlabel("Growth Rate (%)")
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)

    with col4:
        st.subheader("Funding Stage Distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        counts = df["funding_stage"].value_counts()
        ax.pie(counts.values, labels=counts.index, autopct="%1.0f%%",
               colors=["#4CAF50","#2196F3","#FF9800","#9C27B0","#F44336"])
        st.pyplot(fig)

    st.subheader("Raw Data Sample")
    st.dataframe(df.head(50), width="stretch")


# ──────────────────────────────────────────────
# Tab 3 — Simulation
# ──────────────────────────────────────────────

def tab_simulation(ml_agent, rl_agent):
    st.title("🧪 Portfolio Simulation")

    df = load_dataset()
    if df is None:
        st.warning("Run `python data/generate_data.py` first."); return

    col1, col2, col3 = st.columns(3)
    n_startups  = col1.slider("Number of Startups", 50, 500, 100, 50)
    market      = col2.selectbox("Market Condition", ["normal","boom","downturn"])
    run_btn     = col3.button("▶ Run Simulation", type="primary", width="stretch")

    if run_btn or "sim_df" in st.session_state:
        if run_btn:
            with st.spinner("Running simulation …"):
                startups = df.sample(n=min(n_startups, len(df)), random_state=42).to_dict("records")
                sim_df   = simulate(startups, rl_agent=rl_agent, market_condition=market)
                st.session_state["sim_df"] = sim_df

        sim_df = st.session_state["sim_df"]
        stats  = summary_stats(sim_df)

        st.subheader("📋 Agent Performance Summary")
        st.dataframe(
            stats.style
                .highlight_max(subset=["Accuracy %","Total Return"], color="#C8E6C9")
                .highlight_min(subset=["Accuracy %","Total Return"], color="#FFCDD2")
                .format({"Accuracy %": "{:.1f}%", "Total Return": "₹{:.1f}"}),
            width="stretch"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Decision Distribution")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            agents  = ["Random","Rule-Based","ML","RL"]
            x       = np.arange(len(agents))
            for i, dec in enumerate(["INVEST","WATCHLIST","REJECT"]):
                vals = [(sim_df[f"{a}_decision"]==dec).sum() for a in agents]
                ax.bar(x + i*0.25, vals, 0.25,
                       label=dec, color=["#4CAF50","#FFC107","#F44336"][i], alpha=0.85)
            ax.set_xticks(x+0.25); ax.set_xticklabels(agents)
            ax.legend(); ax.spines[["top","right"]].set_visible(False)
            st.pyplot(fig)

        with col_b:
            st.subheader("Cumulative Returns")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            clrs = {"Random":"#9E9E9E","Rule-Based":"#2196F3","ML":"#4CAF50","RL":"#FF9800"}
            for a in agents:
                ax.plot(sim_df[f"{a}_return"].cumsum().values,
                        label=a, color=clrs[a], linewidth=2)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Startup #"); ax.legend()
            ax.spines[["top","right"]].set_visible(False)
            st.pyplot(fig)

        # Export
        csv = sim_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Simulation CSV", csv,
                           "venturex_simulation.csv", "text/csv")


# ──────────────────────────────────────────────
# Tab 4 — RL Training
# ──────────────────────────────────────────────

def tab_rl(rl_agent):
    st.title("🤖 RL Agent Training")

    st.markdown("""
    The **Reinforcement Learning Agent** uses **Q-Learning** to learn an optimal
    investment strategy through repeated interactions with the startup environment.
    - **State**: Discretised composite score (8 bins)
    - **Actions**: Invest / Watchlist / Reject
    - **Reward**: +2 correct invest, +1.5 correct reject, −2.5 wrong invest
    """)

    if rl_agent.trained and rl_agent.rewards_history:
        st.success("✅ RL Agent is trained and ready.")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Reward curve
        rewards = rl_agent.rewards_history
        window  = max(1, len(rewards)//50)
        smoothed= pd.Series(rewards).rolling(window).mean()
        axes[0].plot(rewards, alpha=0.3, color="#4CAF50", linewidth=1)
        axes[0].plot(smoothed, color="#2E7D32", linewidth=2)
        axes[0].set_title("Average Reward per Episode"); axes[0].set_xlabel("Episode")
        axes[0].spines[["top","right"]].set_visible(False)

        # Q-table heatmap
        sns.heatmap(rl_agent.q_table, annot=True, fmt=".2f",
                    xticklabels=["Invest","Watchlist","Reject"],
                    yticklabels=[f"S{i}" for i in range(8)],
                    cmap="RdYlGn", ax=axes[1], linewidths=0.5)
        axes[1].set_title("Q-Table (State × Action)")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("RL Agent training in progress … reload the app to see results.")


# ──────────────────────────────────────────────
# Tab 5 — About
# ──────────────────────────────────────────────

def tab_about():
    st.title("ℹ About VentureX")
    st.markdown("""
    ## VentureX — Intelligent VC Decision Agent

    > *"Instead of guessing which startup will succeed, VentureX helps us decide intelligently."*

    ---

    ### 👥 Team
    | Name | Roll No |
    |------|---------|
    | Rishi Ponda    | S046 |
    | Anmol Singh    | S007 |
    | Priyanshu Padhi| S040 |

    ---

    ### 🧩 System Architecture
    ```
    Startup Data (features)
         ↓
    Preprocessing + EDA
         ↓
    Scoring Engine   ←── financial / market / founder / risk
         ↓
    ML Prediction    ←── Logistic Regression / Random Forest
         ↓
    RL Agent         ←── Q-Learning on composite environment
         ↓
    Decision: INVEST / WATCHLIST / REJECT
    ```

    ### 📦 Tech Stack
    - **ML**: scikit-learn (Logistic Regression, Random Forest, Gradient Boosting)
    - **RL**: Custom Q-Learning (NumPy)
    - **UI**: Streamlit
    - **Data**: Synthetic (NumPy / Faker)
    - **Viz**: Matplotlib, Seaborn

    ### 📊 Scoring Formula
    ```
    score = 0.30 × financial + 0.25 × market + 0.20 × founder − 0.25 × risk
    ```

    | Threshold | Decision   |
    |-----------|------------|
    | ≥ 0.60    | ✅ INVEST   |
    | 0.40–0.60 | ⏳ WATCHLIST|
    | < 0.40    | ❌ REJECT   |
    """)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    startup   = sidebar_inputs()
    ml_agent  = load_ml_agent()
    rl_agent  = get_rl_agent()

    tabs = st.tabs(["🔍 Evaluate", "📊 EDA", "🧪 Simulate", "🤖 RL Agent", "ℹ About"])

    with tabs[0]: tab_evaluate(startup, ml_agent, rl_agent)
    with tabs[1]: tab_eda()
    with tabs[2]: tab_simulation(ml_agent, rl_agent)
    with tabs[3]: tab_rl(rl_agent)
    with tabs[4]: tab_about()


if __name__ == "__main__":
    main()
