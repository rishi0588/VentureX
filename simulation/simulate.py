"""
VentureX Simulation Engine
Simulates 100+ startups through all agents and tracks outcomes.
"""

import os, sys, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scoring.engine import evaluate_startup
from agents.agents  import RandomAgent, RuleBasedAgent, MLAgent, RLAgent

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# Simulate
# ──────────────────────────────────────────────

def simulate(startups: list, rl_agent=None, market_condition: str = "normal"):
    """
    Runs all agents over a list of startup dicts.
    market_condition: 'normal' | 'boom' | 'downturn'
    Returns a DataFrame of results.
    """
    market_multiplier = {"normal": 1.0, "boom": 1.25, "downturn": 0.75}[market_condition]

    agents = {
        "Random":     RandomAgent(),
        "Rule-Based": RuleBasedAgent(),
        "ML":         MLAgent(),
        "RL":         rl_agent if rl_agent else RuleBasedAgent(),  # fallback
    }
    if rl_agent: agents["RL"] = rl_agent

    rows = []
    for i, s in enumerate(startups):
        # Apply market multiplier
        s_adj = dict(s)
        s_adj["revenue"]     *= market_multiplier
        s_adj["market_size"] *= market_multiplier

        scoring = evaluate_startup(s_adj)
        true_success = int(scoring["composite_score"] >= 0.55)

        row = {
            "startup_id":      i + 1,
            "composite_score": scoring["composite_score"],
            "true_outcome":    true_success,
            "market":          market_condition,
        }
        for name, agent in agents.items():
            decision    = agent.decide(s_adj)
            # Was the decision correct?
            correct = (
                (decision == "INVEST"    and true_success == 1) or
                (decision == "REJECT"    and true_success == 0) or
                (decision == "WATCHLIST")
            )
            # Simulated return
            if decision == "INVEST":
                ret = (+3.5 if true_success else -1.0) * market_multiplier
            elif decision == "WATCHLIST":
                ret = 0.2
            else:
                ret = 0.0

            row[f"{name}_decision"] = decision
            row[f"{name}_correct"]  = int(correct)
            row[f"{name}_return"]   = round(ret, 2)

        rows.append(row)

    return pd.DataFrame(rows)


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    agent_names = ["Random", "Rule-Based", "ML", "RL"]
    rows = []
    for a in agent_names:
        invests = (df[f"{a}_decision"] == "INVEST").sum()
        rejects = (df[f"{a}_decision"] == "REJECT").sum()
        watches = (df[f"{a}_decision"] == "WATCHLIST").sum()
        acc     = df[f"{a}_correct"].mean()
        total_r = df[f"{a}_return"].sum()
        rows.append({
            "Agent":       a,
            "Invest":      invests,
            "Watchlist":   watches,
            "Reject":      rejects,
            "Accuracy %":  round(acc * 100, 1),
            "Total Return":round(total_r, 2),
        })
    return pd.DataFrame(rows)


def plot_simulation(df: pd.DataFrame, title_suffix=""):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"VentureX Simulation Results {title_suffix}",
                 fontsize=15, fontweight="bold")

    agents = ["Random", "Rule-Based", "ML", "RL"]
    colors = {"Random": "#9E9E9E", "Rule-Based": "#2196F3",
              "ML": "#4CAF50", "RL": "#FF9800"}

    # 1. Decision distribution
    ax = axes[0, 0]
    decisions_data = {a: [
        (df[f"{a}_decision"] == "INVEST").sum(),
        (df[f"{a}_decision"] == "WATCHLIST").sum(),
        (df[f"{a}_decision"] == "REJECT").sum(),
    ] for a in agents}
    x   = np.arange(3)
    w   = 0.20
    lbls = ["Invest", "Watchlist", "Reject"]
    for i, a in enumerate(agents):
        ax.bar(x + i*w, decisions_data[a], w, label=a, color=colors[a], alpha=0.85)
    ax.set_xticks(x + 1.5*w); ax.set_xticklabels(lbls)
    ax.set_title("Decision Distribution"); ax.legend(fontsize=8)

    # 2. Accuracy comparison
    ax = axes[0, 1]
    accs = [df[f"{a}_correct"].mean() * 100 for a in agents]
    bars = ax.bar(agents, accs, color=[colors[a] for a in agents], alpha=0.85)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=9)
    ax.set_ylim(0, 110); ax.set_title("Decision Accuracy (%)")

    # 3. Cumulative returns
    ax = axes[0, 2]
    for a in agents:
        ax.plot(df[f"{a}_return"].cumsum().values, label=a,
                color=colors[a], linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Cumulative Returns"); ax.legend(fontsize=8)
    ax.set_xlabel("Startup #")

    # 4. Score distribution
    ax = axes[1, 0]
    ax.hist(df["composite_score"], bins=30, color="#4CAF50", alpha=0.75, edgecolor="white")
    ax.axvline(0.60, color="#F44336", linestyle="--", label="Invest threshold")
    ax.axvline(0.40, color="#FF9800", linestyle="--", label="Watchlist threshold")
    ax.set_title("Composite Score Distribution"); ax.legend(fontsize=8)

    # 5. Return boxplot
    ax = axes[1, 1]
    data = [df[f"{a}_return"].values for a in agents]
    bp   = ax.boxplot(data, labels=agents, patch_artist=True)
    for patch, a in zip(bp["boxes"], agents):
        patch.set_facecolor(colors[a]); patch.set_alpha(0.7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Return Distribution per Agent")

    # 6. True success rate by score bucket
    ax = axes[1, 2]
    df["score_bucket"] = pd.cut(df["composite_score"], bins=5)
    rate = df.groupby("score_bucket")["true_outcome"].mean() * 100
    ax.bar(range(len(rate)), rate.values, color="#4CAF50", alpha=0.85)
    ax.set_xticks(range(len(rate)))
    ax.set_xticklabels([str(b) for b in rate.index], rotation=25, fontsize=7)
    ax.set_title("True Success Rate by Score Bucket")
    ax.set_ylabel("Success %")

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"simulation{title_suffix.replace(' ','_')}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Simulation plot saved → {out}")
    return out


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import importlib.util, os

    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "startups.csv")
    if not os.path.exists(data_path):
        from data.generate_data import generate_startup_dataset
        generate_startup_dataset()

    df_data = pd.read_csv(data_path)
    startups = df_data.to_dict("records")[:200]

    # Train RL agent
    rl = RLAgent()
    rl.train(startups[:150], episodes=1500)

    for market in ["normal", "boom", "downturn"]:
        print(f"\n{'='*50}")
        print(f"  Simulating: {market.upper()} market")
        sim_df = simulate(startups[150:200], rl_agent=rl, market_condition=market)
        print(summary_stats(sim_df).to_string(index=False))
        plot_simulation(sim_df, title_suffix=f" ({market})")
        sim_df.to_csv(
            os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "data", f"simulation_{market}.csv"), index=False
        )

    print("\n✅ Simulation complete.")
