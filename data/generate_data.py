import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_startup_dataset(n=600):
    """Generate synthetic startup dataset with realistic distributions."""

    funding_stages = ["Pre-Seed", "Seed", "Series A", "Series B", "Series C"]
    sectors = ["FinTech", "HealthTech", "EdTech", "SaaS", "E-Commerce",
               "DeepTech", "CleanTech", "LogisTech"]

    data = []
    for i in range(n):
        # Core financial features
        revenue         = np.random.lognormal(mean=13.5, sigma=1.8)   # ₹ in lakhs
        growth_rate     = np.random.beta(2, 5) * 150                   # 0–150 %
        burn_rate       = np.random.lognormal(mean=12.0, sigma=1.5)    # ₹ in lakhs
        market_size     = np.random.lognormal(mean=18, sigma=1.5)      # ₹ in crores
        competition_score = np.random.uniform(1, 10)                   # 1=low, 10=high
        founder_experience = np.random.uniform(0, 20)                  # years
        funding_stage   = np.random.choice(funding_stages)
        sector          = np.random.choice(sectors)
        team_size       = np.random.randint(2, 80)
        patents         = np.random.randint(0, 6)
        prev_exits      = np.random.randint(0, 4)

        # Derived risk level (0=low, 1=medium, 2=high)
        runway_months = (revenue * 1.2) / (burn_rate / 12 + 1e-6)
        risk_score_raw = (
            (competition_score / 10) * 0.4 +
            (1 - min(runway_months, 24) / 24) * 0.4 +
            (1 - min(founder_experience, 10) / 10) * 0.2
        )
        if risk_score_raw < 0.35:
            risk_level = 0
        elif risk_score_raw < 0.65:
            risk_level = 1
        else:
            risk_level = 2

        # Compute ground-truth label (success probability)
        fin  = min(revenue / 100000, 1) * 0.35 + min(growth_rate / 100, 1) * 0.65
        mkt  = min(market_size / 1e9, 1) * 0.6 + (1 - competition_score / 10) * 0.4
        fnd  = min(founder_experience / 15, 1) * 0.5 + min(prev_exits / 3, 1) * 0.3 + min(team_size / 30, 1) * 0.2
        rsk  = 1 - risk_score_raw

        composite = 0.30 * fin + 0.25 * mkt + 0.25 * fnd + 0.20 * rsk
        composite += np.random.normal(0, 0.05)   # noise
        composite = np.clip(composite, 0, 1)

        label = 1 if composite >= 0.50 else 0

        data.append({
            "revenue":             round(revenue, 2),
            "growth_rate":         round(growth_rate, 2),
            "burn_rate":           round(burn_rate, 2),
            "market_size":         round(market_size, 2),
            "competition_score":   round(competition_score, 2),
            "founder_experience":  round(founder_experience, 2),
            "funding_stage":       funding_stage,
            "sector":              sector,
            "team_size":           team_size,
            "patents":             patents,
            "prev_exits":          prev_exits,
            "risk_level":          risk_level,
            "success":             label,
        })

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "startups.csv")
    df.to_csv(out, index=False)
    print(f"✅ Dataset saved → {out}  ({len(df)} rows, {df['success'].mean()*100:.1f}% success rate)")
    return df


if __name__ == "__main__":
    generate_startup_dataset()
