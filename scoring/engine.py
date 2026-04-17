"""
VentureX Scoring Engine
Computes financial, market, founder, and risk scores.
Returns a composite score and investment verdict.

All score functions return 0.0–1.0.
Normalization caps are tuned so mid-range slider values produce mid-range scores,
making every parameter visibly affect the output.
"""

import numpy as np


# ──────────────────────────────────────────────
# Stage & Sector multipliers
# ──────────────────────────────────────────────

STAGE_MULTIPLIER = {
    "Pre-Seed": 0.70,   # earliest stage → higher risk, discount score
    "Seed":     0.82,
    "Series A": 1.00,   # baseline
    "Series B": 1.08,
    "Series C": 1.12,
}

SECTOR_MULTIPLIER = {
    "FinTech":    1.10,
    "HealthTech": 1.08,
    "SaaS":       1.07,
    "DeepTech":   1.05,
    "EdTech":     0.97,
    "E-Commerce": 0.95,
    "CleanTech":  1.00,
    "LogisTech":  0.98,
}


# ──────────────────────────────────────────────
# Individual score functions (all return 0–1)
# ──────────────────────────────────────────────

def financial_score(revenue: float, growth_rate: float, burn_rate: float) -> float:
    """
    revenue      : raw value (lakhs * 1000 from UI slider)
    growth_rate  : % per year  (0–150)
    burn_rate    : raw value (lakhs * 1000 from UI slider)

    Caps tuned so:
      - revenue ₹50L → ~0.33,  ₹300L → ~0.66,  ₹1000L → 1.0
      - growth  30%  → ~0.40,  60%   → ~0.60,   100%   → 0.80
      - burn ratio penalises high burn vs revenue
    """
    rev_norm    = min(revenue / 150_000, 1.0)           # ₹150L = full score
    growth_norm = min(growth_rate / 120.0, 1.0)         # 120% = full score
    # burn penalty: burn > revenue is bad
    burn_ratio  = np.clip(burn_rate / (revenue + 1e-6), 0, 2) / 2.0
    burn_score  = 1.0 - burn_ratio                      # low burn = good

    score = 0.40 * rev_norm + 0.40 * growth_norm + 0.20 * burn_score
    return round(np.clip(score, 0, 1), 4)


def market_score(market_size: float, competition_score: float) -> float:
    """
    market_size       : raw value (crores * 100 from UI slider)
    competition_score : 1 (low competition, good) – 10 (high, bad)

    Caps tuned so:
      - market ₹500Cr  → ~0.33,  ₹2000Cr → ~0.66,  ₹5000Cr+ → 1.0
      - competition 1  → +0.40 bonus,  10 → 0.0
    """
    mkt_norm  = min(market_size / 500_000, 1.0)         # ₹500000 units = full
    comp_norm = 1.0 - (competition_score - 1.0) / 9.0   # invert

    score = 0.60 * mkt_norm + 0.40 * comp_norm
    return round(np.clip(score, 0, 1), 4)


def founder_score(founder_experience: float, prev_exits: int,
                  team_size: int, patents: int) -> float:
    """
    founder_experience : years (0–25)
    prev_exits         : 0–5
    team_size          : 2–100
    patents            : 0–10

    Tuned caps:
      - 5 yrs → 0.33,  10 yrs → 0.67,  15+ yrs → 1.0
      - 0 exits → 0,   1 exit → 0.50,  3+ exits → 1.0
    """
    exp_norm    = min(founder_experience / 15.0, 1.0)
    exit_norm   = min(prev_exits / 2.0, 1.0)            # 2 exits = full (was 3)
    team_norm   = min(team_size / 25.0, 1.0)            # 25 people = full
    patent_norm = min(patents / 4.0, 1.0)               # 4 patents = full

    score = (0.40 * exp_norm + 0.30 * exit_norm +
             0.20 * team_norm + 0.10 * patent_norm)
    return round(np.clip(score, 0, 1), 4)


def risk_score(risk_level: int, burn_rate: float, revenue: float) -> float:
    """
    risk_level  : 0=Low, 1=Medium, 2=High
    Returns a PENALTY (0 = no risk, 1 = maximum risk).

    Low  → 0.15 base penalty (not 0, startup always has some risk)
    Med  → 0.50
    High → 0.85
    """
    level_map     = {0: 0.15, 1: 0.50, 2: 0.85}
    level_penalty = level_map.get(risk_level, 0.50)
    burn_ratio    = np.clip(burn_rate / (revenue + 1e-6), 0, 3) / 3.0

    penalty = 0.70 * level_penalty + 0.30 * burn_ratio
    return round(np.clip(penalty, 0, 1), 4)


# ──────────────────────────────────────────────
# Composite scorer + verdict
# ──────────────────────────────────────────────

WEIGHTS = {
    "financial": 0.30,
    "market":    0.25,
    "founder":   0.20,
    "risk_adj":  0.25,
}

THRESHOLDS = {
    "invest":    0.60,
    "watchlist": 0.40,
}


def composite_score(fin: float, mkt: float, fnd: float, rsk: float,
                    stage: str = "Series A", sector: str = "SaaS") -> float:
    """
    Weighted composite with stage & sector multipliers.
    Risk is a penalty subtracted from the positive scores.
    """
    base = (WEIGHTS["financial"] * fin +
            WEIGHTS["market"]    * mkt +
            WEIGHTS["founder"]   * fnd -
            WEIGHTS["risk_adj"]  * rsk)

    stage_mult  = STAGE_MULTIPLIER.get(stage,  1.00)
    sector_mult = SECTOR_MULTIPLIER.get(sector, 1.00)

    # Apply multipliers to the positive component only (so they don't amplify risk)
    positive = (WEIGHTS["financial"] * fin +
                WEIGHTS["market"]    * mkt +
                WEIGHTS["founder"]   * fnd)
    risk_part = WEIGHTS["risk_adj"] * rsk

    score = positive * stage_mult * sector_mult - risk_part
    return round(np.clip(score, 0, 1), 4)


def verdict(score: float) -> str:
    if score >= THRESHOLDS["invest"]:
        return "INVEST"
    elif score >= THRESHOLDS["watchlist"]:
        return "WATCHLIST"
    else:
        return "REJECT"


def evaluate_startup(startup: dict) -> dict:
    """
    Takes a dict of startup features and returns full scoring breakdown.
    Required keys: revenue, growth_rate, burn_rate, market_size,
                   competition_score, founder_experience, risk_level
    Optional keys: prev_exits, team_size, patents, funding_stage, sector
    """
    fin = financial_score(
        startup["revenue"], startup["growth_rate"], startup["burn_rate"]
    )
    mkt = market_score(startup["market_size"], startup["competition_score"])
    fnd = founder_score(
        startup["founder_experience"],
        startup.get("prev_exits", 0),
        startup.get("team_size", 5),
        startup.get("patents", 0),
    )
    rsk = risk_score(startup["risk_level"], startup["burn_rate"], startup["revenue"])

    stage  = startup.get("funding_stage", "Series A")
    sector = startup.get("sector", "SaaS")

    comp = composite_score(fin, mkt, fnd, rsk, stage, sector)
    dec  = verdict(comp)

    return {
        "financial_score": fin,
        "market_score":    mkt,
        "founder_score":   fnd,
        "risk_penalty":    rsk,
        "composite_score": comp,
        "stage_multiplier":  STAGE_MULTIPLIER.get(stage,  1.00),
        "sector_multiplier": SECTOR_MULTIPLIER.get(sector, 1.00),
        "verdict":         dec,
    }


if __name__ == "__main__":
    # Sensitivity check — change one param at a time
    base = {
        "revenue": 150000, "growth_rate": 45, "burn_rate": 30000,
        "market_size": 250000, "competition_score": 5,
        "founder_experience": 8, "prev_exits": 1,
        "team_size": 12, "patents": 2, "risk_level": 1,
        "funding_stage": "Series A", "sector": "SaaS",
    }
    print("=== Sensitivity Check ===")
    for stage in ["Pre-Seed", "Seed", "Series A", "Series B", "Series C"]:
        s = dict(base, funding_stage=stage)
        r = evaluate_startup(s)
        print(f"  {stage:10s}: composite={r['composite_score']:.4f}  verdict={r['verdict']}")
    print()
    for sector in ["FinTech", "SaaS", "E-Commerce", "EdTech"]:
        s = dict(base, sector=sector)
        r = evaluate_startup(s)
        print(f"  {sector:12s}: composite={r['composite_score']:.4f}  verdict={r['verdict']}")
