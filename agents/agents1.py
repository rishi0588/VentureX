"""
VentureX Agents
Four agent types with increasing intelligence.
"""

import os, sys, pickle, random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scoring.engine import evaluate_startup, verdict

ACTIONS = ["INVEST", "WATCHLIST", "REJECT"]
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}


# ──────────────────────────────────────────────
# Base Agent
# ──────────────────────────────────────────────

class BaseAgent:
    name = "Base"
    def decide(self, startup: dict) -> str:
        raise NotImplementedError

    def batch_decide(self, startups):
        return [self.decide(s) for s in startups]


# ──────────────────────────────────────────────
# 1. Random Agent (baseline)
# ──────────────────────────────────────────────

class RandomAgent(BaseAgent):
    name = "Random Agent"
    def decide(self, startup: dict) -> str:
        return random.choice(ACTIONS)


# ──────────────────────────────────────────────
# 2. Rule-Based Agent
# ──────────────────────────────────────────────

class RuleBasedAgent(BaseAgent):
    name = "Rule-Based Agent"

    RULES = {
        "min_growth_rate":   15,
        "max_competition":    7,
        "min_founder_exp":    3,
        "max_risk_level":     2,
        "min_runway_ratio": 0.5,
    }

    def decide(self, startup: dict) -> str:
        invest_score = 0
        if startup.get("growth_rate", 0)        >= self.RULES["min_growth_rate"]:  invest_score += 1
        if startup.get("competition_score", 10) <= self.RULES["max_competition"]:  invest_score += 1
        if startup.get("founder_experience", 0) >= self.RULES["min_founder_exp"]:  invest_score += 1
        if startup.get("risk_level", 2)         <  self.RULES["max_risk_level"]:   invest_score += 1
        ratio = startup.get("revenue", 0) / (startup.get("burn_rate", 1e6) + 1e-6)
        if ratio >= self.RULES["min_runway_ratio"]:                                invest_score += 1

        r = invest_score / 5
        if r >= 0.80:   return "INVEST"
        elif r >= 0.50: return "WATCHLIST"
        else:           return "REJECT"


# ──────────────────────────────────────────────
# 3. ML Agent
# ──────────────────────────────────────────────

class MLAgent(BaseAgent):
    name = "ML Agent (Random Forest)"

    def __init__(self):
        base = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base, "models", "best_model.pkl")
        feat_path  = os.path.join(base, "models", "feature_cols.pkl")
        enc_path   = os.path.join(base, "models", "label_encoders.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f: self.model    = pickle.load(f)
            with open(feat_path,  "rb") as f: self.features = pickle.load(f)
            with open(enc_path,   "rb") as f: self.encoders = pickle.load(f)
            self.ready = True
        else:
            self.ready = False
            print("⚠ ML model not found. Run: python models/train.py")

    def _to_vector(self, startup: dict):
        numeric = [
            startup.get("revenue", 0),
            startup.get("growth_rate", 0),
            startup.get("burn_rate", 0),
            startup.get("market_size", 0),
            startup.get("competition_score", 5),
            startup.get("founder_experience", 0),
            startup.get("team_size", 5),
            startup.get("patents", 0),
            startup.get("prev_exits", 0),
            startup.get("risk_level", 1),
        ]
        try:
            stage_enc  = self.encoders["stage"].transform([startup.get("funding_stage", "Seed")])[0]
            sector_enc = self.encoders["sector"].transform([startup.get("sector", "SaaS")])[0]
        except Exception:
            stage_enc, sector_enc = 0, 0
        return np.array(numeric + [stage_enc, sector_enc]).reshape(1, -1)

    def decide(self, startup: dict) -> str:
        if not self.ready:
            return RuleBasedAgent().decide(startup)
        try:
            prob = self.model.predict_proba(self._to_vector(startup))[0][1]
        except AttributeError:
            pred = self.model.predict(self._to_vector(startup))[0]
            return "INVEST" if pred == 1 else "REJECT"
        except Exception:
            return RuleBasedAgent().decide(startup)
        if prob >= 0.60:   return "INVEST"
        elif prob >= 0.40: return "WATCHLIST"
        else:              return "REJECT"

    def predict_probability(self, startup: dict) -> float:
        if not self.ready:
            return 0.5
        try:
            return float(self.model.predict_proba(self._to_vector(startup))[0][1])
        except AttributeError:
            pred = self.model.predict(self._to_vector(startup))[0]
            return 0.75 if pred == 1 else 0.25
        except Exception:
            return 0.5


# ──────────────────────────────────────────────
# 4. RL Agent — Q-Learning with tuned rewards
#
# Key design decisions vs previous version:
#   • N_STATES = 10 (was 8) → finer score granularity
#   • ALPHA = 0.20 (was 0.15) → learns faster
#   • GAMMA = 0.95 (was 0.90) → values future rewards more
#   • EPSILON_MIN = 0.02 (was 0.05) → more exploitation at convergence
#   • EPSILON_DECAY = 0.997 (was 0.995) → slower decay, more exploration
#   • episodes default = 3000 (was 2000)
#   • Reward matrix rebalanced: big reward for correct invest (+4.0),
#     strong penalty for missing a winner (-3.0),
#     reduced wrong-invest penalty (-1.0 was -2.5)
#     → agent learns to invest aggressively on high-scoring startups
# ──────────────────────────────────────────────

class RLAgent(BaseAgent):
    name = "RL Agent (Q-Learning)"

    N_ACTIONS     = 3
    N_STATES      = 10      # 0.0-0.1, 0.1-0.2, ... 0.9-1.0
    ALPHA         = 0.20    # learning rate
    GAMMA         = 0.95    # discount factor (values future rewards more)
    EPSILON_START = 1.0
    EPSILON_MIN   = 0.02    # almost no randomness at convergence
    EPSILON_DECAY = 0.997   # slower decay → more thorough exploration

    def __init__(self):
        self.q_table         = np.zeros((self.N_STATES, self.N_ACTIONS))
        self.epsilon         = self.EPSILON_START
        self.trained         = False
        self.rewards_history = []

    def _state(self, startup: dict) -> int:
        result = evaluate_startup(startup)
        score  = result["composite_score"]
        return min(int(score * self.N_STATES), self.N_STATES - 1)

    def _reward(self, action_idx: int, startup: dict) -> float:
        """
        Reward matrix explanation:
        ┌─────────────────────────────────────────────────────────┐
        │  true=1 (good startup):                                 │
        │    INVEST    → +4.0  big reward, encourage investing    │
        │    WATCHLIST → -0.5  penalise sitting on fence          │
        │    REJECT    → -3.0  heavily punish missing a winner    │
        │                                                          │
        │  true=0.5 (borderline):                                  │
        │    INVEST    → +0.5  reward taking the calculated risk  │
        │    WATCHLIST → +0.5  also valid, both rewarded equally  │
        │    REJECT    → -0.3  small penalty                      │
        │                                                          │
        │  true=0 (bad startup):                                   │
        │    INVEST    → -1.0  penalise (but not harshly)         │
        │    WATCHLIST → +0.2  acceptable caution                 │
        │    REJECT    → +1.0  correct, reward it                 │
        └─────────────────────────────────────────────────────────┘
        The key change from v1: INVEST+success reward (+4.0) is now
        4x the INVEST+failure penalty (-1.0), so the agent learns
        that investing on strong signals is worth the occasional loss.
        """
        result  = evaluate_startup(startup)
        score   = result["composite_score"]
        truth   = 1 if score >= 0.60 else (0.5 if score >= 0.40 else 0)
        action  = ACTIONS[action_idx]

        reward_matrix = {
            (1,   "INVEST"):    +4.0,   # correct invest → biggest reward
            (0.5, "WATCHLIST"): +0.5,   # watchlist on borderline → ok
            (0,   "REJECT"):    +1.0,   # correct reject → good
            (1,   "WATCHLIST"): -0.5,   # missed a winner → small penalty
            (1,   "REJECT"):    -3.0,   # rejected a winner → big penalty
            (0,   "INVEST"):    -1.0,   # wrong invest → moderate penalty
            (0.5, "INVEST"):    +0.5,   # invested in borderline → rewarded
            (0.5, "REJECT"):    -0.3,   # rejected borderline → small penalty
            (0,   "WATCHLIST"): +0.2,   # watchlisted bad startup → ok
        }
        return reward_matrix.get((truth, action), 0.0)

    def train(self, startups: list, episodes: int = 3000):
        print(f"🤖 Training RL Agent for {episodes} episodes …")
        episode_rewards = []

        for ep in range(episodes):
            ep_reward = 0
            for s in startups:
                state      = self._state(s)
                # ε-greedy action selection
                if random.random() < self.epsilon:
                    action_idx = random.randint(0, self.N_ACTIONS - 1)  # explore
                else:
                    action_idx = int(np.argmax(self.q_table[state]))     # exploit

                reward     = self._reward(action_idx, s)
                ep_reward += reward

                # Q-update: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s') − Q(s,a)]
                best_next = np.max(self.q_table[state])
                self.q_table[state, action_idx] += self.ALPHA * (
                    reward + self.GAMMA * best_next - self.q_table[state, action_idx]
                )

            # Decay epsilon after each full episode
            self.epsilon = max(self.EPSILON_MIN, self.epsilon * self.EPSILON_DECAY)
            episode_rewards.append(ep_reward / len(startups))

            if (ep + 1) % 500 == 0:
                avg = np.mean(episode_rewards[-100:])
                print(f"  Episode {ep+1:4d}/{episodes}  "
                      f"avg reward: {avg:.3f}  ε={self.epsilon:.3f}")

        self.trained          = True
        self.rewards_history  = episode_rewards
        print("✅ RL Agent trained.")

        # Print learned policy
        print("\n📋 Learned policy (Q-table):")
        for i, row in enumerate(self.q_table):
            best = ACTIONS[int(np.argmax(row))]
            lo   = i / self.N_STATES
            hi   = (i + 1) / self.N_STATES
            print(f"   Score {lo:.1f}–{hi:.1f}  →  {best}")

        return episode_rewards

    def decide(self, startup: dict) -> str:
        if not self.trained:
            return RuleBasedAgent().decide(startup)
        state = self._state(startup)
        return ACTIONS[int(np.argmax(self.q_table[state]))]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "q_table":        self.q_table,
                "trained":        self.trained,
                "rewards_history": self.rewards_history,
                "n_states":       self.N_STATES,
            }, f)

    def load(self, path: str):
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.q_table         = data["q_table"]
            self.trained         = data["trained"]
            self.rewards_history = data.get("rewards_history", [])
            self.N_STATES        = data.get("n_states", self.N_STATES)


if __name__ == "__main__":
    sample = {
        "revenue": 200000, "growth_rate": 60, "burn_rate": 25000,
        "market_size": 3000000, "competition_score": 3,
        "founder_experience": 10, "prev_exits": 2,
        "team_size": 15, "patents": 1, "risk_level": 0,
        "funding_stage": "Series A", "sector": "FinTech",
    }
    print("=== Agent Decisions ===")
    for AgentClass in [RandomAgent, RuleBasedAgent]:
        a = AgentClass()
        print(f"  {a.name:25s}: {a.decide(sample)}")
    ml = MLAgent()
    print(f"  {ml.name:25s}: {ml.decide(sample)}  (prob={ml.predict_probability(sample):.2%})")
