"""
VentureX ML Models
Trains Logistic Regression + Random Forest and saves them.
Run: python models/train.py
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve,
    classification_report
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "startups.csv")
MODEL_DIR   = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR    = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────

NUMERIC_FEATURES = [
    "revenue", "growth_rate", "burn_rate", "market_size",
    "competition_score", "founder_experience",
    "team_size", "patents", "prev_exits", "risk_level",
]

def load_and_prepare(path: str):
    df = pd.read_csv(path)

    # Encode categoricals
    le_stage  = LabelEncoder()
    le_sector = LabelEncoder()
    df["funding_stage_enc"] = le_stage.fit_transform(df["funding_stage"])
    df["sector_enc"]        = le_sector.fit_transform(df["sector"])

    # Save encoders
    with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "wb") as f:
        pickle.dump({"stage": le_stage, "sector": le_sector}, f)

    feature_cols = NUMERIC_FEATURES + ["funding_stage_enc", "sector_enc"]
    X = df[feature_cols].values
    y = df["success"].values

    print(f"Dataset: {len(df)} rows | features: {len(feature_cols)} | "
          f"success rate: {y.mean()*100:.1f}%")
    return X, y, feature_cols


# ──────────────────────────────────────────────
# Train & evaluate
# ──────────────────────────────────────────────

def train_and_evaluate(X, y, feature_cols):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, random_state=42,
                                          solver="lbfgs"))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=200, max_depth=8,
                                              random_state=42, n_jobs=-1))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                                  max_depth=4, random_state=42))
        ]),
    }

    results = {}
    best_name, best_f1 = None, 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        pr  = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cv  = cross_val_score(model, X, y, cv=5, scoring="f1").mean()

        results[name] = {
            "model": model, "y_pred": y_pred, "y_prob": y_prob,
            "accuracy": acc, "precision": pr, "recall": rec,
            "f1": f1, "roc_auc": auc, "cv_f1": cv,
        }

        print(f"\n{'─'*45}")
        print(f"  {name}")
        print(f"  Accuracy : {acc:.4f}   Precision: {pr:.4f}")
        print(f"  Recall   : {rec:.4f}   F1-Score : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}   CV F1    : {cv:.4f}")

        if f1 > best_f1:
            best_f1, best_name = f1, name

        # Save each model
        fname = name.lower().replace(" ", "_") + ".pkl"
        with open(os.path.join(MODEL_DIR, fname), "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved → models/{fname}")

    print(f"\n🏆 Best model: {best_name}  (F1={best_f1:.4f})")

    # Save best model alias
    with open(os.path.join(MODEL_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(results[best_name]["model"], f)

    # Save feature list
    with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)

    plot_results(results, feature_cols, X_train, y_test)
    return results, best_name


# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────

def plot_results(results, feature_cols, X_train, y_test):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("VentureX — ML Evaluation", fontsize=16, fontweight="bold", y=1.01)
    colors = ["#4CAF50", "#2196F3", "#FF9800"]

    # 1. Model comparison bar
    ax = axes[0, 0]
    names   = list(results.keys())
    metrics = ["accuracy", "f1", "roc_auc"]
    x = np.arange(len(names))
    w = 0.25
    for i, m in enumerate(metrics):
        vals = [results[n][m] for n in names]
        ax.bar(x + i*w, vals, w, label=m.upper(), color=colors[i], alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels([n.split()[0] for n in names])
    ax.set_ylim(0, 1.05); ax.set_title("Model Comparison"); ax.legend()

    # 2–4. Confusion matrices
    for idx, (name, res) in enumerate(results.items()):
        ax = axes[0, idx+1] if idx < 2 else axes[1, 0]
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax,
                    xticklabels=["Reject","Invest"], yticklabels=["Reject","Invest"])
        ax.set_title(f"Confusion — {name.split()[0]}")
        ax.set_ylabel("True"); ax.set_xlabel("Predicted")

    # 5. ROC curves
    ax = axes[1, 1]
    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name.split()[0]} (AUC={res['roc_auc']:.2f})",
                color=colors[i], linewidth=2)
    ax.plot([0,1],[0,1],"k--", linewidth=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curves"); ax.legend()

    # 6. Feature importance (RF)
    ax = axes[1, 2]
    rf_model = results["Random Forest"]["model"].named_steps["clf"]
    importances = rf_model.feature_importances_
    idx_sorted  = np.argsort(importances)[-10:]
    ax.barh([feature_cols[i] for i in idx_sorted],
            importances[idx_sorted], color="#4CAF50", alpha=0.85)
    ax.set_title("Feature Importance (RF)")

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "ml_evaluation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Plots saved → {out}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Generate data if missing
    if not os.path.exists(DATA_PATH):
        print("Generating dataset …")
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
        from generate_data import generate_startup_dataset
        generate_startup_dataset()

    X, y, feature_cols = load_and_prepare(DATA_PATH)
    train_and_evaluate(X, y, feature_cols)
    print("\n✅ Training complete. All models saved to models/")
