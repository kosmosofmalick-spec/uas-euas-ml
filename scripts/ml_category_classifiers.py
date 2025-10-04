#!/usr/bin/env python3
"""
Binary classifiers for DEG response categories using functional class encodings.

Reads two Excel files (Co + Sc) produced by your pattern-classifier step
(e.g., Co_Gene_Pattern_Classification_with_Annotation.xlsx).
Merges them, builds binary labels for each task (Transient vs Others, Persistent vs Others),
trains RF/XGB/SVM, saves metrics, confusion matrices, SHAP summaries, and an F1 comparison bar.

Required columns in each Excel:
  - 'GeneID', 'Category', 'Assigned_Class'
  - Optional: 'log2FC_WL', 'log2FC_REC' (not required here)

Outputs (into --out):
  - metrics_<task>.txt
  - cm_<model>_<task>.png
  - shap_summary_<model>_<task>.png (RF & XGB)
  - f1_compare_<task>.png
"""
import argparse, os, re, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---- Helpers -----------------------------------------------------------------
def extract_code(s: str) -> str:
    """Robustly extract a functional code from Assigned_Class.
       Tries: [K] form → 'K'; or patterns like R_K_KOG → 'K'; fallback to Assigned_Class."""
    if not isinstance(s, str):
        return "Unclassified"
    m = re.search(r"\[([A-Z]{1,2})\]", s)  # e.g., "... [K] ..."
    if m:
        return m.group(1)
    m = re.search(r"(?:^|_)([A-Z]{1,2})(?:_(?:COG|KOG|eggNOG))?$", s)  # e.g., "R_K_KOG" or "K_COG"
    if m:
        return m.group(1)
    return s  # fallback: use the full string

def load_classification(path: str, species_tag: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    needed = {"GeneID", "Category", "Assigned_Class"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"{species_tag}: missing columns {missing} in {path}")
    df = df.copy()
    df["Species"] = species_tag
    # normalize categories
    df["Category"] = df["Category"].astype(str).str.strip().str.title()
    df["Category"] = df["Category"].replace({"Other": "Antagonistic"})  # as in your script
    # functional code
    df["Functional_Code"] = df["Assigned_Class"].apply(extract_code)
    return df

def save_cm(cm: np.ndarray, labels, title: str, outpath: str):
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def save_text(s: str, outpath: str):
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(s)

# ---- Main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--co", required=True, help="Excel path: Co pattern classification")
    ap.add_argument("--sc", required=True, help="Excel path: Sc pattern classification")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tasks", default="Transient,Persistent",
                    help="Comma-separated positives (choices: Transient,Persistent)")
    ap.add_argument("--use_feature", choices=["code","class"], default="code",
                    help="Use 'code' (Functional_Code) or 'class' (Assigned_Class) as feature")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    co = load_classification(args.co, "Co")
    sc = load_classification(args.sc, "Sc")
    df = pd.concat([co, sc], ignore_index=True)
    df = df.dropna(subset=["Category", "Assigned_Class"])

    # Feature: encode functional code or assigned class
    feat_col = "Functional_Code" if args.use_feature == "code" else "Assigned_Class"
    le_feat = LabelEncoder()
    df["feat_enc"] = le_feat.fit_transform(df[feat_col].astype(str))

    # Define tasks
    positives = [t.strip().title() for t in args.tasks.split(",") if t.strip()]
    tasks = {f"{pos}_vs_Others": pos for pos in positives}

    for task_name, pos_class in tasks.items():
        # Labels
        work = df.copy()
        work["Binary_Label"] = work["Category"].apply(lambda x: 1 if x == pos_class else 0)

        X = work[["feat_enc"]]              # single-feature baseline (as in your approach)
        y = work["Binary_Label"].astype(int)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, stratify=y, random_state=args.seed, test_size=args.test_size
        )

        models = {
            "RandomForest": RandomForestClassifier(n_estimators=300, random_state=args.seed),
            "XGBoost": XGBClassifier(
                n_estimators=300, max_depth=3, subsample=0.9, colsample_bytree=0.9,
                learning_rate=0.1, eval_metric="logloss", random_state=args.seed
            ),
            "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=args.seed),
        }

        f1_scores = {}
        metrics_text = [f"===== {task_name} (positive={pos_class}) =====\n"]

        for name, model in models.items():
            model.fit(Xtr, ytr)
            yhat = model.predict(Xte)

            acc = accuracy_score(yte, yhat)
            f1 = f1_score(yte, yhat, average="binary", zero_division=0)
            f1_scores[name] = f1

            # save metrics
            rep = classification_report(yte, yhat, target_names=[f"Not_{pos_class}", pos_class], zero_division=0)
            metrics_text.append(f"\n--- {name} ---\nAccuracy: {acc:.3f}\nF1: {f1:.3f}\n{rep}")

            # confusion matrix
            cm = confusion_matrix(yte, yhat)
            cm_path = os.path.join(args.out, f"cm_{name}_{task_name}.png")
            save_cm(cm, [f"Not_{pos_class}", pos_class], f"{name} — {task_name}", cm_path)

            # SHAP (tree-based only)
            if name in {"RandomForest", "XGBoost"}:
                try:
                    explainer = shap.Explainer(model, Xtr)
                    sv = explainer(Xte)
                    shap.summary_plot(sv, Xte, feature_names=["feat_enc"], show=False)
                    plt.tight_layout()
                    shap_path = os.path.join(args.out, f"shap_summary_{name}_{task_name}.png")
                    plt.savefig(shap_path, dpi=300, bbox_inches="tight")
                    plt.close()
                except Exception as e:
                    # Save a note instead of failing the run
                    save_text(f"SHAP failed for {name}: {e}\n", os.path.join(args.out, f"shap_{name}_{task_name}.txt"))

        # save overall metrics text
        save_text("\n".join(metrics_text), os.path.join(args.out, f"metrics_{task_name}.txt"))

        # F1 comparison bar
        plt.figure(figsize=(6, 4))
        keys, vals = list(f1_scores.keys()), list(f1_scores.values())
        bars = plt.bar(keys, vals)
        plt.ylabel("F1 score")
        plt.title(f"Model Comparison — {task_name}")
        plt.ylim(0, 1)
        for i, v in enumerate(vals):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"f1_compare_{task_name}.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
