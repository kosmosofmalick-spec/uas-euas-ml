import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def main(args):
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_excel(args.excel)

    if args.label_col not in df.columns:
        raise ValueError(f"Label column '{args.label_col}' not found in Excel.")

    y = df[args.label_col]
    X = df.drop(columns=[args.label_col])

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    Xtr, Xte, ytr, yte = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    rf = RandomForestClassifier(n_estimators=500, random_state=42)
    rf.fit(Xtr, ytr)
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(X.columns[order][::-1], importances[order][::-1])
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'rf_feature_importance.png'), dpi=300)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--excel', required=True)
    p.add_argument('--label_col', default='Treatments')
    p.add_argument('--out', default='out')
    main(p.parse_args())
