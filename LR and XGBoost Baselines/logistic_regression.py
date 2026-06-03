# Logistic Regression on node features only.
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import json, os

def load_data():
    features = pd.read_csv('data/elliptic/elliptic_txs_features.csv', header=None)
    classes  = pd.read_csv('data/elliptic/elliptic_txs_classes.csv')
    times    = pd.read_csv('data/elliptic/elliptic_txs_nodetime.csv')
    features = features.copy()
    features['txId'] = features[0]
    df = features.merge(classes, on='txId').merge(times, on='txId')
    df = df[df['class'] != -1.0]
    X = df.iloc[:, 1:167].values
    y = df['class'].values
    t = df['timestep'].values
    return X, y, t

def main():
    X, y, t = load_data()
    X_train, y_train = X[t <= 33], y[t <= 33]
    X_test,  y_test  = X[t >= 35], y[t >= 35]

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Illicit in test: {(y_test==1.0).sum()} | Licit: {(y_test==0.0).sum()}")

    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:,1]
    preds = model.predict(X_test)

    results = {
        "model": "logistic_regression",
        "auc":         round(roc_auc_score(y_test, probs), 4),
        "weighted_f1": round(f1_score(y_test, preds, average='weighted'), 4),
        "illicit_f1":  round(f1_score(y_test, preds, pos_label=1.0, average='binary'), 4),
    }

    print("LR results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(classification_report(y_test, preds))

    os.makedirs('results', exist_ok=True)
    with open('results/logistic_regression.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("saved results to results/logistic_regression.json")

if __name__ == '__main__':
    main()