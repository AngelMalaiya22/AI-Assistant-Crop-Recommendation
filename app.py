import json
import os

import joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, jsonify, request, send_from_directory

BASE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, "deploy_artifacts")

model = joblib.load(os.path.join(ART, "rf_model.joblib"))
with open(os.path.join(ART, "meta.json"), encoding="utf-8") as f:
    meta = json.load(f)

COLUMNS = meta["columns"]
NUMERIC = list(meta["numeric"])
PREFIX = {"TYPE_OF_CROP": "TYPE_OF_CROP_", "SOIL": "SOIL_", "SEASON": "SEASON_"}

# Dropdown options are read straight from the one-hot column names
OPTIONS = {
    key: sorted(c[len(p):] for c in COLUMNS if c not in NUMERIC and c.startswith(p))
    for key, p in PREFIX.items()
}

# Which original feature does each model column belong to? (used to merge one-hot SHAP values)
def group_of(col):
    if col in NUMERIC:
        return col
    for key, p in PREFIX.items():
        if col.startswith(p):
            return key
    return col

GROUPS = [group_of(c) for c in COLUMNS]
explainer = shap.TreeExplainer(model)

app = Flask(__name__, static_folder="static")


@app.get("/")
def home():
    return send_from_directory("static", "index.html")


@app.get("/api/meta")
def get_meta():
    return jsonify(options=OPTIONS, numeric=meta["numeric"])


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    row = dict.fromkeys(COLUMNS, 0.0)
    warnings = []
    try:
        for feat, rng in meta["numeric"].items():
            val = float(data[feat])
            if not (rng["min"] <= val <= rng["max"]):
                warnings.append(f"{feat}={val} is outside the training range "
                                f"({rng['min']:.1f} to {rng['max']:.1f}); the result may be unreliable.")
            row[feat] = val
        for key, p in PREFIX.items():
            col = p + data[key]
            if col not in row:
                return jsonify(error=f"Unknown {key}"), 400
            row[col] = 1.0
    except (KeyError, ValueError, TypeError):
        return jsonify(error="Please fill in all fields with valid values."), 400

    X = pd.DataFrame([row], columns=COLUMNS)
    proba = model.predict_proba(X)[0]
    order = np.argsort(proba)[::-1][:3]
    best = int(order[0])

    # Local SHAP explanation for the top predicted crop
    sv = explainer.shap_values(X)
    sv_best = sv[best][0] if isinstance(sv, list) else sv[0, :, best]
    merged = {}
    for g, v in zip(GROUPS, sv_best):
        merged[g] = merged.get(g, 0.0) + float(v)

    shown = {**{k: data[k] for k in NUMERIC}, **{k: data[k] for k in PREFIX}}
    contributions = sorted(
        ({"feature": g, "input": shown[g], "shap": v} for g, v in merged.items()),
        key=lambda d: abs(d["shap"]), reverse=True)

    def name(i):
        return meta["crop_names"][str(int(model.classes_[i]))]

    return jsonify(
        predictions=[{"crop": name(int(i)), "probability": float(proba[i])} for i in order],
        explanation={"crop": name(best), "contributions": contributions},
        warnings=warnings)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
