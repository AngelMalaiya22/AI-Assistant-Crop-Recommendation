# 🌾 AI Assistant – Crop Recommendation

A machine-learning web app that recommends the most suitable crop for a field based on soil, season and weather conditions, and **explains every recommendation with SHAP**.

This is **Module 1** of my major project, the **AI Smart Farming Assistant**.

🔗 **Live demo:**[ _add your Render link here_](https://ai-assistant-crop-recommendation.onrender.com/)

---

## ✨ Features

- Recommends **1 crop out of 57** from 9 input conditions
- **Explainable AI:** a SHAP chart shows which inputs pushed the prediction up or down
- One-hot encoded features (soil, season, crop type) are merged back into a single bar each, so the explanation is easy to read
- Warns when an input is outside the range the model was trained on
- Lightweight Flask API with a simple, responsive web page

## 📊 Model

| Item | Details |
|---|---|
| Dataset | 57,000 records, 57 crops (1,000 per crop) |
| Inputs | Type of crop, soil type, season, soil pH, temperature, humidity, N, P, K |
| Train / test split | 80% / 20% |
| Hyperparameter tuning | Optuna |
| Final model | Random Forest |
| Explainability | SHAP `TreeExplainer` |

**Test accuracy after tuning**

| Model | Accuracy |
|---|---|
| **Random Forest** | **98.31%** |
| Decision Tree | 98.19% |
| XGBoost | 98.18% |
| KNN | 96.22% |
| Logistic Regression | 96.10% |
| SVM | 97.01% |

The ML pipeline: EDA → preprocessing → encoding → baseline models → comparison → hyperparameter tuning → explainability (SHAP) → deployment.

## 🧰 Tech stack

Python · scikit-learn · SHAP · Flask · Gunicorn · HTML/CSS/JavaScript · Render

## 📁 Project structure

```
├── app.py                    # Flask API (prediction + SHAP explanation)
├── requirements.txt          # Pinned dependencies
├── static/
│   └── index.html            # Web interface
└── deploy_artifacts/
    ├── rf_model.joblib       # Trained Random Forest
    └── meta.json             # Column order, input ranges, crop names
```

## 🚀 Run locally

```bash
git clone https://github.com/AngelMalaiya22/AI-Assistant-Crop-Recommendation.git
cd AI-Assistant-Crop-Recommendation
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000**.

The model was saved with `scikit-learn==1.5.1`, so keep the pinned versions in `requirements.txt`.

## 🔌 API

`POST /api/predict`

```json
{
  "SEASON": "kharif",
  "SOIL": "Alluvial soil",
  "TYPE_OF_CROP": "Root&tuber",
  "SOIL_PH": 6.6,
  "TEMP": 24.8,
  "RELATIVE_HUMIDITY": 78.2,
  "N": 80.5,
  "P": 47.7,
  "K": 50.5
}
```

Returns the recommended crop, the SHAP contribution of every input, and any out-of-range warnings. `GET /api/meta` returns the dropdown options and input ranges.

## ⚠️ Disclaimer

This is an academic project. Recommendations come from patterns in the training dataset and should be checked with a local agronomist before real farming decisions.

## 👩‍💻 Author

**Angel Malaiya**, B.Tech CSE (AI & ML), Sharda University
GitHub: [@AngelMalaiya22](https://github.com/AngelMalaiya22)
